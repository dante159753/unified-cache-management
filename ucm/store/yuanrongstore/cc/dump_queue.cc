/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include "dump_queue.h"
#include <algorithm>
#include <fmt/format.h>
#include "logger/logger.h"
#ifdef __linux__
#include "thread/cpu_affinity.h"
#endif
#include "trans/device.h"
#include "yuanrong_helper.h"

namespace UC::YuanRongStore {

DumpQueue::~DumpQueue() { Close(); }

Status DumpQueue::Setup(const Config& config, TaskIdSet* failureSet,
                        std::shared_ptr<datasystem::HeteroClient> heteroClient,
                        std::shared_ptr<datasystem::KVClient> kvClient)
{
    config_ = config;
    failureSet_ = failureSet;
    heteroClient_ = std::move(heteroClient);
    kvClient_ = std::move(kvClient);
    backend_ = config.storeBackend;
    waiting_.Setup(config.waitingQueueDepth);
    reaping_.Setup(config.reaperQueueDepth);

    std::promise<Status> started;
    auto future = started.get_future();
    worker_ = std::thread{&DumpQueue::WorkerStage, this, std::ref(started)};
    auto s = future.get();
    if (s.Failure()) { return s; }
    reaper_ = std::thread{&DumpQueue::ReaperStage, this};
    return Status::OK();
}

void DumpQueue::Close()
{
    if (stop_.exchange(true)) { return; }
    if (worker_.joinable()) { worker_.join(); }
    if (reaper_.joinable()) { reaper_.join(); }

    TaskPair pair;
    while (waiting_.TryPop(pair)) {
        failureSet_->Insert(pair.first->id);
        pair.second->Done();
    }
    DumpContext context;
    while (reaping_.TryPop(context)) { Reap(std::move(context)); }
}

void DumpQueue::Submit(TaskPtr task, WaiterPtr waiter)
{
    waiter->Up();
    std::lock_guard<std::mutex> lock(submitMutex_);
    if (waiting_.TryPush({task, waiter})) { return; }
    UC_ERROR("YuanRong dump queue full, task({}) rejected.", task->id);
    failureSet_->Insert(task->id);
    waiter->Done();
}

void DumpQueue::WorkerStage(std::promise<Status>& started)
{
    Trans::Device device;
    auto s = device.Setup(config_.deviceId);
    if (s.Failure()) {
        started.set_value(s);
        return;
    }
    auto prerequisiteStream = device.MakeSharedStream();
    if (!prerequisiteStream) {
        started.set_value(Status::Error("failed to create prerequisite stream"));
        return;
    }
    started.set_value(Status::OK());

#ifdef __linux__
    if (!config_.cpuAffinityCores.empty()) {
        auto affinityStatus = CpuAffinity::SetCpuAffinity4CurrentThread(config_.cpuAffinityCores);
        if (affinityStatus.Failure()) {
            UC_WARN("Failed({}) to set YuanRong dump affinity.", affinityStatus);
        }
    }
#endif
    waiting_.ConsumerLoop(stop_, &DumpQueue::RunOne, this, prerequisiteStream);
}

void DumpQueue::ReaperStage()
{
#ifdef __linux__
    if (!config_.cpuAffinityCores.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(config_.cpuAffinityCores);
        if (s.Failure()) { UC_WARN("Failed({}) to set YuanRong reaper affinity.", s); }
    }
#endif
    reaping_.ConsumerLoop(stop_, &DumpQueue::Reap, this);
}

void DumpQueue::RunOne(std::shared_ptr<Trans::Stream>& prerequisiteStream, TaskPair&& pair)
{
    auto& task = pair.first;
    auto& waiter = pair.second;
    if (!failureSet_->Contains(task->id)) {
        auto s = DumpOne(task, prerequisiteStream);
        if (s.Failure()) {
            UC_ERROR("YuanRong dump task({}) failed: {}.", task->id, s);
            failureSet_->Insert(task->id);
        }
    }
    waiter->Done();
}

Status DumpQueue::DumpOne(TaskPtr task, const std::shared_ptr<Trans::Stream>& prerequisiteStream)
{
    if (task->desc.prerequisiteHandle != 0) {
        auto s =
            prerequisiteStream->WaitEvent(reinterpret_cast<void*>(task->desc.prerequisiteHandle));
        if (s.Failure()) { return s; }
        s = prerequisiteStream->Synchronized();
        if (s.Failure()) { return s; }
    }

    std::vector<std::string> keys;
    std::vector<datasystem::DeviceBlobList> blobLists;
    auto s = BuildKeysAndBlobs(config_, task->desc, keys, blobLists);
    if (s.Failure()) { return s; }
    if (keys.empty()) { return Status::OK(); }

    datasystem::SetParam setParam;
    setParam.writeMode = datasystem::WriteMode::NONE_L2_CACHE_EVICT;
    setParam.existence = datasystem::ExistenceOpt::NONE;
    setParam.cacheType = datasystem::CacheType::MEMORY;
    auto dumpStatus = heteroClient_->MSetD2H(keys, blobLists, setParam);
    if (dumpStatus.IsError()) {
        return Status::Error(fmt::format("YuanRong MSetD2H failed: {}", dumpStatus.ToString()));
    }
    if (backend_ == nullptr) { return Status::OK(); }

    std::vector<bool> exists;
    auto existStatus = heteroClient_->Exist(keys, exists);
    if (existStatus.IsError() || exists.size() != keys.size() ||
        !std::all_of(exists.begin(), exists.end(), [](bool value) { return value; })) {
        return Status::Error(
            fmt::format("YuanRong MSetD2H did not publish all keys: {}", existStatus.ToString()));
    }

    std::vector<datasystem::MetaInfo> metaInfos;
    std::vector<std::string> metaFailedKeys;
    auto metaStatus = heteroClient_->GetMetaInfo(keys, false, metaInfos, metaFailedKeys);
    if (metaStatus.IsError() || metaInfos.size() != keys.size() || !metaFailedKeys.empty()) {
        return Status::Error(fmt::format("YuanRong GetMetaInfo after D2H failed: {}, "
                                         "failed keys({}), meta count({}), key count({})",
                                         metaStatus.ToString(), metaFailedKeys.size(),
                                         metaInfos.size(), keys.size()));
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        auto metaCheck = ValidateYuanRongBlobSizes(keys[i], metaInfos[i], config_.tensorSizes);
        if (metaCheck.Failure()) { return metaCheck; }
    }

    std::vector<datasystem::Optional<datasystem::ReadOnlyBuffer>> buffers;
    auto getStatus = kvClient_->Get(keys, buffers, static_cast<int32_t>(config_.timeoutMs));
    if (getStatus.IsError() || buffers.size() != keys.size()) {
        return Status::Error(
            fmt::format("YuanRong Get after D2H failed: {}", getStatus.ToString()));
    }

    size_t locked = 0;
    Detail::TaskDesc backendTask;
    backendTask.brief = "YuanRong2Posix";
    backendTask.reserve(task->desc.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        auto& buffer = buffers[i];
        if (!buffer) {
            for (size_t j = 0; j < locked; ++j) { (void)buffers[j]->UnRLatch(); }
            return Status::Error(
                fmt::format("YuanRong Get returned no buffer for key({})", keys[i]));
        }
        auto latchStatus = buffer->RLatch();
        if (latchStatus.IsError()) {
            for (size_t j = 0; j < locked; ++j) { (void)buffers[j]->UnRLatch(); }
            return Status::Error(
                fmt::format("failed to latch YuanRong buffer: {}", latchStatus.ToString()));
        }
        ++locked;
        const auto* bufferAddress = buffer->ImmutableData();
        const void* payloadAddress = nullptr;
        auto payloadStatus = GetYuanRongPayloadAddress(
            keys[i], bufferAddress, buffer->GetSize(), metaInfos[i], config_.tensorSizes,
            payloadAddress);
        if (payloadStatus.Failure()) {
            for (size_t j = 0; j < locked; ++j) { (void)buffers[j]->UnRLatch(); }
            return payloadStatus;
        }
        const auto& source = task->desc[i];
        backendTask.push_back(
            Detail::Shard{source.owner, source.index, {const_cast<void*>(payloadAddress)}});
    }

    auto backendResult = backend_->Dump(std::move(backendTask));
    if (!backendResult) {
        ReleaseBuffers(buffers);
        return Status::Error(
            fmt::format("failed to submit Posix dump: {}", backendResult.Error().ToString()));
    }
    reaping_.Push(DumpContext{task->id, backendResult.Value(), std::move(buffers)});
    return Status::OK();
}

void DumpQueue::Reap(DumpContext&& context)
{
    auto s = backend_->Wait(context.backendTaskId);
    if (s.Failure()) {
        UC_ERROR("Background Posix dump({}) for YuanRong task({}) failed: {}.",
                 context.backendTaskId, context.ownerTaskId, s);
    }
    ReleaseBuffers(context.buffers);
}

void DumpQueue::ReleaseBuffers(
    std::vector<datasystem::Optional<datasystem::ReadOnlyBuffer>>& buffers)
{
    for (auto& buffer : buffers) {
        if (!buffer) { continue; }
        auto status = buffer->UnRLatch();
        if (status.IsError()) {
            UC_WARN("Failed to unlock YuanRong read buffer: {}.", status.ToString());
        }
    }
    buffers.clear();
}

}  // namespace UC::YuanRongStore
