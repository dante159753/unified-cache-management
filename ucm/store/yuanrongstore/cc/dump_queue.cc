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
#include "time/now_time.h"
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
    auto taskStart = NowTime::Now();
    auto prereqStart = taskStart;
    if (task->desc.prerequisiteHandle != 0) {
        auto s =
            prerequisiteStream->WaitEvent(reinterpret_cast<void*>(task->desc.prerequisiteHandle));
        if (s.Failure()) { return s; }
        s = prerequisiteStream->Synchronized();
        if (s.Failure()) { return s; }
    }
    auto prereqEnd = NowTime::Now();

    std::vector<std::string> keys;
    std::vector<datasystem::DeviceBlobList> blobLists;
    auto s = BuildKeysAndBlobs(config_, task->desc, keys, blobLists);
    if (s.Failure()) { return s; }
    if (keys.empty()) { return Status::OK(); }
    DeduplicateYuanRongObjects(keys, blobLists, &task->desc);
    const auto totalBytes = TotalBlobBytes(blobLists);

    datasystem::SetParam setParam;
    setParam.writeMode = datasystem::WriteMode::NONE_L2_CACHE_EVICT;
    setParam.existence = datasystem::ExistenceOpt::NONE;
    setParam.cacheType = datasystem::CacheType::MEMORY;
    std::vector<std::string> localSetKeys;
    auto d2hStart = NowTime::Now();
    auto dumpStatus = heteroClient_->MSetD2H(keys, blobLists, setParam, &localSetKeys);
    auto publishEnd = NowTime::Now();
    UC_DEBUG(
        "YuanRong dump task({}) MSetD2H keys={}, local_set={}, bytes={}, prereq={:.3f}ms, "
        "d2h={:.3f}ms, status={}.",
        task->id, keys.size(), localSetKeys.size(), totalBytes, (prereqEnd - prereqStart) * 1e3,
        (publishEnd - d2hStart) * 1e3, dumpStatus.ToString());
    if (localSetKeys.empty()) {
        if (dumpStatus.IsError()) {
            return Status::Error(
                fmt::format("YuanRong MSetD2H failed without any confirmed local key: {}",
                            dumpStatus.ToString()));
        }
        UC_DEBUG("YuanRong dump task({}) has no newly published local key; skipping Posix.",
                 task->id);
        return Status::OK();
    }

    s = SelectYuanRongObjectsByKeys(localSetKeys, keys, task->desc);
    if (s.Failure()) { return s; }
    if (dumpStatus.IsError()) {
        UC_WARN(
            "YuanRong MSetD2H partially failed but confirmed ({} keys) published locally; "
            "persisting the confirmed keys only: {}",
            keys.size(), dumpStatus.ToString());
    }
    if (backend_ == nullptr) {
        UC_DEBUG("YuanRong dump task({}) finished without backend, local_set={}, total={:.3f}ms.",
                 task->id, keys.size(), (NowTime::Now() - taskStart) * 1e3);
        return Status::OK();
    }

    std::vector<datasystem::MetaInfo> metaInfos;
    std::vector<std::string> metaFailedKeys;
    auto metaStart = NowTime::Now();
    auto metaStatus = heteroClient_->GetMetaInfo(keys, false, metaInfos, metaFailedKeys);
    auto metaEnd = NowTime::Now();
    if (metaStatus.IsError() || metaInfos.size() != keys.size() || !metaFailedKeys.empty()) {
        return Status::Error(fmt::format(
            "YuanRong GetMetaInfo after D2H failed: {}, "
            "failed keys({}), meta count({}), key count({})",
            metaStatus.ToString(), metaFailedKeys.size(), metaInfos.size(), keys.size()));
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        auto metaCheck = ValidateYuanRongBlobSizes(keys[i], metaInfos[i], config_.tensorSizes);
        if (metaCheck.Failure()) { return metaCheck; }
    }

    std::vector<datasystem::Optional<datasystem::ReadOnlyBuffer>> buffers;
    auto kvGetStart = NowTime::Now();
    auto getStatus = kvClient_->Get(keys, buffers, static_cast<int32_t>(config_.timeoutMs));
    auto kvGetEnd = NowTime::Now();
    if (getStatus.IsError() || buffers.size() != keys.size()) {
        return Status::Error(
            fmt::format("YuanRong Get after D2H failed: {}", getStatus.ToString()));
    }

    std::vector<datasystem::Optional<datasystem::ReadOnlyBuffer>> lockedBuffers;
    lockedBuffers.reserve(buffers.size());
    Detail::TaskDesc backendTask;
    backendTask.brief = "YuanRong2Posix";
    backendTask.reserve(task->desc.size());
    size_t unavailableBufferCount = 0;
    std::string firstUnavailableBufferKey;
    for (size_t i = 0; i < buffers.size(); ++i) {
        auto& buffer = buffers[i];
        if (!buffer) {
            if (unavailableBufferCount == 0) { firstUnavailableBufferKey = keys[i]; }
            ++unavailableBufferCount;
            continue;
        }
        auto latchStatus = buffer->RLatch();
        if (latchStatus.IsError()) {
            if (unavailableBufferCount == 0) { firstUnavailableBufferKey = keys[i]; }
            ++unavailableBufferCount;
            UC_WARN("Failed to latch YuanRong buffer for key({}), skipping it: {}.", keys[i],
                    latchStatus.ToString());
            continue;
        }
        const auto* bufferAddress = buffer->ImmutableData();
        const void* payloadAddress = nullptr;
        auto payloadStatus =
            GetYuanRongPayloadAddress(keys[i], bufferAddress, buffer->GetSize(), metaInfos[i],
                                      config_.tensorSizes, config_.memoryAlignment, payloadAddress);
        if (payloadStatus.Failure()) {
            (void)buffer->UnRLatch();
            ReleaseBuffers(lockedBuffers);
            return payloadStatus;
        }
        if (config_.ioDirect) {
            auto directStatus = ValidateYuanRongDirectIoPayload(
                keys[i], payloadAddress, config_.objectSize, config_.memoryAlignment);
            if (directStatus.Failure()) {
                (void)buffer->UnRLatch();
                ReleaseBuffers(lockedBuffers);
                return directStatus;
            }
        }
        const auto& source = task->desc[i];
        backendTask.push_back(
            Detail::Shard{source.owner, source.index, {const_cast<void*>(payloadAddress)}});
        lockedBuffers.push_back(std::move(buffer));
    }
    if (unavailableBufferCount != 0) {
        UC_WARN(
            "YuanRong Get returned unavailable buffers({}/{}), first unavailable key({}); "
            "persisting readable buffers only.",
            unavailableBufferCount, buffers.size(), firstUnavailableBufferKey);
    }
    if (backendTask.empty()) {
        return Status::Error("YuanRong Get returned no readable buffer for Posix dump");
    }

    const auto backendKeyCount = backendTask.size();
    auto backendResult = backend_->Dump(std::move(backendTask));
    auto backendSubmitEnd = NowTime::Now();
    if (!backendResult) {
        ReleaseBuffers(lockedBuffers);
        return Status::Error(
            fmt::format("failed to submit Posix dump: {}", backendResult.Error().ToString()));
    }
    UC_INFO(
        "YuanRong dump task({}) backend submit keys={}, local_set_keys={}, "
        "get_meta={:.3f}ms, kvClient_get={:.3f}ms, prepare_backend={:.3f}ms, "
        "d2h_to_backend_submit={:.3f}ms, total={:.3f}ms.",
        task->id, backendKeyCount, keys.size(), (metaEnd - metaStart) * 1e3,
        (kvGetEnd - kvGetStart) * 1e3, (backendSubmitEnd - kvGetEnd) * 1e3,
        (backendSubmitEnd - publishEnd) * 1e3, (backendSubmitEnd - taskStart) * 1e3);
    reaping_.Push(DumpContext{task->id, backendResult.Value(), std::move(lockedBuffers)});
    return Status::OK();
}

void DumpQueue::Reap(DumpContext&& context)
{
    auto waitStart = NowTime::Now();
    auto s = backend_->Wait(context.backendTaskId);
    auto waitMs = (NowTime::Now() - waitStart) * 1e3;
    if (s.Failure()) {
        UC_ERROR("Background Posix dump({}) for YuanRong task({}) failed: {}.",
                 context.backendTaskId, context.ownerTaskId, s);
    } else {
        UC_DEBUG("Background Posix dump({}) for YuanRong task({}) finished, wait {:.3f}ms.",
                 context.backendTaskId, context.ownerTaskId, waitMs);
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
