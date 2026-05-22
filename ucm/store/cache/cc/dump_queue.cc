/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#include "logger/logger.h"
#include "metrics_api.h"
#include "thread/cpu_affinity.h"

namespace UC::CacheStore {

DumpQueue::~DumpQueue()
{
    stop_.store(true);
    if (dispatcher_.joinable()) { dispatcher_.join(); }
    if (syncer_.joinable()) { syncer_.join(); }
    if (dumper_.joinable()) { dumper_.join(); }
}

Status DumpQueue::Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer)
{
    failureSet_ = failureSet;
    buffer_ = buffer;
    backend_ = config.storeBackend;
    deviceId_ = config.deviceId;
    tensorSizes_ = config.tensorSizes;
    streamNumber_ = config.streamNumber;
    dumpD2hPipelineDepth_ = config.dumpD2hPipelineDepth;
    useGdr_ = config.useGdr;
    cpuAffinityCores_ = config.cpuAffinityCores;
    waiting_.Setup(config.waitingQueueDepth);
    syncing_.Setup(config.runningQueueDepth);
    dumping_.Setup(config.runningQueueDepth);
    freeD2hPipelineTokens_.Setup(dumpD2hPipelineDepth_ + 1);
    std::promise<Status> dispatchStarted;
    auto dispatchFut = dispatchStarted.get_future();
    dispatcher_ = std::thread{&DumpQueue::DispatchStage, this, std::ref(dispatchStarted)};
    auto s = dispatchFut.get();
    if (s.Failure()) [[unlikely]] { return s; }
    std::promise<Status> syncStarted;
    auto syncFut = syncStarted.get_future();
    syncer_ = std::thread{&DumpQueue::SyncStage, this, std::ref(syncStarted)};
    s = syncFut.get();
    if (s.Failure()) [[unlikely]] { return s; }
    dumper_ = std::thread{&DumpQueue::BackendDumpStage, this};
    return Status::OK();
}

void DumpQueue::Submit(TaskPtr task, WaiterPtr waiter)
{
    waiter->Up();
    auto success = waiting_.TryPush({task, waiter});
    if (success) { return; }
    UC_ERROR("Waiting queue full, submit dump task({}) failed.", task->id);
    failureSet_->Insert(task->id);
    waiter->Done();
}

void DumpQueue::DispatchStage(std::promise<Status>& started)
{
    for (size_t i = 0; i < dumpD2hPipelineDepth_; ++i) {
        CopyStream stream;
        auto s = stream.Setup(deviceId_, streamNumber_, useGdr_);
        if (s.Failure()) [[unlikely]] {
            started.set_value(s);
            return;
        }
        freeD2hPipelineTokens_.Push(std::move(stream));
    }
    started.set_value(Status::OK());
    if (!cpuAffinityCores_.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    waiting_.ConsumerLoop(stop_, &DumpQueue::DispatchOneTask, this);
}

void DumpQueue::DispatchOneTask(TaskPair&& pair)
{
    auto& task = pair.first;
    auto& waiter = pair.second;
    auto wait = NowTime::Now() - waiter->startTp;
    UC_DEBUG("Cache task({}) start running, wait {:.3f}ms.", task->id, wait * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_queue_wait_duration_ms"), wait * 1e3);
    if (failureSet_->Contains(task->id)) {
        waiter->Done();
        return;
    }
    CopyStream stream;
    if (!AcquireD2hPipelineToken(stream)) {
        waiter->Done();
        return;
    }
    D2hSyncCtx submitted;
    auto s = SubmitD2H(stream, task, waiter, submitted);
    submitted.stream = std::move(stream);
    submitted.status = s;
    syncing_.Push(std::move(submitted));
}

bool DumpQueue::AcquireD2hPipelineToken(CopyStream& stream)
{
    while (!stop_.load(std::memory_order_acquire)) {
        if (freeD2hPipelineTokens_.TryPop(stream)) { return true; }
        std::this_thread::yield();
    }
    return false;
}

Status DumpQueue::SubmitD2H(CopyStream& stream, TaskPtr task, WaiterPtr waiter,
                            D2hSyncCtx& submitted)
{
    auto tp = NowTime::Now();
    submitted.taskHandle = task->id;
    submitted.waiter = waiter;
    submitted.startTp = tp;
    submitted.backendTaskDesc.brief = "Cache2Backend";
    if (task->desc.prerequisiteHandle != 0) {
        auto s = stream.WaitEvent(reinterpret_cast<void*>(task->desc.prerequisiteHandle));
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait prerequisite event for dump task({}).", s,
                     task->id);
            return s;
        }
    }
    const auto nShard = task->desc.size();
    UC_DEBUG("Try to dump ({}) shards.", nShard);
    for (size_t i = 0; i < nShard; i++) {
        auto& shard = task->desc[i];
        auto handle = buffer_->Get(shard.owner, shard.index);
        if (!handle.Owner()) { continue; }
        if (!handle.Ready()) {
            auto s =
                DeviceToHostGatherAsync(stream.NextStream(), shard.addrs.data(), handle.Data());
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to do D2H batch async for task({}).", s, task->id);
                return s;
            }
        }
        submitted.backendTaskDesc.push_back(
            Detail::Shard{shard.owner, shard.index, {handle.Data()}});
        submitted.bufferHandles.push_back(std::move(handle));
    }
    submitted.d2hSubmittedTp = NowTime::Now();
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_shards_total"),
                             static_cast<double>(nShard));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_backend_shards_total"),
                             static_cast<double>(submitted.backendTaskDesc.size()));
    return Status::OK();
}

void DumpQueue::SyncStage(std::promise<Status>& started)
{
    Trans::Device device;
    auto s = device.Setup(deviceId_);
    started.set_value(s);
    if (s.Failure()) [[unlikely]] { return; }
    if (!cpuAffinityCores_.empty()) {
        s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    syncing_.ConsumerLoop(stop_, &DumpQueue::SyncOneTask, this);
}

void DumpQueue::SyncOneTask(D2hSyncCtx&& ctx)
{
    auto taskHandle = ctx.taskHandle;
    auto waiter = ctx.waiter;
    auto s = SyncAndDumpOneTask(std::move(ctx));
    if (s.Failure()) [[unlikely]] { failureSet_->Insert(taskHandle); }
    waiter->Done();
}

Status DumpQueue::SyncAndDumpOneTask(D2hSyncCtx&& ctx)
{
    auto s = ctx.stream.Synchronize();
    auto tpSyncStream = NowTime::Now();
    freeD2hPipelineTokens_.Push(std::move(ctx.stream));
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to sync on stream for task({}).", s, ctx.taskHandle);
        return s;
    }
    if (ctx.status.Failure()) [[unlikely]] { return ctx.status; }
    if (ctx.backendTaskDesc.empty()) { return Status::OK(); }
    for (auto& handle : ctx.bufferHandles) { handle.MarkReady(); }
    auto res = backend_->Dump(std::move(ctx.backendTaskDesc));
    if (!res) [[unlikely]] {
        UC_ERROR("Failed({}) to submit dump task({}) to backend.", res.Error(), ctx.taskHandle);
        return res.Error();
    }
    BackendWaitCtx dumpCtx;
    dumpCtx.taskHandle = ctx.taskHandle;
    dumpCtx.backendTaskHandle = res.Value();
    dumpCtx.bufferHandles = std::move(ctx.bufferHandles);
    dumping_.Push(std::move(dumpCtx));
    auto tpEnd = NowTime::Now();
    UC_DEBUG("Cache task({}) mk_buf={:.3f}ms, sync={:.3f}ms, back={:.3f}ms.", ctx.taskHandle,
             (ctx.d2hSubmittedTp - ctx.startTp) * 1e3,
             (tpSyncStream - ctx.d2hSubmittedTp) * 1e3,
             (tpEnd - tpSyncStream) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_mkbuf_duration_ms"),
                             (ctx.d2hSubmittedTp - ctx.startTp) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_duration_ms"),
                             (tpSyncStream - ctx.d2hSubmittedTp) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_backend_submit_duration_ms"),
                             (tpEnd - tpSyncStream) * 1e3);
    return Status::OK();
}

Status DumpQueue::DeviceToHostGatherAsync(std::shared_ptr<Trans::Stream> stream, void** device,
                                          void* host)
{
    const auto number = tensorSizes_.size();
    for (size_t i = 0, offset = 0; i < number; i++) {
        auto pDevice = device[i];
        auto pHost = (void*)(((int8_t*)host) + offset);
        auto size = tensorSizes_[i];
        auto s = stream->DeviceToHostAsync(pDevice, pHost, size);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to do D2H({}) batch({}/{}) async.", s, size, i, number);
            return s;
        }
        offset += size;
    }
    return Status::OK();
}

void DumpQueue::BackendDumpStage()
{
    if (!cpuAffinityCores_.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    dumping_.ConsumerLoop(stop_, [this](auto&& task) {
        auto s = backend_->Wait(task.backendTaskHandle);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to wait backend({}) for task({}).", s, task.backendTaskHandle,
                     task.taskHandle);
        }
    });
}

}  // namespace UC::CacheStore
