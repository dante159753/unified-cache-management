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
#ifndef UNIFIEDCACHE_CACHE_STORE_CC_DUMP_QUEUE_H
#define UNIFIEDCACHE_CACHE_STORE_CC_DUMP_QUEUE_H

#include <atomic>
#include <future>
#include <thread>
#include "copy_stream.h"
#include "template/hashset.h"
#include "template/spsc_ring_queue.h"
#include "thread/latch.h"
#include "trans_buffer.h"
#include "trans_task.h"
#include "ucmstore_v1.h"

namespace UC::CacheStore {

class DumpQueue {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using TaskIdSet = HashSet<Detail::TaskHandle>;
    struct BackendWaitCtx {
        Detail::TaskHandle taskHandle;
        Detail::TaskHandle backendTaskHandle;
        std::vector<TransBuffer::Handle> bufferHandles;
    };
    struct D2hSyncCtx {
        Detail::TaskHandle taskHandle{0};
        Detail::TaskDesc backendTaskDesc;
        std::vector<TransBuffer::Handle> bufferHandles;
        WaiterPtr waiter;
        CopyStream stream;
        Status status{Status::OK()};
        double startTp{0};
        double d2hSubmittedTp{0};
    };

private:
    alignas(64) std::atomic_bool stop_{false};
    TaskIdSet* failureSet_{nullptr};
    TransBuffer* buffer_{nullptr};
    StoreV1* backend_{nullptr};
    int32_t deviceId_{-1};
    std::vector<size_t> tensorSizes_{};
    size_t streamNumber_{1};
    size_t dumpD2hPipelineDepth_{1};
    bool useGdr_{false};
    std::vector<ssize_t> cpuAffinityCores_{};
    SpscRingQueue<TaskPair> waiting_;
    SpscRingQueue<D2hSyncCtx> syncing_;
    SpscRingQueue<BackendWaitCtx> dumping_;
    SpscRingQueue<CopyStream> freeD2hPipelineTokens_;
    std::thread dispatcher_;
    std::thread syncer_;
    std::thread dumper_;

public:
    ~DumpQueue();
    Status Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer);
    void Submit(TaskPtr task, WaiterPtr waiter);

private:
    void DispatchStage(std::promise<Status>& started);
    void DispatchOneTask(TaskPair&& pair);
    bool AcquireD2hPipelineToken(CopyStream& stream);
    Status SubmitD2H(CopyStream& stream, TaskPtr task, WaiterPtr waiter, D2hSyncCtx& submitted);
    void SyncStage(std::promise<Status>& started);
    void SyncOneTask(D2hSyncCtx&& ctx);
    Status SyncAndDumpOneTask(D2hSyncCtx&& ctx);
    Status DeviceToHostGatherAsync(std::shared_ptr<Trans::Stream> stream, void** device,
                                   void* host);
    void BackendDumpStage();
};

}  // namespace UC::CacheStore

#endif
