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
#ifndef UNIFIEDCACHE_YUANRONG_STORE_CC_DUMP_QUEUE_H
#define UNIFIEDCACHE_YUANRONG_STORE_CC_DUMP_QUEUE_H

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "datasystem/hetero_client.h"
#include "datasystem/kv/read_only_buffer.h"
#include "datasystem/kv_client.h"
#include "datasystem/utils/optional.h"
#include "template/hashset.h"
#include "template/spsc_ring_queue.h"
#include "thread/latch.h"
#include "trans/stream.h"
#include "trans_task.h"
#include "ucmstore_v1.h"
#include "yuanrong_config.h"

namespace UC::YuanRongStore {

class DumpQueue {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using TaskIdSet = HashSet<Detail::TaskHandle>;

    struct DumpContext {
        Detail::TaskHandle ownerTaskId{0};
        Detail::TaskHandle backendTaskId{0};
        std::vector<datasystem::Optional<datasystem::ReadOnlyBuffer>> buffers;
    };

    alignas(64) std::atomic_bool stop_{false};
    TaskIdSet* failureSet_{nullptr};
    Config config_{};
    std::shared_ptr<datasystem::HeteroClient> heteroClient_;
    std::shared_ptr<datasystem::KVClient> kvClient_;
    StoreV1* backend_{nullptr};
    SpscRingQueue<TaskPair> waiting_;
    SpscRingQueue<DumpContext> reaping_;
    std::mutex submitMutex_;
    std::thread worker_;
    std::thread reaper_;

public:
    ~DumpQueue();
    Status Setup(const Config& config, TaskIdSet* failureSet,
                 std::shared_ptr<datasystem::HeteroClient> heteroClient,
                 std::shared_ptr<datasystem::KVClient> kvClient);
    void Submit(TaskPtr task, WaiterPtr waiter);

private:
    void Close();
    void WorkerStage(std::promise<Status>& started);
    void ReaperStage();
    void RunOne(std::shared_ptr<Trans::Stream>& prerequisiteStream, TaskPair&& pair);
    Status DumpOne(TaskPtr task, const std::shared_ptr<Trans::Stream>& prerequisiteStream);
    void Reap(DumpContext&& context);
    static void ReleaseBuffers(
        std::vector<datasystem::Optional<datasystem::ReadOnlyBuffer>>& buffers);
};

}  // namespace UC::YuanRongStore

#endif
