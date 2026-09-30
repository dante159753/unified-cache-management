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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_IO_TRACE_H
#define UNIFIEDCACHE_POSIX_STORE_CC_IO_TRACE_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include "global_config.h"
#include "status/status.h"
#include "template/spsc_ring_queue.h"
#include "type/types.h"

namespace UC::PosixStore {

enum class IoTraceOp : uint16_t {
    READ = 1,
    WRITE,
    OPEN,
    CLOSE,
    RENAME,
    REMOVE,
    SYNC,
    TIMEOUT,
    STATS
};
enum IoTraceFlag : uint16_t {
    TRACE_TMP = 1,
    TRACE_ASYNC = 2,
    TRACE_SUBMIT_FAILED = 4,
    TRACE_NOT_STARTED = 8,
    TRACE_HEALTH = 16,
    TRACE_GC = 32,
    TRACE_FINAL = 32768
};

// Version 1 uses native-endian, fixed-width fields; the header carries the endian marker.
struct IoTraceHeader {
    char magic[8]{'U', 'C', 'M', 'I', 'O', '0', '0', '1'};
    uint32_t endian{0x01020304};
    uint32_t recordBytes{96};
    uint64_t realtimeNs{0};
    uint64_t monotonicNs{0};
    uint64_t sessionId{0};
    uint64_t pid{0};
    uint32_t shardChars{0};
    uint32_t backendCount{0};
    uint64_t reserved{0};
};
struct IoTraceRecord {
    uint64_t startNs;
    uint64_t endNs;
    uint64_t taskId;
    uint64_t ioId;
    uint64_t shard;
    uint64_t offset;
    uint64_t requestedBytes;
    int64_t result;
    Detail::BlockId block;
    uint32_t backend;
    int32_t error;
    uint32_t tid;
    IoTraceOp op;
    uint16_t flags;
};
static_assert(sizeof(IoTraceHeader) == 64);
static_assert(sizeof(IoTraceRecord) == 96);

class IoTrace {
    struct Lane {
        SpscRingQueue<IoTraceRecord> queue;
        std::atomic<uint64_t> produced{0};
        std::atomic<uint64_t> dropped{0};
        std::atomic<uint64_t> consumed{0};
        std::atomic<uint64_t> highWater{0};
        uint32_t tid{0};
    };
    std::unique_ptr<Lane[]> lanes_;
    size_t laneCount_{0};
    size_t capacity_{0};
    std::atomic<size_t> nextLane_{0};
    std::atomic<uint64_t> unregisteredDrops_{0};
    uint64_t sessionId_{0};
    uint64_t written_{0};
    uint64_t writeErrors_{0};
    int fd_{-1};
    std::string path_;
    std::thread writer_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_{false};

public:
    ~IoTrace();
    Status Setup(const Config& config, const std::vector<std::string>& backends);
    void Record(IoTraceRecord record);
    const std::string& Path() const { return path_; }
    static uint64_t MonotonicNanoseconds();

private:
    void WriterLoop(size_t flushMs);
    bool Write(const void* data, size_t bytes);
    IoTraceRecord Stats(bool final) const;
};

struct IoTraceContext {
    IoTrace* trace{nullptr};
    Detail::BlockId block{};
    uint64_t taskId{0};
    uint64_t shard{UINT64_MAX};
    uint32_t backend{0};
    uint16_t flags{0};
};

class IoTraceSpan {
    IoTrace* trace_{nullptr};
    IoTraceRecord record_;

public:
    IoTraceSpan(const IoTraceContext& context, IoTraceOp op, uint64_t offset = 0,
                uint64_t requestedBytes = 0);
    void Finish(int64_t result, int32_t error, uint16_t flags = 0) const;
};

void CloseTraced(int fd, const IoTraceContext& context);

}  // namespace UC::PosixStore
#endif
