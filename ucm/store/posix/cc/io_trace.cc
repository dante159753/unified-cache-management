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
#include "io_trace.h"
#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <sys/syscall.h>
#include <unistd.h>
#include "logger/logger.h"
#include "thread/cpu_affinity.h"

namespace UC::PosixStore {

uint64_t IoTrace::MonotonicNanoseconds()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

Status IoTrace::Setup(const Config& config, const std::vector<std::string>& backends)
{
    if (config.ioTraceDir.empty() || config.ioTraceBufferMb == 0 || config.ioTraceBufferMb > 4096 ||
        config.ioTraceFlushMs == 0 || config.ioTraceFlushMs > 60000) {
        return Status::InvalidParam("invalid posix IO trace directory, buffer or flush interval");
    }
    static std::atomic<uint64_t> sessions{0};
    sessionId_ = ++sessions;
    if (config.openConcurrency > 16384 || config.commitConcurrency > 16384 ||
        config.dataTransConcurrency > 16384 || config.posixGcConcurrency > 16384) {
        return Status::InvalidParam("too many workers for posix IO trace");
    }
    laneCount_ = (config.ioEngine == "aio" ? config.openConcurrency + config.commitConcurrency
                                           : 2 * config.dataTransConcurrency) +
                 config.posixGcConcurrency + 32;
    const auto maxRecords = config.ioTraceBufferMb * 1024 * 1024 / sizeof(IoTraceRecord);
    if (laneCount_ > maxRecords / 2) {
        return Status::InvalidParam("posix IO trace buffer is too small for worker count");
    }
    capacity_ = 2;
    while (capacity_ * 2 <= maxRecords / laneCount_) { capacity_ *= 2; }
    lanes_ = std::make_unique<Lane[]>(laneCount_);
    for (size_t i = 0; i < laneCount_; ++i) { lanes_[i].queue.Setup(capacity_); }
    IoTraceHeader header;
    header.monotonicNs = MonotonicNanoseconds();
    header.realtimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    header.sessionId = sessionId_;
    header.pid = getpid();
    header.shardChars = config.dataDirShardBytes;
    header.backendCount = backends.size();
    std::error_code ec;
    std::filesystem::create_directories(config.ioTraceDir, ec);
    if (ec) { return Status::OsApiError(ec.message()); }
    path_ = fmt::format("{}/posix-{}-{}-{}.bin", config.ioTraceDir, header.pid, header.realtimeNs,
                        sessionId_);
    fd_ = ::open(path_.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
    if (fd_ < 0) { return Status::OsApiError(std::to_string(errno)); }
    if (!Write(&header, sizeof(header))) { return Status::Error("write IO trace header failed"); }
    for (const auto& backend : backends) {
        const auto path = std::filesystem::absolute(backend).lexically_normal().string();
        const uint32_t length = path.size();
        if (!Write(&length, sizeof(length)) || !Write(path.data(), length)) {
            return Status::Error("write IO trace backend failed");
        }
    }
    writer_ = std::thread([this, interval = config.ioTraceFlushMs] { WriterLoop(interval); });
    UC_INFO_UNLIMITED("Posix IO tracing is enabled: path={}, buffer_bytes={}, lanes={}.", path_,
                      laneCount_ * capacity_ * sizeof(IoTraceRecord), laneCount_);
    return Status::OK();
}

IoTrace::~IoTrace()
{
    // Owners must stop IO and GC producers before destroying the layout/trace.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_one();
    if (writer_.joinable()) { writer_.join(); }
    if (fd_ >= 0) { ::close(fd_); }
}

void IoTrace::Record(IoTraceRecord record)
{
    struct Registration {
        uint64_t session;
        size_t lane;
    };
    thread_local std::vector<Registration> registrations;
    auto it = std::find_if(registrations.begin(), registrations.end(),
                           [this](const auto& entry) { return entry.session == sessionId_; });
    if (it == registrations.end()) {
        auto lane = nextLane_.fetch_add(1, std::memory_order_relaxed);
        if (lane < laneCount_) { lanes_[lane].tid = syscall(SYS_gettid); }
        registrations.push_back({sessionId_, lane});
        it = registrations.end() - 1;
    }
    if (it->lane >= laneCount_) {
        unregisteredDrops_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    auto& lane = lanes_[it->lane];
    const auto produced = lane.produced.load(std::memory_order_relaxed) + 1;
    record.ioId = (static_cast<uint64_t>(it->lane) << 48) | produced;
    record.tid = lane.tid;
    const auto consumed = lane.consumed.load(std::memory_order_relaxed);
    const auto accepted = produced - lane.dropped.load(std::memory_order_relaxed);
    if (!lane.queue.TryPush(std::move(record))) {
        lane.dropped.fetch_add(1, std::memory_order_relaxed);
    } else {
        auto depth = std::min<uint64_t>(accepted - consumed, capacity_ - 1);
        if (depth > lane.highWater.load(std::memory_order_relaxed)) {
            lane.highWater.store(depth, std::memory_order_relaxed);
        }
    }
    lane.produced.store(produced, std::memory_order_relaxed);
}

bool IoTrace::Write(const void* data, size_t bytes)
{
    if (writeErrors_) { return false; }
    const auto* buffer = static_cast<const char*>(data);
    while (bytes) {
        const auto count = ::write(fd_, buffer, bytes);
        if (count < 0 && errno == EINTR) { continue; }
        if (count <= 0) {
            ++writeErrors_;
            UC_ERROR_UNLIMITED(
                "Posix IO trace write failed: path={}, errno={}; trace is incomplete.", path_,
                count < 0 ? errno : EIO);
            return false;
        }
        buffer += count;
        bytes -= count;
    }
    return true;
}

IoTraceRecord IoTrace::Stats(bool final) const
{
    IoTraceRecord record{};
    record.op = IoTraceOp::STATS;
    record.startNs = record.endNs = MonotonicNanoseconds();
    record.flags = final ? TRACE_FINAL : 0;
    record.taskId = unregisteredDrops_.load(std::memory_order_relaxed);
    record.shard = record.taskId;
    for (size_t i = 0; i < laneCount_; ++i) {
        record.taskId += lanes_[i].produced.load(std::memory_order_relaxed);
        record.shard += lanes_[i].dropped.load(std::memory_order_relaxed);
        record.offset =
            std::max(record.offset, lanes_[i].highWater.load(std::memory_order_relaxed));
    }
    record.ioId = written_;
    record.requestedBytes = writeErrors_;
    return record;
}

void IoTrace::WriterLoop(size_t flushMs)
{
    CpuAffinity::SetCurrentThreadName("ucm_io_trace");
    std::vector<IoTraceRecord> batch;
    batch.reserve(4096);
    auto lastStats = MonotonicNanoseconds();
    for (;;) {
        bool stopping;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping = stop_;
        }
        bool drained = true;
        for (size_t i = 0; i < laneCount_; ++i) {
            IoTraceRecord record;
            size_t count = 0;
            while (count < 4096 && lanes_[i].queue.TryPop(record)) {
                batch.push_back(record);
                ++count;
                if (batch.size() == batch.capacity()) {
                    if (Write(batch.data(), batch.size() * sizeof(record))) {
                        written_ += batch.size();
                    }
                    batch.clear();
                }
            }
            lanes_[i].consumed.fetch_add(count, std::memory_order_relaxed);
            if (count) { drained = false; }
        }
        if (!batch.empty()) {
            if (Write(batch.data(), batch.size() * sizeof(IoTraceRecord))) {
                written_ += batch.size();
            }
            batch.clear();
        }
        if (stopping && drained) { break; }
        if (MonotonicNanoseconds() - lastStats >= 1000000000ULL) {
            const auto stats = Stats(false);
            Write(&stats, sizeof(stats));
            lastStats = stats.startNs;
        }
        if (drained && !stopping) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(flushMs), [this] { return stop_; });
        }
    }
    const auto stats = Stats(true);
    Write(&stats, sizeof(stats));
    UC_INFO_UNLIMITED(
        "Posix IO trace stopped: path={}, produced={}, written={}, dropped={}, "
        "queue_high_water={}, write_errors={}.",
        path_, stats.taskId, written_, stats.shard, stats.offset, writeErrors_);
}

IoTraceSpan::IoTraceSpan(const IoTraceContext& context, IoTraceOp op, uint64_t offset,
                         uint64_t requestedBytes)
    : trace_(context.trace)
{
    if (!trace_) { return; }
    record_ = {};
    record_.op = op;
    record_.block = context.block;
    record_.taskId = context.taskId;
    record_.shard = context.shard;
    record_.backend = context.backend;
    record_.flags = context.flags;
    record_.offset = offset;
    record_.requestedBytes = requestedBytes;
    record_.startNs = IoTrace::MonotonicNanoseconds();
}

void IoTraceSpan::Finish(int64_t result, int32_t error, uint16_t flags) const
{
    if (!trace_) { return; }
    auto record = record_;
    record.endNs = IoTrace::MonotonicNanoseconds();
    record.result = result;
    record.error = error;
    record.flags |= flags;
    trace_->Record(record);
}

void CloseTraced(int fd, const IoTraceContext& context)
{
    IoTraceSpan span(context, IoTraceOp::CLOSE);
    const auto result = ::close(fd);
    const auto error = result < 0 ? errno : 0;
    span.Finish(result, error);
}

}  // namespace UC::PosixStore
