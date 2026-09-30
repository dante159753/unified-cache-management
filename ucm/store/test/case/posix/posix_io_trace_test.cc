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
#include <cstring>
#include <filesystem>
#include <fstream>
#include "detail/path_base.h"
#include "posix/cc/io_trace.h"
#include "posix/cc/posix_file.h"
#include "posix/cc/trans_manager.h"
#include "ucmstore_v1.h"

extern "C" UC::StoreV1* MakePosixStore();

using namespace UC::PosixStore;

class UCPosixIoTraceTest : public UC::Test::Detail::PathBase {
protected:
    Config MakeConfig(const std::string& engine = "psync")
    {
        Config config;
        config.storageBackends = {Path()};
        config.dataDirShardBytes = 0;
        config.ioEngine = engine;
        config.ioTraceEnable = true;
        config.ioTraceDir = Path() + "trace";
        config.ioTraceBufferMb = 4;
        config.ioTraceFlushMs = 10;
        config.dataTransConcurrency = config.openConcurrency = config.commitConcurrency = 2;
        config.tensorSize = 4096;
        config.shardSize = 8192;
        config.blockSize = 16384;
        return config;
    }

    std::vector<IoTraceRecord> ReadRecords(const Config& config)
    {
        std::vector<IoTraceRecord> records;
        for (const auto& file : std::filesystem::directory_iterator(config.ioTraceDir)) {
            std::ifstream input(file.path(), std::ios::binary);
            IoTraceHeader header;
            input.read(reinterpret_cast<char*>(&header), sizeof(header));
            EXPECT_EQ(std::string(header.magic, 8), "UCMIO001");
            EXPECT_EQ(header.recordBytes, sizeof(IoTraceRecord));
            for (size_t i = 0; i < header.backendCount; ++i) {
                uint32_t length;
                input.read(reinterpret_cast<char*>(&length), sizeof(length));
                input.seekg(length, std::ios::cur);
            }
            IoTraceRecord record;
            while (input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
                if (record.op != IoTraceOp::STATS || (record.flags & TRACE_FINAL)) {
                    records.push_back(record);
                }
            }
            EXPECT_EQ(input.gcount(), 0);
        }
        return records;
    }

    void RoundTrip(const std::string& engine)
    {
        auto config = MakeConfig(engine);
        UC::Detail::BlockId block{};
        block[0] = std::byte{0xab};
        std::vector<uint64_t> dumps, loads;
        alignas(4096) char input[16384];
        alignas(4096) char output[16384]{};
        memset(input, 0x6b, sizeof(input));
        {
            SpaceLayout layout;
            ASSERT_TRUE(layout.Setup(config).Success());
            {
                TransManager manager;
                ASSERT_TRUE(manager.Setup(config, &layout).Success());
                for (size_t layer = 0; layer < 2; ++layer) {
                    UC::Detail::TaskDesc desc;
                    std::vector<void*> buffers{input + layer * 8192};
                    if (engine == "psync") { buffers.push_back(input + layer * 8192 + 4096); }
                    desc.push_back({block, layer, buffers});
                    auto task = manager.GetIoEngine()->Submit({TransTask::Type::DUMP, desc});
                    ASSERT_TRUE(task.HasValue());
                    dumps.push_back(task.Value());
                    ASSERT_TRUE(manager.GetIoEngine()->Wait(task.Value()).Success());
                }
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (!std::filesystem::exists(layout.DataFilePath(block, false)) &&
                       std::chrono::steady_clock::now() < deadline) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                ASSERT_TRUE(std::filesystem::exists(layout.DataFilePath(block, false)));
                for (size_t layer = 0; layer < 2; ++layer) {
                    UC::Detail::TaskDesc desc;
                    std::vector<void*> buffers{output + layer * 8192};
                    if (engine == "psync") { buffers.push_back(output + layer * 8192 + 4096); }
                    desc.push_back({block, layer, buffers});
                    auto task = manager.GetIoEngine()->Submit({TransTask::Type::LOAD, desc});
                    ASSERT_TRUE(task.HasValue());
                    loads.push_back(task.Value());
                    ASSERT_TRUE(manager.GetIoEngine()->Wait(task.Value()).Success());
                }
                ASSERT_EQ(memcmp(input, output, sizeof(input)), 0);
            }
            ASSERT_TRUE(layout.RemoveFile(block).Success());
            ASSERT_TRUE(layout.CommitFile(block, true, 999).Failure());
        }
        auto records = ReadRecords(config);
        ASSERT_FALSE(records.empty());
        auto stats = records.back();
        ASSERT_EQ(stats.op, IoTraceOp::STATS);
        EXPECT_TRUE(stats.flags & TRACE_FINAL);
        EXPECT_EQ(stats.shard, 0);
        EXPECT_EQ(stats.taskId, stats.ioId);
        size_t readBytes = 0, writeBytes = 0, commits = 0, removals = 0;
        for (const auto& record : records) {
            if (record.op == IoTraceOp::STATS) { continue; }
            EXPECT_EQ(record.block, block);
            EXPECT_GE(record.endNs, record.startNs);
            EXPECT_GT(record.tid, 0);
            if (record.op == IoTraceOp::READ || record.op == IoTraceOp::WRITE) {
                const auto dump = record.op == IoTraceOp::WRITE;
                ASSERT_LT(record.shard, 2);
                EXPECT_EQ(record.taskId, dump ? dumps[record.shard] : loads[record.shard]);
                EXPECT_EQ(record.error, 0);
                EXPECT_EQ(record.result, record.requestedBytes);
                EXPECT_EQ(bool(record.flags & TRACE_TMP), dump);
                EXPECT_EQ(bool(record.flags & TRACE_ASYNC), engine == "aio");
                EXPECT_GE(record.offset, record.shard * 8192);
                EXPECT_LT(record.offset, (record.shard + 1) * 8192);
                (dump ? writeBytes : readBytes) += record.result;
            }
            if (record.op == IoTraceOp::RENAME && record.error == 0) {
                EXPECT_EQ(record.taskId, dumps.back());
                ++commits;
            }
            if (record.op == IoTraceOp::REMOVE && (record.flags & TRACE_GC)) {
                EXPECT_EQ(record.taskId, 0);
                ++removals;
            }
            if (record.taskId == 999) { EXPECT_EQ(record.error, ENOENT); }
        }
        EXPECT_EQ(readBytes, sizeof(output));
        EXPECT_EQ(writeBytes, sizeof(input));
        EXPECT_EQ(commits, 1);
        EXPECT_EQ(removals, 1);
    }
};

TEST_F(UCPosixIoTraceTest, PsyncLayerwiseRoundTrip) { RoundTrip("psync"); }
TEST_F(UCPosixIoTraceTest, AioLayerwiseRoundTrip) { RoundTrip("aio"); }

TEST_F(UCPosixIoTraceTest, ShortReadAndOpenError)
{
    auto config = MakeConfig();
    {
        SpaceLayout layout;
        ASSERT_TRUE(layout.Setup(config).Success());
        UC::Detail::BlockId block{};
        PosixFile file(layout.DataFilePath(block, false), layout.TraceContext(block, 123, 0));
        ASSERT_EQ(file.Open(O_RDONLY), UC::Status::NotFound());
        ASSERT_TRUE(file.Open(O_CREAT | O_RDWR).Success());
        char buffer[32]{};
        ASSERT_TRUE(file.Write(buffer, 8, 0).Success());
        ASSERT_TRUE(file.Read(buffer, sizeof(buffer), 0).Failure());
    }
    auto records = ReadRecords(config);
    size_t shortReads = 0, failedOpens = 0;
    for (const auto& record : records) {
        if (record.op == IoTraceOp::READ) {
            EXPECT_EQ(record.taskId, 123);
            EXPECT_EQ(record.result, 8);
            EXPECT_EQ(record.requestedBytes, 32);
            EXPECT_EQ(record.error, 0);
            ++shortReads;
        }
        if (record.op == IoTraceOp::OPEN && record.error == ENOENT) { ++failedOpens; }
    }
    EXPECT_EQ(shortReads, 1);
    EXPECT_EQ(failedOpens, 1);
}

TEST_F(UCPosixIoTraceTest, ConcurrentProducersAndShutdownDrain)
{
    auto config = MakeConfig();
    config.ioTraceBufferMb = 16;
    config.ioTraceFlushMs = 60000;
    {
        IoTrace trace;
        ASSERT_TRUE(trace.Setup(config, config.storageBackends).Success());
        std::vector<std::thread> producers;
        for (size_t task = 1; task <= 8; ++task) {
            producers.emplace_back([&trace, task] {
                for (size_t i = 0; i < 500; ++i) {
                    IoTraceSpan({&trace, {}, task}, IoTraceOp::READ, i * 4096, 4096)
                        .Finish(4096, 0);
                }
            });
        }
        for (auto& producer : producers) { producer.join(); }
    }
    auto records = ReadRecords(config);
    ASSERT_EQ(records.size(), 4001);
    EXPECT_EQ(records.back().taskId, 4000);
    EXPECT_EQ(records.back().ioId, 4000);
    EXPECT_EQ(records.back().shard, 0);
}

TEST_F(UCPosixIoTraceTest, QueueOverflowIsCounted)
{
    auto config = MakeConfig();
    config.ioTraceBufferMb = 1;
    config.ioTraceFlushMs = 60000;
    {
        IoTrace trace;
        ASSERT_TRUE(trace.Setup(config, config.storageBackends).Success());
        for (size_t i = 0; i < 100000; ++i) {
            IoTraceSpan({&trace}, IoTraceOp::WRITE, i, 1).Finish(1, 0);
        }
    }
    auto records = ReadRecords(config);
    ASSERT_FALSE(records.empty());
    EXPECT_EQ(records.back().taskId, 100000);
    EXPECT_GT(records.back().shard, 0);
    EXPECT_EQ(records.back().ioId + records.back().shard, 100000);
}

TEST_F(UCPosixIoTraceTest, DisabledCreatesNoTrace)
{
    auto config = MakeConfig();
    config.ioTraceEnable = false;
    SpaceLayout layout;
    ASSERT_TRUE(layout.Setup(config).Success());
    EXPECT_EQ(layout.TraceContext({}).trace, nullptr);
    EXPECT_FALSE(std::filesystem::exists(config.ioTraceDir));
}

TEST_F(UCPosixIoTraceTest, StoreConfigAndHealthOperations)
{
    auto config = MakeConfig();
    {
        std::unique_ptr<UC::StoreV1> store(MakePosixStore());
        UC::Detail::Dictionary options;
        options.Set("storage_backends", config.storageBackends);
        options.SetNumber("data_dir_shard_bytes", size_t(0));
        options.Set("io_direct", false);
        options.Set("posix_io_trace_enable", true);
        options.Set("posix_io_trace_dir", config.ioTraceDir);
        options.SetNumber("posix_io_trace_buffer_mb", config.ioTraceBufferMb);
        options.SetNumber("posix_io_trace_flush_ms", config.ioTraceFlushMs);
        ASSERT_TRUE(store->Setup(options).Success());
        ASSERT_TRUE(store->CheckHealth().Success());
    }
    auto records = ReadRecords(config);
    ASSERT_EQ(records.size(), 7);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_TRUE(records[i].flags & TRACE_HEALTH);
        EXPECT_EQ(records[i].taskId, 0);
        EXPECT_EQ(records[i].error, 0);
    }
    EXPECT_EQ(records[0].op, IoTraceOp::OPEN);
    EXPECT_EQ(records[1].op, IoTraceOp::WRITE);
    EXPECT_EQ(records[2].op, IoTraceOp::SYNC);
    EXPECT_EQ(records[3].op, IoTraceOp::READ);
    EXPECT_EQ(records[4].op, IoTraceOp::CLOSE);
    EXPECT_EQ(records[5].op, IoTraceOp::REMOVE);
}

TEST_F(UCPosixIoTraceTest, InstancesUseSeparateFilesAndProducerRegistrations)
{
    auto config = MakeConfig();
    {
        IoTrace first, second;
        ASSERT_TRUE(first.Setup(config, config.storageBackends).Success());
        ASSERT_TRUE(second.Setup(config, config.storageBackends).Success());
        EXPECT_NE(first.Path(), second.Path());
        IoTraceSpan({&first, {}, 1}, IoTraceOp::READ).Finish(0, 0);
        IoTraceSpan({&second, {}, 2}, IoTraceOp::READ).Finish(0, 0);
        IoTraceSpan({&first, {}, 3}, IoTraceOp::READ).Finish(0, 0);
    }
    auto records = ReadRecords(config);
    ASSERT_EQ(records.size(), 5);
    uint64_t tasks = 0, dropped = 0, written = 0;
    for (const auto& record : records) {
        if (record.op == IoTraceOp::STATS) {
            dropped += record.shard;
            written += record.ioId;
        } else {
            tasks += record.taskId;
        }
    }
    EXPECT_EQ(tasks, 6);
    EXPECT_EQ(written, 3);
    EXPECT_EQ(dropped, 0);
}

TEST_F(UCPosixIoTraceTest, AioSubmitFailureAndCancellation)
{
    auto config = MakeConfig("aio");
    {
        SpaceLayout layout;
        ASSERT_TRUE(layout.Setup(config).Success());
        AioImpl aio;
        ASSERT_TRUE(aio.Setup(10).Success());
        alignas(4096) char buffer[4096]{};
        auto makeIo = [&] {
            AioImpl::Io io{};
            io.fd = -1;
            io.length = sizeof(buffer);
            io.buffer = buffer;
            io.tag = 321;
            io.trace = layout.TraceContext({}, io.tag, 0, TRACE_ASYNC);
            io.callback = [](AioImpl::Result result) { EXPECT_EQ(result.error, ECANCELED); };
            return io;
        };
        TestHooks::SetAioSubmitHook([](aio_context_t, int64_t, iocb**) {
            errno = EIO;
            return -1;
        });
        EXPECT_TRUE(aio.ReadAsync(makeIo()).Failure());
        TestHooks::SetAioSubmitHook([](aio_context_t, int64_t, iocb**) {
            errno = EAGAIN;
            return -1;
        });
        EXPECT_EQ(aio.WriteAsync(makeIo()), UC::Status::Timeout());
        TestHooks::SetAioSubmitHook([](aio_context_t, int64_t, iocb**) { return 1; });
        TestHooks::SetAioCancelHook([](aio_context_t, iocb*, io_event*) { return 0; });
        EXPECT_TRUE(aio.ReadAsync(makeIo()).Success());
        aio.CancelTask(321);
        TestHooks::ClearAioHooks();
    }
    auto records = ReadRecords(config);
    ASSERT_EQ(records.size(), 4);
    EXPECT_EQ(records[0].error, EIO);
    EXPECT_TRUE(records[0].flags & TRACE_SUBMIT_FAILED);
    EXPECT_EQ(records[1].error, ETIMEDOUT);
    EXPECT_TRUE(records[1].flags & TRACE_SUBMIT_FAILED);
    EXPECT_GE(records[1].endNs - records[1].startNs, 10000000);
    EXPECT_EQ(records[2].error, ECANCELED);
    EXPECT_FALSE(records[2].flags & TRACE_SUBMIT_FAILED);
}
