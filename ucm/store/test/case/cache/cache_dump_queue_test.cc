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
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <string>
#include "cache/cc/dump_queue.h"
#include "detail/data_generator.h"
#include "detail/mock_store.h"
#include "detail/random.h"
#include "detail/types_helper.h"

class UCCacheDumpQueueTest : public testing::Test {
public:
    UC::Test::Detail::Random rd;
    static UC::Detail::TaskHandle NextId(UC::Detail::TaskDesc = {})
    {
        static std::atomic<size_t> id{1};
        return id.fetch_add(1, std::memory_order_relaxed);
    }
    static UC::CacheStore::Config MakeConfig(UC::StoreV1* backend, const std::string& uniqueId,
                                             size_t tensorSize = 32768)
    {
        UC::CacheStore::Config config;
        config.storeBackend = backend;
        config.tensorSizes = {tensorSize};
        config.shardSize = tensorSize;
        config.blockSize = config.shardSize;
        config.deviceId = 0;
        config.bufferCapacity = config.shardSize * 1024;
        config.uniqueId = uniqueId;
        config.shareBufferEnable = true;
        return config;
    }
    static UC::Detail::TaskDesc MakeDesc(const std::string& block, size_t shardIdx, void* buffer)
    {
        auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId(block.c_str());
        return UC::Detail::TaskDesc{
            {blockId, shardIdx, {buffer}}
        };
    }
    static std::string MakeBlockName(size_t i)
    {
        return std::to_string(i) + "1234567890123456";
    }
};

TEST_F(UCCacheDumpQueueTest, DumpOneBlock)
{
    using namespace UC::CacheStore;
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Dump).WillOnce(testing::Invoke(NextId));
    UC::Latch finish{};
    finish.Up();
    EXPECT_CALL(backend, Wait).WillOnce(testing::Invoke([&finish](auto) {
        finish.Done();
        return UC::Status::OK();
    }));
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    size_t tensorSize = 32768;
    auto config = MakeConfig(&backend, rd.RandomString(10), tensorSize);
    config.dumpD2hPipelineDepth = 1;
    TransBuffer buffer;
    DumpQueue dumpQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = dumpQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    constexpr size_t shardIdx = 0;
    UC::Test::Detail::DataGenerator data{1, config.blockSize};
    data.Generate();
    auto desc = MakeDesc("a1b2c3d4e5f6789012345678901234ab", shardIdx, data.Buffer());
    auto task = std::make_shared<TransTask>(TransTask::Type::DUMP, desc);
    auto waiter = std::make_shared<UC::Latch>();
    dumpQ.Submit(task, waiter);
    waiter->Wait();
    ASSERT_FALSE(failureSet.Contains(task->id));
    finish.Wait();
}

TEST_F(UCCacheDumpQueueTest, DumpBlockWhileBackendSubmitFailed)
{
    using namespace UC::CacheStore;
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Dump).WillOnce(testing::Return(UC::Status::Error()));
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    size_t tensorSize = 32768;
    auto config = MakeConfig(&backend, rd.RandomString(10), tensorSize);
    TransBuffer buffer;
    DumpQueue dumpQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = dumpQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    constexpr size_t shardIdx = 0;
    UC::Test::Detail::DataGenerator data{1, config.blockSize};
    data.Generate();
    auto desc = MakeDesc("a1b2c3d4e5f6789012345678901234ab", shardIdx, data.Buffer());
    auto task = std::make_shared<TransTask>(TransTask::Type::DUMP, desc);
    auto waiter = std::make_shared<UC::Latch>();
    dumpQ.Submit(task, waiter);
    waiter->Wait();
    ASSERT_TRUE(failureSet.Contains(task->id));
}

TEST_F(UCCacheDumpQueueTest, DumpMultipleBlocksThroughStreamGroups)
{
    using namespace UC::CacheStore;
    UC::Test::Detail::MockStore backend;
    constexpr size_t nTask = 8;
    UC::Latch finish{};
    for (size_t i = 0; i < nTask; ++i) { finish.Up(); }
    EXPECT_CALL(backend, Dump).Times(nTask).WillRepeatedly(testing::Invoke(NextId));
    EXPECT_CALL(backend, Wait).Times(nTask).WillRepeatedly(testing::Invoke([&finish](auto) {
        finish.Done();
        return UC::Status::OK();
    }));
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    auto config = MakeConfig(&backend, rd.RandomString(10));
    config.dumpD2hPipelineDepth = 2;
    TransBuffer buffer;
    DumpQueue dumpQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = dumpQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    UC::Test::Detail::DataGenerator data{1, config.blockSize};
    data.Generate();
    std::array<std::shared_ptr<UC::Latch>, nTask> waiters;
    std::array<std::shared_ptr<TransTask>, nTask> tasks;
    for (size_t i = 0; i < nTask; ++i) {
        auto block = MakeBlockName(i);
        auto desc = MakeDesc(block, 0, data.Buffer());
        tasks[i] = std::make_shared<TransTask>(TransTask::Type::DUMP, desc);
        waiters[i] = std::make_shared<UC::Latch>();
        dumpQ.Submit(tasks[i], waiters[i]);
    }
    for (size_t i = 0; i < nTask; ++i) {
        waiters[i]->Wait();
        ASSERT_FALSE(failureSet.Contains(tasks[i]->id));
    }
    ASSERT_TRUE(finish.WaitForDuration(5000));
}

TEST_F(UCCacheDumpQueueTest, DumpWaitsEveryBackendTaskWithoutHandleOrdering)
{
    using namespace UC::CacheStore;
    UC::Test::Detail::MockStore backend;
    UC::Latch finish{};
    finish.Up();
    finish.Up();
    {
        testing::InSequence seq;
        EXPECT_CALL(backend, Dump).WillOnce(testing::Return(UC::Detail::TaskHandle{2}));
        EXPECT_CALL(backend, Dump).WillOnce(testing::Return(UC::Detail::TaskHandle{1}));
    }
    EXPECT_CALL(backend, Wait(2)).WillOnce(testing::Invoke([&finish](auto) {
        finish.Done();
        return UC::Status::OK();
    }));
    EXPECT_CALL(backend, Wait(1)).WillOnce(testing::Invoke([&finish](auto) {
        finish.Done();
        return UC::Status::OK();
    }));
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    auto config = MakeConfig(&backend, rd.RandomString(10));
    TransBuffer buffer;
    DumpQueue dumpQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = dumpQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    UC::Test::Detail::DataGenerator data{1, config.blockSize};
    data.Generate();
    std::array<std::shared_ptr<UC::Latch>, 2> waiters;
    std::array<std::shared_ptr<TransTask>, 2> tasks;
    for (size_t i = 0; i < tasks.size(); ++i) {
        auto block = MakeBlockName(i);
        auto desc = MakeDesc(block, 0, data.Buffer());
        tasks[i] = std::make_shared<TransTask>(TransTask::Type::DUMP, desc);
        waiters[i] = std::make_shared<UC::Latch>();
        dumpQ.Submit(tasks[i], waiters[i]);
    }
    for (auto& waiter : waiters) { waiter->Wait(); }
    ASSERT_TRUE(finish.WaitForDuration(5000));
}

TEST_F(UCCacheDumpQueueTest, DumpKeepsBufferHandleUntilBackendWaitFinishes)
{
    using namespace UC::CacheStore;
    UC::Test::Detail::MockStore backend;
    UC::Latch waitStarted{};
    UC::Latch releaseWait{};
    UC::Latch waitFinished{};
    waitStarted.Up();
    releaseWait.Up();
    waitFinished.Up();
    EXPECT_CALL(backend, Dump).WillOnce(testing::Return(UC::Detail::TaskHandle{1}));
    EXPECT_CALL(backend, Wait).WillOnce(testing::Invoke([&](auto) {
        waitStarted.Done();
        releaseWait.Wait();
        waitFinished.Done();
        return UC::Status::OK();
    }));
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    auto config = MakeConfig(&backend, rd.RandomString(10));
    TransBuffer buffer;
    DumpQueue dumpQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = dumpQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    constexpr const char* block = "a1b2c3d4e5f6789012345678901234ab";
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId(block);
    UC::Test::Detail::DataGenerator data{1, config.blockSize};
    data.Generate();
    auto desc = MakeDesc(block, 0, data.Buffer());
    auto task = std::make_shared<TransTask>(TransTask::Type::DUMP, desc);
    auto waiter = std::make_shared<UC::Latch>();
    dumpQ.Submit(task, waiter);
    waiter->Wait();
    auto started = waitStarted.WaitForDuration(5000);
    EXPECT_TRUE(started);
    if (started) {
        auto handle = buffer.Get(blockId, 0);
        ASSERT_FALSE(handle.Owner());
    }
    releaseWait.Done();
    ASSERT_TRUE(waitFinished.WaitForDuration(5000));
    auto handle = buffer.Get(blockId, 0);
    ASSERT_TRUE(handle.Owner());
}
