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
#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "yuanrong_helper.h"

namespace {

UC::Detail::BlockId MakeBlock(std::initializer_list<uint8_t> bytes)
{
    UC::Detail::BlockId block{};
    size_t index = 0;
    for (auto byte : bytes) { block[index++] = static_cast<std::byte>(byte); }
    return block;
}

}  // namespace

TEST(YuanRongHelperTest, BuildKeysAndBlobsMapsShardToOneContiguousYuanRongObject)
{
    using namespace UC::YuanRongStore;

    Config config;
    config.nameSpace = "ns";
    config.deviceId = 3;
    config.tensorSizes = {64, 96};

    auto block = MakeBlock({0xab, 0xcd});
    std::array<char, 64> tensor0{};
    std::array<char, 96> tensor1{};
    UC::Detail::TaskDesc desc{
        UC::Detail::Shard{block, 7, {tensor0.data(), tensor1.data()}}
    };

    std::vector<std::string> keys;
    std::vector<datasystem::DeviceBlobList> blobLists;
    auto status = BuildKeysAndBlobs(config, desc, keys, blobLists);

    ASSERT_TRUE(status.Success()) << status.ToString();
    ASSERT_EQ(keys.size(), 1);
    EXPECT_EQ(keys[0], "ucm_ns_abcd0000000000000000000000000000_7");
    ASSERT_EQ(blobLists.size(), 1);
    EXPECT_EQ(blobLists[0].deviceIdx, 3);
    EXPECT_EQ(blobLists[0].srcOffset, 0);
    ASSERT_EQ(blobLists[0].blobs.size(), 2);
    EXPECT_EQ(blobLists[0].blobs[0].pointer, tensor0.data());
    EXPECT_EQ(blobLists[0].blobs[0].size, 64);
    EXPECT_EQ(blobLists[0].blobs[1].pointer, tensor1.data());
    EXPECT_EQ(blobLists[0].blobs[1].size, 96);
}

TEST(YuanRongHelperTest, BuildKeysAndBlobsRejectsAddressCountMismatch)
{
    using namespace UC::YuanRongStore;

    Config config;
    config.nameSpace = "ns";
    config.deviceId = 0;
    config.tensorSizes = {64, 96};
    auto block = MakeBlock({0x01});
    std::array<char, 64> tensor0{};
    UC::Detail::TaskDesc desc{
        UC::Detail::Shard{block, 0, {tensor0.data()}}
    };

    std::vector<std::string> keys;
    std::vector<datasystem::DeviceBlobList> blobLists;
    auto status = BuildKeysAndBlobs(config, desc, keys, blobLists);

    EXPECT_TRUE(status.Failure());
}

TEST(YuanRongHelperTest, DeduplicateYuanRongObjectsKeepsFirstShardForEachKey)
{
    using namespace UC::YuanRongStore;

    Config config;
    config.nameSpace = "ns";
    config.deviceId = 0;
    config.tensorSizes = {64};
    auto block0 = MakeBlock({0x01});
    auto block1 = MakeBlock({0x02});
    std::array<char, 64> tensor0{};
    std::array<char, 64> tensor1{};
    std::array<char, 64> tensor2{};
    UC::Detail::TaskDesc desc{
        UC::Detail::Shard{block0, 0, {tensor0.data()}},
        UC::Detail::Shard{block1, 0, {tensor1.data()}},
        UC::Detail::Shard{block0, 0, {tensor2.data()}},
    };

    std::vector<std::string> keys;
    std::vector<datasystem::DeviceBlobList> blobLists;
    auto status = BuildKeysAndBlobs(config, desc, keys, blobLists);
    ASSERT_TRUE(status.Success()) << status.ToString();

    DeduplicateYuanRongObjects(keys, blobLists, &desc);

    ASSERT_EQ(keys.size(), 2);
    ASSERT_EQ(blobLists.size(), 2);
    ASSERT_EQ(desc.size(), 2);
    EXPECT_EQ(keys[0], "ucm_ns_01000000000000000000000000000000_0");
    EXPECT_EQ(keys[1], "ucm_ns_02000000000000000000000000000000_0");
    EXPECT_EQ(blobLists[0].blobs[0].pointer, tensor0.data());
    EXPECT_EQ(blobLists[1].blobs[0].pointer, tensor1.data());
    EXPECT_EQ(desc[0].addrs[0], tensor0.data());
    EXPECT_EQ(desc[1].addrs[0], tensor1.data());
}

TEST(YuanRongHelperTest, FailedIndexesHandlesTotalFailureAndPartialFailure)
{
    using namespace UC::YuanRongStore;

    std::vector<std::string> keys{"k0", "k1", "k2"};

    EXPECT_TRUE(FailedIndexes(keys, {}, false).empty());

    auto all = FailedIndexes(keys, {}, true);
    ASSERT_EQ(all.size(), 3);
    EXPECT_EQ(all[0], 0);
    EXPECT_EQ(all[1], 1);
    EXPECT_EQ(all[2], 2);

    auto partial = FailedIndexes(keys, {"k2", "k0"}, false);
    ASSERT_EQ(partial.size(), 2);
    EXPECT_EQ(partial[0], 0);
    EXPECT_EQ(partial[1], 2);
}

TEST(YuanRongHelperTest, SelectMissingYuanRongObjectsKeepsAlignedMissingEntries)
{
    using namespace UC::YuanRongStore;

    auto block0 = MakeBlock({0x01});
    auto block1 = MakeBlock({0x02});
    auto block2 = MakeBlock({0x03});
    std::array<char, 64> tensor0{};
    std::array<char, 64> tensor1{};
    std::array<char, 64> tensor2{};
    std::vector<std::string> keys{"k0", "k1", "k2"};
    std::vector<datasystem::DeviceBlobList> blobLists(3);
    blobLists[0].blobs.push_back({tensor0.data(), tensor0.size()});
    blobLists[1].blobs.push_back({tensor1.data(), tensor1.size()});
    blobLists[2].blobs.push_back({tensor2.data(), tensor2.size()});
    UC::Detail::TaskDesc desc{
        UC::Detail::Shard{block0, 0, {tensor0.data()}},
        UC::Detail::Shard{block1, 0, {tensor1.data()}},
        UC::Detail::Shard{block2, 0, {tensor2.data()}},
    };

    auto status = SelectMissingYuanRongObjects({true, false, true}, keys, blobLists, desc);

    ASSERT_TRUE(status.Success()) << status.ToString();
    ASSERT_EQ(keys.size(), 1);
    ASSERT_EQ(blobLists.size(), 1);
    ASSERT_EQ(desc.size(), 1);
    EXPECT_EQ(keys[0], "k1");
    EXPECT_EQ(blobLists[0].blobs[0].pointer, tensor1.data());
    EXPECT_EQ(desc[0].owner, block1);
    EXPECT_EQ(desc[0].addrs[0], tensor1.data());
}

TEST(YuanRongHelperTest, SelectPublishedYuanRongObjectsKeepsAlignedPublishedEntries)
{
    using namespace UC::YuanRongStore;

    std::array<char, 1> tensor0{};
    std::array<char, 1> tensor1{};
    std::array<char, 1> tensor2{};
    std::vector<std::string> keys{"key-a", "key-b", "key-c"};
    UC::Detail::TaskDesc desc;
    desc.push_back(UC::Detail::Shard{MakeBlock({0x01}), 10, {tensor0.data()}});
    desc.push_back(UC::Detail::Shard{MakeBlock({0x02}), 20, {tensor1.data()}});
    desc.push_back(UC::Detail::Shard{MakeBlock({0x03}), 30, {tensor2.data()}});

    auto status = SelectPublishedYuanRongObjects({true, false, true}, keys, desc);

    ASSERT_TRUE(status.Success()) << status.ToString();
    ASSERT_EQ(keys.size(), 2);
    EXPECT_EQ(keys[0], "key-a");
    EXPECT_EQ(keys[1], "key-c");
    ASSERT_EQ(desc.size(), 2);
    EXPECT_EQ(desc[0].index, 10);
    EXPECT_EQ(desc[1].index, 30);
    EXPECT_EQ(desc[0].addrs[0], tensor0.data());
    EXPECT_EQ(desc[1].addrs[0], tensor2.data());
}

TEST(YuanRongHelperTest, RecoveryBatchRangesCoverAllIndexesWithoutOverlap)
{
    using namespace UC::YuanRongStore;

    auto ranges = RecoveryBatchRanges(119, 32);

    ASSERT_EQ(ranges.size(), 4);
    EXPECT_EQ(ranges[0], std::make_pair(size_t{0}, size_t{32}));
    EXPECT_EQ(ranges[1], std::make_pair(size_t{32}, size_t{64}));
    EXPECT_EQ(ranges[2], std::make_pair(size_t{64}, size_t{96}));
    EXPECT_EQ(ranges[3], std::make_pair(size_t{96}, size_t{119}));
    EXPECT_TRUE(RecoveryBatchRanges(0, 32).empty());
    EXPECT_TRUE(RecoveryBatchRanges(10, 0).empty());
}

TEST(YuanRongHelperTest, ComposedBufferHeaderMapsPayloadAfterYuanRongHeader)
{
    using namespace UC::YuanRongStore;

    constexpr size_t memoryAlignment = 4096;
    std::vector<size_t> tensorSizes{64, 96, 128};
    std::vector<uint8_t> buffer(YuanRongComposedObjectSize(tensorSizes, memoryAlignment));

    void* payloadAddress = nullptr;
    auto initStatus = InitYuanRongComposedBuffer("key-a", buffer.data(), buffer.size(), tensorSizes,
                                                 memoryAlignment, payloadAddress);
    ASSERT_TRUE(initStatus.Success()) << initStatus.ToString();

    const auto headerSize = YuanRongHeaderSize(tensorSizes.size(), memoryAlignment);
    EXPECT_EQ(headerSize, memoryAlignment);
    EXPECT_EQ(payloadAddress, buffer.data() + headerSize);

    auto* offsets = reinterpret_cast<uint64_t*>(buffer.data());
    EXPECT_EQ(offsets[0], tensorSizes.size());
    EXPECT_EQ(offsets[1], headerSize);
    EXPECT_EQ(offsets[2], headerSize + tensorSizes[0]);
    EXPECT_EQ(offsets[3], headerSize + tensorSizes[0] + tensorSizes[1]);
    EXPECT_EQ(offsets[4], buffer.size());

    datasystem::MetaInfo metaInfo;
    metaInfo.blobSizeList = {64, 96, 128};
    const void* readPayloadAddress = nullptr;
    auto getStatus = GetYuanRongPayloadAddress("key-a", buffer.data(), buffer.size(), metaInfo,
                                               tensorSizes, memoryAlignment, readPayloadAddress);
    ASSERT_TRUE(getStatus.Success()) << getStatus.ToString();
    EXPECT_EQ(readPayloadAddress, payloadAddress);
}

TEST(YuanRongHelperTest, PayloadAddressRejectsInvalidComposedHeader)
{
    using namespace UC::YuanRongStore;

    constexpr size_t memoryAlignment = 64;
    std::vector<size_t> tensorSizes{64, 96};
    std::vector<uint8_t> buffer(YuanRongComposedObjectSize(tensorSizes, memoryAlignment));
    void* payloadAddress = nullptr;
    ASSERT_TRUE(InitYuanRongComposedBuffer("key-a", buffer.data(), buffer.size(), tensorSizes,
                                           memoryAlignment, payloadAddress)
                    .Success());

    auto* offsets = reinterpret_cast<uint64_t*>(buffer.data());
    offsets[2] += 1;

    datasystem::MetaInfo metaInfo;
    metaInfo.blobSizeList = {64, 96};
    const void* readPayloadAddress = nullptr;
    auto status = GetYuanRongPayloadAddress("key-a", buffer.data(), buffer.size(), metaInfo,
                                            tensorSizes, memoryAlignment, readPayloadAddress);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(readPayloadAddress, nullptr);
}

TEST(YuanRongHelperTest, DirectIoPayloadRequiresAlignedAddressAndSize)
{
    using namespace UC::YuanRongStore;

    constexpr size_t alignment = 4096;
    alignas(alignment) std::array<uint8_t, alignment * 2> buffer{};

    EXPECT_TRUE(
        ValidateYuanRongDirectIoPayload("key-a", buffer.data(), alignment, alignment).Success());
    EXPECT_TRUE(ValidateYuanRongDirectIoPayload("key-a", buffer.data() + 1, alignment, alignment)
                    .Failure());
    EXPECT_TRUE(ValidateYuanRongDirectIoPayload("key-a", buffer.data(), alignment - 1, alignment)
                    .Failure());
    EXPECT_TRUE(ValidateYuanRongDirectIoPayload("key-a", buffer.data(), alignment, 0).Failure());
}
