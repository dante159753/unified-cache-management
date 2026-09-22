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
#include <algorithm>
#include <ascend_hal.h>
#include <chrono>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <vector>
#include "data_strategy.h"
#include "trans/ascend/hal/hal_host_buffers.h"
#include "trans/device.h"

struct drv_mem_handle {
    uint64_t token;
    bool imported;
};

namespace {

using testing::_;
using testing::ElementsAre;
using testing::HasSubstr;
using testing::Return;
using UC::Expected;
using UC::Status;
using UC::Cache2::CtrlLayout;
using UC::Cache2::DataStrategy;
using UC::Trans::HalHostBuffers;

constexpr size_t MiB = size_t{1} << 20;
constexpr size_t GiB = size_t{1} << 30;
constexpr uint64_t localToken = 0xabcdef1200000001ULL;
constexpr uint64_t peerToken = 0x1234567800000000ULL;

struct FakeHal {
    std::map<std::string, std::set<size_t>> failures;
    std::map<std::string, size_t> calls;
    std::map<void*, size_t> reservations;
    std::set<drv_mem_handle_t*> handles;
    std::map<void*, drv_mem_handle_t*> mappings;
    std::vector<uint32_t> pages;
    std::vector<uint32_t> importDevices;
    std::vector<uint64_t> importedTokens;
    std::vector<uint64_t> releasedTokens;
    std::vector<int32_t> setupDevices;
    size_t hugeGranularity{2 * MiB};
    size_t normalGranularity{2 * MiB};
    size_t allocationBytes{};
    size_t reserveBytes{};
    drv_mem_prop allocationProp{};

    bool Fail(const std::string& api) { return failures[api].count(++calls[api]) != 0; }
};

FakeHal hal;

Expected<CtrlLayout::RankDataDesc> PeerDesc(size_t rank)
{
    CtrlLayout::RankDataDesc desc;
    desc.handle.store(peerToken + rank);
    return desc;
}

class Cache2HalTest : public testing::Test {
protected:
    testing::NiceMock<CtrlLayout> ctrl;

    void SetUp() override
    {
        hal = FakeHal{};
        ON_CALL(ctrl, SlotCount()).WillByDefault(Return(9));
        ON_CALL(ctrl, SetRankDesc(_, _)).WillByDefault(Return(Status::OK()));
        ON_CALL(ctrl, GetRankDesc(_)).WillByDefault(PeerDesc);
    }

    void TearDown() override
    {
        EXPECT_TRUE(hal.mappings.empty());
        EXPECT_TRUE(hal.handles.empty());
        EXPECT_TRUE(hal.reservations.empty());
        // Keep later tests independent even if a regression leaks resources.
        for (drv_mem_handle_t* handle : hal.handles) { delete handle; }
        for (const auto& [addr, bytes] : hal.reservations) { munmap(addr, bytes); }
    }
};

}  // namespace

UC::Status UC::Trans::Device::Setup(int32_t deviceId)
{
    hal.setupDevices.push_back(deviceId);
    return hal.Fail("device") ? Status::Error("device setup failed") : Status::OK();
}

extern "C" {

drvError_t halMemGetAllocationGranularity(const drv_mem_prop* prop,
                                          drv_mem_granularity_options option, size_t* granularity)
{
    EXPECT_EQ(prop->side, MEM_HOST_SIDE);
    EXPECT_EQ(prop->devid, 0U);
    EXPECT_EQ(prop->mem_type, MEM_DDR_TYPE);
    EXPECT_EQ(option, MEM_ALLOC_GRANULARITY_RECOMMENDED);
    hal.pages.push_back(prop->pg_type);
    if (hal.Fail("granularity")) { return DRV_ERROR_INVALID_VALUE; }
    *granularity =
        prop->pg_type == MEM_HUGE_PAGE_TYPE ? hal.hugeGranularity : hal.normalGranularity;
    return DRV_ERROR_NONE;
}

drvError_t halMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr,
                                uint64_t flags)
{
    EXPECT_EQ(alignment, 0U);
    EXPECT_EQ(addr, nullptr);
    EXPECT_EQ(flags, 0U);
    EXPECT_EQ(size % GiB, 0U);
    EXPECT_TRUE(hal.reservations.empty());
    hal.reserveBytes = size;
    if (hal.Fail("reserve")) { return DRV_ERROR_INVALID_VALUE; }
    // Reserve virtual addresses only; tests neither allocate nor touch 1 GiB of RAM.
    void* raw =
        mmap(nullptr, size + GiB, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (raw == MAP_FAILED) { return DRV_ERROR_INVALID_VALUE; }
    uintptr_t start = reinterpret_cast<uintptr_t>(raw);
    uintptr_t aligned = (start + GiB - 1) / GiB * GiB;
    size_t prefix = aligned - start;
    if (prefix != 0) { EXPECT_EQ(munmap(raw, prefix), 0); }
    EXPECT_EQ(munmap(reinterpret_cast<void*>(aligned + size), GiB - prefix), 0);
    *ptr = reinterpret_cast<void*>(aligned);
    hal.reservations.emplace(*ptr, size);
    return DRV_ERROR_NONE;
}

drvError_t halMemAddressFree(void* ptr)
{
    EXPECT_TRUE(hal.handles.empty());
    EXPECT_TRUE(hal.mappings.empty());
    auto reservation = hal.reservations.find(ptr);
    if (reservation == hal.reservations.end()) {
        ADD_FAILURE() << "free of unreserved address";
        return DRV_ERROR_INVALID_VALUE;
    }
    EXPECT_EQ(munmap(ptr, reservation->second), 0);
    hal.reservations.erase(reservation);
    return DRV_ERROR_NONE;
}

drvError_t halMemCreate(drv_mem_handle_t** handle, size_t size, const drv_mem_prop* prop,
                        uint64_t flags)
{
    EXPECT_EQ(prop->side, MEM_HOST_SIDE);
    EXPECT_EQ(prop->devid, 0U);
    EXPECT_EQ(prop->mem_type, MEM_DDR_TYPE);
    EXPECT_EQ(prop->module_id, 0U);
    EXPECT_EQ(prop->reserve, 0U);
    EXPECT_EQ(flags, 0U);
    hal.allocationBytes = size;
    hal.allocationProp = *prop;
    if (hal.Fail("create")) { return DRV_ERROR_INVALID_VALUE; }
    *handle = new drv_mem_handle_t{localToken, false};
    hal.handles.insert(*handle);
    return DRV_ERROR_NONE;
}

drvError_t halMemRelease(drv_mem_handle_t* handle)
{
    EXPECT_TRUE(hal.mappings.empty()) << "unmap all ranks before releasing handles";
    if (hal.handles.erase(handle) != 1) {
        ADD_FAILURE() << "release of unknown handle";
        return DRV_ERROR_INVALID_VALUE;
    }
    if (!handle->imported) {
        EXPECT_TRUE(hal.handles.empty()) << "release imports before the owner";
    }
    hal.releasedTokens.push_back(handle->token);
    delete handle;
    return DRV_ERROR_NONE;
}

drvError_t halMemMap(void* ptr, size_t size, size_t offset, drv_mem_handle_t* handle,
                     uint64_t flags)
{
    EXPECT_EQ(size, hal.allocationBytes);
    EXPECT_EQ(offset, 0U);
    EXPECT_EQ(flags, 0U);
    EXPECT_EQ(hal.handles.count(handle), 1U);
    EXPECT_EQ(hal.mappings.count(ptr), 0U);
    if (hal.Fail("map")) { return DRV_ERROR_INVALID_VALUE; }
    hal.mappings.emplace(ptr, handle);
    return DRV_ERROR_NONE;
}

drvError_t halMemUnmap(void* ptr)
{
    EXPECT_EQ(hal.mappings.erase(ptr), 1U);
    return DRV_ERROR_NONE;
}

drvError_t halMemExportToShareableHandle(drv_mem_handle_t* handle, drv_mem_handle_type type,
                                         uint64_t flags, uint64_t* shareHandle)
{
    EXPECT_EQ(hal.handles.count(handle), 1U);
    EXPECT_EQ(type, MEM_HANDLE_TYPE_NONE);
    EXPECT_EQ(flags, 0U);
    if (hal.Fail("export")) { return DRV_ERROR_INVALID_VALUE; }
    *shareHandle = handle->token;
    return DRV_ERROR_NONE;
}

drvError_t halMemShareHandleSetAttribute(uint64_t shareHandle, ShareHandleAttrType type,
                                         ShareHandleAttr attr)
{
    EXPECT_EQ(shareHandle, localToken);
    EXPECT_EQ(type, SHR_HANDLE_ATTR_NO_WLIST_IN_SERVER);
    EXPECT_EQ(attr.enableFlag, SHR_HANDLE_NO_WLIST_ENABLE);
    for (unsigned int value : attr.rsv) { EXPECT_EQ(value, 0U); }
    return hal.Fail("attribute") ? DRV_ERROR_INVALID_VALUE : DRV_ERROR_NONE;
}

drvError_t halMemImportFromShareableHandle(uint64_t shareHandle, uint32_t deviceId,
                                           drv_mem_handle_t** handle)
{
    if (hal.Fail("import")) { return DRV_ERROR_INVALID_VALUE; }
    hal.importDevices.push_back(deviceId);
    hal.importedTokens.push_back(shareHandle);
    *handle = new drv_mem_handle_t{shareHandle, true};
    hal.handles.insert(*handle);
    return DRV_ERROR_NONE;
}

}  // extern "C"

namespace {

class RankAddressTest : public Cache2HalTest, public testing::WithParamInterface<size_t> {};

TEST_P(RankAddressTest, OnlyOwnerExposesHostSlotsAndPeersExposeDeviceSlots)
{
    const size_t owner = GetParam();
    bool published = false;
    EXPECT_CALL(ctrl, SetRankDesc(owner, _))
        .WillOnce([&](size_t, const CtrlLayout::RankDataDesc& desc) {
            EXPECT_EQ(desc.handle.load(), localToken);
            published = true;
            return Status::OK();
        });
    EXPECT_CALL(ctrl, GetRankDesc(owner)).Times(0);
    EXPECT_CALL(ctrl, GetRankDesc(testing::Ne(owner))).Times(2).WillRepeatedly([&](size_t rank) {
        EXPECT_TRUE(published);
        return PeerDesc(rank);
    });
    DataStrategy strategy;
    EXPECT_EQ(strategy.DataAt(0), nullptr);
    EXPECT_EQ(strategy.DeviceDataAt(0), nullptr);
    strategy.Setup(ctrl, 7, 1000, 3, owner, 0);
    ASSERT_EQ(hal.reservations.size(), 1U);
    std::byte* base = static_cast<std::byte*>(hal.reservations.begin()->first);
    for (size_t slot = 0; slot < 9; ++slot) {
        SCOPED_TRACE(slot);
        const size_t rank = slot / 3;
        void* expected = base + rank * 2 * MiB + (slot % 3) * 1000;
        EXPECT_EQ(strategy.DataAt(slot), rank == owner ? expected : nullptr);
        EXPECT_EQ(strategy.DeviceDataAt(slot), rank == owner ? nullptr : expected);
        EXPECT_EQ(hal.mappings.at(base + rank * 2 * MiB)->token,
                  rank == owner ? localToken : peerToken + rank);
    }
    EXPECT_EQ(strategy.DataAt(9), nullptr);
    EXPECT_EQ(strategy.DeviceDataAt(9), nullptr);
    EXPECT_EQ(strategy.DataAt(std::numeric_limits<size_t>::max()), nullptr);
    EXPECT_EQ(strategy.DeviceDataAt(std::numeric_limits<size_t>::max()), nullptr);
    EXPECT_THAT(hal.setupDevices, ElementsAre(7));
    EXPECT_THAT(hal.importDevices, ElementsAre(7, 7));
    EXPECT_THAT(hal.pages, ElementsAre(MEM_HUGE_PAGE_TYPE));
    EXPECT_THROW(strategy.Setup(ctrl, 7, 1000, 3, owner, 0), std::logic_error);
}

INSTANTIATE_TEST_SUITE_P(Owners, RankAddressTest, testing::Values(0, 1, 2));

TEST_F(Cache2HalTest, AlignsRankStrideAndReservationIndependently)
{
    HalHostBuffers buffers;
    buffers.Setup(4, 513 * MiB + 1, 3, 1);
    EXPECT_EQ(hal.allocationBytes, 514 * MiB);
    EXPECT_EQ(hal.reserveBytes, 2 * GiB);
    EXPECT_EQ(buffers.Owner(), 1U);
    EXPECT_EQ(buffers.DeviceId(), 4);
    EXPECT_EQ(buffers.RankCount(), 3U);
    EXPECT_EQ(buffers.HostData(0), nullptr);
    EXPECT_EQ(buffers.DeviceData(0), nullptr);
    EXPECT_EQ(buffers.DeviceData(1), nullptr);
    EXPECT_EQ(buffers.HostData(3), nullptr);
    std::byte* base = static_cast<std::byte*>(hal.reservations.begin()->first);
    EXPECT_EQ(buffers.HostData(1), base + 514 * MiB);
    buffers.ImportHandle(2, peerToken + 2);
    EXPECT_EQ(buffers.DeviceData(2), base + 1028 * MiB);
    EXPECT_EQ(buffers.HostData(2), nullptr);
    EXPECT_THROW(buffers.ImportHandle(1, peerToken), std::invalid_argument);
    EXPECT_THROW(buffers.ImportHandle(3, peerToken), std::invalid_argument);
    EXPECT_THROW(buffers.ImportHandle(2, peerToken), std::logic_error);
    EXPECT_THROW(buffers.Setup(4, MiB, 3, 1), std::logic_error);
}

TEST_F(Cache2HalTest, UninitializedBuffersRejectHandleOperations)
{
    HalHostBuffers buffers;
    EXPECT_EQ(buffers.HostData(0), nullptr);
    EXPECT_EQ(buffers.DeviceData(0), nullptr);
    EXPECT_THROW(buffers.ExportHandle(), std::logic_error);
    EXPECT_THROW(buffers.ImportHandle(0, peerToken), std::logic_error);
}

TEST_F(Cache2HalTest, SingleRankNeedsNoImport)
{
    ON_CALL(ctrl, SlotCount()).WillByDefault(Return(3));
    EXPECT_CALL(ctrl, GetRankDesc(_)).Times(0);
    DataStrategy strategy;
    strategy.Setup(ctrl, 0, 1000, 3, 0, 0);
    EXPECT_NE(strategy.DataAt(2), nullptr);
    EXPECT_EQ(strategy.DeviceDataAt(2), nullptr);
    EXPECT_TRUE(hal.importDevices.empty());
}

class PageFallbackTest : public Cache2HalTest, public testing::WithParamInterface<const char*> {};

TEST_P(PageFallbackTest, CleansHugeAttemptAndRecomputesNormalPageStride)
{
    hal.failures[GetParam()] = {1};
    hal.normalGranularity = 4 * MiB;
    DataStrategy strategy;
    strategy.Setup(ctrl, 7, 1000, 3, 1, 0);
    EXPECT_THAT(hal.pages, ElementsAre(MEM_HUGE_PAGE_TYPE, MEM_NORMAL_PAGE_TYPE));
    EXPECT_THAT(hal.setupDevices, ElementsAre(7));
    EXPECT_EQ(hal.allocationProp.pg_type, MEM_NORMAL_PAGE_TYPE);
    EXPECT_EQ(hal.allocationBytes, 4 * MiB);
    std::byte* base = static_cast<std::byte*>(hal.reservations.begin()->first);
    EXPECT_EQ(strategy.DataAt(3), base + 4 * MiB);
    EXPECT_EQ(strategy.DeviceDataAt(6), base + 8 * MiB);
}

INSTANTIATE_TEST_SUITE_P(LocalFailures, PageFallbackTest,
                         testing::Values("granularity", "reserve", "create", "map"));

class SetupFailureTest : public Cache2HalTest, public testing::WithParamInterface<const char*> {};

TEST_P(SetupFailureTest, ReleasesResourcesAndAllowsRetry)
{
    hal.failures[GetParam()] = {1, 2};
    DataStrategy strategy;
    EXPECT_THROW(strategy.Setup(ctrl, 7, 1000, 3, 1, 0), std::runtime_error);
    EXPECT_EQ(strategy.DataAt(3), nullptr);
    EXPECT_EQ(strategy.DeviceDataAt(0), nullptr);
    EXPECT_TRUE(hal.handles.empty());
    EXPECT_TRUE(hal.mappings.empty());
    EXPECT_TRUE(hal.reservations.empty());
    hal.failures.clear();
    EXPECT_NO_THROW(strategy.Setup(ctrl, 2, 1000, 3, 0, 0));
    EXPECT_NE(strategy.DataAt(0), nullptr);
    EXPECT_NE(strategy.DeviceDataAt(3), nullptr);
}

INSTANTIATE_TEST_SUITE_P(HalFailures, SetupFailureTest,
                         testing::Values("device", "granularity", "reserve", "create", "map",
                                         "export", "attribute", "import"));

TEST_F(Cache2HalTest, FailedSecondPeerMapReleasesEvenTheUnmappedImportedHandle)
{
    hal.failures["map"] = {3};
    DataStrategy strategy;
    EXPECT_THROW(strategy.Setup(ctrl, 7, 1000, 3, 1, 0), std::runtime_error);
    EXPECT_THAT(hal.releasedTokens, ElementsAre(peerToken, peerToken + 2, localToken));
    EXPECT_TRUE(hal.reservations.empty());
}

class InvalidGranularityTest : public Cache2HalTest, public testing::WithParamInterface<size_t> {};

TEST_P(InvalidGranularityTest, RejectsInvalidDriverGranularityBeforeReserving)
{
    hal.hugeGranularity = GetParam();
    hal.normalGranularity = GetParam();
    HalHostBuffers buffers;
    EXPECT_THROW(buffers.Setup(7, MiB, 3, 1), std::runtime_error);
    EXPECT_EQ(hal.calls["reserve"], 0U);
}

INSTANTIATE_TEST_SUITE_P(InvalidGranularities, InvalidGranularityTest,
                         testing::Values(size_t{0}, size_t{3}, 2 * GiB));

struct InvalidLayout {
    const char* name;
    int32_t device;
    size_t slotSize;
    size_t slotsPerRank;
    size_t totalSlots;
    size_t owner;
};

class InvalidLayoutTest : public Cache2HalTest,
                          public testing::WithParamInterface<InvalidLayout> {};

TEST_P(InvalidLayoutTest, RejectsInvalidLayoutBeforeDeviceSetup)
{
    const InvalidLayout layout = GetParam();
    ON_CALL(ctrl, SlotCount()).WillByDefault(Return(layout.totalSlots));
    DataStrategy strategy;
    EXPECT_THROW(
        strategy.Setup(ctrl, layout.device, layout.slotSize, layout.slotsPerRank, layout.owner, 0),
        std::invalid_argument);
    EXPECT_TRUE(hal.setupDevices.empty());
}

INSTANTIATE_TEST_SUITE_P(InvalidLayouts, InvalidLayoutTest,
                         testing::Values(InvalidLayout{"NegativeDevice", -1, 1000, 3, 9, 1},
                                         InvalidLayout{"ZeroSlotSize", 7, 0, 3, 9, 1},
                                         InvalidLayout{"ZeroSlotsPerRank", 7, 1000, 0, 9, 1},
                                         InvalidLayout{"ZeroTotalSlots", 7, 1000, 3, 0, 1},
                                         InvalidLayout{"UnevenSlotCounts", 7, 1000, 3, 8, 1},
                                         InvalidLayout{"InvalidOwner", 7, 1000, 3, 9, 3}),
                         [](const testing::TestParamInfo<InvalidLayout>& info) {
                             return info.param.name;
                         });

TEST_F(Cache2HalTest, PublishFailureDoesNotReadPeersAndReleasesAllocation)
{
    EXPECT_CALL(ctrl, SetRankDesc(1, _)).WillOnce(Return(Status::Error("publish failed")));
    EXPECT_CALL(ctrl, GetRankDesc(_)).Times(0);
    DataStrategy strategy;
    EXPECT_THROW(strategy.Setup(ctrl, 7, 1000, 3, 1, 0), std::runtime_error);
    EXPECT_TRUE(hal.reservations.empty());
}

TEST_F(Cache2HalTest, DefaultTimeoutAllowsWaitingForPeerHandles)
{
    EXPECT_CALL(ctrl, GetRankDesc(0))
        .WillOnce(Return(Status::NotFound()))
        .WillOnce(Return(Status::Retry()))
        .WillOnce(PeerDesc);
    EXPECT_CALL(ctrl, GetRankDesc(2)).WillOnce(PeerDesc);
    DataStrategy strategy;
    EXPECT_NO_THROW(strategy.Setup(ctrl, 7, 1000, 3, 1));
    EXPECT_THAT(hal.importedTokens, ElementsAre(peerToken, peerToken + 2));
}

TEST_F(Cache2HalTest, ZeroTimeoutReadsUnavailablePeerOnceAndReportsContext)
{
    EXPECT_CALL(ctrl, GetRankDesc(0)).WillOnce(Return(Status::NotFound()));
    EXPECT_CALL(ctrl, GetRankDesc(2)).Times(0);
    DataStrategy strategy;
    try {
        strategy.Setup(ctrl, 7, 1000, 3, 1, 0);
        FAIL() << "missing peer must time out";
    } catch (const std::runtime_error& error) {
        EXPECT_THAT(error.what(), HasSubstr("GetRankDesc timed out"));
        EXPECT_THAT(error.what(), HasSubstr("owner=1 device=7 rank=0 timeout_ms=0"));
    }
    EXPECT_TRUE(hal.reservations.empty());
}

TEST_F(Cache2HalTest, AllPeersShareOneDeadline)
{
    EXPECT_CALL(ctrl, GetRankDesc(0)).WillOnce([](size_t rank) {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        return PeerDesc(rank);
    });
    // The first peer has already consumed the entire 20 ms budget.
    EXPECT_CALL(ctrl, GetRankDesc(2)).WillOnce(Return(Status::NotFound()));
    DataStrategy strategy;
    EXPECT_THROW(strategy.Setup(ctrl, 7, 1000, 3, 1, 20), std::runtime_error);
    EXPECT_THAT(hal.releasedTokens, ElementsAre(peerToken, localToken));
    EXPECT_TRUE(hal.reservations.empty());
}

TEST_F(Cache2HalTest, UnexpectedControlExceptionAlsoCleansUpAndAllowsRetry)
{
    EXPECT_CALL(ctrl, GetRankDesc(0)).WillOnce([](size_t) -> Expected<CtrlLayout::RankDataDesc> {
        throw std::bad_alloc();
    });
    DataStrategy strategy;
    EXPECT_THROW(strategy.Setup(ctrl, 7, 1000, 3, 1, 0), std::bad_alloc);
    EXPECT_TRUE(hal.reservations.empty());
    testing::Mock::VerifyAndClearExpectations(&ctrl);
    EXPECT_NO_THROW(strategy.Setup(ctrl, 7, 1000, 3, 1, 0));
}

TEST_F(Cache2HalTest, ImportsFullWidthHandleWithoutTreatingMaximumAsInvalid)
{
    EXPECT_CALL(ctrl, GetRankDesc(0)).WillOnce([](size_t) {
        CtrlLayout::RankDataDesc desc;
        desc.handle.store(std::numeric_limits<uint64_t>::max());
        return Expected<CtrlLayout::RankDataDesc>(std::move(desc));
    });
    EXPECT_CALL(ctrl, GetRankDesc(2)).WillOnce(PeerDesc);
    DataStrategy strategy;
    strategy.Setup(ctrl, 7, 1000, 3, 1, 0);
    EXPECT_THAT(hal.importedTokens,
                ElementsAre(std::numeric_limits<uint64_t>::max(), peerToken + 2));
}

}  // namespace
