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
#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include "logger/logger.h"
#include "meta_manager.h"
#include "time/stopwatch.h"
#include "ucmstore_v1.h"

namespace UC::FakeStore {

class FakeStore : public StoreV1 {
    MetaManager metaMgr_;

public:
    Status Setup(const Detail::Dictionary& inConfig) override
    {
        Config config;
        inConfig.Get("unique_id", config.uniqueId);
        inConfig.GetNumber("buffer_number", config.bufferNumber);
        inConfig.Get("share_buffer_enable", config.shareBufferEnable);
        inConfig.Get("fake_always_hit", config.alwaysHit);
        inConfig.Get("fake_fail_load", config.failLoad);
        inConfig.GetNumbers("fake_load_delay_us", loadDelaysUs_);
        auto s = CheckConfig(config);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed to check config params: {}.", s);
            return s;
        }
        s = metaMgr_.Setup(config);
        if (s.Failure()) [[unlikely]] { return s; }
        alwaysHit_ = config.alwaysHit;
        failLoad_ = config.failLoad;
        ShowConfig(config);
        return Status::OK();
    }
    std::string Readme() const override { return "FakeStore"; }
    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num) override
    {
        if (alwaysHit_) { return std::vector<uint8_t>(num, true); }
        std::vector<uint8_t> founds(num);
        StopWatch sw;
        std::transform(blocks, blocks + num, founds.begin(),
                       [this](const Detail::BlockId& block) { return metaMgr_.Exist(block); });
        UC_DEBUG("Fake lookup({}) costs {:.3f}ms.", num, sw.Elapsed().count() * 1e3);
        return founds;
    }
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) override
    {
        if (alwaysHit_) { return static_cast<ssize_t>(num) - 1; }
        ssize_t index = -1;
        StopWatch sw;
        for (size_t i = 0; i < num && metaMgr_.Exist(blocks[i]); i++) {
            index = static_cast<ssize_t>(i);
        }
        UC_DEBUG("Fake Lookup({}/{}) costs {:.3f}ms.", index, num, sw.Elapsed().count() * 1e3);
        return index;
    }
    Expected<ssize_t> LookupOnReverse(const Detail::BlockId* blocks, size_t num) override
    {
        StopWatch sw;
        for (ssize_t i = static_cast<ssize_t>(num) - 1; i >= 0; --i) {
            if (metaMgr_.Exist(blocks[i])) {
                UC_DEBUG("Fake reverse lookup({}/{}) costs {:.3f}ms.", i, num,
                         sw.Elapsed().count() * 1e3);
                return i;
            }
        }
        UC_DEBUG("Fake reverse lookup(-1/{}) costs {:.3f}ms.", num, sw.Elapsed().count() * 1e3);
        return static_cast<ssize_t>(-1);
    }
    void Prefetch(const Detail::BlockId* blocks, size_t num) override {}
    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override
    {
        if (failLoad_) { return Status::Error("injected fake load failure"); }
        auto handle = NextId();
        if (!loadDelaysUs_.empty()) {
            const auto sequence = loadSequence_.fetch_add(1, std::memory_order_relaxed);
            const auto readyAt = std::chrono::steady_clock::now() +
                                 std::chrono::microseconds(loadDelaysUs_[sequence %
                                                                       loadDelaysUs_.size()]);
            std::lock_guard<std::mutex> lock(loadMutex_);
            loadReadyAt_[handle] = readyAt;
        }
        return handle;
    }
    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override
    {
        StopWatch sw;
        std::for_each(task.begin(), task.end(),
                      [this](const Detail::Shard& shard) { metaMgr_.Insert(shard.owner); });
        UC_DEBUG("Fake dump({}) costs {:.3f}ms.", task.size(), sw.Elapsed().count() * 1e3);
        return NextId();
    }
    Expected<bool> Check(Detail::TaskHandle taskId) override
    {
        std::lock_guard<std::mutex> lock(loadMutex_);
        const auto found = loadReadyAt_.find(taskId);
        return found == loadReadyAt_.end() || std::chrono::steady_clock::now() >= found->second;
    }
    Status Wait(Detail::TaskHandle taskId) override
    {
        std::chrono::steady_clock::time_point readyAt;
        {
            std::lock_guard<std::mutex> lock(loadMutex_);
            const auto found = loadReadyAt_.find(taskId);
            if (found == loadReadyAt_.end()) { return Status::OK(); }
            readyAt = found->second;
            loadReadyAt_.erase(found);
        }
        std::this_thread::sleep_until(readyAt);
        return Status::OK();
    }

private:
    static Detail::TaskHandle NextId() noexcept
    {
        static std::atomic<Detail::TaskHandle> idSeed{1};
        return idSeed.fetch_add(1, std::memory_order_relaxed);
    };
    Status CheckConfig(const Config& config)
    {
        if (config.uniqueId.empty()) { return Status::InvalidParam("invalid unique id"); }
        if (config.bufferNumber < 1024) {
            return Status::InvalidParam("too small buffer number({})", config.bufferNumber);
        }
        if (!config.shareBufferEnable) { return Status::InvalidParam("buffer must be shared"); }
        return Status::OK();
    }
    void ShowConfig(const Config& config)
    {
        const auto& ns = Readme();
        std::string buildType = UCM_BUILD_TYPE;
        if (buildType.empty()) { buildType = "Release"; }
        UC_INFO("{}-{}({}).", ns, UCM_COMMIT_ID, buildType);
        UC_INFO("Set {}::UniqueId to {}.", ns, config.uniqueId);
        UC_INFO("Set {}::BufferNumber to {}.", ns, config.bufferNumber);
        UC_INFO("Set {}::ShareBufferEnable to {}.", ns, config.shareBufferEnable);
        UC_INFO("Set {}::AlwaysHit to {}.", ns, config.alwaysHit);
        UC_INFO("Set {}::FailLoad to {}.", ns, config.failLoad);
    }

    bool alwaysHit_{false};
    bool failLoad_{false};
    std::vector<uint64_t> loadDelaysUs_;
    std::atomic<size_t> loadSequence_{0};
    std::mutex loadMutex_;
    std::unordered_map<Detail::TaskHandle, std::chrono::steady_clock::time_point> loadReadyAt_;
};

}  // namespace UC::FakeStore

extern "C" UC::StoreV1* MakeFakeStore() { return new UC::FakeStore::FakeStore(); }
