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
#include <cctype>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include "datasystem/hetero_client.h"
#include "datasystem/kv_client.h"
#include "logger/logger.h"
#include "task_manager.h"
#include "time/now_time.h"
#include "ucmstore_v1.h"
#include "yuanrong_config.h"
#include "yuanrong_helper.h"

namespace UC::YuanRongStore {

class YuanRongStore : public StoreV1 {
    Config config_{};
    std::shared_ptr<datasystem::HeteroClient> heteroClient_;
    std::shared_ptr<datasystem::KVClient> kvClient_;
    TaskManager taskManager_;
    bool taskManagerEnabled_{false};

public:
    ~YuanRongStore() override = default;

    Status Setup(const Detail::Dictionary& input) override
    {
        config_ = ParseConfig(input);
        auto s = CheckConfig(config_);
        if (s.Failure()) { return s; }
        s = ResolveDeviceMemoryPreRegistration(config_, std::getenv("DS_RH2D_LINK_TYPE"));
        if (s.Failure()) { return s; }

        datasystem::ConnectOptions options;
        options.host = config_.host;
        options.port = config_.port;
        options.requestTimeoutMs = static_cast<int32_t>(config_.timeoutMs);
        options.enableRemoteH2D = config_.enableRemoteH2D;

        heteroClient_ = std::make_shared<datasystem::HeteroClient>(options);
        auto heteroStatus = heteroClient_->Init();
        if (heteroStatus.IsError()) {
            return Status::Error("failed to initialize YuanRong HeteroClient: " +
                                 heteroStatus.ToString());
        }

        kvClient_ = std::make_shared<datasystem::KVClient>(options);
        auto kvStatus = kvClient_->Init();
        if (kvStatus.IsError()) {
            return Status::Error("failed to initialize YuanRong KVClient: " + kvStatus.ToString());
        }

        if (config_.deviceId >= 0) {
            s = taskManager_.Setup(config_, heteroClient_, kvClient_);
            if (s.Failure()) { return s; }
            taskManagerEnabled_ = true;
        }
        ShowConfig(config_);
        return Status::OK();
    }

    std::string Readme() const override { return "YuanRongStore"; }

    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num) override
    {
        if (num == 0) { return std::vector<uint8_t>{}; }
        auto lookupResult = LookupYuanRong(blocks, num);
        if (!lookupResult) { return lookupResult.Error(); }
        const auto& exists = lookupResult.Value();

        std::vector<uint8_t> result(num, 0);
        std::vector<Detail::BlockId> missBlocks;
        std::vector<size_t> missIndexes;
        missBlocks.reserve(num);
        missIndexes.reserve(num);
        for (size_t i = 0; i < num; ++i) {
            result[i] = exists[i] ? 1 : 0;
            if (!exists[i]) {
                missBlocks.push_back(blocks[i]);
                missIndexes.push_back(i);
            }
        }
        if (missBlocks.empty() || config_.storeBackend == nullptr) { return result; }

        auto backendStart = NowTime::Now();
        auto backendResult = config_.storeBackend->Lookup(missBlocks.data(), missBlocks.size());
        auto backendEnd = NowTime::Now();
        if (!backendResult) { return backendResult.Error(); }
        const auto& backendExists = backendResult.Value();
        if (backendExists.size() != missBlocks.size()) {
            return Status::Error("backend Lookup returned an unexpected result size");
        }
        for (size_t i = 0; i < missIndexes.size(); ++i) {
            result[missIndexes[i]] = backendExists[i];
        }
        UC_DEBUG("YuanRong Lookup queried Posix misses={}/{}, cost={:.3f}ms.", missBlocks.size(),
                 num, (backendEnd - backendStart) * 1e3);
        return result;
    }

    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) override
    {
        if (num == 0) { return static_cast<ssize_t>(-1); }
        auto lookupResult = LookupYuanRong(blocks, num);
        if (!lookupResult) { return lookupResult.Error(); }
        const auto& exists = lookupResult.Value();

        std::vector<Detail::BlockId> missBlocks;
        std::vector<size_t> missIndexes;
        missBlocks.reserve(num);
        missIndexes.reserve(num);
        for (size_t i = 0; i < num; ++i) {
            if (!exists[i]) {
                missBlocks.push_back(blocks[i]);
                missIndexes.push_back(i);
            }
        }
        if (missBlocks.empty()) { return static_cast<ssize_t>(num) - 1; }
        if (config_.storeBackend == nullptr) {
            return static_cast<ssize_t>(missIndexes.front()) - 1;
        }

        auto backendStart = NowTime::Now();
        auto backendResult =
            config_.storeBackend->LookupOnPrefix(missBlocks.data(), missBlocks.size());
        auto backendEnd = NowTime::Now();
        if (!backendResult) { return backendResult.Error(); }
        ssize_t result = -1;
        auto status = ResolveTieredPrefixHit(num, missIndexes, backendResult.Value(), result);
        if (status.Failure()) { return status; }
        UC_DEBUG("YuanRong LookupOnPrefix queried Posix misses={}/{}, cost={:.3f}ms, result={}.",
                 missBlocks.size(), num, (backendEnd - backendStart) * 1e3, result);
        return result;
    }

    void Prefetch(const Detail::BlockId* blocks, size_t num) override
    {
        (void)blocks;
        (void)num;
    }

    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override
    {
        if (!taskManagerEnabled_) { return Status::Unsupported(); }
        return taskManager_.Submit(TransTask{TransTask::Type::LOAD, std::move(task)});
    }

    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override
    {
        if (!taskManagerEnabled_) { return Status::Unsupported(); }
        return taskManager_.Submit(TransTask{TransTask::Type::DUMP, std::move(task)});
    }

    Expected<bool> Check(Detail::TaskHandle taskId) override
    {
        if (!taskManagerEnabled_) { return Status::Unsupported(); }
        return taskManager_.Check(taskId);
    }

    Status Wait(Detail::TaskHandle taskId) override
    {
        if (!taskManagerEnabled_) { return Status::Unsupported(); }
        return taskManager_.Wait(taskId);
    }

    Status RegisterMemory(void* baseAddr, size_t totalSize) override
    {
        if (!config_.enableDeviceMemoryPreRegistration) { return Status::OK(); }
        if (baseAddr == nullptr) {
            return Status::InvalidParam("YuanRong device memory address cannot be null");
        }
        if (totalSize == 0) {
            return Status::InvalidParam("YuanRong device memory size must be greater than zero");
        }
        if (heteroClient_ == nullptr) {
            return Status::Error("YuanRong HeteroClient is not initialized");
        }

        auto status = heteroClient_->PreRegisterDeviceMemory(
            std::vector<void*>{baseAddr}, std::vector<uint64_t>{static_cast<uint64_t>(totalSize)});
        if (status.IsError()) {
            UC_ERROR("YuanRong PreRegisterDeviceMemory failed: addr={}, size={}, error={}",
                     baseAddr, totalSize, status.ToString());
            return Status::Error("YuanRong PreRegisterDeviceMemory failed: " + status.ToString());
        }
        return Status::OK();
    }

private:
    Expected<std::vector<bool>> LookupYuanRong(const Detail::BlockId* blocks, size_t num)
    {
        std::vector<std::string> keys;
        keys.reserve(num);
        for (size_t i = 0; i < num; ++i) { keys.push_back(MakeLookupKey(config_, blocks[i])); }

        std::vector<bool> exists;
        auto start = NowTime::Now();
        auto status = heteroClient_->Exist(keys, exists);
        auto end = NowTime::Now();
        if (status.IsError() && exists.empty()) {
            return Status::Error("YuanRong Exist failed: " + status.ToString());
        }
        if (exists.size() != num) {
            return Status::Error("YuanRong Exist returned an unexpected result size");
        }
        const auto hitCount = std::count(exists.begin(), exists.end(), true);
        UC_DEBUG("YuanRong Lookup Exist keys={}, hit={}, cost={:.3f}ms, status={}.", num, hitCount,
                 (end - start) * 1e3, status.ToString());
        return exists;
    }

    static Config ParseConfig(const Detail::Dictionary& input)
    {
        Config config;
        input.Get("yuanrong_host", config.host);
        input.GetNumber("yuanrong_port", config.port);
        input.Get("yuanrong_namespace", config.nameSpace);
        if (config.nameSpace.empty()) { input.Get("unique_id", config.nameSpace); }
        input.Get("yuanrong_enable_remote_h2d", config.enableRemoteH2D);
        input.GetNumber("device_id", config.deviceId);
        input.GetNumbers("tensor_size_list", config.tensorSizes);
        input.GetNumber("shard_size", config.shardSize);
        input.GetNumber("block_size", config.blockSize);
        input.GetNumber("yuanrong_memory_alignment", config.memoryAlignment);
        input.GetNumber("yuanrong_timeout_ms", config.timeoutMs);
        input.GetNumber("yuanrong_waiting_queue_depth", config.waitingQueueDepth);
        input.GetNumber("yuanrong_load_worker_count", config.loadWorkerCount);
        input.GetNumber("yuanrong_dump_prerequisite_worker_count",
                        config.dumpPrerequisiteWorkerCount);
        input.GetNumber("yuanrong_recovery_batch_size", config.recoveryBatchSize);
        input.GetNumber("yuanrong_host_buffer_count", config.hostBufferCount);
        config.hostBufferCountExplicit = config.hostBufferCount != 0;
        input.GetNumber("yuanrong_host_buffer_capacity_gb", config.hostBufferCapacityGb);
        input.GetNumber("yuanrong_h2d_stream_count", config.h2dStreamCount);
        input.GetNumber("yuanrong_backfill_worker_count", config.backfillWorkerCount);
        input.GetNumber("yuanrong_backfill_queue_depth", config.backfillQueueDepth);
        input.GetNumber("yuanrong_reaper_queue_depth", config.reaperQueueDepth);
        input.Get("cpu_affinity_cores", config.cpuAffinityCores);
        input.Get("io_direct", config.ioDirect);
        input.Get("posix_io_engine", config.posixIoEngine);
        input.Get("store_backend", config.storeBackend);
        config.objectSize =
            std::accumulate(config.tensorSizes.begin(), config.tensorSizes.end(), size_t{0});
        DeriveHostBufferCount(config);
        return config;
    }

    static void DeriveHostBufferCount(Config& config)
    {
        if (config.deviceId < 0 || config.storeBackend == nullptr) {
            config.hostBufferCount = 0;
            return;
        }
        if (config.hostBufferCountExplicit || config.objectSize == 0 ||
            config.hostBufferCapacityGb == 0 ||
            config.hostBufferCapacityGb > (std::numeric_limits<uint64_t>::max() >> 30)) {
            return;
        }
        const auto capacityBytes = static_cast<uint64_t>(config.hostBufferCapacityGb) << 30;
        config.hostBufferCount = DeriveYuanRongHostBufferCount(
            config.objectSize, config.recoveryBatchSize, config.loadWorkerCount,
            config.backfillWorkerCount, capacityBytes);
    }

    static Status CheckConfig(const Config& config)
    {
        if (config.host.empty()) { return Status::InvalidParam("yuanrong_host is required"); }
        if (config.port <= 0 || config.port > 65535) {
            return Status::InvalidParam("invalid yuanrong_port({})", config.port);
        }
        if (config.nameSpace.empty()) {
            return Status::InvalidParam("yuanrong_namespace or unique_id is required");
        }
        auto validChar = [](unsigned char ch) {
            return std::isalnum(ch) != 0 || ch == '-' || ch == '_' || ch == '.';
        };
        if (!std::all_of(config.nameSpace.begin(), config.nameSpace.end(), validChar)) {
            return Status::InvalidParam("yuanrong_namespace contains unsupported characters");
        }
        if (config.timeoutMs > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            return Status::InvalidParam("yuanrong_timeout_ms is too large");
        }
        if (config.memoryAlignment == 0 || config.memoryAlignment > 4096 ||
            (config.memoryAlignment & (config.memoryAlignment - 1)) != 0) {
            return Status::InvalidParam(
                "yuanrong_memory_alignment must be a power of two in (0, 4096]");
        }
        if (config.waitingQueueDepth <= 1) {
            return Status::InvalidParam("yuanrong_waiting_queue_depth({}) must be greater than 1",
                                        config.waitingQueueDepth);
        }
        if (config.loadWorkerCount == 0) {
            return Status::InvalidParam("yuanrong_load_worker_count must be greater than 0");
        }
        if (config.dumpPrerequisiteWorkerCount == 0) {
            return Status::InvalidParam(
                "yuanrong_dump_prerequisite_worker_count must be greater than 0");
        }
        if (config.h2dStreamCount == 0) {
            return Status::InvalidParam("yuanrong_h2d_stream_count must be greater than 0");
        }
        if (config.reaperQueueDepth <= 1) {
            return Status::InvalidParam("yuanrong_reaper_queue_depth({}) must be greater than 1",
                                        config.reaperQueueDepth);
        }
        if (config.storeBackend != nullptr && config.posixIoEngine != "psync" &&
            config.posixIoEngine != "aio") {
            return Status::InvalidParam("invalid posix_io_engine({}) for YuanRong|Posix",
                                        config.posixIoEngine);
        }
        if (config.storeBackend != nullptr && config.posixIoEngine == "aio" && !config.ioDirect) {
            return Status::InvalidParam(
                "YuanRong|Posix posix_io_engine=aio requires io_direct=true");
        }
        if (config.deviceId < 0) { return Status::OK(); }
        if (config.tensorSizes.empty() || config.objectSize == 0) {
            return Status::InvalidParam("tensor_size_list is required in worker mode");
        }
        if (config.shardSize == 0 || config.blockSize == 0 ||
            config.blockSize % config.shardSize != 0) {
            return Status::InvalidParam("invalid shard/block size");
        }
        if (config.storeBackend != nullptr) {
            if (config.recoveryBatchSize == 0) {
                return Status::InvalidParam("yuanrong_recovery_batch_size must be greater than 0");
            }
            if (config.backfillWorkerCount == 0) {
                return Status::InvalidParam(
                    "yuanrong_backfill_worker_count must be greater than 0");
            }
            if (config.backfillQueueDepth == 0) {
                return Status::InvalidParam("yuanrong_backfill_queue_depth must be greater than 0");
            }
            if (!config.hostBufferCountExplicit &&
                (config.hostBufferCapacityGb == 0 ||
                 config.hostBufferCapacityGb > (std::numeric_limits<uint64_t>::max() >> 30))) {
                return Status::InvalidParam("invalid yuanrong_host_buffer_capacity_gb({})",
                                            config.hostBufferCapacityGb);
            }
            if (config.hostBufferCount < config.recoveryBatchSize) {
                if (config.hostBufferCountExplicit) {
                    return Status::InvalidParam(
                        "yuanrong_host_buffer_count({}) must be greater than or equal to "
                        "yuanrong_recovery_batch_size({})",
                        config.hostBufferCount, config.recoveryBatchSize);
                }
                return Status::InvalidParam(
                    "YuanRong host buffer capacity({}GB) can provide {} buffers, fewer than "
                    "yuanrong_recovery_batch_size({}); increase "
                    "yuanrong_host_buffer_capacity_gb, reduce the batch size, or set "
                    "yuanrong_host_buffer_count explicitly",
                    config.hostBufferCapacityGb, config.hostBufferCount, config.recoveryBatchSize);
            }
            if (config.hostBufferCount >= std::numeric_limits<uint32_t>::max()) {
                return Status::InvalidParam("yuanrong_host_buffer_count({}) must be less than {}",
                                            config.hostBufferCount,
                                            std::numeric_limits<uint32_t>::max());
            }
        }
        if (config.ioDirect && config.storeBackend != nullptr) {
            constexpr size_t directIoAlignment = 4096;
            if (config.memoryAlignment != directIoAlignment) {
                return Status::InvalidParam(
                    "YuanRong|Posix io_direct requires yuanrong_memory_alignment=4096");
            }
            if (config.objectSize % directIoAlignment != 0) {
                return Status::InvalidParam(
                    "YuanRong object size must be aligned to 4096 bytes for io_direct");
            }
        }
        return Status::OK();
    }

    static void ShowConfig(const Config& config)
    {
        constexpr const char* name = "YuanRongStore";
        UC_INFO("{}::Host = {}", name, config.host);
        UC_INFO("{}::Port = {}", name, config.port);
        UC_INFO("{}::Namespace = {}", name, config.nameSpace);
        UC_INFO("{}::EnableRemoteH2D = {}", name, config.enableRemoteH2D);
        UC_INFO("{}::DeviceMemoryPreRegistration = {}", name,
                config.enableDeviceMemoryPreRegistration);
        UC_INFO("{}::DeviceId = {}", name, config.deviceId);
        UC_INFO("{}::ObjectSize = {}", name, config.objectSize);
        UC_INFO("{}::MemoryAlignment = {}", name, config.memoryAlignment);
        UC_INFO("{}::IoDirect = {}", name, config.ioDirect);
        UC_INFO("{}::PosixIoEngine = {}", name, config.posixIoEngine);
        UC_INFO("{}::TimeoutMs = {}", name, config.timeoutMs);
        UC_INFO("{}::LoadWorkerCount = {}", name, config.loadWorkerCount);
        UC_INFO("{}::DumpPrerequisiteWorkerCount = {}", name, config.dumpPrerequisiteWorkerCount);
        UC_INFO("{}::RecoveryBatchSize = {}", name, config.recoveryBatchSize);
        UC_INFO("{}::HostBufferCount = {}", name, config.hostBufferCount);
        UC_INFO("{}::HostBufferCountSource = {}", name,
                config.deviceId < 0 || config.storeBackend == nullptr
                    ? "disabled"
                    : (config.hostBufferCountExplicit ? "explicit" : "derived"));
        UC_INFO("{}::HostBufferCapacityGb = {}", name, config.hostBufferCapacityGb);
        UC_INFO("{}::H2DStreamCount = {}", name, config.h2dStreamCount);
        UC_INFO("{}::BackfillWorkerCount = {}", name, config.backfillWorkerCount);
        UC_INFO("{}::BackfillQueueDepth = {}", name, config.backfillQueueDepth);
        UC_INFO("{}::StoreBackend = {}", name,
                config.storeBackend ? config.storeBackend->Readme() : "none");
    }
};

}  // namespace UC::YuanRongStore

extern "C" UC::StoreV1* MakeYuanRongStore() { return new UC::YuanRongStore::YuanRongStore(); }
