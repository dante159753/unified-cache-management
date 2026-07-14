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
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include "datasystem/hetero_client.h"
#include "datasystem/kv_client.h"
#include "logger/logger.h"
#include "task_manager.h"
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
        std::vector<std::string> keys;
        keys.reserve(num);
        for (size_t i = 0; i < num; ++i) { keys.push_back(MakeLookupKey(config_, blocks[i])); }

        std::vector<bool> exists;
        auto status = heteroClient_->Exist(keys, exists);
        if (status.IsError() && exists.empty()) {
            return Status::Error("YuanRong Exist failed: " + status.ToString());
        }
        if (exists.size() != num) {
            return Status::Error("YuanRong Exist returned an unexpected result size");
        }

        std::vector<uint8_t> result(num, 0);
        bool hasMiss = false;
        for (size_t i = 0; i < num; ++i) {
            result[i] = exists[i] ? 1 : 0;
            hasMiss = hasMiss || !exists[i];
        }
        if (!hasMiss || config_.storeBackend == nullptr) { return result; }

        auto backendResult = config_.storeBackend->Lookup(blocks, num);
        if (!backendResult) { return backendResult.Error(); }
        const auto& backendExists = backendResult.Value();
        if (backendExists.size() != num) {
            return Status::Error("backend Lookup returned an unexpected result size");
        }
        for (size_t i = 0; i < num; ++i) { result[i] = result[i] || backendExists[i]; }
        return result;
    }

    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) override
    {
        auto result = Lookup(blocks, num);
        if (!result) { return result.Error(); }
        const auto& exists = result.Value();
        for (size_t i = 0; i < exists.size(); ++i) {
            if (!exists[i]) { return static_cast<ssize_t>(i) - 1; }
        }
        return static_cast<ssize_t>(exists.size()) - 1;
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
        (void)baseAddr;
        (void)totalSize;
        return Status::OK();
    }

private:
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
        input.GetNumber("yuanrong_recovery_batch_size", config.recoveryBatchSize);
        input.GetNumber("yuanrong_host_buffer_count", config.hostBufferCount);
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
        return config;
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
        if (config.waitingQueueDepth <= 1 || config.loadWorkerCount == 0 ||
            config.recoveryBatchSize == 0 || config.hostBufferCount < config.recoveryBatchSize ||
            config.hostBufferCount >= std::numeric_limits<uint32_t>::max() ||
            config.h2dStreamCount == 0 || config.backfillWorkerCount == 0 ||
            config.backfillQueueDepth == 0 || config.reaperQueueDepth <= 1) {
            return Status::InvalidParam("invalid YuanRong queue depth");
        }
        if (config.storeBackend != nullptr && config.posixIoEngine != "psync") {
            return Status::Unsupported();
        }
        if (config.deviceId < 0) { return Status::OK(); }
        if (config.tensorSizes.empty() || config.objectSize == 0) {
            return Status::InvalidParam("tensor_size_list is required in worker mode");
        }
        if (config.shardSize == 0 || config.blockSize == 0 ||
            config.blockSize % config.shardSize != 0) {
            return Status::InvalidParam("invalid shard/block size");
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
        UC_INFO("{}::DeviceId = {}", name, config.deviceId);
        UC_INFO("{}::ObjectSize = {}", name, config.objectSize);
        UC_INFO("{}::MemoryAlignment = {}", name, config.memoryAlignment);
        UC_INFO("{}::IoDirect = {}", name, config.ioDirect);
        UC_INFO("{}::TimeoutMs = {}", name, config.timeoutMs);
        UC_INFO("{}::LoadWorkerCount = {}", name, config.loadWorkerCount);
        UC_INFO("{}::RecoveryBatchSize = {}", name, config.recoveryBatchSize);
        UC_INFO("{}::HostBufferCount = {}", name, config.hostBufferCount);
        UC_INFO("{}::H2DStreamCount = {}", name, config.h2dStreamCount);
        UC_INFO("{}::BackfillWorkerCount = {}", name, config.backfillWorkerCount);
        UC_INFO("{}::BackfillQueueDepth = {}", name, config.backfillQueueDepth);
        UC_INFO("{}::StoreBackend = {}", name,
                config.storeBackend ? config.storeBackend->Readme() : "none");
    }
};

}  // namespace UC::YuanRongStore

extern "C" UC::StoreV1* MakeYuanRongStore() { return new UC::YuanRongStore::YuanRongStore(); }
