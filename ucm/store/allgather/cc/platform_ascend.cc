#include <acl/acl.h>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include "platform_runtime.h"

namespace ucm_segmented_copy {
void Launch(void* stream, void* descriptors, void* coreOffsets, uint32_t usedCores);
}

namespace ucm_compact_scatter {
void Launch(void* stream, void* receiveBuffer, void* destinationAddresses, void* routes,
            void* chunks, uint32_t rowCount, uint32_t chunksPerBlock, uint32_t tensorCount,
            uint64_t rankStride, uint64_t shardSize, uint32_t usedCores);
void LaunchRemote(void* stream, void* peerBuffers, void* destinationAddresses, void* routes,
                  void* chunks, uint32_t rowCount, uint32_t chunksPerBlock, uint32_t tensorCount,
                  uint64_t shardSize, uint32_t usedCores);
}  // namespace ucm_compact_scatter

namespace UC::AllGatherStore {
namespace {

Status AclStatus(aclError code, const char* operation)
{
    if (code == ACL_SUCCESS) { return Status::OK(); }
    const auto* recent = aclGetRecentErrMsg();
    return Status::Error(fmt::format("{} failed({}): {}", operation, code,
                                     recent != nullptr ? recent : "no recent ACL message"));
}

constexpr size_t kBootstrapKeyBytes = 32;
constexpr size_t kIpcKeyBytes = 256;
class AscendPlatformRuntime final : public PlatformRuntime {
public:
    const char* Name() const override { return "ascend"; }

    Status SetDevice(int32_t deviceId) override
    {
        return AclStatus(aclrtSetDevice(deviceId), "aclrtSetDevice");
    }

    Status AllocateDevice(void** data, size_t bytes, bool zero) override
    {
        auto status =
            AclStatus(aclrtMalloc(data, bytes, ACL_MEM_TYPE_HIGH_BAND_WIDTH), "aclrtMalloc");
        if (status.Success() && zero) {
            status = AclStatus(aclrtMemset(*data, bytes, 0, bytes), "aclrtMemset");
            if (status.Failure()) {
                (void)aclrtFree(*data);
                *data = nullptr;
            }
        }
        return status;
    }

    void FreeDevice(void* data) override
    {
        if (data != nullptr) { (void)aclrtFree(data); }
    }

    Status AllocateHost(void** data, size_t bytes) override
    {
        return AclStatus(aclrtMallocHost(data, bytes), "aclrtMallocHost");
    }

    void FreeHost(void* data) override
    {
        if (data != nullptr) { (void)aclrtFreeHost(data); }
    }

    Status CreateStream(StreamHandle* stream) override
    {
        return AclStatus(aclrtCreateStream(reinterpret_cast<aclrtStream*>(stream)),
                         "aclrtCreateStream");
    }

    void DestroyStream(StreamHandle stream) override
    {
        if (stream != nullptr) { (void)aclrtDestroyStream(static_cast<aclrtStream>(stream)); }
    }

    Status SynchronizeStream(StreamHandle stream) override
    {
        return AclStatus(aclrtSynchronizeStream(static_cast<aclrtStream>(stream)),
                         "aclrtSynchronizeStream");
    }

    Status CreateEvent(EventHandle* event, bool timing) override
    {
        (void)timing;
        return AclStatus(aclrtCreateEvent(reinterpret_cast<aclrtEvent*>(event)),
                         "aclrtCreateEvent");
    }

    void DestroyEvent(EventHandle event) override
    {
        if (event != nullptr) { (void)aclrtDestroyEvent(static_cast<aclrtEvent>(event)); }
    }

    Status RecordEvent(EventHandle event, StreamHandle stream) override
    {
        return AclStatus(
            aclrtRecordEvent(static_cast<aclrtEvent>(event), static_cast<aclrtStream>(stream)),
            "aclrtRecordEvent");
    }

    Expected<bool> QueryEvent(EventHandle event) override
    {
        aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
        auto result = aclrtQueryEventStatus(static_cast<aclrtEvent>(event), &status);
        if (result != ACL_SUCCESS) { return AclStatus(result, "aclrtQueryEventStatus"); }
        return status == ACL_EVENT_RECORDED_STATUS_COMPLETE;
    }

    Status SynchronizeEvent(EventHandle event) override
    {
        return AclStatus(aclrtSynchronizeEvent(static_cast<aclrtEvent>(event)),
                         "aclrtSynchronizeEvent");
    }

    Status WaitEvent(StreamHandle stream, EventHandle event) override
    {
        return AclStatus(
            aclrtStreamWaitEvent(static_cast<aclrtStream>(stream), static_cast<aclrtEvent>(event)),
            "aclrtStreamWaitEvent");
    }

    Status EventElapsedTime(float* milliseconds, EventHandle start, EventHandle end) override
    {
        return AclStatus(aclrtEventElapsedTime(milliseconds, static_cast<aclrtEvent>(start),
                                               static_cast<aclrtEvent>(end)),
                         "aclrtEventElapsedTime");
    }

    Status CopyHostToDevice(void* destination, size_t destinationBytes, const void* source,
                            size_t bytes, StreamHandle stream) override
    {
        if (bytes > destinationBytes) {
            return Status::InvalidParam("host-to-device copy exceeds destination({}/{})", bytes,
                                        destinationBytes);
        }
        if (stream == nullptr) {
            return AclStatus(aclrtMemcpy(destination, destinationBytes, source, bytes,
                                         ACL_MEMCPY_HOST_TO_DEVICE),
                             "aclrtMemcpy");
        }
        return AclStatus(
            aclrtMemcpyAsync(destination, destinationBytes, source, bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE, static_cast<aclrtStream>(stream)),
            "aclrtMemcpyAsync");
    }

    Status CopyDeviceToHost(void* destination, size_t destinationBytes, const void* source,
                            size_t bytes, StreamHandle stream) override
    {
        if (bytes > destinationBytes) {
            return Status::InvalidParam("device-to-host copy exceeds destination({}/{})", bytes,
                                        destinationBytes);
        }
        if (stream == nullptr) {
            return AclStatus(aclrtMemcpy(destination, destinationBytes, source, bytes,
                                         ACL_MEMCPY_DEVICE_TO_HOST),
                             "aclrtMemcpy");
        }
        return AclStatus(
            aclrtMemcpyAsync(destination, destinationBytes, source, bytes,
                             ACL_MEMCPY_DEVICE_TO_HOST, static_cast<aclrtStream>(stream)),
            "aclrtMemcpyAsync");
    }

    Status CopyDeviceBatchAsync(StreamHandle stream, const DeviceCopy* copies,
                                size_t count) override
    {
        if (count == 0) { return Status::OK(); }
        if (stream == nullptr || copies == nullptr) {
            return Status::InvalidParam("invalid device batch copy input");
        }
        for (size_t i = 0; i < count; ++i) {
            const auto& copy = copies[i];
            if (copy.destination == nullptr || copy.source == nullptr || copy.bytes == 0) {
                return Status::InvalidParam("invalid device batch copy descriptor({})", i);
            }
            const auto code =
                aclrtMemcpyAsync(copy.destination, copy.bytes, copy.source, copy.bytes,
                                 ACL_MEMCPY_DEFAULT, static_cast<aclrtStream>(stream));
            if (code != ACL_SUCCESS) {
                const auto* recent = aclGetRecentErrMsg();
                return Status::Error(
                    fmt::format("aclrtMemcpyAsync failed({}) at device batch item({}): {}", code, i,
                                recent != nullptr ? recent : "no recent ACL message"));
            }
        }
        return Status::OK();
    }

    bool SupportsRemoteScatter() const override { return true; }

    Expected<int32_t> IpcProcessId() override
    {
        int32_t processId = 0;
        auto status = AclStatus(aclrtDeviceGetBareTgid(&processId), "aclrtDeviceGetBareTgid");
        if (status.Failure()) { return status; }
        return processId;
    }

    Expected<std::vector<uint8_t>> ExportDeviceMemory(void* data, size_t bytes) override
    {
        std::vector<uint8_t> key(kIpcKeyBytes, 0);
        auto status =
            AclStatus(aclrtIpcMemGetExportKey(data, bytes, reinterpret_cast<char*>(key.data()),
                                              key.size(), ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT),
                      "aclrtIpcMemGetExportKey");
        if (status.Failure()) { return status; }
        return key;
    }

    Status AuthorizeDeviceMemory(const std::vector<uint8_t>& handle,
                                 const std::vector<int32_t>& processIds) override
    {
        if (handle.size() != kIpcKeyBytes || processIds.empty()) {
            return Status::InvalidParam("invalid Ascend IPC authorization input");
        }
        auto ids = processIds;
        return AclStatus(aclrtIpcMemSetImportPid(reinterpret_cast<const char*>(handle.data()),
                                                 ids.data(), ids.size()),
                         "aclrtIpcMemSetImportPid");
    }

    Status OpenDeviceMemory(const std::vector<uint8_t>& handle, void** data) override
    {
        if (handle.size() != kIpcKeyBytes) {
            return Status::InvalidParam("invalid Ascend IPC memory handle size({})", handle.size());
        }
        auto status =
            AclStatus(aclrtIpcMemImportByKey(data, reinterpret_cast<const char*>(handle.data()),
                                             ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS),
                      "aclrtIpcMemImportByKey");
        if (status.Success()) {
            std::lock_guard<std::mutex> lock(importsMutex_);
            imports_[*data] = std::string(reinterpret_cast<const char*>(handle.data()));
        }
        return status;
    }

    void CloseDeviceMemory(void* data) override
    {
        std::string key;
        {
            std::lock_guard<std::mutex> lock(importsMutex_);
            const auto found = imports_.find(data);
            if (found == imports_.end()) { return; }
            key = std::move(found->second);
            imports_.erase(found);
        }
        (void)aclrtIpcMemClose(key.c_str());
    }

    Expected<std::vector<uint8_t>> CreateBootstrapKey() override
    {
        std::vector<uint8_t> key(kBootstrapKeyBytes);
        std::random_device random;
        for (auto& byte : key) { byte = static_cast<uint8_t>(random()); }
        return key;
    }

    size_t BootstrapKeySize() const override { return kBootstrapKeyBytes; }

    Status LaunchSegmentedCopy(StreamHandle stream, void* descriptors, void* coreOffsets,
                               uint32_t usedWorkers) override
    {
        ucm_segmented_copy::Launch(stream, descriptors, coreOffsets, usedWorkers);
        return Status::OK();
    }

    Status LaunchCompactScatter(StreamHandle stream, void* receiveBuffer,
                                void* destinationAddresses, void* routes, void* chunks,
                                uint32_t rowCount, uint32_t chunksPerBlock, uint32_t tensorCount,
                                uint64_t rankStride, uint64_t shardSize,
                                uint32_t usedWorkers) override
    {
        ucm_compact_scatter::Launch(stream, receiveBuffer, destinationAddresses, routes, chunks,
                                    rowCount, chunksPerBlock, tensorCount, rankStride, shardSize,
                                    usedWorkers);
        return Status::OK();
    }

    Status LaunchRemoteScatter(StreamHandle stream, void* peerBuffers, void* destinationAddresses,
                               void* routes, void* chunks, uint32_t rowCount,
                               uint32_t chunksPerBlock, uint32_t tensorCount, uint64_t shardSize,
                               uint32_t usedWorkers) override
    {
        ucm_compact_scatter::LaunchRemote(stream, peerBuffers, destinationAddresses, routes, chunks,
                                          rowCount, chunksPerBlock, tensorCount, shardSize,
                                          usedWorkers);
        return Status::OK();
    }

private:
    std::mutex importsMutex_;
    std::unordered_map<void*, std::string> imports_;
};

}  // namespace

std::shared_ptr<PlatformRuntime> CreatePlatformRuntime()
{
    return std::make_shared<AscendPlatformRuntime>();
}

}  // namespace UC::AllGatherStore
