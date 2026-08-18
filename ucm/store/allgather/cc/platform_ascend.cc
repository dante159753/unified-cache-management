#include "platform_runtime.h"

#include <acl/acl.h>
#include <hccl/hccl.h>

#include <cstring>

namespace ucm_segmented_copy {
void Launch(void* stream, void* descriptors, void* coreOffsets, uint32_t usedCores);
}

namespace ucm_compact_scatter {
void Launch(void* stream, void* receiveBuffer, void* destinationAddresses, void* routes,
            void* chunks, uint32_t rowCount,
            uint32_t chunksPerBlock, uint32_t tensorCount,
            uint64_t rankStride, uint64_t shardSize, uint32_t usedCores);
}

namespace UC::AllGatherStore {
namespace {

Status AclStatus(aclError code, const char* operation)
{
    if (code == ACL_SUCCESS) { return Status::OK(); }
    return Status::Error(fmt::format("{} failed({})", operation, code));
}

Status HcclStatus(HcclResult code, const char* operation)
{
    if (code == HCCL_SUCCESS) { return Status::OK(); }
    return Status::Error(fmt::format("{} failed({})", operation, static_cast<int>(code)));
}

class AscendPlatformRuntime final : public PlatformRuntime {
public:
    const char* Name() const override { return "ascend"; }

    Status SetDevice(int32_t deviceId) override
    {
        return AclStatus(aclrtSetDevice(deviceId), "aclrtSetDevice");
    }

    Status AllocateDevice(void** data, size_t bytes, bool zero) override
    {
        auto status = AclStatus(aclrtMalloc(data, bytes, ACL_MEM_TYPE_HIGH_BAND_WIDTH),
                                "aclrtMalloc");
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
        return AclStatus(aclrtRecordEvent(static_cast<aclrtEvent>(event),
                                          static_cast<aclrtStream>(stream)),
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
        return AclStatus(aclrtStreamWaitEvent(static_cast<aclrtStream>(stream),
                                              static_cast<aclrtEvent>(event)),
                         "aclrtStreamWaitEvent");
    }

    Status EventElapsedTime(float* milliseconds, EventHandle start,
                            EventHandle end) override
    {
        return AclStatus(aclrtEventElapsedTime(milliseconds,
                                               static_cast<aclrtEvent>(start),
                                               static_cast<aclrtEvent>(end)),
                         "aclrtEventElapsedTime");
    }

    Status CopyHostToDevice(void* destination, size_t destinationBytes,
                            const void* source, size_t bytes,
                            StreamHandle stream) override
    {
        if (bytes > destinationBytes) {
            return Status::InvalidParam("host-to-device copy exceeds destination({}/{})",
                                        bytes, destinationBytes);
        }
        if (stream == nullptr) {
            return AclStatus(aclrtMemcpy(destination, destinationBytes, source, bytes,
                                         ACL_MEMCPY_HOST_TO_DEVICE),
                             "aclrtMemcpy");
        }
        return AclStatus(aclrtMemcpyAsync(destination, destinationBytes, source, bytes,
                                          ACL_MEMCPY_HOST_TO_DEVICE,
                                          static_cast<aclrtStream>(stream)),
                         "aclrtMemcpyAsync");
    }

    Expected<std::vector<uint8_t>> CreateRootInfo() override
    {
        HcclRootInfo info{};
        auto status = HcclStatus(HcclGetRootInfo(&info), "HcclGetRootInfo");
        if (status.Failure()) { return status; }
        return std::vector<uint8_t>(reinterpret_cast<uint8_t*>(info.internal),
                                    reinterpret_cast<uint8_t*>(info.internal) +
                                        HCCL_ROOT_INFO_BYTES);
    }

    size_t RootInfoSize() const override { return HCCL_ROOT_INFO_BYTES; }

    Status CreateCollective(uint32_t rank, uint32_t worldSize, uint32_t bufferMb,
                            uint32_t expansionMode,
                            const std::vector<uint8_t>& rootInfo,
                            CollectiveHandle* collective) override
    {
        if (rootInfo.size() != HCCL_ROOT_INFO_BYTES) {
            return Status::InvalidParam("invalid HCCL root info size({})", rootInfo.size());
        }
        HcclRootInfo info{};
        std::memcpy(info.internal, rootInfo.data(), HCCL_ROOT_INFO_BYTES);
        HcclCommConfig config{};
        HcclCommConfigInit(&config);
        config.hcclBufferSize = bufferMb;
        config.hcclOpExpansionMode = expansionMode;
        return HcclStatus(HcclCommInitRootInfoConfig(
                              worldSize, &info, rank, &config,
                              reinterpret_cast<HcclComm*>(collective)),
                          "HcclCommInitRootInfoConfig");
    }

    void DestroyCollective(CollectiveHandle collective) override
    {
        if (collective != nullptr) { (void)HcclCommDestroy(static_cast<HcclComm>(collective)); }
    }

    Status AllGather(void* send, void* receive, size_t bytes,
                     CollectiveHandle collective, StreamHandle stream) override
    {
        return HcclStatus(HcclAllGather(send, receive, bytes, HCCL_DATA_TYPE_INT8,
                                        static_cast<HcclComm>(collective),
                                        static_cast<aclrtStream>(stream)),
                          "HcclAllGather");
    }

    bool SupportsAllGatherV() const override { return true; }

    Status AllGatherV(void* send, size_t sendBytes, void* receive,
                      const uint64_t* receiveBytes,
                      const uint64_t* receiveDisplacements,
                      CollectiveHandle collective, StreamHandle stream) override
    {
        return HcclStatus(
            HcclAllGatherV(send, sendBytes, receive, receiveBytes,
                           receiveDisplacements, HCCL_DATA_TYPE_INT8,
                           static_cast<HcclComm>(collective),
                           static_cast<aclrtStream>(stream)),
            "HcclAllGatherV");
    }

    Status LaunchSegmentedCopy(StreamHandle stream, void* descriptors,
                               void* coreOffsets, uint32_t usedWorkers) override
    {
        ucm_segmented_copy::Launch(stream, descriptors, coreOffsets, usedWorkers);
        return Status::OK();
    }

    Status LaunchCompactScatter(StreamHandle stream, void* receiveBuffer,
                                void* destinationAddresses, void* routes, void* chunks,
                                uint32_t rowCount,
                                uint32_t chunksPerBlock,
                                uint32_t tensorCount, uint64_t rankStride,
                                uint64_t shardSize, uint32_t usedWorkers) override
    {
        ucm_compact_scatter::Launch(stream, receiveBuffer, destinationAddresses, routes,
                                    chunks, rowCount, chunksPerBlock, tensorCount,
                                    rankStride, shardSize, usedWorkers);
        return Status::OK();
    }
};

}  // namespace

std::shared_ptr<PlatformRuntime> CreatePlatformRuntime()
{
    return std::make_shared<AscendPlatformRuntime>();
}

}  // namespace UC::AllGatherStore
