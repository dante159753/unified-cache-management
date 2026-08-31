#ifndef UNIFIEDCACHE_ALLGATHER_PLATFORM_RUNTIME_H
#define UNIFIEDCACHE_ALLGATHER_PLATFORM_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "status/status.h"

namespace UC::AllGatherStore {

using StreamHandle = void*;
using EventHandle = void*;

struct DeviceCopy {
    void* destination;
    const void* source;
    size_t bytes;
};

class PlatformRuntime {
public:
    virtual ~PlatformRuntime() = default;

    virtual const char* Name() const = 0;
    virtual Status SetDevice(int32_t deviceId) = 0;
    virtual Status AllocateDevice(void** data, size_t bytes, bool zero) = 0;
    virtual void FreeDevice(void* data) = 0;
    virtual Status AllocateHost(void** data, size_t bytes) = 0;
    virtual void FreeHost(void* data) = 0;

    virtual Status CreateStream(StreamHandle* stream) = 0;
    virtual void DestroyStream(StreamHandle stream) = 0;
    virtual Status SynchronizeStream(StreamHandle stream) = 0;
    virtual Status CreateEvent(EventHandle* event, bool timing = false) = 0;
    virtual void DestroyEvent(EventHandle event) = 0;
    virtual Status RecordEvent(EventHandle event, StreamHandle stream) = 0;
    virtual Expected<bool> QueryEvent(EventHandle event) = 0;
    virtual Status SynchronizeEvent(EventHandle event) = 0;
    virtual Status WaitEvent(StreamHandle stream, EventHandle event) = 0;
    virtual Status EventElapsedTime(float* milliseconds, EventHandle start, EventHandle end) = 0;

    virtual Status CopyHostToDevice(void* destination, size_t destinationBytes, const void* source,
                                    size_t bytes, StreamHandle stream) = 0;
    virtual Status CopyDeviceToHost(void* destination, size_t destinationBytes, const void* source,
                                    size_t bytes, StreamHandle stream) = 0;
    virtual Status CopyDeviceBatchAsync(StreamHandle stream, const DeviceCopy* copies,
                                        size_t count) = 0;
    virtual bool SupportsRemoteScatter() const = 0;
    virtual Expected<int32_t> IpcProcessId() = 0;
    virtual Expected<std::vector<uint8_t>> ExportDeviceMemory(void* data, size_t bytes) = 0;
    virtual Status AuthorizeDeviceMemory(const std::vector<uint8_t>& handle,
                                         const std::vector<int32_t>& processIds) = 0;
    virtual Status OpenDeviceMemory(const std::vector<uint8_t>& handle, void** data) = 0;
    virtual void CloseDeviceMemory(void* data) = 0;

    virtual Expected<std::vector<uint8_t>> CreateBootstrapKey() = 0;
    virtual size_t BootstrapKeySize() const = 0;
    virtual Status LaunchSegmentedCopy(StreamHandle stream, void* descriptors, void* coreOffsets,
                                       uint32_t usedWorkers) = 0;
    virtual Status LaunchCompactScatter(StreamHandle stream, void* receiveBuffer,
                                        void* destinationAddresses, void* routes, void* chunks,
                                        uint32_t rowCount, uint32_t chunksPerBlock,
                                        uint32_t tensorCount, uint64_t rankStride,
                                        uint64_t shardSize, uint32_t usedWorkers) = 0;
    virtual Status LaunchRemoteScatter(StreamHandle stream, void* peerBuffers,
                                       void* destinationAddresses, void* routes, void* chunks,
                                       uint32_t rowCount, uint32_t chunksPerBlock,
                                       uint32_t tensorCount, uint64_t shardSize,
                                       uint32_t usedWorkers) = 0;
};

std::shared_ptr<PlatformRuntime> CreatePlatformRuntime();

}  // namespace UC::AllGatherStore

#endif
