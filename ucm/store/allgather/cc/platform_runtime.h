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
using CollectiveHandle = void*;

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

    virtual Expected<std::vector<uint8_t>> CreateRootInfo() = 0;
    virtual size_t RootInfoSize() const = 0;
    virtual Status CreateCollective(uint32_t rank, uint32_t worldSize, uint32_t bufferMb,
                                    uint32_t expansionMode, const std::vector<uint8_t>& rootInfo,
                                    CollectiveHandle* collective) = 0;
    virtual void DestroyCollective(CollectiveHandle collective) = 0;
    virtual Status AllGather(void* send, void* receive, size_t bytes, CollectiveHandle collective,
                             StreamHandle stream) = 0;
    virtual bool SupportsAllGatherV() const { return false; }
    virtual Status AllGatherV(void* send, size_t sendBytes, void* receive,
                              const uint64_t* receiveBytes, const uint64_t* receiveDisplacements,
                              CollectiveHandle collective, StreamHandle stream)
    {
        (void)send;
        (void)sendBytes;
        (void)receive;
        (void)receiveBytes;
        (void)receiveDisplacements;
        (void)collective;
        (void)stream;
        return Status::Error("AllGatherV is not supported");
    }

    virtual Status LaunchSegmentedCopy(StreamHandle stream, void* descriptors, void* coreOffsets,
                                       uint32_t usedWorkers) = 0;
    virtual Status LaunchCompactScatter(StreamHandle stream, void* receiveBuffer,
                                        void* destinationAddresses, void* routes, void* chunks,
                                        uint32_t rowCount, uint32_t chunksPerBlock,
                                        uint32_t tensorCount, uint64_t rankStride,
                                        uint64_t shardSize, uint32_t usedWorkers) = 0;
    virtual Status LaunchFramedScatter(StreamHandle stream, void* receiveBuffer,
                                       void* destinationAddresses, void* chunks, void* taskError,
                                       uint32_t rowCount, uint32_t worldSize, uint32_t windowBlocks,
                                       uint32_t chunksPerBlock, uint32_t tensorCount,
                                       uint64_t frameStride, uint64_t metadataBytes,
                                       uint64_t shardSize, uint64_t sequence, uint32_t round,
                                       uint32_t usedWorkers) = 0;
};

std::shared_ptr<PlatformRuntime> CreatePlatformRuntime();

}  // namespace UC::AllGatherStore

#endif
