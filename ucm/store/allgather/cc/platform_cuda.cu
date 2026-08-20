#include <cstring>
#include <cuda_runtime.h>
#include <nccl.h>
#include "frame_protocol.h"
#include "platform_runtime.h"

namespace UC::AllGatherStore {
namespace {

Status CudaStatus(cudaError_t code, const char* operation)
{
    if (code == cudaSuccess) { return Status::OK(); }
    return Status::Error(fmt::format("{} failed({}): {}", operation, static_cast<int>(code),
                                     cudaGetErrorString(code)));
}

Status NcclStatus(ncclResult_t code, const char* operation)
{
    if (code == ncclSuccess) { return Status::OK(); }
    return Status::Error(fmt::format("{} failed({}): {}", operation, static_cast<int>(code),
                                     ncclGetErrorString(code)));
}

__device__ void CopyBytes(const uint8_t* source, uint8_t* destination, uint64_t bytes)
{
    if (((reinterpret_cast<uintptr_t>(source) | reinterpret_cast<uintptr_t>(destination) | bytes) &
         (sizeof(uint4) - 1)) == 0) {
        const auto* vectorSource = reinterpret_cast<const uint4*>(source);
        auto* vectorDestination = reinterpret_cast<uint4*>(destination);
        const uint64_t vectors = bytes / sizeof(uint4);
        for (uint64_t offset = threadIdx.x; offset < vectors; offset += blockDim.x) {
            vectorDestination[offset] = vectorSource[offset];
        }
        return;
    }
    for (uint64_t offset = threadIdx.x; offset < bytes; offset += blockDim.x) {
        destination[offset] = source[offset];
    }
}

__global__ void SegmentedCopyKernel(const uint64_t* descriptors, const uint32_t* workerOffsets)
{
    const uint32_t worker = blockIdx.x;
    for (uint32_t task = workerOffsets[worker]; task < workerOffsets[worker + 1]; ++task) {
        const uint64_t row = static_cast<uint64_t>(task) * 3;
        const auto* source = reinterpret_cast<const uint8_t*>(descriptors[row]);
        auto* destination = reinterpret_cast<uint8_t*>(descriptors[row + 1]);
        const uint64_t bytes = descriptors[row + 2];
        CopyBytes(source, destination, bytes);
    }
}

__global__ void CompactScatterKernel(const uint8_t* receiveBuffer,
                                     const uint64_t* destinationAddresses, const uint32_t* routes,
                                     const uint64_t* chunks, uint32_t rowCount,
                                     uint32_t chunksPerBlock, uint32_t tensorCount,
                                     uint64_t rankStride, uint64_t shardSize)
{
    const uint32_t taskCount = rowCount * chunksPerBlock;
    for (uint32_t task = blockIdx.x; task < taskCount; task += gridDim.x) {
        const uint32_t row = task / chunksPerBlock;
        const uint32_t chunk = task - row * chunksPerBlock;
        const uint64_t chunkOffset = static_cast<uint64_t>(chunk) * 4;
        const uint32_t tensor = static_cast<uint32_t>(chunks[chunkOffset]);
        const uint64_t destinationAddress =
            destinationAddresses[static_cast<uint64_t>(row) * tensorCount + tensor];
        if (destinationAddress == 0) { continue; }
        const uint64_t routeOffset = static_cast<uint64_t>(row) * 2;
        const uint64_t owner = routes[routeOffset];
        const uint64_t ownerSlot = routes[routeOffset + 1];
        const auto* source =
            receiveBuffer + owner * rankStride + ownerSlot * shardSize + chunks[chunkOffset + 1];
        auto* destination =
            reinterpret_cast<uint8_t*>(destinationAddress) + chunks[chunkOffset + 2];
        const uint64_t bytes = chunks[chunkOffset + 3];
        CopyBytes(source, destination, bytes);
    }
}

__global__ void FramedScatterKernel(const uint8_t* receiveBuffer,
                                    const uint64_t* destinationAddresses, const uint64_t* chunks,
                                    uint32_t* taskError, uint32_t rowCount, uint32_t worldSize,
                                    uint32_t windowBlocks, uint32_t chunksPerBlock,
                                    uint32_t tensorCount, uint64_t frameStride,
                                    uint64_t metadataBytes, uint64_t shardSize, uint64_t sequence,
                                    uint32_t round)
{
    constexpr uint32_t frameMagic = 0x55434631;
    constexpr uint32_t frameVersion = 1;
    constexpr uint32_t headerFields = 8;
    constexpr uint32_t recordFields = 4;
    const uint32_t taskCount = worldSize * windowBlocks * chunksPerBlock;
    for (uint32_t task = blockIdx.x; task < taskCount; task += gridDim.x) {
        const uint32_t recordTask = task / chunksPerBlock;
        const uint32_t chunk = task - recordTask * chunksPerBlock;
        const uint32_t owner = recordTask / windowBlocks;
        const uint32_t recordIndex = recordTask - owner * windowBlocks;
        const auto* frame = receiveBuffer + owner * frameStride;
        const auto* header = reinterpret_cast<const uint32_t*>(frame);
        const uint64_t frameSequence =
            static_cast<uint64_t>(header[3]) | (static_cast<uint64_t>(header[4]) << 32);
        const uint32_t validCount = header[2];
        if (header[0] != frameMagic || header[1] != frameVersion || frameSequence != sequence ||
            (round != kAnyFrameRound && header[5] != round) || validCount > windowBlocks) {
            if (chunk == 0 && recordIndex == 0) { *taskError = 1; }
            continue;
        }
        if (recordIndex >= validCount) { continue; }
        const auto* record = header + headerFields + recordIndex * recordFields;
        const uint32_t row = record[0];
        const uint32_t status = record[1];
        const uint32_t payloadSlot = record[2];
        if (status != 0 || row >= rowCount || payloadSlot >= windowBlocks) {
            if (chunk == 0) { *taskError = 1; }
            continue;
        }
        const uint64_t chunkOffset = static_cast<uint64_t>(chunk) * 4;
        const uint32_t tensor = static_cast<uint32_t>(chunks[chunkOffset]);
        const uint64_t destinationAddress =
            destinationAddresses[static_cast<uint64_t>(row) * tensorCount + tensor];
        if (destinationAddress == 0) { continue; }
        const auto* source = frame + metadataBytes +
                             static_cast<uint64_t>(payloadSlot) * shardSize +
                             chunks[chunkOffset + 1];
        auto* destination =
            reinterpret_cast<uint8_t*>(destinationAddress) + chunks[chunkOffset + 2];
        CopyBytes(source, destination, chunks[chunkOffset + 3]);
    }
}

class CudaPlatformRuntime final : public PlatformRuntime {
public:
    const char* Name() const override { return "cuda"; }

    Status SetDevice(int32_t deviceId) override
    {
        return CudaStatus(cudaSetDevice(deviceId), "cudaSetDevice");
    }

    Status AllocateDevice(void** data, size_t bytes, bool zero) override
    {
        auto status = CudaStatus(cudaMalloc(data, bytes), "cudaMalloc");
        if (status.Success() && zero) {
            status = CudaStatus(cudaMemset(*data, 0, bytes), "cudaMemset");
            if (status.Failure()) {
                (void)cudaFree(*data);
                *data = nullptr;
            }
        }
        return status;
    }

    void FreeDevice(void* data) override
    {
        if (data != nullptr) { (void)cudaFree(data); }
    }

    Status AllocateHost(void** data, size_t bytes) override
    {
        return CudaStatus(cudaMallocHost(data, bytes), "cudaMallocHost");
    }

    void FreeHost(void* data) override
    {
        if (data != nullptr) { (void)cudaFreeHost(data); }
    }

    Status CreateStream(StreamHandle* stream) override
    {
        return CudaStatus(cudaStreamCreateWithFlags(reinterpret_cast<cudaStream_t*>(stream),
                                                    cudaStreamNonBlocking),
                          "cudaStreamCreateWithFlags");
    }

    void DestroyStream(StreamHandle stream) override
    {
        if (stream != nullptr) { (void)cudaStreamDestroy(static_cast<cudaStream_t>(stream)); }
    }

    Status SynchronizeStream(StreamHandle stream) override
    {
        return CudaStatus(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)),
                          "cudaStreamSynchronize");
    }

    Status CreateEvent(EventHandle* event, bool timing) override
    {
        const auto flags = timing ? cudaEventDefault : cudaEventDisableTiming;
        return CudaStatus(cudaEventCreateWithFlags(reinterpret_cast<cudaEvent_t*>(event), flags),
                          "cudaEventCreateWithFlags");
    }

    void DestroyEvent(EventHandle event) override
    {
        if (event != nullptr) { (void)cudaEventDestroy(static_cast<cudaEvent_t>(event)); }
    }

    Status RecordEvent(EventHandle event, StreamHandle stream) override
    {
        return CudaStatus(
            cudaEventRecord(static_cast<cudaEvent_t>(event), static_cast<cudaStream_t>(stream)),
            "cudaEventRecord");
    }

    Expected<bool> QueryEvent(EventHandle event) override
    {
        auto status = cudaEventQuery(static_cast<cudaEvent_t>(event));
        if (status == cudaSuccess) { return true; }
        if (status == cudaErrorNotReady) { return false; }
        return CudaStatus(status, "cudaEventQuery");
    }

    Status SynchronizeEvent(EventHandle event) override
    {
        return CudaStatus(cudaEventSynchronize(static_cast<cudaEvent_t>(event)),
                          "cudaEventSynchronize");
    }

    Status WaitEvent(StreamHandle stream, EventHandle event) override
    {
        return CudaStatus(
            cudaStreamWaitEvent(static_cast<cudaStream_t>(stream), static_cast<cudaEvent_t>(event)),
            "cudaStreamWaitEvent");
    }

    Status EventElapsedTime(float* milliseconds, EventHandle start, EventHandle end) override
    {
        return CudaStatus(cudaEventElapsedTime(milliseconds, static_cast<cudaEvent_t>(start),
                                               static_cast<cudaEvent_t>(end)),
                          "cudaEventElapsedTime");
    }

    Status CopyHostToDevice(void* destination, size_t destinationBytes, const void* source,
                            size_t bytes, StreamHandle stream) override
    {
        if (bytes > destinationBytes) {
            return Status::InvalidParam("host-to-device copy exceeds destination({}/{})", bytes,
                                        destinationBytes);
        }
        if (stream == nullptr) {
            return CudaStatus(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
                              "cudaMemcpy");
        }
        return CudaStatus(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice,
                                          static_cast<cudaStream_t>(stream)),
                          "cudaMemcpyAsync");
    }

    Status CopyDeviceToHost(void* destination, size_t destinationBytes, const void* source,
                            size_t bytes, StreamHandle stream) override
    {
        if (bytes > destinationBytes) {
            return Status::InvalidParam("device-to-host copy exceeds destination({}/{})", bytes,
                                        destinationBytes);
        }
        if (stream == nullptr) {
            return CudaStatus(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost),
                              "cudaMemcpy");
        }
        return CudaStatus(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToHost,
                                          static_cast<cudaStream_t>(stream)),
                          "cudaMemcpyAsync");
    }

    Expected<std::vector<uint8_t>> CreateRootInfo() override
    {
        ncclUniqueId id{};
        auto status = NcclStatus(ncclGetUniqueId(&id), "ncclGetUniqueId");
        if (status.Failure()) { return status; }
        const auto* begin = reinterpret_cast<const uint8_t*>(&id);
        return std::vector<uint8_t>(begin, begin + sizeof(id));
    }

    size_t RootInfoSize() const override { return sizeof(ncclUniqueId); }

    Status CreateCollective(uint32_t rank, uint32_t worldSize, uint32_t bufferMb,
                            uint32_t expansionMode, const std::vector<uint8_t>& rootInfo,
                            CollectiveHandle* collective) override
    {
        (void)bufferMb;
        (void)expansionMode;
        if (rootInfo.size() != sizeof(ncclUniqueId)) {
            return Status::InvalidParam("invalid NCCL root info size({})", rootInfo.size());
        }
        ncclUniqueId id{};
        std::memcpy(&id, rootInfo.data(), sizeof(id));
        return NcclStatus(
            ncclCommInitRank(reinterpret_cast<ncclComm_t*>(collective), worldSize, id, rank),
            "ncclCommInitRank");
    }

    void DestroyCollective(CollectiveHandle collective) override
    {
        if (collective != nullptr) { (void)ncclCommDestroy(static_cast<ncclComm_t>(collective)); }
    }

    Status AllGather(void* send, void* receive, size_t bytes, CollectiveHandle collective,
                     StreamHandle stream) override
    {
        return NcclStatus(
            ncclAllGather(send, receive, bytes, ncclInt8, static_cast<ncclComm_t>(collective),
                          static_cast<cudaStream_t>(stream)),
            "ncclAllGather");
    }

    Status LaunchSegmentedCopy(StreamHandle stream, void* descriptors, void* coreOffsets,
                               uint32_t usedWorkers) override
    {
        SegmentedCopyKernel<<<usedWorkers, 256, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint64_t*>(descriptors), static_cast<const uint32_t*>(coreOffsets));
        return CudaStatus(cudaGetLastError(), "SegmentedCopyKernel");
    }

    Status LaunchCompactScatter(StreamHandle stream, void* receiveBuffer,
                                void* destinationAddresses, void* routes, void* chunks,
                                uint32_t rowCount, uint32_t chunksPerBlock, uint32_t tensorCount,
                                uint64_t rankStride, uint64_t shardSize,
                                uint32_t usedWorkers) override
    {
        CompactScatterKernel<<<usedWorkers, 256, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint8_t*>(receiveBuffer),
            static_cast<const uint64_t*>(destinationAddresses),
            static_cast<const uint32_t*>(routes), static_cast<const uint64_t*>(chunks), rowCount,
            chunksPerBlock, tensorCount, rankStride, shardSize);
        return CudaStatus(cudaGetLastError(), "CompactScatterKernel");
    }

    Status LaunchFramedScatter(StreamHandle stream, void* receiveBuffer, void* destinationAddresses,
                               void* chunks, void* taskError, uint32_t rowCount, uint32_t worldSize,
                               uint32_t windowBlocks, uint32_t chunksPerBlock, uint32_t tensorCount,
                               uint64_t frameStride, uint64_t metadataBytes, uint64_t shardSize,
                               uint64_t sequence, uint32_t round, uint32_t usedWorkers) override
    {
        FramedScatterKernel<<<usedWorkers, 256, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint8_t*>(receiveBuffer),
            static_cast<const uint64_t*>(destinationAddresses),
            static_cast<const uint64_t*>(chunks), static_cast<uint32_t*>(taskError), rowCount,
            worldSize, windowBlocks, chunksPerBlock, tensorCount, frameStride, metadataBytes,
            shardSize, sequence, round);
        return CudaStatus(cudaGetLastError(), "FramedScatterKernel");
    }
};

}  // namespace

std::shared_ptr<PlatformRuntime> CreatePlatformRuntime()
{
    return std::make_shared<CudaPlatformRuntime>();
}

}  // namespace UC::AllGatherStore
