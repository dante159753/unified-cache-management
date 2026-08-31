#include <cstring>
#include <cuda_runtime.h>
#include <random>
#include <unistd.h>
#include "platform_runtime.h"

namespace UC::AllGatherStore {
namespace {

Status CudaStatus(cudaError_t code, const char* operation)
{
    if (code == cudaSuccess) { return Status::OK(); }
    return Status::Error(fmt::format("{} failed({}): {}", operation, static_cast<int>(code),
                                     cudaGetErrorString(code)));
}

constexpr size_t kBootstrapKeyBytes = 32;

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

__global__ void RemoteScatterKernel(const uint8_t* const* peerBuffers,
                                    const uint64_t* destinationAddresses, const uint32_t* routes,
                                    const uint64_t* chunks, uint32_t rowCount,
                                    uint32_t chunksPerBlock, uint32_t tensorCount,
                                    uint64_t shardSize)
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
        const uint32_t owner = routes[routeOffset];
        const uint32_t ownerSlot = routes[routeOffset + 1];
        const auto* source = peerBuffers[owner] + static_cast<uint64_t>(ownerSlot) * shardSize +
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
            auto status = CudaStatus(
                cudaMemcpyAsync(copy.destination, copy.source, copy.bytes, cudaMemcpyDefault,
                                static_cast<cudaStream_t>(stream)),
                "cudaMemcpyAsync");
            if (status.Failure()) { return status; }
        }
        return Status::OK();
    }

    bool SupportsRemoteScatter() const override { return true; }

    Expected<int32_t> IpcProcessId() override { return static_cast<int32_t>(getpid()); }

    Expected<std::vector<uint8_t>> ExportDeviceMemory(void* data, size_t) override
    {
        cudaIpcMemHandle_t handle{};
        auto status = CudaStatus(cudaIpcGetMemHandle(&handle, data), "cudaIpcGetMemHandle");
        if (status.Failure()) { return status; }
        const auto* begin = reinterpret_cast<const uint8_t*>(&handle);
        return std::vector<uint8_t>(begin, begin + sizeof(handle));
    }

    Status AuthorizeDeviceMemory(const std::vector<uint8_t>&,
                                 const std::vector<int32_t>&) override
    {
        return Status::OK();
    }

    Status OpenDeviceMemory(const std::vector<uint8_t>& handle, void** data) override
    {
        if (handle.size() != sizeof(cudaIpcMemHandle_t)) {
            return Status::InvalidParam("invalid CUDA IPC memory handle size({})", handle.size());
        }
        cudaIpcMemHandle_t value{};
        std::memcpy(&value, handle.data(), sizeof(value));
        return CudaStatus(cudaIpcOpenMemHandle(data, value, cudaIpcMemLazyEnablePeerAccess),
                          "cudaIpcOpenMemHandle");
    }

    void CloseDeviceMemory(void* data) override
    {
        if (data != nullptr) { (void)cudaIpcCloseMemHandle(data); }
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

    Status LaunchRemoteScatter(StreamHandle stream, void* peerBuffers, void* destinationAddresses,
                               void* routes, void* chunks, uint32_t rowCount,
                               uint32_t chunksPerBlock, uint32_t tensorCount, uint64_t shardSize,
                               uint32_t usedWorkers) override
    {
        RemoteScatterKernel<<<usedWorkers, 256, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint8_t* const*>(peerBuffers),
            static_cast<const uint64_t*>(destinationAddresses),
            static_cast<const uint32_t*>(routes), static_cast<const uint64_t*>(chunks), rowCount,
            chunksPerBlock, tensorCount, shardSize);
        return CudaStatus(cudaGetLastError(), "RemoteScatterKernel");
    }

};

}  // namespace

std::shared_ptr<PlatformRuntime> CreatePlatformRuntime()
{
    return std::make_shared<CudaPlatformRuntime>();
}

}  // namespace UC::AllGatherStore
