#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t kChunkFields = 4;
constexpr uint32_t kRouteFields = 2;
constexpr uint32_t kTileBytes = 32 * 1024;
constexpr uint32_t kBufferCount = 2;
constexpr uint32_t kDataBlockBytes = 32;

class CompactScatterKernel {
public:
    __aicore__ inline void Init(GM_ADDR source, GM_ADDR destinationAddresses, GM_ADDR routes,
                                GM_ADDR chunks, uint32_t rowCount, uint32_t chunksPerBlock,
                                uint32_t tensorCount, uint64_t rankStride, uint64_t shardSize,
                                uint32_t coreCount, bool remote)
    {
        sourceAddress_ = reinterpret_cast<uint64_t>(source);
        peerBuffers_ = reinterpret_cast<__gm__ uint64_t*>(source);
        destinationAddresses_ = reinterpret_cast<__gm__ uint64_t*>(destinationAddresses);
        routes_ = reinterpret_cast<__gm__ uint32_t*>(routes);
        chunks_ = reinterpret_cast<__gm__ uint64_t*>(chunks);
        rowCount_ = rowCount;
        chunksPerBlock_ = chunksPerBlock;
        tensorCount_ = tensorCount;
        rankStride_ = rankStride;
        shardSize_ = shardSize;
        coreCount_ = coreCount;
        remote_ = remote;
        pipe_.InitBuffer(copyQueue_, kBufferCount, kTileBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t taskCount = rowCount_ * chunksPerBlock_;
        for (uint32_t task = GetBlockIdx(); task < taskCount; task += coreCount_) {
            const uint32_t row = task / chunksPerBlock_;
            const uint32_t chunk = task - row * chunksPerBlock_;
            const uint64_t chunkOffset = static_cast<uint64_t>(chunk) * kChunkFields;
            const uint32_t tensor = static_cast<uint32_t>(chunks_[chunkOffset]);
            const uint64_t destination =
                destinationAddresses_[static_cast<uint64_t>(row) * tensorCount_ + tensor];
            if (destination == 0) { continue; }

            const uint64_t routeOffset = static_cast<uint64_t>(row) * kRouteFields;
            const uint64_t owner = routes_[routeOffset];
            const uint64_t ownerSlot = routes_[routeOffset + 1];
            const uint64_t sourceBase =
                remote_ ? peerBuffers_[owner] : sourceAddress_ + owner * rankStride_;
            CopyTask(sourceBase + ownerSlot * shardSize_ + chunks_[chunkOffset + 1],
                     destination + chunks_[chunkOffset + 2], chunks_[chunkOffset + 3]);
        }
    }

private:
    __aicore__ inline void CopyTask(uint64_t source, uint64_t destination, uint64_t bytes)
    {
        for (uint64_t offset = 0; offset < bytes; offset += kTileBytes) {
            const uint32_t tile =
                static_cast<uint32_t>(bytes - offset < kTileBytes ? bytes - offset : kTileBytes);
            GlobalTensor<uint8_t> input;
            GlobalTensor<uint8_t> output;
            input.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(source + offset), tile);
            output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(destination + offset), tile);

            LocalTensor<uint8_t> local = copyQueue_.AllocTensor<uint8_t>();
            if (((source + offset) % kDataBlockBytes) == 0 &&
                ((destination + offset) % kDataBlockBytes) == 0 && (tile % kDataBlockBytes) == 0) {
                DataCopy(local, input, tile);
            } else {
                DataCopyExtParams params{1, tile, 0, 0, 0};
                DataCopyPadExtParams<uint8_t> padding{false, 0, 0, 0};
                DataCopyPad(local, input, params, padding);
            }
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<uint8_t>();
            if (((source + offset) % kDataBlockBytes) == 0 &&
                ((destination + offset) % kDataBlockBytes) == 0 && (tile % kDataBlockBytes) == 0) {
                DataCopy(output, local, tile);
            } else {
                DataCopyExtParams params{1, tile, 0, 0, 0};
                DataCopyPad(output, local, params);
            }
            copyQueue_.FreeTensor(local);
        }
    }

    TPipe pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, kBufferCount> copyQueue_;
    __gm__ uint64_t* peerBuffers_;
    __gm__ uint64_t* destinationAddresses_;
    __gm__ uint32_t* routes_;
    __gm__ uint64_t* chunks_;
    uint32_t rowCount_;
    uint32_t chunksPerBlock_;
    uint32_t tensorCount_;
    uint32_t coreCount_;
    uint64_t sourceAddress_;
    uint64_t rankStride_;
    uint64_t shardSize_;
    bool remote_;
};

}  // namespace

extern "C" __global__ __aicore__ void ucm_compact_scatter_kernel(
    GM_ADDR source, GM_ADDR destinationAddresses, GM_ADDR routes, GM_ADDR chunks, uint32_t rowCount,
    uint32_t chunksPerBlock, uint32_t tensorCount, uint64_t rankStride, uint64_t shardSize,
    uint32_t coreCount, bool remote)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CompactScatterKernel kernel;
    kernel.Init(source, destinationAddresses, routes, chunks, rowCount, chunksPerBlock, tensorCount,
                rankStride, shardSize, coreCount, remote);
    kernel.Process();
}

namespace ucm_compact_scatter {

void Launch(void* stream, void* receiveBuffer, void* destinationAddresses, void* routes,
            void* chunks, uint32_t rowCount, uint32_t chunksPerBlock, uint32_t tensorCount,
            uint64_t rankStride, uint64_t shardSize, uint32_t usedCores)
{
    ucm_compact_scatter_kernel<<<usedCores, nullptr, stream>>>(
        static_cast<uint8_t*>(receiveBuffer), static_cast<uint64_t*>(destinationAddresses),
        static_cast<uint32_t*>(routes), static_cast<uint64_t*>(chunks), rowCount, chunksPerBlock,
        tensorCount, rankStride, shardSize, usedCores, false);
}

void LaunchRemote(void* stream, void* peerBuffers, void* destinationAddresses, void* routes,
                  void* chunks, uint32_t rowCount, uint32_t chunksPerBlock, uint32_t tensorCount,
                  uint64_t shardSize, uint32_t usedCores)
{
    ucm_compact_scatter_kernel<<<usedCores, nullptr, stream>>>(
        static_cast<uint64_t*>(peerBuffers), static_cast<uint64_t*>(destinationAddresses),
        static_cast<uint32_t*>(routes), static_cast<uint64_t*>(chunks), rowCount, chunksPerBlock,
        tensorCount, 0, shardSize, usedCores, true);
}

}  // namespace ucm_compact_scatter
