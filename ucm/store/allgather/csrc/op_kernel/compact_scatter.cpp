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
    __aicore__ inline void Init(GM_ADDR receiveBuffer,
                                GM_ADDR destinationAddresses,
                                GM_ADDR routes,
                                GM_ADDR chunks,
                                uint32_t rowCount,
                                uint32_t chunksPerBlock,
                                uint32_t tensorCount,
                                uint64_t rankStride,
                                uint64_t shardSize,
                                uint32_t coreCount)
    {
        receiveAddress_ = reinterpret_cast<uint64_t>(receiveBuffer);
        destinationAddresses_ =
            reinterpret_cast<__gm__ uint64_t*>(destinationAddresses);
        routes_ = reinterpret_cast<__gm__ uint32_t*>(routes);
        chunks_ = reinterpret_cast<__gm__ uint64_t*>(chunks);
        rowCount_ = rowCount;
        chunksPerBlock_ = chunksPerBlock;
        tensorCount_ = tensorCount;
        rankStride_ = rankStride;
        shardSize_ = shardSize;
        coreCount_ = coreCount;
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
            if (destination == 0) {
                continue;
            }

            const uint64_t routeOffset = static_cast<uint64_t>(row) * kRouteFields;
            const uint64_t owner = routes_[routeOffset];
            const uint64_t ownerSlot = routes_[routeOffset + 1];
            CopyTask(
                receiveAddress_ + owner * rankStride_ + ownerSlot * shardSize_ +
                    chunks_[chunkOffset + 1],
                destination + chunks_[chunkOffset + 2],
                chunks_[chunkOffset + 3]);
        }
    }

private:
    __aicore__ inline void CopyTask(uint64_t srcAddress, uint64_t dstAddress,
                                    uint64_t bytes)
    {
        GlobalTensor<uint8_t> src;
        GlobalTensor<uint8_t> dst;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(srcAddress), bytes);
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(dstAddress), bytes);

        LocalTensor<uint8_t> local = copyQueue_.AllocTensor<uint8_t>();
        if ((srcAddress % kDataBlockBytes) == 0 &&
            (dstAddress % kDataBlockBytes) == 0 &&
            (bytes % kDataBlockBytes) == 0) {
            DataCopy(local, src, static_cast<uint32_t>(bytes));
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(bytes), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> padding{false, 0, 0, 0};
            DataCopyPad(local, src, params, padding);
        }
        copyQueue_.EnQue(local);
        local = copyQueue_.DeQue<uint8_t>();
        if ((srcAddress % kDataBlockBytes) == 0 &&
            (dstAddress % kDataBlockBytes) == 0 &&
            (bytes % kDataBlockBytes) == 0) {
            DataCopy(dst, local, static_cast<uint32_t>(bytes));
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(bytes), 0, 0, 0};
            DataCopyPad(dst, local, params);
        }
        copyQueue_.FreeTensor(local);
    }

    TPipe pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, kBufferCount> copyQueue_;
    __gm__ uint64_t* destinationAddresses_;
    __gm__ uint32_t* routes_;
    __gm__ uint64_t* chunks_;
    uint32_t rowCount_;
    uint32_t chunksPerBlock_;
    uint32_t tensorCount_;
    uint32_t coreCount_;
    uint64_t receiveAddress_;
    uint64_t rankStride_;
    uint64_t shardSize_;
};

}

extern "C" __global__ __aicore__ void ucm_compact_scatter_kernel(
    GM_ADDR receiveBuffer, GM_ADDR destinationAddresses, GM_ADDR routes,
    GM_ADDR chunks, uint32_t rowCount, uint32_t chunksPerBlock,
    uint32_t tensorCount, uint64_t rankStride, uint64_t shardSize,
    uint32_t coreCount)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CompactScatterKernel kernel;
    kernel.Init(receiveBuffer, destinationAddresses, routes, chunks, rowCount,
                chunksPerBlock, tensorCount, rankStride, shardSize, coreCount);
    kernel.Process();
}

namespace ucm_compact_scatter {

void Launch(void* stream, void* receiveBuffer, void* destinationAddresses,
            void* routes, void* chunks, uint32_t rowCount,
            uint32_t chunksPerBlock, uint32_t tensorCount,
            uint64_t rankStride, uint64_t shardSize, uint32_t usedCores)
{
    ucm_compact_scatter_kernel<<<usedCores, nullptr, stream>>>(
        static_cast<uint8_t*>(receiveBuffer),
        static_cast<uint64_t*>(destinationAddresses),
        static_cast<uint32_t*>(routes),
        static_cast<uint64_t*>(chunks),
        rowCount, chunksPerBlock, tensorCount, rankStride, shardSize, usedCores);
}

}
