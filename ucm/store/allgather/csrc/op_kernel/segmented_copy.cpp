#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t kDescriptorFields = 3;
constexpr uint32_t kTileBytes = 32 * 1024;
constexpr uint32_t kBufferCount = 2;
constexpr uint32_t kDataBlockBytes = 32;

class SegmentedCopyKernel {
public:
    __aicore__ inline void Init(GM_ADDR descriptors, GM_ADDR coreOffsets)
    {
        descriptors_ = reinterpret_cast<__gm__ uint64_t*>(descriptors);
        coreOffsets_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(coreOffsets));
        pipe_.InitBuffer(copyQueue_, kBufferCount, kTileBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t begin = coreOffsets_.GetValue(core);
        const uint32_t end = coreOffsets_.GetValue(core + 1);
        for (uint32_t task = begin; task < end; ++task) {
            const uint64_t row = static_cast<uint64_t>(task) * kDescriptorFields;
            CopyTask(descriptors_[row], descriptors_[row + 1], descriptors_[row + 2]);
        }
    }

private:
    __aicore__ inline void CopyTask(uint64_t srcAddress, uint64_t dstAddress, uint64_t bytes)
    {
        GlobalTensor<uint8_t> src;
        GlobalTensor<uint8_t> dst;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(srcAddress), bytes);
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(dstAddress), bytes);

        for (uint64_t offset = 0; offset < bytes; offset += kTileBytes) {
            const uint32_t current = static_cast<uint32_t>(
                bytes - offset < kTileBytes ? bytes - offset : kTileBytes);
            LocalTensor<uint8_t> local = copyQueue_.AllocTensor<uint8_t>();
            if (((srcAddress + offset) % kDataBlockBytes) == 0 &&
                ((dstAddress + offset) % kDataBlockBytes) == 0 &&
                (current % kDataBlockBytes) == 0) {
                DataCopy(local, src[offset], current);
            } else {
                DataCopyExtParams params{1, current, 0, 0, 0};
                DataCopyPadExtParams<uint8_t> padding{false, 0, 0, 0};
                DataCopyPad(local, src[offset], params, padding);
            }
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<uint8_t>();
            if (((srcAddress + offset) % kDataBlockBytes) == 0 &&
                ((dstAddress + offset) % kDataBlockBytes) == 0 &&
                (current % kDataBlockBytes) == 0) {
                DataCopy(dst[offset], local, current);
            } else {
                DataCopyExtParams params{1, current, 0, 0, 0};
                DataCopyPad(dst[offset], local, params);
            }
            copyQueue_.FreeTensor(local);
        }
    }

    TPipe pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, kBufferCount> copyQueue_;
    GlobalTensor<uint32_t> coreOffsets_;
    __gm__ uint64_t* descriptors_;
};

}

extern "C" __global__ __aicore__ void ucm_segmented_copy_kernel(
    GM_ADDR descriptors, GM_ADDR coreOffsets)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SegmentedCopyKernel kernel;
    kernel.Init(descriptors, coreOffsets);
    kernel.Process();
}

namespace ucm_segmented_copy {

void Launch(void* stream, void* descriptors, void* coreOffsets, uint32_t usedCores)
{
    ucm_segmented_copy_kernel<<<usedCores, nullptr, stream>>>(
        static_cast<uint64_t*>(descriptors), static_cast<uint32_t*>(coreOffsets));
}

}
