#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <acl/acl_rt.h>

#include <algorithm>
#include <cstdint>

namespace ucm_segmented_copy {
void Launch(void* stream, void* descriptors, void* coreOffsets, uint32_t usedCores);
}

namespace ucm_compact_scatter {
void Launch(void* stream, void* receiveBuffer, void* destinationAddresses,
            void* routes, void* chunks, uint32_t rowCount,
            uint32_t chunksPerBlock, uint32_t tensorCount,
            uint64_t rankStride, uint64_t shardSize, uint32_t usedCores);
}

namespace {

void Copy(const torch::Tensor& descriptors, const torch::Tensor& coreOffsets, int64_t usedCores)
{
    TORCH_CHECK(descriptors.device().type() == c10::DeviceType::PrivateUse1,
                "descriptors must be an NPU tensor");
    TORCH_CHECK(coreOffsets.device().type() == c10::DeviceType::PrivateUse1,
                "core_offsets must be an NPU tensor");
    TORCH_CHECK(descriptors.scalar_type() == torch::kInt64,
                "descriptors must have dtype int64");
    TORCH_CHECK(coreOffsets.scalar_type() == torch::kInt32,
                "core_offsets must have dtype int32");
    TORCH_CHECK(descriptors.is_contiguous() && descriptors.dim() == 2 &&
                    descriptors.size(1) == 3,
                "descriptors must be contiguous [N, 3]");
    TORCH_CHECK(coreOffsets.is_contiguous() && coreOffsets.dim() == 1,
                "core_offsets must be contiguous");
    TORCH_CHECK(usedCores > 0 && usedCores <= 40, "used_cores must be in [1, 40]");
    TORCH_CHECK(coreOffsets.size(0) == usedCores + 1,
                "core_offsets length must be used_cores + 1");

    ucm_segmented_copy::Launch(
        c10_npu::getCurrentNPUStream().stream(),
        descriptors.data_ptr<int64_t>(),
        coreOffsets.data_ptr<int32_t>(),
        static_cast<uint32_t>(usedCores));
}

void WaitEvent(int64_t eventHandle)
{
    if (eventHandle == 0) { return; }
    auto result = aclrtStreamWaitEvent(
        c10_npu::getCurrentNPUStream().stream(),
        reinterpret_cast<aclrtEvent>(eventHandle));
    TORCH_CHECK(result == ACL_SUCCESS, "aclrtStreamWaitEvent failed: ", result);
}

void Scatter(const torch::Tensor& receiveBuffer,
             const torch::Tensor& destinationAddresses,
             const torch::Tensor& routes,
             const torch::Tensor& chunks,
             int64_t rowCount,
             int64_t rankStride,
             int64_t shardSize)
{
    const auto npu = c10::DeviceType::PrivateUse1;
    TORCH_CHECK(receiveBuffer.device().type() == npu,
                "receive_buffer must be an NPU tensor");
    TORCH_CHECK(destinationAddresses.device().type() == npu,
                "destination_addresses must be an NPU tensor");
    TORCH_CHECK(routes.device().type() == npu,
                "routes must be an NPU tensor");
    TORCH_CHECK(chunks.device().type() == npu,
                "chunks must be an NPU tensor");
    TORCH_CHECK(receiveBuffer.scalar_type() == torch::kUInt8,
                "receive_buffer must have dtype uint8");
    TORCH_CHECK(destinationAddresses.scalar_type() == torch::kInt64,
                "destination_addresses must have dtype int64");
    TORCH_CHECK(routes.scalar_type() == torch::kInt32,
                "routes must have dtype int32");
    TORCH_CHECK(chunks.scalar_type() == torch::kInt64,
                "chunks must have dtype int64");
    TORCH_CHECK(receiveBuffer.is_contiguous() && receiveBuffer.dim() == 1,
                "receive_buffer must be contiguous and one-dimensional");
    TORCH_CHECK(destinationAddresses.is_contiguous() &&
                    destinationAddresses.dim() == 2,
                "destination_addresses must be contiguous and two-dimensional");
    TORCH_CHECK(routes.is_contiguous() && routes.dim() == 2 && routes.size(1) == 2,
                "routes must be contiguous [N, 2]");
    TORCH_CHECK(chunks.is_contiguous() && chunks.dim() == 2 && chunks.size(1) == 4,
                "chunks must be contiguous [N, 4]");
    TORCH_CHECK(rowCount > 0 && rowCount <= destinationAddresses.size(0) &&
                    rowCount <= routes.size(0),
                "row_count exceeds metadata capacity");
    TORCH_CHECK(chunks.size(0) > 0, "chunks must not be empty");
    TORCH_CHECK(rankStride > 0 && shardSize > 0,
                "rank_stride and shard_size must be positive");

    const int64_t taskCount = rowCount * chunks.size(0);
    const uint32_t usedCores = static_cast<uint32_t>(std::min<int64_t>(40, taskCount));
    ucm_compact_scatter::Launch(
        c10_npu::getCurrentNPUStream().stream(),
        receiveBuffer.data_ptr<uint8_t>(),
        destinationAddresses.data_ptr<int64_t>(),
        routes.data_ptr<int32_t>(),
        chunks.data_ptr<int64_t>(),
        static_cast<uint32_t>(rowCount),
        static_cast<uint32_t>(chunks.size(0)),
        static_cast<uint32_t>(destinationAddresses.size(1)),
        static_cast<uint64_t>(rankStride),
        static_cast<uint64_t>(shardSize),
        usedCores);
}

}

PYBIND11_MODULE(ucm_segmented_copy, module)
{
    module.def("copy", &Copy);
    module.def("scatter", &Scatter);
    module.def("wait_event", &WaitEvent);
}
