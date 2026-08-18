#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "memory_plan.h"
#include "platform_runtime.h"

namespace py = pybind11;

namespace {

std::vector<uint8_t> CreateRootInfo()
{
    auto platform = UC::AllGatherStore::CreatePlatformRuntime();
    auto result = platform->CreateRootInfo();
    if (!result) { throw std::runtime_error(result.Error().ToString()); }
    return std::move(result.Value());
}

py::dict CalculateMemoryPlan(const std::vector<size_t>& tensorSizes, size_t shardSize,
                             size_t worldSize, bool replicated, size_t loadSlots,
                             size_t dumpSlots, size_t windowBlocks)
{
    const auto plan = UC::AllGatherStore::CalculateStageMemoryPlan(
        tensorSizes, shardSize, worldSize, replicated, loadSlots, dumpSlots,
        windowBlocks);
    py::dict result;
    result["window_blocks"] = plan.windowBlocks;
    result["shard_size"] = plan.shardSize;
    result["tensor_count"] = plan.tensorCount;
    result["chunk_count"] = plan.chunkCount;
    result["load_send_bytes"] = plan.loadSendBytes;
    result["load_receive_bytes"] = plan.loadReceiveBytes;
    result["dump_send_bytes"] = plan.dumpSendBytes;
    result["metadata_bytes"] = plan.MetadataBytes();
    result["total_bytes"] = plan.TotalBytes();
    return result;
}

}  // namespace

PYBIND11_MODULE(ucm_allgather_runtime, module)
{
    module.def("create_root_info", &CreateRootInfo);
    module.def("root_info_size", [] {
        return UC::AllGatherStore::CreatePlatformRuntime()->RootInfoSize();
    });
    module.def("platform_name", [] {
        return UC::AllGatherStore::CreatePlatformRuntime()->Name();
    });
    module.def("calculate_memory_plan", &CalculateMemoryPlan, py::arg("tensor_sizes"),
               py::arg("shard_size"), py::arg("world_size"), py::arg("replicated"),
               py::arg("load_slots") = 2, py::arg("dump_slots") = 2,
               py::arg("window_blocks") = UC::AllGatherStore::kDefaultWindowBlocks);
}
