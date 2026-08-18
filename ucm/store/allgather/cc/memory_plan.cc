#include "memory_plan.h"

#include <numeric>
#include <stdexcept>

namespace UC::AllGatherStore {

size_t StageMemoryPlan::PayloadBytes() const
{
    return loadSendBytes + loadReceiveBytes + dumpSendBytes;
}

size_t StageMemoryPlan::MetadataBytes() const
{
    return chunkLayoutBytes + dumpDescriptorBytes + dumpOffsetBytes + loadDestinationBytes +
           loadRouteBytes;
}

size_t StageMemoryPlan::TotalBytes() const { return PayloadBytes() + MetadataBytes(); }

StageMemoryPlan CalculateStageMemoryPlan(const std::vector<size_t>& tensorSizes,
                                         size_t shardSize, size_t worldSize, bool replicated,
                                         size_t loadSlots, size_t dumpSlots,
                                         size_t windowBlocks)
{
    if (tensorSizes.empty() || shardSize == 0 || worldSize == 0 || loadSlots == 0 ||
        dumpSlots == 0 || windowBlocks == 0) {
        throw std::invalid_argument("invalid allgather memory plan parameter");
    }
    const auto logicalSize =
        std::accumulate(tensorSizes.begin(), tensorSizes.end(), static_cast<size_t>(0));
    if (logicalSize == 0 || logicalSize > shardSize) {
        throw std::invalid_argument("allgather tensor sizes do not fit shard size");
    }
    size_t chunkCount = 0;
    for (const auto size : tensorSizes) {
        chunkCount += (size + kCopyChunkBytes - 1) / kCopyChunkBytes;
    }
    const bool collectiveEnabled = replicated && worldSize > 1;
    const size_t windowPayloadBytes = windowBlocks * shardSize;
    const size_t maxWindowRows = windowBlocks * (collectiveEnabled ? worldSize : 1);

    StageMemoryPlan plan;
    plan.shardSize = shardSize;
    plan.tensorCount = tensorSizes.size();
    plan.chunkCount = chunkCount;
    plan.worldSize = worldSize;
    plan.windowBlocks = windowBlocks;
    plan.replicated = replicated;
    plan.loadSlots = loadSlots;
    plan.dumpSlots = dumpSlots;
    plan.loadSendBytes = loadSlots * windowPayloadBytes;
    plan.loadReceiveBytes =
        collectiveEnabled ? loadSlots * worldSize * windowPayloadBytes : 0;
    plan.dumpSendBytes = dumpSlots * windowPayloadBytes;
    plan.chunkLayoutBytes = chunkCount * 4 * sizeof(uint64_t);
    plan.dumpDescriptorBytes =
        dumpSlots * windowBlocks * chunkCount * 3 * sizeof(uint64_t);
    plan.dumpOffsetBytes = dumpSlots * (kMaxCopyWorkers + 1) * sizeof(uint32_t);
    plan.loadDestinationBytes =
        loadSlots * maxWindowRows * tensorSizes.size() * sizeof(uint64_t);
    plan.loadRouteBytes = loadSlots * maxWindowRows * 2 * sizeof(uint32_t);
    return plan;
}

size_t CalculateCollectiveBytes(uint32_t bufferMb, bool collectiveEnabled,
                                size_t collectiveGroupCount)
{
    if (!collectiveEnabled) { return 0; }
    if (bufferMb == 0 || collectiveGroupCount == 0) {
        throw std::invalid_argument("collective buffer parameters must be positive");
    }
    return static_cast<size_t>(bufferMb) * 1024 * 1024 * kCollectiveBufferCopies *
           collectiveGroupCount;
}

}  // namespace UC::AllGatherStore
