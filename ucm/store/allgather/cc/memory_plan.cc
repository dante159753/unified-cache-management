#include "memory_plan.h"
#include <algorithm>
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

StageMemoryPlan CalculateStageMemoryPlan(const std::vector<size_t>& tensorSizes, size_t shardSize,
                                         size_t worldSize, bool replicated, size_t loadSlots,
                                         size_t dumpSlots, size_t windowBlocks, size_t receiveSlots,
                                         bool remoteScatter)
{
    if (tensorSizes.empty() || shardSize == 0 || worldSize == 0 || loadSlots == 0 ||
        dumpSlots == 0 || windowBlocks == 0 || receiveSlots == 0) {
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
    const bool distributed = replicated && worldSize > 1;
    const bool collectiveEnabled = distributed && !remoteScatter;
    const size_t windowPayloadBytes = windowBlocks * shardSize;
    const size_t frameBytes =
        windowPayloadBytes + (collectiveEnabled ? CalculateFrameMetadataBytes(windowBlocks) : 0);
    const size_t maxWindowRows = windowBlocks * (distributed ? worldSize : 1);

    StageMemoryPlan plan;
    plan.shardSize = shardSize;
    plan.tensorCount = tensorSizes.size();
    plan.chunkCount = chunkCount;
    plan.worldSize = worldSize;
    plan.windowBlocks = windowBlocks;
    plan.replicated = replicated;
    plan.loadSlots = loadSlots;
    // Receive areas only cover the collective-to-scatter handoff, which one
    // serialized collective stream saturates with a couple of buffers. They do
    // not scale with the send slots that give the picker its choices.
    plan.receiveSlots = std::min(receiveSlots, loadSlots);
    plan.dumpSlots = dumpSlots;
    plan.loadSendBytes = loadSlots * frameBytes;
    plan.loadReceiveBytes = collectiveEnabled ? plan.receiveSlots * worldSize * frameBytes : 0;
    plan.dumpSendBytes = dumpSlots * windowPayloadBytes;
    plan.chunkLayoutBytes = chunkCount * 4 * sizeof(uint64_t);
    plan.dumpDescriptorBytes = dumpSlots * windowBlocks * chunkCount * 3 * sizeof(uint64_t);
    plan.dumpOffsetBytes = dumpSlots * (kMaxCopyWorkers + 1) * sizeof(uint32_t);
    // Framed mode keeps row addresses on the task, not per slot.
    plan.loadDestinationBytes =
        collectiveEnabled ? 0 : loadSlots * maxWindowRows * tensorSizes.size() * sizeof(uint64_t);
    plan.loadRouteBytes = collectiveEnabled
                              ? 0
                              : loadSlots * maxWindowRows * 2 * sizeof(uint32_t) +
                                    (remoteScatter ? loadSlots * worldSize * sizeof(void*) : 0);
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
