#ifndef UNIFIEDCACHE_ALLGATHER_MEMORY_PLAN_H
#define UNIFIEDCACHE_ALLGATHER_MEMORY_PLAN_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace UC::AllGatherStore {

constexpr size_t kDefaultWindowBlocks = 4;
constexpr size_t kDefaultReceiveSlotCount = 2;
constexpr size_t kCopyChunkBytes = 32 * 1024;
constexpr size_t kMaxCopyWorkers = 40;
constexpr size_t kCollectiveBufferCopies = 2;
constexpr size_t kFrameHeaderBytes = 32;
constexpr size_t kFrameRecordBytes = 16;

constexpr size_t AlignFrameBytes(size_t bytes) { return (bytes + 31) / 32 * 32; }

constexpr size_t CalculateFrameMetadataBytes(size_t windowBlocks)
{
    return AlignFrameBytes(kFrameHeaderBytes + windowBlocks * kFrameRecordBytes);
}

struct StageMemoryPlan {
    size_t shardSize{};
    size_t tensorCount{};
    size_t chunkCount{};
    size_t worldSize{};
    size_t windowBlocks{};
    bool replicated{};
    size_t loadSlots{};
    size_t receiveSlots{};
    size_t dumpSlots{};
    size_t loadSendBytes{};
    size_t loadReceiveBytes{};
    size_t dumpSendBytes{};
    size_t chunkLayoutBytes{};
    size_t dumpDescriptorBytes{};
    size_t dumpOffsetBytes{};
    size_t loadDestinationBytes{};
    size_t loadRouteBytes{};

    size_t PayloadBytes() const;
    size_t MetadataBytes() const;
    size_t TotalBytes() const;
};

StageMemoryPlan CalculateStageMemoryPlan(const std::vector<size_t>& tensorSizes, size_t shardSize,
                                         size_t worldSize, bool replicated, size_t loadSlots,
                                         size_t dumpSlots,
                                         size_t windowBlocks = kDefaultWindowBlocks,
                                         size_t receiveSlots = kDefaultReceiveSlotCount,
                                         bool remoteScatter = false);
size_t CalculateCollectiveBytes(uint32_t bufferMb, bool collectiveEnabled,
                                size_t collectiveGroupCount = 1);

}  // namespace UC::AllGatherStore

#endif
