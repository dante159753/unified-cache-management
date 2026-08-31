#ifndef UNIFIEDCACHE_ALLGATHER_MEMORY_PLAN_H
#define UNIFIEDCACHE_ALLGATHER_MEMORY_PLAN_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace UC::AllGatherStore {

constexpr size_t kDefaultWindowBlocks = 4;
constexpr size_t kCopyChunkBytes = 32 * 1024;
constexpr size_t kMaxCopyWorkers = 40;

struct StageMemoryPlan {
    size_t shardSize{};
    size_t tensorCount{};
    size_t chunkCount{};
    size_t worldSize{};
    size_t windowBlocks{};
    bool replicated{};
    size_t loadSlots{};
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
                                         bool bufferedRemoteScatter = false);

}  // namespace UC::AllGatherStore

#endif
