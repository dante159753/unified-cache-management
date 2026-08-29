#ifndef UNIFIEDCACHE_ALLGATHER_FRAME_PROTOCOL_H
#define UNIFIEDCACHE_ALLGATHER_FRAME_PROTOCOL_H

#include <cstddef>
#include <cstdint>
#include "memory_plan.h"

namespace UC::AllGatherStore {

constexpr uint32_t kFrameMagic = 0x55434631;
constexpr uint32_t kFrameVersion = 1;
constexpr uint32_t kFrameStatusReady = 0;
constexpr uint32_t kFrameStatusBackendFailed = 1;
constexpr uint32_t kAnyFrameRound = UINT32_MAX;

struct FrameHeader {
    uint32_t magic{kFrameMagic};
    uint32_t version{kFrameVersion};
    uint32_t validCount{};
    uint32_t sequenceLow{};
    uint32_t sequenceHigh{};
    uint32_t round{};
    uint32_t reserved0{};
    uint32_t reserved1{};
};

struct FrameRecord {
    uint32_t row{};
    uint32_t status{kFrameStatusReady};
    uint32_t payloadSlot{};
    uint32_t reserved{};
};

static_assert(sizeof(FrameHeader) == kFrameHeaderBytes);
static_assert(sizeof(FrameRecord) == kFrameRecordBytes);

inline size_t FrameBytes(size_t windowBlocks, size_t shardSize)
{
    return CalculateFrameMetadataBytes(windowBlocks) + windowBlocks * shardSize;
}

}  // namespace UC::AllGatherStore

#endif
