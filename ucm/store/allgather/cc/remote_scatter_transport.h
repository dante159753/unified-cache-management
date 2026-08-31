#ifndef UNIFIEDCACHE_ALLGATHER_REMOTE_SCATTER_TRANSPORT_H
#define UNIFIEDCACHE_ALLGATHER_REMOTE_SCATTER_TRANSPORT_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "platform_runtime.h"
#include "status/status.h"

namespace UC::AllGatherStore {

class RemoteScatterTransport {
public:
    struct ReadyRanks {
        uint64_t readyMask{};
        uint64_t failedMask{};
    };

    static Expected<std::unique_ptr<RemoteScatterTransport>> Create(
        std::shared_ptr<PlatformRuntime> platform, const std::vector<uint8_t>& bootstrapKey,
        const std::string& stageSignature, uint32_t rank, uint32_t worldSize,
        const std::vector<void*>& localBuffers, size_t bufferBytes);

    ~RemoteScatterTransport();

    const std::vector<void*>& PeerBuffers(size_t slot) const { return peerBuffers_[slot]; }
    Status PublishReady(size_t slot, uint64_t generation, uint64_t windowTag, bool failed);
    Expected<ReadyRanks> WaitReady(size_t slot, uint64_t generation, uint64_t windowTag,
                                   uint64_t pendingMask);
    void PublishConsumed(size_t slot, uint64_t generation);
    Status WaitConsumed(size_t slot, uint64_t generation);

private:
    RemoteScatterTransport(std::shared_ptr<PlatformRuntime> platform, uint32_t rank,
                           uint32_t worldSize, size_t slotCount);
    Status Setup(const std::vector<uint8_t>& bootstrapKey, const std::string& stageSignature,
                 const std::vector<void*>& localBuffers, size_t bufferBytes);

    std::shared_ptr<PlatformRuntime> platform_;
    uint32_t rank_;
    uint32_t worldSize_;
    size_t slotCount_;
    std::string sharedName_;
    bool sharedLinked_{false};
    void* mapping_{nullptr};
    size_t mappingBytes_{0};
    std::vector<std::vector<void*>> peerBuffers_;
};

}  // namespace UC::AllGatherStore

#endif
