#ifndef UNIFIEDCACHE_ALLGATHER_RUNTIME_H
#define UNIFIEDCACHE_ALLGATHER_RUNTIME_H

#include <condition_variable>
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "platform_runtime.h"
#include "status/status.h"

namespace UC::AllGatherStore {

/**
 * @brief Streams and the single collective domain shared by AllGather stages.
 *
 * The runtime owns exactly one communicator and one collective stream, so every
 * collective this process issues through it is serialized in submission order.
 * Overlap comes from the per-slot streams, which carry metadata copies and
 * scatter kernels beside the collective rather than from extra communicators.
 */
class AllGatherRuntime {
public:
    /* The Status argument carries the runtime's fatal state. When it is a
     * failure the work must abandon its task instead of touching the
     * communicator, which is how queued tasks drain after a poisoning. */
    using Work = std::function<void(PlatformRuntime&, StreamHandle, CollectiveHandle, Status)>;

    static Expected<std::shared_ptr<AllGatherRuntime>> Acquire(
        const std::string& key, int32_t deviceId, uint32_t rank, uint32_t worldSize,
        uint32_t collectiveBufferMb, uint32_t collectiveMode,
        const std::vector<uint8_t>& rootInfo, bool collectiveEnabled, size_t slotStreamCount);

    ~AllGatherRuntime();
    Status Submit(Work work);
    Status SubmitDump(Work work);

    /**
     * @brief Mark the collective domain unusable after a fatal failure.
     *
     * A communicator that failed mid-collective has already diverged from its
     * peers, so every later task on it would deepen the skew. Poisoning rejects
     * new work and lets queued work fail fast instead.
     */
    void Poison(Status reason);
    Status FatalStatus() const;

    /** Stream carrying every collective. Serialized across stages and tasks. */
    StreamHandle CollectiveStream() const { return collectiveStream_; }
    /** Side stream owned by one load slot: metadata copies and scatter kernels. */
    StreamHandle SlotStream(size_t index) const;
    /** Stream used to join slot work and publish task completion. */
    StreamHandle CompletionStream() const { return completionStream_; }
    CollectiveHandle Collective() const { return collective_; }

private:
    AllGatherRuntime(std::string key, int32_t deviceId);
    Status Setup(uint32_t rank, uint32_t worldSize, uint32_t collectiveBufferMb,
                 uint32_t collectiveMode, const std::vector<uint8_t>& rootInfo,
                 bool collectiveEnabled, size_t slotStreamCount);
    Status WarmupCollective(uint32_t worldSize);
    Status Submit(Work work, bool dump);
    void Loop(bool dump);

    std::string key_;
    int32_t deviceId_;
    uint32_t rank_{0};
    uint32_t worldSize_{0};
    uint32_t collectiveBufferMb_{0};
    uint32_t collectiveMode_{0};
    bool collectiveEnabled_{false};
    std::vector<uint8_t> rootInfo_;
    std::shared_ptr<PlatformRuntime> platform_;
    StreamHandle collectiveStream_{nullptr};
    StreamHandle completionStream_{nullptr};
    StreamHandle dumpStream_{nullptr};
    std::vector<StreamHandle> slotStreams_;
    CollectiveHandle collective_{nullptr};
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<Work> queue_;
    std::deque<Work> dumpQueue_;
    bool stopping_{false};
    Status fatal_{Status::OK()};
    std::thread thread_;
    std::thread dumpThread_;
};

}  // namespace UC::AllGatherStore

#endif
