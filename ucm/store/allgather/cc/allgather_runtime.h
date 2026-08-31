#ifndef UNIFIEDCACHE_ALLGATHER_RUNTIME_H
#define UNIFIEDCACHE_ALLGATHER_RUNTIME_H

#include <condition_variable>
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

class AllGatherRuntime {
public:
    using Work = std::function<void(PlatformRuntime&, StreamHandle, Status)>;

    static Expected<std::shared_ptr<AllGatherRuntime>> Acquire(const std::string& key,
                                                               int32_t deviceId, uint32_t rank,
                                                               uint32_t worldSize,
                                                               size_t slotStreamCount);

    ~AllGatherRuntime();
    Status Submit(Work work);
    Status SubmitDump(Work work);
    void Poison(Status reason);
    Status FatalStatus() const;

    StreamHandle ProgressStream() const { return progressStream_; }
    StreamHandle SlotStream(size_t index) const;
    StreamHandle CompletionStream() const { return completionStream_; }

private:
    AllGatherRuntime(std::string key, int32_t deviceId);
    Status Setup(uint32_t rank, uint32_t worldSize, size_t slotStreamCount);
    Status Submit(Work work, bool dump);
    void Loop(bool dump);

    std::string key_;
    int32_t deviceId_;
    uint32_t rank_{0};
    uint32_t worldSize_{0};
    std::shared_ptr<PlatformRuntime> platform_;
    StreamHandle progressStream_{nullptr};
    StreamHandle completionStream_{nullptr};
    StreamHandle dumpStream_{nullptr};
    std::vector<StreamHandle> slotStreams_;
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
