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
    using Work = std::function<void(PlatformRuntime&, StreamHandle, CollectiveHandle)>;

    static Expected<std::shared_ptr<AllGatherRuntime>> Acquire(
        const std::string& key, int32_t deviceId, uint32_t rank, uint32_t worldSize,
        uint32_t collectiveBufferMb, uint32_t collectiveMode,
        const std::vector<uint8_t>& rootInfo,
        bool collectiveEnabled, size_t loadGroupCount);

    ~AllGatherRuntime();
    Status Submit(Work work);
    Status SubmitDump(Work work);
    StreamHandle LoadStream(size_t index) const;
    CollectiveHandle LoadCollective(size_t index) const;

private:
    AllGatherRuntime(std::string key, int32_t deviceId);
    Status Setup(uint32_t rank, uint32_t worldSize, uint32_t collectiveBufferMb,
                 uint32_t collectiveMode,
                 const std::vector<uint8_t>& rootInfo, bool collectiveEnabled,
                 size_t loadGroupCount);
    Status Submit(Work work, bool dump);
    void Loop(bool dump);

    std::string key_;
    int32_t deviceId_;
    uint32_t collectiveMode_{0};
    std::shared_ptr<PlatformRuntime> platform_;
    std::vector<StreamHandle> loadStreams_;
    StreamHandle dumpStream_{nullptr};
    std::vector<CollectiveHandle> collectives_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<Work> queue_;
    std::deque<Work> dumpQueue_;
    bool stopping_{false};
    std::thread thread_;
    std::thread dumpThread_;
};

}  // namespace UC::AllGatherStore

#endif
