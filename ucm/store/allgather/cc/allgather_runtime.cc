#include "allgather_runtime.h"

#include <unordered_map>
#include <utility>

namespace UC::AllGatherStore {
namespace {

std::mutex gRegistryMutex;
std::unordered_map<std::string, std::weak_ptr<AllGatherRuntime>> gRegistry;

}  // namespace

AllGatherRuntime::AllGatherRuntime(std::string key, int32_t deviceId)
    : key_(std::move(key)), deviceId_(deviceId)
{
}

Expected<std::shared_ptr<AllGatherRuntime>> AllGatherRuntime::Acquire(
    const std::string& key, int32_t deviceId, uint32_t rank, uint32_t worldSize,
    uint32_t collectiveBufferMb, const std::vector<uint8_t>& rootInfo,
    bool collectiveEnabled, size_t loadGroupCount)
{
    std::lock_guard<std::mutex> lock(gRegistryMutex);
    const auto found = gRegistry.find(key);
    if (found != gRegistry.end()) {
        if (auto runtime = found->second.lock()) {
            if (runtime->loadStreams_.size() < loadGroupCount) {
                return Status::InvalidParam(
                    "allgather runtime has fewer load groups than requested({}/{})",
                    loadGroupCount, runtime->loadStreams_.size());
            }
            return std::move(runtime);
        }
    }
    auto runtime = std::shared_ptr<AllGatherRuntime>(new AllGatherRuntime(key, deviceId));
    auto status = runtime->Setup(rank, worldSize, collectiveBufferMb, rootInfo,
                                 collectiveEnabled, loadGroupCount);
    if (status.Failure()) { return status; }
    gRegistry[key] = runtime;
    return std::move(runtime);
}

Status AllGatherRuntime::Setup(uint32_t rank, uint32_t worldSize,
                               uint32_t collectiveBufferMb,
                               const std::vector<uint8_t>& rootInfo,
                               bool collectiveEnabled, size_t loadGroupCount)
{
    if (loadGroupCount == 0) {
        return Status::InvalidParam("allgather load group count must be positive");
    }
    platform_ = CreatePlatformRuntime();
    if (platform_ == nullptr) { return Status::Error("allgather platform runtime unavailable"); }
    auto status = platform_->SetDevice(deviceId_);
    if (status.Failure()) { return status; }
    loadStreams_.reserve(loadGroupCount);
    for (size_t i = 0; i < loadGroupCount; ++i) {
        StreamHandle stream = nullptr;
        status = platform_->CreateStream(&stream);
        if (status.Failure()) { return status; }
        loadStreams_.push_back(stream);
    }
    status = platform_->CreateStream(&dumpStream_);
    if (status.Failure()) { return status; }
    if (collectiveEnabled) {
        if (rootInfo.size() % loadGroupCount != 0) {
            return Status::InvalidParam(
                "invalid allgather root info size({}) for {} groups",
                rootInfo.size(), loadGroupCount);
        }
        const size_t rootInfoSize = rootInfo.size() / loadGroupCount;
        collectives_.reserve(loadGroupCount);
        for (size_t i = 0; i < loadGroupCount; ++i) {
            std::vector<uint8_t> groupRootInfo(
                rootInfo.begin() + i * rootInfoSize,
                rootInfo.begin() + (i + 1) * rootInfoSize);
            CollectiveHandle collective = nullptr;
            status = platform_->CreateCollective(rank, worldSize, collectiveBufferMb,
                                                 groupRootInfo, &collective);
            if (status.Failure()) { return status; }
            collectives_.push_back(collective);
        }
    }
    thread_ = std::thread(&AllGatherRuntime::Loop, this, false);
    dumpThread_ = std::thread(&AllGatherRuntime::Loop, this, true);
    return Status::OK();
}

AllGatherRuntime::~AllGatherRuntime()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    condition_.notify_all();
    if (thread_.joinable()) { thread_.join(); }
    if (dumpThread_.joinable()) { dumpThread_.join(); }
    if (platform_ != nullptr) { (void)platform_->SetDevice(deviceId_); }
    for (const auto collective : collectives_) {
        platform_->DestroyCollective(collective);
    }
    collectives_.clear();
    for (const auto stream : loadStreams_) { platform_->DestroyStream(stream); }
    loadStreams_.clear();
    if (dumpStream_ != nullptr) {
        platform_->DestroyStream(dumpStream_);
        dumpStream_ = nullptr;
    }
}

Status AllGatherRuntime::Submit(Work work)
{
    return Submit(std::move(work), false);
}

Status AllGatherRuntime::SubmitDump(Work work)
{
    return Submit(std::move(work), true);
}

StreamHandle AllGatherRuntime::LoadStream(size_t index) const
{
    return loadStreams_[index % loadStreams_.size()];
}

CollectiveHandle AllGatherRuntime::LoadCollective(size_t index) const
{
    return collectives_.empty() ? nullptr : collectives_[index % collectives_.size()];
}

Status AllGatherRuntime::Submit(Work work, bool dump)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) { return Status::Error("allgather runtime is stopping"); }
        (dump ? dumpQueue_ : queue_).push_back(std::move(work));
    }
    condition_.notify_all();
    return Status::OK();
}

void AllGatherRuntime::Loop(bool dump)
{
    (void)platform_->SetDevice(deviceId_);
    auto& queue = dump ? dumpQueue_ : queue_;
    const auto stream = dump ? dumpStream_ : loadStreams_.front();
    const auto collective = collectives_.empty() ? nullptr : collectives_.front();
    while (true) {
        Work work;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this, &queue] { return stopping_ || !queue.empty(); });
            if (stopping_ && queue.empty()) { return; }
            work = std::move(queue.front());
            queue.pop_front();
        }
        work(*platform_, stream, collective);
    }
}

}  // namespace UC::AllGatherStore
