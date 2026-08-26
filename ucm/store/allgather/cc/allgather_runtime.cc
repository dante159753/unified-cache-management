#include "allgather_runtime.h"

#include <unordered_map>
#include <utility>

#include "logger/logger.h"

namespace UC::AllGatherStore {
namespace {

std::mutex gRegistryMutex;
std::unordered_map<std::string, std::weak_ptr<AllGatherRuntime>> gRegistry;

/* Payload of the startup probe collective. Small enough to be free, large
 * enough to force HCCL to negotiate its transport resources. */
constexpr size_t kWarmupBytes = 32;

}  // namespace

AllGatherRuntime::AllGatherRuntime(std::string key, int32_t deviceId)
    : key_(std::move(key)), deviceId_(deviceId)
{
}

Expected<std::shared_ptr<AllGatherRuntime>> AllGatherRuntime::Acquire(
    const std::string& key, int32_t deviceId, uint32_t rank, uint32_t worldSize,
    uint32_t collectiveBufferMb, uint32_t collectiveMode, const std::vector<uint8_t>& rootInfo,
    bool collectiveEnabled, size_t slotStreamCount)
{
    std::lock_guard<std::mutex> lock(gRegistryMutex);
    const auto found = gRegistry.find(key);
    if (found != gRegistry.end()) {
        if (auto runtime = found->second.lock()) {
            if (runtime->deviceId_ != deviceId) {
                return Status::InvalidParam("allgather runtime device mismatch({}/{})", deviceId,
                                            runtime->deviceId_);
            }
            if (runtime->rank_ != rank || runtime->worldSize_ != worldSize) {
                return Status::InvalidParam("allgather runtime rank mismatch({}:{}/{}:{})", rank,
                                            worldSize, runtime->rank_, runtime->worldSize_);
            }
            if (runtime->collectiveEnabled_ != collectiveEnabled) {
                return Status::InvalidParam("allgather runtime collective state mismatch");
            }
            if (runtime->collectiveBufferMb_ != collectiveBufferMb) {
                return Status::InvalidParam("allgather runtime collective buffer mismatch({}/{})",
                                            collectiveBufferMb, runtime->collectiveBufferMb_);
            }
            if (runtime->collectiveMode_ != collectiveMode) {
                return Status::InvalidParam(
                    "allgather runtime collective mode mismatch({}/{})", collectiveMode,
                    runtime->collectiveMode_);
            }
            if (collectiveEnabled && runtime->rootInfo_ != rootInfo) {
                return Status::InvalidParam("allgather runtime root info mismatch");
            }
            if (runtime->slotStreams_.size() < slotStreamCount) {
                return Status::InvalidParam(
                    "allgather runtime has fewer slot streams than requested({}/{})",
                    slotStreamCount, runtime->slotStreams_.size());
            }
            return std::move(runtime);
        }
    }
    auto runtime = std::shared_ptr<AllGatherRuntime>(new AllGatherRuntime(key, deviceId));
    auto status = runtime->Setup(rank, worldSize, collectiveBufferMb, collectiveMode, rootInfo,
                                 collectiveEnabled, slotStreamCount);
    if (status.Failure()) { return status; }
    gRegistry[key] = runtime;
    return std::move(runtime);
}

Status AllGatherRuntime::Setup(uint32_t rank, uint32_t worldSize, uint32_t collectiveBufferMb,
                               uint32_t collectiveMode, const std::vector<uint8_t>& rootInfo,
                               bool collectiveEnabled, size_t slotStreamCount)
{
    if (slotStreamCount == 0) {
        return Status::InvalidParam("allgather slot stream count must be positive");
    }
    rank_ = rank;
    worldSize_ = worldSize;
    collectiveBufferMb_ = collectiveBufferMb;
    collectiveEnabled_ = collectiveEnabled;
    rootInfo_ = rootInfo;
    platform_ = CreatePlatformRuntime();
    collectiveMode_ = collectiveMode;
    if (platform_ == nullptr) { return Status::Error("allgather platform runtime unavailable"); }
    auto status = platform_->SetDevice(deviceId_);
    if (status.Failure()) { return status; }
    status = platform_->CreateStream(&collectiveStream_);
    if (status.Failure()) { return status; }
    status = platform_->CreateStream(&completionStream_);
    if (status.Failure()) { return status; }
    status = platform_->CreateStream(&dumpStream_);
    if (status.Failure()) { return status; }
    slotStreams_.reserve(slotStreamCount);
    for (size_t i = 0; i < slotStreamCount; ++i) {
        StreamHandle stream = nullptr;
        status = platform_->CreateStream(&stream);
        if (status.Failure()) { return status; }
        slotStreams_.push_back(stream);
    }
    if (collectiveEnabled) {
        if (rootInfo.empty()) { return Status::InvalidParam("empty allgather root info"); }
        status = platform_->CreateCollective(rank, worldSize, collectiveBufferMb, collectiveMode_,
                                             rootInfo, &collective_);
        if (status.Failure()) { return status; }
        status = WarmupCollective(worldSize);
        if (status.Failure()) { return status; }
    }
    thread_ = std::thread(&AllGatherRuntime::Loop, this, false);
    dumpThread_ = std::thread(&AllGatherRuntime::Loop, this, true);
    return Status::OK();
}

Status AllGatherRuntime::WarmupCollective(uint32_t worldSize)
{
    /* HCCL negotiates transport resources on the first collective of a
     * communicator. Doing that lazily would put the negotiation on the serving
     * path, next to the model's own collectives. Run one probe here so the cost
     * and any failure land at startup instead. */
    void* send = nullptr;
    void* receive = nullptr;
    auto status = platform_->AllocateDevice(&send, kWarmupBytes, true);
    if (status.Failure()) { return status; }
    status = platform_->AllocateDevice(&receive, kWarmupBytes * worldSize, true);
    if (status.Success()) {
        status = platform_->AllGather(send, receive, kWarmupBytes, collective_, collectiveStream_);
    }
    if (status.Success()) { status = platform_->SynchronizeStream(collectiveStream_); }
    platform_->FreeDevice(send);
    platform_->FreeDevice(receive);
    if (status.Failure()) {
        return Status::Error(
            fmt::format("allgather communicator warmup failed on runtime {}: {}", key_, status));
    }
    UC_INFO("AllGather runtime {} warmed up its communicator.", key_);
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
    if (collective_ != nullptr) {
        platform_->DestroyCollective(collective_);
        collective_ = nullptr;
    }
    for (const auto stream : slotStreams_) { platform_->DestroyStream(stream); }
    slotStreams_.clear();
    for (auto* stream : {&collectiveStream_, &completionStream_, &dumpStream_}) {
        if (*stream != nullptr) {
            platform_->DestroyStream(*stream);
            *stream = nullptr;
        }
    }
}

Status AllGatherRuntime::Submit(Work work) { return Submit(std::move(work), false); }

Status AllGatherRuntime::SubmitDump(Work work) { return Submit(std::move(work), true); }

StreamHandle AllGatherRuntime::SlotStream(size_t index) const
{
    return slotStreams_[index % slotStreams_.size()];
}

Status AllGatherRuntime::Submit(Work work, bool dump)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) { return Status::Error("allgather runtime is stopping"); }
        if (fatal_.Failure()) { return fatal_; }
        (dump ? dumpQueue_ : queue_).push_back(std::move(work));
    }
    condition_.notify_all();
    return Status::OK();
}

void AllGatherRuntime::Poison(Status reason)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (fatal_.Failure()) { return; }
        fatal_ = std::move(reason);
    }
    UC_ERROR("AllGather runtime {} is poisoned: {}. Queued tasks will fail and the "
             "communicator will not be used again.",
             key_, FatalStatus());
    condition_.notify_all();
}

Status AllGatherRuntime::FatalStatus() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return fatal_;
}

void AllGatherRuntime::Loop(bool dump)
{
    (void)platform_->SetDevice(deviceId_);
    auto& queue = dump ? dumpQueue_ : queue_;
    const auto stream = dump ? dumpStream_ : collectiveStream_;
    while (true) {
        Work work;
        Status fatal = Status::OK();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this, &queue] { return stopping_ || !queue.empty(); });
            if (stopping_ && queue.empty()) { return; }
            work = std::move(queue.front());
            queue.pop_front();
            fatal = fatal_;
        }
        work(*platform_, stream, collective_, fatal);
    }
}

}  // namespace UC::AllGatherStore
