#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include "allgather_runtime.h"
#include "logger/logger.h"
#include "memory_plan.h"
#include "metrics_api.h"
#include "platform_runtime.h"
#include "remote_scatter_transport.h"
#include "ucmstore_v1.h"

namespace UC::AllGatherStore {
namespace {

constexpr size_t kDefaultLoadSlots = 2;
constexpr size_t kDefaultDumpSlots = 2;
using Clock = std::chrono::steady_clock;

enum class RemoteScatterMode { Kernel, BatchedRemoteScatter, BatchCopy, CopyThenScatter };

bool UsesRemoteScatterKernel(RemoteScatterMode mode)
{
    return mode == RemoteScatterMode::Kernel || mode == RemoteScatterMode::BatchedRemoteScatter;
}

const char* RemoteScatterModeName(RemoteScatterMode mode)
{
    switch (mode) {
        case RemoteScatterMode::Kernel: return "kernel";
        case RemoteScatterMode::BatchedRemoteScatter: return "batched_remote_scatter";
        case RemoteScatterMode::BatchCopy: return "batch_copy";
        case RemoteScatterMode::CopyThenScatter: return "copy_then_scatter";
    }
    return "unknown";
}

double Milliseconds(Clock::duration duration)
{
    return std::chrono::duration<double, std::milli>(duration).count();
}

int64_t Microseconds(Clock::time_point point)
{
    if (point == Clock::time_point{}) { return 0; }
    return std::chrono::duration_cast<std::chrono::microseconds>(point.time_since_epoch()).count();
}

struct DeviceBuffer {
    std::shared_ptr<PlatformRuntime> platform;
    void* data{nullptr};
    size_t size{0};

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : platform(std::move(other.platform)), data(other.data), size(other.size)
    {
        other.data = nullptr;
        other.size = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this == &other) { return *this; }
        Reset();
        platform = std::move(other.platform);
        data = other.data;
        size = other.size;
        other.data = nullptr;
        other.size = 0;
        return *this;
    }
    ~DeviceBuffer() { Reset(); }

    Status Allocate(std::shared_ptr<PlatformRuntime> value, size_t bytes, bool zero = false)
    {
        if (bytes == 0) { return Status::OK(); }
        platform = std::move(value);
        auto status = platform->AllocateDevice(&data, bytes, zero);
        if (status.Success()) { size = bytes; }
        return status;
    }

    void Reset()
    {
        if (data != nullptr && platform != nullptr) { platform->FreeDevice(data); }
        data = nullptr;
        size = 0;
    }
};

template <typename T>
struct HostBuffer {
    std::shared_ptr<PlatformRuntime> platform;
    T* data{nullptr};
    size_t count{0};

    HostBuffer() = default;
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    ~HostBuffer()
    {
        if (data != nullptr && platform != nullptr) { platform->FreeHost(data); }
    }
    Status Allocate(std::shared_ptr<PlatformRuntime> value, size_t elements)
    {
        platform = std::move(value);
        auto status = platform->AllocateHost(reinterpret_cast<void**>(&data), elements * sizeof(T));
        if (status.Success()) { count = elements; }
        return status;
    }
};

/**
 * @brief One staged window on its way from the backend into remote scatter.
 *
 * Send slots size the reordering freedom: the more of them there are, the more
 * candidate windows the progress thread can choose from when networked storage
 * completes reads out of order. They are cheap - one window of payload each.
 */
struct LoadSlot {
    std::shared_ptr<PlatformRuntime> platform;
    DeviceBuffer send;
    DeviceBuffer receive;
    /* Local-coalesced mode only: the scatter reads the send buffer directly. */
    DeviceBuffer destinations;
    DeviceBuffer routes;
    HostBuffer<uint64_t> hostDestinations;
    HostBuffer<uint32_t> hostRoutes;
    /* The send buffer has been consumed, so the backend may refill it. */
    EventHandle sendFree{nullptr};
    DeviceBuffer peerBuffers;
    uint64_t generation{0};
    bool inFlight{false};
    bool consumedPublished{false};

    ~LoadSlot()
    {
        if (sendFree != nullptr && platform != nullptr) { platform->DestroyEvent(sendFree); }
    }
};

struct DumpSlot {
    std::shared_ptr<PlatformRuntime> platform;
    DeviceBuffer send;
    DeviceBuffer descriptors;
    DeviceBuffer offsets;
    HostBuffer<uint64_t> hostDescriptors;
    HostBuffer<uint32_t> hostOffsets;
    EventHandle ready{nullptr};

    ~DumpSlot()
    {
        if (ready != nullptr && platform != nullptr) { platform->DestroyEvent(ready); }
    }
};

struct WindowPlan {
    std::vector<size_t> rows;
    std::vector<uint32_t> owners;
    std::vector<uint32_t> ownerSlots;
    std::vector<size_t> ownerCounts;
    std::vector<size_t> ownerOffsets;
    size_t rankStrideBlocks{};
};

struct PendingBackend {
    size_t window{};
    size_t slot{};
    Detail::TaskHandle handle{};
    bool submitted{false};
    Status status{Status::OK()};
    size_t ownedRows{};
    Clock::time_point slotReadyAt;
    Clock::time_point backendSubmitStartedAt;
    Clock::time_point backendSubmittedAt;
};

struct LoadMetrics {
    double queueWaitMs{};
    double prefetchMs{};
    double metadataMs{};
    double slotWaitMs{};
    double backendSubmitMs{};
    double backendWaitMs{};
    double remoteReadySubmitMs{};
    double scatterSubmitMs{};
    double remoteReadyDeviceMs{};
    double scatterDeviceMs{};
    double syncMs{};
    double totalMs{};
    size_t windows{};
};

struct DumpMetrics {
    double queueWaitMs{};
    double backendSubmitMs{};
    double backendWaitMs{};
    double syncMs{};
    double totalMs{};
    size_t windows{};
};

struct StageTraceRecord {
    size_t window{};
    size_t slot{};
    size_t rows{};
    size_t ownedRows{};
    uint64_t remoteScatterBytes{};
    Clock::time_point slotReadyAt;
    Clock::time_point backendSubmitStartedAt;
    Clock::time_point backendSubmittedAt;
    Clock::time_point backendWaitStartedAt;
    Clock::time_point backendReadyAt;
    Clock::time_point remoteReadyStartedAt;
    Clock::time_point remoteReadySubmittedAt;
};

struct TimingWindow {
    PlatformRuntime* platform{nullptr};
    EventHandle start{nullptr};
    EventHandle remoteReady{nullptr};
    EventHandle scatterDone{nullptr};
    bool complete{false};

    TimingWindow() = default;
    TimingWindow(const TimingWindow&) = delete;
    TimingWindow& operator=(const TimingWindow&) = delete;
    TimingWindow(TimingWindow&& other) noexcept
        : platform(other.platform),
          start(other.start),
          remoteReady(other.remoteReady),
          scatterDone(other.scatterDone),
          complete(other.complete)
    {
        other.platform = nullptr;
        other.start = nullptr;
        other.remoteReady = nullptr;
        other.scatterDone = nullptr;
        other.complete = false;
    }
    ~TimingWindow()
    {
        if (platform == nullptr) { return; }
        platform->DestroyEvent(start);
        platform->DestroyEvent(remoteReady);
        platform->DestroyEvent(scatterDone);
    }

    bool Setup(PlatformRuntime& value)
    {
        platform = &value;
        if (platform->CreateEvent(&start, true).Failure() ||
            platform->CreateEvent(&remoteReady, true).Failure() ||
            platform->CreateEvent(&scatterDone, true).Failure()) {
            return false;
        }
        return true;
    }

    bool Valid() const
    {
        return complete && start != nullptr && remoteReady != nullptr && scatterDone != nullptr;
    }
};

struct TaskState {
    enum class Operation { Load, Dump };
    explicit TaskState(Operation operation, Detail::TaskDesc input, uint64_t sequence, bool profile,
                       bool stageTrace, std::shared_ptr<PlatformRuntime> platform)
        : operation(operation),
          input(std::move(input)),
          queuedAt(Clock::now()),
          profile(profile),
          stageTrace(stageTrace),
          sequence(sequence),
          platform(std::move(platform))
    {
    }
    Operation operation;
    Detail::TaskDesc input;
    std::vector<WindowPlan> windows;
    std::mutex mutex;
    std::condition_variable condition;
    Status status{Status::OK()};
    bool done{false};
    Clock::time_point queuedAt;
    Clock::time_point enqueuedAt;
    bool profile{false};
    bool stageTrace{false};
    uint64_t sequence{};
    std::shared_ptr<PlatformRuntime> platform;
};

/** Communication failures are process-level, not request-level. */
bool IsFatalCommunication(const Status& status) { return status == Status::StoreUnhealthy(); }

uint32_t Owner(const Detail::BlockId& id, uint32_t worldSize)
{
    uint64_t value = 0;
    std::memcpy(&value, id.data(), std::min(sizeof(value), id.size()));
    return static_cast<uint32_t>(value % worldSize);
}

uint64_t WindowTag(const TaskState& task, const WindowPlan& window)
{
    uint64_t hash = 1469598103934665603ULL;
    const auto update = [&hash](const void* data, size_t bytes) {
        const auto* input = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < bytes; ++i) {
            hash ^= input[i];
            hash *= 1099511628211ULL;
        }
    };
    for (const auto row : window.rows) {
        const auto& shard = task.input[row];
        update(shard.owner.data(), shard.owner.size());
        update(&shard.index, sizeof(shard.index));
    }
    return hash;
}

}  // namespace

class AllGatherStore final : public StoreV1 {
public:
    ~AllGatherStore() override
    {
        if (platform_ != nullptr) { (void)platform_->SetDevice(deviceId_); }
        std::vector<std::shared_ptr<TaskState>> tasks;
        {
            std::lock_guard<std::mutex> lock(tasksMutex_);
            stopping_ = true;
            for (const auto& item : tasks_) { tasks.push_back(item.second); }
        }
        for (const auto& task : tasks) {
            std::unique_lock<std::mutex> lock(task->mutex);
            task->condition.wait(lock, [&task] { return task->done; });
        }
        runtime_.reset();
    }

    Status Setup(const Detail::Dictionary& config) override
    {
        config.Get("store_backend", backend_);
        config.GetNumber("device_id", deviceId_);
        config.GetNumber("shard_size", shardSize_);
        config.GetNumber("block_size", blockSize_);
        config.GetNumbers("tensor_size_list", tensorSizes_);
        config.GetNumber("allgather_rank", rank_);
        config.GetNumber("allgather_world_size", worldSize_);
        config.Get("allgather_replicated_data", replicated_);
        config.Get("allgather_scatter_only", scatterOnly_);
        config.GetNumber("allgather_load_slots", loadSlotCount_);
        config.GetNumber("allgather_dump_slots", dumpSlotCount_);
        config.GetNumber("allgather_runtime_slot_streams", runtimeSlotStreamCount_);
        config.GetNumber("allgather_scatter_aiv_cores", scatterAivCores_);
        config.GetNumber("allgather_profile_sample_every", profileSampleEvery_);
        config.GetNumber("allgather_stage_trace_sample_every", stageTraceSampleEvery_);
        config.Get("allgather_separate_dump_queue", separateDumpQueue_);
        config.Get("allgather_load_skip_remote_scatter", skipRemoteScatter_);
        config.Get("allgather_load_skip_scatter", skipLoadScatter_);
        config.Get("allgather_load_backend_only", loadBackendOnly_);
        std::string remoteScatterMode = "batch_copy";
        config.Get("allgather_remote_scatter_mode", remoteScatterMode);
        config.Get("allgather_runtime_key", runtimeKey_);
        config.GetNumbers("allgather_remote_scatter_key", bootstrapKey_);

        if (backend_ == nullptr) { return Status::InvalidParam("invalid store backend"); }
        if (deviceId_ < 0) { return Status::OK(); }
        if (worldSize_ == 0 || rank_ >= worldSize_) {
            return Status::InvalidParam("invalid allgather rank({}/{})", rank_, worldSize_);
        }
        if (runtimeKey_.empty()) { return Status::InvalidParam("empty allgather runtime key"); }
        if (config.Contains("allgather_fused_buffer_capacity_mb")) {
            return Status::InvalidParam("allgather_fused_buffer_capacity_mb is not supported");
        }
        config.GetNumber("allgather_window_blocks_per_rank", windowBlocks_);
        if (windowBlocks_ == 0) {
            return Status::InvalidParam("allgather window must be positive");
        }
        platform_ = CreatePlatformRuntime();
        if (platform_ == nullptr) {
            return Status::Error("allgather platform runtime unavailable");
        }
        auto status = platform_->SetDevice(deviceId_);
        if (status.Failure()) { return status; }
        distributedLoadEnabled_ = replicated_ && !scatterOnly_ && worldSize_ > 1;
        if (remoteScatterMode == "kernel") {
            remoteScatterMode_ = RemoteScatterMode::Kernel;
        } else if (remoteScatterMode == "batched_remote_scatter") {
            remoteScatterMode_ = RemoteScatterMode::BatchedRemoteScatter;
        } else if (remoteScatterMode == "batch_copy") {
            remoteScatterMode_ = RemoteScatterMode::BatchCopy;
        } else if (remoteScatterMode == "copy_then_scatter") {
            remoteScatterMode_ = RemoteScatterMode::CopyThenScatter;
        } else {
            return Status::InvalidParam("unsupported remote scatter mode({})", remoteScatterMode);
        }
        if (distributedLoadEnabled_ && !platform_->SupportsRemoteScatter()) {
            return Status::Error(
                fmt::format("remote scatter is unavailable on {}", platform_->Name()));
        }
        if (distributedLoadEnabled_ && bootstrapKey_.empty()) {
            return Status::InvalidParam("empty remote scatter bootstrap key");
        }
        if (loadSlotCount_ == 0) {
            return Status::InvalidParam("allgather load slot count must be positive");
        }
        if (scatterAivCores_ == 0 || scatterAivCores_ > kMaxCopyWorkers) {
            return Status::InvalidParam("allgather scatter AIV cores must be in [1, {}]",
                                        kMaxCopyWorkers);
        }
        const size_t requiredSlotStreams = loadSlotCount_;
        if (runtimeSlotStreamCount_ == 0) { runtimeSlotStreamCount_ = requiredSlotStreams; }
        if (runtimeSlotStreamCount_ < requiredSlotStreams) {
            return Status::InvalidParam(
                "allgather runtime slot streams({}) are fewer than required streams({})",
                runtimeSlotStreamCount_, requiredSlotStreams);
        }
        loadStorageWorldSize_ = distributedLoadEnabled_ ? worldSize_ : 1;
        loadStorageRank_ = distributedLoadEnabled_ ? rank_ : 0;
        dumpStorageWorldSize_ = replicated_ ? worldSize_ : 1;
        dumpStorageRank_ = replicated_ ? rank_ : 0;
        try {
            plan_ = CalculateStageMemoryPlan(
                tensorSizes_, shardSize_, worldSize_, distributedLoadEnabled_, loadSlotCount_,
                dumpSlotCount_, windowBlocks_,
                remoteScatterMode_ == RemoteScatterMode::CopyThenScatter);
        } catch (const std::exception& error) {
            return Status::InvalidParam("invalid allgather memory plan: {}", error.what());
        }

        status = AllocateBuffers();
        if (status.Failure()) { return status; }
        if (distributedLoadEnabled_) {
            status = SetupRemoteScatter();
            if (status.Failure()) { return status; }
        }
        auto runtime = AllGatherRuntime::Acquire(runtimeKey_, deviceId_, rank_, worldSize_,
                                                 runtimeSlotStreamCount_);
        if (!runtime) { return runtime.Error(); }
        runtime_ = runtime.Value();
        UC_INFO(
            "AllGatherStore: shard={}, world={}, window_blocks={}, load_slots={}, "
            "runtime_slot_streams={}, dump_slots={}, payload_bytes={}, metadata_bytes={}, "
            "scatter_only={}, skip_remote_scatter={}, skip_load_scatter={}, load_backend_only={}, "
            "remote_scatter={}, remote_scatter_mode={}, scatter_aiv_cores={}, "
            "platform={}.",
            shardSize_, worldSize_, windowBlocks_, loadSlotCount_, runtimeSlotStreamCount_,
            dumpSlotCount_, plan_.PayloadBytes(), plan_.MetadataBytes(), scatterOnly_,
            skipRemoteScatter_, skipLoadScatter_, loadBackendOnly_, distributedLoadEnabled_,
            RemoteScatterModeName(remoteScatterMode_), scatterAivCores_, platform_->Name());
        return Status::OK();
    }

    std::string Readme() const override { return "AllGatherStore"; }

    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num) override
    {
        return backend_->Lookup(blocks, num);
    }

    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) override
    {
        return backend_->LookupOnPrefix(blocks, num);
    }

    Expected<ssize_t> LookupOnReverse(const Detail::BlockId* blocks, size_t num) override
    {
        return backend_->LookupOnReverse(blocks, num);
    }

    void Prefetch(const Detail::BlockId* blocks, size_t num) override
    {
        if (deviceId_ < 0 || !distributedLoadEnabled_) {
            backend_->Prefetch(blocks, num);
            return;
        }
        std::vector<Detail::BlockId> owned;
        for (size_t i = 0; i < num; ++i) {
            if (Owner(blocks[i], loadStorageWorldSize_) == loadStorageRank_) {
                owned.push_back(blocks[i]);
            }
        }
        if (!owned.empty()) { backend_->Prefetch(owned.data(), owned.size()); }
    }

    void Prefetch(const Detail::Shard* shards, size_t num) override
    {
        if (deviceId_ < 0 || !distributedLoadEnabled_) {
            backend_->Prefetch(shards, num);
            return;
        }
        std::vector<Detail::Shard> owned;
        owned.reserve(num);
        for (size_t i = 0; i < num; ++i) {
            if (Owner(shards[i].owner, loadStorageWorldSize_) == loadStorageRank_) {
                owned.push_back({shards[i].owner, shards[i].index, {}});
            }
        }
        if (!owned.empty()) { backend_->Prefetch(owned.data(), owned.size()); }
    }

    Status CheckHealth() override { return backend_->CheckHealth(); }

    Expected<Detail::TaskHandle> Load(Detail::TaskDesc input) override
    {
        if (deviceId_ < 0) { return Status::Error("allgather load is unavailable on scheduler"); }
        return Submit(TaskState::Operation::Load, std::move(input));
    }

    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc input) override
    {
        if (deviceId_ < 0) { return Status::Error("allgather dump is unavailable on scheduler"); }
        return Submit(TaskState::Operation::Dump, std::move(input));
    }

    Expected<bool> Check(Detail::TaskHandle handle) override
    {
        if (deviceId_ < 0) { return Status::Error("allgather check is unavailable on scheduler"); }
        auto task = FindTask(handle);
        if (!task) { return Status::InvalidParam("invalid allgather task({})", handle); }
        std::lock_guard<std::mutex> lock(task->mutex);
        return bool(task->done);
    }

    Status Wait(Detail::TaskHandle handle) override
    {
        if (deviceId_ < 0) { return Status::Error("allgather wait is unavailable on scheduler"); }
        auto task = FindTask(handle);
        if (!task) { return Status::InvalidParam("invalid allgather task({})", handle); }
        std::unique_lock<std::mutex> lock(task->mutex);
        task->condition.wait(lock, [&task] { return task->done; });
        auto status = task->status;
        lock.unlock();
        EraseTask(handle, task);
        return status;
    }

private:
    Status AllocateBuffers()
    {
        const size_t windowBytes = windowBlocks_ * shardSize_;
        const size_t maxRows = windowBlocks_ * (distributedLoadEnabled_ ? worldSize_ : 1);
        loadSlots_.reserve(loadSlotCount_);
        for (size_t i = 0; i < loadSlotCount_; ++i) {
            auto slot = std::make_unique<LoadSlot>();
            slot->platform = platform_;
            auto status = slot->send.Allocate(platform_, windowBytes);
            if (status.Failure()) { return status; }
            if (distributedLoadEnabled_ &&
                remoteScatterMode_ == RemoteScatterMode::CopyThenScatter) {
                status = slot->receive.Allocate(platform_, worldSize_ * windowBytes);
                if (status.Failure()) { return status; }
            }
            status = slot->destinations.Allocate(platform_,
                                                 maxRows * tensorSizes_.size() * sizeof(uint64_t));
            if (status.Failure()) { return status; }
            status = slot->routes.Allocate(platform_, maxRows * 2 * sizeof(uint32_t));
            if (status.Failure()) { return status; }
            status = slot->hostDestinations.Allocate(platform_, maxRows * tensorSizes_.size());
            if (status.Failure()) { return status; }
            status = slot->hostRoutes.Allocate(platform_, maxRows * 2);
            if (status.Failure()) { return status; }
            status = platform_->CreateEvent(&slot->sendFree);
            if (status.Failure()) { return status; }
            if (distributedLoadEnabled_ && UsesRemoteScatterKernel(remoteScatterMode_)) {
                status = slot->peerBuffers.Allocate(platform_, worldSize_ * sizeof(void*));
                if (status.Failure()) { return status; }
            }
            loadSlots_.push_back(std::move(slot));
        }
        dumpSlots_.reserve(dumpSlotCount_);
        for (size_t i = 0; i < dumpSlotCount_; ++i) {
            auto slot = std::make_unique<DumpSlot>();
            slot->platform = platform_;
            auto status = slot->send.Allocate(platform_, windowBytes, true);
            if (status.Failure()) { return status; }
            status = slot->descriptors.Allocate(
                platform_, windowBlocks_ * plan_.chunkCount * 3 * sizeof(uint64_t));
            if (status.Failure()) { return status; }
            status = slot->offsets.Allocate(platform_, (kMaxCopyWorkers + 1) * sizeof(uint32_t));
            if (status.Failure()) { return status; }
            status =
                slot->hostDescriptors.Allocate(platform_, windowBlocks_ * plan_.chunkCount * 3);
            if (status.Failure()) { return status; }
            status = slot->hostOffsets.Allocate(platform_, kMaxCopyWorkers + 1);
            if (status.Failure()) { return status; }
            status = platform_->CreateEvent(&slot->ready);
            if (status.Failure()) { return status; }
            dumpSlots_.push_back(std::move(slot));
        }
        auto status = chunkLayout_.Allocate(platform_, plan_.chunkLayoutBytes);
        if (status.Failure()) { return status; }
        std::vector<uint64_t> chunks;
        size_t sourceOffset = 0;
        for (size_t tensor = 0; tensor < tensorSizes_.size(); ++tensor) {
            tensorOffsets_.push_back(sourceOffset);
            for (size_t offset = 0; offset < tensorSizes_[tensor]; offset += kCopyChunkBytes) {
                chunks.push_back(tensor);
                chunks.push_back(sourceOffset + offset);
                chunks.push_back(offset);
                chunks.push_back(std::min(kCopyChunkBytes, tensorSizes_[tensor] - offset));
            }
            sourceOffset += tensorSizes_[tensor];
        }
        return platform_->CopyHostToDevice(chunkLayout_.data, chunkLayout_.size, chunks.data(),
                                           chunks.size() * sizeof(uint64_t), nullptr);
    }

    Status SetupRemoteScatter()
    {
        std::vector<void*> buffers;
        buffers.reserve(loadSlots_.size());
        for (const auto& slot : loadSlots_) { buffers.push_back(slot->send.data); }
        std::ostringstream signature;
        signature << "shard=" << shardSize_ << ":window=" << windowBlocks_
                  << ":slots=" << loadSlotCount_ << ":tensors=";
        for (const auto size : tensorSizes_) { signature << size << ','; }
        signature << ":mode=" << RemoteScatterModeName(remoteScatterMode_);
        auto transport =
            RemoteScatterTransport::Create(platform_, bootstrapKey_, signature.str(), rank_,
                                           worldSize_, buffers, windowBlocks_ * shardSize_);
        if (!transport) { return transport.Error(); }
        remoteTransport_ = std::move(transport.Value());
        if (UsesRemoteScatterKernel(remoteScatterMode_)) {
            for (size_t slot = 0; slot < loadSlots_.size(); ++slot) {
                const auto& peers = remoteTransport_->PeerBuffers(slot);
                auto status = platform_->CopyHostToDevice(
                    loadSlots_[slot]->peerBuffers.data, loadSlots_[slot]->peerBuffers.size,
                    peers.data(), peers.size() * sizeof(void*), nullptr);
                if (status.Failure()) { return status; }
            }
        }
        return Status::OK();
    }

    Expected<Detail::TaskHandle> Submit(TaskState::Operation operation, Detail::TaskDesc input)
    {
        if (input.empty()) { return Status::InvalidParam("empty allgather task"); }
        for (const auto& shard : input) {
            if (shard.addrs.size() != tensorSizes_.size()) {
                return Status::InvalidParam("invalid allgather address count({}/{})",
                                            shard.addrs.size(), tensorSizes_.size());
            }
        }
        const auto sampleSequence = taskSequence_.fetch_add(1, std::memory_order_relaxed);
        const auto sequence = operation == TaskState::Operation::Load
                                  ? loadTaskSequence_.fetch_add(1, std::memory_order_relaxed)
                                  : 0;
        const bool profile = profileSampleEvery_ > 0 && sampleSequence % profileSampleEvery_ == 0;
        const bool stageTrace = operation == TaskState::Operation::Load &&
                                stageTraceSampleEvery_ > 0 &&
                                sequence % stageTraceSampleEvery_ == 0;
        auto task = std::make_shared<TaskState>(operation, std::move(input), sequence, profile,
                                                stageTrace, platform_);
        task->windows = BuildWindows(task->input, operation);
        const auto handle = nextHandle_.fetch_add(1);
        size_t queueDepth = 0;
        {
            std::lock_guard<std::mutex> lock(tasksMutex_);
            if (stopping_) { return Status::Error("allgather store is stopping"); }
            tasks_.emplace(handle, task);
            queueDepth = tasks_.size();
        }
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_task_queue_depth"),
                                 static_cast<double>(queueDepth));
        auto work = [this, task](PlatformRuntime& platform, StreamHandle stream, Status fatal) {
            if (fatal.Failure()) {
                Finish(task, fatal);
                return;
            }
            try {
                if (task->operation == TaskState::Operation::Load) {
                    ProcessLoad(task, platform);
                } else {
                    ProcessDump(task, platform, stream);
                }
            } catch (const std::exception& error) {
                Finish(task,
                       Status::Error(fmt::format("allgather progress failed: {}", error.what())));
            } catch (...) {
                Finish(task, Status::Error("allgather progress failed"));
            }
        };
        task->enqueuedAt = Clock::now();
        auto status = operation == TaskState::Operation::Dump && separateDumpQueue_
                          ? runtime_->SubmitDump(std::move(work))
                          : runtime_->Submit(std::move(work));
        if (status.Failure()) {
            std::lock_guard<std::mutex> lock(tasksMutex_);
            tasks_.erase(handle);
            return status;
        }
        return Detail::TaskHandle(handle);
    }

    std::vector<WindowPlan> BuildWindows(const Detail::TaskDesc& input,
                                         TaskState::Operation operation) const
    {
        const auto storageWorldSize =
            operation == TaskState::Operation::Load ? loadStorageWorldSize_ : dumpStorageWorldSize_;
        std::vector<size_t> counts(storageWorldSize, 0);
        std::vector<uint32_t> owners(input.size());
        std::vector<size_t> ownerSlots(input.size());
        std::vector<size_t> order(input.size());
        std::iota(order.begin(), order.end(), 0);
        size_t windowCount = 0;
        for (const auto row : order) {
            owners[row] = Owner(input[row].owner, storageWorldSize);
            ownerSlots[row] = counts[owners[row]]++;
            windowCount = std::max(windowCount, ownerSlots[row] / windowBlocks_ + 1);
        }
        std::vector<WindowPlan> windows(windowCount);
        for (auto& window : windows) {
            window.ownerCounts.assign(storageWorldSize, 0);
            window.ownerOffsets.assign(storageWorldSize, 0);
        }
        for (const auto row : order) {
            const size_t window = ownerSlots[row] / windowBlocks_;
            windows[window].rows.push_back(row);
            windows[window].owners.push_back(owners[row]);
            windows[window].ownerSlots.push_back(ownerSlots[row] % windowBlocks_);
            ++windows[window].ownerCounts[owners[row]];
        }
        for (auto& window : windows) {
            auto rows = std::move(window.rows);
            auto rowOwners = std::move(window.owners);
            auto rowOwnerSlots = std::move(window.ownerSlots);
            window.rows.reserve(rows.size());
            window.owners.reserve(rows.size());
            window.ownerSlots.reserve(rows.size());
            for (size_t owner = 0; owner < storageWorldSize; ++owner) {
                window.ownerOffsets[owner] = window.rows.size();
                for (size_t i = 0; i < rows.size(); ++i) {
                    if (rowOwners[i] != owner) { continue; }
                    window.rows.push_back(rows[i]);
                    window.owners.push_back(rowOwners[i]);
                    window.ownerSlots.push_back(rowOwnerSlots[i]);
                }
            }
            window.rankStrideBlocks = windowBlocks_;
        }
        return windows;
    }

    PendingBackend SubmitLoadWindow(const std::shared_ptr<TaskState>& task, size_t windowIndex,
                                    size_t slotIndex, LoadMetrics& metrics)
    {
        auto& slot = *loadSlots_[slotIndex];
        PendingBackend pending{windowIndex, slotIndex, 0, false, Status::OK()};
        try {
            // The caller only hands over slots whose previous consumers have
            // released the send buffer. The backend fills it from its own stream.
            pending.slotReadyAt = Clock::now();
            Detail::TaskDesc backendTask;
            const auto& window = task->windows[windowIndex];
            for (size_t i = 0; i < window.rows.size(); ++i) {
                if (window.owners[i] != loadStorageRank_) { continue; }
                auto shard = task->input[window.rows[i]];
                shard.addrs = {static_cast<uint8_t*>(slot.send.data) +
                               window.ownerSlots[i] * shardSize_};
                backendTask.push_back(std::move(shard));
            }
            pending.ownedRows = backendTask.size();
            if (backendTask.empty()) { return pending; }
            backendTask.brief = "AllGatherLoadWindow";
            const auto started = Clock::now();
            pending.backendSubmitStartedAt = started;
            auto result = backend_->Load(std::move(backendTask));
            pending.backendSubmittedAt = Clock::now();
            metrics.backendSubmitMs += Milliseconds(pending.backendSubmittedAt - started);
            if (!result) {
                pending.status = result.Error();
            } else {
                pending.handle = result.Value();
                pending.submitted = true;
            }
        } catch (const std::exception& error) {
            pending.status = Status::Error(
                fmt::format("allgather load window submission failed: {}", error.what()));
        } catch (...) {
            pending.status = Status::Error("allgather load window submission failed");
        }
        return pending;
    }

    Status PrepareLoadMetadata(const std::shared_ptr<TaskState>& task, const WindowPlan& window,
                               LoadSlot& slot, PlatformRuntime& platform, StreamHandle stream) const
    {
        auto& destinations = slot.hostDestinations;
        size_t destinationOffset = 0;
        for (size_t i = 0; i < window.rows.size(); ++i) {
            const auto& shard = task->input[window.rows[i]];
            for (const auto address : shard.addrs) {
                destinations.data[destinationOffset++] = reinterpret_cast<uint64_t>(address);
            }
        }
        if (distributedLoadEnabled_ && remoteScatterMode_ == RemoteScatterMode::BatchCopy) {
            return Status::OK();
        }
        auto status = platform.CopyHostToDevice(slot.destinations.data, slot.destinations.size,
                                                destinations.data,
                                                destinationOffset * sizeof(uint64_t), stream);
        if (status.Failure()) { return status; }

        auto& routes = slot.hostRoutes;
        size_t routeOffset = 0;
        for (size_t i = 0; i < window.rows.size(); ++i) {
            routes.data[routeOffset++] = window.owners[i];
            routes.data[routeOffset++] = window.ownerSlots[i];
        }
        return platform.CopyHostToDevice(slot.routes.data, slot.routes.size, routes.data,
                                         routeOffset * sizeof(uint32_t), stream);
    }

    Status ScatterWindow(const WindowPlan& window, LoadSlot& slot, PlatformRuntime& platform,
                         StreamHandle stream)
    {
        const auto taskCount = window.rows.size() * plan_.chunkCount;
        const auto usedWorkers = static_cast<uint32_t>(std::min(scatterAivCores_, taskCount));
        if (usedWorkers == 0) { return Status::OK(); }
        return platform.LaunchCompactScatter(
            stream, slot.send.data, slot.destinations.data, slot.routes.data, chunkLayout_.data,
            window.rows.size(), plan_.chunkCount, tensorSizes_.size(),
            window.rankStrideBlocks * shardSize_, shardSize_, usedWorkers);
    }

    Status BatchCopyReadyRanks(const WindowPlan& window, size_t slotIndex, LoadSlot& slot,
                               PlatformRuntime& platform, StreamHandle stream, uint64_t readyMask)
    {
        const auto& peers = remoteTransport_->PeerBuffers(slotIndex);
        std::vector<DeviceCopy> copies;
        copies.reserve(window.rows.size() * tensorSizes_.size());
        for (uint32_t owner = 0; owner < worldSize_; ++owner) {
            if ((readyMask & (uint64_t{1} << owner)) == 0) { continue; }
            const size_t firstRow = window.ownerOffsets[owner];
            const size_t lastRow = firstRow + window.ownerCounts[owner];
            const auto* peer = static_cast<const uint8_t*>(peers[owner]);
            for (size_t row = firstRow; row < lastRow; ++row) {
                const auto* source = peer + window.ownerSlots[row] * shardSize_;
                for (size_t tensor = 0; tensor < tensorSizes_.size(); ++tensor) {
                    auto* destination = reinterpret_cast<void*>(
                        slot.hostDestinations.data[row * tensorSizes_.size() + tensor]);
                    if (destination == nullptr || tensorSizes_[tensor] == 0) { continue; }
                    copies.push_back(
                        {destination, source + tensorOffsets_[tensor], tensorSizes_[tensor]});
                }
            }
        }
        return platform.CopyDeviceBatchAsync(stream, copies.data(), copies.size());
    }

    Status ScatterBufferedReadyRanks(const WindowPlan& window, LoadSlot& slot,
                                     PlatformRuntime& platform, StreamHandle stream,
                                     uint64_t readyMask)
    {
        for (uint32_t first = 0; first < worldSize_;) {
            if ((readyMask & (uint64_t{1} << first)) == 0) {
                ++first;
                continue;
            }
            uint32_t last = first + 1;
            while (last < worldSize_ && (readyMask & (uint64_t{1} << last)) != 0) { ++last; }
            const size_t rowOffset = window.ownerOffsets[first];
            size_t rowCount = 0;
            for (uint32_t owner = first; owner < last; ++owner) {
                rowCount += window.ownerCounts[owner];
            }
            if (rowCount != 0) {
                const auto taskCount = rowCount * plan_.chunkCount;
                const auto usedWorkers =
                    static_cast<uint32_t>(std::min(scatterAivCores_, taskCount));
                auto* destinations = static_cast<uint8_t*>(slot.destinations.data) +
                                     rowOffset * tensorSizes_.size() * sizeof(uint64_t);
                auto* routes =
                    static_cast<uint8_t*>(slot.routes.data) + rowOffset * 2 * sizeof(uint32_t);
                auto status = platform.LaunchCompactScatter(
                    stream, slot.receive.data, destinations, routes, chunkLayout_.data, rowCount,
                    plan_.chunkCount, tensorSizes_.size(), windowBlocks_ * shardSize_, shardSize_,
                    usedWorkers);
                if (status.Failure()) { return status; }
            }
            first = last;
        }
        return Status::OK();
    }

    Status CopyThenScatterReadyRanks(const WindowPlan& window, size_t slotIndex, LoadSlot& slot,
                                     PlatformRuntime& platform, StreamHandle stream,
                                     uint64_t readyMask)
    {
        const auto& peers = remoteTransport_->PeerBuffers(slotIndex);
        std::vector<DeviceCopy> copies;
        copies.reserve(worldSize_);
        const size_t rankStride = windowBlocks_ * shardSize_;
        for (uint32_t owner = 0; owner < worldSize_; ++owner) {
            if ((readyMask & (uint64_t{1} << owner)) == 0 || window.ownerCounts[owner] == 0) {
                continue;
            }
            copies.push_back({static_cast<uint8_t*>(slot.receive.data) + owner * rankStride,
                              peers[owner], window.ownerCounts[owner] * shardSize_});
        }
        auto status = platform.CopyDeviceBatchAsync(stream, copies.data(), copies.size());
        if (status.Failure()) { return status; }
        return ScatterBufferedReadyRanks(window, slot, platform, stream, readyMask);
    }

    Status ScatterReadyRanks(const WindowPlan& window, size_t slotIndex, LoadSlot& slot,
                             PlatformRuntime& platform, StreamHandle stream, uint64_t readyMask)
    {
        if (remoteScatterMode_ == RemoteScatterMode::BatchCopy) {
            return BatchCopyReadyRanks(window, slotIndex, slot, platform, stream, readyMask);
        }
        if (remoteScatterMode_ == RemoteScatterMode::CopyThenScatter) {
            return CopyThenScatterReadyRanks(window, slotIndex, slot, platform, stream, readyMask);
        }
        for (uint32_t first = 0; first < worldSize_;) {
            if ((readyMask & (uint64_t{1} << first)) == 0) {
                ++first;
                continue;
            }
            uint32_t last = first + 1;
            while (last < worldSize_ && (readyMask & (uint64_t{1} << last)) != 0) { ++last; }
            const size_t rowOffset = window.ownerOffsets[first];
            size_t rowCount = 0;
            for (uint32_t owner = first; owner < last; ++owner) {
                rowCount += window.ownerCounts[owner];
            }
            if (rowCount != 0) {
                const auto taskCount = rowCount * plan_.chunkCount;
                const auto usedWorkers =
                    static_cast<uint32_t>(std::min(scatterAivCores_, taskCount));
                auto* destinations = static_cast<uint8_t*>(slot.destinations.data) +
                                     rowOffset * tensorSizes_.size() * sizeof(uint64_t);
                auto* routes =
                    static_cast<uint8_t*>(slot.routes.data) + rowOffset * 2 * sizeof(uint32_t);
                auto status = platform.LaunchRemoteScatter(
                    stream, slot.peerBuffers.data, destinations, routes, chunkLayout_.data,
                    rowCount, plan_.chunkCount, tensorSizes_.size(), shardSize_, usedWorkers);
                if (status.Failure()) { return status; }
            }
            first = last;
        }
        return Status::OK();
    }

    void ProcessLoadBackendOnly(const std::shared_ptr<TaskState>& task)
    {
        const auto taskStarted = Clock::now();
        LoadMetrics metrics;
        metrics.queueWaitMs = Milliseconds(taskStarted - task->queuedAt);
        metrics.windows = task->windows.size();
        Status firstError = Status::OK();
        std::deque<size_t> freeSlots;
        std::deque<PendingBackend> active;
        for (size_t i = 0; i < loadSlots_.size(); ++i) { freeSlots.push_back(i); }
        size_t nextWindow = 0;
        const auto prefetchStarted = Clock::now();
        Prefetch(task->input.data(), task->input.size());
        metrics.prefetchMs = Milliseconds(Clock::now() - prefetchStarted);
        auto submitReadyWindows = [&] {
            while (nextWindow < task->windows.size() && !freeSlots.empty()) {
                const auto slot = freeSlots.front();
                freeSlots.pop_front();
                active.push_back(SubmitLoadWindow(task, nextWindow, slot, metrics));
                ++nextWindow;
            }
        };
        submitReadyWindows();
        while (!active.empty()) {
            const auto waitStarted = Clock::now();
            auto pending = std::move(active.front());
            active.pop_front();
            if (pending.status.Success() && pending.submitted) {
                pending.status = backend_->Wait(pending.handle);
            }
            metrics.backendWaitMs += Milliseconds(Clock::now() - waitStarted);
            if (pending.status.Failure() && firstError.Success()) { firstError = pending.status; }
            freeSlots.push_back(pending.slot);
            submitReadyWindows();
        }
        metrics.totalMs = Milliseconds(Clock::now() - taskStarted);
        RecordLoadMetrics(metrics, task->profile);
        Finish(task, firstError);
    }

    void ProcessLoad(const std::shared_ptr<TaskState>& task, PlatformRuntime& platform)
    {
        if (loadBackendOnly_) {
            ProcessLoadBackendOnly(task);
            return;
        }
        const auto taskStarted = Clock::now();
        LoadMetrics metrics;
        metrics.queueWaitMs = Milliseconds(taskStarted - task->queuedAt);
        metrics.windows = task->windows.size();
        Status firstError = Status::OK();
        std::vector<PendingBackend> pending(loadSlots_.size());
        std::vector<bool> submitted(loadSlots_.size(), false);
        std::vector<bool> taskSlots(loadSlots_.size(), false);
        std::vector<TimingWindow> timingWindows;
        if (task->profile) { timingWindows.reserve(task->windows.size()); }
        std::vector<StageTraceRecord> stageTraceRecords;
        if (task->stageTrace) { stageTraceRecords.reserve(task->windows.size()); }

        const auto prefetchStarted = Clock::now();
        Prefetch(task->input.data(), task->input.size());
        metrics.prefetchMs = Milliseconds(Clock::now() - prefetchStarted);
        auto reclaim = [&](size_t slotIndex) {
            auto& slot = *loadSlots_[slotIndex];
            if (!slot.inFlight) { return Status::OK(); }
            const auto started = Clock::now();
            auto status = Status::OK();
            if (!slot.consumedPublished) {
                status = platform.SynchronizeEvent(slot.sendFree);
                if (status.Success() && distributedLoadEnabled_ && !skipRemoteScatter_) {
                    remoteTransport_->PublishConsumed(slotIndex, slot.generation);
                    slot.consumedPublished = true;
                }
            }
            if (status.Success() && distributedLoadEnabled_ && !skipRemoteScatter_) {
                status = remoteTransport_->WaitConsumed(slotIndex, slot.generation);
            }
            metrics.slotWaitMs += Milliseconds(Clock::now() - started);
            if (status.Success()) { slot.inFlight = false; }
            return status;
        };
        auto submit = [&](size_t windowIndex, size_t slotIndex) {
            pending[slotIndex] = SubmitLoadWindow(task, windowIndex, slotIndex, metrics);
            submitted[slotIndex] = true;
        };
        const size_t firstSlot = nextLoadSlot_;
        nextLoadSlot_ =
            (nextLoadSlot_ + task->windows.size() % loadSlots_.size()) % loadSlots_.size();
        for (size_t window = 0; window < std::min(task->windows.size(), loadSlots_.size());
             ++window) {
            const size_t slotIndex = (firstSlot + window) % loadSlots_.size();
            auto status = reclaim(slotIndex);
            if (status.Failure()) {
                firstError = status;
                if (IsFatalCommunication(status)) { runtime_->Poison(status); }
                break;
            }
            submit(window, slotIndex);
        }

        for (size_t windowIndex = 0; windowIndex < task->windows.size(); ++windowIndex) {
            const size_t slotIndex = (firstSlot + windowIndex) % loadSlots_.size();
            if (!submitted[slotIndex]) {
                auto status = reclaim(slotIndex);
                if (status.Failure()) {
                    firstError = status;
                    if (IsFatalCommunication(status)) { runtime_->Poison(status); }
                    break;
                }
                submit(windowIndex, slotIndex);
            }
            auto current = std::move(pending[slotIndex]);
            submitted[slotIndex] = false;
            const auto backendWaitStartedAt = Clock::now();
            if (current.status.Success() && current.submitted) {
                current.status = backend_->Wait(current.handle);
            }
            const auto backendReadyAt = Clock::now();
            metrics.backendWaitMs += Milliseconds(backendReadyAt - backendWaitStartedAt);
            auto& slot = *loadSlots_[slotIndex];
            const auto stream = runtime_->SlotStream(slotIndex);
            const auto& window = task->windows[windowIndex];
            const auto metadataStarted = Clock::now();
            auto roundStatus = PrepareLoadMetadata(task, window, slot, platform, stream);
            metrics.metadataMs += Milliseconds(Clock::now() - metadataStarted);
            if (roundStatus.Failure() && firstError.Success()) { firstError = roundStatus; }
            if (current.status.Failure() && firstError.Success()) { firstError = current.status; }

            TimingWindow* timing = nullptr;
            if (task->profile) {
                timingWindows.emplace_back();
                if (timingWindows.back().Setup(platform) &&
                    platform.RecordEvent(timingWindows.back().start, stream).Success()) {
                    timing = &timingWindows.back();
                }
            }
            const auto readyStartedAt = Clock::now();
            const bool localFailed = current.status.Failure() || roundStatus.Failure();
            bool peerFailed = localFailed;
            auto status = Status::OK();
            if (distributedLoadEnabled_ && !skipRemoteScatter_) {
                const bool batchedRemoteScatter =
                    remoteScatterMode_ == RemoteScatterMode::BatchedRemoteScatter;
                const uint64_t generation = slot.generation + 1;
                const uint64_t windowTag = WindowTag(*task, window);
                status =
                    remoteTransport_->PublishReady(slotIndex, generation, windowTag, localFailed);
                metrics.remoteReadySubmitMs += Milliseconds(Clock::now() - readyStartedAt);
                uint64_t pendingMask =
                    worldSize_ == 64 ? ~uint64_t{0} : (uint64_t{1} << worldSize_) - 1;
                bool scatterTimingStarted = false;
                while (status.Success() && pendingMask != 0) {
                    const auto waitStarted = Clock::now();
                    auto ready =
                        remoteTransport_->WaitReady(slotIndex, generation, windowTag, pendingMask);
                    metrics.remoteReadySubmitMs += Milliseconds(Clock::now() - waitStarted);
                    if (!ready) {
                        status = ready.Error();
                        break;
                    }
                    pendingMask &= ~ready.Value().readyMask;
                    peerFailed = peerFailed || ready.Value().failedMask != 0;
                    if (peerFailed || skipLoadScatter_ || batchedRemoteScatter) { continue; }
                    if (timing != nullptr && !scatterTimingStarted) {
                        if (platform.RecordEvent(timing->remoteReady, stream).Success()) {
                            scatterTimingStarted = true;
                        } else {
                            timing = nullptr;
                        }
                    }
                    const auto started = Clock::now();
                    auto scatterStatus = ScatterReadyRanks(window, slotIndex, slot, platform,
                                                           stream, ready.Value().readyMask);
                    metrics.scatterSubmitMs += Milliseconds(Clock::now() - started);
                    if (scatterStatus.Failure() && firstError.Success()) {
                        firstError = scatterStatus;
                    }
                    if (scatterStatus.Failure()) { peerFailed = true; }
                }
                if (status.Success() && !peerFailed && !skipLoadScatter_ && batchedRemoteScatter) {
                    if (timing != nullptr && !scatterTimingStarted) {
                        if (platform.RecordEvent(timing->remoteReady, stream).Success()) {
                            scatterTimingStarted = true;
                        } else {
                            timing = nullptr;
                        }
                    }
                    const auto started = Clock::now();
                    const uint64_t allRanks =
                        worldSize_ == 64 ? ~uint64_t{0} : (uint64_t{1} << worldSize_) - 1;
                    auto scatterStatus =
                        ScatterReadyRanks(window, slotIndex, slot, platform, stream, allRanks);
                    metrics.scatterSubmitMs += Milliseconds(Clock::now() - started);
                    if (scatterStatus.Failure() && firstError.Success()) {
                        firstError = scatterStatus;
                    }
                    if (scatterStatus.Failure()) { peerFailed = true; }
                }
                slot.generation = generation;
                if (peerFailed && firstError.Success()) {
                    firstError = Status::Error("remote scatter owner backend load failed");
                }
            } else if (!distributedLoadEnabled_ && !localFailed && !skipLoadScatter_) {
                if (timing != nullptr &&
                    platform.RecordEvent(timing->remoteReady, stream).Failure()) {
                    timing = nullptr;
                }
                const auto started = Clock::now();
                auto scatterStatus = ScatterWindow(window, slot, platform, stream);
                metrics.scatterSubmitMs += Milliseconds(Clock::now() - started);
                if (scatterStatus.Failure() && firstError.Success()) { firstError = scatterStatus; }
            }
            const auto readySubmittedAt = Clock::now();
            if (status.Failure()) {
                if (firstError.Success()) { firstError = status; }
                if (IsFatalCommunication(status)) { runtime_->Poison(status); }
                break;
            }
            if (timing != nullptr) {
                if (platform.RecordEvent(timing->scatterDone, stream).Success()) {
                    timing->complete = true;
                } else {
                    timing = nullptr;
                }
            }
            status = platform.RecordEvent(slot.sendFree, stream);
            if (status.Failure()) {
                status = Status::StoreUnhealthy(
                    fmt::format("failed to record remote scatter slot completion: {}", status));
                if (firstError.Success()) { firstError = status; }
                runtime_->Poison(status);
                break;
            }
            slot.inFlight = true;
            slot.consumedPublished = false;
            taskSlots[slotIndex] = true;
            if (task->stageTrace) {
                stageTraceRecords.push_back(StageTraceRecord{
                    windowIndex, slotIndex, window.rows.size(), current.ownedRows,
                    window.rows.size() * shardSize_ * (distributedLoadEnabled_ ? worldSize_ : 1),
                    current.slotReadyAt, current.backendSubmitStartedAt, current.backendSubmittedAt,
                    backendWaitStartedAt, backendReadyAt, readyStartedAt, readySubmittedAt});
            }
        }

        const auto completionStream = runtime_->CompletionStream();
        for (size_t slot = 0; slot < taskSlots.size(); ++slot) {
            if (!taskSlots[slot]) { continue; }
            auto status = platform.WaitEvent(completionStream, loadSlots_[slot]->sendFree);
            if (status.Failure() && firstError.Success()) { firstError = status; }
        }
        const auto started = Clock::now();
        auto completionStatus = platform.SynchronizeStream(completionStream);
        metrics.syncMs = Milliseconds(Clock::now() - started);
        if (completionStatus.Failure() && firstError.Success()) { firstError = completionStatus; }
        if (completionStatus.Success() && distributedLoadEnabled_ && !skipRemoteScatter_) {
            for (size_t slotIndex = 0; slotIndex < taskSlots.size(); ++slotIndex) {
                auto& slot = *loadSlots_[slotIndex];
                if (!taskSlots[slotIndex] || !slot.inFlight || slot.consumedPublished) { continue; }
                remoteTransport_->PublishConsumed(slotIndex, slot.generation);
                slot.consumedPublished = true;
            }
        }
        if (task->profile && completionStatus.Success()) {
            for (auto& timing : timingWindows) {
                if (!timing.Valid()) { continue; }
                float elapsed = 0.0F;
                if (platform.EventElapsedTime(&elapsed, timing.start, timing.remoteReady)
                        .Success()) {
                    metrics.remoteReadyDeviceMs += elapsed;
                }
                if (platform.EventElapsedTime(&elapsed, timing.remoteReady, timing.scatterDone)
                        .Success()) {
                    metrics.scatterDeviceMs += elapsed;
                }
            }
        }
        metrics.totalMs = Milliseconds(Clock::now() - taskStarted);
        RecordLoadMetrics(metrics, task->profile);
        RecordStageTrace(*task, taskStarted, stageTraceRecords);
        Finish(task, firstError);
    }

    void RecordStageTrace(const TaskState& task, Clock::time_point dequeuedAt,
                          const std::vector<StageTraceRecord>& records) const
    {
        for (const auto& record : records) {
            UC_INFO(
                "AG_STAGE device={} rank={} shard={} task={} window={} slot={} rows={} owned={} "
                "bytes={} received_us={} enqueued_us={} dequeued_us={} slot_ready_us={} "
                "backend_submit_begin_us={} backend_submit_end_us={} "
                "backend_wait_begin_us={} backend_ready_us={} "
                "remote_ready_begin_us={} remote_ready_end_us={}.",
                deviceId_, rank_, shardSize_, task.sequence, record.window, record.slot,
                record.rows, record.ownedRows, record.remoteScatterBytes,
                Microseconds(task.queuedAt), Microseconds(task.enqueuedAt),
                Microseconds(dequeuedAt), Microseconds(record.slotReadyAt),
                Microseconds(record.backendSubmitStartedAt),
                Microseconds(record.backendSubmittedAt), Microseconds(record.backendWaitStartedAt),
                Microseconds(record.backendReadyAt), Microseconds(record.remoteReadyStartedAt),
                Microseconds(record.remoteReadySubmittedAt));
        }
    }

    static void RecordLoadMetrics(const LoadMetrics& metrics, bool profiled)
    {
        if (!profiled) { return; }
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_windows"),
                                 static_cast<double>(metrics.windows));
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_task_queue_wait_ms"),
                                 metrics.queueWaitMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_prefetch_ms"),
                                 metrics.prefetchMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_metadata_ms"),
                                 metrics.metadataMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_slot_reclaim_wait_ms"),
                                 metrics.slotWaitMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_backend_submit_ms"),
                                 metrics.backendSubmitMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_inner_wait_ms"),
                                 metrics.backendWaitMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_remote_ready_submit_ms"),
                                 metrics.remoteReadySubmitMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_scatter_submit_ms"),
                                 metrics.scatterSubmitMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_sync_ms"), metrics.syncMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_total_ms"), metrics.totalMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_remote_ready_device_ms"),
                                 metrics.remoteReadyDeviceMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_scatter_device_ms"),
                                 metrics.scatterDeviceMs);
        UC_INFO(
            "AllGather load profile: queue_ms={:.3f}, prefetch_ms={:.3f}, metadata_ms={:.3f}, "
            "slot_ms={:.3f}, "
            "backend_submit_ms={:.3f}, backend_wait_ms={:.3f}, "
            "remote_ready_submit_ms={:.3f}, remote_ready_device_ms={:.3f}, "
            "scatter_submit_ms={:.3f}, scatter_device_ms={:.3f}, "
            "sync_ms={:.3f}, total_ms={:.3f}, windows={}.",
            metrics.queueWaitMs, metrics.prefetchMs, metrics.metadataMs, metrics.slotWaitMs,
            metrics.backendSubmitMs, metrics.backendWaitMs, metrics.remoteReadySubmitMs,
            metrics.remoteReadyDeviceMs, metrics.scatterSubmitMs, metrics.scatterDeviceMs,
            metrics.syncMs, metrics.totalMs, metrics.windows);
    }

    static std::pair<std::vector<uint64_t>, std::vector<uint32_t>> BalanceDescriptors(
        std::vector<std::array<uint64_t, 3>> descriptors, size_t maxWorkers = kMaxCopyWorkers)
    {
        if (descriptors.empty()) { return {}; }
        const size_t workers = std::min({kMaxCopyWorkers, maxWorkers, descriptors.size()});
        std::vector<std::vector<std::array<uint64_t, 3>>> bins(workers);
        std::vector<size_t> loads(workers, 0);
        std::sort(descriptors.begin(), descriptors.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs[2] > rhs[2]; });
        for (const auto& descriptor : descriptors) {
            const auto target = std::min_element(loads.begin(), loads.end()) - loads.begin();
            bins[target].push_back(descriptor);
            loads[target] += descriptor[2];
        }
        std::vector<uint64_t> ordered;
        std::vector<uint32_t> offsets{0};
        for (const auto& bin : bins) {
            for (const auto& descriptor : bin) {
                ordered.insert(ordered.end(), descriptor.begin(), descriptor.end());
            }
            offsets.push_back(ordered.size() / 3);
        }
        return {std::move(ordered), std::move(offsets)};
    }

    PendingBackend SubmitDumpWindow(const std::shared_ptr<TaskState>& task, size_t windowIndex,
                                    size_t slotIndex, PlatformRuntime& platform,
                                    StreamHandle stream, DumpMetrics& metrics)
    {
        auto& slot = *dumpSlots_[slotIndex];
        const auto& window = task->windows[windowIndex];
        std::vector<std::array<uint64_t, 3>> descriptors;
        Detail::TaskDesc backendTask;
        for (size_t i = 0; i < window.rows.size(); ++i) {
            if (window.owners[i] != dumpStorageRank_) { continue; }
            const auto& source = task->input[window.rows[i]];
            size_t shardOffset = 0;
            for (size_t tensor = 0; tensor < tensorSizes_.size(); ++tensor) {
                const auto address = reinterpret_cast<uint64_t>(source.addrs[tensor]);
                for (size_t offset = 0; address != 0 && offset < tensorSizes_[tensor];
                     offset += kCopyChunkBytes) {
                    descriptors.push_back(
                        {address + offset,
                         reinterpret_cast<uint64_t>(slot.send.data) +
                             window.ownerSlots[i] * shardSize_ + shardOffset + offset,
                         std::min(kCopyChunkBytes, tensorSizes_[tensor] - offset)});
                }
                shardOffset += tensorSizes_[tensor];
            }
            auto shard = source;
            shard.addrs = {static_cast<uint8_t*>(slot.send.data) +
                           window.ownerSlots[i] * shardSize_};
            backendTask.push_back(std::move(shard));
        }
        PendingBackend pending{windowIndex, slotIndex, 0, false, Status::OK()};
        if (backendTask.empty()) { return pending; }
        auto balanced = BalanceDescriptors(std::move(descriptors));
        auto status = Status::OK();
        if (!balanced.first.empty()) {
            std::copy(balanced.first.begin(), balanced.first.end(), slot.hostDescriptors.data);
            std::copy(balanced.second.begin(), balanced.second.end(), slot.hostOffsets.data);
            status = platform.CopyHostToDevice(slot.descriptors.data, slot.descriptors.size,
                                               slot.hostDescriptors.data,
                                               balanced.first.size() * sizeof(uint64_t), stream);
            if (status.Failure()) { throw std::runtime_error(status.ToString()); }
            status = platform.CopyHostToDevice(slot.offsets.data, slot.offsets.size,
                                               slot.hostOffsets.data,
                                               balanced.second.size() * sizeof(uint32_t), stream);
            if (status.Failure()) { throw std::runtime_error(status.ToString()); }
            const auto usedWorkers = static_cast<uint32_t>(balanced.second.size() - 1);
            status = platform.LaunchSegmentedCopy(stream, slot.descriptors.data, slot.offsets.data,
                                                  usedWorkers);
            if (status.Failure()) { throw std::runtime_error(status.ToString()); }
        }
        status = platform.RecordEvent(slot.ready, stream);
        if (status.Failure()) { throw std::runtime_error(status.ToString()); }
        backendTask.brief = "AllGatherDumpWindow";
        backendTask.prerequisiteHandle = reinterpret_cast<uintptr_t>(slot.ready);
        const auto started = Clock::now();
        auto result = backend_->Dump(std::move(backendTask));
        metrics.backendSubmitMs += Milliseconds(Clock::now() - started);
        if (!result) { throw std::runtime_error(result.Error().ToString()); }
        pending.handle = result.Value();
        pending.submitted = true;
        return pending;
    }

    void ProcessDump(const std::shared_ptr<TaskState>& task, PlatformRuntime& platform,
                     StreamHandle stream)
    {
        const auto taskStarted = Clock::now();
        DumpMetrics metrics;
        metrics.queueWaitMs = Milliseconds(taskStarted - task->queuedAt);
        metrics.windows = task->windows.size();
        Status firstError = Status::OK();
        if (task->input.prerequisiteHandle != 0) {
            auto status = platform.WaitEvent(
                stream, reinterpret_cast<EventHandle>(task->input.prerequisiteHandle));
            if (status.Failure()) { firstError = status; }
        }
        std::deque<PendingBackend> active;
        size_t nextWindow = 0;
        while (nextWindow < task->windows.size() && active.size() < dumpSlotCount_) {
            active.push_back(
                SubmitDumpWindow(task, nextWindow, active.size(), platform, stream, metrics));
            ++nextWindow;
        }
        while (!active.empty()) {
            auto pending = active.front();
            active.pop_front();
            if (pending.submitted) {
                const auto started = Clock::now();
                auto status = backend_->Wait(pending.handle);
                metrics.backendWaitMs += Milliseconds(Clock::now() - started);
                if (status.Failure() && firstError.Success()) { firstError = status; }
            }
            if (nextWindow < task->windows.size()) {
                active.push_back(
                    SubmitDumpWindow(task, nextWindow, pending.slot, platform, stream, metrics));
                ++nextWindow;
            }
        }
        const auto syncStarted = Clock::now();
        auto status = platform.SynchronizeStream(stream);
        metrics.syncMs = Milliseconds(Clock::now() - syncStarted);
        if (status.Failure() && firstError.Success()) { firstError = status; }
        metrics.totalMs = Milliseconds(Clock::now() - taskStarted);
        RecordDumpMetrics(metrics, task->profile);
        Finish(task, firstError);
    }

    static void RecordDumpMetrics(const DumpMetrics& metrics, bool profiled)
    {
        if (!profiled) { return; }
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_dump_windows"),
                                 static_cast<double>(metrics.windows));
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_task_queue_wait_ms"),
                                 metrics.queueWaitMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_dump_backend_submit_ms"),
                                 metrics.backendSubmitMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_dump_backend_wait_ms"),
                                 metrics.backendWaitMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_dump_sync_ms"), metrics.syncMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_dump_total_ms"), metrics.totalMs);
        UC_INFO(
            "AllGather dump profile: queue_ms={:.3f}, backend_submit_ms={:.3f}, "
            "backend_wait_ms={:.3f}, sync_ms={:.3f}, total_ms={:.3f}, windows={}.",
            metrics.queueWaitMs, metrics.backendSubmitMs, metrics.backendWaitMs, metrics.syncMs,
            metrics.totalMs, metrics.windows);
    }

    static void Finish(const std::shared_ptr<TaskState>& task, Status status)
    {
        {
            std::lock_guard<std::mutex> lock(task->mutex);
            if (task->done) { return; }
            task->status = std::move(status);
            task->done = true;
        }
        task->condition.notify_all();
    }

    std::shared_ptr<TaskState> FindTask(Detail::TaskHandle handle)
    {
        std::lock_guard<std::mutex> lock(tasksMutex_);
        const auto found = tasks_.find(handle);
        return found == tasks_.end() ? nullptr : found->second;
    }

    void EraseTask(Detail::TaskHandle handle, const std::shared_ptr<TaskState>& task)
    {
        std::lock_guard<std::mutex> lock(tasksMutex_);
        const auto found = tasks_.find(handle);
        if (found != tasks_.end() && found->second == task) { tasks_.erase(found); }
    }

    StoreV1* backend_{nullptr};
    int32_t deviceId_{-1};
    size_t shardSize_{0};
    size_t blockSize_{0};
    std::vector<size_t> tensorSizes_;
    uint32_t rank_{0};
    uint32_t worldSize_{1};
    uint32_t loadStorageRank_{0};
    uint32_t loadStorageWorldSize_{1};
    uint32_t dumpStorageRank_{0};
    uint32_t dumpStorageWorldSize_{1};
    bool replicated_{false};
    bool scatterOnly_{false};
    bool distributedLoadEnabled_{false};
    size_t windowBlocks_{kDefaultWindowBlocks};
    size_t loadSlotCount_{kDefaultLoadSlots};
    size_t runtimeSlotStreamCount_{0};
    size_t dumpSlotCount_{kDefaultDumpSlots};
    size_t scatterAivCores_{1};
    RemoteScatterMode remoteScatterMode_{RemoteScatterMode::BatchCopy};
    std::string runtimeKey_;
    std::vector<uint8_t> bootstrapKey_;
    std::shared_ptr<PlatformRuntime> platform_;
    StageMemoryPlan plan_;
    DeviceBuffer chunkLayout_;
    std::vector<std::unique_ptr<LoadSlot>> loadSlots_;
    std::vector<size_t> tensorOffsets_;
    std::unique_ptr<RemoteScatterTransport> remoteTransport_;
    std::vector<std::unique_ptr<DumpSlot>> dumpSlots_;
    std::shared_ptr<AllGatherRuntime> runtime_;
    std::atomic<Detail::TaskHandle> nextHandle_{1};
    std::atomic<uint64_t> taskSequence_{0};
    std::atomic<uint64_t> loadTaskSequence_{0};
    size_t nextLoadSlot_{0};
    size_t profileSampleEvery_{0};
    size_t stageTraceSampleEvery_{0};
    bool separateDumpQueue_{true};
    bool skipRemoteScatter_{false};
    bool skipLoadScatter_{false};
    bool loadBackendOnly_{false};
    std::mutex tasksMutex_;
    std::unordered_map<Detail::TaskHandle, std::shared_ptr<TaskState>> tasks_;
    bool stopping_{false};
};

}  // namespace UC::AllGatherStore

extern "C" UC::StoreV1* MakeAllGatherStore() { return new UC::AllGatherStore::AllGatherStore(); }
