#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
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
#include "frame_protocol.h"
#include "logger/logger.h"
#include "memory_plan.h"
#include "metrics_api.h"
#include "platform_runtime.h"
#include "remote_scatter_transport.h"
#include "ucmstore_v1.h"

namespace UC::AllGatherStore {
namespace {

constexpr uint32_t kDefaultCollectiveBufferMb = 8;
constexpr size_t kDefaultLoadSlots = 2;
constexpr size_t kDefaultDumpSlots = 2;
constexpr size_t kDefaultReceiveSlots = 2;
using Clock = std::chrono::steady_clock;

Status ParseCollectiveMode(const std::string& value, uint32_t* mode, std::string* canonical)
{
    std::string normalized = value;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (normalized == "auto" || normalized == "default") {
        *mode = 0;
        *canonical = "auto";
    } else if (normalized == "host" || normalized == "host_ts") {
        *mode = 1;
        *canonical = "host";
    } else if (normalized == "aicpu" || normalized == "ai_cpu" || normalized == "aicpu_ts" ||
               normalized == "ai_cpu_ts") {
        *mode = 2;
        *canonical = "aicpu_ts";
    } else if (normalized == "aiv") {
        *mode = 3;
        *canonical = "aiv";
    } else {
        return Status::InvalidParam(
            "invalid allgather collective mode({}); expected auto, host, aicpu_ts, or aiv", value);
    }
    return Status::OK();
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
 * @brief One staged window on its way from the backend into the collective.
 *
 * Send slots size the reordering freedom: the more of them there are, the more
 * candidate windows the progress thread can choose from when networked storage
 * completes reads out of order. They are cheap - one window of payload each.
 */
struct LoadSlot {
    std::shared_ptr<PlatformRuntime> platform;
    DeviceBuffer send;
    /* Local-coalesced mode only: the scatter reads the send buffer directly. */
    DeviceBuffer destinations;
    DeviceBuffer routes;
    HostBuffer<uint64_t> hostDestinations;
    HostBuffer<uint32_t> hostRoutes;
    HostBuffer<uint8_t> hostFrame;
    /* The send buffer has been consumed, so the backend may refill it. */
    EventHandle sendFree{nullptr};
    EventHandle ready{nullptr};
    DeviceBuffer peerBuffers;
    uint64_t generation{0};
    bool inFlight{false};

    ~LoadSlot()
    {
        if (sendFree != nullptr && platform != nullptr) { platform->DestroyEvent(sendFree); }
        if (ready != nullptr && platform != nullptr) { platform->DestroyEvent(ready); }
    }
};

/**
 * @brief Landing area for one collective, plus the stream that drains it.
 *
 * Receive buffers are world_size times larger than send buffers but only need to
 * cover the collective-to-scatter handoff. Collectives are serialized on one
 * stream, so two or three of these saturate that handoff no matter how many send
 * slots exist. Sizing them together with send slots is what used to make the
 * receive area the dominant allocation.
 */
struct ReceiveSlot {
    std::shared_ptr<PlatformRuntime> platform;
    DeviceBuffer receive;
    /* Metadata for this round is on the stream. The previous round that used
     * this slot scattered on the same stream, so waiting for `prepared` also
     * proves the receive buffer is free. */
    EventHandle prepared{nullptr};
    EventHandle scattered{nullptr};
    bool inFlight{false};

    ~ReceiveSlot()
    {
        if (platform == nullptr) { return; }
        for (const auto event : {prepared, scattered}) {
            if (event != nullptr) { platform->DestroyEvent(event); }
        }
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
    std::vector<uint64_t> frameCounts;
    std::vector<uint64_t> frameDisplacements;
    /* Only the local-coalesced path uses this, as the scatter rank stride. */
    size_t collectiveBlocks{};
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
    double placementMs{};
    double slotWaitMs{};
    double backendSubmitMs{};
    double backendWaitMs{};
    double collectiveSubmitMs{};
    double scatterSubmitMs{};
    double collectiveDeviceMs{};
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
    uint64_t collectiveBytes{};
    Clock::time_point slotReadyAt;
    Clock::time_point backendSubmitStartedAt;
    Clock::time_point backendSubmittedAt;
    Clock::time_point backendWaitStartedAt;
    Clock::time_point backendReadyAt;
    Clock::time_point collectiveStartedAt;
    Clock::time_point collectiveSubmittedAt;
};

struct TimingWindow {
    PlatformRuntime* platform{nullptr};
    EventHandle start{nullptr};
    EventHandle collectiveDone{nullptr};
    EventHandle scatterDone{nullptr};
    bool complete{false};

    TimingWindow() = default;
    TimingWindow(const TimingWindow&) = delete;
    TimingWindow& operator=(const TimingWindow&) = delete;
    TimingWindow(TimingWindow&& other) noexcept
        : platform(other.platform),
          start(other.start),
          collectiveDone(other.collectiveDone),
          scatterDone(other.scatterDone),
          complete(other.complete)
    {
        other.platform = nullptr;
        other.start = nullptr;
        other.collectiveDone = nullptr;
        other.scatterDone = nullptr;
        other.complete = false;
    }
    ~TimingWindow()
    {
        if (platform == nullptr) { return; }
        platform->DestroyEvent(start);
        platform->DestroyEvent(collectiveDone);
        platform->DestroyEvent(scatterDone);
    }

    bool Setup(PlatformRuntime& value)
    {
        platform = &value;
        if (platform->CreateEvent(&start, true).Failure() ||
            platform->CreateEvent(&collectiveDone, true).Failure() ||
            platform->CreateEvent(&scatterDone, true).Failure()) {
            return false;
        }
        return true;
    }

    bool Valid() const
    {
        return complete && start != nullptr && collectiveDone != nullptr && scatterDone != nullptr;
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
    ~TaskState()
    {
        if (metadataReady != nullptr && platform != nullptr) {
            platform->DestroyEvent(metadataReady);
        }
    }

    Operation operation;
    Detail::TaskDesc input;
    std::vector<WindowPlan> windows;
    std::mutex mutex;
    std::condition_variable condition;
    Status status{Status::OK()};
    bool done{false};
    DeviceBuffer deviceError;
    HostBuffer<uint32_t> hostError;
    DeviceBuffer destinations;
    HostBuffer<uint64_t> hostDestinations;
    DeviceBuffer locationSend;
    DeviceBuffer locationReceive;
    HostBuffer<uint8_t> hostLocationSend;
    HostBuffer<uint8_t> hostLocationReceive;
    bool placementRequired{false};
    EventHandle metadataReady{nullptr};
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
        config.GetNumber("allgather_load_groups", requestedLoadGroups_);
        config.GetNumber("allgather_dump_slots", dumpSlotCount_);
        config.GetNumber("allgather_receive_slots", receiveSlotCount_);
        config.GetNumber("allgather_runtime_slot_streams", runtimeSlotStreamCount_);
        config.GetNumber("allgather_hccl_buffer_mb", collectiveBufferMb_);
        config.GetNumber("allgather_collective_buffer_mb", collectiveBufferMb_);
        config.Get("allgather_collective_mode", collectiveModeName_);
        config.Get("allgather_collective_variable_counts", variableCollectiveRequested_);
        config.GetNumber("allgather_scatter_aiv_cores", scatterAivCores_);
        config.GetNumber("allgather_profile_sample_every", profileSampleEvery_);
        config.GetNumber("allgather_stage_trace_sample_every", stageTraceSampleEvery_);
        config.Get("allgather_separate_dump_queue", separateDumpQueue_);
        config.Get("allgather_load_skip_collective", skipLoadCollective_);
        config.Get("allgather_load_skip_scatter", skipLoadScatter_);
        config.Get("allgather_load_backend_only", loadBackendOnly_);
        config.Get("allgather_runtime_key", runtimeKey_);
        config.GetNumbers("allgather_hccl_root_info", rootInfo_);
        config.GetNumbers("allgather_collective_root_info", rootInfo_);

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
        remoteScatterEnabled_ = distributedLoadEnabled_ && platform_->SupportsRemoteScatter();
        collectiveEnabled_ = distributedLoadEnabled_ && !remoteScatterEnabled_;
        status = ParseCollectiveMode(collectiveModeName_, &collectiveMode_, &collectiveModeName_);
        if (status.Failure()) { return status; }
        if (loadSlotCount_ == 0) {
            return Status::InvalidParam("allgather load slot count must be positive");
        }
        if (!remoteScatterEnabled_ && receiveSlotCount_ == 0) {
            return Status::InvalidParam("allgather receive slot count must be positive");
        }
        if (scatterAivCores_ == 0 || scatterAivCores_ > kMaxCopyWorkers) {
            return Status::InvalidParam("allgather scatter AIV cores must be in [1, {}]",
                                        kMaxCopyWorkers);
        }
        receiveSlotCount_ = remoteScatterEnabled_ ? 0 : std::min(receiveSlotCount_, loadSlotCount_);
        const size_t requiredSlotStreams =
            remoteScatterEnabled_ ? loadSlotCount_ : receiveSlotCount_;
        if (runtimeSlotStreamCount_ == 0) { runtimeSlotStreamCount_ = requiredSlotStreams; }
        if (runtimeSlotStreamCount_ < requiredSlotStreams) {
            return Status::InvalidParam(
                "allgather runtime slot streams({}) are fewer than required streams({})",
                runtimeSlotStreamCount_, requiredSlotStreams);
        }
        if (requestedLoadGroups_ > 1) {
            UC_WARN(
                "allgather_load_groups={} is no longer supported; the stage uses one "
                "ordered data path. Raise allgather_load_slots for pipelining instead.",
                requestedLoadGroups_);
        }
        loadStorageWorldSize_ = distributedLoadEnabled_ ? worldSize_ : 1;
        loadStorageRank_ = distributedLoadEnabled_ ? rank_ : 0;
        dumpStorageWorldSize_ = replicated_ ? worldSize_ : 1;
        dumpStorageRank_ = replicated_ ? rank_ : 0;
        try {
            plan_ = CalculateStageMemoryPlan(
                tensorSizes_, shardSize_, worldSize_, distributedLoadEnabled_, loadSlotCount_,
                dumpSlotCount_, windowBlocks_, std::max<size_t>(receiveSlotCount_, 1),
                remoteScatterEnabled_);
        } catch (const std::exception& error) {
            return Status::InvalidParam("invalid allgather memory plan: {}", error.what());
        }

        variableCollectiveEnabled_ =
            collectiveEnabled_ && variableCollectiveRequested_ && platform_->SupportsAllGatherV();
        if (collectiveEnabled_ && variableCollectiveRequested_ && !variableCollectiveEnabled_) {
            UC_WARN(
                "allgather_collective_variable_counts is unavailable on {}; using fixed "
                "AllGather",
                platform_->Name());
        }
        status = AllocateBuffers();
        if (status.Failure()) { return status; }
        if (remoteScatterEnabled_) {
            status = SetupRemoteScatter();
            if (status.Failure()) { return status; }
        }
        auto runtime = AllGatherRuntime::Acquire(runtimeKey_, deviceId_, rank_, worldSize_,
                                                 collectiveBufferMb_, collectiveMode_, rootInfo_,
                                                 collectiveEnabled_, runtimeSlotStreamCount_);
        if (!runtime) { return runtime.Error(); }
        runtime_ = runtime.Value();
        UC_INFO(
            "AllGatherStore: shard={}, world={}, window_blocks={}, load_slots={}, "
            "receive_slots={}, runtime_slot_streams={}, dump_slots={}, frame_bytes={}, "
            "collective_message_bytes={}, "
            "payload_bytes={}, "
            "metadata_bytes={}, collective_buffer_bytes={}, scatter_only={}, "
            "skip_load_collective={}, skip_load_scatter={}, load_backend_only={}, "
            "collective_mode={}, variable_counts={}, remote_scatter={}, scatter_aiv_cores={}, "
            "platform={}.",
            shardSize_, worldSize_, windowBlocks_, loadSlotCount_, receiveSlotCount_,
            runtimeSlotStreamCount_, dumpSlotCount_, frameBytes_,
            collectiveEnabled_ ? worldSize_ * frameBytes_ : 0, plan_.PayloadBytes(),
            plan_.MetadataBytes(),
            CalculateCollectiveBytes(collectiveBufferMb_, collectiveEnabled_, 1), scatterOnly_,
            skipLoadCollective_, skipLoadScatter_, loadBackendOnly_, collectiveModeName_,
            variableCollectiveEnabled_, remoteScatterEnabled_, scatterAivCores_, platform_->Name());
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
        return task->done;
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
        if (status.Success() && !skipLoadCollective_ && task->hostError.data != nullptr &&
            task->hostError.data[0] != 0) {
            status = Status::Error("allgather owner backend load failed");
        }
        EraseTask(handle, task);
        return status;
    }

private:
    Status AllocateBuffers()
    {
        const size_t windowBytes = windowBlocks_ * shardSize_;
        frameMetadataBytes_ = collectiveEnabled_ ? CalculateFrameMetadataBytes(windowBlocks_) : 0;
        frameBytes_ = windowBytes + frameMetadataBytes_;
        const size_t maxRows = windowBlocks_ * (distributedLoadEnabled_ ? worldSize_ : 1);
        loadSlots_.reserve(loadSlotCount_);
        for (size_t i = 0; i < loadSlotCount_; ++i) {
            auto slot = std::make_unique<LoadSlot>();
            slot->platform = platform_;
            auto status = slot->send.Allocate(platform_, frameBytes_);
            if (status.Failure()) { return status; }
            if (collectiveEnabled_) {
                // Row addresses live on the task in framed mode, not on the slot.
                status = slot->hostFrame.Allocate(platform_, frameMetadataBytes_);
                if (status.Failure()) { return status; }
            } else {
                status = slot->destinations.Allocate(
                    platform_, maxRows * tensorSizes_.size() * sizeof(uint64_t));
                if (status.Failure()) { return status; }
                status = slot->routes.Allocate(platform_, maxRows * 2 * sizeof(uint32_t));
                if (status.Failure()) { return status; }
                status = slot->hostDestinations.Allocate(platform_, maxRows * tensorSizes_.size());
                if (status.Failure()) { return status; }
                status = slot->hostRoutes.Allocate(platform_, maxRows * 2);
                if (status.Failure()) { return status; }
            }
            status = platform_->CreateEvent(&slot->sendFree);
            if (status.Failure()) { return status; }
            if (remoteScatterEnabled_) {
                status = platform_->CreateInterprocessEvent(&slot->ready);
                if (status.Failure()) { return status; }
                status = slot->peerBuffers.Allocate(platform_, worldSize_ * sizeof(void*));
                if (status.Failure()) { return status; }
            }
            loadSlots_.push_back(std::move(slot));
        }
        receiveSlots_.reserve(receiveSlotCount_);
        for (size_t i = 0; i < receiveSlotCount_; ++i) {
            auto slot = std::make_unique<ReceiveSlot>();
            slot->platform = platform_;
            if (collectiveEnabled_) {
                auto status = slot->receive.Allocate(platform_, worldSize_ * frameBytes_);
                if (status.Failure()) { return status; }
            }
            for (auto* event : {&slot->prepared, &slot->scattered}) {
                auto status = platform_->CreateEvent(event);
                if (status.Failure()) { return status; }
            }
            receiveSlots_.push_back(std::move(slot));
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
        std::vector<EventHandle> events;
        buffers.reserve(loadSlots_.size());
        events.reserve(loadSlots_.size());
        for (const auto& slot : loadSlots_) {
            buffers.push_back(slot->send.data);
            events.push_back(slot->ready);
        }
        std::ostringstream signature;
        signature << "shard=" << shardSize_ << ":window=" << windowBlocks_
                  << ":slots=" << loadSlotCount_ << ":tensors=";
        for (const auto size : tensorSizes_) { signature << size << ','; }
        auto transport = RemoteScatterTransport::Create(platform_, rootInfo_, signature.str(),
                                                        rank_, worldSize_, buffers, events);
        if (!transport) { return transport.Error(); }
        remoteTransport_ = std::move(transport.Value());
        for (size_t slot = 0; slot < loadSlots_.size(); ++slot) {
            const auto& peers = remoteTransport_->PeerBuffers(slot);
            auto status = platform_->CopyHostToDevice(
                loadSlots_[slot]->peerBuffers.data, loadSlots_[slot]->peerBuffers.size,
                peers.data(), peers.size() * sizeof(void*), nullptr);
            if (status.Failure()) { return status; }
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
        task->placementRequired = operation == TaskState::Operation::Load && collectiveEnabled_ &&
                                  !skipLoadCollective_ && !loadBackendOnly_ &&
                                  task->windows.size() > 1;
        if (operation == TaskState::Operation::Load && collectiveEnabled_ && !loadBackendOnly_) {
            auto status = task->deviceError.Allocate(platform_, sizeof(uint32_t), true);
            if (status.Failure()) { return status; }
            status = task->hostError.Allocate(platform_, 1);
            if (status.Failure()) { return status; }
            task->hostError.data[0] = 0;
            {
                const auto destinationCount = task->input.size() * tensorSizes_.size();
                status =
                    task->destinations.Allocate(platform_, destinationCount * sizeof(uint64_t));
                if (status.Failure()) { return status; }
                status = task->hostDestinations.Allocate(platform_, destinationCount);
                if (status.Failure()) { return status; }
                size_t offset = 0;
                for (const auto& shard : task->input) {
                    for (const auto address : shard.addrs) {
                        task->hostDestinations.data[offset++] = reinterpret_cast<uint64_t>(address);
                    }
                }
                status = platform_->CreateEvent(&task->metadataReady);
                if (status.Failure()) { return status; }
                if (task->placementRequired) {
                    status = task->locationSend.Allocate(platform_, task->input.size());
                    if (status.Failure()) { return status; }
                    status =
                        task->locationReceive.Allocate(platform_, task->input.size() * worldSize_);
                    if (status.Failure()) { return status; }
                    status = task->hostLocationSend.Allocate(platform_, task->input.size());
                    if (status.Failure()) { return status; }
                    status = task->hostLocationReceive.Allocate(platform_,
                                                                task->input.size() * worldSize_);
                    if (status.Failure()) { return status; }
                }
            }
        }
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
        auto work = [this, task](PlatformRuntime& platform, StreamHandle stream,
                                 CollectiveHandle collective, Status fatal) {
            if (fatal.Failure()) {
                // The collective domain already diverged from its peers. Running
                // this task would only add another mismatched round.
                Finish(task, fatal);
                return;
            }
            try {
                if (task->operation == TaskState::Operation::Load) {
                    ProcessLoad(task, platform, stream, collective);
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
                                         TaskState::Operation operation,
                                         const std::vector<DataLocation>* locations = nullptr) const
    {
        const auto storageWorldSize =
            operation == TaskState::Operation::Load ? loadStorageWorldSize_ : dumpStorageWorldSize_;
        std::vector<size_t> counts(storageWorldSize, 0);
        std::vector<uint32_t> owners(input.size());
        std::vector<size_t> ownerSlots(input.size());
        std::vector<size_t> order(input.size());
        std::iota(order.begin(), order.end(), 0);
        if (locations != nullptr) {
            std::stable_sort(order.begin(), order.end(), [locations](size_t lhs, size_t rhs) {
                return static_cast<uint8_t>((*locations)[lhs]) >
                       static_cast<uint8_t>((*locations)[rhs]);
            });
        }
        size_t windowCount = 0;
        for (const auto row : order) {
            owners[row] = Owner(input[row].owner, storageWorldSize);
            ownerSlots[row] = counts[owners[row]]++;
            windowCount = std::max(windowCount, ownerSlots[row] / windowBlocks_ + 1);
        }
        std::vector<WindowPlan> windows(windowCount);
        for (auto& window : windows) { window.ownerCounts.assign(storageWorldSize, 0); }
        for (const auto row : order) {
            const size_t window = ownerSlots[row] / windowBlocks_;
            windows[window].rows.push_back(row);
            windows[window].owners.push_back(owners[row]);
            windows[window].ownerSlots.push_back(ownerSlots[row] % windowBlocks_);
            ++windows[window].ownerCounts[owners[row]];
        }
        for (auto& window : windows) { window.collectiveBlocks = windowBlocks_; }
        if (operation == TaskState::Operation::Load && collectiveEnabled_ &&
            variableCollectiveEnabled_ && windows.size() == 1) {
            auto& window = windows.front();
            window.frameCounts.resize(worldSize_);
            window.frameDisplacements.resize(worldSize_);
            for (size_t owner = 0; owner < worldSize_; ++owner) {
                window.frameCounts[owner] =
                    frameMetadataBytes_ + window.ownerCounts[owner] * shardSize_;
                window.frameDisplacements[owner] = owner * frameBytes_;
            }
        }
        return windows;
    }

    Status PrepareLoadPlan(const std::shared_ptr<TaskState>& task, PlatformRuntime& platform,
                           CollectiveHandle collective, LoadMetrics& metrics)
    {
        const auto started = Clock::now();
        if (!task->placementRequired) {
            metrics.windows = task->windows.size();
            return Status::OK();
        }
        std::vector<DataLocation> locations(task->input.size(), DataLocation::MISSING);
        std::vector<Detail::Shard> localShards;
        std::vector<size_t> localRows;
        localShards.reserve(task->input.size());
        localRows.reserve(task->input.size());
        for (size_t row = 0; row < task->input.size(); ++row) {
            if (Owner(task->input[row].owner, loadStorageWorldSize_) != loadStorageRank_) {
                continue;
            }
            localShards.push_back({task->input[row].owner, task->input[row].index, {}});
            localRows.push_back(row);
        }

        Status lookupStatus = Status::OK();
        if (!localShards.empty()) {
            auto result = backend_->LookupDataLocation(localShards.data(), localShards.size());
            if (!result) {
                lookupStatus = result.Error();
            } else if (result.Value().size() != localRows.size()) {
                lookupStatus = Status::Error("location lookup returned unexpected result size");
            } else {
                std::vector<Detail::Shard> prefetch;
                for (size_t i = 0; i < localRows.size(); ++i) {
                    locations[localRows[i]] = result.Value()[i];
                    if (result.Value()[i] == DataLocation::BACKEND) {
                        prefetch.push_back(localShards[i]);
                    }
                }
                if (!prefetch.empty()) { backend_->Prefetch(prefetch.data(), prefetch.size()); }
            }
        }

        if (collectiveEnabled_) {
            std::memset(task->hostLocationSend.data, 0, task->hostLocationSend.count);
            for (const auto row : localRows) {
                task->hostLocationSend.data[row] = static_cast<uint8_t>(locations[row]);
            }
            const auto stream = runtime_->CollectiveStream();
            auto status = platform.CopyHostToDevice(
                task->locationSend.data, task->locationSend.size, task->hostLocationSend.data,
                task->hostLocationSend.count, stream);
            if (status.Success()) {
                status = platform.AllGather(task->locationSend.data, task->locationReceive.data,
                                            task->input.size(), collective, stream);
            }
            if (status.Success()) {
                status = platform.CopyDeviceToHost(
                    task->hostLocationReceive.data, task->hostLocationReceive.count,
                    task->locationReceive.data, task->locationReceive.size, stream);
            }
            if (status.Success()) { status = platform.SynchronizeStream(stream); }
            if (status.Failure()) {
                return Status::StoreUnhealthy(
                    fmt::format("allgather placement exchange failed: {}", status));
            }
            for (size_t row = 0; row < task->input.size(); ++row) {
                const auto owner = Owner(task->input[row].owner, worldSize_);
                const auto encoded =
                    task->hostLocationReceive.data[owner * task->input.size() + row];
                if (encoded <= static_cast<uint8_t>(DataLocation::CACHE_READY)) {
                    locations[row] = static_cast<DataLocation>(encoded);
                } else {
                    locations[row] = DataLocation::MISSING;
                }
            }
        }
        task->windows = BuildWindows(task->input, TaskState::Operation::Load, &locations);
        metrics.windows = task->windows.size();
        metrics.placementMs = Milliseconds(Clock::now() - started);
        return lookupStatus;
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
                shard.addrs = {static_cast<uint8_t*>(slot.send.data) + frameMetadataBytes_ +
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

    Status PrepareLoadMetadata(const std::shared_ptr<TaskState>& task, size_t windowIndex,
                               const WindowPlan& window, LoadSlot& slot, bool backendFailed,
                               PlatformRuntime& platform, StreamHandle stream) const
    {
        if (!collectiveEnabled_) {
            auto& destinations = slot.hostDestinations;
            size_t destinationOffset = 0;
            for (size_t i = 0; i < window.rows.size(); ++i) {
                const auto& shard = task->input[window.rows[i]];
                for (const auto address : shard.addrs) {
                    destinations.data[destinationOffset++] = reinterpret_cast<uint64_t>(address);
                }
            }
            auto status = platform.CopyHostToDevice(slot.destinations.data, slot.destinations.size,
                                                    destinations.data,
                                                    destinationOffset * sizeof(uint64_t), stream);
            if (status.Failure()) { return status; }
        } else {
            std::memset(slot.hostFrame.data, 0, frameMetadataBytes_);
            auto* header = reinterpret_cast<FrameHeader*>(slot.hostFrame.data);
            header->magic = kFrameMagic;
            header->version = kFrameVersion;
            header->sequenceLow = static_cast<uint32_t>(task->sequence);
            header->sequenceHigh = static_cast<uint32_t>(task->sequence >> 32);
            header->round = static_cast<uint32_t>(windowIndex);
            auto* records =
                reinterpret_cast<FrameRecord*>(slot.hostFrame.data + sizeof(FrameHeader));
            for (size_t i = 0; i < window.rows.size(); ++i) {
                if (window.owners[i] != loadStorageRank_) { continue; }
                auto& record = records[header->validCount++];
                record.row = static_cast<uint32_t>(window.rows[i]);
                record.status = backendFailed ? kFrameStatusBackendFailed : kFrameStatusReady;
                record.payloadSlot = window.ownerSlots[i];
            }
            return platform.CopyHostToDevice(slot.send.data, slot.send.size, slot.hostFrame.data,
                                             frameMetadataBytes_, stream);
        }

        auto& routes = slot.hostRoutes;
        size_t routeOffset = 0;
        for (size_t i = 0; i < window.rows.size(); ++i) {
            routes.data[routeOffset++] = window.owners[i];
            routes.data[routeOffset++] = window.ownerSlots[i];
        }
        return platform.CopyHostToDevice(slot.routes.data, slot.routes.size, routes.data,
                                         routeOffset * sizeof(uint32_t), stream);
    }

    Status ScatterWindow(const std::shared_ptr<TaskState>& task, const WindowPlan& window,
                         LoadSlot& slot, ReceiveSlot* receiveSlot, PlatformRuntime& platform,
                         StreamHandle stream)
    {
        const auto taskCount =
            (collectiveEnabled_ ? worldSize_ * windowBlocks_ : window.rows.size()) *
            plan_.chunkCount;
        const auto usedWorkers = static_cast<uint32_t>(std::min(scatterAivCores_, taskCount));
        if (usedWorkers == 0) { return Status::OK(); }
        if (remoteScatterEnabled_) {
            return platform.LaunchRemoteScatter(
                stream, slot.peerBuffers.data, slot.destinations.data, slot.routes.data,
                chunkLayout_.data, window.rows.size(), plan_.chunkCount, tensorSizes_.size(),
                shardSize_, usedWorkers);
        }
        void* receive =
            collectiveEnabled_ && !skipLoadCollective_ ? receiveSlot->receive.data : slot.send.data;
        if (collectiveEnabled_) {
            // Repeating the valid local frame preserves scatter work in the
            // collective ablation without reading uninitialized receive frames.
            const auto frameStride = skipLoadCollective_ ? 0 : frameBytes_;
            return platform.LaunchFramedScatter(
                stream, receive, task->destinations.data, chunkLayout_.data, task->deviceError.data,
                static_cast<uint32_t>(task->input.size()), worldSize_,
                static_cast<uint32_t>(windowBlocks_), static_cast<uint32_t>(plan_.chunkCount),
                static_cast<uint32_t>(tensorSizes_.size()), frameStride, frameMetadataBytes_,
                shardSize_, task->sequence, kAnyFrameRound, usedWorkers);
        }
        return platform.LaunchCompactScatter(
            stream, receive, slot.destinations.data, slot.routes.data, chunkLayout_.data,
            window.rows.size(), plan_.chunkCount, tensorSizes_.size(),
            window.collectiveBlocks * shardSize_, shardSize_, usedWorkers);
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
        Prefetch(task->input.data(), task->input.size());
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

    void ProcessRemoteLoad(const std::shared_ptr<TaskState>& task, PlatformRuntime& platform)
    {
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

        Prefetch(task->input.data(), task->input.size());
        auto reclaim = [&](size_t slotIndex) {
            auto& slot = *loadSlots_[slotIndex];
            if (!slot.inFlight) { return Status::OK(); }
            const auto started = Clock::now();
            auto status = platform.SynchronizeEvent(slot.sendFree);
            if (status.Success()) {
                remoteTransport_->PublishConsumed(slotIndex, slot.generation);
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
        for (size_t window = 0; window < std::min(task->windows.size(), loadSlots_.size());
             ++window) {
            auto status = reclaim(window);
            if (status.Failure()) {
                firstError = status;
                if (IsFatalCommunication(status)) { runtime_->Poison(status); }
                break;
            }
            submit(window, window);
        }

        for (size_t windowIndex = 0; windowIndex < task->windows.size(); ++windowIndex) {
            const size_t slotIndex = windowIndex % loadSlots_.size();
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
            auto roundStatus = PrepareLoadMetadata(task, windowIndex, window, slot,
                                                   current.status.Failure(), platform, stream);
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
            auto readyStatus = platform.RecordEvent(slot.ready, stream);
            const uint64_t generation = slot.generation + 1;
            const uint64_t windowTag = WindowTag(*task, window);
            const bool localFailed =
                current.status.Failure() || roundStatus.Failure() || readyStatus.Failure();
            auto status =
                remoteTransport_->PublishReady(slotIndex, generation, windowTag, localFailed);
            if (readyStatus.Failure() && firstError.Success()) { firstError = readyStatus; }
            auto peerFailed = [&]() -> Expected<bool> {
                if (status.Failure()) { return status; }
                return remoteTransport_->WaitReady(slotIndex, generation, windowTag, stream);
            }();
            const auto readySubmittedAt = Clock::now();
            metrics.collectiveSubmitMs += Milliseconds(readySubmittedAt - readyStartedAt);
            if (!peerFailed) {
                if (firstError.Success()) { firstError = peerFailed.Error(); }
                if (IsFatalCommunication(peerFailed.Error())) {
                    runtime_->Poison(peerFailed.Error());
                }
                break;
            }
            slot.generation = generation;
            if (peerFailed.Value() && firstError.Success()) {
                firstError = Status::Error("remote scatter owner backend load failed");
            }
            if (timing != nullptr &&
                platform.RecordEvent(timing->collectiveDone, stream).Failure()) {
                timing = nullptr;
            }
            if (!peerFailed.Value() && !skipLoadScatter_) {
                const auto started = Clock::now();
                auto scatterStatus = ScatterWindow(task, window, slot, nullptr, platform, stream);
                metrics.scatterSubmitMs += Milliseconds(Clock::now() - started);
                if (scatterStatus.Failure() && firstError.Success()) { firstError = scatterStatus; }
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
            taskSlots[slotIndex] = true;
            if (task->stageTrace) {
                stageTraceRecords.push_back(StageTraceRecord{
                    windowIndex, slotIndex, window.rows.size(), current.ownedRows,
                    window.rows.size() * shardSize_ * worldSize_, current.slotReadyAt,
                    current.backendSubmitStartedAt, current.backendSubmittedAt,
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
        if (task->profile && completionStatus.Success()) {
            for (auto& timing : timingWindows) {
                if (!timing.Valid()) { continue; }
                float elapsed = 0.0F;
                if (platform.EventElapsedTime(&elapsed, timing.start, timing.collectiveDone)
                        .Success()) {
                    metrics.collectiveDeviceMs += elapsed;
                }
                if (platform.EventElapsedTime(&elapsed, timing.collectiveDone, timing.scatterDone)
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

    void ProcessLoad(const std::shared_ptr<TaskState>& task, PlatformRuntime& platform,
                     StreamHandle stream, CollectiveHandle collective)
    {
        if (loadBackendOnly_) {
            ProcessLoadBackendOnly(task);
            return;
        }
        if (remoteScatterEnabled_) {
            ProcessRemoteLoad(task, platform);
            return;
        }
        (void)stream;
        const auto taskStarted = Clock::now();
        LoadMetrics metrics;
        metrics.queueWaitMs = Milliseconds(taskStarted - task->queuedAt);
        Status firstError = Status::OK();
        Status localBackendError = Status::OK();
        bool collectiveFailed = false;
        std::deque<PendingBackend> active;
        auto planStatus = PrepareLoadPlan(task, platform, collective, metrics);
        if (IsFatalCommunication(planStatus)) {
            runtime_->Poison(planStatus);
            metrics.totalMs = Milliseconds(Clock::now() - taskStarted);
            RecordLoadMetrics(metrics, task->profile);
            Finish(task, planStatus);
            return;
        }
        if (planStatus.Failure()) { firstError = planStatus; }
        std::vector<TimingWindow> timingWindows;
        std::vector<StageTraceRecord> stageTraceRecords;
        if (task->stageTrace) { stageTraceRecords.reserve(task->windows.size()); }
        bool profiling = task->profile;
        if (profiling) {
            try {
                timingWindows.reserve(task->windows.size());
            } catch (...) {
                profiling = false;
            }
        }
        size_t nextWindow = 0;
        std::deque<size_t> freeSlots;
        std::deque<size_t> recyclingSlots;
        for (size_t i = 0; i < loadSlots_.size(); ++i) {
            (loadSlots_[i]->inFlight ? recyclingSlots : freeSlots).push_back(i);
        }
        std::deque<size_t> freeReceiveSlots;
        std::deque<size_t> recyclingReceiveSlots;
        std::vector<bool> taskReceiveSlots(receiveSlots_.size(), false);
        for (size_t i = 0; i < receiveSlots_.size(); ++i) {
            (receiveSlots_[i]->inFlight ? recyclingReceiveSlots : freeReceiveSlots).push_back(i);
        }
        if (collectiveEnabled_) {
            const auto metadataStream = runtime_->SlotStream(0);
            auto status = platform.CopyHostToDevice(
                task->destinations.data, task->destinations.size, task->hostDestinations.data,
                task->hostDestinations.count * sizeof(uint64_t), metadataStream);
            if (status.Success()) {
                status = platform.RecordEvent(task->metadataReady, metadataStream);
            }
            if (status.Failure()) {
                firstError = status;
            } else {
                // Every slot stream scatters through task->destinations, so all of
                // them must observe the copy, not only the stream that issued it.
                for (size_t i = 0; i < loadSlots_.size(); ++i) {
                    status = platform.WaitEvent(runtime_->SlotStream(i), task->metadataReady);
                    if (status.Failure() && firstError.Success()) { firstError = status; }
                }
            }
        }
        auto reclaimSlots = [&](bool blocking) {
            for (auto slot = recyclingSlots.begin(); slot != recyclingSlots.end();) {
                auto queried = platform.QueryEvent(loadSlots_[*slot]->sendFree);
                if (queried && !queried.Value()) {
                    ++slot;
                    continue;
                }
                if (!queried && firstError.Success()) { firstError = queried.Error(); }
                loadSlots_[*slot]->inFlight = false;
                freeSlots.push_back(*slot);
                slot = recyclingSlots.erase(slot);
            }
            if (!blocking || !freeSlots.empty() || recyclingSlots.empty()) { return; }
            // Collectives run in submission order on one stream, so the oldest
            // recycling send slot is always the next one to be released.
            const auto slot = recyclingSlots.front();
            recyclingSlots.pop_front();
            const auto started = Clock::now();
            auto status = platform.SynchronizeEvent(loadSlots_[slot]->sendFree);
            metrics.slotWaitMs += Milliseconds(Clock::now() - started);
            if (status.Failure() && firstError.Success()) { firstError = status; }
            loadSlots_[slot]->inFlight = false;
            freeSlots.push_back(slot);
        };
        auto submitReadyWindows = [&] {
            while (nextWindow < task->windows.size() && !freeSlots.empty()) {
                const auto slot = freeSlots.front();
                freeSlots.pop_front();
                active.push_back(SubmitLoadWindow(task, nextWindow, slot, metrics));
                ++nextWindow;
            }
        };
        auto reclaimReceiveSlots = [&](bool blocking) {
            for (auto slot = recyclingReceiveSlots.begin(); slot != recyclingReceiveSlots.end();) {
                auto queried = platform.QueryEvent(receiveSlots_[*slot]->scattered);
                if (queried && !queried.Value()) {
                    ++slot;
                    continue;
                }
                if (!queried && firstError.Success()) { firstError = queried.Error(); }
                receiveSlots_[*slot]->inFlight = false;
                freeReceiveSlots.push_back(*slot);
                slot = recyclingReceiveSlots.erase(slot);
            }
            if (!blocking || !freeReceiveSlots.empty() || recyclingReceiveSlots.empty()) { return; }
            const auto slot = recyclingReceiveSlots.front();
            recyclingReceiveSlots.pop_front();
            const auto started = Clock::now();
            auto status = platform.SynchronizeEvent(receiveSlots_[slot]->scattered);
            metrics.slotWaitMs += Milliseconds(Clock::now() - started);
            if (status.Failure() && firstError.Success()) { firstError = status; }
            receiveSlots_[slot]->inFlight = false;
            freeReceiveSlots.push_back(slot);
        };
        reclaimSlots(true);
        submitReadyWindows();
        while (!active.empty()) {
            const auto waitStarted = Clock::now();
            const auto backendWaitStartedAt = waitStarted;
            auto pending = std::move(active.front());
            active.pop_front();
            if (pending.status.Success() && pending.submitted) {
                pending.status = backend_->Wait(pending.handle);
            }
            const auto backendReadyAt = Clock::now();
            metrics.backendWaitMs += Milliseconds(backendReadyAt - waitStarted);
            auto& slot = *loadSlots_[pending.slot];
            // Receive slots rotate with the collective order, which is the same on
            // every rank. Metadata copies and the scatter run on that slot's
            // stream; the collective always runs on the one collective stream.
            reclaimReceiveSlots(false);
            if (freeReceiveSlots.empty()) { reclaimReceiveSlots(true); }
            if (freeReceiveSlots.empty()) {
                if (firstError.Success()) {
                    firstError = Status::Error("allgather receive slot reclamation failed");
                }
                break;
            }
            const auto receiveIndex = freeReceiveSlots.front();
            freeReceiveSlots.pop_front();
            auto& receiveSlot = *receiveSlots_[receiveIndex];
            const auto slotStream = runtime_->SlotStream(receiveIndex);
            const auto collectiveStream = runtime_->CollectiveStream();
            const auto loadCollective = runtime_->Collective();
            bool backendFailed = pending.status.Failure();
            if (backendFailed && localBackendError.Success()) {
                localBackendError = pending.status;
            }
            TimingWindow* timing = nullptr;
            if (profiling) {
                try {
                    timingWindows.emplace_back();
                    if (timingWindows.back().Setup(platform)) {
                        timing = &timingWindows.back();
                        if (platform.RecordEvent(timing->start, collectiveStream).Failure()) {
                            timing = nullptr;
                        }
                    }
                } catch (...) {
                    profiling = false;
                    timing = nullptr;
                }
            }
            const auto& window = task->windows[pending.window];
            auto roundStatus = PrepareLoadMetadata(task, pending.window, window, slot,
                                                   backendFailed, platform, slotStream);
            if (roundStatus.Failure() && firstError.Success()) { firstError = roundStatus; }
            Clock::time_point collectiveStartedAt;
            auto collectiveSubmittedAt = backendReadyAt;
            if (collectiveEnabled_ && !skipLoadCollective_) {
                collectiveStartedAt = Clock::now();
                // Waiting for `prepared` also covers the previous round's scatter:
                // both are queued on this slot's stream, in that order.
                auto status = platform.RecordEvent(receiveSlot.prepared, slotStream);
                if (status.Success()) {
                    status = platform.WaitEvent(collectiveStream, receiveSlot.prepared);
                }
                if (status.Success()) {
                    if (!window.frameCounts.empty()) {
                        status = platform.AllGatherV(
                            slot.send.data, window.frameCounts[rank_], receiveSlot.receive.data,
                            window.frameCounts.data(), window.frameDisplacements.data(),
                            loadCollective, collectiveStream);
                    } else {
                        status = platform.AllGather(slot.send.data, receiveSlot.receive.data,
                                                    frameBytes_, loadCollective, collectiveStream);
                    }
                }
                collectiveSubmittedAt = Clock::now();
                metrics.collectiveSubmitMs +=
                    Milliseconds(collectiveSubmittedAt - collectiveStartedAt);
                if (status.Failure()) {
                    // A collective that failed to enqueue never reached the wire,
                    // so this rank has already fallen out of step with its peers.
                    // Issuing the remaining rounds would only widen the gap.
                    roundStatus = status;
                    collectiveFailed = true;
                    status = Status::StoreUnhealthy(fmt::format(
                        "allgather collective failed, communicator is unusable: {}", status));
                    if (firstError.Success() || !IsFatalCommunication(firstError)) {
                        firstError = status;
                    }
                    runtime_->Poison(status);
                }
            }
            if (task->stageTrace) {
                stageTraceRecords.push_back(StageTraceRecord{
                    pending.window, pending.slot, window.rows.size(), pending.ownedRows,
                    collectiveEnabled_
                        ? (window.frameCounts.empty()
                               ? worldSize_ * frameBytes_
                               : std::accumulate(window.frameCounts.begin(),
                                                 window.frameCounts.end(), uint64_t{0}))
                        : window.collectiveBlocks * shardSize_,
                    pending.slotReadyAt, pending.backendSubmitStartedAt, pending.backendSubmittedAt,
                    backendWaitStartedAt, backendReadyAt, collectiveStartedAt,
                    collectiveSubmittedAt});
            }
            if (collectiveEnabled_ && !skipLoadCollective_) {
                // The scatter reads the receive buffer, so it has to trail the
                // collective that fills it. This is a device-side dependency:
                // the host never waits here.
                auto status = platform.RecordEvent(slot.sendFree, collectiveStream);
                if (status.Success()) { status = platform.WaitEvent(slotStream, slot.sendFree); }
                if (status.Failure() && firstError.Success()) { firstError = status; }
            }
            if (timing != nullptr &&
                platform.RecordEvent(timing->collectiveDone, collectiveStream).Failure()) {
                timing = nullptr;
            }
            if (roundStatus.Success() && (!backendFailed || collectiveEnabled_) &&
                !skipLoadScatter_) {
                const auto started = Clock::now();
                auto scatterStatus =
                    ScatterWindow(task, window, slot, &receiveSlot, platform, slotStream);
                metrics.scatterSubmitMs += Milliseconds(Clock::now() - started);
                if (scatterStatus.Failure()) { firstError = scatterStatus; }
            }
            if (timing != nullptr) {
                if (platform.RecordEvent(timing->scatterDone, slotStream).Failure()) {
                    timing = nullptr;
                } else {
                    timing->complete = true;
                }
            }
            auto scatteredStatus = platform.RecordEvent(receiveSlot.scattered, slotStream);
            if (scatteredStatus.Failure() && firstError.Success()) { firstError = scatteredStatus; }
            if (!collectiveEnabled_ || skipLoadCollective_) {
                // No collective read the send buffer, so the scatter is what frees
                // it. Record on the same stream, after the scatter.
                auto status = platform.RecordEvent(slot.sendFree, slotStream);
                if (status.Failure() && firstError.Success()) { firstError = status; }
            }
            receiveSlot.inFlight = true;
            taskReceiveSlots[receiveIndex] = true;
            recyclingReceiveSlots.push_back(receiveIndex);
            slot.inFlight = true;
            recyclingSlots.push_back(pending.slot);
            if (collectiveFailed) { break; }
            // Issue the next collective before paying for slot recycling: the
            // host only blocks when every slot is genuinely still in flight.
            reclaimSlots(false);
            submitReadyWindows();
            if (active.empty() && nextWindow < task->windows.size()) {
                reclaimSlots(true);
                submitReadyWindows();
            }
        }
        for (const auto& entry : active) {
            // The loop stopped early; these backend loads still own their slots.
            if (entry.submitted) { (void)backend_->Wait(entry.handle); }
            loadSlots_[entry.slot]->inFlight = false;
        }
        const auto completionStream = runtime_->CompletionStream();
        for (size_t i = 0; i < receiveSlots_.size(); ++i) {
            if (!taskReceiveSlots[i]) { continue; }
            // Every scatter ran on a receive slot's stream, and those streams are
            // FIFO, so the latest scatter per slot covers all of this task's work.
            auto waitStatus = platform.WaitEvent(completionStream, receiveSlots_[i]->scattered);
            if (waitStatus.Failure() && firstError.Success()) { firstError = waitStatus; }
        }
        auto status = Status::OK();
        if (collectiveEnabled_ && !skipLoadCollective_) {
            status = platform.CopyDeviceToHost(task->hostError.data, sizeof(uint32_t),
                                               task->deviceError.data, sizeof(uint32_t),
                                               completionStream);
            if (status.Failure() && firstError.Success()) { firstError = status; }
        }
        const auto syncStarted = Clock::now();
        status = platform.SynchronizeStream(completionStream);
        metrics.syncMs = Milliseconds(Clock::now() - syncStarted);
        if (status.Failure() && firstError.Success()) { firstError = status; }
        if (profiling && status.Success()) {
            for (auto& timing : timingWindows) {
                if (!timing.Valid()) { continue; }
                float elapsed = 0.0F;
                if (platform.EventElapsedTime(&elapsed, timing.start, timing.collectiveDone)
                        .Success()) {
                    metrics.collectiveDeviceMs += elapsed;
                }
                if (platform.EventElapsedTime(&elapsed, timing.collectiveDone, timing.scatterDone)
                        .Success()) {
                    metrics.scatterDeviceMs += elapsed;
                }
            }
        }
        if (firstError.Success() && localBackendError.Failure()) { firstError = localBackendError; }
        if (firstError.Success() && collectiveEnabled_ && !skipLoadCollective_ &&
            task->hostError.data[0] != 0) {
            firstError = Status::Error("allgather owner backend load failed");
        }
        metrics.totalMs = Milliseconds(Clock::now() - taskStarted);
        RecordLoadMetrics(metrics, task->profile && profiling);
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
                "collective_begin_us={} collective_end_us={}.",
                deviceId_, rank_, shardSize_, task.sequence, record.window, record.slot,
                record.rows, record.ownedRows, record.collectiveBytes, Microseconds(task.queuedAt),
                Microseconds(task.enqueuedAt), Microseconds(dequeuedAt),
                Microseconds(record.slotReadyAt), Microseconds(record.backendSubmitStartedAt),
                Microseconds(record.backendSubmittedAt), Microseconds(record.backendWaitStartedAt),
                Microseconds(record.backendReadyAt), Microseconds(record.collectiveStartedAt),
                Microseconds(record.collectiveSubmittedAt));
        }
    }

    static void RecordLoadMetrics(const LoadMetrics& metrics, bool profiled)
    {
        if (!profiled) { return; }
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_windows"),
                                 static_cast<double>(metrics.windows));
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_task_queue_wait_ms"),
                                 metrics.queueWaitMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_placement_ms"),
                                 metrics.placementMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_slot_reclaim_wait_ms"),
                                 metrics.slotWaitMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_backend_submit_ms"),
                                 metrics.backendSubmitMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_inner_wait_ms"),
                                 metrics.backendWaitMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_collective_submit_ms"),
                                 metrics.collectiveSubmitMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_scatter_submit_ms"),
                                 metrics.scatterSubmitMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_sync_ms"), metrics.syncMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_total_ms"), metrics.totalMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_collective_device_ms"),
                                 metrics.collectiveDeviceMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("allgather_load_scatter_device_ms"),
                                 metrics.scatterDeviceMs);
        UC_INFO(
            "AllGather load profile: queue_ms={:.3f}, placement_ms={:.3f}, slot_ms={:.3f}, "
            "backend_submit_ms={:.3f}, backend_wait_ms={:.3f}, "
            "collective_submit_ms={:.3f}, collective_device_ms={:.3f}, "
            "scatter_submit_ms={:.3f}, scatter_device_ms={:.3f}, "
            "sync_ms={:.3f}, total_ms={:.3f}, windows={}.",
            metrics.queueWaitMs, metrics.placementMs, metrics.slotWaitMs, metrics.backendSubmitMs,
            metrics.backendWaitMs, metrics.collectiveSubmitMs, metrics.collectiveDeviceMs,
            metrics.scatterSubmitMs, metrics.scatterDeviceMs, metrics.syncMs, metrics.totalMs,
            metrics.windows);
    }

    static std::pair<std::vector<uint64_t>, std::vector<uint32_t>> BalanceDescriptors(
        std::vector<std::array<uint64_t, 3>> descriptors)
    {
        if (descriptors.empty()) { return {}; }
        const size_t workers = std::min(kMaxCopyWorkers, descriptors.size());
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
    bool remoteScatterEnabled_{false};
    bool collectiveEnabled_{false};
    size_t windowBlocks_{kDefaultWindowBlocks};
    size_t frameMetadataBytes_{0};
    size_t frameBytes_{0};
    size_t loadSlotCount_{kDefaultLoadSlots};
    size_t requestedLoadGroups_{1};
    size_t receiveSlotCount_{kDefaultReceiveSlots};
    size_t runtimeSlotStreamCount_{0};
    size_t dumpSlotCount_{kDefaultDumpSlots};
    uint32_t collectiveBufferMb_{kDefaultCollectiveBufferMb};
    std::string collectiveModeName_{"host"};
    uint32_t collectiveMode_{1};
    bool variableCollectiveRequested_{true};
    bool variableCollectiveEnabled_{false};
    size_t scatterAivCores_{1};
    std::string runtimeKey_;
    std::vector<uint8_t> rootInfo_;
    std::shared_ptr<PlatformRuntime> platform_;
    StageMemoryPlan plan_;
    DeviceBuffer chunkLayout_;
    std::vector<std::unique_ptr<LoadSlot>> loadSlots_;
    std::unique_ptr<RemoteScatterTransport> remoteTransport_;
    std::vector<std::unique_ptr<ReceiveSlot>> receiveSlots_;
    std::vector<std::unique_ptr<DumpSlot>> dumpSlots_;
    std::shared_ptr<AllGatherRuntime> runtime_;
    std::atomic<Detail::TaskHandle> nextHandle_{1};
    std::atomic<uint64_t> taskSequence_{0};
    std::atomic<uint64_t> loadTaskSequence_{0};
    size_t profileSampleEvery_{0};
    size_t stageTraceSampleEvery_{0};
    bool separateDumpQueue_{true};
    bool skipLoadCollective_{false};
    bool skipLoadScatter_{false};
    bool loadBackendOnly_{false};
    std::mutex tasksMutex_;
    std::unordered_map<Detail::TaskHandle, std::shared_ptr<TaskState>> tasks_;
    bool stopping_{false};
};

}  // namespace UC::AllGatherStore

extern "C" UC::StoreV1* MakeAllGatherStore() { return new UC::AllGatherStore::AllGatherStore(); }
