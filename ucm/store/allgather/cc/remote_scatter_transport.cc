#include "remote_scatter_transport.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>

namespace UC::AllGatherStore {
namespace {

constexpr uint32_t kMagic = 0x55434950;
constexpr uint32_t kVersion = 4;
constexpr size_t kMaxHandleBytes = 256;
constexpr auto kSetupTimeout = std::chrono::seconds(120);
constexpr auto kDataTimeout = std::chrono::seconds(120);

struct alignas(64) SharedHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t worldSize;
    uint32_t slotCount;
    uint32_t memoryHandleBytes;
    uint64_t joinedMask;
    uint64_t authorizedMask;
    uint64_t openedMask;
    uint32_t unlinked;
};

size_t AlignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

uint64_t Fnv1a(uint64_t hash, const void* data, size_t bytes)
{
    const auto* input = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= input[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string SharedName(const std::vector<uint8_t>& key, const std::string& signature)
{
    auto hash = Fnv1a(1469598103934665603ULL, key.data(), key.size());
    hash = Fnv1a(hash, signature.data(), signature.size());
    std::ostringstream output;
    output << "/ucm_remote_scatter_" << std::hex << hash;
    return output.str();
}

struct Layout {
    size_t processIds;
    size_t memoryHandles;
    size_t windowTags;
    size_t ready;
    size_t consumed;
    size_t bytes;
};

Layout SharedLayout(uint32_t worldSize, size_t slotCount)
{
    const size_t entries = static_cast<size_t>(worldSize) * slotCount;
    Layout layout{};
    layout.processIds = AlignUp(sizeof(SharedHeader), 64);
    layout.memoryHandles = AlignUp(layout.processIds + worldSize * sizeof(int32_t), 64);
    layout.windowTags = AlignUp(layout.memoryHandles + entries * kMaxHandleBytes, 64);
    layout.ready = layout.windowTags + entries * sizeof(uint64_t);
    layout.consumed = layout.ready + entries * sizeof(uint64_t);
    layout.bytes = layout.consumed + entries * sizeof(uint64_t);
    return layout;
}

uint64_t RankMask(uint32_t worldSize)
{
    return worldSize == 64 ? ~uint64_t{0} : (uint64_t{1} << worldSize) - 1;
}

template <typename Predicate>
Status WaitUntil(Predicate predicate, std::chrono::steady_clock::duration timeout,
                 const char* operation)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return Status::Error(fmt::format("{} timed out", operation));
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return Status::OK();
}

uint64_t Load(const uint64_t* value) { return __atomic_load_n(value, __ATOMIC_ACQUIRE); }

void Store(uint64_t* value, uint64_t input) { __atomic_store_n(value, input, __ATOMIC_RELEASE); }

}  // namespace

RemoteScatterTransport::RemoteScatterTransport(std::shared_ptr<PlatformRuntime> platform,
                                               uint32_t rank, uint32_t worldSize, size_t slotCount)
    : platform_(std::move(platform)), rank_(rank), worldSize_(worldSize), slotCount_(slotCount)
{
}

Expected<std::unique_ptr<RemoteScatterTransport>> RemoteScatterTransport::Create(
    std::shared_ptr<PlatformRuntime> platform, const std::vector<uint8_t>& bootstrapKey,
    const std::string& stageSignature, uint32_t rank, uint32_t worldSize,
    const std::vector<void*>& localBuffers, size_t bufferBytes)
{
    if (platform == nullptr || !platform->SupportsRemoteScatter()) {
        return Status::Error("remote scatter is unavailable on this platform");
    }
    if (bootstrapKey.empty() || worldSize < 2 || worldSize > 64 || rank >= worldSize ||
        localBuffers.empty() || bufferBytes == 0 || localBuffers.size() > UINT32_MAX) {
        return Status::InvalidParam("invalid remote scatter transport configuration");
    }
    if (std::any_of(localBuffers.begin(), localBuffers.end(),
                    [](const auto value) { return value == nullptr; })) {
        return Status::InvalidParam("null remote scatter IPC resource");
    }
    auto result = std::unique_ptr<RemoteScatterTransport>(
        new RemoteScatterTransport(std::move(platform), rank, worldSize, localBuffers.size()));
    auto status = result->Setup(bootstrapKey, stageSignature, localBuffers, bufferBytes);
    if (status.Failure()) { return status; }
    return std::move(result);
}

Status RemoteScatterTransport::Setup(const std::vector<uint8_t>& bootstrapKey,
                                     const std::string& stageSignature,
                                     const std::vector<void*>& localBuffers, size_t bufferBytes)
{
    sharedName_ = SharedName(bootstrapKey, stageSignature);
    const auto layout = SharedLayout(worldSize_, slotCount_);
    mappingBytes_ = layout.bytes;
    int descriptor = -1;
    if (rank_ == 0) {
        descriptor = shm_open(sharedName_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        if (descriptor < 0) {
            return Status::Error(
                fmt::format("failed to create remote scatter shared state({})", errno));
        }
        sharedLinked_ = true;
        if (ftruncate(descriptor, mappingBytes_) != 0) {
            close(descriptor);
            shm_unlink(sharedName_.c_str());
            sharedLinked_ = false;
            return Status::Error(
                fmt::format("failed to size remote scatter shared state({})", errno));
        }
    } else {
        auto status = WaitUntil(
            [&] {
                descriptor = shm_open(sharedName_.c_str(), O_RDWR, 0600);
                return descriptor >= 0;
            },
            kSetupTimeout, "opening remote scatter shared state");
        if (status.Failure()) { return status; }
    }
    mapping_ = mmap(nullptr, mappingBytes_, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
    close(descriptor);
    if (mapping_ == MAP_FAILED) {
        mapping_ = nullptr;
        if (rank_ == 0) {
            shm_unlink(sharedName_.c_str());
            sharedLinked_ = false;
        }
        return Status::Error(fmt::format("failed to map remote scatter shared state({})", errno));
    }
    auto* header = static_cast<SharedHeader*>(mapping_);
    if (rank_ == 0) {
        std::memset(mapping_, 0, mappingBytes_);
        header->version = kVersion;
        header->worldSize = worldSize_;
        header->slotCount = slotCount_;
        __atomic_store_n(&header->magic, kMagic, __ATOMIC_RELEASE);
    } else {
        auto status =
            WaitUntil([&] { return __atomic_load_n(&header->magic, __ATOMIC_ACQUIRE) == kMagic; },
                      kSetupTimeout, "initializing remote scatter shared state");
        if (status.Failure()) { return status; }
    }
    if (header->version != kVersion || header->worldSize != worldSize_ ||
        header->slotCount != slotCount_) {
        return Status::InvalidParam("remote scatter shared state layout mismatch");
    }

    auto processId = platform_->IpcProcessId();
    if (!processId) { return processId.Error(); }
    auto* processIds =
        reinterpret_cast<int32_t*>(static_cast<uint8_t*>(mapping_) + layout.processIds);
    processIds[rank_] = processId.Value();

    std::vector<std::vector<uint8_t>> memoryHandles(slotCount_);
    for (size_t slot = 0; slot < slotCount_; ++slot) {
        auto memory = platform_->ExportDeviceMemory(localBuffers[slot], bufferBytes);
        if (!memory) { return memory.Error(); }
        memoryHandles[slot] = std::move(memory.Value());
        if (memoryHandles[slot].size() > kMaxHandleBytes) {
            return Status::Error("remote scatter IPC handle is too large");
        }
    }
    if (rank_ == 0) {
        __atomic_store_n(&header->memoryHandleBytes,
                         static_cast<uint32_t>(memoryHandles.front().size()), __ATOMIC_RELEASE);
    } else {
        auto status = WaitUntil(
            [&] { return __atomic_load_n(&header->memoryHandleBytes, __ATOMIC_ACQUIRE) != 0; },
            kSetupTimeout, "publishing remote scatter handle sizes");
        if (status.Failure()) { return status; }
    }
    for (size_t slot = 0; slot < slotCount_; ++slot) {
        if (memoryHandles[slot].size() != header->memoryHandleBytes) {
            return Status::InvalidParam("inconsistent remote scatter IPC handle size");
        }
        const size_t index = static_cast<size_t>(rank_) * slotCount_ + slot;
        std::memcpy(
            static_cast<uint8_t*>(mapping_) + layout.memoryHandles + index * kMaxHandleBytes,
            memoryHandles[slot].data(), memoryHandles[slot].size());
    }
    __atomic_fetch_or(&header->joinedMask, uint64_t{1} << rank_, __ATOMIC_RELEASE);
    auto status = WaitUntil([&] { return Load(&header->joinedMask) == RankMask(worldSize_); },
                            kSetupTimeout, "exchanging remote scatter IPC handles");
    if (status.Failure()) { return status; }

    std::vector<int32_t> peerProcessIds;
    peerProcessIds.reserve(worldSize_ - 1);
    for (uint32_t peer = 0; peer < worldSize_; ++peer) {
        if (peer != rank_) { peerProcessIds.push_back(processIds[peer]); }
    }
    for (const auto& handle : memoryHandles) {
        status = platform_->AuthorizeDeviceMemory(handle, peerProcessIds);
        if (status.Failure()) { return status; }
    }
    __atomic_fetch_or(&header->authorizedMask, uint64_t{1} << rank_, __ATOMIC_RELEASE);
    status = WaitUntil([&] { return Load(&header->authorizedMask) == RankMask(worldSize_); },
                       kSetupTimeout, "authorizing remote scatter IPC handles");
    if (status.Failure()) { return status; }

    peerBuffers_.assign(slotCount_, std::vector<void*>(worldSize_, nullptr));
    for (size_t slot = 0; slot < slotCount_; ++slot) {
        for (uint32_t peer = 0; peer < worldSize_; ++peer) {
            if (peer == rank_) {
                peerBuffers_[slot][peer] = localBuffers[slot];
                continue;
            }
            const size_t index = static_cast<size_t>(peer) * slotCount_ + slot;
            std::vector<uint8_t> memory(header->memoryHandleBytes);
            std::memcpy(
                memory.data(),
                static_cast<uint8_t*>(mapping_) + layout.memoryHandles + index * kMaxHandleBytes,
                memory.size());
            status = platform_->OpenDeviceMemory(memory, &peerBuffers_[slot][peer]);
            if (status.Failure()) { return status; }
        }
    }
    __atomic_fetch_or(&header->openedMask, uint64_t{1} << rank_, __ATOMIC_RELEASE);
    status = WaitUntil([&] { return Load(&header->openedMask) == RankMask(worldSize_); },
                       kSetupTimeout, "opening remote scatter IPC handles");
    if (status.Failure()) { return status; }
    if (rank_ == 0) {
        if (shm_unlink(sharedName_.c_str()) != 0) {
            return Status::Error(
                fmt::format("failed to unlink remote scatter shared state({})", errno));
        }
        sharedLinked_ = false;
        __atomic_store_n(&header->unlinked, 1U, __ATOMIC_RELEASE);
    }
    status = WaitUntil([&] { return __atomic_load_n(&header->unlinked, __ATOMIC_ACQUIRE) != 0; },
                       kSetupTimeout, "unlinking remote scatter shared state");
    return status;
}

RemoteScatterTransport::~RemoteScatterTransport()
{
    if (platform_ != nullptr) {
        for (size_t slot = 0; slot < peerBuffers_.size(); ++slot) {
            for (uint32_t peer = 0; peer < worldSize_; ++peer) {
                if (peer == rank_) { continue; }
                platform_->CloseDeviceMemory(peerBuffers_[slot][peer]);
            }
        }
    }
    if (mapping_ != nullptr) { munmap(mapping_, mappingBytes_); }
    if (rank_ == 0 && sharedLinked_) { shm_unlink(sharedName_.c_str()); }
}

Status RemoteScatterTransport::PublishReady(size_t slot, uint64_t generation, uint64_t windowTag,
                                            bool failed)
{
    if (slot >= slotCount_ || generation == 0 || generation > (UINT64_MAX >> 1)) {
        return Status::InvalidParam("invalid remote scatter ready generation");
    }
    const auto layout = SharedLayout(worldSize_, slotCount_);
    auto* windowTags =
        reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(mapping_) + layout.windowTags);
    auto* ready = reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(mapping_) + layout.ready);
    const size_t index = static_cast<size_t>(rank_) * slotCount_ + slot;
    Store(windowTags + index, windowTag);
    Store(ready + index, (generation << 1) | static_cast<uint64_t>(failed));
    return Status::OK();
}

Expected<RemoteScatterTransport::ReadyRanks> RemoteScatterTransport::WaitReady(size_t slot,
                                                                               uint64_t generation,
                                                                               uint64_t windowTag,
                                                                               uint64_t pendingMask)
{
    if (slot >= slotCount_ || generation == 0 || pendingMask == 0 ||
        (pendingMask & ~RankMask(worldSize_)) != 0) {
        return Status::InvalidParam("invalid remote scatter wait generation");
    }
    const auto layout = SharedLayout(worldSize_, slotCount_);
    auto* windowTags =
        reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(mapping_) + layout.windowTags);
    auto* ready = reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(mapping_) + layout.ready);
    const auto deadline = std::chrono::steady_clock::now() + kDataTimeout;
    for (;;) {
        ReadyRanks result;
        for (uint32_t peer = 0; peer < worldSize_; ++peer) {
            const uint64_t rankBit = uint64_t{1} << peer;
            if ((pendingMask & rankBit) == 0) { continue; }
            const size_t index = static_cast<size_t>(peer) * slotCount_ + slot;
            const uint64_t state = Load(ready + index);
            const uint64_t peerGeneration = state >> 1;
            if (peerGeneration < generation) { continue; }
            if (peerGeneration != generation) {
                return Status::StoreUnhealthy("remote scatter producer generation overrun");
            }
            if (Load(windowTags + index) != windowTag) {
                return Status::StoreUnhealthy("remote scatter window identity mismatch");
            }
            result.readyMask |= rankBit;
            if ((state & 1) != 0) { result.failedMask |= rankBit; }
        }
        if (result.readyMask != 0) { return result; }
        if (std::chrono::steady_clock::now() >= deadline) {
            return Status::StoreUnhealthy("waiting for remote scatter producer timed out");
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

void RemoteScatterTransport::PublishConsumed(size_t slot, uint64_t generation)
{
    const auto layout = SharedLayout(worldSize_, slotCount_);
    auto* consumed = reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(mapping_) + layout.consumed);
    Store(consumed + static_cast<size_t>(rank_) * slotCount_ + slot, generation);
}

Status RemoteScatterTransport::WaitConsumed(size_t slot, uint64_t generation)
{
    if (slot >= slotCount_ || generation == 0) {
        return Status::InvalidParam("invalid remote scatter consumed generation");
    }
    const auto layout = SharedLayout(worldSize_, slotCount_);
    auto* consumed = reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(mapping_) + layout.consumed);
    auto status = WaitUntil(
        [&] {
            for (uint32_t peer = 0; peer < worldSize_; ++peer) {
                if (Load(consumed + static_cast<size_t>(peer) * slotCount_ + slot) < generation) {
                    return false;
                }
            }
            return true;
        },
        kDataTimeout, "waiting for remote scatter consumers");
    return status.Success() ? status : Status::StoreUnhealthy(status.ToString());
}

}  // namespace UC::AllGatherStore
