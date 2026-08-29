#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <pthread.h>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

constexpr int kMaxTp = 16;
constexpr int kThreads = 256;
constexpr int kSkipped = 77;

enum class Mode { IpcPull, RemoteScatter, Multicast };

struct Options {
    std::string mode = "remote_scatter";
    int tp = 4;
    uint64_t blocks = 64;
    uint64_t maxTransferBlocks = 0;
    std::vector<uint64_t> tensorSizes{131072, 16384, 256};
    int warmup = 10;
    int iterations = 50;
    int scatterCtas = 0;
    uint64_t seed = 1;
    bool roundRobin = false;
    bool verify = true;
};

struct RankResult {
    double averageMs{};
    double p50Ms{};
    double p95Ms{};
    double minimumMs{};
    double maximumMs{};
    uint64_t errors{};
};

struct SharedState {
    pthread_barrier_t barrier;
    std::atomic<int> multicastUnsupported{0};
    std::array<cudaIpcMemHandle_t, kMaxTp> sendHandles{};
    std::array<cudaIpcEventHandle_t, kMaxTp> readyHandles{};
    std::array<size_t, kMaxTp> allocationGranularities{};
    size_t multicastBytes{};
    std::array<RankResult, kMaxTp> results{};
};

struct Layout {
    struct Window {
        uint64_t blockBegin{};
        uint64_t blockCount{};
        std::vector<uint64_t> ownerFirstSlots;
        std::vector<uint64_t> ownerCounts;
    };

    uint64_t shardSize{};
    std::vector<uint64_t> tensorOffsets;
    std::vector<uint32_t> owners;
    std::vector<uint64_t> ownerSlots;
    std::vector<uint64_t> ownerCounts;
    std::vector<uint64_t> ownerBases;
    std::vector<uint64_t> packedIndices;
    std::vector<Window> windows;
};

[[noreturn]] void ThrowCuda(cudaError_t code, const char* expression, const char* file, int line)
{
    std::ostringstream output;
    output << expression << " failed at " << file << ':' << line << ": " << cudaGetErrorString(code)
           << " (" << static_cast<int>(code) << ')';
    throw std::runtime_error(output.str());
}

[[noreturn]] void ThrowDriver(CUresult code, const char* expression, const char* file, int line)
{
    const char* name = nullptr;
    const char* message = nullptr;
    (void)cuGetErrorName(code, &name);
    (void)cuGetErrorString(code, &message);
    std::ostringstream output;
    output << expression << " failed at " << file << ':' << line << ": "
           << (name == nullptr ? "CUDA_ERROR" : name) << " - "
           << (message == nullptr ? "unknown" : message) << " (" << static_cast<int>(code) << ')';
    throw std::runtime_error(output.str());
}

#define CUDA_CHECK(expr)                                                         \
    do {                                                                         \
        const cudaError_t code = (expr);                                         \
        if (code != cudaSuccess) { ThrowCuda(code, #expr, __FILE__, __LINE__); } \
    } while (false)

#define CU_CHECK(expr)                                                              \
    do {                                                                            \
        const CUresult code = (expr);                                               \
        if (code != CUDA_SUCCESS) { ThrowDriver(code, #expr, __FILE__, __LINE__); } \
    } while (false)

void Barrier(SharedState* shared)
{
    const int status = pthread_barrier_wait(&shared->barrier);
    if (status != 0 && status != PTHREAD_BARRIER_SERIAL_THREAD) {
        throw std::runtime_error("pthread_barrier_wait failed");
    }
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0) { return value; }
    return (value + alignment - 1) / alignment * alignment;
}

uint64_t SplitMix64(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

uint64_t ParseBytes(std::string value)
{
    if (value.empty()) { throw std::invalid_argument("empty byte value"); }
    uint64_t multiplier = 1;
    const char suffix = value.back();
    if (suffix == 'k' || suffix == 'K') {
        multiplier = 1024;
        value.pop_back();
    } else if (suffix == 'm' || suffix == 'M') {
        multiplier = 1024 * 1024;
        value.pop_back();
    }
    const uint64_t parsed = std::stoull(value);
    if (parsed > std::numeric_limits<uint64_t>::max() / multiplier) {
        throw std::overflow_error("byte value overflow");
    }
    return parsed * multiplier;
}

std::vector<uint64_t> ParseTensorSizes(const std::string& value)
{
    std::vector<uint64_t> result;
    std::stringstream input(value);
    std::string token;
    while (std::getline(input, token, ',')) { result.push_back(ParseBytes(token)); }
    if (result.empty() ||
        std::any_of(result.begin(), result.end(), [](uint64_t size) { return size == 0; })) {
        throw std::invalid_argument("tensor sizes must be positive");
    }
    return result;
}

void Usage(const char* program)
{
    std::cout << "Usage: " << program << " [options]\n"
              << "  --mode all|ipc_pull|remote_scatter|multicast\n"
              << "  --tp N\n"
              << "  --blocks N                       (total blocks per iteration)\n"
              << "  --max-transfer-blocks N          (0 = transfer all blocks at once)\n"
              << "  --tensor-sizes BYTES[,BYTES...]  (K/M suffixes accepted)\n"
              << "  --warmup N\n"
              << "  --iters N\n"
              << "  --scatter-ctas N                (0 = 4 x SM count)\n"
              << "  --owner hash|round_robin\n"
              << "  --seed N\n"
              << "  --no-verify\n";
}

Options ParseOptions(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) { throw std::invalid_argument("missing value for " + argument); }
            return argv[i];
        };
        if (argument == "--mode") {
            options.mode = next();
        } else if (argument == "--tp") {
            options.tp = std::stoi(next());
        } else if (argument == "--blocks") {
            options.blocks = std::stoull(next());
        } else if (argument == "--max-transfer-blocks" || argument == "--window-blocks") {
            options.maxTransferBlocks = std::stoull(next());
        } else if (argument == "--tensor-sizes") {
            options.tensorSizes = ParseTensorSizes(next());
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(next());
        } else if (argument == "--iters") {
            options.iterations = std::stoi(next());
        } else if (argument == "--scatter-ctas") {
            options.scatterCtas = std::stoi(next());
        } else if (argument == "--owner") {
            const auto owner = next();
            if (owner != "hash" && owner != "round_robin") {
                throw std::invalid_argument("owner must be hash or round_robin");
            }
            options.roundRobin = owner == "round_robin";
        } else if (argument == "--seed") {
            options.seed = std::stoull(next());
        } else if (argument == "--no-verify") {
            options.verify = false;
        } else if (argument == "--help" || argument == "-h") {
            Usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (options.tp < 1 || options.tp > kMaxTp) {
        throw std::invalid_argument("tp must be in [1, 16]");
    }
    if (options.blocks == 0 || options.warmup < 0 || options.iterations < 1 ||
        options.scatterCtas < 0) {
        throw std::invalid_argument("invalid non-positive benchmark parameter");
    }
    if (options.mode != "all" && options.mode != "ipc_pull" && options.mode != "remote_scatter" &&
        options.mode != "multicast") {
        throw std::invalid_argument("invalid mode");
    }
    return options;
}

Layout BuildLayout(const Options& options)
{
    Layout layout;
    layout.tensorOffsets.reserve(options.tensorSizes.size());
    for (const auto size : options.tensorSizes) {
        layout.tensorOffsets.push_back(layout.shardSize);
        if (layout.shardSize > std::numeric_limits<uint64_t>::max() - size) {
            throw std::overflow_error("shard size overflow");
        }
        layout.shardSize += size;
    }
    layout.owners.resize(options.blocks);
    layout.ownerSlots.resize(options.blocks);
    layout.ownerCounts.assign(options.tp, 0);
    for (uint64_t block = 0; block < options.blocks; ++block) {
        const uint32_t owner =
            options.roundRobin
                ? static_cast<uint32_t>(block % options.tp)
                : static_cast<uint32_t>(SplitMix64(block ^ options.seed) % options.tp);
        layout.owners[block] = owner;
        layout.ownerSlots[block] = layout.ownerCounts[owner]++;
    }
    layout.ownerBases.resize(options.tp);
    uint64_t base = 0;
    for (int rank = 0; rank < options.tp; ++rank) {
        layout.ownerBases[rank] = base;
        base += layout.ownerCounts[rank];
    }
    layout.packedIndices.resize(options.blocks);
    for (uint64_t block = 0; block < options.blocks; ++block) {
        layout.packedIndices[block] =
            layout.ownerBases[layout.owners[block]] + layout.ownerSlots[block];
    }
    const uint64_t windowBlocks = options.maxTransferBlocks == 0
                                      ? options.blocks
                                      : std::min(options.maxTransferBlocks, options.blocks);
    for (uint64_t begin = 0; begin < options.blocks; begin += windowBlocks) {
        Layout::Window window;
        window.blockBegin = begin;
        window.blockCount = std::min(windowBlocks, options.blocks - begin);
        window.ownerFirstSlots.assign(options.tp, 0);
        window.ownerCounts.assign(options.tp, 0);
        for (uint64_t block = begin; block < begin + window.blockCount; ++block) {
            const uint32_t owner = layout.owners[block];
            if (window.ownerCounts[owner] == 0) {
                window.ownerFirstSlots[owner] = layout.ownerSlots[block];
            }
            ++window.ownerCounts[owner];
        }
        layout.windows.push_back(std::move(window));
    }
    return layout;
}

std::string ModeName(Mode mode)
{
    switch (mode) {
        case Mode::IpcPull: return "ipc_pull";
        case Mode::RemoteScatter: return "remote_scatter";
        case Mode::Multicast: return "multicast";
    }
    return "unknown";
}

__device__ uint8_t Pattern(uint64_t block, uint64_t byteOffset)
{
    return static_cast<uint8_t>((block * 131 + byteOffset * 17 + 7) & 0xff);
}

__global__ void InitializeSend(uint8_t* output, const uint64_t* slotBlocks, uint64_t slots,
                               uint64_t shardSize)
{
    const uint64_t bytes = slots * shardSize;
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < bytes;
         index += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        const uint64_t slot = index / shardSize;
        const uint64_t offset = index - slot * shardSize;
        output[index] = Pattern(slotBlocks[slot], offset);
    }
}

__device__ void CopyBytes(const uint8_t* source, uint8_t* destination, uint64_t bytes)
{
    if (((reinterpret_cast<uintptr_t>(source) | reinterpret_cast<uintptr_t>(destination) | bytes) &
         15) == 0) {
        const auto* vectorSource = reinterpret_cast<const uint4*>(source);
        auto* vectorDestination = reinterpret_cast<uint4*>(destination);
        for (uint64_t index = threadIdx.x; index < bytes / sizeof(uint4); index += blockDim.x) {
            vectorDestination[index] = vectorSource[index];
        }
        return;
    }
    for (uint64_t index = threadIdx.x; index < bytes; index += blockDim.x) {
        destination[index] = source[index];
    }
}

__global__ void ScatterPacked(const uint8_t* packed, uint8_t* const* destinations,
                              const uint64_t* packedIndices, const uint64_t* tensorOffsets,
                              const uint64_t* tensorSizes, uint64_t blockBegin, uint64_t blockCount,
                              uint32_t tensorCount, uint64_t shardSize)
{
    const uint64_t taskCount = blockCount * tensorCount;
    for (uint64_t task = blockIdx.x; task < taskCount; task += gridDim.x) {
        const uint64_t windowBlock = task / tensorCount;
        const uint64_t block = blockBegin + windowBlock;
        const uint32_t tensor = static_cast<uint32_t>(task - windowBlock * tensorCount);
        const auto* source = packed + packedIndices[block] * shardSize + tensorOffsets[tensor];
        auto* destination = destinations[tensor] + block * tensorSizes[tensor];
        CopyBytes(source, destination, tensorSizes[tensor]);
    }
}

__global__ void ScatterRemote(uint8_t* const* sources, uint8_t* const* destinations,
                              const uint32_t* owners, const uint64_t* ownerSlots,
                              const uint64_t* tensorOffsets, const uint64_t* tensorSizes,
                              uint64_t blockBegin, uint64_t blockCount, uint32_t tensorCount,
                              uint64_t shardSize)
{
    const uint64_t taskCount = blockCount * tensorCount;
    for (uint64_t task = blockIdx.x; task < taskCount; task += gridDim.x) {
        const uint64_t windowBlock = task / tensorCount;
        const uint64_t block = blockBegin + windowBlock;
        const uint32_t tensor = static_cast<uint32_t>(task - windowBlock * tensorCount);
        const auto* source =
            sources[owners[block]] + ownerSlots[block] * shardSize + tensorOffsets[tensor];
        auto* destination = destinations[tensor] + block * tensorSizes[tensor];
        CopyBytes(source, destination, tensorSizes[tensor]);
    }
}

__global__ void VerifyDestinations(uint8_t* const* destinations, const uint64_t* tensorOffsets,
                                   const uint64_t* tensorSizes, uint64_t blocks,
                                   uint32_t tensorCount, unsigned long long* errors)
{
    const uint64_t samples = blocks * tensorCount * 3;
    for (uint64_t sample = blockIdx.x * blockDim.x + threadIdx.x; sample < samples;
         sample += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        const uint64_t item = sample / 3;
        const uint32_t point = static_cast<uint32_t>(sample - item * 3);
        const uint64_t block = item / tensorCount;
        const uint32_t tensor = static_cast<uint32_t>(item - block * tensorCount);
        const uint64_t size = tensorSizes[tensor];
        const uint64_t offset = point == 0 ? 0 : (point == 1 ? size / 2 : size - 1);
        const uint8_t actual = destinations[tensor][block * size + offset];
        const uint8_t expected = Pattern(block, tensorOffsets[tensor] + offset);
        if (actual != expected) { atomicAdd(errors, 1ULL); }
    }
}

#if CUDA_VERSION >= 12010
__global__ void MulticastBroadcast(const uint64_t* source, uint64_t* multicastDestination,
                                   uint64_t words)
{
#if __CUDA_ARCH__ >= 900
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < words;
         index += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        const uint64_t value = source[index];
        auto* destination = multicastDestination + index;
        asm volatile("multimem.st.relaxed.sys.global.u64 [%0], %1;"
                     :
                     : "l"(destination), "l"(value)
                     : "memory");
    }
#endif
}
#endif

template <typename T>
T* CopyVectorToDevice(const std::vector<T>& values)
{
    T* result = nullptr;
    CUDA_CHECK(cudaMalloc(&result, std::max<size_t>(values.size() * sizeof(T), 1)));
    if (!values.empty()) {
        CUDA_CHECK(
            cudaMemcpy(result, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    return result;
}

int ScatterCtas(const Options& options, uint64_t tasks)
{
    if (options.scatterCtas > 0) {
        return static_cast<int>(std::min<uint64_t>(options.scatterCtas, tasks));
    }
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    return static_cast<int>(std::max<uint64_t>(1, std::min<uint64_t>(sms * 4, tasks)));
}

void SendFd(int socket, int fd)
{
    char byte = 0;
    iovec io{&byte, 1};
    std::array<char, CMSG_SPACE(sizeof(int))> control{};
    msghdr message{};
    message.msg_iov = &io;
    message.msg_iovlen = 1;
    message.msg_control = control.data();
    message.msg_controllen = control.size();
    cmsghdr* header = CMSG_FIRSTHDR(&message);
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_RIGHTS;
    header->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(header), &fd, sizeof(fd));
    if (sendmsg(socket, &message, 0) != 1) {
        throw std::runtime_error("sendmsg failed: " + std::string(std::strerror(errno)));
    }
}

int ReceiveFd(int socket)
{
    char byte = 0;
    iovec io{&byte, 1};
    std::array<char, CMSG_SPACE(sizeof(int))> control{};
    msghdr message{};
    message.msg_iov = &io;
    message.msg_iovlen = 1;
    message.msg_control = control.data();
    message.msg_controllen = control.size();
    if (recvmsg(socket, &message, 0) != 1) {
        throw std::runtime_error("recvmsg failed: " + std::string(std::strerror(errno)));
    }
    cmsghdr* header = CMSG_FIRSTHDR(&message);
    if (header == nullptr || header->cmsg_level != SOL_SOCKET || header->cmsg_type != SCM_RIGHTS) {
        throw std::runtime_error("received message has no file descriptor");
    }
    int fd = -1;
    std::memcpy(&fd, CMSG_DATA(header), sizeof(fd));
    return fd;
}

class RankBuffers {
public:
    RankBuffers(int rank, const Options& options, const Layout& layout, SharedState* shared,
                bool openPeerMemory)
        : rank_(rank),
          options_(options),
          layout_(layout),
          shared_(shared),
          openPeerMemory_(openPeerMemory)
    {
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (rank_ >= deviceCount) {
            throw std::runtime_error("rank exceeds visible CUDA device count");
        }
        CUDA_CHECK(cudaSetDevice(rank_));
        CUDA_CHECK(cudaFree(nullptr));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CUDA_CHECK(
            cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming | cudaEventInterprocess));

        const uint64_t sendBytes = layout_.ownerCounts[rank_] * layout_.shardSize;
        CUDA_CHECK(cudaMalloc(&send_, std::max<uint64_t>(sendBytes, 1)));
        CUDA_CHECK(cudaIpcGetMemHandle(&shared_->sendHandles[rank_], send_));
        CUDA_CHECK(cudaIpcGetEventHandle(&shared_->readyHandles[rank_], ready_));

        std::vector<uint64_t> slotBlocks;
        slotBlocks.reserve(layout_.ownerCounts[rank_]);
        for (uint64_t block = 0; block < options_.blocks; ++block) {
            if (layout_.owners[block] == static_cast<uint32_t>(rank_)) {
                slotBlocks.push_back(block);
            }
        }
        uint64_t* deviceSlotBlocks = CopyVectorToDevice(slotBlocks);
        if (sendBytes > 0) {
            InitializeSend<<<std::min<uint64_t>(4096, (sendBytes + kThreads - 1) / kThreads),
                             kThreads, 0, stream_>>>(send_, deviceSlotBlocks, slotBlocks.size(),
                                                     layout_.shardSize);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaFree(deviceSlotBlocks));

        destinations_.resize(options_.tensorSizes.size());
        for (size_t tensor = 0; tensor < options_.tensorSizes.size(); ++tensor) {
            CUDA_CHECK(
                cudaMalloc(&destinations_[tensor], options_.blocks * options_.tensorSizes[tensor]));
            CUDA_CHECK(cudaMemsetAsync(destinations_[tensor], 0,
                                       options_.blocks * options_.tensorSizes[tensor], stream_));
        }
        deviceDestinations_ = CopyVectorToDevice(destinations_);
        deviceOwners_ = CopyVectorToDevice(layout_.owners);
        deviceOwnerSlots_ = CopyVectorToDevice(layout_.ownerSlots);
        devicePackedIndices_ = CopyVectorToDevice(layout_.packedIndices);
        deviceTensorOffsets_ = CopyVectorToDevice(layout_.tensorOffsets);
        deviceTensorSizes_ = CopyVectorToDevice(options_.tensorSizes);
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        Barrier(shared_);
        peerSends_.resize(options_.tp);
        peerReady_.resize(options_.tp);
        for (int peer = 0; peer < options_.tp; ++peer) {
            if (peer == rank_) {
                peerSends_[peer] = send_;
                peerReady_[peer] = ready_;
                continue;
            }
            if (openPeerMemory_) {
                int accessible = 0;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&accessible, rank_, peer));
                if (!accessible) {
                    throw std::runtime_error("CUDA peer access is unavailable from rank " +
                                             std::to_string(rank_) + " to rank " +
                                             std::to_string(peer));
                }
                CUDA_CHECK(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&peerSends_[peer]),
                                                shared_->sendHandles[peer],
                                                cudaIpcMemLazyEnablePeerAccess));
            }
            CUDA_CHECK(cudaIpcOpenEventHandle(&peerReady_[peer], shared_->readyHandles[peer]));
        }
        devicePeerSends_ = CopyVectorToDevice(peerSends_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        Barrier(shared_);
    }

    ~RankBuffers()
    {
        (void)cudaSetDevice(rank_);
        for (int peer = 0; peer < options_.tp; ++peer) {
            if (peer != rank_ && peerSends_[peer] != nullptr) {
                (void)cudaIpcCloseMemHandle(peerSends_[peer]);
            }
            if (peer != rank_ && peerReady_[peer] != nullptr) {
                (void)cudaEventDestroy(peerReady_[peer]);
            }
        }
        for (auto* destination : destinations_) { (void)cudaFree(destination); }
        for (auto* pointer :
             {reinterpret_cast<void*>(deviceDestinations_),
              reinterpret_cast<void*>(devicePeerSends_), reinterpret_cast<void*>(deviceOwners_),
              reinterpret_cast<void*>(deviceOwnerSlots_),
              reinterpret_cast<void*>(devicePackedIndices_),
              reinterpret_cast<void*>(deviceTensorOffsets_),
              reinterpret_cast<void*>(deviceTensorSizes_)}) {
            if (pointer != nullptr) { (void)cudaFree(pointer); }
        }
        if (send_ != nullptr) { (void)cudaFree(send_); }
        if (ready_ != nullptr) { (void)cudaEventDestroy(ready_); }
        if (stream_ != nullptr) { (void)cudaStreamDestroy(stream_); }
    }

    void RecordReady() { CUDA_CHECK(cudaEventRecord(ready_, stream_)); }

    void WaitAllReady()
    {
        for (int peer = 0; peer < options_.tp; ++peer) {
            CUDA_CHECK(cudaStreamWaitEvent(stream_, peerReady_[peer], 0));
        }
    }

    void ScatterPackedBuffer(const uint8_t* packed, const Layout::Window& window)
    {
        const uint64_t tasks = window.blockCount * options_.tensorSizes.size();
        ScatterPacked<<<ScatterCtas(options_, tasks), kThreads, 0, stream_>>>(
            packed, deviceDestinations_, devicePackedIndices_, deviceTensorOffsets_,
            deviceTensorSizes_, window.blockBegin, window.blockCount, options_.tensorSizes.size(),
            layout_.shardSize);
        CUDA_CHECK(cudaGetLastError());
    }

    void ScatterRemoteBuffers(const Layout::Window& window)
    {
        const uint64_t tasks = window.blockCount * options_.tensorSizes.size();
        ScatterRemote<<<ScatterCtas(options_, tasks), kThreads, 0, stream_>>>(
            devicePeerSends_, deviceDestinations_, deviceOwners_, deviceOwnerSlots_,
            deviceTensorOffsets_, deviceTensorSizes_, window.blockBegin, window.blockCount,
            options_.tensorSizes.size(), layout_.shardSize);
        CUDA_CHECK(cudaGetLastError());
    }

    uint64_t Verify()
    {
        if (!options_.verify) { return 0; }
        unsigned long long* deviceErrors = nullptr;
        CUDA_CHECK(cudaMalloc(&deviceErrors, sizeof(*deviceErrors)));
        CUDA_CHECK(cudaMemsetAsync(deviceErrors, 0, sizeof(*deviceErrors), stream_));
        const uint64_t samples = options_.blocks * options_.tensorSizes.size() * 3;
        VerifyDestinations<<<std::min<uint64_t>(1024, (samples + kThreads - 1) / kThreads),
                             kThreads, 0, stream_>>>(deviceDestinations_, deviceTensorOffsets_,
                                                     deviceTensorSizes_, options_.blocks,
                                                     options_.tensorSizes.size(), deviceErrors);
        CUDA_CHECK(cudaGetLastError());
        unsigned long long errors = 0;
        CUDA_CHECK(cudaMemcpyAsync(&errors, deviceErrors, sizeof(errors), cudaMemcpyDeviceToHost,
                                   stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK(cudaFree(deviceErrors));
        return errors;
    }

    cudaStream_t Stream() const { return stream_; }
    uint8_t* Send() const { return send_; }
    const std::vector<uint8_t*>& PeerSends() const { return peerSends_; }

private:
    int rank_;
    const Options& options_;
    const Layout& layout_;
    SharedState* shared_;
    bool openPeerMemory_;
    cudaStream_t stream_{};
    cudaEvent_t ready_{};
    uint8_t* send_{};
    std::vector<uint8_t*> destinations_;
    std::vector<uint8_t*> peerSends_;
    std::vector<cudaEvent_t> peerReady_;
    uint8_t** deviceDestinations_{};
    uint8_t** devicePeerSends_{};
    uint32_t* deviceOwners_{};
    uint64_t* deviceOwnerSlots_{};
    uint64_t* devicePackedIndices_{};
    uint64_t* deviceTensorOffsets_{};
    uint64_t* deviceTensorSizes_{};
};

#if CUDA_VERSION >= 12010
class MulticastAllocation {
public:
    MulticastAllocation(int rank, const Options& options, const Layout& layout, SharedState* shared,
                        const std::vector<std::array<int, 2>>& sockets)
        : rank_(rank), options_(options), shared_(shared)
    {
        CU_CHECK(cuInit(0));
        CU_CHECK(cuDeviceGet(&device_, rank_));
        int supported = 0;
        CU_CHECK(
            cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device_));
        if (!supported) { shared_->multicastUnsupported.store(1); }

        CUmemAllocationProp allocationProp{};
        allocationProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        allocationProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        allocationProp.location.id = device_;
        allocationProp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        size_t allocationGranularity = 0;
        CU_CHECK(cuMemGetAllocationGranularity(&allocationGranularity, &allocationProp,
                                               CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        shared_->allocationGranularities[rank_] = allocationGranularity;
        Barrier(shared_);
        if (shared_->multicastUnsupported.load() != 0) { return; }

        if (rank_ == 0) {
            CUmulticastObjectProp multicastProp{};
            multicastProp.numDevices = options_.tp;
            multicastProp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
            multicastProp.size = options_.blocks * layout.shardSize;
            size_t multicastGranularity = 0;
            CU_CHECK(cuMulticastGetGranularity(&multicastGranularity, &multicastProp,
                                               CU_MULTICAST_GRANULARITY_RECOMMENDED));
            uint64_t alignment = multicastGranularity;
            for (int peer = 0; peer < options_.tp; ++peer) {
                alignment = std::lcm<uint64_t>(alignment, shared_->allocationGranularities[peer]);
            }
            shared_->multicastBytes = AlignUp(options_.blocks * layout.shardSize, alignment);
            multicastProp.size = shared_->multicastBytes;
            CU_CHECK(cuMulticastCreate(&multicastHandle_, &multicastProp));
            int exported = -1;
            CU_CHECK(cuMemExportToShareableHandle(&exported, multicastHandle_,
                                                  CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
            for (int peer = 1; peer < options_.tp; ++peer) { SendFd(sockets[peer][0], exported); }
            close(exported);
        } else {
            const int imported = ReceiveFd(sockets[rank_][1]);
            CU_CHECK(cuMemImportFromShareableHandle(
                &multicastHandle_, reinterpret_cast<void*>(static_cast<intptr_t>(imported)),
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
            close(imported);
        }

        CU_CHECK(cuMulticastAddDevice(multicastHandle_, device_));
        Barrier(shared_);

        allocationProp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        CU_CHECK(cuMemCreate(&localHandle_, shared_->multicastBytes, &allocationProp, 0));
        CU_CHECK(
            cuMulticastBindMem(multicastHandle_, 0, localHandle_, 0, shared_->multicastBytes, 0));

        CU_CHECK(cuMemAddressReserve(&localAddress_, shared_->multicastBytes, 0, 0, 0));
        CU_CHECK(cuMemMap(localAddress_, shared_->multicastBytes, 0, localHandle_, 0));
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device_;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_CHECK(cuMemSetAccess(localAddress_, shared_->multicastBytes, &access, 1));

        CU_CHECK(cuMemAddressReserve(&multicastAddress_, shared_->multicastBytes, 0, 0, 0));
        CU_CHECK(cuMemMap(multicastAddress_, shared_->multicastBytes, 0, multicastHandle_, 0));
        CU_CHECK(cuMemSetAccess(multicastAddress_, shared_->multicastBytes, &access, 1));
        active_ = true;
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(localAddress_), 0, shared_->multicastBytes));
        Barrier(shared_);
    }

    ~MulticastAllocation()
    {
        if (!active_) { return; }
        (void)cudaSetDevice(rank_);
        (void)cuMemUnmap(multicastAddress_, shared_->multicastBytes);
        (void)cuMemAddressFree(multicastAddress_, shared_->multicastBytes);
        (void)cuMemUnmap(localAddress_, shared_->multicastBytes);
        (void)cuMemAddressFree(localAddress_, shared_->multicastBytes);
        (void)cuMulticastUnbind(multicastHandle_, device_, 0, shared_->multicastBytes);
        (void)cuMemRelease(localHandle_);
        (void)cuMemRelease(multicastHandle_);
    }

    bool Active() const { return active_; }
    uint8_t* Local() const { return reinterpret_cast<uint8_t*>(localAddress_); }
    uint8_t* Multicast() const { return reinterpret_cast<uint8_t*>(multicastAddress_); }

private:
    int rank_;
    const Options& options_;
    SharedState* shared_;
    CUdevice device_{};
    CUmemGenericAllocationHandle multicastHandle_{};
    CUmemGenericAllocationHandle localHandle_{};
    CUdeviceptr localAddress_{};
    CUdeviceptr multicastAddress_{};
    bool active_{};
};
#endif

RankResult Summarize(std::vector<double> samples, uint64_t errors)
{
    std::sort(samples.begin(), samples.end());
    auto percentile = [&](double value) {
        const size_t index =
            static_cast<size_t>(std::ceil(value * static_cast<double>(samples.size() - 1)));
        return samples[index];
    };
    RankResult result;
    result.averageMs = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    result.p50Ms = percentile(0.50);
    result.p95Ms = percentile(0.95);
    result.minimumMs = samples.front();
    result.maximumMs = samples.back();
    result.errors = errors;
    return result;
}

int RunRank(Mode mode, int rank, const Options& options, const Layout& layout, SharedState* shared,
            const std::vector<std::array<int, 2>>& sockets)
{
    RankBuffers buffers(rank, options, layout, shared, mode != Mode::Multicast);
    uint8_t* packedReceive = nullptr;
#if CUDA_VERSION >= 12010
    std::unique_ptr<MulticastAllocation> multicast;
#endif
    if (mode == Mode::IpcPull) {
        CUDA_CHECK(cudaMalloc(&packedReceive, options.blocks * layout.shardSize));
    } else if (mode == Mode::Multicast) {
#if CUDA_VERSION >= 12010
        multicast = std::make_unique<MulticastAllocation>(rank, options, layout, shared, sockets);
        if (!multicast->Active()) {
            Barrier(shared);
            return kSkipped;
        }
#else
        shared->multicastUnsupported.store(1);
        Barrier(shared);
        return kSkipped;
#endif
    }

    std::vector<double> samples;
    samples.reserve(options.iterations);
    const int totalIterations = options.warmup + options.iterations;
    for (int iteration = 0; iteration < totalIterations; ++iteration) {
        Barrier(shared);
        const auto started = std::chrono::steady_clock::now();
        for (const auto& window : layout.windows) {
            if (mode == Mode::Multicast) {
#if CUDA_VERSION >= 12010
                const uint64_t bytes = window.ownerCounts[rank] * layout.shardSize;
                if ((bytes & 7) != 0) {
                    throw std::runtime_error("multicast owner payload must be 8-byte aligned");
                }
                if (bytes > 0) {
                    const int ctas =
                        std::min<uint64_t>(4096, (bytes / 8 + kThreads - 1) / kThreads);
                    const uint64_t ownerOffset = window.ownerFirstSlots[rank] * layout.shardSize;
                    const uint64_t packedOffset =
                        (layout.ownerBases[rank] + window.ownerFirstSlots[rank]) * layout.shardSize;
                    MulticastBroadcast<<<ctas, kThreads, 0, buffers.Stream()>>>(
                        reinterpret_cast<const uint64_t*>(buffers.Send() + ownerOffset),
                        reinterpret_cast<uint64_t*>(multicast->Multicast() + packedOffset),
                        bytes / 8);
                    CUDA_CHECK(cudaGetLastError());
                }
#endif
                buffers.RecordReady();
                Barrier(shared);
                buffers.WaitAllReady();
            }

            if (mode == Mode::IpcPull) {
                for (int owner = 0; owner < options.tp; ++owner) {
                    const uint64_t bytes = window.ownerCounts[owner] * layout.shardSize;
                    if (bytes == 0) { continue; }
                    const uint64_t ownerOffset = window.ownerFirstSlots[owner] * layout.shardSize;
                    const uint64_t packedOffset =
                        (layout.ownerBases[owner] + window.ownerFirstSlots[owner]) *
                        layout.shardSize;
                    auto* destination = packedReceive + packedOffset;
                    const auto* source = buffers.PeerSends()[owner] + ownerOffset;
                    if (owner == rank) {
                        CUDA_CHECK(cudaMemcpyAsync(destination, source, bytes,
                                                   cudaMemcpyDeviceToDevice, buffers.Stream()));
                    } else {
                        CUDA_CHECK(cudaMemcpyPeerAsync(destination, rank, source, owner, bytes,
                                                       buffers.Stream()));
                    }
                }
                buffers.ScatterPackedBuffer(packedReceive, window);
            } else if (mode == Mode::RemoteScatter) {
                buffers.ScatterRemoteBuffers(window);
            } else {
#if CUDA_VERSION >= 12010
                buffers.ScatterPackedBuffer(multicast->Local(), window);
#endif
            }
            if (mode == Mode::Multicast) { Barrier(shared); }
        }
        CUDA_CHECK(cudaStreamSynchronize(buffers.Stream()));
        const auto finished = std::chrono::steady_clock::now();
        Barrier(shared);
        if (iteration >= options.warmup) {
            samples.push_back(
                std::chrono::duration<double, std::milli>(finished - started).count());
        }
    }

    const uint64_t errors = buffers.Verify();
    shared->results[rank] = Summarize(std::move(samples), errors);
    Barrier(shared);
    if (packedReceive != nullptr) { CUDA_CHECK(cudaFree(packedReceive)); }
    return 0;
}

SharedState* CreateSharedState(int tp)
{
    void* memory = mmap(nullptr, sizeof(SharedState), PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) {
        throw std::runtime_error("mmap failed: " + std::string(std::strerror(errno)));
    }
    auto* shared = new (memory) SharedState;
    pthread_barrierattr_t attributes;
    if (pthread_barrierattr_init(&attributes) != 0 ||
        pthread_barrierattr_setpshared(&attributes, PTHREAD_PROCESS_SHARED) != 0 ||
        pthread_barrier_init(&shared->barrier, &attributes, tp) != 0) {
        throw std::runtime_error("failed to initialize process-shared barrier");
    }
    pthread_barrierattr_destroy(&attributes);
    return shared;
}

void DestroySharedState(SharedState* shared)
{
    pthread_barrier_destroy(&shared->barrier);
    shared->~SharedState();
    munmap(shared, sizeof(SharedState));
}

void PrintResult(Mode mode, const Options& options, const Layout& layout, const SharedState* shared)
{
    double criticalAverage = 0;
    double criticalP50 = 0;
    double criticalP95 = 0;
    double minimum = std::numeric_limits<double>::max();
    double maximum = 0;
    uint64_t errors = 0;
    for (int rank = 0; rank < options.tp; ++rank) {
        const auto& result = shared->results[rank];
        criticalAverage = std::max(criticalAverage, result.averageMs);
        criticalP50 = std::max(criticalP50, result.p50Ms);
        criticalP95 = std::max(criticalP95, result.p95Ms);
        minimum = std::min(minimum, result.minimumMs);
        maximum = std::max(maximum, result.maximumMs);
        errors += result.errors;
    }
    const double uniqueBytes = static_cast<double>(options.blocks * layout.shardSize);
    const double uniqueGbps = uniqueBytes / criticalAverage / 1.0e6;
    const double deliveredGbps = uniqueGbps * options.tp;
    std::cout << std::fixed << std::setprecision(3) << "mode=" << ModeName(mode)
              << " critical_avg_ms=" << criticalAverage << " critical_p50_ms=" << criticalP50
              << " critical_p95_ms=" << criticalP95 << " min_rank_iter_ms=" << minimum
              << " max_rank_iter_ms=" << maximum << " unique_GBps=" << uniqueGbps
              << " delivered_GBps=" << deliveredGbps << " verify_errors=" << errors << '\n';
}

int RunMode(Mode mode, const Options& options, const Layout& layout)
{
    std::cout.flush();
    std::cerr.flush();
    SharedState* shared = CreateSharedState(options.tp);
    std::vector<std::array<int, 2>> sockets(options.tp, {-1, -1});
    if (mode == Mode::Multicast) {
        for (int rank = 1; rank < options.tp; ++rank) {
            if (socketpair(AF_UNIX, SOCK_DGRAM, 0, sockets[rank].data()) != 0) {
                throw std::runtime_error("socketpair failed: " + std::string(std::strerror(errno)));
            }
        }
    }

    std::vector<pid_t> children;
    for (int rank = 0; rank < options.tp; ++rank) {
        const pid_t child = fork();
        if (child < 0) { throw std::runtime_error("fork failed"); }
        if (child == 0) {
            try {
                const int status = RunRank(mode, rank, options, layout, shared, sockets);
                std::cout.flush();
                std::cerr.flush();
                _exit(status);
            } catch (const std::exception& error) {
                std::cerr << "rank=" << rank << " mode=" << ModeName(mode)
                          << " failed: " << error.what() << std::endl;
                _exit(2);
            }
        }
        children.push_back(child);
    }

    bool failed = false;
    bool skipped = false;
    size_t remaining = children.size();
    while (remaining > 0) {
        int status = 0;
        const pid_t child = waitpid(-1, &status, 0);
        if (child < 0) {
            if (errno == EINTR) { continue; }
            failed = true;
            break;
        }
        --remaining;
        const int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128;
        if (code == kSkipped) {
            skipped = true;
        } else if (code != 0) {
            failed = true;
            for (const pid_t peer : children) {
                if (peer != child) { kill(peer, SIGKILL); }
            }
        }
    }
    while (waitpid(-1, nullptr, WNOHANG) > 0) {}
    for (auto& pair : sockets) {
        for (const int socket : pair) {
            if (socket >= 0) { close(socket); }
        }
    }

    int result = 0;
    if (failed) {
        result = 1;
    } else if (skipped) {
        std::cout << "mode=multicast skipped: CUDA multicast is unavailable in this build or "
                     "on at least one selected GPU\n";
    } else {
        PrintResult(mode, options, layout, shared);
        for (int rank = 0; rank < options.tp; ++rank) {
            if (shared->results[rank].errors != 0) { result = 1; }
        }
    }
    DestroySharedState(shared);
    return result;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = ParseOptions(argc, argv);
        const Layout layout = BuildLayout(options);
        std::cout << "tp=" << options.tp << " blocks=" << options.blocks << " max_transfer_blocks="
                  << (options.maxTransferBlocks == 0 ? options.blocks : options.maxTransferBlocks)
                  << " windows=" << layout.windows.size()
                  << " tensors=" << options.tensorSizes.size()
                  << " shard_bytes=" << layout.shardSize << " owner_counts=";
        for (int rank = 0; rank < options.tp; ++rank) {
            if (rank != 0) { std::cout << ','; }
            std::cout << layout.ownerCounts[rank];
        }
        std::cout << " warmup=" << options.warmup << " iters=" << options.iterations << '\n';

        std::vector<Mode> modes;
        if (options.mode == "all" || options.mode == "ipc_pull") { modes.push_back(Mode::IpcPull); }
        if (options.mode == "all" || options.mode == "remote_scatter") {
            modes.push_back(Mode::RemoteScatter);
        }
        if (options.mode == "all" || options.mode == "multicast") {
            modes.push_back(Mode::Multicast);
        }
        int result = 0;
        for (const auto mode : modes) { result |= RunMode(mode, options, layout); }
        return result;
    } catch (const std::exception& error) {
        std::cerr << "cuda_replication_scatter_bench failed: " << error.what() << '\n';
        return 1;
    }
}
