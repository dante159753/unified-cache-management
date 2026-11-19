#include <acl/acl.h>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <linux/mempolicy.h>
#include <numeric>
#include <pthread.h>
#include <random>
#include <sched.h>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

enum class Mode { SharedSequential, ShardedSequential, ShardedRandom };

struct Options {
    size_t tp = 1;
    std::vector<int> devices;
    std::vector<int> numaNodes;
    int sharedNumaNode = -2;
    size_t blockSize = 1024 * 1024;
    size_t blockCount = 10240;
    size_t deviceBufferSize = 0;
    size_t warmup = 3;
    size_t iterations = 10;
    uint64_t seed = 20260901;
    std::string mode = "all";
};

struct Stats {
    double average = 0.0;
    double minimum = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p99 = 0.0;
    double maximum = 0.0;
};

struct SharedControl {
    pthread_barrier_t barrier;
    double elapsedUs[1];
};

[[noreturn]] void Fail(const std::string& message) { throw std::runtime_error(message); }

void CheckAcl(aclError result, const std::string& operation)
{
    if (result == ACL_SUCCESS) { return; }
    std::ostringstream os;
    os << operation << " failed, ret=" << result;
    const char* recent = aclGetRecentErrMsg();
    if (recent != nullptr) { os << ", msg=" << recent; }
    Fail(os.str());
}

size_t AlignUp(size_t value, size_t alignment)
{
    if (value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        Fail("size alignment overflow");
    }
    return (value + alignment - 1) / alignment * alignment;
}

size_t ParseSize(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    size_t pos = 0;
    const double number = std::stod(value, &pos);
    const auto suffix = value.substr(pos);
    double scale = 1.0;
    if (suffix.empty() || suffix == "b") {
        scale = 1.0;
    } else if (suffix == "k" || suffix == "kb" || suffix == "kib") {
        scale = 1024.0;
    } else if (suffix == "m" || suffix == "mb" || suffix == "mib") {
        scale = 1024.0 * 1024.0;
    } else if (suffix == "g" || suffix == "gb" || suffix == "gib") {
        scale = 1024.0 * 1024.0 * 1024.0;
    } else {
        Fail("invalid size suffix: " + suffix);
    }
    if (number <= 0 || number * scale > static_cast<double>(std::numeric_limits<size_t>::max())) {
        Fail("size must be positive and fit in size_t");
    }
    return static_cast<size_t>(number * scale);
}

size_t ParseCount(const std::string& value, const std::string& name, bool allowZero = false)
{
    size_t pos = 0;
    const auto result = std::stoull(value, &pos);
    if (pos != value.size() || (!allowZero && result == 0) ||
        result > std::numeric_limits<size_t>::max()) {
        Fail("invalid " + name + ": " + value);
    }
    return static_cast<size_t>(result);
}

int ParseInt(const std::string& value, const std::string& name, bool allowNegative = false)
{
    size_t pos = 0;
    const auto result = std::stoll(value, &pos);
    if (pos != value.size() || (!allowNegative && result < 0) ||
        result < std::numeric_limits<int>::min() || result > std::numeric_limits<int>::max()) {
        Fail("invalid " + name + ": " + value);
    }
    return static_cast<int>(result);
}

std::string Trim(std::string value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) { return {}; }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<int> ParseIntList(const std::string& value, const std::string& name)
{
    std::vector<int> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = Trim(std::move(item));
        if (item.empty()) { Fail("empty item in " + name); }
        result.push_back(ParseInt(item, name));
    }
    if (result.empty()) { Fail(name + " must not be empty"); }
    return result;
}

std::vector<int> ParseRangeList(const std::string& value)
{
    std::vector<int> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = Trim(std::move(item));
        const auto dash = item.find('-');
        if (dash == std::string::npos) {
            result.push_back(ParseInt(item, "range"));
            continue;
        }
        const int first = ParseInt(item.substr(0, dash), "range");
        const int last = ParseInt(item.substr(dash + 1), "range");
        if (last < first) { Fail("descending range is not supported: " + item); }
        for (int current = first;; ++current) {
            result.push_back(current);
            if (current == last) { break; }
        }
    }
    return result;
}

std::string ReadTextFile(const std::string& path)
{
    std::ifstream input(path);
    if (!input) { Fail("cannot read " + path); }
    std::ostringstream output;
    output << input.rdbuf();
    return output.str();
}

std::vector<int> OnlineNumaNodes()
{
    try {
        auto nodes = ParseRangeList(ReadTextFile("/sys/devices/system/node/online"));
        if (!nodes.empty()) { return nodes; }
    } catch (const std::exception&) {
    }
    return {0};
}

void PrintUsage(const char* program)
{
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --tp N                         Number of ranks/devices\n"
        << "  --devices D0,D1,...            Device ids; default 0..TP-1\n"
        << "  --block-size SIZE              Bytes per H2D copy, suffixes K/M/G allowed\n"
        << "  --blocks N                     Total logical block count\n"
        << "  --device-buffer-size SIZE      Reusable HBM destination; default full data size\n"
        << "  --numa-nodes N0,N1,...         Owner NUMA node per rank\n"
        << "  --shared-numa-node N           NUMA node for the single shared segment\n"
        << "  --mode all|shared-sequential|sharded-sequential|sharded-random\n"
        << "  --warmup N                     Warmup iterations\n"
        << "  --iters N                      Measured iterations\n"
        << "  --seed N                       Base seed for independent random orders\n";
}

Options ParseOptions(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) { Fail("missing value for " + arg); }
        const std::string value = argv[++i];
        if (arg == "--tp") {
            options.tp = ParseCount(value, "tp");
        } else if (arg == "--devices") {
            options.devices = ParseIntList(value, "devices");
        } else if (arg == "--block-size") {
            options.blockSize = ParseSize(value);
        } else if (arg == "--blocks") {
            options.blockCount = ParseCount(value, "blocks");
        } else if (arg == "--device-buffer-size") {
            options.deviceBufferSize = ParseSize(value);
        } else if (arg == "--numa-nodes") {
            options.numaNodes = ParseIntList(value, "numa-nodes");
        } else if (arg == "--shared-numa-node") {
            options.sharedNumaNode = ParseInt(value, "shared-numa-node", true);
        } else if (arg == "--mode") {
            options.mode = value;
        } else if (arg == "--warmup") {
            options.warmup = ParseCount(value, "warmup", true);
        } else if (arg == "--iters") {
            options.iterations = ParseCount(value, "iters");
        } else if (arg == "--seed") {
            options.seed = std::stoull(value);
        } else {
            Fail("unknown option: " + arg);
        }
    }

    if (options.tp > 64) { Fail("tp must not exceed 64"); }
    if (options.blockCount < options.tp) { Fail("blocks must be at least tp"); }
    if (options.blockSize > std::numeric_limits<size_t>::max() / options.blockCount) {
        Fail("block-size * blocks overflows size_t");
    }
    if (options.deviceBufferSize != 0 && options.deviceBufferSize < options.blockSize) {
        Fail("device-buffer-size must be at least one block");
    }
    if (options.devices.empty()) {
        options.devices.resize(options.tp);
        std::iota(options.devices.begin(), options.devices.end(), 0);
    } else if (options.devices.size() != options.tp) {
        Fail("devices count must equal tp");
    }
    if (options.numaNodes.empty()) {
        const auto online = OnlineNumaNodes();
        options.numaNodes.resize(options.tp);
        for (size_t rank = 0; rank < options.tp; ++rank) {
            options.numaNodes[rank] = online[rank % online.size()];
        }
    } else if (options.numaNodes.size() == 1) {
        options.numaNodes.resize(options.tp, options.numaNodes.front());
    } else if (options.numaNodes.size() != options.tp) {
        Fail("numa-nodes must contain one node or tp nodes");
    }
    if (options.sharedNumaNode == -2) { options.sharedNumaNode = options.numaNodes.front(); }
    const std::vector<std::string> modes{"all", "shared-sequential", "sharded-sequential",
                                         "sharded-random"};
    if (std::find(modes.begin(), modes.end(), options.mode) == modes.end()) {
        Fail("unknown mode: " + options.mode);
    }
    return options;
}

std::vector<unsigned long> NodeMask(int node)
{
    constexpr size_t bits = sizeof(unsigned long) * 8;
    std::vector<unsigned long> mask(static_cast<size_t>(node) / bits + 1, 0);
    mask[static_cast<size_t>(node) / bits] |= 1UL << (static_cast<size_t>(node) % bits);
    return mask;
}

void BindMapping(void* address, size_t bytes, int node)
{
    if (node < 0) { return; }
    auto mask = NodeMask(node);
    const auto maxNode = mask.size() * sizeof(unsigned long) * 8;
    const auto flags = static_cast<unsigned long>(MPOL_MF_MOVE | MPOL_MF_STRICT);
    if (syscall(SYS_mbind, address, bytes, MPOL_BIND, mask.data(), maxNode, flags) != 0) {
        Fail("mbind(node=" + std::to_string(node) + ") failed: " + std::strerror(errno));
    }
}

int PickCpu(int node, size_t rank)
{
    if (node < 0) { return -1; }
    try {
        const auto cpus = ParseRangeList(
            ReadTextFile("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist"));
        if (!cpus.empty()) { return cpus[rank % cpus.size()]; }
    } catch (const std::exception&) {
    }
    return -1;
}

void PinCurrentProcess(int cpu)
{
    if (cpu < 0) { return; }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        Fail("sched_setaffinity(cpu=" + std::to_string(cpu) + ") failed: " + std::strerror(errno));
    }
}

void Barrier(SharedControl* control)
{
    const int result = pthread_barrier_wait(&control->barrier);
    if (result != 0 && result != PTHREAD_BARRIER_SERIAL_THREAD) {
        Fail("pthread_barrier_wait failed: " + std::to_string(result));
    }
}

class AscendRuntime {
public:
    explicit AscendRuntime(int device) : device_(device)
    {
        CheckAcl(aclInit(nullptr), "aclInit");
        initialized_ = true;
        CheckAcl(aclrtSetDevice(device_), "aclrtSetDevice");
        deviceSet_ = true;
        CheckAcl(aclrtCreateStream(&stream_), "aclrtCreateStream");
    }

    ~AscendRuntime()
    {
        if (stream_ != nullptr) { (void)aclrtDestroyStream(stream_); }
        if (deviceSet_) { (void)aclrtResetDevice(device_); }
        if (initialized_) { (void)aclFinalize(); }
    }

    aclrtStream Stream() const { return stream_; }

private:
    int device_ = 0;
    aclrtStream stream_ = nullptr;
    bool initialized_ = false;
    bool deviceSet_ = false;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) : bytes_(bytes)
    {
        CheckAcl(aclrtMalloc(&address_, bytes_, ACL_MEM_TYPE_HIGH_BAND_WIDTH), "aclrtMalloc");
    }

    ~DeviceBuffer()
    {
        if (address_ != nullptr) { (void)aclrtFree(address_); }
    }

    void* Data() const { return address_; }
    size_t Size() const { return bytes_; }

private:
    void* address_ = nullptr;
    size_t bytes_ = 0;
};

class RegisteredMapping {
public:
    RegisteredMapping() = default;
    RegisteredMapping(const RegisteredMapping&) = delete;
    RegisteredMapping& operator=(const RegisteredMapping&) = delete;

    RegisteredMapping(RegisteredMapping&& other) noexcept { MoveFrom(other); }

    RegisteredMapping& operator=(RegisteredMapping&& other) noexcept
    {
        if (this != &other) {
            Reset();
            MoveFrom(other);
        }
        return *this;
    }

    ~RegisteredMapping() { Reset(); }

    static RegisteredMapping Open(const std::string& name, size_t bytes)
    {
        RegisteredMapping result;
        const auto pageSize = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        result.mappedBytes_ = AlignUp(bytes, pageSize);
        const int fd = shm_open(name.c_str(), O_RDWR, 0600);
        if (fd < 0) { Fail("shm_open(" + name + ") failed: " + std::strerror(errno)); }
        result.address_ =
            mmap(nullptr, result.mappedBytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (result.address_ == MAP_FAILED) {
            result.address_ = nullptr;
            Fail("mmap(" + name + ") failed: " + std::strerror(errno));
        }
        CheckAcl(aclrtHostRegisterV2(result.address_, result.mappedBytes_,
                                     ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED),
                 "aclrtHostRegisterV2(" + name + ")");
        result.registered_ = true;
        return result;
    }

    void* Data() const { return address_; }

private:
    void Reset()
    {
        if (address_ == nullptr) { return; }
        if (registered_) { (void)aclrtHostUnregister(address_); }
        (void)munmap(address_, mappedBytes_);
        address_ = nullptr;
        registered_ = false;
    }

    void MoveFrom(RegisteredMapping& other)
    {
        address_ = other.address_;
        mappedBytes_ = other.mappedBytes_;
        registered_ = other.registered_;
        other.address_ = nullptr;
        other.mappedBytes_ = 0;
        other.registered_ = false;
    }

    void* address_ = nullptr;
    size_t mappedBytes_ = 0;
    bool registered_ = false;
};

void CreateSegment(const std::string& name, size_t bytes, int numaNode)
{
    const auto pageSize = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const auto mappedBytes = AlignUp(bytes, pageSize);
    const int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) { Fail("creating " + name + " failed: " + std::strerror(errno)); }
    if (mappedBytes > static_cast<size_t>(std::numeric_limits<off_t>::max()) ||
        ftruncate(fd, static_cast<off_t>(mappedBytes)) != 0) {
        const auto message = std::string("ftruncate(") + name + ") failed: " + std::strerror(errno);
        close(fd);
        shm_unlink(name.c_str());
        Fail(message);
    }
    void* address = mmap(nullptr, mappedBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (address == MAP_FAILED) {
        shm_unlink(name.c_str());
        Fail("mmap(" + name + ") failed: " + std::strerror(errno));
    }
    try {
        BindMapping(address, mappedBytes, numaNode);
        (void)madvise(address, mappedBytes, MADV_HUGEPAGE);
        std::memset(address, 0x5A, mappedBytes);
    } catch (...) {
        munmap(address, mappedBytes);
        shm_unlink(name.c_str());
        throw;
    }
    munmap(address, mappedBytes);
}

size_t BlocksOwnedBy(size_t owner, size_t blockCount, size_t tp)
{
    return (blockCount + tp - 1 - owner) / tp;
}

std::vector<Mode> SelectedModes(const std::string& mode)
{
    if (mode == "shared-sequential") { return {Mode::SharedSequential}; }
    if (mode == "sharded-sequential") { return {Mode::ShardedSequential}; }
    if (mode == "sharded-random") { return {Mode::ShardedRandom}; }
    return {Mode::SharedSequential, Mode::ShardedSequential, Mode::ShardedRandom};
}

std::string ModeName(Mode mode)
{
    if (mode == Mode::SharedSequential) { return "shared-sequential"; }
    if (mode == Mode::ShardedSequential) { return "sharded-sequential"; }
    return "sharded-random";
}

Stats Summarize(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    auto percentile = [&](double fraction) {
        const auto index = static_cast<size_t>(std::ceil(fraction * values.size())) - 1;
        return values[std::min(index, values.size() - 1)];
    };
    Stats result;
    result.average = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    result.minimum = values.front();
    result.p50 = percentile(0.50);
    result.p90 = percentile(0.90);
    result.p99 = percentile(0.99);
    result.maximum = values.back();
    return result;
}

std::string SegmentName(pid_t parentPid, Mode mode, size_t owner)
{
    return "/ucm_h2d_" + std::to_string(parentPid) + "_" + ModeName(mode) + "_" +
           std::to_string(owner);
}

void PrintResult(const Options& options, Mode mode, const Stats& stats)
{
    const auto uniqueBytes = static_cast<double>(options.blockSize) * options.blockCount;
    const auto deliveredBytes = uniqueBytes * options.tp;
    const double seconds = stats.average * 1e-6;
    std::cout << std::fixed << std::setprecision(3) << "mode=" << ModeName(mode)
              << " critical_us(avg/min/p50/p90/p99/max)=" << stats.average << '/' << stats.minimum
              << '/' << stats.p50 << '/' << stats.p90 << '/' << stats.p99 << '/' << stats.maximum
              << " unique_GBps=" << uniqueBytes / seconds / 1e9
              << " delivered_GBps=" << deliveredBytes / seconds / 1e9
              << " per_npu_GBps=" << uniqueBytes / seconds / 1e9 << std::endl;
}

void RunMode(const Options& options, Mode mode, size_t rank, pid_t parentPid,
             SharedControl* control, const DeviceBuffer& device, aclrtStream stream)
{
    const bool shared = mode == Mode::SharedSequential;
    const size_t segmentCount = shared ? 1 : options.tp;
    if ((shared && rank == 0) || !shared) {
        const size_t owner = shared ? 0 : rank;
        const size_t blocks =
            shared ? options.blockCount : BlocksOwnedBy(owner, options.blockCount, options.tp);
        const int node = shared ? options.sharedNumaNode : options.numaNodes[owner];
        CreateSegment(SegmentName(parentPid, mode, owner), blocks * options.blockSize, node);
    }
    Barrier(control);

    std::vector<RegisteredMapping> sources;
    sources.reserve(segmentCount);
    for (size_t owner = 0; owner < segmentCount; ++owner) {
        const size_t blocks =
            shared ? options.blockCount : BlocksOwnedBy(owner, options.blockCount, options.tp);
        sources.push_back(RegisteredMapping::Open(SegmentName(parentPid, mode, owner),
                                                  blocks * options.blockSize));
    }
    Barrier(control);
    if ((shared && rank == 0) || !shared) {
        const size_t owner = shared ? 0 : rank;
        if (shm_unlink(SegmentName(parentPid, mode, owner).c_str()) != 0) {
            Fail("shm_unlink failed: " + std::string(std::strerror(errno)));
        }
    }
    Barrier(control);

    std::vector<size_t> order(options.blockCount);
    std::iota(order.begin(), order.end(), 0);
    std::vector<double> criticalLatencies;
    criticalLatencies.reserve(options.iterations);
    const size_t totalIterations = options.warmup + options.iterations;
    for (size_t iteration = 0; iteration < totalIterations; ++iteration) {
        if (mode == Mode::ShardedRandom) {
            std::iota(order.begin(), order.end(), 0);
            std::mt19937_64 generator(options.seed ^ (rank * 0x9E3779B97F4A7C15ULL) ^
                                      (iteration * 0xD1B54A32D192ED03ULL));
            std::shuffle(order.begin(), order.end(), generator);
        }
        Barrier(control);
        const auto begin = Clock::now();
        for (size_t block : order) {
            const size_t owner = shared ? 0 : block % options.tp;
            const size_t localBlock = shared ? block : block / options.tp;
            const auto* source =
                static_cast<const char*>(sources[owner].Data()) + localBlock * options.blockSize;
            const size_t deviceSlots = device.Size() / options.blockSize;
            auto* destination =
                static_cast<char*>(device.Data()) + (block % deviceSlots) * options.blockSize;
            CheckAcl(aclrtMemcpyAsync(destination, options.blockSize, source, options.blockSize,
                                      ACL_MEMCPY_HOST_TO_DEVICE, stream),
                     "aclrtMemcpyAsync(H2D)");
        }
        CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
        control->elapsedUs[rank] =
            std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
        Barrier(control);
        if (rank == 0) {
            const auto critical =
                *std::max_element(control->elapsedUs, control->elapsedUs + options.tp);
            if (iteration >= options.warmup) { criticalLatencies.push_back(critical); }
        }
        Barrier(control);
    }
    if (rank == 0) { PrintResult(options, mode, Summarize(criticalLatencies)); }
    Barrier(control);
    sources.clear();
    Barrier(control);
}

void RunChild(const Options& options, size_t rank, pid_t parentPid, SharedControl* control)
{
    PinCurrentProcess(PickCpu(options.numaNodes[rank], rank));
    AscendRuntime runtime(options.devices[rank]);
    const size_t totalBytes = options.blockSize * options.blockCount;
    const size_t requestedDeviceBytes =
        options.deviceBufferSize == 0 ? totalBytes : std::min(totalBytes, options.deviceBufferSize);
    const size_t deviceBytes = requestedDeviceBytes / options.blockSize * options.blockSize;
    DeviceBuffer device(deviceBytes);
    for (Mode mode : SelectedModes(options.mode)) {
        RunMode(options, mode, rank, parentPid, control, device, runtime.Stream());
    }
}

void CleanupSegments(const Options& options, pid_t parentPid)
{
    for (Mode mode : SelectedModes(options.mode)) {
        const size_t count = mode == Mode::SharedSequential ? 1 : options.tp;
        for (size_t owner = 0; owner < count; ++owner) {
            (void)shm_unlink(SegmentName(parentPid, mode, owner).c_str());
        }
    }
}

int Run(const Options& options)
{
    const size_t controlBytes = sizeof(SharedControl) + (options.tp - 1) * sizeof(double);
    auto* control = static_cast<SharedControl*>(
        mmap(nullptr, controlBytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    if (control == MAP_FAILED) {
        Fail("shared control mmap failed: " + std::string(strerror(errno)));
    }
    pthread_barrierattr_t attributes;
    if (pthread_barrierattr_init(&attributes) != 0 ||
        pthread_barrierattr_setpshared(&attributes, PTHREAD_PROCESS_SHARED) != 0 ||
        pthread_barrier_init(&control->barrier, &attributes, options.tp) != 0) {
        munmap(control, controlBytes);
        Fail("process-shared barrier initialization failed");
    }
    pthread_barrierattr_destroy(&attributes);

    const auto totalBytes = options.blockSize * options.blockCount;
    const auto requestedDeviceBytes =
        options.deviceBufferSize == 0 ? totalBytes : std::min(totalBytes, options.deviceBufferSize);
    const auto deviceBytes = requestedDeviceBytes / options.blockSize * options.blockSize;
    std::cout << std::fixed << std::setprecision(3) << "tp=" << options.tp
              << " block_size=" << options.blockSize << " blocks=" << options.blockCount
              << " unique_bytes=" << totalBytes
              << " unique_GB=" << static_cast<double>(totalBytes) / 1e9
              << " unique_GiB=" << static_cast<double>(totalBytes) / (1024.0 * 1024.0 * 1024.0)
              << " delivered_GB_per_iteration="
              << static_cast<double>(totalBytes) * options.tp / 1e9 << " hbm_GiB_per_npu="
              << static_cast<double>(deviceBytes) / (1024.0 * 1024.0 * 1024.0)
              << " shm_GiB=" << static_cast<double>(totalBytes) / (1024.0 * 1024.0 * 1024.0)
              << std::endl;
    std::cout << "devices=";
    for (size_t i = 0; i < options.tp; ++i) {
        if (i != 0) { std::cout << ','; }
        std::cout << options.devices[i];
    }
    std::cout << " numa_nodes=";
    for (size_t i = 0; i < options.tp; ++i) {
        if (i != 0) { std::cout << ','; }
        std::cout << options.numaNodes[i];
    }
    std::cout << " shared_numa_node=" << options.sharedNumaNode << std::endl;

    const pid_t parentPid = getpid();
    std::vector<pid_t> children;
    children.reserve(options.tp);
    for (size_t rank = 0; rank < options.tp; ++rank) {
        const pid_t pid = fork();
        if (pid < 0) { Fail("fork failed: " + std::string(strerror(errno))); }
        if (pid == 0) {
            try {
                RunChild(options, rank, parentPid, control);
                _exit(0);
            } catch (const std::exception& error) {
                std::cerr << "rank=" << rank << " error: " << error.what() << std::endl;
                _exit(1);
            }
        }
        children.push_back(pid);
    }

    bool failed = false;
    size_t remaining = children.size();
    while (remaining > 0) {
        int status = 0;
        const pid_t pid = waitpid(-1, &status, 0);
        if (pid < 0) {
            if (errno == EINTR) { continue; }
            failed = true;
            break;
        }
        --remaining;
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            failed = true;
            for (pid_t child : children) {
                if (child != pid) { (void)kill(child, SIGKILL); }
            }
            while (remaining > 0) {
                if (waitpid(-1, nullptr, 0) > 0) { --remaining; }
            }
        }
    }

    CleanupSegments(options, parentPid);
    pthread_barrier_destroy(&control->barrier);
    munmap(control, controlBytes);
    return failed ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        return Run(ParseOptions(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << std::endl;
        return 1;
    }
}
