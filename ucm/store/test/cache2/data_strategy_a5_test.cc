/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include <acl/acl.h>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include "data_strategy.h"
#include "trans/device.h"

namespace {

using UC::Cache2::CtrlLayout;
using UC::Cache2::DataStrategy;
using UC::Cache2::TwoRankControl;
using Clock = std::chrono::steady_clock;

struct Options {
    int32_t devices[2]{0, 1};
    size_t slotBytes{8 * 1024 * 1024};
    size_t slotsPerRank{256};
    size_t timeoutMs{600 * 1000};
};

void Log(size_t rank, const std::string& message)
{
    fmt::print("[rank={} pid={}] {}\n", rank, getpid(), message);
    std::fflush(stdout);
}

void CheckAcl(aclError ret, const char* api)
{
    if (ret != ACL_SUCCESS) { throw std::runtime_error(fmt::format("{} ret={}", api, ret)); }
}

void CheckStatus(const UC::Status& status, const char* api)
{
    if (status.Failure()) { throw std::runtime_error(fmt::format("{} status={}", api, status)); }
}

uint8_t Pattern(size_t owner, size_t slot, unsigned round, size_t offset)
{
    return static_cast<uint8_t>((owner + 1) * 37 + slot * 17 + round * 53 +
                                (offset ^ (offset >> 8)));
}

void Verify(const volatile uint8_t* data, size_t bytes, size_t owner, size_t slot, unsigned round)
{
    for (size_t offset = 0; offset < bytes; ++offset) {
        const uint8_t expected = Pattern(owner, slot, round, offset);
        const uint8_t actual = data[offset];
        if (actual != expected) {
            throw std::runtime_error(fmt::format(
                "MISMATCH owner={} slot={} round={} offset={} actual={:02x} expected={:02x}", owner,
                slot, round, offset, static_cast<unsigned>(actual),
                static_cast<unsigned>(expected)));
        }
    }
}

int RunRank(size_t rank, TwoRankControl& shared, const Options& options)
{
    const size_t peer = 1 - rank;
    UC::Trans::Device device;
    bool initialized = false;
    int result = 0;
    {
        DataStrategy strategy;
        CtrlLayout ctrl(shared, options.slotsPerRank);
        void* hbm = nullptr;
        void* readback = nullptr;
        aclrtStream stream = nullptr;
        auto barrier = [&](unsigned phase) {
            shared.phases[rank].store(phase, std::memory_order_release);
            const Clock::time_point deadline =
                Clock::now() + std::chrono::milliseconds(options.timeoutMs);
            while (shared.phases[peer].load(std::memory_order_acquire) < phase) {
                if (shared.failed.load(std::memory_order_acquire)) {
                    throw std::runtime_error("peer failed");
                }
                if (Clock::now() >= deadline) {
                    throw std::runtime_error(fmt::format("barrier timed out: phase={}", phase));
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        };

        try {
            Log(rank, fmt::format("SETUP device={} slot_bytes={} slots_per_rank={} timeout_ms={}",
                                  options.devices[rank], options.slotBytes, options.slotsPerRank,
                                  options.timeoutMs));
            CheckStatus(device.Init(), "Device::Init");
            initialized = true;
            CheckStatus(strategy.Setup(ctrl, options.devices[rank], options.slotBytes,
                                       options.slotsPerRank, rank, options.timeoutMs),
                        "DataStrategy::Setup");
            CheckAcl(aclrtMalloc(&hbm, options.slotBytes, ACL_MEM_MALLOC_NORMAL_ONLY),
                     "aclrtMalloc(HBM)");
            CheckAcl(aclrtMallocHost(&readback, options.slotBytes), "aclrtMallocHost(readback)");
            CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");

            for (unsigned round = 0; round < 2; ++round) {
                for (size_t slot = 0; slot < options.slotsPerRank; ++slot) {
                    const size_t slotIdx = rank * options.slotsPerRank + slot;
                    uint8_t* own = static_cast<uint8_t*>(strategy.DataAt(slotIdx));
                    if (own == nullptr || strategy.DeviceDataAt(slotIdx) != nullptr) {
                        throw std::runtime_error(
                            fmt::format("invalid owner address: slot={}", slot));
                    }
                    Log(rank, fmt::format("CPU_WRITE_READ round={} slot={} addr={} bytes={}", round,
                                          slot, static_cast<void*>(own), options.slotBytes));
                    for (size_t offset = 0; offset < options.slotBytes; ++offset) {
                        own[offset] = Pattern(rank, slot, round, offset);
                    }
                    Verify(own, options.slotBytes, rank, slot, round);
                    Log(rank, fmt::format("  PASS CPU_WRITE_READ round={} slot={}", round, slot));
                }
                barrier(2 * round + 1);

                for (size_t slot = 0; slot < options.slotsPerRank; ++slot) {
                    const size_t slotIdx = peer * options.slotsPerRank + slot;
                    void* imported = strategy.DeviceDataAt(slotIdx);
                    if (imported == nullptr || strategy.DataAt(slotIdx) != nullptr) {
                        throw std::runtime_error(
                            fmt::format("invalid peer address: slot={}", slot));
                    }
                    Log(rank,
                        fmt::format("D2D_READ round={} owner={} slot={} src={} dst={} bytes={}",
                                    round, peer, slot, imported, hbm, options.slotBytes));
                    CheckAcl(aclrtMemcpyAsync(hbm, options.slotBytes, imported, options.slotBytes,
                                              ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                             "aclrtMemcpyAsync(import -> HBM, D2D)");
                    CheckAcl(aclrtMemcpyAsync(readback, options.slotBytes, hbm, options.slotBytes,
                                              ACL_MEMCPY_DEVICE_TO_HOST, stream),
                             "aclrtMemcpyAsync(HBM -> readback, D2H)");
                    CheckAcl(aclrtSynchronizeStreamWithTimeout(
                                 stream, static_cast<int32_t>(options.timeoutMs)),
                             "aclrtSynchronizeStreamWithTimeout");
                    Verify(static_cast<uint8_t*>(readback), options.slotBytes, peer, slot, round);
                    Log(rank, fmt::format("  PASS D2D_READ round={} owner={} slot={}", round, peer,
                                          slot));
                }
                // Owners cannot overwrite or release buffers until both D2D reads complete.
                barrier(2 * round + 2);
            }
        } catch (const std::exception& error) {
            shared.failed.store(true, std::memory_order_release);
            Log(rank, fmt::format("FAIL {}", error.what()));
            result = 1;
        }

        auto cleanup = [&](aclError ret, const char* api) {
            if (ret != ACL_SUCCESS) {
                shared.failed.store(true, std::memory_order_release);
                Log(rank, fmt::format("FAIL cleanup {} ret={}", api, ret));
                result = 1;
            }
        };
        if (stream != nullptr) {
            cleanup(
                aclrtSynchronizeStreamWithTimeout(stream, static_cast<int32_t>(options.timeoutMs)),
                "aclrtSynchronizeStreamWithTimeout");
            cleanup(aclrtDestroyStream(stream), "aclrtDestroyStream");
        }
        if (hbm != nullptr) { cleanup(aclrtFree(hbm), "aclrtFree"); }
        if (readback != nullptr) { cleanup(aclrtFreeHost(readback), "aclrtFreeHost"); }
    }
    if (initialized) {
        const UC::Status reset = device.Reset(options.devices[rank]);
        const UC::Status finalize = device.Finalize();
        if (reset.Failure() || finalize.Failure()) {
            Log(rank, fmt::format("FAIL runtime cleanup reset={} finalize={}", reset, finalize));
            result = 1;
        }
    }
    if (result != 0) { shared.failed.store(true, std::memory_order_release); }
    Log(rank, result == 0 ? "RANK_RESULT PASS" : "RANK_RESULT FAIL");
    return result;
}

size_t ParseNumber(const char* text)
{
    size_t value = 0;
    const char* end = text + std::strlen(text);
    const std::from_chars_result parsed = std::from_chars(text, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
        throw std::runtime_error(fmt::format("invalid numeric argument: {}", text));
    }
    return value;
}

}  // namespace

int main(int argc, char** argv)
{
    if ((argc == 2 && std::strcmp(argv[1], "--help") == 0) ||
        (argc != 1 && (argc < 3 || argc > 6))) {
        fmt::print(
            "Usage: {} [device0 device1 [slot_bytes [slots_per_rank [timeout_ms]]]]\n"
            "Defaults: 0 1 8388608 256 600000 (2 GiB per rank)\n",
            argv[0]);
        return argc == 2 && std::strcmp(argv[1], "--help") == 0 ? 0 : 2;
    }
    Options options;
    try {
        if (argc >= 3) {
            for (size_t rank = 0; rank < 2; ++rank) {
                const size_t device = ParseNumber(argv[rank + 1]);
                if (device > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                    throw std::runtime_error("device ID exceeds int32_t");
                }
                options.devices[rank] = static_cast<int32_t>(device);
            }
        }
        if (argc >= 4) { options.slotBytes = ParseNumber(argv[3]); }
        if (argc >= 5) { options.slotsPerRank = ParseNumber(argv[4]); }
        if (argc >= 6) { options.timeoutMs = ParseNumber(argv[5]); }
        if (options.devices[0] == options.devices[1] || options.slotBytes == 0 ||
            options.slotsPerRank == 0 || options.timeoutMs == 0 ||
            options.timeoutMs > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error(
                "use two distinct device IDs and positive sizes/timeout (int32 ms)");
        }
    } catch (const std::exception& error) {
        fmt::print(stderr, "{}\n", error.what());
        return 2;
    }

    void* memory = mmap(nullptr, sizeof(TwoRankControl), PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) {
        std::perror("mmap(control)");
        return 1;
    }
    auto* shared = new (memory) TwoRankControl{};
    pid_t children[2]{-1, -1};
    bool failed = false;
    // The parent never initializes ACL; each child gets its own runtime after fork.
    for (size_t rank = 0; rank < 2; ++rank) {
        children[rank] = fork();
        if (children[rank] == 0) { _exit(RunRank(rank, *shared, options)); }
        if (children[rank] < 0) {
            std::perror("fork");
            failed = true;
            break;
        }
    }

    const Clock::time_point deadline =
        Clock::now() + std::chrono::milliseconds(options.timeoutMs) + std::chrono::seconds(60);
    for (;;) {
        bool running = false;
        for (size_t rank = 0; rank < 2; ++rank) {
            if (children[rank] <= 0) { continue; }
            int status = 0;
            const pid_t result = waitpid(children[rank], &status, WNOHANG);
            if (result == children[rank]) {
                if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                    Log(rank, fmt::format("CHILD_EXIT status={}", status));
                    failed = true;
                }
                children[rank] = -1;
            } else if (result < 0 && errno != EINTR) {
                std::perror("waitpid");
                failed = true;
            } else {
                running = true;
            }
        }
        failed = failed || shared->failed.load(std::memory_order_acquire);
        if (failed || !running) { break; }
        if (Clock::now() >= deadline) {
            fmt::print(stderr, "FAIL child process timeout\n");
            failed = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    for (pid_t child : children) {
        if (child <= 0) { continue; }
        kill(child, SIGKILL);
        while (waitpid(child, nullptr, 0) < 0 && errno == EINTR) {}
    }
    shared->~TwoRankControl();
    munmap(memory, sizeof(TwoRankControl));
    fmt::print("MULTI_RESULT {}\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
