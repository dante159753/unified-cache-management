/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_STORAGE_SLOWDOWN_H
#define UNIFIEDCACHE_POSIX_STORE_CC_STORAGE_SLOWDOWN_H

#include <chrono>
#include <thread>
#include "global_config.h"

namespace UC::PosixStore {

class StorageSlowdown {
public:
    void Setup(const Config& config)
    {
        readDelayMs_ = config.storageSlowdownReadDelayMs;
        writeDelayMs_ = config.storageSlowdownWriteDelayMs;
        readBandwidthMBps_ = config.storageSlowdownReadBandwidthMBps;
        writeBandwidthMBps_ = config.storageSlowdownWriteBandwidthMBps;
    }
    void ApplyRead(size_t bytes) const { Apply(readDelayMs_, readBandwidthMBps_, bytes); }
    void ApplyWrite(size_t bytes) const { Apply(writeDelayMs_, writeBandwidthMBps_, bytes); }

private:
    static void Apply(double delayMs, double bandwidthMBps, size_t bytes)
    {
        auto sleepMs = delayMs;
        if (bandwidthMBps > 0.0 && bytes > 0) {
            sleepMs += static_cast<double>(bytes) * 1000.0 / (bandwidthMBps * 1024.0 * 1024.0);
        }
        if (sleepMs <= 0.0) { return; }
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(sleepMs));
    }

    double readDelayMs_{0.0};
    double writeDelayMs_{0.0};
    double readBandwidthMBps_{0.0};
    double writeBandwidthMBps_{0.0};
};

}  // namespace UC::PosixStore

#endif
