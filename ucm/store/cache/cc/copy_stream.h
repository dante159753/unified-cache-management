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
#ifndef UNIFIEDCACHE_CACHE_STORE_CC_COPY_STREAM_H
#define UNIFIEDCACHE_CACHE_STORE_CC_COPY_STREAM_H

#include <atomic>
#include <memory>
#include <vector>
#include "logger/logger.h"
#include "status/status.h"
#include "thread/latch.h"
#include "trans/device.h"

namespace UC::CacheStore {

class CopyStream {
    int32_t deviceId_{-1};
    size_t streamNumber_{0};
    size_t streamIndex_{0};
    bool useGdr_{false};
    std::vector<std::shared_ptr<Trans::Stream>> streams_;

public:
    class Completion {
        friend class CopyStream;
        std::shared_ptr<UC::Latch> latch_{nullptr};
        std::shared_ptr<std::atomic_bool> success_{nullptr};
        std::vector<std::shared_ptr<Trans::Stream>> streams_{};

    public:
        bool Valid() const noexcept { return latch_ != nullptr; }
        Status Wait() const
        {
            if (!latch_) { return Status::OK(); }
            latch_->Wait();
            if (success_ && !success_->load(std::memory_order_acquire)) {
                return Status::Error("copy stream callback failed");
            }
            return Status::OK();
        }
    };

    Status Setup(const int32_t deviceId, const size_t streamNumber, const bool useGdr)
    {
        Trans::Device device;
        auto s = device.Setup(deviceId);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to setup device({}).", s, deviceId);
            return s;
        }
        streams_.reserve(streamNumber);
        for (size_t i = 0; i < streamNumber; ++i) {
            std::shared_ptr<Trans::Stream> stream =
                useGdr ? device.MakeGdrStream() : device.MakeSharedStream();
            if (!stream) [[unlikely]] {
                UC_ERROR("Failed to make stream on device({}).", deviceId);
                return Status::Error();
            }
            streams_.push_back(std::move(stream));
        }
        deviceId_ = deviceId;
        streamNumber_ = streamNumber;
        useGdr_ = useGdr;
        return Status::OK();
    }
    std::shared_ptr<Trans::Stream> NextStream() noexcept
    {
        if (streamNumber_ == 0) [[unlikely]] { return nullptr; }
        auto& stream = streams_[streamIndex_];
        streamIndex_ = (streamIndex_ + 1) % streamNumber_;
        return stream;
    }
    Status WaitEvent(void* event) noexcept
    {
        auto status = Status::OK();
        for (auto& stream : streams_) {
            auto s = stream->WaitEvent(event);
            if (s.Success()) { continue; }
            UC_ERROR("Failed({}) to wait event on stream on device({}).", s, deviceId_);
            if (status.Success()) { status = s; }
        }
        return status;
    }
    Status AppendCompletion(Completion& completion)
    {
        completion = Completion{};
        if (streamNumber_ == 0) [[unlikely]] { return Status::OK(); }
        if (useGdr_) { return Synchronize(); }
        auto latch = std::make_shared<UC::Latch>();
        auto success = std::make_shared<std::atomic_bool>(true);
        latch->Set(streams_.size());
        Completion pending;
        pending.latch_ = latch;
        pending.success_ = success;
        pending.streams_ = streams_;
        for (auto& stream : streams_) {
            auto s = stream->AppendCallback([latch, success](bool ok) {
                if (!ok) { success->store(false, std::memory_order_release); }
                latch->Done();
            });
            if (s.Success()) { continue; }
            UC_ERROR("Failed({}) to append completion callback on device({}).", s, deviceId_);
            (void)Synchronize();
            return s;
        }
        completion = std::move(pending);
        return Status::OK();
    }
    Status Synchronize() noexcept
    {
        auto status = Status::OK();
        for (auto& stream : streams_) {
            auto s = stream->Synchronized();
            if (s.Success()) { continue; }
            UC_ERROR("Failed({}) to synchronize stream on device({}).", s, deviceId_);
            if (status.Success()) { status = s; }
        }
        return status;
    }
};

}  // namespace UC::CacheStore

#endif
