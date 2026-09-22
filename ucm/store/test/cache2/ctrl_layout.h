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
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include "status/status.h"

namespace UC::Cache2 {

struct TwoRankControl {
    static_assert(std::atomic<size_t>::is_always_lock_free);
    static_assert(std::atomic<unsigned>::is_always_lock_free);
    static_assert(std::atomic<bool>::is_always_lock_free);

    std::atomic<size_t> handles[2]{};
    std::atomic<bool> published[2]{};
    std::atomic<unsigned> phases[2]{};
    std::atomic<bool> failed{false};
};

// Only used by the hardware test; HAL allocation and device setup remain real.
class CtrlLayout {
    TwoRankControl& control_;
    size_t slotsPerRank_;

public:
    struct RankDataDesc {
        std::atomic<size_t> handle{0};

        RankDataDesc() = default;
        RankDataDesc(const RankDataDesc& other)
            : handle(other.handle.load(std::memory_order_relaxed))
        {
        }
        RankDataDesc& operator=(const RankDataDesc& other)
        {
            handle.store(other.handle.load(std::memory_order_relaxed), std::memory_order_relaxed);
            return *this;
        }
    };

    CtrlLayout(TwoRankControl& control, size_t slotsPerRank)
        : control_(control), slotsPerRank_(slotsPerRank)
    {
    }

    size_t SlotCount() const { return 2 * slotsPerRank_; }

    Status SetRankDesc(size_t rank, const RankDataDesc& desc)
    {
        control_.handles[rank].store(desc.handle.load(std::memory_order_relaxed),
                                     std::memory_order_relaxed);
        control_.published[rank].store(true, std::memory_order_release);
        return Status::OK();
    }

    Expected<RankDataDesc> GetRankDesc(size_t rank) const
    {
        if (!control_.published[rank].load(std::memory_order_acquire)) {
            return Status::NotFound();
        }
        RankDataDesc desc;
        desc.handle.store(control_.handles[rank].load(std::memory_order_relaxed),
                          std::memory_order_relaxed);
        return desc;
    }
};

}  // namespace UC::Cache2
