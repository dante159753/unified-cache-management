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
 */
#include "store_health_state.h"
#include <gtest/gtest.h>
#include <limits>
#include <thread>

namespace UC::Test {

using PipelineStore::StoreHealthConfig;
using PipelineStore::StoreHealthState;
using namespace std::chrono_literals;

namespace {

bool RecordAndUpdatePassiveHealth(StoreHealthState& state, bool healthy, uint64_t generation,
                                  StoreHealthState::Time now)
{
    state.RecordIo(healthy, generation, now);
    if (!state.PassiveThresholdExceeded(generation, now)) { return false; }
    return state.UpdatePassiveHealth(generation, now);
}

}  // namespace

TEST(UCStoreHealthStateTest, RecordingAndThresholdQueriesDoNotChangeHealth)
{
    StoreHealthConfig config;
    config.passiveMinSamples = 1;
    StoreHealthState state(config);
    const auto now = StoreHealthState::Time{};

    state.RecordIo(false, 0, now);
    EXPECT_EQ(state.GetPassiveWindowStats(now).failures, 1);
    EXPECT_TRUE(state.Enabled());
    EXPECT_TRUE(state.PassiveThresholdExceeded(0, now));
    EXPECT_TRUE(state.Enabled());
    EXPECT_EQ(state.Generation(), 0);

    EXPECT_TRUE(state.UpdatePassiveHealth(0, now));
    EXPECT_FALSE(state.Enabled());
}

TEST(UCStoreHealthStateTest, ThresholdObservationRemainsValidUntilStateTransition)
{
    StoreHealthConfig config;
    config.passiveMinSamples = 1;
    config.passiveFailureRatio = 0.5;
    config.healthWindowSize = 1;
    config.failureThreshold = 1;
    config.initialCooldown = 0s;
    StoreHealthState state(config);
    const auto now = StoreHealthState::Time{};

    state.RecordIo(false, 0, now);
    ASSERT_TRUE(state.PassiveThresholdExceeded(0, now));
    state.RecordIo(true, 0, now);
    state.RecordIo(true, 0, now);
    EXPECT_FALSE(state.PassiveThresholdExceeded(0, now));
    EXPECT_TRUE(state.UpdatePassiveHealth(0, now));
    EXPECT_FALSE(state.Enabled());
    EXPECT_FALSE(state.UpdatePassiveHealth(0, now));
    EXPECT_EQ(state.Generation(), 1);

    ASSERT_TRUE(state.RecordProbe(true, 1, now, now));
    EXPECT_FALSE(state.UpdatePassiveHealth(0, now));
    EXPECT_TRUE(state.Enabled());
    EXPECT_EQ(state.Generation(), 2);
}

TEST(UCStoreHealthStateTest, PassiveRatioUsesTaskTotalsAndStrictThreshold)
{
    StoreHealthConfig config;
    StoreHealthState state(config);
    auto now = StoreHealthState::Time{};
    for (size_t i = 0; i < 99; ++i) {
        EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, true, 0, now));
    }
    EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, false, 0, now));
    EXPECT_TRUE(state.Enabled());
    EXPECT_EQ(state.SampleCount(), 0);
    EXPECT_TRUE(RecordAndUpdatePassiveHealth(state, false, 0, now + 1s));
    EXPECT_FALSE(state.Enabled());
}

TEST(UCStoreHealthStateTest, RequiresTenSamplesWithoutDilutionFromProbes)
{
    StoreHealthConfig config;
    StoreHealthState state(config);
    auto now = StoreHealthState::Time{};
    EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, false, 0, now));
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, true, 0, now));
    }
    for (size_t i = 0; i < 20; ++i) { state.RecordProbe(true, 0, now, now); }
    EXPECT_TRUE(state.Enabled());
    EXPECT_TRUE(RecordAndUpdatePassiveHealth(state, true, 0, now));
    EXPECT_FALSE(state.Enabled());
}

TEST(UCStoreHealthStateTest, MinimumSamplesExcludeExpiredResults)
{
    StoreHealthConfig config;
    StoreHealthState state(config);
    auto now = StoreHealthState::Time{};
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, true, 0, now));
    }
    now += 60s;
    EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, false, 0, now));
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, true, 0, now));
    }
    EXPECT_TRUE(RecordAndUpdatePassiveHealth(state, true, 0, now));
    EXPECT_FALSE(state.Enabled());
}

TEST(UCStoreHealthStateTest, PassiveBucketsExpireAtWindowBoundaryAndAfterIdleGap)
{
    StoreHealthConfig config;
    config.passiveMinSamples = 1;
    config.passiveFailureRatio = 0.5;
    StoreHealthState state(config);
    auto now = StoreHealthState::Time{};
    EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, true, 0, now));
    EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, false, 0, now + 1s));
    EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, true, 0, now + 60s));
    EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, false, 0, now + 61s));
    EXPECT_TRUE(state.Enabled());
    EXPECT_TRUE(RecordAndUpdatePassiveHealth(state, false, 0, now + 600s));
}

TEST(UCStoreHealthStateTest, FailureRemainsInLastBucketBeforeExpiry)
{
    StoreHealthConfig config;
    config.passiveMinSamples = 1;
    config.passiveFailureRatio = 0.5;
    StoreHealthState state(config);
    auto now = StoreHealthState::Time{};
    EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, true, 0, now));
    EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, true, 0, now));
    EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, false, 0, now + 1s));
    EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, true, 0, now + 60s));
    EXPECT_TRUE(RecordAndUpdatePassiveHealth(state, false, 0, now + 60s));
}

TEST(UCStoreHealthStateTest, PassiveSnapshotExcludesExpiredBucketsWithoutNewIo)
{
    StoreHealthConfig config;
    StoreHealthState state(config);
    auto now = StoreHealthState::Time{};
    EXPECT_EQ(state.GetPassiveWindowStats(now).total, 0);
    EXPECT_EQ(state.GetPassiveWindowStats(now).failures, 0);
    EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, true, 0, now));
    EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, false, 0, now + 1s));
    auto stats = state.GetPassiveWindowStats(now + 59s);
    EXPECT_EQ(stats.total, 2);
    EXPECT_EQ(stats.failures, 1);
    stats = state.GetPassiveWindowStats(now + 60s);
    EXPECT_EQ(stats.total, 1);
    EXPECT_EQ(stats.failures, 1);
    stats = state.GetPassiveWindowStats(now + 61s);
    EXPECT_EQ(stats.total, 0);
    EXPECT_EQ(stats.failures, 0);
    EXPECT_TRUE(state.Enabled());
}

TEST(UCStoreHealthStateTest, BucketReuseExcludesOldFailuresWhenNewIoSucceeds)
{
    StoreHealthConfig config;
    StoreHealthState state(config);
    const auto old = StoreHealthState::Time{};
    const auto now = old + config.passiveWindow;
    state.RecordIo(false, 0, old);
    state.RecordIo(true, 0, now);
    const auto stats = state.GetPassiveWindowStats(now);
    EXPECT_EQ(stats.total, 1);
    EXPECT_EQ(stats.failures, 0);
    EXPECT_FALSE(state.PassiveThresholdExceeded(0, now));
}

TEST(UCStoreHealthStateTest, LateFailuresCannotHideRecentFailure)
{
    StoreHealthConfig config;
    config.passiveMinSamples = 1;
    StoreHealthState state(config);
    const auto old = StoreHealthState::Time{};
    const auto now = old + config.passiveWindow;
    state.RecordIo(false, 0, now);
    state.RecordIo(false, 0, old);
    const auto stats = state.GetPassiveWindowStats(now);
    EXPECT_EQ(stats.total, 1);
    EXPECT_EQ(stats.failures, 1);
    EXPECT_TRUE(state.PassiveThresholdExceeded(0, now));
}

TEST(UCStoreHealthStateTest, PassiveTimestampBoundsDoNotWrap)
{
    StoreHealthConfig config;
    StoreHealthState state(config);
    const auto now = StoreHealthState::Time{} + std::chrono::seconds{UINT32_MAX};
    state.RecordIo(false, 0, now);
    state.RecordIo(false, 0, now + 1s);
    state.RecordIo(false, 0, StoreHealthState::Time{} - 1s);
    const auto stats = state.GetPassiveWindowStats(now);
    EXPECT_EQ(stats.total, 1);
    EXPECT_EQ(stats.failures, 1);
    EXPECT_EQ(state.GetPassiveWindowStats(now + 59s).failures, 1);
    EXPECT_EQ(state.GetPassiveWindowStats(now + 60s).total, 0);
    EXPECT_EQ(state.GetPassiveWindowStats(now + 60s).failures, 0);
}

TEST(UCStoreHealthStateTest, RecoveryRequiresFreshProbeWindowAndElapsedCooldown)
{
    StoreHealthConfig config;
    config.initialCooldown = 30s;
    config.passiveMinSamples = 1;
    config.healthWindowSize = 2;
    config.failureThreshold = 2;
    StoreHealthState state(config);
    auto now = StoreHealthState::Time{};
    state.RecordProbe(true, 0, now, now);
    state.RecordProbe(true, 0, now, now);
    EXPECT_TRUE(RecordAndUpdatePassiveHealth(state, false, 0, now));
    EXPECT_FALSE(state.RecordProbe(true, 0, now, now + 1s));
    EXPECT_EQ(state.SampleCount(), 0);
    EXPECT_FALSE(state.RecordProbe(true, 1, now + 10s, now + 10s));
    EXPECT_FALSE(state.RecordProbe(true, 1, now + 20s, now + 20s));
    EXPECT_FALSE(state.RecordProbe(true, 1, now + 29s, now + 31s));
    EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, true, 1, now + 40s));
    EXPECT_FALSE(state.Enabled());
    EXPECT_TRUE(state.RecordProbe(true, 1, now + 40s, now + 40s));
    EXPECT_TRUE(state.Enabled());
    EXPECT_EQ(state.Generation(), 2);
    EXPECT_EQ(state.GetPassiveWindowStats(now + 40s).total, 0);
    EXPECT_EQ(state.GetPassiveWindowStats(now + 40s).failures, 0);
    EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, false, 0, now + 41s));
    EXPECT_FALSE(state.RecordProbe(false, 1, now + 30s, now + 41s));
    EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, true, 2, now + 41s));
}

TEST(UCStoreHealthStateTest, ProbeFailureInterruptsRecovery)
{
    StoreHealthConfig config;
    config.healthWindowSize = 2;
    config.failureThreshold = 1;
    StoreHealthState state(config);
    auto now = StoreHealthState::Time{};
    EXPECT_TRUE(state.RecordProbe(false, 0, now, now));
    EXPECT_FALSE(state.RecordProbe(true, 1, now + 30s, now + 30s));
    EXPECT_FALSE(state.RecordProbe(false, 1, now + 40s, now + 40s));
    EXPECT_FALSE(state.RecordProbe(true, 1, now + 50s, now + 50s));
    EXPECT_TRUE(state.RecordProbe(true, 1, now + 60s, now + 60s));
}

TEST(UCStoreHealthStateTest, BackoffUsesHealthyUptimeCapsAndResetsAfterStability)
{
    StoreHealthConfig config;
    config.initialCooldown = 30s;
    config.passiveMinSamples = 1;
    config.healthWindowSize = 1;
    config.failureThreshold = 1;
    config.maxCooldown = 100s;
    config.stableResetAfter = 20s;
    StoreHealthState state(config);
    auto now = StoreHealthState::Time{};
    for (auto expected : {30s, 60s, 100s, 100s}) {
        const auto generation = state.Generation();
        EXPECT_TRUE(RecordAndUpdatePassiveHealth(state, false, generation, now));
        EXPECT_EQ(state.Cooldown(), expected);
        EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, false, generation, now + 1s));
        EXPECT_FALSE(state.RecordProbe(false, state.Generation(), now + 2s, now + 2s));
        EXPECT_EQ(state.Cooldown(), expected);
        now += expected;
        EXPECT_TRUE(state.RecordProbe(true, state.Generation(), now, now));
        now += 1s;
    }
    now += 20s;
    EXPECT_TRUE(RecordAndUpdatePassiveHealth(state, false, state.Generation(), now));
    EXPECT_EQ(state.Cooldown(), 30s);
}

TEST(UCStoreHealthStateTest, CanDisablePassiveDetection)
{
    StoreHealthConfig config;
    config.passiveMinSamples = 1;
    config.passiveEnabled = false;
    StoreHealthState state(config);
    auto now = StoreHealthState::Time{};
    EXPECT_FALSE(RecordAndUpdatePassiveHealth(state, false, 0, now));
    EXPECT_TRUE(state.Enabled());
    EXPECT_FALSE(state.RecordProbe(false, 0, now, now));
    EXPECT_TRUE(state.RecordProbe(false, 0, now, now));
}

TEST(UCStoreHealthStateTest, ValidatesPassiveAndBackoffConfiguration)
{
    StoreHealthConfig config;
    EXPECT_TRUE(config.Validate().Success());
    for (double ratio : {-0.1, 1.0, std::numeric_limits<double>::quiet_NaN()}) {
        config.passiveFailureRatio = ratio;
        EXPECT_EQ(config.Validate(), Status::InvalidParam());
    }
    config = StoreHealthConfig{};
    config.passiveWindow = 0s;
    EXPECT_EQ(config.Validate(), Status::InvalidParam());
    config = StoreHealthConfig{};
    config.passiveMinSamples = 0;
    EXPECT_EQ(config.Validate(), Status::InvalidParam());
    config = StoreHealthConfig{};
    config.maxCooldown = 1s;
    EXPECT_EQ(config.Validate(), Status::InvalidParam());
    config = StoreHealthConfig{};
    config.backoffFactor = std::numeric_limits<double>::infinity();
    EXPECT_EQ(config.Validate(), Status::InvalidParam());
    config = StoreHealthConfig{};
    config.stableResetAfter = 0s;
    EXPECT_EQ(config.Validate(), Status::InvalidParam());
}

TEST(UCStoreHealthStateTest, ConcurrentIoPreservesCountsAndSnapshotBounds)
{
    StoreHealthConfig config;
    config.passiveFailureRatio = 0.9;
    StoreHealthState state(config);
    const auto now = StoreHealthState::Time{} + 100s;
    std::atomic<bool> start{false};
    std::atomic<bool> finished{false};
    std::atomic<bool> invalidSnapshot{false};
    std::thread reader([&] {
        while (!finished.load()) {
            const auto stats = state.GetPassiveWindowStats(now);
            if (stats.failures > stats.total) { invalidSnapshot.store(true); }
        }
    });
    std::vector<std::thread> writers;
    for (size_t i = 0; i < 8; ++i) {
        writers.emplace_back([&] {
            while (!start.load()) { std::this_thread::yield(); }
            for (size_t j = 0; j < 10000; ++j) { state.RecordIo(j % 10 != 0, 0, now); }
        });
    }
    start.store(true);
    for (auto& writer : writers) { writer.join(); }
    finished.store(true);
    reader.join();
    const auto stats = state.GetPassiveWindowStats(now);
    EXPECT_EQ(stats.total, 80000);
    EXPECT_EQ(stats.failures, 8000);
    EXPECT_FALSE(invalidSnapshot.load());
    EXPECT_TRUE(state.Enabled());
}

TEST(UCStoreHealthStateTest, ConcurrentBucketReusePreservesCounts)
{
    StoreHealthConfig config;
    StoreHealthState state(config);
    const auto old = StoreHealthState::Time{} + 1s;
    const auto now = old + config.passiveWindow;
    state.RecordIo(false, 0, old);
    std::atomic<bool> start{false};
    std::vector<std::thread> writers;
    for (size_t i = 0; i < 8; ++i) {
        writers.emplace_back([&] {
            while (!start.load()) { std::this_thread::yield(); }
            for (size_t j = 0; j < 10000; ++j) { state.RecordIo(j % 10 != 0, 0, now); }
        });
    }
    start.store(true);
    for (auto& writer : writers) { writer.join(); }
    const auto stats = state.GetPassiveWindowStats(now);
    EXPECT_EQ(stats.total, 80000);
    EXPECT_EQ(stats.failures, 8000);
}

TEST(UCStoreHealthStateTest, ConcurrentLateUpdatesCannotOverwriteNewSecond)
{
    StoreHealthConfig config;
    StoreHealthState state(config);
    const auto old = StoreHealthState::Time{} + 1s;
    const auto now = old + config.passiveWindow;
    state.RecordIo(false, 0, now);
    std::atomic<bool> start{false};
    std::vector<std::thread> writers;
    for (size_t i = 0; i < 8; ++i) {
        writers.emplace_back([&, i] {
            while (!start.load()) { std::this_thread::yield(); }
            for (size_t j = 0; j < 10000; ++j) { state.RecordIo(false, 0, i % 2 == 0 ? old : now); }
        });
    }
    start.store(true);
    for (auto& writer : writers) { writer.join(); }
    const auto stats = state.GetPassiveWindowStats(now);
    EXPECT_EQ(stats.total, 40001);
    EXPECT_EQ(stats.failures, 40001);
    EXPECT_TRUE(state.PassiveThresholdExceeded(0, now));
}

TEST(UCStoreHealthStateTest, RecoveryRejectsStaleCandidatesDuringConcurrentIo)
{
    StoreHealthConfig config;
    config.passiveMinSamples = 1;
    config.healthWindowSize = 1;
    config.failureThreshold = 1;
    config.initialCooldown = 0s;
    StoreHealthState state(config);
    const auto now = StoreHealthState::Time{} + 100s;
    state.RecordIo(false, 0, now);
    ASSERT_TRUE(state.PassiveThresholdExceeded(0, now));
    std::atomic<bool> stop{false};
    std::atomic<size_t> ready{0};
    std::vector<std::thread> writers;
    for (size_t i = 0; i < 8; ++i) {
        writers.emplace_back([&] {
            state.RecordIo(false, 0, now);
            ready.fetch_add(1);
            while (!stop.load()) { state.RecordIo(false, 0, now); }
        });
    }
    while (ready.load() != writers.size()) { std::this_thread::yield(); }
    EXPECT_TRUE(state.UpdatePassiveHealth(0, now));
    EXPECT_TRUE(state.RecordProbe(true, 1, now, now));
    EXPECT_LE(state.GetPassiveWindowStats(now).total, writers.size());
    EXPECT_FALSE(state.UpdatePassiveHealth(0, now));
    stop.store(true);
    for (auto& writer : writers) { writer.join(); }
    EXPECT_LE(state.GetPassiveWindowStats(now).total, writers.size());
    EXPECT_EQ(state.Generation(), 2);
    const auto before = state.GetPassiveWindowStats(now);
    state.RecordIo(false, 2, now);
    EXPECT_TRUE(state.PassiveThresholdExceeded(2, now));
    EXPECT_EQ(state.GetPassiveWindowStats(now).failures, before.failures + 1);
}

}  // namespace UC::Test
