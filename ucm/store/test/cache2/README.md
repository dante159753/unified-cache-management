# A5 DataStrategy two-process test

This hardware test is built only with `BUILD_UNIT_TESTS=ON` and
`RUNTIME_ENVIRONMENT=ascend-a5`. It requires two visible Ascend devices, CANN,
and the HAL driver library. It does not mock HAL or ACL.

`CtrlLayout` currently has inline placeholders. This target compiles unchanged
`DataStrategy` sources next to a test-only control block implementing `SlotCount`,
`SetRankDesc`, and `GetRankDesc`. A shared anonymous `mmap` carries real HAL handles
between the two child processes. Production `CtrlLayout` is unchanged.

## Build and run

From the repository root, after loading the CANN environment:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build-a5 -DCMAKE_BUILD_TYPE=Release \
  -DRUNTIME_ENVIRONMENT=ascend-a5 -DBUILD_UNIT_TESTS=ON -DBUILD_UCM_SPARSE=OFF
cmake --build build-a5 --target cache2_data_strategy_a5.test -j
./build-a5/ucm/store/test/cache2/cache2_data_strategy_a5.test 0 1
```

Arguments are positional: `device0 device1 [slot_bytes [slots_per_rank [timeout_ms]]]`.
Defaults are `0 1 8388608 256 600000`: 256 slots of 8 MiB per rank (2 GiB per rank,
4 GiB of shared host allocations in total) and a ten-minute timeout. Each copy
transfers exactly one complete slot. Device IDs are logical IDs in the current
container. For example:

```bash
./build-a5/ucm/store/test/cache2/cache2_data_strategy_a5.test 2 3 8388608 256 60000
```

Use `-DASCEND_ROOT=...` and `-DASCEND_DRIVER_ROOT=...` if the SDK/driver is installed
elsewhere. The existing `UCM_TRANS_HAL_INCLUDE_DIR` and `UCM_TRANS_HAL_LIBRARY`
CMake options can also specify the exact HAL header directory and library.

CTest runs the default device pair, with no hardware-based skip:

```bash
ctest --test-dir build-a5 -R '^cache2\.data_strategy_a5$' -V --output-on-failure
```

## What is checked

Each child initializes ACL after `fork`, then calls the real `DataStrategy::Setup`
with its own device ID. In each of two rounds:

1. The owner writes every byte of each local slot through `DataAt`, then reads and
   checks it on the CPU. The pattern includes owner, slot, round, and byte offset.
2. Both processes wait until all owner writes have completed.
3. Each process uses the peer's `DeviceDataAt` as the **source** of an asynchronous
   D2D copy into separately allocated local HBM, then D2H copies that HBM into a
   pinned readback buffer. It copies one complete 8 MiB slot at a time,
   synchronizes the stream, and checks every byte. Each rank reuses 8 MiB of
   temporary HBM and 8 MiB of pinned readback memory with the default sizes.
4. Both processes finish reading before either owner rewrites or releases memory.

The imported buffer is only read. Owner slots must not expose `DeviceDataAt`, and
peer slots must not expose `DataAt`. Logs include rank, PID, slot, and round; a
mismatch prints its offset and actual/expected bytes. The parent prints
`MULTI_RESULT PASS` only if both children exit successfully, and terminates/reaps
remaining children on failure or timeout.
