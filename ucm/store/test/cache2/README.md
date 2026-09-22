# Cache2 HAL unit tests

Run on Linux without CANN or an NPU:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_UNIT_TESTS=ON -DRUNTIME_ENVIRONMENT=simu -DBUILD_UCM_SPARSE=OFF
cmake --build build --target cache2_hal.test -j
ctest --test-dir build -L cache2_hal --output-on-failure --no-tests=error
```

The existing `cpp_gtest` CI job also builds and runs these tests. The isolated
target compiles the production `HalHostBuffers` and `DataStrategy` implementations
with fake HAL calls and `Device::Setup`. Its virtual address reservations do not
allocate physical host memory.

`CtrlLayout` currently has placeholder methods on `feature_a5`, so CMake copies
the unchanged `DataStrategy` sources into the build directory next to a mock
control block. Source changes trigger CMake regeneration. The mocks and HAL
compile definition are private to this test target.

Coverage includes owner/peer address selection, allocation alignment, huge-page
fallback, handle publication/import, timeout/retry, partial-failure cleanup, and
initialization retry. These tests do not validate the driver's ABI or actual
cross-process sharing, AIO, or device transfers; those require hardware tests.
