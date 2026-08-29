# Pipeline-integrated AllGather Store

## Summary

This design implements AllGather as a native `StoreV1` stage inside
`UcmPipelineStore`.

The logical pipeline is:

```text
vLLM connector
      |
UcmPipelineStore
      |
AllGatherStore        owns block routing, fused device buffers, replication,
      |               scatter/gather kernels, and task completion
CacheStore            owns rank-private DRAM cache and Cache -> backend fill
      |
PosixStore            owns persistent I/O
```

The configuration name is `AllGather|Cache|Posix`. `PipelineStore.Stack()` is
called in reverse construction order: Posix, Cache, then AllGather. The
AllGather stage receives the Cache stage through the existing `store_backend`
entry in `Detail::Dictionary`.

## Goals

- Make AllGather a normal composable `StoreV1` stage rather than a second store
  framework around `UcmPipelineStore`.
- Keep CacheStore responsible for cache hits, POSIX fallback, cache readiness,
  H2D into a supplied staging address, and backend error propagation.
- Store each replicated block on one hash owner. Ascend uses HCCL plus a fused
  scatter kernel; CUDA uses one remote-scatter kernel that reads owner slots
  through CUDA IPC without an NCCL payload collective.
- Allocate CacheStore host memory per rank. Do not map and register one shared
  CacheStore buffer on every accelerator.
- Make W4 a fixed first-version invariant. Derive each stage's aligned shard
  size from startup KV-layout parameters, calculate the exact buffer plan, and
  allocate for W4 rather than shrinking the window to fit a capacity limit.
- Reserve the complete AllGather device-memory plan before vLLM decides how many
  KV-cache blocks fit, so later fused-buffer allocation cannot cause an OOM.
- Preserve bounded load/dump slots and FA/WA shared progress.
- Support replicated MLA data and local transfer aggregation for non-MLA data
  through one stage implementation.
- Keep scheduler-side lookup and garbage-collection behavior compatible with
  `UcmPipelineStore`.

## Non-goals for the first version

- Changing vLLM KV-cache tensor layout or block-allocation policy. The
  integration only subtracts an explicit UCM reservation from memory available
  to the existing allocator.
- Replacing CacheStore or PosixStore queues.
- GDR or Cache SDMA-direct into AllGather-owned staging buffers. The first
  version rejects these combinations rather than silently using a wrong buffer
  registration.
- Dynamic TP membership during the lifetime of one Store.
- Dynamically reducing W4 when memory is tight. Startup fails if the fixed plan
  cannot be reserved or allocated.
- Making a failing owner error visible to every rank with an error AllReduce.

## Why the stage is native

`PipelineStore` composes native `StoreV1` objects and passes task handles through
the top stage. Implementing AllGather as another `StoreV1` gives it the same
lifetime, health wrapping, task semantics, and backend composition rules as
CacheStore and PosixStore. It also removes Python polling and Python-owned fused
buffer state from the load hot path.

The Ascend stage uses HCCL through a native platform-runtime interface and a
dedicated communicator. CUDA has no UCM payload communicator: the TP group is
used only during startup to broadcast an IPC bootstrap key, then a POSIX
shared-memory control area exchanges CUDA IPC handles and slot generations.
Neither data path invokes a PyTorch process group from the native progress
thread.

## Pipeline construction

Add a builder for `AllGather|Cache|Posix` in
`ucm/store/pipeline/connector.py`.

The input configuration describes the external tensor layout. The builder
derives separate configurations for each stage:

| Field | AllGatherStore | CacheStore | PosixStore |
|---|---|---|---|
| `tensor_size_list` | original tensor sizes | `[shard_size]` | not used |
| `shard_size` | padded packed shard | padded packed shard | padded packed shard |
| `block_size` | unchanged | unchanged | unchanged |
| `share_buffer_enable` | not used | forced `false` | not used |
| `local_rank_size` | TP world | forced `1` | not used |
| `tensor_size` | not used | not used | `shard_size` on workers |
| `gpu_kv_buffer_addrs` | original destinations | empty in v1 | not used |

The construction order is:

```python
pipeline.Stack("Posix", libposixstore, posix_config)
pipeline.Stack("Cache", libcachestore, packed_cache_config)
pipeline.Stack("AllGather", liballgatherstore, allgather_config)
```

For the first version, the builder validates:

- `cache_sdma_direct == false`;
- `use_gdr == false`;
- `share_buffer_enable` is either absent or overridden to `false` for the
  internal CacheStore;
- `shard_size >= sum(tensor_size_list)` and direct-I/O alignment is preserved;
- `allgather_window_blocks_per_rank` is positive and identical on every rank;
- `allgather_fused_buffer_capacity_mb` is rejected for this pipeline because it
  cannot be a meaningful cap when W4 is fixed;
- rank, world size, communicator identity, and replicated/local mode are
  internally consistent.

## Configuration surface

An MLA configuration uses the ordinary pipeline connector:

```yaml
ucm_connector_name: UcmPipelineStore
ucm_connector_config:
  store_pipeline: "AllGather|Cache|Posix"
  storage_backends: /data/yanzhao/ucm/kvdata
  io_direct: true
  posix_io_engine: aio
  cache_buffer_capacity_gb: 8
  allgather_load_slots: 2
  allgather_dump_slots: 2
  allgather_window_blocks_per_rank_fa: 4
  allgather_window_blocks_per_rank_wa: 64
  allgather_load_slots_fa: 8
  allgather_load_slots_wa: 2
  allgather_collective_buffer_mb: 8
  allgather_collective_mode: host
```

The vLLM integration injects worker-only values:

- `allgather_rank`;
- `allgather_world_size`;
- `allgather_replicated_data`;
- `allgather_scatter_only`;
- `allgather_layerwise`;
- `allgather_runtime_key`;
- `allgather_collective_root_info` for the first stage using a runtime key.

The former `allgather_hccl_buffer_mb` and `allgather_hccl_root_info` names remain
accepted as Ascend compatibility aliases.

`allgather_collective_mode` selects the communication engine of the UCM-owned
communicator. It accepts `host`, `aicpu_ts`, `aiv`, and `auto`. Atlas A3 must use
`aicpu_ts`, which maps to `hcclOpExpansionMode=2` (AI CPU+TS). The setting is
communicator-local and does not change the process-wide
`HCCL_OP_EXPANSION_MODE` or the vLLM tensor-parallel communicator. `host` remains
the default to preserve the current Atlas A2 behavior.

Pipeline detection is the only supported activation path. Code checks the
parsed pipeline stages instead of a standalone connector name.

## Startup memory plan and vLLM reservation

The pipeline does not expose a fused-buffer capacity for the first version.
Once vLLM has resolved the concrete KV layout, UCM calculates an
`AllGatherMemoryPlan` from startup parameters:

- the aligned packed shard size for each FA/WA stage;
- fixed window size `W = 4`;
- TP/storage world size;
- configured load and dump slot counts;
- fixed-size device metadata arenas;
- the collective send and receive reservation shared by FA and WA on that
  worker. The default reservation is `2 * allgather_collective_buffer_mb` per
  accelerator.

The allocator and the vLLM memory estimator must call the same native planning
function. Model-specific shard arithmetic must not be duplicated in the vLLM
patch. The plan records every aligned allocation and produces one
`total_reserved_bytes` value per accelerator. Per-stage payload and metadata are summed,
while the shared collective-runtime reservation is counted exactly once.

For vLLM-Ascend 0.23, the integration patches
`NPUWorker.determine_available_memory()`. After profiling computes the normal
available KV memory, but before the value is logged and returned, it applies:

```text
raw_available_kv_bytes = requested_memory - non_kv_cache_memory
available_kv_bytes = raw_available_kv_bytes - allgather_reserved_bytes
```

The CUDA integration applies the same reservation to
`GPUWorker.determine_available_memory()`.

The patch logs the raw KV budget, UCM AllGather reservation, and final KV
budget. The reservation is zero unless the selected pipeline contains
AllGather. A non-positive final budget is a startup error with the full memory
plan in the diagnostic. Version adapters with an explicit
`kv_cache_memory_bytes` branch must subtract the reservation there as well, so
a manual KV-cache limit cannot bypass accounting.

Memory planning and allocation occur in this order:

1. vLLM profiles the model without allocating AllGather device buffers.
2. UCM derives the exact plan from the resolved startup KV layout.
3. vLLM subtracts `total_reserved_bytes` and allocates fewer KV-cache blocks.
4. Pipeline construction allocates the planned AllGather buffers.
5. Setup verifies actual requested allocation bytes equal the reservation.

This ordering is an invariant: allocating the buffers before profiling and
then subtracting them again would double count the same memory. Any layout or
configuration mismatch between planning and Store setup fails startup instead
of growing the buffer pool beyond the reservation.

## Replication and shared progress runtime

The FA and WA pipelines on one worker must use one collective FIFO. Two
independent communicators or progress threads allow rank-local readiness to
change collective order and can deadlock.

`liballgatherstore.so` owns a process-local runtime registry keyed by
`allgather_runtime_key`. A runtime contains:

- one HCCL communicator for Ascend, or no communicator for CUDA;
- one dedicated accelerator progress stream;
- one condition-variable-driven progress thread;
- one FIFO containing tasks from every registered AllGather stage;
- a fatal runtime status and reference count.

The vLLM integration creates one runtime key per engine, DP rank, TP group, and
device. FA and WA use that same key. Different DP groups use different keys.

On the TP group's rank zero, a small native binding creates an HCCL root-info
blob on Ascend or a random IPC bootstrap key on CUDA. The existing vLLM TP group
broadcasts it during Store construction. Ascend stages use it to initialize the
shared communicator. CUDA stages use it to derive the shared-memory control
name and exchange every load slot's CUDA IPC memory/event handles. TP1 creates
a progress runtime without either replication mechanism.

The runtime is reference counted. The communicator and progress stream are
destroyed only after the final stage has drained its tasks and released the
runtime.

## Data ownership and layout

For replicated data, the storage owner is stable and independent of tensor
layout:

```text
owner = little_endian_u64(block_id[0:8]) % world_size
```

Only the owner calls CacheStore for that block. CacheStore and PosixStore see
one packed tensor of `shard_size` bytes per block. AllGatherStore sees the
original destination/source tensor list and owns the mapping between the packed
shard and those tensors.

`Lookup` and `LookupOnPrefix` are not owner-filtered. They query the complete
block sequence through CacheStore, which checks rank-local DRAM first and then
the shared POSIX backend for misses. Ownership filtering applies only to data
movement and prefetch work.

The AllGather stage keeps the current 32 KiB chunk layout. Zero addresses and
zero-sized ghost tensors are metadata only and are never passed to a kernel.
Direct-I/O padding is transferred and persisted but never scattered to a model
tensor.

When `allgather_replicated_data` is false, storage world size is one. The stage
still packs or scatters fragmented tensors through fused buffers, but it skips
hash partitioning and collective communication. This is the non-MLA
transfer-coalescing mode.

## Fused buffer pool

Each stage allocates rank-local device buffers. CacheStore host memory is
rank-private in collective mode and shared across ranks in scatter-only mode.

For window size `W`, aligned packed shard size `S`, TP size `P`, load
slots `L`, dump slots `D`, and replicated-mode flag `R`, one stage allocates:

```text
load send       L * W * S
load receive    receive_slots * P * W * S    Ascend replicated mode only
dump send       D * W * S
```

CUDA replicated mode has no receive allocation. Each rank exports its `L` send
slots, imports the peers' slots, and stores one `P`-entry pointer table per load
slot. TP1 and non-replicated mode also omit receive allocations.

The plan additionally includes the exact aligned sizes of every explicit device
allocation: chunk-layout tables, dump descriptors and offsets, and bounded load
destination/route metadata arenas. There is no best-effort window clamp: all W4
buffers are allocated, or setup fails. The shard size, each plan component, the
total reservation, and actual allocated bytes are logged once at setup.

Slots have `free`, `active`, and `reclaimable` states. A completion event guards
reuse. When every slot is busy, the oldest reclaimable slot is synchronized;
the submission thread never synchronizes a slot that is still owned by an
active task.

## Load state machine

`AllGatherStore::Load()` receives the original block IDs, shard indices, and
destination tensor addresses.

1. Validate the task shape and build a deterministic ownership/window plan.
2. Build destination and route metadata.
3. Enqueue the task into the shared runtime FIFO and return an AllGather task
   handle immediately.
4. Prime at most `allgather_load_slots` windows:
   - acquire a load slot;
   - build a backend task containing only locally owned blocks;
   - give CacheStore one packed destination address per block;
   - submit `CacheStore::Load()` without waiting.
5. The shared progress thread processes tasks in FIFO order. For each window:
   - wait for the owner-local CacheStore task;
   - on Ascend, enter HCCL AllGather and scatter its receive buffer;
   - on CUDA, publish the owner slot generation, wait for all owners, and launch
     remote scatter directly from their CUDA IPC mappings;
   - record slot completion and prime the next queued window.
6. After all windows have been submitted, synchronize the progress stream once
   for the task, publish terminal status, and wake `Wait()` callers.

CacheStore remains responsible for the important asynchronous dependency:

```text
POSIX completion -> Cache buffer ready -> packed H2D completion
```

AllGatherStore starts replication only after the CacheStore backend task for
that window completes. On CUDA, a producer cannot refill a slot until every
rank has consumed the previous generation. Two or more load slots allow
CacheStore work for later windows to overlap with replication/scatter for the
current window.

## Dump state machine

Dump does not require a collective.

1. Partition blocks with the same owner function used by load.
2. For each window, acquire a dump slot.
3. Wait on the caller prerequisite event once, then launch one segmented-copy
   kernel for locally owned blocks into the packed slot.
4. Record a device event and submit only those packed owner blocks to
   CacheStore. The event is forwarded through `TaskDesc.prerequisiteHandle`, so
   CacheStore D2H cannot read the slot before packing finishes.
5. Release the slot after the CacheStore task completes.

Each block is inserted into CacheStore and PosixStore by exactly one TP rank.

## Task handles and thread safety

AllGatherStore owns task handles returned to PipelineStore. Backend task handles
never escape the stage.

Each task records:

- operation, FIFO sequence, and deterministic windows;
- active and unprimed windows;
- backend handles and fused slots;
- device metadata buffers;
- first local error and terminal status;
- a condition variable for `Wait()`.

Public Store methods remain thread-safe as required by `StoreV1`.

- `Check()` is non-blocking. It only reads terminal state.
- `Wait()` sleeps on the task condition and returns the stored status.
- the progress thread exclusively mutates active load-window and slot state;
- dump slot mutation is protected by the stage lock;
- `Close()` rejects new work, drains submitted tasks, joins the progress
  runtime, releases buffers, and is idempotent.

No pending path busy-polls `backend->Check()`. CacheStore completion is consumed
through its task wait in the dedicated progress thread; idle progress threads
sleep on a condition variable.

## Error and synchronization semantics

Local failures include Cache/Posix misses or I/O errors, H2D errors, collective
errors, metadata allocation errors, kernel errors, and runtime shutdown.

There is no error AllReduce. If an owner-local backend load fails, that rank:

- records the first local error;
- still executes the remaining fixed collective sequence with a valid staging
  allocation so peers do not deadlock;
- skips local scatter when its task is already invalid;
- returns the error from `Wait()` after the required sequence completes.

Other ranks may return success. The vLLM request-level failure path discards the
loaded KV when any worker reports failure. This keeps errors off the collective
critical path without weakening collective ordering.

Task success is published only after one final progress-stream synchronize.
Once `Wait()` returns, every scatter for that task is complete and consumers on
the caller stream may use the destination tensors. Dump uses the existing
prerequisite-event contract instead of synchronizing the caller stream.

A fatal collective or progress-thread failure marks the shared runtime failed, aborts
all queued tasks on every registered FA/WA stage in the process, and causes new
submissions to fail immediately.

## Lookup, prefetch, scheduler, and health checks

Scheduler instances have `device_id < 0`. Their AllGather stage allocates no
device resources and creates no communicator. Lookup, prefix lookup, and prefetch
pass through to CacheStore/PosixStore. Load, dump, check, and wait are
worker-only operations and return an error if called on a scheduler instance.

Worker prefetch filters to owner blocks in replicated mode. Worker lookup and
prefix lookup pass through without owner filtering so that CacheStore can check
both local DRAM and the shared PosixStore for every requested block.

`AllGatherStore::CheckHealth()` must not execute a collective because health
checks are not guaranteed to run simultaneously on every rank. It reports only
local runtime setup/fatal state. CacheStore and PosixStore retain their own
pipeline health wrappers.

## Metrics

The new stage preserves the current metric names where their meaning is
unchanged and adds stage boundaries needed to diagnose the pipeline:

- task FIFO depth and wait time;
- active/unprimed windows and slot-reclaim wait;
- CacheStore submit count and backend wait time per window;
- collective submit and completion time;
- scatter/gather submit and completion time;
- final stream synchronization time;
- logical unique bytes, local host-ingress bytes, and final delivered bytes;
- task, backend, H2D, collective, kernel, and metadata error counters;
- configured window size, planned/reserved/actual fused-pool HBM bytes, and the raw
  and post-reservation vLLM KV-memory budgets.

Metrics must distinguish actual rank-local DRAM-to-device ingress from logical KV
delivered after AllGather. Aggregate logical delivery is not reported as raw
H2D bandwidth.

## Proposed code layout

```text
ucm/store/allgather/
  cc/
    allgather_store.{h,cc}
    allgather_runtime.{h,cc}
    buffer_pool.{h,cc}
    load_task.{h,cc}
    dump_task.{h,cc}
  csrc/op_kernel/
    compact_scatter.cpp
    segmented_copy.cpp
  csrc/op_host/
    kernel_launcher.{h,cc}
    runtime_binding.cpp
  CMakeLists.txt

ucm/store/pipeline/
  connector.py
  cpy/pipeline_store.py.cc
  allgather_store_design.md
```

The kernel launcher is runtime-neutral and accepts raw device pointers and a
platform stream. The native platform adapter invokes it directly.

## Migration plan

1. Add native runtime, buffer pool, kernel launcher, and fake-backend unit tests.
2. Add the AllGather `StoreV1` stage and
   `AllGather|Cache|Empty` for deterministic tests.
3. Add `AllGather|Cache|Posix` with stage-specific configuration rewriting.
4. Add the shared native memory-plan calculator and the vLLM-Ascend 0.23 KV
   budget reservation patch before enabling device-buffer allocation.
5. Change vLLM integration to inject runtime metadata based on pipeline
   capabilities.
6. Run direct Store A/B tests against baseline commit `bf69d67`.
7. Run DeepSeek V4 Flash end-to-end A/B tests on A2.

## Validation and acceptance criteria

Correctness:

- TP1/2/4/8 dump then Cache-hit load for DeepSeek V4 FA and WA layouts;
- Posix-hit load with Cache misses, AIO, direct I/O, and W4;
- missing key, backend failure, H2D failure, and owner failure injection;
- identical collective order with different owner-local completion order;
- non-MLA local-coalesced mode without a collective;
- scheduler lookup/prefix lookup and GC behavior;
- repeated construction/destruction without surviving threads, communicators,
  events, or device buffers.
- AllGather-disabled, explicit KV-budget, insufficient-memory, and FA/WA
  multi-stage reservation cases;
- no device-buffer allocation during model profiling and no double subtraction;
- byte-for-byte equality between planned reservation and Store allocation.

Performance:

- use 32 FA plus 32 WA blocks, 313,958,400 logical bytes per load;
- measure TP1/2/4/8 with W4 and the same load-slot count as the baseline;
- report submit, Cache wait, collective, scatter, sync, and total latency;
- report unique host ingress and aggregate logical delivery separately;
- Cache-hit median must be no more than 5% slower than baseline `bf69d67`;
- Posix-hit throughput must not regress by more than 5%;
- task submission and metadata construction should improve because the outer
  Python wrapper is removed;
- rank-private CacheStore allocation and fused-pool HBM must match the logged
  startup plan, and the vLLM KV-cache budget must be reduced by exactly that
  reservation.

The baseline branch remains the rollback and comparison point throughout the
implementation.
