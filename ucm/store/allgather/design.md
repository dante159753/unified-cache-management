# UcmAllGatherStore

## Scope

For replicated MLA KV cache, a stable block-id hash assigns every persisted
block to one TP rank. The owner alone performs CacheStore/POSIX I/O; load
reconstructs the original KV tensors on every TP rank through batched
AllGather and a segmented-copy AscendC kernel.

For non-replicated TP layouts, every rank keeps its existing rank-specific key
and persists all local blocks. The same fused staging buffers aggregate small
D2H/H2D transfers, but load scatters directly from the local buffer without
creating a TP process group or issuing an AllGather. Scheduler instances pass
through to PipelineStore, preserving lookup semantics and the persisted data
format.

The vLLM HMA path creates separate FA and WA stores. Its TP-local tensors are
replicated before persistence, so every rank passes the complete dump key set
to `UcmAllGatherStore`; the store's hash ownership replaces HMA's ordinary
contiguous TP key slicing.

## Interface and configuration

`UcmAllGatherStore` implements `UcmKVStoreBaseV1`. It accepts the ordinary
PipelineStore configuration plus:

| Key | Default | Meaning |
| --- | ---: | --- |
| `allgather_window_blocks_per_rank` | 4 | Requested owner blocks per rank in one collective window |
| `allgather_load_slots` | 2 | Rank-local load pipeline depth |
| `allgather_dump_slots` | 2 | Rank-local dump pipeline depth |
| `allgather_fused_buffer_capacity_mb` | 256 layerwise, otherwise 2048 | Hard NPU staging-pool budget |
| `allgather_rank` | required | Rank in the TP device group |
| `allgather_world_size` | required | TP group size |
| `allgather_replicated_data` | false | Enables hash ownership and TP AllGather for replicated layouts; false selects rank-local transfer aggregation |
| `allgather_scatter_only` | false | For replicated layouts, keeps hash-partitioned dump but loads every block from shared CacheStore and skips TP AllGather |
| `allgather_collective_mode` | `host` | Collective engine: `host`, `aicpu_ts`, `aiv`, or `auto` |
| `allgather_collective_variable_counts` | `true` | Use `HcclAllGatherV` for deterministic single-window loads and omit unused payload slots |
| `allgather_scatter_aiv_cores` | 1 | Maximum AIV cores used by one load scatter kernel, in `[1, 40]` |
| `allgather_dynamic_windows` | false | Dispatches whichever in-flight owner load completes first instead of waiting in source-window order |

The actual window is reduced if the requested window does not fit the budget.
Initialization fails when even one block per rank cannot fit. The store logs
the requested and actual window, payload bytes, descriptor bytes, total bytes,
and configured capacity.

## Ownership and windowing

For a block id `b`, `owner(b) = little_endian_u64(b[0:8]) % TP`. Owner-local
slot numbers are assigned in input order. Window `j` contains every block whose
owner-local slot is in `[jW, (j+1)W)`. Every rank derives the same load windows
without exchanging metadata.

In rank-local aggregation mode the effective storage world size is one, so all
local blocks are owned and window slots follow input order.

In scatter-only mode, dump ownership remains hash-partitioned across TP ranks,
but load ownership has an effective world size of one. Each rank loads the full
window from shared CacheStore into its local fused send buffer and scatters it
without allocating a receive buffer or creating a collective communicator.

All collectives use fixed tensors:

- send: `W * S` bytes;
- receive: `TP * W * S` bytes.

`S` is the direct-I/O-aligned Store shard size. Unused send slots may contain
old data but are never referenced by scatter descriptors.

The completion-ordered load path keeps the collective allocation fixed but
formats each rank contribution as a self-describing frame:

```text
[frame header][block records][compacted shard payload][unused tail]
```

The header identifies the task sequence and source window and gives the number
of valid block records. A block record identifies its task-global destination
row, payload slot, and status. Metadata travels in the same fixed-size
AllGather as the payload; there is no shared-memory round plan and no metadata
collective. Scatter parses every received rank frame on device and computes
local tensor addresses from the task-wide destination table and immutable model
layout. Different ranks may contribute different source windows to one
collective because every record is self-describing.

For a single-window load, every rank derives the same per-owner valid counts
from the request. When variable counts are enabled, HcclAllGatherV transfers
only each frame's metadata and populated payload while retaining the fixed
`rank * frame_bytes` receive displacement expected by scatter. Multi-window
completion order is rank-local, so those loads keep fixed AllGather unless a
preceding count exchange is added.

## Rank-local fused-buffer pool

The worker allocates the large payload pool once during Store initialization.
`load_data` allocates task-lifetime metadata tensors whose total size is linear in
the requested block count; payload tensors are never allocated on the hot path.

Each load slot owns:

- `W * S` send payload;
- `TP * W * S` AllGather receive payload;
- one reusable NPU completion event.

Each collective load task owns one destination-address table covering all of
its windows. It is uploaded once before backend completion polling starts, and
every load stream waits on that upload event. Non-collective aggregation keeps
the slot-local destination and `(owner, owner_slot)` route tables.

Each dump slot owns:

- `W * S` zero-initialized pack payload;
- `W * chunks_per_block` NPU descriptors;
- 41 per-core offsets.

The model layout is converted once into a shared NPU chunk template. Each
template row contains tensor index, source offset, destination offset, and byte
count. Load planning uploads a task-wide destination-address and route table.
Dump keeps persistent descriptor and offset arrays for its source-to-staging
pack.

For TP size `T`, window `W`, shard bytes `S`, load slots `Kl`, and dump slots
`Kd`, payload memory is:

```text
Kl * (T + 1) * W * S + Kd * W * S
```

Scatter-only payload memory is `Kl * W * S + Kd * W * S`.

Persistent descriptor and offset buffers are included in the hard capacity
calculation. Load-task metadata uses
`blocks * (tensor_count * 8 + 8)` bytes outside the payload pool and is released
with its task.
With the measured DeepSeek-V4 direct-layout shard `S = 3,186,688` bytes,
`W=32`, `Kl=Kd=2`, payload is about 1.14 GiB at TP4 and 1.90 GiB at TP8.
Layerwise shards are much smaller; the 256 MiB default normally retains W=32.

## Load pipeline

`load_data` immediately submits up to `Kl` owner-local PipelineStore loads, so
POSIX/H2D starts before completion. With dynamic windows enabled, the progress
worker polls all in-flight backend handles and processes whichever completes
first:

1. poll the owner-local PipelineStore loads until any window is ready;
2. AllGather the fixed send payload into the slot receive payload;
3. launch compact scatter into the original vLLM tensor addresses;
4. record the slot completion event and submit the next window.

Owner-local failures are retained while every rank continues the same payload
collective sequence. After every window has submitted its AllGather and scatter,
the current stream is synchronized once. A rank then reports its local failure
to vLLM; the request-level load fails if any rank reports failure, so the store
does not add a separate error collective.

This removes head-of-line blocking when an earlier source window misses or has
slower POSIX I/O. The Nth ready window always uses collective group and stream
`N % load_group_count`, independent of its buffer slot. All ranks therefore
submit the same collective ordinal to the same communicator even when their
source-window completion orders differ. Every rank still submits exactly the
same number of collectives for a task, while the task sequence prevents frames
from crossing task boundaries. Setting `allgather_dynamic_windows=false`
retains the source-window FIFO behavior for A/B validation.

This overlaps owner I/O for later windows with collective/scatter for the
current window. A completed slot is reused after event query succeeds. If the
pool is exhausted, only the oldest reclaimable slot event is synchronized;
there is no device-wide or current-stream synchronization. Concurrent load
tasks exceeding the configured pipeline depth remain in a FIFO queue. Finishing
the head task primes the next task, preserving identical collective order on all
TP ranks while bounding temporary HBM by the configured slot count.

Async vLLM loads are progressed by one dedicated FIFO worker per rank. FA and
WA stores on the same worker share this progressor, so their collective order
is identical on every TP rank. The worker sleeps on a condition variable while
idle and performs owner I/O waits,
AllGather, compact scatter, and stream completion on a dedicated NPU stream
outside the scheduler thread. Metadata uploads record an event on the caller's
stream; the progress stream waits for that event before scatter.
`check` only reads task completion state and never waits for I/O or the device.
`wait` blocks on the same completion condition. Keeping one progress worker per
rank preserves task order; embedded global rows remove the former requirement
that owner-local window readiness be identical.
The shared progressor owns one dedicated HCCL process group with the same TP
membership, so its background collectives cannot interleave with model
collectives on the vLLM TP group. The group is initialized with a barrier on
the worker thread before the progress thread can issue its first collective.
Its HCCL buffer defaults to 8 MiB instead of the runtime's 200 MiB default;
`allgather_hccl_buffer_mb` can override it for larger transfer windows.
The collective engine is configured on the UCM communicator rather than through
the process-wide `HCCL_OP_EXPANSION_MODE`. `aicpu_ts` selects
`hcclOpExpansionMode=2`, the non-AIV high-bandwidth mode available on Atlas A3.
`host` preserves the Atlas A2 behavior, while `auto` delegates to the platform
default.

On an owner failure, the affected rank skips local scatter but still enters all
payload collectives with its valid staging allocation, preserving collective
order. The failing rank raises after the final stream synchronization.

In the completion-ordered path, a backend miss or I/O failure produces a
`FAILED` block record with no payload instead of removing that block from the
round. Every rank receives the record, skips its scatter, and marks the local
task failed. This propagates owner data failures without an error AllReduce and
also guarantees that failed blocks count toward task completion. Collective,
H2D, metadata-allocation, and kernel-launch failures remain rank-local runtime
failures because a failed collective cannot carry its own error record.

The current destination table is task-owned and remains alive until all slot
events complete, so dynamic windows cannot write through reused metadata. A
future cross-task completion scheduler would need a shared task table with
`task_slot + generation`; that reuse mechanism is deliberately outside this
phase.

## Dump pipeline

The vLLM connector submits only blocks owned by the current rank. Each window:

1. waits for the caller prerequisite event on the current stream once;
2. packs original tensor segments into a reusable dump payload;
3. records an ACL event after the pack kernel;
4. passes the payload and event to rank-private PipelineStore D2H/POSIX.

The buffer remains leased until PipelineStore completion. If all dump slots are
busy, submission waits only for the oldest dump window and then reuses its
slot. `check` finalizes completed windows so poll-based callers cannot leak
events or pool leases.

## Copy kernels

Load scatter receives the fixed AllGather buffer, an `int64[rows, tensors]`
destination table, an `int32[rows, 2]` route table, and the immutable
`int64[chunks, 4]` model-layout template. A kernel task identifies one
`(block, chunk)` pair and derives source and destination addresses on device.
At most 40 AIV cores consume tasks with `task = core_id + n * core_count`, so
the host no longer expands, sorts, or balances per-chunk descriptors.

The fixed chunk size is at most 32 KiB. This fits one queue buffer, keeps all
writes disjoint, and naturally supports any tensor list supplied by the
connector without model- or rank-count-specific code. Address-zero ghost
tensors are skipped on device. The only dynamic load metadata is proportional
to `rows * (tensor_count * 8 + 8)` bytes.

Dump pack retains the segmented-copy kernel. It receives an NPU
`int64[N, 3]` array of source address, destination address, and byte count,
plus `int32[C + 1]` per-core descriptor ranges. Host planning splits copies
into at most 32 KiB chunks and balances bytes over at most 40 AIV cores.

Each core copies GM to UB to GM with two 32 KiB queue buffers. Copies whose
source, destination, and size are 32-byte aligned use `DataCopy`; remaining
copies use `DataCopyPad`. Copy ranges are disjoint, so neither kernel needs
atomics or cross-core synchronization.

Segmented pack and compact scatter are built as separate AscendC libraries.
CANN 8.5.1 must register each generated kernel binary independently; placing
both global kernel entries in one library makes binary registration fail.

## Host-memory registration

The inner PipelineStore is forced to `share_buffer_enable=false` and
`local_rank_size=1`. Every rank therefore allocates and registers its own
CacheStore host memory with its local NPU. This avoids a single shared host
allocation being registered against all devices while retaining the existing
CacheStore/POSIX implementation.

## Constraints

- All TP ranks must submit load tasks in the same order and with identical task
  row layouts. Source-window completion order may differ by rank.
- Load-pool mutation is owned by the per-rank progress worker. Submission,
  completion state, and prepared metadata lifetime are protected by its
  condition lock.
- A completed load has synchronized the progress worker's current NPU stream,
  so KV consumers may run on the caller's current stream. The recorded
  completion event protects slot reuse.
- Address `0` entries are metadata-only ghost tensors and are skipped.
- Direct-I/O tail padding is gathered and persisted but never scattered into
  model tensors.
