# Remote-scatter pipeline store

## Goal

The pipeline stage replicates owner-partitioned KV data without a UCM-owned
collective communicator. Each rank loads only its owned packed shards, exports
its load slots through device-memory IPC, and scatters directly from every
owner's remote mapping into the rank-local model tensors.

The same state machine is used on Ascend and CUDA. Ascend uses
`aclrtIpcMemGetExportKey`; CUDA uses CUDA IPC memory. HCCL/NCCL payload
collectives, IPC events, gathered receive buffers, framed metadata, and
collective memory reservations are not part of this design.

## Pipeline and ownership

The stage remains the top element of `AllGather|Cache|Posix` or
`AllGather|Cache|Fake`:

```text
model tensors <-> RemoteScatterStore <-> packed CacheStore <-> PosixStore
```

For replicated MLA data:

```text
owner = little_endian_u64(block_id[0:8]) % TP
```

Only the owner submits the packed shard to CacheStore. Lookup still passes the
complete block sequence through the lower stores, while load, dump, and
prefetch filter data movement by owner.

Non-replicated models use the same packed load and scatter implementation with
storage world size one. No IPC transport is created in that mode.

## Startup protocol

The vLLM TP group broadcasts a 32-byte random bootstrap key. It is used only to
derive the POSIX shared-memory control name; it is not a communication
communicator.

For every load slot, each rank:

1. allocates one `W * shard_size` device buffer;
2. obtains its device-visible process ID and exports the buffer;
3. publishes process ID and export handle in the shared control area;
4. authorizes all peer process IDs to import its handles;
5. imports every peer slot with peer access enabled;
6. copies the peer-pointer table to device memory.

The lower-store `Wait()` contract guarantees that packed H2D is complete before
the owner publishes ready. The release/acquire shared-state transition therefore
provides the cross-process handoff without a separate IPC event.

## Load state machine

For a load task, every rank constructs the same deterministic windows and
routes. Window `n` uses slot `n % load_slots`.

1. Submit lower-store loads for the initial free slots. Each task contains only
   locally owned rows and writes packed shards into its rank-local slot.
2. Wait for the current owner load. CacheStore remains responsible for
   `POSIX completion -> cache ready -> packed H2D completion`.
3. Copy destination and `(owner, owner_slot)` route tables to the slot stream.
4. Publish `(generation, window_tag, failed)` in shared host memory.
5. Poll all pending owners together. As soon as any owners are ready, launch
   scatter for their owner-contiguous rows while slower owners keep loading.
   Adjacent ready owners are coalesced into one kernel; when every owner is ready
   together, the window still uses one kernel.
6. If any owner failed, stop submitting new scatter work and return load failure
   after all owners have completed the protocol.
7. Record slot completion. Before reuse, publish local consumption and wait
   until all ranks consumed that generation.
8. Join the slot completion events once and finish the task.

The generation prevents an old consumer from observing a refilled slot. The
window tag detects rank task/order divergence before a kernel can route data
into the wrong KV tensors. Backend failure travels in the same shared state, so
no error AllReduce is needed.

## Kernel interface

The remote-scatter launcher receives:

```text
peer_buffers[TP]             uint64 device addresses
destinations[rows][tensors] uint64 device addresses
routes[rows]                 (owner, owner_slot)
chunks[]                     (tensor, packed_offset, tensor_offset, bytes)
row_count
chunks_per_block
tensor_count
shard_size
```

Each AIV task copies one chunk:

```text
source = peer_buffers[owner]
       + owner_slot * shard_size
       + packed_offset
destination = destinations[row][tensor] + tensor_offset
```

Chunks remain 32 KiB. Only the final chunk of each tensor may be shorter. This
keeps work balanced across one or two AIV cores and avoids the severe two-core
imbalance observed with one-task-per-tensor GLM 5.2 shapes. Each chunk uses a
32 KiB UB tile with double buffering and 32-byte aligned `DataCopy` when
possible; unaligned tails use padded copy instructions.

There is no type conversion or numerical calculation. The operator is purely
memory-bound.

## Load dispatch modes

`allgather_remote_scatter_mode` selects how ready owner buffers are materialized:

- `batch_copy` issues one runtime device copy for every destination tensor and
  does not use AIV;
- `copy_then_scatter` copies each ready owner's packed range into a local receive
  buffer, then launches the compact-scatter kernel on that buffer;
- `kernel` launches direct remote scatter whenever one or more owners become
  ready, preserving overlap with slower owners;
- `batched_remote_scatter` waits for every owner in the window and submits one
  direct remote-scatter launch for all rows, tensors, and chunks.

The batched mode uses the same peer-pointer, route, destination, and 32 KiB
chunk tables as the standalone remote-scatter benchmark. It removes repeated
launches and is preferred for cache-hit windows with low readiness skew. The
dynamic `kernel` mode remains available for mixed cache/POSIX windows where
overlapping early-owner scatter with backend stragglers can be more valuable.

## Memory plan

For window blocks per rank `W`, packed shard bytes `S`, TP size `P`, load slots
`L`, and dump slots `D`, one stage allocates:

```text
load payload       L * W * S
dump payload       D * W * S
destinations       L * max_rows * tensor_count * 8
routes             L * max_rows * 2 * 4
peer pointer table L * P * 8             replicated mode only
chunk/dump metadata
```

`max_rows = W * P` for replicated data and `W` otherwise. There is no
`P * W * S` receive area and no HCCL/NCCL buffer reservation. vLLM subtracts
the exact sum of these stage allocations from the KV-cache budget.

## Streams and threads

FA and WA stages share one process-local progress runtime keyed by engine, TP
group, and device. The runtime owns:

- one sleeping FIFO progress thread for load tasks;
- one independent dump thread and stream;
- one stream per load slot;
- one completion stream.

Different slot streams allow owner H2D for later windows to overlap with the
current scatter. Slot generation and consumption, rather than communicator
order, control reuse.

## Errors and teardown

Setup fails if IPC export, authorization, or import fails. At runtime, shared
state timeouts, generation overruns, and window mismatches poison the shared
runtime because later tasks cannot safely continue. Ordinary backend failure
is propagated through the slot failure bit and does not poison the transport.

Teardown first drains submitted tasks, joins both progress threads, closes peer
IPC mappings, destroys events and streams, and frees local buffers. Rank zero
unlinks the shared control object after every rank has imported the startup
handles.

## Validation

Required coverage:

- TP1/2/4/8 Cache-hit dump/load for DeepSeek V4 FA/WA and GLM 5.2 layerwise
  tensor lists;
- Ascend and CUDA peer-import correctness with deterministic payload checks;
- Posix-hit load with AIO and direct I/O;
- backend failure, mismatched window tag, generation overrun, and timeout;
- repeated construction and teardown without surviving mappings or workers;
- byte-for-byte equality between native memory planning and vLLM reservation;
- performance comparison against baseline commit `ef2c66a` using identical
  window, slot, AIV-core, token, and concurrency settings.
