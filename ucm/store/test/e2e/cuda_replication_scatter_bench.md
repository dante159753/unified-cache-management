# CUDA replication + scatter benchmark

Production UCM uses `remote_scatter`. This standalone benchmark retains the
other two paths only as diagnostic comparison baselines:

- `ipc_pull`: each rank pulls every owner's packed allocation with
  `cudaMemcpyPeerAsync`, then scatters from a local receive buffer;
- `remote_scatter`: the scatter kernel reads the owners' CUDA IPC allocations
  directly;
- `multicast`: every owner broadcasts its packed segment with Hopper
  `multimem.st` into CUDA multicast/VMM replicas, then each rank scatters from
  its local replica.

The parent process forks one worker per visible GPU. CUDA is initialized only in
the children. Owner buffers and interprocess events are exchanged with CUDA IPC;
the multicast object is exchanged with a Unix-domain socket and a POSIX file
descriptor. No NCCL dependency is required.

## Build on H100

CUDA multicast requires CUDA 12.1 or newer, an `sm_90` build, NVSwitch multicast
support, and a working Fabric Manager.

```bash
nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo \
  ucm/store/test/e2e/cuda_replication_scatter_bench.cu \
  -o /tmp/cuda_replication_scatter_bench \
  -lcuda -Xcompiler -pthread
```

## Run

Tensor sizes are bytes per block. `K` and `M` suffixes use powers of two.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/tmp/cuda_replication_scatter_bench \
  --mode remote_scatter \
  --tp 4 \
  --blocks 64 \
  --max-transfer-blocks 16 \
  --tensor-sizes 131072,16384,256 \
  --warmup 10 \
  --iters 100
```

Use a real model's complete per-block tensor list, for example:

```bash
/tmp/cuda_replication_scatter_bench \
  --mode remote_scatter \
  --tp 8 \
  --blocks 256 \
  --tensor-sizes 151552,4096 \
  --owner hash \
  --seed 1
```

Relevant options:

```text
--mode all|ipc_pull|remote_scatter|multicast
--tp N
--blocks N
--max-transfer-blocks N
--tensor-sizes BYTES[,BYTES...]
--owner hash|round_robin
--scatter-ctas N
--warmup N
--iters N
--no-verify
```

`--blocks` is the total number of logical blocks completed by one measured
iteration. `--max-transfer-blocks` bounds each transfer/scatter window; `0` or
omitting it uses one window containing all blocks. The final partial window uses
its actual block count, and each owner transfers only its blocks in that window.

`critical_*` is the maximum latency across ranks. `unique_GBps` counts each
logical block once; `delivered_GBps` counts the copy present on every rank and is
therefore `TP * unique_GBps`. Verification samples the first, middle, and last
byte of every `(block, tensor)` destination.

The timed region includes all transfer/scatter windows and the multicast
readiness synchronization required between producer and consumer processes. It
excludes buffer allocation, IPC handle exchange, data generation, and
correctness verification. Multicast is reported as skipped when the toolkit or
any selected device lacks multicast support.
