# Ascend SHM/NUMA H2D benchmark

`shm_numa_h2d_bench.cpp` only measures host shared-memory to NPU copies. It does
not initialize UCM, run a cache backend, perform collectives, or scatter data.

The three modes copy the same logical data set of `block_size * blocks` bytes to
every NPU:

- `shared-sequential`: one SHM segment contains the whole data set. Every rank
  registers that segment and traverses its blocks in the same order.
- `sharded-sequential`: rank `r` owns blocks whose index modulo TP equals `r`.
  Its SHM segment is bound to `numa_nodes[r]`. Every rank maps and registers all
  segments, then all ranks traverse logical blocks in the same order.
- `sharded-random`: allocation and registration are identical to
  `sharded-sequential`, but every rank independently shuffles the block order on
  every iteration. Shuffle time is outside the measured interval.

The aggregate SHM capacity remains `block_size * blocks`, not that amount per
rank. Every NPU receives the full data set, so the actual aggregate H2D bytes per
iteration are `block_size * blocks * TP`.

## Build

Run in an Ascend container with CANN installed:

```bash
ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
g++ -O3 -std=c++17 -D_GNU_SOURCE \
  -I"${ASCEND_HOME_PATH}/include" \
  ucm/store/test/e2e/shm_numa_h2d_bench.cpp \
  -L"${ASCEND_HOME_PATH}/lib64" -lascendcl -lpthread -lrt \
  -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
  -o /tmp/shm_numa_h2d_bench
```

The program requires CANN headers that provide `aclrtHostRegisterV2`.

## Run

For a 10 GiB logical data set made of 1 MiB blocks on TP4:

```bash
/tmp/shm_numa_h2d_bench \
  --tp 4 \
  --devices 0,1,2,3 \
  --block-size 1m \
  --blocks 10240 \
  --device-buffer-size 256m \
  --numa-nodes 0,2,4,6 \
  --shared-numa-node 0 \
  --mode all \
  --warmup 3 \
  --iters 10
```

`--numa-nodes` should be set to the NUMA nodes selected for the corresponding
NPUs. If omitted, online NUMA nodes are assigned round-robin. The single-buffer
baseline is allocated on `--shared-numa-node`; it defaults to the first entry in
`--numa-nodes`. Set it to `-1` to leave placement to the kernel.

`--device-buffer-size` makes the benchmark reuse a ring of HBM destinations. It
does not change source traversal or transferred bytes, and is useful when the
selected NPUs do not have enough free HBM for a destination as large as the
logical data set. If omitted, the destination occupies the full logical size.

The process needs permission to call `mbind`, `/dev/shm` must have at least the
logical data size available. Each selected NPU needs either that much free HBM
or the amount selected by `--device-buffer-size`. Modes run sequentially and
release their SHM mappings before the next mode starts.

The result reports two bandwidths:

- `unique_GBps`: logical KV bytes divided by the slowest-rank latency.
- `delivered_GBps`: actual bytes copied by all TP ranks divided by that latency.

`per_npu_GBps` equals `unique_GBps` because every NPU receives the full data set.
