from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing
import secrets
import statistics
import time

from _allgather_cache_store_benchmark import (
    _broadcast_root_info,
    _find_free_port,
    _group_max_seconds,
    _make_block_ids,
    _parse_devices,
    _percentile,
    _setup_device,
    _synchronize,
)


FA_TENSOR_BYTES = (131072, 16384, 256) * 21
WA_TENSOR_BYTES = (131072,) * 43 + (32768, 8192) * 21


def _aligned_size(sizes: tuple[int, ...]) -> int:
    return (sum(sizes) + 4095) // 4096 * 4096


def _make_tensors(torch, sizes: tuple[int, ...], blocks: int, device: str):
    return [
        [
            torch.empty(size // 2, dtype=torch.bfloat16, device=device)
            for size in sizes
        ]
        for _ in range(blocks)
    ]


def _store_config(
    stage: str,
    sizes: tuple[int, ...],
    rank: int,
    world_size: int,
    device_id: int,
    unique_id: str,
    runtime_key: str,
    root_info: list[int],
    load_slots: int,
    load_groups: int,
    window_blocks: int,
    buffer_mb: int,
    variable_counts: bool,
) -> dict[str, object]:
    shard_size = _aligned_size(sizes)
    return {
        "store_pipeline": "AllGather|Cache|Empty",
        "unique_id": f"{unique_id}-{stage}",
        "device_id": device_id,
        "tensor_size_list": list(sizes),
        "shard_size": shard_size,
        "block_size": shard_size,
        "share_buffer_enable": False,
        "cache_load_backend_only": False,
        "cache_buffer_capacity_gb": 8,
        "cache_stream_number": 4,
        "cache_load_exclusive_buffer_number": 32,
        "cache_numa_node": rank,
        "cache_sdma_direct": False,
        "timeout_ms": 120000,
        "allgather_rank": rank,
        "allgather_world_size": world_size,
        "allgather_replicated_data": True,
        "allgather_runtime_key": runtime_key,
        "allgather_collective_root_info": root_info,
        "allgather_window_blocks_per_rank": window_blocks,
        "allgather_load_slots": load_slots,
        "allgather_load_groups": load_groups,
        "allgather_dump_slots": 2,
        "allgather_collective_buffer_mb": buffer_mb,
        "allgather_async_completion": True,
        "allgather_separate_dump_queue": True,
        "allgather_collective_variable_counts": variable_counts,
    }


def _worker(
    rank: int,
    devices: tuple[int, ...],
    args: argparse.Namespace,
    unique_id: str,
    master_port: int,
) -> None:
    torch, device, backend = _setup_device("npu", devices[rank])
    import torch.distributed as dist

    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=len(devices),
        timeout=datetime.timedelta(milliseconds=120000),
    )
    fa_store = None
    wa_store = None
    try:
        fa_roots = _broadcast_root_info(torch, dist, rank, device, 4)
        wa_roots = (
            fa_roots
            if args.share_runtime
            else _broadcast_root_info(torch, dist, rank, device, 1)
        )
        runtime_base = f"fawa-cache-benchmark-{unique_id}"
        fa_runtime = runtime_base if args.share_runtime else f"{runtime_base}:fa"
        wa_runtime = runtime_base if args.share_runtime else f"{runtime_base}:wa"

        from ucm.store.pipeline.connector import UcmPipelineStore

        fa_store = UcmPipelineStore(
            _store_config(
                "fa",
                FA_TENSOR_BYTES,
                rank,
                len(devices),
                devices[rank],
                unique_id,
                fa_runtime,
                fa_roots,
                4,
                4,
                args.fa_window_blocks,
                args.buffer_mb,
                args.variable_counts,
            )
        )
        wa_store = UcmPipelineStore(
            _store_config(
                "wa",
                WA_TENSOR_BYTES,
                rank,
                len(devices),
                devices[rank],
                unique_id,
                wa_runtime,
                wa_roots,
                1,
                1,
                args.wa_window_blocks,
                args.buffer_mb,
                args.variable_counts,
            )
        )

        fa_ids = _make_block_ids(args.block_id_seed, 0, args.fa_blocks)
        wa_ids = fa_ids[-1:]
        fa_indexes = [0] * len(fa_ids)
        wa_indexes = [0]
        fa_tensors = _make_tensors(torch, FA_TENSOR_BYTES, len(fa_ids), device)
        wa_tensors = _make_tensors(torch, WA_TENSOR_BYTES, 1, device)

        fa_dump = fa_store.dump(fa_ids, fa_indexes, fa_tensors)
        wa_dump = wa_store.dump(wa_ids, wa_indexes, wa_tensors)
        fa_store.wait(fa_dump)
        wa_store.wait(wa_dump)
        _synchronize(torch, "npu")

        latencies = []
        for iteration in range(args.warmup + args.iterations):
            dist.barrier()
            _synchronize(torch, "npu")
            started = time.perf_counter()
            fa_task = fa_store.load(fa_ids, fa_indexes, fa_tensors)
            wa_task = wa_store.load(wa_ids, wa_indexes, wa_tensors)
            fa_store.wait(fa_task)
            wa_store.wait(wa_task)
            _synchronize(torch, "npu")
            elapsed = _group_max_seconds(
                torch, dist, time.perf_counter() - started, device
            )
            if iteration >= args.warmup:
                latencies.append(elapsed)

        if rank == 0:
            unique_bytes = _aligned_size(FA_TENSOR_BYTES) * args.fa_blocks
            unique_bytes += _aligned_size(WA_TENSOR_BYTES)
            mean_seconds = statistics.mean(latencies)
            result = {
                "runtime": "shared" if args.share_runtime else "split",
                "world_size": len(devices),
                "buffer_mb": args.buffer_mb,
                "fa_blocks": args.fa_blocks,
                "fa_window_blocks": args.fa_window_blocks,
                "wa_window_blocks": args.wa_window_blocks,
                "variable_counts": args.variable_counts,
                "mean_ms": mean_seconds * 1000,
                "p50_ms": _percentile(latencies, 50) * 1000,
                "p95_ms": _percentile(latencies, 95) * 1000,
                "unique_kv_gbps": unique_bytes / mean_seconds / 1e9,
                "aggregate_h2d_gbps": unique_bytes
                * len(devices)
                / mean_seconds
                / 1e9,
            }
            print("FAWA_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    finally:
        if wa_store is not None:
            wa_store.close()
        if fa_store is not None:
            fa_store.close()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--fa-blocks", type=int, default=240)
    parser.add_argument("--fa-window-blocks", type=int, default=4)
    parser.add_argument("--wa-window-blocks", type=int, default=4)
    parser.add_argument("--buffer-mb", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--block-id-seed", type=int, default=20260814)
    parser.add_argument("--share-runtime", action="store_true")
    parser.add_argument("--variable-counts", action="store_true")
    args = parser.parse_args()
    args.devices = _parse_devices(parser, args.devices)
    if min(
        args.fa_blocks,
        args.fa_window_blocks,
        args.wa_window_blocks,
        args.buffer_mb,
        args.iterations,
    ) <= 0 or args.warmup < 0:
        parser.error("numeric arguments must be positive, and warmup non-negative")

    context = multiprocessing.get_context("spawn")
    unique_id = secrets.token_hex(8)
    master_port = _find_free_port()
    processes = [
        context.Process(
            target=_worker,
            args=(rank, args.devices, args, unique_id, master_port),
        )
        for rank in range(len(args.devices))
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    if failures:
        raise SystemExit(f"workers failed: {failures}")


if __name__ == "__main__":
    main()
