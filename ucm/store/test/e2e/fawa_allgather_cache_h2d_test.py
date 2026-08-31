from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing
import secrets
import statistics
import time

from _allgather_cache_store_benchmark import (
    _broadcast_remote_scatter_key,
    _find_free_port,
    _group_max_seconds,
    _make_block_ids,
    _parse_devices,
    _parse_numa_nodes,
    _percentile,
    _setup_device,
    _synchronize,
)

FA_TENSOR_BYTES = (131072, 16384, 256) * 21 + (4096,) * 20
WA_TENSOR_BYTES = (131072,) * 44 + (32768, 8192) * 21


def _aligned_size(sizes: tuple[int, ...]) -> int:
    return (sum(sizes) + 4095) // 4096 * 4096


def _make_tensors(torch, sizes: tuple[int, ...], blocks: int, device: str):
    return [
        [torch.empty(size // 2, dtype=torch.bfloat16, device=device) for size in sizes]
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
    remote_scatter_key: list[int],
    load_slots: int,
    window_blocks: int,
    scatter_only: bool,
    load_backend_only: bool,
    cache_numa_node: int | None,
) -> dict[str, object]:
    shard_size = _aligned_size(sizes)
    config: dict[str, object] = {
        "store_pipeline": "AllGather|Cache|Empty",
        "unique_id": f"{unique_id}-{stage}",
        "device_id": device_id,
        "tensor_size_list": list(sizes),
        "shard_size": shard_size,
        "block_size": shard_size,
        "share_buffer_enable": scatter_only,
        "local_rank_size": world_size if scatter_only else 1,
        "cache_load_backend_only": False,
        "cache_buffer_capacity_gb": 8,
        "cache_stream_number": 4,
        "cache_load_exclusive_buffer_number": 32,
        "cache_sdma_direct": False,
        "timeout_ms": 120000,
        "allgather_rank": rank,
        "allgather_world_size": world_size,
        "allgather_replicated_data": True,
        "allgather_scatter_only": scatter_only,
        "allgather_runtime_key": runtime_key,
        "allgather_remote_scatter_key": remote_scatter_key,
        "allgather_window_blocks_per_rank": window_blocks,
        "allgather_load_slots": load_slots,
        "allgather_dump_slots": 2,
        "allgather_separate_dump_queue": True,
        "allgather_load_backend_only": load_backend_only,
    }
    if cache_numa_node is not None:
        config["cache_numa_node"] = cache_numa_node
    return config


def _worker(
    rank: int,
    devices: tuple[int, ...],
    args: argparse.Namespace,
    unique_id: str,
    master_port: int,
) -> None:
    torch, device, backend = _setup_device(args.device_type, devices[rank])
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
        fa_roots = (
            []
            if args.scatter_only
            else _broadcast_remote_scatter_key(torch, dist, rank, device)
        )
        wa_roots = (
            fa_roots
            if args.share_runtime
            else (
                []
                if args.scatter_only
                else _broadcast_remote_scatter_key(torch, dist, rank, device)
            )
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
                args.fa_window_blocks,
                args.scatter_only,
                args.load_backend_only,
                None if args.cache_numa_nodes is None else args.cache_numa_nodes[rank],
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
                args.wa_window_blocks,
                args.scatter_only,
                args.load_backend_only,
                None if args.cache_numa_nodes is None else args.cache_numa_nodes[rank],
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
        _synchronize(torch, args.device_type)

        latencies = []
        for iteration in range(args.warmup + args.iterations):
            dist.barrier()
            _synchronize(torch, args.device_type)
            started = time.perf_counter()
            fa_task = fa_store.load(fa_ids, fa_indexes, fa_tensors)
            wa_task = wa_store.load(wa_ids, wa_indexes, wa_tensors)
            fa_store.wait(fa_task)
            wa_store.wait(wa_task)
            _synchronize(torch, args.device_type)
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
                "fa_blocks": args.fa_blocks,
                "fa_window_blocks": args.fa_window_blocks,
                "wa_window_blocks": args.wa_window_blocks,
                "scatter_only": args.scatter_only,
                "load_backend_only": args.load_backend_only,
                "mean_ms": mean_seconds * 1000,
                "p50_ms": _percentile(latencies, 50) * 1000,
                "p95_ms": _percentile(latencies, 95) * 1000,
                "unique_kv_gbps": unique_bytes / mean_seconds / 1e9,
                "aggregate_h2d_gbps": unique_bytes * len(devices) / mean_seconds / 1e9,
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
    parser.add_argument("--device-type", choices=("npu", "cuda"), default="npu")
    parser.add_argument("--cache-numa-nodes")
    parser.add_argument("--fa-blocks", type=int, default=240)
    parser.add_argument("--fa-window-blocks", type=int, default=4)
    parser.add_argument("--wa-window-blocks", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--block-id-seed", type=int, default=20260814)
    parser.add_argument("--share-runtime", action="store_true")
    parser.add_argument("--scatter-only", action="store_true")
    parser.add_argument("--load-backend-only", action="store_true")
    args = parser.parse_args()
    args.devices = _parse_devices(parser, args.devices)
    args.cache_numa_nodes = _parse_numa_nodes(
        parser, args.cache_numa_nodes, len(args.devices)
    )
    if (
        min(
            args.fa_blocks,
            args.fa_window_blocks,
            args.wa_window_blocks,
            args.iterations,
        )
        <= 0
        or args.warmup < 0
    ):
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
