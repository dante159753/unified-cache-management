from __future__ import annotations

import argparse
import datetime
import json
import math
import multiprocessing
import os
import random
import secrets
import statistics
import time

from _allgather_cache_store_benchmark import (
    _broadcast_root_info,
    _find_free_port,
    _group_max_seconds,
    _make_block_ids,
    _parse_devices,
    _parse_numa_nodes,
    _percentile,
    _setup_device,
    _synchronize,
)

DEFAULT_TENSOR_BYTES = (131072, 16384, 32768)


def _parse_tensor_bytes(parser: argparse.ArgumentParser, value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(size) for size in value.split(","))
    except ValueError:
        parser.error("--tensor-bytes must be a comma-separated integer list")
    if not sizes or any(size <= 0 or size % 2 != 0 for size in sizes):
        parser.error("--tensor-bytes must contain positive bfloat16-aligned sizes")
    return sizes


def _aligned_size(sizes: tuple[int, ...]) -> int:
    return (sum(sizes) + 4095) // 4096 * 4096


def _owner(block_id: bytes, world_size: int) -> int:
    return int.from_bytes(block_id[:8], byteorder="little") % world_size


def _make_requests(
    seed: int, round_index: int, concurrency: int, blocks_per_request: int
) -> list[list[bytes]]:
    batch_base = round_index * concurrency
    return [
        _make_block_ids(seed, batch_base + request, blocks_per_request)
        for request in range(concurrency)
    ]


def _make_tensors(torch, sizes: tuple[int, ...], blocks: int, device: str):
    return [
        [torch.empty(size // 2, dtype=torch.bfloat16, device=device) for size in sizes]
        for _ in range(blocks)
    ]


def _make_aligned_tensor(torch, size: int, device: str, alignment: int = 4096):
    storage = torch.empty(size + alignment, dtype=torch.uint8, device=device)
    offset = (-storage.data_ptr()) % alignment
    tensor = storage[offset : offset + size].view(torch.bfloat16)
    tensor.storage_ref = storage
    return tensor


def _base_config(
    args: argparse.Namespace,
    tensor_sizes: tuple[int, ...],
    shard_size: int,
    device_id: int,
    unique_id: str,
) -> dict[str, object]:
    return {
        "unique_id": unique_id,
        "device_id": device_id,
        "tensor_size_list": list(tensor_sizes),
        "shard_size": shard_size,
        "block_size": shard_size,
        "storage_backends": list(args.storage_backends),
        "posix_io_engine": "aio",
        "io_direct": True,
        "posix_data_trans_concurrency": args.posix_concurrency,
        "posix_lookup_concurrency": args.posix_lookup_concurrency,
        "posix_gc_enable": False,
        "timeout_ms": args.timeout_ms,
    }


def _make_posix_seed_store(
    args: argparse.Namespace,
    tensor_sizes: tuple[int, ...],
    shard_size: int,
    rank: int,
    device_id: int,
    unique_id: str,
):
    from ucm.store.pipeline.connector import UcmPipelineStore

    config = _base_config(
        args, tensor_sizes, shard_size, device_id, f"{unique_id}-seed-r{rank}"
    )
    config.update(
        {
            "store_pipeline": "Posix",
            "tensor_size": shard_size,
            "tensor_size_list": [shard_size],
            "share_buffer_enable": True,
        }
    )
    return UcmPipelineStore(config)


def _make_load_store(
    args: argparse.Namespace,
    tensor_sizes: tuple[int, ...],
    shard_size: int,
    rank: int,
    world_size: int,
    device_id: int,
    unique_id: str,
    root_info: list[int],
):
    from ucm.store.pipeline.connector import UcmPipelineStore

    config = _base_config(
        args, tensor_sizes, shard_size, device_id, f"{unique_id}-load-r{rank}"
    )
    config.update(
        {
            "store_pipeline": "AllGather|Cache|Posix",
            "share_buffer_enable": False,
            "cache_load_backend_only": False,
            "cache_buffer_capacity_gb": args.cache_capacity_gb,
            "cache_stream_number": args.cache_streams,
            "cache_load_exclusive_buffer_number": 32,
            "cache_sdma_direct": False,
            "allgather_rank": rank,
            "allgather_world_size": world_size,
            "allgather_replicated_data": True,
            "allgather_runtime_key": f"allgather-cache-posix-mixed-{unique_id}",
            "allgather_collective_root_info": root_info,
            "allgather_window_blocks_per_rank": args.window_blocks,
            "allgather_load_slots": args.load_slots,
            "allgather_receive_slots": args.receive_slots,
            "allgather_dump_slots": 1,
            "allgather_collective_buffer_mb": args.collective_buffer_mb,
            "allgather_collective_mode": args.collective_mode,
            "allgather_collective_variable_counts": args.variable_counts,
            "allgather_scatter_aiv_cores": args.scatter_aiv_cores,
            "allgather_profile_sample_every": args.profile_sample_every,
            "allgather_async_completion": True,
            "allgather_separate_dump_queue": True,
        }
    )
    if args.cache_numa_nodes is not None:
        config["cache_numa_node"] = args.cache_numa_nodes[rank]
    return UcmPipelineStore(config)


def _seed_posix(
    torch,
    store,
    requests: list[list[bytes]],
    rank: int,
    world_size: int,
    shard_size: int,
) -> None:
    packed = _make_aligned_tensor(torch, shard_size, "cpu")
    packed.fill_(rank + 1)
    tasks = []
    for block_ids in requests:
        owned = [
            block_id for block_id in block_ids if _owner(block_id, world_size) == rank
        ]
        if owned:
            tasks.append(store.dump(owned, [0] * len(owned), [[packed]] * len(owned)))
    for task in tasks:
        store.wait(task)

    owned_ids = [
        block_id
        for request in requests
        for block_id in request
        if _owner(block_id, world_size) == rank
    ]
    if owned_ids and not bool(store.lookup(owned_ids).all()):
        raise RuntimeError(f"rank {rank} failed to seed every owned block into POSIX")


def _warm_selection(
    requests: list[list[bytes]], cache_ratio: float, seed: int, round_index: int
) -> tuple[list[int], list[int]]:
    total = sum(len(request) for request in requests)
    warm_count = round(total * cache_ratio)
    rng = random.Random(seed ^ (round_index * 0x9E3779B97F4A7C15))
    positions = sorted(rng.sample(range(total), warm_count))
    per_request = [0] * len(requests)
    request_size = len(requests[0])
    for position in positions:
        per_request[position // request_size] += 1
    return positions, per_request


def _warm_cache(
    store,
    requests: list[list[bytes]],
    tensors,
    positions: list[int],
) -> None:
    flat_ids = [block_id for request in requests for block_id in request]
    warm_ids = [flat_ids[position] for position in positions]
    warm_tensors = [tensors[position] for position in positions]
    if not warm_ids:
        return
    task = store.load(warm_ids, [0] * len(warm_ids), warm_tensors)
    store.wait(task)


def _run_round(
    torch,
    dist,
    store,
    requests: list[list[bytes]],
    tensors,
    device: str,
    batch_load_requests: bool,
) -> dict[str, float]:
    request_tensors = []
    offset = 0
    for request in requests:
        request_tensors.append(tensors[offset : offset + len(request)])
        offset += len(request)

    dist.barrier()
    _synchronize(torch, "npu")
    started = time.perf_counter()
    if batch_load_requests:
        flat_ids = [block_id for request in requests for block_id in request]
        tasks = [store.load(flat_ids, [0] * len(flat_ids), tensors)]
    else:
        tasks = [
            store.load(block_ids, [0] * len(block_ids), dst)
            for block_ids, dst in zip(requests, request_tensors)
        ]
    submitted = time.perf_counter()
    for task in tasks:
        store.wait(task)
    _synchronize(torch, "npu")
    completed = time.perf_counter()
    return {
        "submit_seconds": _group_max_seconds(torch, dist, submitted - started, device),
        "wait_seconds": _group_max_seconds(torch, dist, completed - submitted, device),
        "total_seconds": _group_max_seconds(torch, dist, completed - started, device),
    }


def _worker(
    rank: int,
    devices: tuple[int, ...],
    args: argparse.Namespace,
    tensor_sizes: tuple[int, ...],
    shard_size: int,
    unique_id: str,
    master_port: int,
) -> None:
    torch, device, backend = _setup_device("npu", devices[rank])
    import torch.distributed as dist

    world_size = len(devices)
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(milliseconds=args.timeout_ms),
    )
    seed_store = None
    load_store = None
    try:
        round_count = args.warmup + args.iterations
        requests_by_round = [
            _make_requests(
                args.block_id_seed,
                round_index,
                args.concurrency,
                args.blocks_per_request,
            )
            for round_index in range(round_count)
        ]

        seed_store = _make_posix_seed_store(
            args,
            tensor_sizes,
            shard_size,
            rank,
            devices[rank],
            unique_id,
        )
        seed_started = time.perf_counter()
        for requests in requests_by_round:
            _seed_posix(
                torch,
                seed_store,
                requests,
                rank,
                world_size,
                shard_size,
            )
        _synchronize(torch, "npu")
        seed_store.close()
        seed_store = None
        dist.barrier()
        seed_seconds = _group_max_seconds(
            torch, dist, time.perf_counter() - seed_started, device
        )

        root_info = _broadcast_root_info(torch, dist, rank, device)
        load_store = _make_load_store(
            args,
            tensor_sizes,
            shard_size,
            rank,
            world_size,
            devices[rank],
            unique_id,
            root_info,
        )
        tensors = _make_tensors(
            torch,
            tensor_sizes,
            args.concurrency * args.blocks_per_request,
            device,
        )

        records = []
        placement = None
        for round_index, requests in enumerate(requests_by_round):
            positions, per_request = _warm_selection(
                requests, args.cache_ratio, args.cache_seed, round_index
            )
            _warm_cache(load_store, requests, tensors, positions)
            _synchronize(torch, "npu")
            record = _run_round(
                torch,
                dist,
                load_store,
                requests,
                tensors,
                device,
                args.batch_load_requests,
            )
            if placement is None:
                flat_ids = [block_id for request in requests for block_id in request]
                position_set = set(positions)
                owner_blocks = [0] * world_size
                owner_cache_blocks = [0] * world_size
                for position, block_id in enumerate(flat_ids):
                    owner = _owner(block_id, world_size)
                    owner_blocks[owner] += 1
                    if position in position_set:
                        owner_cache_blocks[owner] += 1
                placement = {
                    "owner_blocks": owner_blocks,
                    "owner_cache_blocks": owner_cache_blocks,
                    "cache_blocks_per_request_min": min(per_request),
                    "cache_blocks_per_request_max": max(per_request),
                    "cache_blocks_per_request_mean": statistics.mean(per_request),
                }
            if round_index >= args.warmup:
                records.append(record)
            if rank == 0 and args.print_each:
                print(
                    "MIXED_ROUND "
                    + json.dumps(
                        {
                            "round": round_index,
                            "warmup": round_index < args.warmup,
                            "submit_ms": record["submit_seconds"] * 1000,
                            "wait_ms": record["wait_seconds"] * 1000,
                            "total_ms": record["total_seconds"] * 1000,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        if rank == 0:
            total_blocks = args.concurrency * args.blocks_per_request
            cache_blocks = round(total_blocks * args.cache_ratio)
            posix_blocks = total_blocks - cache_blocks
            totals = [record["total_seconds"] for record in records]
            submits = [record["submit_seconds"] for record in records]
            waits = [record["wait_seconds"] for record in records]
            mean_seconds = statistics.mean(totals)
            result = {
                "world_size": world_size,
                "devices": list(devices),
                "input_tokens": args.input_tokens,
                "concurrency": args.concurrency,
                "batch_load_requests": args.batch_load_requests,
                "aggregate_input_tokens": args.input_tokens * args.concurrency,
                "block_tokens": args.block_tokens,
                "blocks_per_request": args.blocks_per_request,
                "total_blocks": total_blocks,
                "cache_ratio": args.cache_ratio,
                "cache_blocks": cache_blocks,
                "posix_blocks": posix_blocks,
                "tensor_sizes": list(tensor_sizes),
                "shard_size": shard_size,
                "window_blocks_per_rank": args.window_blocks,
                "load_slots": args.load_slots,
                "receive_slots": args.receive_slots,
                "collective_mode": args.collective_mode,
                "variable_counts": args.variable_counts,
                "scatter_aiv_cores": args.scatter_aiv_cores,
                "iterations": args.iterations,
                "seed_posix_seconds": seed_seconds,
                "submit_ms_mean": statistics.mean(submits) * 1000,
                "wait_ms_mean": statistics.mean(waits) * 1000,
                "total_ms_mean": mean_seconds * 1000,
                "total_ms_p50": _percentile(totals, 50) * 1000,
                "total_ms_p90": _percentile(totals, 90) * 1000,
                "unique_kv_gbps": total_blocks * shard_size / mean_seconds / 1e9,
                "aggregate_h2d_gbps": (
                    total_blocks * shard_size * world_size / mean_seconds / 1e9
                ),
                "input_tokens_per_second": (
                    args.input_tokens * args.concurrency / mean_seconds
                ),
                **placement,
            }
            print(
                "ALLGATHER_CACHE_POSIX_MIXED_RESULT "
                + json.dumps(result, sort_keys=True),
                flush=True,
            )
    finally:
        if load_store is not None:
            load_store.close()
        if seed_store is not None:
            seed_store.close()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark AllGather|Cache|Posix after seeding every block in POSIX and "
            "a random subset in Cache"
        )
    )
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--cache-numa-nodes")
    parser.add_argument(
        "--storage-backend",
        action="append",
        dest="storage_backends",
        default=None,
        help="POSIX data directory; repeat to use multiple backends",
    )
    parser.add_argument(
        "--tensor-bytes", default=",".join(str(size) for size in DEFAULT_TENSOR_BYTES)
    )
    parser.add_argument("--input-tokens", type=int, default=16 * 1024)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--block-tokens", type=int, default=128)
    parser.add_argument("--cache-ratio", type=float, default=0.4)
    parser.add_argument("--cache-seed", type=int, default=20260825)
    parser.add_argument("--block-id-seed", type=int, default=20260826)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--window-mb", type=float, default=8.0)
    parser.add_argument("--window-blocks", type=int, default=0)
    parser.add_argument("--load-slots", type=int, default=4)
    parser.add_argument("--receive-slots", type=int, default=4)
    parser.add_argument("--cache-capacity-gb", type=int, default=4)
    parser.add_argument("--cache-streams", type=int, default=4)
    parser.add_argument("--collective-buffer-mb", type=int, default=8)
    parser.add_argument(
        "--collective-mode",
        choices=("auto", "host", "aicpu_ts", "aiv"),
        default="host",
    )
    parser.add_argument("--disable-variable-counts", action="store_true")
    parser.add_argument("--scatter-aiv-cores", type=int, default=1)
    parser.add_argument("--posix-concurrency", type=int, default=128)
    parser.add_argument("--posix-lookup-concurrency", type=int, default=32)
    parser.add_argument("--profile-sample-every", type=int, default=0)
    parser.add_argument("--timeout-ms", type=int, default=300000)
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--print-each", action="store_true")
    parser.add_argument("--batch-load-requests", action="store_true")
    args = parser.parse_args()

    args.devices = _parse_devices(parser, args.devices)
    args.cache_numa_nodes = _parse_numa_nodes(
        parser, args.cache_numa_nodes, len(args.devices)
    )
    args.storage_backends = args.storage_backends or [
        "/mnt/kvcache-local/ucm-allgather-cache-posix-mixed"
    ]
    tensor_sizes = _parse_tensor_bytes(parser, args.tensor_bytes)
    shard_size = _aligned_size(tensor_sizes)
    if (
        min(
            args.input_tokens,
            args.concurrency,
            args.block_tokens,
            args.iterations,
            args.load_slots,
            args.receive_slots,
            args.cache_capacity_gb,
            args.cache_streams,
            args.collective_buffer_mb,
            args.scatter_aiv_cores,
            args.posix_concurrency,
            args.posix_lookup_concurrency,
            args.timeout_ms,
        )
        <= 0
    ):
        parser.error("numeric size and concurrency arguments must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if not 0.0 <= args.cache_ratio <= 1.0:
        parser.error("--cache-ratio must be in [0, 1]")
    if args.window_mb <= 0:
        parser.error("--window-mb must be positive")
    if args.window_blocks < 0:
        parser.error("--window-blocks must be non-negative")
    if args.scatter_aiv_cores > 40:
        parser.error("--scatter-aiv-cores must not exceed 40")
    if not 0 <= args.master_port <= 65535:
        parser.error("--master-port must be 0 or a valid TCP port")
    if args.block_id_seed < 0 or args.block_id_seed >= 1 << 64:
        parser.error("--block-id-seed must fit in an unsigned 64-bit integer")

    args.blocks_per_request = math.ceil(args.input_tokens / args.block_tokens)
    args.window_blocks = args.window_blocks or max(
        4, int(args.window_mb * 1024 * 1024) // shard_size
    )
    args.variable_counts = not args.disable_variable_counts
    for path in args.storage_backends:
        os.makedirs(path, exist_ok=True)

    context = multiprocessing.get_context("spawn")
    unique_id = secrets.token_hex(8)
    master_port = args.master_port or _find_free_port()
    processes = [
        context.Process(
            target=_worker,
            args=(
                rank,
                args.devices,
                args,
                tensor_sizes,
                shard_size,
                unique_id,
                master_port,
            ),
        )
        for rank in range(len(args.devices))
    ]
    try:
        for process in processes:
            process.start()
        while any(process.is_alive() for process in processes):
            failed = next(
                (process for process in processes if process.exitcode not in (None, 0)),
                None,
            )
            if failed is not None:
                raise RuntimeError(
                    f"benchmark worker pid={failed.pid} exited with code "
                    f"{failed.exitcode}"
                )
            time.sleep(0.1)
        failed = next((process for process in processes if process.exitcode != 0), None)
        if failed is not None:
            raise RuntimeError(
                f"benchmark worker pid={failed.pid} exited with code {failed.exitcode}"
            )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()


if __name__ == "__main__":
    main()
