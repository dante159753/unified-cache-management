from __future__ import annotations

import argparse
import datetime
import hashlib
import math
import multiprocessing
import secrets
import socket
import statistics
import time
from dataclasses import dataclass

DTYPE_BYTES = {
    "uint8": 1,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}


@dataclass(frozen=True)
class Workload:
    tensor_shapes: tuple[tuple[int, ...], ...]
    tensor_sizes: tuple[int, ...]
    logical_bytes: int
    shard_size: int


def _positive(parser: argparse.ArgumentParser, name: str, value: int) -> int:
    if value <= 0:
        parser.error(f"{name} must be positive, got {value}")
    return value


def _parse_shape(parser: argparse.ArgumentParser, value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(dimension) for dimension in value.lower().split("x"))
    except ValueError:
        parser.error(
            f"invalid tensor shape {value!r}; expected dimensions such as 64x512"
        )
    if not shape or any(dimension <= 0 for dimension in shape):
        parser.error(f"tensor shape must contain positive dimensions, got {value!r}")
    return shape


def _resolve_workload(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> Workload:
    element_size = DTYPE_BYTES[args.dtype]
    if args.tensor_bytes:
        try:
            sizes = tuple(int(value) for value in args.tensor_bytes.split(","))
        except ValueError:
            parser.error("--tensor-bytes must be a comma-separated integer list")
        if not sizes or any(size <= 0 for size in sizes):
            parser.error("--tensor-bytes must contain positive values")
        if any(size % element_size != 0 for size in sizes):
            parser.error(
                f"every tensor byte size must be divisible by dtype size {element_size}"
            )
        shapes = tuple((size // element_size,) for size in sizes)
    else:
        shapes = tuple(
            _parse_shape(parser, value) for value in (args.tensor_shape or ["65536"])
        )
        sizes = tuple(math.prod(shape) * element_size for shape in shapes)

    if args.tensor_count is not None:
        _positive(parser, "--tensor-count", args.tensor_count)
        if len(shapes) == 1:
            shapes = shapes * args.tensor_count
            sizes = sizes * args.tensor_count
        elif len(shapes) != args.tensor_count:
            parser.error(
                "--tensor-count must equal the number of --tensor-shape/--tensor-bytes "
                f"entries, got {args.tensor_count} and {len(shapes)}"
            )

    logical_bytes = sum(sizes)
    shard_size = (logical_bytes + 4095) // 4096 * 4096
    return Workload(shapes, sizes, logical_bytes, shard_size)


def _parse_devices(parser: argparse.ArgumentParser, value: str) -> tuple[int, ...]:
    try:
        devices = tuple(int(device) for device in value.split(","))
    except ValueError:
        parser.error("--devices must be a comma-separated integer list")
    if not devices or any(device < 0 for device in devices):
        parser.error("--devices must contain non-negative device ids")
    if len(set(devices)) != len(devices):
        parser.error("--devices must not contain duplicates")
    return devices


def _parse_numa_nodes(
    parser: argparse.ArgumentParser, value: str | None, world_size: int
) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        nodes = tuple(int(node) for node in value.split(","))
    except ValueError:
        parser.error("--cache-numa-nodes must be a comma-separated integer list")
    if len(nodes) != world_size or any(node < 0 for node in nodes):
        parser.error("--cache-numa-nodes must contain one non-negative node per device")
    return nodes


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_block_ids(seed: int, batch: int, block_count: int) -> list[bytes]:
    seed_bytes = seed.to_bytes(8, byteorder="little")
    batch_bytes = batch.to_bytes(8, byteorder="little")
    return [
        hashlib.sha256(
            seed_bytes + batch_bytes + index.to_bytes(8, byteorder="little")
        ).digest()[:16]
        for index in range(block_count)
    ]


def _owner_counts(
    seed: int, batch: int, block_count: int, world_size: int
) -> list[int]:
    counts = [0] * world_size
    for block_id in _make_block_ids(seed, batch, block_count):
        owner = int.from_bytes(block_id[:8], byteorder="little") % world_size
        counts[owner] += 1
    return counts


def _setup_device(device_type: str, device_id: int):
    import torch

    if device_type == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(device_id)
        return torch, f"npu:{device_id}", "hccl"
    torch.cuda.set_device(device_id)
    return torch, f"cuda:{device_id}", "nccl"


def _synchronize(torch, device_type: str) -> None:
    if device_type == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


def _broadcast_root_info(
    torch, dist, rank: int, device: str, group_count: int
) -> list[int]:
    from ucm.store.allgather import ucm_allgather_runtime

    size = int(ucm_allgather_runtime.root_info_size())
    result = []
    for _ in range(group_count):
        if rank == 0:
            root_info = ucm_allgather_runtime.create_root_info()
            tensor = torch.tensor(root_info, dtype=torch.uint8, device=device)
        else:
            tensor = torch.empty(size, dtype=torch.uint8, device=device)
        dist.broadcast(tensor, src=0)
        result.extend(tensor.cpu().tolist())
    return result


def _make_tensors(torch, workload: Workload, block_count: int, device: str, dtype: str):
    torch_dtype = getattr(torch, dtype)
    return [
        [
            torch.empty(shape, dtype=torch_dtype, device=device)
            for shape in workload.tensor_shapes
        ]
        for _ in range(block_count)
    ]


def _make_store_config(
    args: argparse.Namespace,
    direction: str,
    store_mode: str,
    workload: Workload,
    rank: int,
    world_size: int,
    device_id: int,
    unique_id: str,
    root_info: list[int],
) -> dict[str, object]:
    use_allgather = store_mode == "allgather"
    scatter_only = use_allgather and args.scatter_only
    backend = "Empty" if direction != "h2d" or args.verify_roundtrip else "Fake"
    pipeline = f"AllGather|Cache|{backend}" if use_allgather else f"Cache|{backend}"
    config: dict[str, object] = {
        "store_pipeline": pipeline,
        "unique_id": unique_id,
        "device_id": device_id,
        "tensor_size_list": list(workload.tensor_sizes),
        "shard_size": workload.shard_size,
        "block_size": workload.shard_size,
        "share_buffer_enable": not use_allgather or scatter_only,
        "cache_load_backend_only": False,
        "cache_buffer_capacity_gb": args.cache_capacity_gb,
        "cache_stream_number": args.cache_streams,
        "cache_sdma_direct": False,
        "io_direct": True,
        "timeout_ms": args.timeout_ms,
    }
    if scatter_only:
        config["local_rank_size"] = world_size
    if args.cache_numa_nodes is not None:
        config["cache_numa_node"] = args.cache_numa_nodes[rank]
    if use_allgather:
        replicated_data = not args.local_coalesced
        config.update(
            {
                "allgather_rank": rank,
                "allgather_world_size": world_size,
                "allgather_replicated_data": replicated_data,
                "allgather_scatter_only": scatter_only,
                "allgather_runtime_key": f"allgather-cache-benchmark-{unique_id}",
                "allgather_window_blocks_per_rank": args.window_blocks,
                "allgather_load_slots": args.load_slots,
                "allgather_load_groups": args.load_groups,
                "allgather_dump_slots": args.dump_slots,
                "allgather_collective_buffer_mb": args.collective_buffer_mb,
                "allgather_collective_mode": args.collective_mode,
                "allgather_profile_sample_every": args.profile_sample_every,
                "allgather_async_completion": not args.sync_completion,
                "allgather_separate_dump_queue": not args.shared_dump_queue,
                "allgather_collective_count_crop": not args.disable_count_crop,
                "allgather_collective_variable_counts": args.variable_counts,
                "allgather_dynamic_windows": not args.disable_dynamic_windows,
            }
        )
        if replicated_data and not scatter_only and world_size > 1:
            config["allgather_collective_root_info"] = root_info
    if direction == "h2d" and not args.verify_roundtrip:
        config["fake_always_hit"] = True
        config["fake_fail_load"] = args.fail_rank == rank
        if args.fake_load_delay_us:
            config["fake_load_delay_us"] = args.fake_load_delay_us
        config["buffer_number"] = max(1024, args.blocks * 4)
    return config


def _group_max_seconds(torch, dist, elapsed: float, device: str) -> float:
    value = torch.tensor([elapsed], dtype=torch.float32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def _run_iterations(
    torch,
    dist,
    store,
    args: argparse.Namespace,
    workload: Workload,
    direction: str,
    rank: int,
    world_size: int,
    device_type: str,
    device: str,
) -> list[float]:
    tensors = _make_tensors(torch, workload, args.blocks, device, args.dtype)
    dump_tensors = (
        _make_tensors(torch, workload, args.blocks, device, args.dtype)
        if args.mixed_dump
        else None
    )
    if dump_tensors is not None:
        for row_tensors in dump_tensors:
            for tensor in row_tensors:
                tensor.zero_()
        _synchronize(torch, device_type)
    shard_indexes = [0] * args.blocks
    timings: list[float] = []

    if direction == "h2d" and args.fail_rank >= 0:
        block_ids = _make_block_ids(args.block_id_seed, 0, args.blocks)
        task = store.load(block_ids, shard_indexes, tensors)
        failed = False
        try:
            store.wait(task)
        except RuntimeError:
            failed = True
        failure_count = torch.tensor([int(failed)], dtype=torch.int32, device=device)
        dist.all_reduce(failure_count)
        dist.barrier()
        if int(failure_count.item()) != world_size:
            raise RuntimeError(
                "expected the injected owner failure on every rank, got "
                f"{failure_count.item()}/{world_size}"
            )
        if rank == 0:
            print("ALLGATHER_FAILURE_DRAIN_PASS", flush=True)
        return timings

    if direction == "h2d":
        block_ids = _make_block_ids(args.block_id_seed, 0, args.blocks)
        if args.verify_roundtrip:
            for row, row_tensors in enumerate(tensors):
                for tensor_index, tensor in enumerate(row_tensors):
                    tensor.fill_((row * len(row_tensors) + tensor_index + 1) % 251)
            _synchronize(torch, device_type)
            dumped = store.dump(block_ids, shard_indexes, tensors)
            store.wait(dumped)
            _synchronize(torch, device_type)
            dist.barrier()
        if not args.verify_roundtrip:
            found = store.lookup(block_ids)
            if not bool(found.all()):
                raise RuntimeError(
                    "benchmark cache did not report every block as present"
                )
        if args.verify_roundtrip:
            for row_tensors in tensors:
                for tensor in row_tensors:
                    tensor.zero_()
        prime = store.load(block_ids, shard_indexes, tensors)
        store.wait(prime)
        _synchronize(torch, device_type)
        if args.verify_roundtrip:
            for row, row_tensors in enumerate(tensors):
                for tensor_index, tensor in enumerate(row_tensors):
                    expected = (row * len(row_tensors) + tensor_index + 1) % 251
                    if not bool(torch.all(tensor == expected).item()):
                        raise RuntimeError(
                            f"roundtrip mismatch at row {row}, tensor {tensor_index}"
                        )
            if rank == 0:
                print("ALLGATHER_CACHE_ROUNDTRIP_PASS", flush=True)

    for iteration in range(args.warmup + args.iterations):
        warmup = iteration < args.warmup
        block_ids = (
            _make_block_ids(args.block_id_seed, 0, args.blocks)
            if direction == "h2d"
            else _make_block_ids(args.block_id_seed, iteration + 1, args.blocks)
        )
        dist.barrier()
        _synchronize(torch, device_type)
        started = time.perf_counter()
        if dump_tensors is not None:
            tasks = []
            for task_index in range(args.inflight_tasks):
                dump_block_ids = _make_block_ids(
                    args.block_id_seed,
                    iteration * args.inflight_tasks + task_index + 10_000,
                    args.blocks,
                )
                tasks.append(store.dump(dump_block_ids, shard_indexes, dump_tensors))
                tasks.append(store.load(block_ids, shard_indexes, tensors))
        else:
            tasks = [
                (
                    store.load(block_ids, shard_indexes, tensors)
                    if direction == "h2d"
                    else store.dump(block_ids, shard_indexes, tensors)
                )
                for _ in range(args.inflight_tasks)
            ]
        for task in tasks:
            store.wait(task)
        _synchronize(torch, device_type)
        group_seconds = _group_max_seconds(
            torch, dist, time.perf_counter() - started, device
        )
        task_seconds = group_seconds / args.inflight_tasks
        if not warmup:
            timings.append(task_seconds)
        if rank == 0 and args.print_each:
            phase = "warmup" if warmup else "benchmark"
            physical_bytes = workload.shard_size * args.blocks
            print(
                f"phase={phase}, iteration={iteration:03d}, direction={direction}, "
                f"latency={task_seconds * 1e3:.3f}ms, "
                f"cache_bw={physical_bytes / task_seconds / 1e9:.3f}GB/s"
            )
    return timings


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _print_summary(
    direction: str,
    store_mode: str,
    timings: list[float],
    workload: Workload,
    args: argparse.Namespace,
    world_size: int,
) -> None:
    latencies_ms = [elapsed * 1e3 for elapsed in timings]
    physical_bytes = workload.shard_size * args.blocks
    logical_bytes = workload.logical_bytes * args.blocks
    owner_world_size = 1 if args.local_coalesced else world_size
    owner_counts = _owner_counts(args.block_id_seed, 0, args.blocks, owner_world_size)
    owner_window_count = max(
        (count + args.window_blocks - 1) // args.window_blocks for count in owner_counts
    )
    cache_bandwidths = [physical_bytes / elapsed / 1e9 for elapsed in timings]
    logical_multiplier = world_size if direction == "h2d" else 1
    logical_bandwidths = [
        logical_bytes * logical_multiplier / elapsed / 1e9 for elapsed in timings
    ]
    benchmark_name = (
        "Scatter-only Shared Cache"
        if args.scatter_only
        else ("AllGather Cache" if store_mode == "allgather" else "Shared Cache")
    )
    print(f"\n================ {benchmark_name} benchmark ================")
    print(
        f"direction={direction}, devices={args.devices}, world_size={world_size}, "
        f"blocks={args.blocks}, inflight_tasks={args.inflight_tasks}, dtype={args.dtype}, "
        f"mixed_dump={args.mixed_dump}, local_coalesced={args.local_coalesced}, "
        f"scatter_only={args.scatter_only}, "
        f"dynamic_windows={not args.disable_dynamic_windows}, "
        f"tensor_count={len(workload.tensor_shapes)}"
    )
    print(f"tensor_shapes={list(workload.tensor_shapes)}")
    print(
        f"tensor_sizes={list(workload.tensor_sizes)}, logical_bytes_per_block="
        f"{workload.logical_bytes}, cache_bytes_per_block={workload.shard_size}"
    )
    print(
        f"block_id_seed={args.block_id_seed}, owner_block_counts={owner_counts}, "
        f"owner_window_count={owner_window_count}"
    )
    print(
        f"latency_ms: avg={statistics.mean(latencies_ms):.3f}, "
        f"p50={_percentile(latencies_ms, 50):.3f}, "
        f"p90={_percentile(latencies_ms, 90):.3f}, "
        f"p99={_percentile(latencies_ms, 99):.3f}, "
        f"min={min(latencies_ms):.3f}, max={max(latencies_ms):.3f}"
    )
    print(
        f"cache_{direction}_GBps: avg={statistics.mean(cache_bandwidths):.3f}, "
        f"max={max(cache_bandwidths):.3f}"
    )
    logical_name = "materialized_logical" if direction == "h2d" else "logical"
    print(
        f"{logical_name}_GBps: avg={statistics.mean(logical_bandwidths):.3f}, "
        f"max={max(logical_bandwidths):.3f}"
    )


def _worker(
    rank: int,
    devices: tuple[int, ...],
    device_type: str,
    direction: str,
    store_mode: str,
    args: argparse.Namespace,
    workload: Workload,
    unique_id: str,
    master_port: int,
) -> None:
    torch, device, backend = _setup_device(device_type, devices[rank])
    import torch.distributed as dist

    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=len(devices),
        timeout=datetime.timedelta(milliseconds=args.timeout_ms),
    )
    store = None
    try:
        root_info = (
            _broadcast_root_info(torch, dist, rank, device, args.load_groups)
            if store_mode == "allgather"
            and not args.local_coalesced
            and not args.scatter_only
            and len(devices) > 1
            else []
        )
        from ucm.store.pipeline.connector import UcmPipelineStore

        config = _make_store_config(
            args,
            direction,
            store_mode,
            workload,
            rank,
            len(devices),
            devices[rank],
            unique_id,
            root_info,
        )
        store = UcmPipelineStore(config)
        timings = _run_iterations(
            torch,
            dist,
            store,
            args,
            workload,
            direction,
            rank,
            len(devices),
            device_type,
            device,
        )
        if rank == 0 and args.fail_rank < 0:
            _print_summary(direction, store_mode, timings, workload, args, len(devices))
    finally:
        if store is not None:
            store.close()
        dist.destroy_process_group()


def _build_parser(direction: str, store_mode: str) -> argparse.ArgumentParser:
    store_name = (
        "AllGatherStore + CacheStore"
        if store_mode == "allgather"
        else "shared CacheStore"
    )
    parser = argparse.ArgumentParser(
        description=(
            f"Benchmark {store_name} cache-hit H2D"
            if direction == "h2d"
            else f"Benchmark {store_name} D2H dump"
        )
    )
    parser.add_argument("--devices", default="0", help="comma-separated device ids")
    parser.add_argument(
        "--cache-numa-nodes",
        help="comma-separated CacheStore NUMA nodes in device order",
    )
    parser.add_argument("--device-type", choices=("npu", "cuda"), default="npu")
    parser.add_argument("--dtype", choices=tuple(DTYPE_BYTES), default="bfloat16")
    tensor_group = parser.add_mutually_exclusive_group()
    tensor_group.add_argument(
        "--tensor-shape",
        action="append",
        help="one tensor shape such as 64x512; repeat for multiple tensors",
    )
    tensor_group.add_argument(
        "--tensor-bytes",
        help="comma-separated tensor byte sizes, for example 131072,16384,32768",
    )
    parser.add_argument(
        "--tensor-count",
        type=int,
        help="repeat one shape/byte-size this many times, or validate an explicit list",
    )
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--inflight-tasks", type=int, default=1)
    parser.add_argument("--cache-capacity-gb", type=int, default=4)
    parser.add_argument("--cache-streams", type=int, default=4)
    parser.add_argument("--load-slots", type=int, default=2)
    parser.add_argument("--load-groups", type=int, default=1)
    parser.add_argument("--dump-slots", type=int, default=2)
    parser.add_argument("--window-blocks", type=int, default=4)
    parser.add_argument("--collective-buffer-mb", type=int, default=8)
    parser.add_argument(
        "--collective-mode",
        choices=("auto", "host", "aicpu_ts", "aiv"),
        default="host",
    )
    parser.add_argument("--timeout-ms", type=int, default=120000)
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--block-id-seed", type=int, default=20260812)
    parser.add_argument("--print-each", action="store_true")
    parser.add_argument("--fail-rank", type=int, default=-1)
    parser.add_argument("--profile-sample-every", type=int, default=0)
    parser.add_argument("--verify-roundtrip", action="store_true")
    parser.add_argument("--sync-completion", action="store_true")
    parser.add_argument("--mixed-dump", action="store_true")
    parser.add_argument("--shared-dump-queue", action="store_true")
    parser.add_argument("--local-coalesced", action="store_true")
    parser.add_argument("--scatter-only", action="store_true")
    parser.add_argument("--disable-count-crop", action="store_true")
    parser.add_argument("--variable-counts", action="store_true")
    parser.add_argument("--disable-dynamic-windows", action="store_true")
    parser.add_argument(
        "--fake-load-delay-us",
        type=lambda value: [int(delay) for delay in value.split(",")],
        default=[],
        help="comma-separated FakeStore load delays, cycled by backend submission order",
    )
    return parser


def run(direction: str, store_mode: str = "allgather") -> None:
    if store_mode not in ("allgather", "shared-cache"):
        raise ValueError(f"unsupported store mode: {store_mode}")
    parser = _build_parser(direction, store_mode)
    args = parser.parse_args()
    devices = _parse_devices(parser, args.devices)
    args.cache_numa_nodes = _parse_numa_nodes(
        parser, args.cache_numa_nodes, len(devices)
    )
    workload = _resolve_workload(parser, args)
    for name in (
        "blocks",
        "iterations",
        "inflight_tasks",
        "cache_capacity_gb",
        "cache_streams",
        "load_slots",
        "load_groups",
        "dump_slots",
        "window_blocks",
        "collective_buffer_mb",
        "timeout_ms",
    ):
        _positive(parser, f"--{name.replace('_', '-')}", getattr(args, name))
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.load_groups > args.load_slots:
        parser.error("--load-groups must not exceed --load-slots")
    if args.master_port < 0 or args.master_port > 65535:
        parser.error("--master-port must be 0 or a valid TCP port")
    if args.block_id_seed < 0 or args.block_id_seed >= 1 << 64:
        parser.error("--block-id-seed must fit in an unsigned 64-bit integer")
    if args.fail_rank < -1 or args.fail_rank >= len(devices):
        parser.error("--fail-rank must be -1 or a rank in --devices")
    if args.profile_sample_every < 0:
        parser.error("--profile-sample-every must be non-negative")
    if any(delay < 0 for delay in args.fake_load_delay_us):
        parser.error("--fake-load-delay-us must contain non-negative integers")
    if args.local_coalesced and store_mode != "allgather":
        parser.error("--local-coalesced requires the AllGatherStore benchmark")
    if args.scatter_only and store_mode != "allgather":
        parser.error("--scatter-only requires the AllGatherStore benchmark")
    if args.scatter_only and args.local_coalesced:
        parser.error("--scatter-only and --local-coalesced are mutually exclusive")
    if direction != "h2d" and args.inflight_tasks != 1:
        parser.error("--inflight-tasks is currently supported only for h2d")
    if direction != "h2d" and args.mixed_dump:
        parser.error("--mixed-dump is supported only for h2d")
    args.devices = ",".join(str(device) for device in devices)
    master_port = args.master_port or _find_free_port()
    unique_id = secrets.token_hex(8)
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_worker,
            args=(
                rank,
                devices,
                args.device_type,
                direction,
                store_mode,
                args,
                workload,
                unique_id,
                master_port,
            ),
        )
        for rank in range(len(devices))
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
                    f"benchmark worker pid={failed.pid} exited with code {failed.exitcode}"
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
