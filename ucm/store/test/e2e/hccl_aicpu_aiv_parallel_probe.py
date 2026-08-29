#!/usr/bin/env python3

import argparse
import datetime
import os
import statistics
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure whether HCCL AI CPU+TS and AIV collectives overlap."
    )
    parser.add_argument("--mb", type=float, default=32.0)
    parser.add_argument("--aicpu-mb", type=float)
    parser.add_argument("--aiv-mb", type=float)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--hccl-buffer-mb", type=int, default=8)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument(
        "--submit-order",
        choices=("aiv-first", "aicpu-first"),
        default="aiv-first",
    )
    args = parser.parse_args()
    for name in ("mb", "iters", "rounds", "hccl_buffer_mb", "timeout_s"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("aicpu_mb", "aiv_mb"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    args.aicpu_mb = args.mb if args.aicpu_mb is None else args.aicpu_mb
    args.aiv_mb = args.mb if args.aiv_mb is None else args.aiv_mb
    return args


def _group_options(torch_npu, mode: int, buffer_mb: int):
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    options.hccl_config = {
        "hccl_buffer_size": buffer_mb,
        "hccl_op_expansion_mode": mode,
    }
    return options


def _new_group(dist, torch_npu, mode: int, buffer_mb: int):
    ranks = list(range(dist.get_world_size()))
    group = dist.new_group(
        ranks,
        backend="hccl",
        pg_options=_group_options(torch_npu, mode, buffer_mb),
    )
    dist.barrier(group=group)
    return group


def _enqueue(dist, group, output, source, iters: int):
    works = []
    for _ in range(iters):
        works.append(
            dist.all_gather_into_tensor(
                output,
                source,
                group=group,
                async_op=True,
            )
        )
    return works


def _wait(works) -> None:
    for work in works:
        work.wait()


def _submit(dist, group, output, source, iters: int) -> None:
    _wait(_enqueue(dist, group, output, source, iters))


def _measure(torch, dist, operation) -> float:
    dist.barrier()
    torch.npu.synchronize()
    start = time.perf_counter()
    operation()
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    maximum = torch.tensor([elapsed_ms], dtype=torch.float32, device="npu")
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(maximum.item())


def _measure_concurrent_completion(
    torch,
    dist,
    aicpu_group,
    aiv_group,
    aicpu_output,
    aicpu_source,
    aiv_output,
    aiv_source,
    iters: int,
    timeout_s: int,
    submit_order: str,
) -> tuple[float, float, float]:
    dist.barrier()
    torch.npu.synchronize()
    start = time.perf_counter()
    aicpu_works = []
    aiv_works = []
    for _ in range(iters):
        if submit_order == "aiv-first":
            aiv_works.extend(_enqueue(dist, aiv_group, aiv_output, aiv_source, 1))
            aicpu_works.extend(
                _enqueue(dist, aicpu_group, aicpu_output, aicpu_source, 1)
            )
        else:
            aicpu_works.extend(
                _enqueue(dist, aicpu_group, aicpu_output, aicpu_source, 1)
            )
            aiv_works.extend(_enqueue(dist, aiv_group, aiv_output, aiv_source, 1))

    aicpu_done_ms = None
    aiv_done_ms = None
    deadline = start + timeout_s
    while aicpu_done_ms is None or aiv_done_ms is None:
        now = time.perf_counter()
        if aicpu_done_ms is None and aicpu_works[-1].is_completed():
            aicpu_done_ms = (now - start) * 1000.0
        if aiv_done_ms is None and aiv_works[-1].is_completed():
            aiv_done_ms = (now - start) * 1000.0
        if now >= deadline:
            raise TimeoutError("concurrent HCCL completion polling timed out")

    _wait(aicpu_works)
    _wait(aiv_works)
    torch.npu.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0
    maximums = torch.tensor(
        [aicpu_done_ms, aiv_done_ms, total_ms],
        dtype=torch.float32,
        device="npu",
    )
    dist.all_reduce(maximums, op=dist.ReduceOp.MAX)
    return tuple(float(value) for value in maximums.cpu().tolist())


def _validate(torch, output, world_size: int, value_base: int) -> None:
    rows = output.view(world_size, -1)
    for rank in range(world_size):
        expected = float(value_base + rank)
        if not bool(torch.all(rows[rank] == expected).item()):
            raise RuntimeError(
                f"data corruption: expected value {expected} in gathered rank {rank}"
            )


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    args = _parse_args()

    import torch
    import torch.distributed as dist
    import torch_npu

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(
        backend="hccl",
        init_method="env://",
        timeout=datetime.timedelta(seconds=args.timeout_s),
    )

    aicpu_group = None
    aiv_group = None
    try:
        aicpu_group = _new_group(dist, torch_npu, 2, args.hccl_buffer_mb)
        aiv_group = _new_group(dist, torch_npu, 3, args.hccl_buffer_mb)

        element_size = torch.empty((), dtype=torch.float16).element_size()
        aicpu_elements = max(1, int(args.aicpu_mb * 1024 * 1024) // element_size)
        aiv_elements = max(1, int(args.aiv_mb * 1024 * 1024) // element_size)
        aicpu_source = torch.full(
            (aicpu_elements,), float(10 + rank), dtype=torch.float16, device="npu"
        )
        aiv_source = torch.full(
            (aiv_elements,), float(100 + rank), dtype=torch.float16, device="npu"
        )
        aicpu_output = torch.empty(
            aicpu_elements * world_size, dtype=torch.float16, device="npu"
        )
        aiv_output = torch.empty(
            aiv_elements * world_size, dtype=torch.float16, device="npu"
        )
        torch.npu.synchronize()

        def aicpu_only():
            _submit(
                dist,
                aicpu_group,
                aicpu_output,
                aicpu_source,
                args.iters,
            )

        def aiv_only():
            _submit(dist, aiv_group, aiv_output, aiv_source, args.iters)

        def serial():
            aicpu_only()
            torch.npu.synchronize()
            aiv_only()

        for _ in range(args.warmup):
            _measure_concurrent_completion(
                torch,
                dist,
                aicpu_group,
                aiv_group,
                aicpu_output,
                aicpu_source,
                aiv_output,
                aiv_source,
                args.iters,
                args.timeout_s,
                args.submit_order,
            )

        samples = {
            name: []
            for name in (
                "aicpu_ts",
                "aiv",
                "serial",
                "concurrent",
                "concurrent_aicpu_done",
                "concurrent_aiv_done",
            )
        }
        operations = (
            ("aicpu_ts", aicpu_only),
            ("aiv", aiv_only),
            ("serial", serial),
        )
        for _ in range(args.rounds):
            for name, operation in operations:
                samples[name].append(_measure(torch, dist, operation))
            aicpu_done_ms, aiv_done_ms, concurrent_ms = _measure_concurrent_completion(
                torch,
                dist,
                aicpu_group,
                aiv_group,
                aicpu_output,
                aicpu_source,
                aiv_output,
                aiv_source,
                args.iters,
                args.timeout_s,
                args.submit_order,
            )
            samples["concurrent_aicpu_done"].append(aicpu_done_ms)
            samples["concurrent_aiv_done"].append(aiv_done_ms)
            samples["concurrent"].append(concurrent_ms)

        _validate(torch, aicpu_output, world_size, 10)
        _validate(torch, aiv_output, world_size, 100)
        dist.barrier()

        medians = {name: _median(values) for name, values in samples.items()}
        speedup = medians["serial"] / medians["concurrent"]
        ideal = max(medians["aicpu_ts"], medians["aiv"])
        ideal_ratio = medians["concurrent"] / ideal
        aicpu_latency_ratio = medians["concurrent_aicpu_done"] / medians["aicpu_ts"]
        aiv_latency_ratio = medians["concurrent_aiv_done"] / medians["aiv"]
        aicpu_latency_delta = medians["concurrent_aicpu_done"] - medians["aicpu_ts"]
        aiv_latency_delta = medians["concurrent_aiv_done"] - medians["aiv"]
        completion_poll_tail = max(
            0.0,
            medians["concurrent"]
            - max(
                medians["concurrent_aicpu_done"],
                medians["concurrent_aiv_done"],
            ),
        )
        if medians["concurrent"] <= medians["serial"] * 0.8:
            verdict = "OVERLAP"
        elif medians["concurrent"] <= medians["serial"] * 0.95:
            verdict = "PARTIAL_OVERLAP"
        else:
            verdict = "SERIALIZED_OR_NO_MEASURABLE_OVERLAP"

        if rank == 0:
            print("data_check=PASS")
            print(
                f"world_size={world_size} aicpu_mb_per_rank={args.aicpu_mb:g} "
                f"aiv_mb_per_rank={args.aiv_mb:g} iters={args.iters} "
                f"rounds={args.rounds} submit_order={args.submit_order}"
            )
            for name in (
                "aicpu_ts",
                "aiv",
                "serial",
                "concurrent_aicpu_done",
                "concurrent_aiv_done",
                "concurrent",
            ):
                values = ", ".join(f"{value:.3f}" for value in samples[name])
                print(f"{name}_ms median={medians[name]:.3f} samples=[{values}]")
            print(
                "aicpu_latency_when_concurrent="
                f"{aicpu_latency_ratio:.3f}x delta_ms={aicpu_latency_delta:.3f}"
            )
            print(
                "aiv_latency_when_concurrent="
                f"{aiv_latency_ratio:.3f}x delta_ms={aiv_latency_delta:.3f}"
            )
            print(f"completion_poll_tail_ms={completion_poll_tail:.3f}")
            print(f"parallel_speedup_vs_serial={speedup:.3f}x")
            print(f"concurrent_to_ideal_max_ratio={ideal_ratio:.3f}")
            print(f"verdict={verdict}")
    finally:
        if aiv_group is not None:
            dist.destroy_process_group(aiv_group)
        if aicpu_group is not None:
            dist.destroy_process_group(aicpu_group)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
