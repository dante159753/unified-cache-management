from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

DEFAULT_WINDOW_BLOCKS = 4
WINDOW_BLOCKS = DEFAULT_WINDOW_BLOCKS
COPY_CHUNK_BYTES = 32 * 1024
MAX_COPY_WORKERS = 40
MAX_AIV_CORES = MAX_COPY_WORKERS
DEFAULT_LOAD_SLOTS = 2
DEFAULT_FA_LOAD_SLOTS = 4
DEFAULT_WA_LOAD_SLOTS = 1
DEFAULT_DUMP_SLOTS = 2


@dataclass(frozen=True)
class AllGatherStageMemoryPlan:
    shard_size: int
    tensor_count: int
    chunk_count: int
    world_size: int
    window_blocks: int
    replicated: bool
    load_slots: int
    dump_slots: int
    load_send_bytes: int
    load_receive_bytes: int
    dump_send_bytes: int
    chunk_layout_bytes: int
    dump_descriptor_bytes: int
    dump_offset_bytes: int
    load_destination_bytes: int
    load_route_bytes: int

    @property
    def payload_bytes(self) -> int:
        return self.load_send_bytes + self.load_receive_bytes + self.dump_send_bytes

    @property
    def metadata_bytes(self) -> int:
        return (
            self.chunk_layout_bytes
            + self.dump_descriptor_bytes
            + self.dump_offset_bytes
            + self.load_destination_bytes
            + self.load_route_bytes
        )

    @property
    def total_bytes(self) -> int:
        return self.payload_bytes + self.metadata_bytes


@dataclass(frozen=True)
class AllGatherWorkerMemoryPlan:
    stages: tuple[AllGatherStageMemoryPlan, ...]

    @property
    def total_bytes(self) -> int:
        return sum(stage.total_bytes for stage in self.stages)


def _positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def calculate_stage_memory_plan(
    tensor_sizes: Sequence[int],
    shard_size: int,
    world_size: int,
    replicated: bool,
    load_slots: int = DEFAULT_LOAD_SLOTS,
    dump_slots: int = DEFAULT_DUMP_SLOTS,
    window_blocks: int = DEFAULT_WINDOW_BLOCKS,
    buffered_remote_scatter: bool = False,
) -> AllGatherStageMemoryPlan:
    sizes = tuple(int(size) for size in tensor_sizes)
    if not sizes or any(size < 0 for size in sizes):
        raise ValueError("tensor_sizes must contain non-negative sizes")
    shard_size = _positive("shard_size", shard_size)
    world_size = _positive("world_size", world_size)
    load_slots = _positive("load_slots", load_slots)
    dump_slots = _positive("dump_slots", dump_slots)
    window_blocks = _positive("window_blocks", window_blocks)
    if sum(sizes) > shard_size:
        raise ValueError(
            f"tensor sizes total {sum(sizes)} exceeds shard_size {shard_size}"
        )

    chunk_count = sum(
        (size + COPY_CHUNK_BYTES - 1) // COPY_CHUNK_BYTES for size in sizes
    )
    if chunk_count == 0:
        raise ValueError("tensor_sizes must contain at least one non-empty tensor")

    distributed = bool(replicated) and world_size > 1
    window_payload_bytes = window_blocks * shard_size
    max_window_rows = window_blocks * (world_size if distributed else 1)

    return AllGatherStageMemoryPlan(
        shard_size=shard_size,
        tensor_count=len(sizes),
        chunk_count=chunk_count,
        world_size=world_size,
        window_blocks=window_blocks,
        replicated=bool(replicated),
        load_slots=load_slots,
        dump_slots=dump_slots,
        load_send_bytes=load_slots * window_payload_bytes,
        load_receive_bytes=(
            load_slots * world_size * window_payload_bytes
            if buffered_remote_scatter and distributed
            else 0
        ),
        dump_send_bytes=dump_slots * window_payload_bytes,
        chunk_layout_bytes=chunk_count * 4 * 8,
        dump_descriptor_bytes=dump_slots * window_blocks * chunk_count * 3 * 8,
        dump_offset_bytes=dump_slots * (MAX_COPY_WORKERS + 1) * 4,
        load_destination_bytes=(load_slots * max_window_rows * len(sizes) * 8),
        load_route_bytes=(
            load_slots * max_window_rows * 2 * 4
            + (load_slots * world_size * 8 if distributed else 0)
        ),
    )


def calculate_worker_memory_plan(
    stages: Iterable[AllGatherStageMemoryPlan],
) -> AllGatherWorkerMemoryPlan:
    return AllGatherWorkerMemoryPlan(tuple(stages))
