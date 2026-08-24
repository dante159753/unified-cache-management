from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

DEFAULT_WINDOW_BLOCKS = 4
WINDOW_BLOCKS = DEFAULT_WINDOW_BLOCKS
COPY_CHUNK_BYTES = 32 * 1024
MAX_COPY_WORKERS = 40
MAX_AIV_CORES = MAX_COPY_WORKERS
DEFAULT_LOAD_SLOTS = 2
# Each AllGather stage owns exactly one collective domain. Concurrent
# communicators on one device were a source of HCCL internal errors, and
# pipelining now comes from the per-slot side streams instead.
COLLECTIVE_GROUPS_PER_STAGE = 1
# Receive areas only cover the collective-to-scatter handoff, which one
# serialized collective stream saturates with a couple of buffers.
DEFAULT_RECEIVE_SLOTS = 2
DEFAULT_LOAD_GROUPS = COLLECTIVE_GROUPS_PER_STAGE
DEFAULT_FA_LOAD_SLOTS = 4
DEFAULT_FA_LOAD_GROUPS = COLLECTIVE_GROUPS_PER_STAGE
DEFAULT_WA_LOAD_SLOTS = 1
DEFAULT_WA_LOAD_GROUPS = COLLECTIVE_GROUPS_PER_STAGE
DEFAULT_DUMP_SLOTS = 2
DEFAULT_COLLECTIVE_BUFFER_MB = 8
COLLECTIVE_BUFFER_COPIES = 2
FRAME_HEADER_BYTES = 32
FRAME_RECORD_BYTES = 16
DEFAULT_HCCL_BUFFER_MB = DEFAULT_COLLECTIVE_BUFFER_MB
HCCL_BUFFER_COPIES = COLLECTIVE_BUFFER_COPIES


@dataclass(frozen=True)
class AllGatherStageMemoryPlan:
    shard_size: int
    tensor_count: int
    chunk_count: int
    world_size: int
    window_blocks: int
    replicated: bool
    load_slots: int
    receive_slots: int
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
    collective_buffer_mb: int
    collective_group_count: int = DEFAULT_LOAD_GROUPS

    @property
    def collective_bytes(self) -> int:
        if not any(stage.replicated and stage.world_size > 1 for stage in self.stages):
            return 0
        return (
            self.collective_buffer_mb
            * 1024
            * 1024
            * COLLECTIVE_BUFFER_COPIES
            * self.collective_group_count
        )

    @property
    def hccl_buffer_mb(self) -> int:
        return self.collective_buffer_mb

    @property
    def hccl_bytes(self) -> int:
        return self.collective_bytes

    @property
    def total_bytes(self) -> int:
        return sum(stage.total_bytes for stage in self.stages) + self.collective_bytes


def _positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def calculate_frame_metadata_bytes(window_blocks: int) -> int:
    return (FRAME_HEADER_BYTES + window_blocks * FRAME_RECORD_BYTES + 31) // 32 * 32


def calculate_stage_memory_plan(
    tensor_sizes: Sequence[int],
    shard_size: int,
    world_size: int,
    replicated: bool,
    load_slots: int = DEFAULT_LOAD_SLOTS,
    dump_slots: int = DEFAULT_DUMP_SLOTS,
    window_blocks: int = DEFAULT_WINDOW_BLOCKS,
    receive_slots: int = DEFAULT_RECEIVE_SLOTS,
) -> AllGatherStageMemoryPlan:
    sizes = tuple(int(size) for size in tensor_sizes)
    if not sizes or any(size < 0 for size in sizes):
        raise ValueError("tensor_sizes must contain non-negative sizes")
    shard_size = _positive("shard_size", shard_size)
    world_size = _positive("world_size", world_size)
    load_slots = _positive("load_slots", load_slots)
    dump_slots = _positive("dump_slots", dump_slots)
    window_blocks = _positive("window_blocks", window_blocks)
    receive_slots = min(_positive("receive_slots", receive_slots), load_slots)
    if sum(sizes) > shard_size:
        raise ValueError(
            f"tensor sizes total {sum(sizes)} exceeds shard_size {shard_size}"
        )

    chunk_count = sum(
        (size + COPY_CHUNK_BYTES - 1) // COPY_CHUNK_BYTES for size in sizes
    )
    if chunk_count == 0:
        raise ValueError("tensor_sizes must contain at least one non-empty tensor")

    window_payload_bytes = window_blocks * shard_size
    collective_enabled = bool(replicated) and world_size > 1
    frame_bytes = window_payload_bytes + (
        calculate_frame_metadata_bytes(window_blocks) if collective_enabled else 0
    )
    max_window_rows = window_blocks * (world_size if collective_enabled else 1)

    return AllGatherStageMemoryPlan(
        shard_size=shard_size,
        tensor_count=len(sizes),
        chunk_count=chunk_count,
        world_size=world_size,
        window_blocks=window_blocks,
        replicated=bool(replicated),
        load_slots=load_slots,
        receive_slots=receive_slots,
        dump_slots=dump_slots,
        load_send_bytes=load_slots * frame_bytes,
        load_receive_bytes=(
            receive_slots * world_size * frame_bytes if collective_enabled else 0
        ),
        dump_send_bytes=dump_slots * window_payload_bytes,
        chunk_layout_bytes=chunk_count * 4 * 8,
        dump_descriptor_bytes=(dump_slots * window_blocks * chunk_count * 3 * 8),
        dump_offset_bytes=dump_slots * (MAX_COPY_WORKERS + 1) * 4,
        load_destination_bytes=(
            0 if collective_enabled else load_slots * max_window_rows * len(sizes) * 8
        ),
        load_route_bytes=(
            0 if collective_enabled else load_slots * max_window_rows * 2 * 4
        ),
    )


def calculate_worker_memory_plan(
    stages: Iterable[AllGatherStageMemoryPlan],
    hccl_buffer_mb: int = DEFAULT_HCCL_BUFFER_MB,
    *,
    collective_buffer_mb: int | None = None,
    collective_group_count: int = DEFAULT_LOAD_GROUPS,
) -> AllGatherWorkerMemoryPlan:
    stage_tuple = tuple(stages)
    if not stage_tuple:
        return AllGatherWorkerMemoryPlan((), 0)
    return AllGatherWorkerMemoryPlan(
        stage_tuple,
        _positive(
            "collective_buffer_mb",
            hccl_buffer_mb if collective_buffer_mb is None else collective_buffer_mb,
        ),
        _positive("collective_group_count", collective_group_count),
    )
