import pytest

from ucm.store.allgather.memory_plan import (
    calculate_frame_metadata_bytes,
    calculate_stage_memory_plan,
    calculate_worker_memory_plan,
)


def test_tp8_w4_memory_plan() -> None:
    plan = calculate_stage_memory_plan(
        [1_000_000, 2_000_000],
        shard_size=3_002_368,
        world_size=8,
        replicated=True,
        load_slots=2,
        dump_slots=2,
    )

    frame_bytes = 4 * 3_002_368 + calculate_frame_metadata_bytes(4)
    assert plan.load_send_bytes == 2 * frame_bytes
    assert plan.load_receive_bytes == 2 * 8 * frame_bytes
    assert plan.dump_send_bytes == 2 * 4 * 3_002_368
    assert plan.receive_slots == 2
    assert plan.load_destination_bytes == 0
    assert plan.load_route_bytes == 0
    assert plan.window_blocks == 4

    worker = calculate_worker_memory_plan([plan], hccl_buffer_mb=32)
    assert worker.hccl_bytes == 64 * 1024 * 1024
    assert worker.total_bytes == plan.total_bytes + worker.hccl_bytes


def test_tp1_aliases_load_receive_buffer() -> None:
    plan = calculate_stage_memory_plan([4096], 4096, world_size=1, replicated=True)
    assert plan.load_receive_bytes == 0
    assert calculate_worker_memory_plan([plan]).hccl_bytes == 0


def test_collective_groups_scale_hccl_reservation() -> None:
    plan = calculate_stage_memory_plan([4096], 4096, world_size=8, replicated=True)
    worker = calculate_worker_memory_plan(
        [plan], hccl_buffer_mb=8, collective_group_count=4
    )

    assert worker.hccl_bytes == 4 * 2 * 8 * 1024 * 1024


def test_non_replicated_tp8_has_no_collective_buffer() -> None:
    plan = calculate_stage_memory_plan([4096], 4096, world_size=8, replicated=False)
    assert plan.load_receive_bytes == 0
    assert calculate_worker_memory_plan([plan]).hccl_bytes == 0


def test_invalid_plan_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds shard_size"):
        calculate_stage_memory_plan([4097], 4096, 8, True)


def test_window_blocks_scale_payload_and_metadata() -> None:
    plan = calculate_stage_memory_plan(
        [4096], 4096, world_size=8, replicated=True, window_blocks=64
    )

    assert plan.window_blocks == 64
    frame_bytes = 64 * 4096 + calculate_frame_metadata_bytes(64)
    assert plan.load_send_bytes == 2 * frame_bytes
    assert plan.load_receive_bytes == 2 * 8 * frame_bytes
    assert plan.dump_send_bytes == 2 * 64 * 4096
    assert plan.load_destination_bytes == 0
    assert plan.load_route_bytes == 0


def test_receive_slots_are_independent_and_capped_by_load_slots() -> None:
    plan = calculate_stage_memory_plan(
        [4096],
        4096,
        world_size=8,
        replicated=True,
        load_slots=4,
        receive_slots=3,
    )

    frame_bytes = 4 * 4096 + calculate_frame_metadata_bytes(4)
    assert plan.receive_slots == 3
    assert plan.load_send_bytes == 4 * frame_bytes
    assert plan.load_receive_bytes == 3 * 8 * frame_bytes

    capped = calculate_stage_memory_plan(
        [4096],
        4096,
        world_size=8,
        replicated=True,
        load_slots=2,
        receive_slots=4,
    )
    assert capped.receive_slots == 2
