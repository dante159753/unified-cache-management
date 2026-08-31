import pytest

from ucm.store.allgather.memory_plan import (
    calculate_stage_memory_plan,
    calculate_worker_memory_plan,
)


def test_tp8_remote_scatter_memory_plan() -> None:
    plan = calculate_stage_memory_plan(
        [1_000_000, 2_000_000],
        shard_size=3_002_368,
        world_size=8,
        replicated=True,
        load_slots=2,
        dump_slots=2,
    )

    rows = 4 * 8
    assert plan.load_send_bytes == 2 * 4 * 3_002_368
    assert plan.load_receive_bytes == 0
    assert plan.dump_send_bytes == 2 * 4 * 3_002_368
    assert plan.load_destination_bytes == 2 * rows * 2 * 8
    assert plan.load_route_bytes == 2 * rows * 2 * 4 + 2 * 8 * 8
    assert calculate_worker_memory_plan([plan]).total_bytes == plan.total_bytes


def test_tp1_omits_peer_pointer_tables() -> None:
    plan = calculate_stage_memory_plan([4096], 4096, world_size=1, replicated=True)
    assert plan.load_route_bytes == 2 * 4 * 2 * 4


def test_non_replicated_tp8_uses_local_row_capacity() -> None:
    plan = calculate_stage_memory_plan([4096], 4096, world_size=8, replicated=False)
    assert plan.load_destination_bytes == 2 * 4 * 8
    assert plan.load_route_bytes == 2 * 4 * 2 * 4


def test_invalid_plan_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds shard_size"):
        calculate_stage_memory_plan([4097], 4096, 8, True)


def test_window_blocks_scale_payload_and_metadata() -> None:
    plan = calculate_stage_memory_plan(
        [4096], 4096, world_size=8, replicated=True, window_blocks=64
    )

    rows = 64 * 8
    assert plan.load_send_bytes == 2 * 64 * 4096
    assert plan.dump_send_bytes == 2 * 64 * 4096
    assert plan.load_destination_bytes == 2 * rows * 8
    assert plan.load_route_bytes == 2 * rows * 2 * 4 + 2 * 8 * 8


def test_copy_then_scatter_reserves_full_receive_slots() -> None:
    plan = calculate_stage_memory_plan(
        [4096],
        4096,
        world_size=8,
        replicated=True,
        load_slots=2,
        window_blocks=64,
        buffered_remote_scatter=True,
    )

    assert plan.load_receive_bytes == 2 * 8 * 64 * 4096
    assert plan.payload_bytes == (
        plan.load_send_bytes + plan.load_receive_bytes + plan.dump_send_bytes
    )
