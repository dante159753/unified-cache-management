from types import SimpleNamespace

import pytest

from ucm.store.allgather import reservation


def _worker(
    page_sizes,
    blocks_per_chunk=1,
    replicated=True,
    sliding_windows=None,
):
    sliding_windows = sliding_windows or [None] * len(page_sizes)
    specs = {
        f"layer.{index}": SimpleNamespace(
            page_size_bytes=size,
            sliding_window=sliding_windows[index],
        )
        for index, size in enumerate(page_sizes)
    }
    connector = SimpleNamespace(blocks_per_chunk=blocks_per_chunk)
    model_runner = SimpleNamespace(
        get_kv_cache_spec=lambda: specs,
        kv_connector=connector,
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        model_config=SimpleNamespace(is_deepseek_mla=replicated),
    )
    return SimpleNamespace(model_runner=model_runner, vllm_config=vllm_config)


def _install_native_plan(monkeypatch):
    calls = []

    def calculate_memory_plan(
        tensor_sizes,
        shard_size,
        world_size,
        replicated,
        load_slots,
        dump_slots,
        window_blocks,
        receive_slots,
    ):
        calls.append(
            (
                list(tensor_sizes),
                shard_size,
                world_size,
                replicated,
                load_slots,
                dump_slots,
                window_blocks,
                receive_slots,
            )
        )
        return {"total_bytes": shard_size}

    import ucm.store.allgather as package

    monkeypatch.setattr(
        package,
        "ucm_allgather_runtime",
        SimpleNamespace(calculate_memory_plan=calculate_memory_plan),
        raising=False,
    )
    return calls


def test_layerwise_reservation_reuses_largest_stage_and_keeps_chunk_blocks(monkeypatch):
    config = {
        "use_layerwise": True,
        "allgather_collective_buffer_mb": 1,
    }
    monkeypatch.setattr(reservation, "get_allgather_pipeline_config", lambda _: config)
    calls = _install_native_plan(monkeypatch)

    result = reservation.calculate_vllm_reservation(
        _worker([4096, 8192], blocks_per_chunk=3)
    )

    assert [call[0] for call in calls] == [
        [4096, 4096, 4096],
        [8192, 8192, 8192],
    ]
    assert [call[1] for call in calls] == [12288, 24576]
    assert result == 24576 + 2 * 1024 * 1024


def test_pipeline_config_reads_layerwise_from_launch_config():
    connector_config = {"store_pipeline": "AllGather|Cache|Fake"}
    launch_config = {
        "use_layerwise": True,
        "ucm_connectors": [
            {
                "ucm_connector_name": "UcmPipelineStore",
                "ucm_connector_config": connector_config,
            }
        ],
    }
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=launch_config,
        )
    )

    result = reservation.get_allgather_pipeline_config(vllm_config)

    assert result["use_layerwise"] is True
    assert "use_layerwise" not in connector_config


def test_non_layerwise_reservation_repeats_tensor_layout_per_chunk(monkeypatch):
    config = {
        "use_layerwise": False,
        "allgather_collective_buffer_mb": 1,
    }
    monkeypatch.setattr(reservation, "get_allgather_pipeline_config", lambda _: config)
    calls = _install_native_plan(monkeypatch)

    reservation.calculate_vllm_reservation(
        _worker([4096, 8192], blocks_per_chunk=2, replicated=False)
    )

    assert calls == [([4096, 8192, 4096, 8192], 24576, 8, False, 2, 2, 4, 2)]


def test_scatter_only_reservation_omits_collective_memory(monkeypatch):
    config = {
        "use_layerwise": False,
        "allgather_scatter_only": True,
        "allgather_collective_buffer_mb": 8,
    }
    monkeypatch.setattr(reservation, "get_allgather_pipeline_config", lambda _: config)
    calls = _install_native_plan(monkeypatch)

    result = reservation.calculate_vllm_reservation(_worker([4096]))

    assert calls == [([4096], 4096, 8, False, 2, 2, 4, 2)]
    assert result == 4096


def test_window_blocks_are_reserved(monkeypatch):
    config = {
        "use_layerwise": True,
        "allgather_window_blocks_per_rank": 64,
    }
    monkeypatch.setattr(reservation, "get_allgather_pipeline_config", lambda _: config)
    calls = _install_native_plan(monkeypatch)

    reservation.calculate_vllm_reservation(_worker([4096]))

    assert calls[0][-2] == 64


def test_fawa_stage_windows_are_reserved(monkeypatch):
    config = {
        "use_layerwise": True,
        "allgather_window_blocks_per_rank_fa": 4,
        "allgather_window_blocks_per_rank_wa": 64,
    }
    monkeypatch.setattr(reservation, "get_allgather_pipeline_config", lambda _: config)
    calls = _install_native_plan(monkeypatch)

    reservation.calculate_vllm_reservation(
        _worker([151552, 4096], sliding_windows=[None, 8192])
    )

    assert [call[-2] for call in calls] == [4, 64]


def test_fawa_stage_slots_are_reserved(monkeypatch):
    config = {
        "use_layerwise": True,
        "allgather_load_slots_fa": 8,
        "allgather_load_slots_wa": 2,
    }
    monkeypatch.setattr(reservation, "get_allgather_pipeline_config", lambda _: config)
    calls = _install_native_plan(monkeypatch)

    reservation.calculate_vllm_reservation(
        _worker([151552, 4096], sliding_windows=[None, 8192])
    )

    assert [call[4] for call in calls] == [8, 2]


def test_fawa_default_load_slots_and_groups_are_reserved(monkeypatch):
    config = {
        "use_layerwise": True,
        "allgather_collective_buffer_mb": 1,
    }
    monkeypatch.setattr(reservation, "get_allgather_pipeline_config", lambda _: config)
    calls = _install_native_plan(monkeypatch)

    result = reservation.calculate_vllm_reservation(
        _worker([151552, 4096], sliding_windows=[None, 8192])
    )

    assert [call[4] for call in calls] == [4, 1]
    assert result == 151552 + 4096 + 2 * 1024 * 1024


def test_explicit_blocks_per_chunk_overrides_connector(monkeypatch):
    config = {
        "use_layerwise": True,
        "allgather_blocks_per_chunk": 4,
    }
    monkeypatch.setattr(reservation, "get_allgather_pipeline_config", lambda _: config)
    calls = _install_native_plan(monkeypatch)

    reservation.calculate_vllm_reservation(_worker([4096], blocks_per_chunk=2))

    assert calls[0][0] == [4096] * 4


def test_invalid_blocks_per_chunk_is_rejected(monkeypatch):
    config = {
        "use_layerwise": True,
        "allgather_blocks_per_chunk": 0,
    }
    monkeypatch.setattr(reservation, "get_allgather_pipeline_config", lambda _: config)

    with pytest.raises(ValueError, match="blocks_per_chunk"):
        reservation.calculate_vllm_reservation(_worker([4096]))
