from __future__ import annotations

from typing import Any

from ucm.store.allgather.memory_plan import (
    COLLECTIVE_BUFFER_COPIES,
    COLLECTIVE_GROUPS_PER_STAGE,
    DEFAULT_COLLECTIVE_BUFFER_MB,
    DEFAULT_DUMP_SLOTS,
    DEFAULT_FA_LOAD_SLOTS,
    DEFAULT_LOAD_SLOTS,
    DEFAULT_RECEIVE_SLOTS,
    DEFAULT_WA_LOAD_SLOTS,
)


def _align(value: int, alignment: int = 4096) -> int:
    return (value + alignment - 1) // alignment * alignment


def _blocks_per_chunk(worker: Any, config: dict[str, Any]) -> int:
    explicit = config.get("allgather_blocks_per_chunk")
    if explicit is not None:
        value = int(explicit)
    else:
        model_runner = getattr(worker, "model_runner", None)
        connector = None
        for name in ("kv_connector", "_kv_connector", "connector"):
            connector = getattr(model_runner, name, None)
            if connector is not None:
                break
        value = int(getattr(connector, "blocks_per_chunk", 1))
    if value <= 0:
        raise ValueError(f"allgather blocks_per_chunk must be positive, got {value}")
    return value


def _stage_suffix(spec: Any) -> str:
    nested_specs = getattr(spec, "kv_cache_specs", None)
    representative = next(iter(nested_specs.values())) if nested_specs else spec
    return "wa" if getattr(representative, "sliding_window", None) is not None else "fa"


def _window_blocks(spec: Any, config: dict[str, Any]) -> int:
    suffix = _stage_suffix(spec)
    value = int(
        config.get(
            f"allgather_window_blocks_per_rank_{suffix}",
            config.get("allgather_window_blocks_per_rank", 4),
        )
    )
    if value <= 0:
        raise ValueError(f"allgather window blocks must be positive, got {value}")
    return value


def _stage_number(spec: Any, config: dict[str, Any], name: str, default: int) -> int:
    suffix = _stage_suffix(spec)
    value = int(config.get(f"{name}_{suffix}", config.get(name, default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def get_allgather_pipeline_config(vllm_config: Any) -> dict[str, Any] | None:
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return None

    from ucm.utils import Config

    launch_config = Config(kv_transfer_config).get_config()
    connectors = launch_config.get("ucm_connectors", [])
    if len(connectors) != 1:
        return None
    connector = connectors[0]
    config = dict(connector.get("ucm_connector_config", {}))
    if connector.get("ucm_connector_name") != "UcmPipelineStore":
        return None
    if "AllGather" not in str(config.get("store_pipeline", "")).split("|"):
        return None
    config["use_layerwise"] = bool(launch_config.get("use_layerwise", False))
    return config


def calculate_vllm_reservation(worker: Any) -> int:
    config = get_allgather_pipeline_config(worker.vllm_config)
    if config is None:
        return 0

    specs = worker.model_runner.get_kv_cache_spec()
    if not specs:
        return 0
    spec_values = list(specs.values())
    stage_suffixes = {_stage_suffix(spec) for spec in spec_values}
    is_fawa = stage_suffixes == {"fa", "wa"}
    page_sizes = [int(spec.page_size_bytes) for spec in spec_values]
    if any(size <= 0 for size in page_sizes):
        raise ValueError("KV cache specs contain a non-positive page size")

    use_layerwise = bool(config.get("use_layerwise", False))
    tp_size = int(worker.vllm_config.parallel_config.tensor_parallel_size)
    replicated = bool(worker.vllm_config.model_config.is_deepseek_mla) and not bool(
        config.get("allgather_scatter_only", False)
    )
    load_slots = int(config.get("allgather_load_slots", DEFAULT_LOAD_SLOTS))
    dump_slots = int(config.get("allgather_dump_slots", DEFAULT_DUMP_SLOTS))
    collective_buffer_mb = int(
        config.get(
            "allgather_collective_buffer_mb",
            config.get("allgather_hccl_buffer_mb", DEFAULT_COLLECTIVE_BUFFER_MB),
        )
    )
    blocks_per_chunk = _blocks_per_chunk(worker, config)

    def default_load_slots(spec: Any) -> int:
        if not is_fawa:
            return DEFAULT_LOAD_SLOTS
        return (
            DEFAULT_WA_LOAD_SLOTS
            if _stage_suffix(spec) == "wa"
            else DEFAULT_FA_LOAD_SLOTS
        )

    if is_fawa:
        stage_specs = {
            suffix: next(spec for spec in spec_values if _stage_suffix(spec) == suffix)
            for suffix in stage_suffixes
        }
        # FA and WA run on separate runtimes, so they hold one communicator
        # each. Any allgather_load_groups value in the config is ignored.
        collective_group_count = len(stage_specs) * COLLECTIVE_GROUPS_PER_STAGE
    else:
        collective_group_count = COLLECTIVE_GROUPS_PER_STAGE

    if use_layerwise:
        stage_inputs = [
            (
                [size] * blocks_per_chunk,
                _align(size * blocks_per_chunk),
                _window_blocks(spec, config),
                _stage_number(
                    spec, config, "allgather_load_slots", default_load_slots(spec)
                ),
                _stage_number(spec, config, "allgather_dump_slots", DEFAULT_DUMP_SLOTS),
            )
            for size, spec in zip(page_sizes, spec_values)
        ]
    else:
        tensor_sizes = page_sizes * blocks_per_chunk
        stage_inputs = [
            (
                tensor_sizes,
                _align(sum(tensor_sizes)),
                int(config.get("allgather_window_blocks_per_rank", 4)),
                load_slots,
                dump_slots,
            )
        ]

    from ucm.store.allgather import ucm_allgather_runtime

    receive_slots = int(config.get("allgather_receive_slots", DEFAULT_RECEIVE_SLOTS))
    if receive_slots <= 0:
        raise ValueError("allgather_receive_slots must be positive")

    stage_totals = [
        int(
            ucm_allgather_runtime.calculate_memory_plan(
                tensor_sizes,
                shard_size,
                tp_size,
                replicated,
                stage_load_slots,
                stage_dump_slots,
                window_blocks,
                min(receive_slots, stage_load_slots),
            )["total_bytes"]
        )
        for (
            tensor_sizes,
            shard_size,
            window_blocks,
            stage_load_slots,
            stage_dump_slots,
        ) in stage_inputs
    ]
    # One layerwise runtime reuses its buffers across layers. FA and WA are the
    # exception because they are backed by separate runtimes.
    total = sum(stage_totals) if not use_layerwise or is_fawa else max(stage_totals)
    if replicated and tp_size > 1:
        total += (
            collective_buffer_mb
            * 1024
            * 1024
            * COLLECTIVE_BUFFER_COPIES
            * collective_group_count
        )
    return total
