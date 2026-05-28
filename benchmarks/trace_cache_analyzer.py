import argparse
import gzip
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

GB = 1024**3
DTypeBytes = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
}


class LRUCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, capacity)
        self.items: OrderedDict[str, None] = OrderedDict()
        self.evictions = 0
        self.peak_blocks = 0

    def __contains__(self, key: str) -> bool:
        return key in self.items

    def touch(self, key: str) -> None:
        if key in self.items:
            self.items.move_to_end(key)

    def put(self, key: str) -> None:
        if self.capacity <= 0:
            return
        if key in self.items:
            self.items.move_to_end(key)
            return
        self.items[key] = None
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
            self.evictions += 1
        self.peak_blocks = max(self.peak_blocks, len(self.items))

    def stats(self, block_bytes: int, configured_gb: float) -> dict:
        return {
            "configured_gb": configured_gb,
            "capacity_blocks": self.capacity,
            "final_blocks": len(self.items),
            "peak_blocks": self.peak_blocks,
            "evictions": self.evictions,
            "final_bytes": len(self.items) * block_bytes,
            "peak_bytes": self.peak_blocks * block_bytes,
        }


def dtype_size_bytes(dtype: str) -> int:
    try:
        return DTypeBytes[dtype]
    except KeyError as exc:
        supported = ", ".join(sorted(DTypeBytes))
        raise ValueError(
            f"Unsupported KV cache dtype: {dtype}. Supported: {supported}"
        ) from exc


def load_model_config(model_path: str | Path | None) -> dict:
    if model_path is None:
        return {}
    config_path = Path(model_path)
    if config_path.is_dir():
        config_path = config_path / "config.json"
    if not config_path.exists():
        raise ValueError(f"Model config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Model config must be a JSON object: {config_path}")
    return config


def _config_int(config: dict, key: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Model config field {key} must be an integer") from exc


def _infer_head_dim(config: dict) -> int | None:
    head_dim = _config_int(config, "head_dim")
    if head_dim is not None:
        return head_dim
    hidden_size = _config_int(config, "hidden_size")
    num_attention_heads = _config_int(config, "num_attention_heads")
    if hidden_size is None or num_attention_heads is None:
        return None
    if num_attention_heads <= 0:
        raise ValueError(
            "Model config field num_attention_heads must be greater than 0"
        )
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "Cannot infer head_dim because hidden_size is not divisible by num_attention_heads"
        )
    return hidden_size // num_attention_heads


def resolve_model_params(args: argparse.Namespace) -> dict:
    config = load_model_config(getattr(args, "model_path", None))
    kv_lora_rank = args.kv_lora_rank
    if kv_lora_rank is None:
        kv_lora_rank = _config_int(config, "kv_lora_rank")

    qk_rope_head_dim = args.qk_rope_head_dim
    if qk_rope_head_dim is None:
        qk_rope_head_dim = _config_int(config, "qk_rope_head_dim")

    attention_type = args.attention_type
    if attention_type is None:
        attention_type = (
            "mla"
            if kv_lora_rank is not None and qk_rope_head_dim is not None
            else "gqa"
        )

    num_layers = args.num_layers
    if num_layers is None:
        num_layers = _config_int(config, "num_hidden_layers")
    if num_layers is None:
        raise ValueError(
            "--num-layers is required when it cannot be inferred from --model-path"
        )

    num_kv_heads = args.num_kv_heads
    if num_kv_heads is None:
        num_kv_heads = _config_int(config, "num_key_value_heads")
    if num_kv_heads is None:
        num_kv_heads = _config_int(config, "num_attention_heads")

    head_dim = args.head_dim
    if head_dim is None:
        head_dim = _infer_head_dim(config)

    return {
        "attention_type": attention_type,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
    }


def calculate_block_bytes(
    *,
    attention_type: str,
    num_layers: int,
    trace_block_size: int,
    kv_cache_dtype: str,
    tp: int,
    num_kv_heads: int | None,
    head_dim: int | None,
    kv_lora_rank: int | None,
    qk_rope_head_dim: int | None,
) -> int:
    dtype_bytes = dtype_size_bytes(kv_cache_dtype)
    if num_layers <= 0:
        raise ValueError("--num-layers must be greater than 0")
    if trace_block_size <= 0:
        raise ValueError("--trace-block-size must be greater than 0")
    if tp <= 0:
        raise ValueError("--tp must be greater than 0")

    if attention_type == "gqa":
        if num_kv_heads is None or head_dim is None:
            raise ValueError("--num-kv-heads and --head-dim are required for GQA")
        return (
            2
            * num_layers
            * trace_block_size
            * num_kv_heads
            * head_dim
            * dtype_bytes
            // tp
        )

    if attention_type == "mla":
        if kv_lora_rank is None or qk_rope_head_dim is None:
            raise ValueError(
                "--kv-lora-rank and --qk-rope-head-dim are required for MLA"
            )
        return (
            num_layers
            * trace_block_size
            * (kv_lora_rank + qk_rope_head_dim)
            * dtype_bytes
            // tp
        )

    raise ValueError(f"Unsupported attention type: {attention_type}")


def empty_model_params(args: argparse.Namespace) -> dict:
    return {
        "attention_type": args.attention_type,
        "num_layers": args.num_layers,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "kv_lora_rank": args.kv_lora_rank,
        "qk_rope_head_dim": args.qk_rope_head_dim,
    }


def capacity_gb_to_blocks(capacity_gb: float, block_bytes: int) -> int:
    if capacity_gb <= 0:
        return 0
    return int(capacity_gb * GB) // block_bytes


def read_trace_records(trace_paths: Iterable[str | Path]) -> list[dict]:
    records = []
    for trace_path in trace_paths:
        path = Path(trace_path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8-sig", errors="ignore") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                hash_ids = record.get("hash_ids")
                if not isinstance(hash_ids, list):
                    raise ValueError(f"{path}:{line_no} missing list field hash_ids")
                record["_trace_path"] = str(path)
                records.append(record)
    return records


def _hit_prefix(cache: LRUCache, block_ids: list[str], start: int) -> int:
    pos = start
    while pos < len(block_ids) and block_ids[pos] in cache:
        cache.touch(block_ids[pos])
        pos += 1
    return pos - start


def _new_totals() -> dict:
    return {
        "total_requests": 0,
        "total_query_blocks": 0,
        "gpu_hit_blocks": 0,
        "memory_hit_blocks": 0,
        "disk_hit_blocks": 0,
        "miss_blocks": 0,
    }


def _add_request_totals(
    totals: dict,
    *,
    query_blocks: int,
    gpu_hits: int,
    memory_hits: int,
    disk_hits: int,
    miss_blocks: int,
) -> None:
    totals["total_requests"] += 1
    totals["total_query_blocks"] += query_blocks
    totals["gpu_hit_blocks"] += gpu_hits
    totals["memory_hit_blocks"] += memory_hits
    totals["disk_hit_blocks"] += disk_hits
    totals["miss_blocks"] += miss_blocks


def _add_hit_rates(totals: dict) -> None:
    total_blocks = totals["total_query_blocks"]
    if total_blocks:
        totals["gpu_hit_rate"] = totals["gpu_hit_blocks"] / total_blocks
        totals["memory_hit_rate"] = totals["memory_hit_blocks"] / total_blocks
        totals["disk_hit_rate"] = totals["disk_hit_blocks"] / total_blocks
        totals["total_hit_rate"] = (
            totals["gpu_hit_blocks"]
            + totals["memory_hit_blocks"]
            + totals["disk_hit_blocks"]
        ) / total_blocks
    else:
        totals["gpu_hit_rate"] = 0.0
        totals["memory_hit_rate"] = 0.0
        totals["disk_hit_rate"] = 0.0
        totals["total_hit_rate"] = 0.0


def _aggregate_cache_stats(
    caches: list[LRUCache],
    block_bytes: int,
    configured_gb_per_cache: float,
) -> dict:
    final_blocks = sum(len(cache.items) for cache in caches)
    peak_blocks = sum(cache.peak_blocks for cache in caches)
    return {
        "configured_gb": configured_gb_per_cache * len(caches),
        "configured_gb_per_node": configured_gb_per_cache,
        "capacity_blocks": sum(cache.capacity for cache in caches),
        "capacity_blocks_per_node": caches[0].capacity if caches else 0,
        "final_blocks": final_blocks,
        "peak_blocks": peak_blocks,
        "evictions": sum(cache.evictions for cache in caches),
        "final_bytes": final_blocks * block_bytes,
        "peak_bytes": peak_blocks * block_bytes,
    }


def analyze_records(
    records: list[dict],
    *,
    block_bytes: int,
    capacities_blocks: dict[str, int],
    trace_paths: list[str],
    include_requests: bool = False,
    configured_gb: dict[str, float] | None = None,
    num_nodes: int = 1,
    random_seed: int | None = None,
) -> dict:
    if num_nodes <= 0:
        raise ValueError("--num-nodes must be greater than 0")
    configured_gb = configured_gb or {"gpu": 0.0, "memory": 0.0, "disk": 0.0}
    rng = random.Random(random_seed)
    gpu_caches = [LRUCache(capacities_blocks["gpu"]) for _ in range(num_nodes)]
    memory_caches = [LRUCache(capacities_blocks["memory"]) for _ in range(num_nodes)]
    disk = LRUCache(capacities_blocks["disk"])
    totals = _new_totals()
    node_totals = [_new_totals() for _ in range(num_nodes)]
    request_summaries = []

    for request_index, record in enumerate(records):
        node_index = 0 if num_nodes == 1 else rng.randrange(num_nodes)
        gpu = gpu_caches[node_index]
        memory = memory_caches[node_index]
        block_ids = [str(block_id) for block_id in record["hash_ids"]]
        pos = 0
        gpu_hits = _hit_prefix(gpu, block_ids, pos)
        pos += gpu_hits

        memory_hits = _hit_prefix(memory, block_ids, pos)
        for block_id in block_ids[pos : pos + memory_hits]:
            gpu.put(block_id)
        pos += memory_hits

        disk_hits = _hit_prefix(disk, block_ids, pos)
        for block_id in block_ids[pos : pos + disk_hits]:
            memory.put(block_id)
            gpu.put(block_id)
        pos += disk_hits

        miss_blocks = len(block_ids) - pos
        for block_id in block_ids:
            disk.put(block_id)
            memory.put(block_id)
            gpu.put(block_id)

        request_totals = {
            "query_blocks": len(block_ids),
            "gpu_hits": gpu_hits,
            "memory_hits": memory_hits,
            "disk_hits": disk_hits,
            "miss_blocks": miss_blocks,
        }
        _add_request_totals(totals, **request_totals)
        _add_request_totals(node_totals[node_index], **request_totals)

        if include_requests:
            request_summaries.append(
                {
                    "request_index": request_index,
                    "node_index": node_index,
                    "trace_path": record.get("_trace_path"),
                    "timestamp": record.get("timestamp"),
                    "input_length": record.get("input_length"),
                    "output_length": record.get("output_length"),
                    "query_blocks": len(block_ids),
                    "gpu_hit_blocks": gpu_hits,
                    "memory_hit_blocks": memory_hits,
                    "disk_hit_blocks": disk_hits,
                    "miss_blocks": miss_blocks,
                }
            )

    _add_hit_rates(totals)
    for node_total in node_totals:
        _add_hit_rates(node_total)

    result = {
        "trace_paths": trace_paths,
        "num_nodes": num_nodes,
        "block_bytes": block_bytes,
        "totals": totals,
        "layers": {
            "gpu": _aggregate_cache_stats(
                gpu_caches,
                block_bytes,
                configured_gb["gpu"],
            ),
            "memory": _aggregate_cache_stats(
                memory_caches,
                block_bytes,
                configured_gb["memory"],
            ),
            "disk": disk.stats(block_bytes, configured_gb["disk"]),
        },
        "nodes": [
            {
                "node_index": node_index,
                "totals": node_totals[node_index],
                "layers": {
                    "gpu": gpu_caches[node_index].stats(
                        block_bytes, configured_gb["gpu"]
                    ),
                    "memory": memory_caches[node_index].stats(
                        block_bytes,
                        configured_gb["memory"],
                    ),
                },
            }
            for node_index in range(num_nodes)
        ],
    }
    if include_requests:
        result["requests"] = request_summaries
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze GPU/UCM memory/UCM filesystem cache hit rates from trace JSONL files."
    )
    parser.add_argument("--trace-path", nargs="+", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--attention-type", choices=["gqa", "mla"])
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-kv-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--kv-lora-rank", type=int)
    parser.add_argument("--qk-rope-head-dim", type=int)
    parser.add_argument(
        "--kv-cache-dtype",
        choices=sorted(DTypeBytes),
        default="bfloat16",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--trace-block-size", type=int, default=512)
    parser.add_argument("--block-bytes", type=int)
    parser.add_argument("--gpu-kv-cache-gb", type=float, required=True)
    parser.add_argument("--ucm-memory-cache-gb", type=float, required=True)
    parser.add_argument("--ucm-filesystem-cache-gb", type=float, required=True)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-requests", action="store_true")
    return parser.parse_args()


def build_analysis(args: argparse.Namespace) -> dict:
    if args.block_bytes is not None:
        if args.block_bytes <= 0:
            raise ValueError("--block-bytes must be greater than 0")
        model_params = empty_model_params(args)
        block_bytes = args.block_bytes
        block_bytes_source = "override"
    else:
        model_params = resolve_model_params(args)
        block_bytes = calculate_block_bytes(
            attention_type=model_params["attention_type"],
            num_layers=model_params["num_layers"],
            trace_block_size=args.trace_block_size,
            kv_cache_dtype=args.kv_cache_dtype,
            tp=args.tp,
            num_kv_heads=model_params["num_kv_heads"],
            head_dim=model_params["head_dim"],
            kv_lora_rank=model_params["kv_lora_rank"],
            qk_rope_head_dim=model_params["qk_rope_head_dim"],
        )
        block_bytes_source = "estimated"
    configured_gb = {
        "gpu": args.gpu_kv_cache_gb,
        "memory": args.ucm_memory_cache_gb,
        "disk": args.ucm_filesystem_cache_gb,
    }
    capacities_blocks = {
        layer: capacity_gb_to_blocks(capacity_gb, block_bytes)
        for layer, capacity_gb in configured_gb.items()
    }
    records = read_trace_records(args.trace_path)
    result = analyze_records(
        records,
        block_bytes=block_bytes,
        capacities_blocks=capacities_blocks,
        trace_paths=[str(path) for path in args.trace_path],
        include_requests=args.include_requests,
        configured_gb=configured_gb,
        num_nodes=args.num_nodes,
        random_seed=args.random_seed,
    )
    result["config"] = {
        "model_path": str(args.model_path) if args.model_path is not None else None,
        **model_params,
        "block_bytes_override": args.block_bytes,
        "block_bytes_source": block_bytes_source,
        "kv_cache_dtype": args.kv_cache_dtype,
        "tp": args.tp,
        "trace_block_size": args.trace_block_size,
        "num_nodes": args.num_nodes,
        "random_seed": args.random_seed,
    }
    return result


def print_summary(result: dict) -> None:
    totals = result["totals"]
    print("Trace cache analysis")
    print(f"  traces: {len(result['trace_paths'])}")
    print(f"  nodes: {result['num_nodes']}")
    print(f"  block_bytes: {result['block_bytes']}")
    print(f"  total_requests: {totals['total_requests']}")
    print(f"  total_query_blocks: {totals['total_query_blocks']}")
    print(f"  gpu_hit_rate: {totals['gpu_hit_rate']:.6f}")
    print(f"  memory_hit_rate: {totals['memory_hit_rate']:.6f}")
    print(f"  disk_hit_rate: {totals['disk_hit_rate']:.6f}")
    print(f"  total_hit_rate: {totals['total_hit_rate']:.6f}")
    for layer_name, layer_stats in result["layers"].items():
        print(
            "  "
            f"{layer_name}: capacity_blocks={layer_stats['capacity_blocks']}, "
            f"final_blocks={layer_stats['final_blocks']}, "
            f"peak_blocks={layer_stats['peak_blocks']}, "
            f"evictions={layer_stats['evictions']}"
        )


def main() -> None:
    args = parse_args()
    try:
        result = build_analysis(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print_summary(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
