import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "trace_cache_analyzer.py"


def load_analyzer():
    spec = importlib.util.spec_from_file_location("trace_cache_analyzer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TraceCacheAnalyzerTest(unittest.TestCase):
    def test_gqa_block_bytes_uses_kv_cache_dtype(self):
        analyzer = load_analyzer()

        block_bytes = analyzer.calculate_block_bytes(
            attention_type="gqa",
            num_layers=2,
            trace_block_size=4,
            kv_cache_dtype="bfloat16",
            tp=2,
            num_kv_heads=2,
            head_dim=8,
            kv_lora_rank=None,
            qk_rope_head_dim=None,
        )

        self.assertEqual(block_bytes, 256)

    def test_three_tier_prefix_lru_analysis(self):
        analyzer = load_analyzer()
        records = [
            {
                "timestamp": 1.0,
                "input_length": 1024,
                "output_length": 1,
                "hash_ids": ["a", "b"],
            },
            {
                "timestamp": 2.0,
                "input_length": 1024,
                "output_length": 1,
                "hash_ids": ["a", "b"],
            },
            {
                "timestamp": 3.0,
                "input_length": 1536,
                "output_length": 1,
                "hash_ids": ["c", "d", "e"],
            },
            {
                "timestamp": 4.0,
                "input_length": 1024,
                "output_length": 1,
                "hash_ids": ["a", "b"],
            },
        ]

        result = analyzer.analyze_records(
            records,
            block_bytes=1,
            capacities_blocks={"gpu": 2, "memory": 5, "disk": 8},
            trace_paths=["trace.jsonl"],
            include_requests=True,
        )

        self.assertEqual(result["totals"]["total_requests"], 4)
        self.assertEqual(result["totals"]["total_query_blocks"], 9)
        self.assertEqual(result["totals"]["gpu_hit_blocks"], 2)
        self.assertEqual(result["totals"]["memory_hit_blocks"], 2)
        self.assertEqual(result["totals"]["disk_hit_blocks"], 0)
        self.assertEqual(result["totals"]["miss_blocks"], 5)
        self.assertEqual(result["layers"]["gpu"]["evictions"], 5)
        self.assertEqual(result["layers"]["memory"]["peak_blocks"], 5)
        self.assertEqual(result["requests"][3]["memory_hit_blocks"], 2)

    def test_multi_node_random_load_balancing_uses_shared_disk(self):
        analyzer = load_analyzer()
        records = [
            {
                "timestamp": 1.0,
                "input_length": 512,
                "output_length": 1,
                "hash_ids": ["a"],
            },
            {
                "timestamp": 2.0,
                "input_length": 512,
                "output_length": 1,
                "hash_ids": ["a"],
            },
            {
                "timestamp": 3.0,
                "input_length": 512,
                "output_length": 1,
                "hash_ids": ["a"],
            },
        ]

        result = analyzer.analyze_records(
            records,
            block_bytes=1,
            capacities_blocks={"gpu": 1, "memory": 1, "disk": 1},
            trace_paths=["trace.jsonl"],
            include_requests=True,
            num_nodes=2,
            random_seed=1,
        )

        self.assertEqual(
            [request["node_index"] for request in result["requests"]],
            [0, 0, 1],
        )
        self.assertEqual(result["totals"]["gpu_hit_blocks"], 1)
        self.assertEqual(result["totals"]["memory_hit_blocks"], 0)
        self.assertEqual(result["totals"]["disk_hit_blocks"], 1)
        self.assertEqual(result["totals"]["miss_blocks"], 1)
        self.assertEqual(result["nodes"][0]["totals"]["gpu_hit_blocks"], 1)
        self.assertEqual(result["nodes"][1]["totals"]["disk_hit_blocks"], 1)
        self.assertEqual(result["layers"]["gpu"]["capacity_blocks"], 2)
        self.assertEqual(result["layers"]["disk"]["capacity_blocks"], 1)

    def test_cli_reads_multiple_traces_and_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            first = tmp_path / "first.jsonl"
            second = tmp_path / "second.jsonl"
            output = tmp_path / "result.json"
            first.write_text(
                json.dumps(
                    {
                        "timestamp": 1.0,
                        "input_length": 512,
                        "output_length": 1,
                        "hash_ids": ["a"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    {
                        "timestamp": 2.0,
                        "input_length": 512,
                        "output_length": 1,
                        "hash_ids": ["a"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--trace-path",
                    str(first),
                    str(second),
                    "--attention-type",
                    "gqa",
                    "--num-layers",
                    "1",
                    "--num-kv-heads",
                    "1",
                    "--head-dim",
                    "1",
                    "--kv-cache-dtype",
                    "float16",
                    "--trace-block-size",
                    "1",
                    "--gpu-kv-cache-gb",
                    "0.00000001",
                    "--ucm-memory-cache-gb",
                    "0.00000001",
                    "--ucm-filesystem-cache-gb",
                    "0.00000001",
                    "--num-nodes",
                    "2",
                    "--random-seed",
                    "1",
                    "--output",
                    str(output),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["totals"]["total_requests"], 2)
            self.assertEqual(data["totals"]["gpu_hit_blocks"], 1)
            self.assertEqual(data["trace_paths"], [str(first), str(second)])
            self.assertEqual(data["config"]["num_nodes"], 2)
            self.assertEqual(data["config"]["random_seed"], 1)

    def test_cli_uses_block_bytes_without_model_params(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            trace = tmp_path / "trace.jsonl"
            output = tmp_path / "result.json"
            trace.write_text(
                json.dumps(
                    {
                        "timestamp": 1.0,
                        "input_length": 512,
                        "output_length": 1,
                        "hash_ids": ["a", "b"],
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "timestamp": 2.0,
                        "input_length": 512,
                        "output_length": 1,
                        "hash_ids": ["a", "b"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--trace-path",
                    str(trace),
                    "--block-bytes",
                    "4",
                    "--gpu-kv-cache-gb",
                    "0.00000001",
                    "--ucm-memory-cache-gb",
                    "0.00000001",
                    "--ucm-filesystem-cache-gb",
                    "0.00000001",
                    "--output",
                    str(output),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["block_bytes"], 4)
            self.assertEqual(data["config"]["block_bytes_source"], "override")
            self.assertEqual(data["config"]["attention_type"], None)
            self.assertEqual(data["layers"]["gpu"]["capacity_blocks"], 2)
            self.assertEqual(data["totals"]["gpu_hit_blocks"], 2)

    def test_cli_infers_gqa_model_config_from_model_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_path = tmp_path / "model"
            model_path.mkdir()
            trace = tmp_path / "trace.jsonl"
            output = tmp_path / "result.json"
            (model_path / "config.json").write_text(
                json.dumps(
                    {
                        "hidden_size": 16,
                        "num_attention_heads": 2,
                        "num_hidden_layers": 2,
                        "num_key_value_heads": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            trace.write_text(
                json.dumps(
                    {
                        "timestamp": 1.0,
                        "input_length": 512,
                        "output_length": 1,
                        "hash_ids": ["a"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--trace-path",
                    str(trace),
                    "--model-path",
                    str(model_path),
                    "--kv-cache-dtype",
                    "float16",
                    "--trace-block-size",
                    "4",
                    "--gpu-kv-cache-gb",
                    "0.00000001",
                    "--ucm-memory-cache-gb",
                    "0.00000001",
                    "--ucm-filesystem-cache-gb",
                    "0.00000001",
                    "--output",
                    str(output),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["block_bytes"], 256)
            self.assertEqual(data["config"]["attention_type"], "gqa")
            self.assertEqual(data["config"]["num_layers"], 2)
            self.assertEqual(data["config"]["num_kv_heads"], 1)
            self.assertEqual(data["config"]["head_dim"], 8)

    def test_model_path_infers_mla_model_config(self):
        analyzer = load_analyzer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir)
            (model_path / "config.json").write_text(
                json.dumps(
                    {
                        "num_hidden_layers": 2,
                        "kv_lora_rank": 4,
                        "qk_rope_head_dim": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                model_path=model_path,
                attention_type=None,
                num_layers=None,
                num_kv_heads=None,
                head_dim=None,
                kv_lora_rank=None,
                qk_rope_head_dim=None,
            )

            params = analyzer.resolve_model_params(args)

            self.assertEqual(params["attention_type"], "mla")
            self.assertEqual(params["num_layers"], 2)
            self.assertEqual(params["kv_lora_rank"], 4)
            self.assertEqual(params["qk_rope_head_dim"], 2)


if __name__ == "__main__":
    unittest.main()
