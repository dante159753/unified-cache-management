"""CPU tests for the FAWA layerwise connector's load and dump order.

The connector methods are loaded from source with ``ast`` so that the tests
need neither vLLM nor torch, in the same way as ``test_ucm_v41_compat.py``.
"""

import ast
import importlib.util
from concurrent.futures import Future
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Optional
from unittest.mock import Mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
HMA_PATH = REPO_ROOT / "ucm/integration/vllm/hma_connector.py"
LAYER_TUPLE_PATH = REPO_ROOT / "ucm/integration/vllm/fawa_layer_tuple.py"

BASE_METHODS = {
    "_submit_load_task",
    "_wait_load_task",
    "_submit_dump_task",
    "_submit_store_dump",
    "_collect_dump_plan",
}
LAYERWISE_METHODS = {
    "_reset_step_state",
    "_transfer_bytes_per_key",
    "_store_for_label",
    "_kv_layer_names_of",
    "_shard_keys_of",
    "start_load_kv",
    "_load_keys_for_store",
    "_extract_load_row_ptrs",
    "_submit_next_tuple_load",
    "_fill_load_pipeline",
    "_finish_loading_tuple",
    "_loading_tuple_has_arrived",
    "_load_tuples_until",
    "_finish_step_loads",
    "wait_for_layer_load",
    "save_kv_layer",
    "_dump_shard",
    "wait_for_save",
}
DATA_CLASSES = {
    "FAWALoadTask",
    "FAWADumpTask",
    "FAWAStoreDumpRows",
    "FAWADumpPlan",
    "FAWARequestDispatchMeta",
    "UCMFAWAConnectorMetadata",
}


def _load_layer_tuple_module():
    spec = importlib.util.spec_from_file_location("fawa_layer_tuple", LAYER_TUPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


layer_tuple = _load_layer_tuple_module()
metrics_updates = []


def _load_layerwise_connector_class():
    """Build a plain class holding the tested methods of both connectors."""
    tree = ast.parse(HMA_PATH.read_text(encoding="utf-8"))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    # Keep source order: later dataclasses refer to earlier ones.
    body = [node for node in tree.body if getattr(node, "name", None) in DATA_CLASSES]
    base_methods = [
        node
        for node in classes["UCMFAWAConnector"].body
        if getattr(node, "name", None) in BASE_METHODS
    ]
    # Subclass methods come last so that they override base methods.
    layerwise_methods = [
        node
        for node in classes["UCMFAWALayerWiseConnector"].body
        if getattr(node, "name", None) in LAYERWISE_METHODS
    ]
    body.append(
        ast.ClassDef(
            name="LayerWiseConnectorUnderTest",
            bases=[],
            keywords=[],
            body=base_methods + layerwise_methods,
            decorator_list=[],
        )
    )
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "__name__": __name__,
        "dataclass": dataclass,
        "field": field,
        "cached_property": cached_property,
        "Future": Future,
        "Optional": Optional,
        "np": np,
        "torch": NS(Tensor=object),
        "Task": object,
        "UcmKVStoreBaseV1": object,
        "KVConnectorMetadata": object,
        "split_layer_name": layer_tuple.split_layer_name,
        "is_draft_layer_name": layer_tuple.is_draft_layer_name,
        "logger": NS(error=lambda *_args: None, info=lambda *_args: None),
        "ucmmetrics": NS(update_stats=metrics_updates.append),
    }
    exec(compile(module, str(HMA_PATH), "exec"), namespace)
    return namespace


SYMBOLS = _load_layerwise_connector_class()
FAWARequestDispatchMeta = SYMBOLS["FAWARequestDispatchMeta"]
UCMFAWAConnectorMetadata = SYMBOLS["UCMFAWAConnectorMetadata"]


class RecordingStore:
    """Fake FA or WA store that records each shard transfer.

    ``events`` is shared by both stores and lists every submit and wait in
    order, so tests can see how many tuples were loading at the same time.
    """

    def __init__(self, label, transfers, events):
        self.label = label
        self.transfers = transfers
        self.events = events
        self.waited = []
        self.tasks = []

    def load_data(self, keys, shard_indices, ptrs):
        return self._record("load", keys, shard_indices, ptrs)

    def dump_data(self, keys, shard_indices, ptrs, event_handle):
        return self._record("dump", keys, shard_indices, ptrs)

    def _record(self, kind, keys, shard_indices, ptrs):
        (shard_index,) = set(shard_indices)
        self.transfers.append(
            (kind, self.label, shard_index, list(keys), np.asarray(ptrs).tolist())
        )
        self.events.append(("submit", kind, self.label, shard_index))
        task = NS(
            kind=kind, label=self.label, shard_index=shard_index, store=self, done=False
        )
        self.tasks.append(task)
        return task

    def wait(self, task):
        self.waited.append((task.kind, task.label, task.shard_index))
        self.events.append(("wait", task.kind, task.label, task.shard_index))


class FakeRankConsistency:
    def submit_load(self, store, _block_ids_by_request, keys, shard_indices, ptrs, **_):
        return store.load_data(keys, shard_indices, ptrs)

    def check_load(self, task):
        return task.done

    def wait_load(self, task):
        task.store.wait(task)

    def submit_dump(
        self, store, _block_ids_by_request, keys, shard_indices, ptrs, event_handle
    ):
        return store.dump_data(keys, shard_indices, ptrs, event_handle)


def layerwise_connector():
    """Connector for three layer tuples.

    Tuple 0 holds layer 0 SWA (WA) and layer 1 attention (FA), tuple 1 holds
    layer 2 SWA and layer 3 attention, and tuple 2 holds the draft layer SWA.
    """

    def column(group_id, layer_name):
        return layer_tuple.StoreRowColumn(group_id, 0, layer_name, 0, 128)

    fa_layers = ["model.layers.1.self_attn.attn", "model.layers.3.self_attn.attn"]
    wa_layers = [
        "model.layers.0.self_attn.swa_cache",
        "model.layers.2.self_attn.swa_cache",
        "mtp.0.self_attn.swa_cache",
    ]
    tuple_slot_by_layer = {
        fa_layers[0]: 0,
        wa_layers[0]: 0,
        fa_layers[1]: 1,
        wa_layers[1]: 1,
        wa_layers[2]: 2,
    }
    metrics_updates.clear()
    connector = SYMBOLS["LayerWiseConnectorUnderTest"]()
    connector.transfers, connector.events = [], []
    connector.fa_store = RecordingStore("FA", connector.transfers, connector.events)
    connector.wa_store = RecordingStore("WA", connector.transfers, connector.events)
    connector.tuple_slot_by_layer = tuple_slot_by_layer
    connector.fa_group_ids, connector.window_group_ids = [0], [1]
    connector.shard_plans = {
        "FA": layer_tuple.build_layer_tuple_shard_plan(
            [column(0, name) for name in fa_layers], tuple_slot_by_layer
        ),
        "WA": layer_tuple.build_layer_tuple_shard_plan(
            [column(1, name) for name in wa_layers], tuple_slot_by_layer
        ),
    }
    connector.shard_bytes = {"FA": 4096, "WA": 4096}
    connector.shard_index_by_layer = {
        label: {
            layer_name: shard_index
            for shard_index, layer_names in enumerate(plan.shard_layer_names)
            for layer_name in layer_names
        }
        for label, plan in connector.shard_plans.items()
    }
    connector.shard_keys_by_tuple_slot = {
        0: [("FA", 0), ("WA", 0)],
        1: [("FA", 1), ("WA", 1)],
        2: [("WA", 2)],
    }
    connector.kv_layer_names_by_decoder_layer = {
        layer_tuple.split_layer_name(name)[0]: frozenset([name])
        for name in tuple_slot_by_layer
    }
    connector._reset_step_state()
    connector.load_prefetch_depth = 1
    connector._connector_metadata = None
    connector._get_connector_metadata = lambda: connector._connector_metadata
    connector.is_mla = False
    connector.is_v41_layout = False
    connector.tp_size, connector.tp_rank = 1, 0
    connector.wa_dump_block_wise = True
    connector.tp_dump_tasks = {}
    connector._get_dump_event_handle = lambda: 7
    connector.device = NS(destroy_event_handle=Mock(), synchronize=Mock())
    connector._poll_completed_dump_tasks = Mock()
    connector._handle_load_err = Mock()
    connector._record_counter = Mock()
    connector._rank_consistency = FakeRankConsistency()
    # Full-row pointer values are 10 * column + key position, so tests can
    # see which row column and key each shard pointer came from.
    connector._extract_fa_ptr = lambda keys, *_args, **_kwargs: np.array(
        [[10 * col + key for col in range(2)] for key in range(len(keys))],
        dtype=np.uint64,
    )
    connector._extract_wa_ptr = lambda keys, _block_ids: np.array(
        [[10 * col + key for col in range(3)] for key in range(len(keys))],
        dtype=np.uint64,
    )
    return connector


def load_metadata(load_keys):
    return UCMFAWAConnectorMetadata(
        {
            "req": FAWARequestDispatchMeta(
                load_keys=load_keys,
                load_hash_end=len(load_keys),
                load_vllm_block_ids=([10, 11][: len(load_keys)], [10]),
            )
        }
    )


def submitted_shards(connector):
    return [(label, shard) for _, label, shard, _, _ in connector.transfers]


def assert_one_tuple_loading_at_a_time(connector):
    """Every tuple's loads are waited before the next tuple is submitted."""
    tuple_of = {("FA", 0): 0, ("WA", 0): 0, ("FA", 1): 1, ("WA", 1): 1, ("WA", 2): 2}
    unfinished = set()
    for action, _, label, shard in connector.events:
        if action == "submit":
            assert {tuple_of[key] for key in unfinished} <= {tuple_of[label, shard]}
            unfinished.add((label, shard))
        else:
            unfinished.discard((label, shard))


def test_start_load_kv_submits_only_the_first_tuple():
    connector = layerwise_connector()
    connector._connector_metadata = load_metadata([b"a", b"b"])

    connector.start_load_kv(None)

    assert submitted_shards(connector) == [("FA", 0), ("WA", 0)]
    fa_shard_0, wa_shard_0 = connector.transfers
    assert fa_shard_0[3] == [b"a", b"b"]
    assert fa_shard_0[4] == [[0], [1]]
    # WA rows only exist for the final matched boundary.
    assert wa_shard_0[3] == [b"b"]
    assert wa_shard_0[4] == [[0]]


def test_each_layer_waits_for_its_tuple_then_starts_the_next_one():
    connector = layerwise_connector()
    connector._connector_metadata = load_metadata([b"a", b"b"])
    connector.start_load_kv(None)

    # The hooks pass the attention module name, not a cache name.
    connector.wait_for_layer_load("model.layers.0.self_attn.attn")
    assert connector.wa_store.waited == [("load", "WA", 0)]
    assert connector.fa_store.waited == [("load", "FA", 0)]
    assert submitted_shards(connector)[2:] == [("FA", 1), ("WA", 1)]
    assert connector.transfers[2][4] == [[10], [11]]

    # Tuple 1 is still loading, so nothing new is submitted.
    connector.wait_for_layer_load("model.layers.1.self_attn.attn")
    assert len(connector.transfers) == 4

    # Tuple 1 has arrived: the next hook collects it and submits tuple 2.
    for task in connector.fa_store.tasks + connector.wa_store.tasks:
        task.done = True
    connector.wait_for_layer_load("model.layers.1.self_attn.attn")
    assert connector.fa_store.waited[-1] == ("load", "FA", 1)
    assert submitted_shards(connector)[4:] == [("WA", 2)]

    connector.wait_for_layer_load("model.layers.2.self_attn.attn")
    connector.wait_for_layer_load("model.layers.3.self_attn.attn")
    # The draft layer skips the hook; wait_for_save finishes its load.
    connector.wait_for_save()

    assert connector.wa_store.waited == [
        ("load", "WA", 0),
        ("load", "WA", 1),
        ("load", "WA", 2),
    ]
    assert_one_tuple_loading_at_a_time(connector)
    # FA: 2 shards x 2 keys, WA: 3 shards x 1 boundary key.
    assert metrics_updates == [{"load_bytes_total": 7 * 4096}]
    connector._handle_load_err.assert_not_called()
    connector.device.synchronize.assert_not_called()


def test_a_later_layer_loads_the_tuples_before_it_one_at_a_time():
    connector = layerwise_connector()
    connector._connector_metadata = load_metadata([b"a"])
    connector.start_load_kv(None)

    connector.wait_for_layer_load("model.layers.2.self_attn.attn")

    assert connector.fa_store.waited == [("load", "FA", 0), ("load", "FA", 1)]
    assert submitted_shards(connector) == [
        ("FA", 0),
        ("WA", 0),
        ("FA", 1),
        ("WA", 1),
        ("WA", 2),
    ]
    assert_one_tuple_loading_at_a_time(connector)


def test_prefetch_depth_keeps_two_tuples_loading():
    connector = layerwise_connector()
    connector.load_prefetch_depth = 2
    connector._connector_metadata = load_metadata([b"a", b"b"])

    connector.start_load_kv(None)
    # Tuples 0 and 1 both start loading before a layer asks for them.
    assert submitted_shards(connector) == [
        ("FA", 0),
        ("WA", 0),
        ("FA", 1),
        ("WA", 1),
    ]

    # Layer 0 waits only for tuple 0; the freed slot starts tuple 2.
    connector.wait_for_layer_load("model.layers.0.self_attn.attn")
    assert connector.fa_store.waited == [("load", "FA", 0)]
    assert connector.wa_store.waited == [("load", "WA", 0)]
    assert submitted_shards(connector)[4:] == [("WA", 2)]

    for task in connector.fa_store.tasks + connector.wa_store.tasks:
        task.done = True
    connector.wait_for_layer_load("model.layers.1.self_attn.attn")
    connector.wait_for_layer_load("model.layers.2.self_attn.attn")
    connector.wait_for_layer_load("model.layers.3.self_attn.attn")
    connector.wait_for_save()

    assert connector.wa_store.waited == [
        ("load", "WA", 0),
        ("load", "WA", 1),
        ("load", "WA", 2),
    ]
    tuple_of = {("FA", 0): 0, ("WA", 0): 0, ("FA", 1): 1, ("WA", 1): 1, ("WA", 2): 2}
    unfinished = set()
    for action, _, label, shard in connector.events:
        if action == "submit":
            unfinished.add((label, shard))
            assert len({tuple_of[key] for key in unfinished}) <= 2
        else:
            unfinished.discard((label, shard))
    assert metrics_updates == [{"load_bytes_total": 7 * 4096}]
    connector._handle_load_err.assert_not_called()


def test_v41_synchronizes_before_submitting_loads():
    connector = layerwise_connector()
    connector.is_v41_layout = True
    connector._connector_metadata = load_metadata([b"a"])

    connector.start_load_kv(None)

    connector.device.synchronize.assert_called_once_with()


def test_target_layer_that_skips_the_load_hook_is_recomputed():
    connector = layerwise_connector()
    connector._connector_metadata = load_metadata([b"a"])
    connector.start_load_kv(None)
    connector.wait_for_layer_load("model.layers.0.self_attn.attn")
    connector.wait_for_layer_load("model.layers.1.self_attn.attn")

    # Layers 2 and 3 read their caches without waiting for tuple 1.
    connector.wait_for_save()

    assert len(connector.fa_store.waited) == 2
    assert len(connector.wa_store.waited) == 3
    assert_one_tuple_loading_at_a_time(connector)
    # Tuple 2 only holds the draft layer, so only tuple 1 triggers recompute.
    assert connector._handle_load_err.call_args_list == [(("req",),)]


def test_a_tuple_is_dumped_once_all_its_layers_ran():
    connector = layerwise_connector()
    connector._connector_metadata = UCMFAWAConnectorMetadata(
        {
            "req": FAWARequestDispatchMeta(
                dump_keys=[b"c"],
                dump_hash_start=2,
                dump_hash_end=3,
                dump_vllm_block_ids=([12], [12]),
            )
        }
    )

    def dumped():
        return [(label, shard) for _, label, shard, _, _ in connector.transfers]

    connector.save_kv_layer("model.layers.0.self_attn.attn", None, None)
    assert dumped() == [("WA", 0)]
    connector.save_kv_layer("model.layers.0.self_attn.attn", None, None)
    assert dumped() == [("WA", 0)]
    connector.save_kv_layer("model.layers.1.self_attn.v41_attn", None, None)
    assert dumped() == [("WA", 0), ("FA", 0)]

    # The draft layer and layers 2-3 never call the hook, as in an ACL graph
    # replay; wait_for_save dumps the shards that are left.
    connector.wait_for_save()

    assert dumped() == [("WA", 0), ("FA", 0), ("FA", 1), ("WA", 1), ("WA", 2)]
    assert all(keys == [b"c"] for _, _, _, keys, _ in connector.transfers)
    assert connector.transfers[2][4] == [[10]]
    assert connector.transfers[4][4] == [[20]]
    assert len(connector.tp_dump_tasks[("req",)]) == 5
    assert metrics_updates == [{"save_bytes_total": 5 * 4096}]
