# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""Unit tests for scheduler-level async load (load_async) in UCM connectors."""

from types import SimpleNamespace

import pytest

vllm = pytest.importorskip("vllm")

from ucm.integration.vllm.ucm_connector import PendingLoadTask  # noqa: E402
from ucm.integration.vllm.ucm_connector import (
    RequestHasher,
    RequestMeta,
    UCMConnectorMetadata,
    UCMDirectConnector,
    UCMLayerWiseConnector,
    UCMWorkerMetadata,
)


class FakeTask:
    def __init__(self, task_id):
        self.task_id = task_id


class FakeStore:
    """Scriptable store: control per-task check/wait behavior."""

    def __init__(self, lookup_hit_blocks=0):
        self.lookup_hit_blocks = lookup_hit_blocks
        self.submitted = []
        self.in_flight: dict[int, bool] = {}
        self.check_errors: set[int] = set()
        self.wait_errors: set[int] = set()
        self.waited: list[int] = []
        self.submit_error = False
        self._next_id = 0

    def lookup_on_prefix(self, block_ids):
        return min(self.lookup_hit_blocks, len(block_ids)) - 1

    def load_data(self, block_ids, shard_index, dst_addr):
        if self.submit_error:
            raise RuntimeError("injected submit failure")
        task = FakeTask(self._next_id)
        self._next_id += 1
        self.submitted.append((list(block_ids), list(shard_index)))
        self.in_flight[task.task_id] = True
        return task

    def check(self, task):
        if task.task_id in self.check_errors:
            raise RuntimeError("injected check failure")
        return not self.in_flight.get(task.task_id, False)

    def wait(self, task):
        self.waited.append(task.task_id)
        if task.task_id in self.wait_errors:
            raise RuntimeError("injected wait failure")

    def complete(self, task_id):
        self.in_flight[task_id] = False


class FakeLayout:
    def __init__(self, n_layers=2, n_ptrs=2):
        self.n_layers = n_layers
        self.n_ptrs = n_ptrs

    def extract_block_addrs(self, vllm_block_ids, layer_first=False):
        import numpy as np

        n = len(vllm_block_ids)
        if layer_first:
            return np.zeros((self.n_layers, n, self.n_ptrs), dtype=np.int64)
        return np.zeros((n, self.n_layers, self.n_ptrs), dtype=np.int64)


class FakeMetrics:
    def __init__(self):
        self.stats = []

    def update_stats(self, stats):
        self.stats.append(stats)


BLOCK_SIZE = 4


def make_scheduler_connector(load_async=True, lookup_hit_blocks=8):
    conn = UCMDirectConnector.__new__(UCMDirectConnector)
    conn.load_async = load_async
    conn.block_size = BLOCK_SIZE
    conn.hash_block_size = BLOCK_SIZE
    conn.cp_world_size = 1
    conn.enable_record_traces = False
    conn.persist_token_threshold = 0
    conn.requests_meta = {}
    conn.store = FakeStore(lookup_hit_blocks=lookup_hit_blocks)
    conn._other_rank_hashers = []
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model="fake-model", dtype="bf16"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    conn.request_hasher = RequestHasher(vllm_config, 0)
    conn._seed = conn.request_hasher("UCM_HASH_SEED")
    conn._async_dump_req_ids = set()
    return conn


def make_worker_connector(cls=UCMDirectConnector):
    conn = cls.__new__(cls)
    conn.load_async = True
    conn.tp_rank = 0
    conn.tp_size = 1
    conn.is_mla = False
    conn._skip_null_vllm_blocks = False
    conn.store = FakeStore()
    conn.kv_cache_layout = FakeLayout()
    conn.block_data_size = 1024
    conn.requests_meta = {}
    conn._invalid_block_ids = set()
    conn._connector_worker_meta = UCMWorkerMetadata()
    conn._pending_load_reqs = {}
    conn._ready_recving = set()
    conn._async_dump_req_ids = set()
    conn._pending_dump_tasks = []
    if cls is UCMLayerWiseConnector:
        conn.layer_ids = [0, 1]
        conn.first_layer_id = 0
    return conn


def make_fawa_worker_connector():
    from ucm.integration.vllm.hma_connector import UCMFAWAConnector

    conn = UCMFAWAConnector.__new__(UCMFAWAConnector)
    conn.store = FakeStore()
    conn.block_data_size = 0
    conn.tp_dump_tasks = {}
    conn._invalid_block_ids = set()
    conn._connector_worker_meta = UCMWorkerMetadata()
    conn._pending_load_reqs = {}
    conn._ready_recving = set()
    return conn


def add_pending_load(conn, request_id="req-0"):
    task = conn.store.load_data([b"h0"], [0], [[0]])
    conn._pending_load_reqs[request_id] = PendingLoadTask(
        request_id=request_id,
        tasks=[(conn.store, task)],
        vllm_block_ids=[100],
        num_blocks=1,
        submit_ms=0,
    )
    return task


def make_request(request_id="req-0", num_blocks=8):
    token_ids = list(range(num_blocks * BLOCK_SIZE))
    return SimpleNamespace(
        request_id=request_id,
        all_token_ids=token_ids,
        num_tokens=len(token_ids),
        max_tokens=16,
    )


def make_scheduler_output(new_reqs=(), num_scheduled_tokens=None):
    return SimpleNamespace(
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], resumed_from_preemption=[]
        ),
        finished_req_ids=set(),
        num_scheduled_tokens=num_scheduled_tokens or {},
        preempted_req_ids=set(),
    )


def patch_metrics(monkeypatch):
    from ucm.integration.vllm import ucm_connector as mod

    fake = FakeMetrics()
    monkeypatch.setattr(mod, "ucmmetrics", fake)
    monkeypatch.setattr(mod, "_record_counter", lambda *a, **k: None)
    return fake


class TestSchedulerSide:
    def test_lookup_returns_async_flag(self, monkeypatch):
        patch_metrics(monkeypatch)
        conn = make_scheduler_connector(load_async=True, lookup_hit_blocks=8)
        request = make_request()
        hit_tokens, load_async = conn.get_num_new_matched_tokens(request, 0)
        assert hit_tokens > 0
        assert load_async is True
        assert conn.requests_meta[request.request_id].do_async_load is True

    def test_lookup_miss_returns_sync_flag(self, monkeypatch):
        patch_metrics(monkeypatch)
        conn = make_scheduler_connector(load_async=True, lookup_hit_blocks=0)
        request = make_request()
        hit_tokens, load_async = conn.get_num_new_matched_tokens(request, 0)
        assert hit_tokens == 0
        assert load_async is False

    def test_lookup_async_disabled(self, monkeypatch):
        patch_metrics(monkeypatch)
        conn = make_scheduler_connector(load_async=False, lookup_hit_blocks=8)
        request = make_request()
        hit_tokens, load_async = conn.get_num_new_matched_tokens(request, 0)
        assert hit_tokens > 0
        assert load_async is False

    def test_dispatch_emitted_once_then_wakeup_skips_load(self, monkeypatch):
        patch_metrics(monkeypatch)
        conn = make_scheduler_connector(load_async=True, lookup_hit_blocks=6)
        request = make_request(num_blocks=8)
        hit_tokens, load_async = conn.get_num_new_matched_tokens(request, 0)
        assert load_async is True
        assert hit_tokens == 6 * BLOCK_SIZE

        hit_blocks = hit_tokens // BLOCK_SIZE
        all_block_ids = list(range(100, 108))
        blocks = SimpleNamespace(get_block_ids=lambda: (all_block_ids,))
        conn.update_state_after_alloc(request, blocks, hit_tokens)
        req_meta = conn.requests_meta[request.request_id]
        assert req_meta.load_async_pending is True

        meta = conn.build_connector_meta(make_scheduler_output())
        assert isinstance(meta, UCMConnectorMetadata)
        dispatch = meta.request_meta[request.request_id]
        assert dispatch.is_async_load is True
        assert len(dispatch.load_block_ids[0]) == hit_blocks
        assert dispatch.load_block_ids[1] == all_block_ids[:hit_blocks]
        assert len(dispatch.dump_block_ids[0]) == 0
        assert req_meta.load_async_pending is False
        assert req_meta.async_loaded is True

        meta2 = conn.build_connector_meta(make_scheduler_output())
        assert request.request_id not in meta2.request_meta

        remaining = request.num_tokens - hit_tokens
        new_req = SimpleNamespace(
            req_id=request.request_id, block_ids=[list(range(100, 108))]
        )
        meta3 = conn.build_connector_meta(
            make_scheduler_output(
                new_reqs=[new_req],
                num_scheduled_tokens={request.request_id: remaining},
            )
        )
        dispatch3 = meta3.request_meta[request.request_id]
        assert dispatch3.is_async_load is False
        assert len(dispatch3.load_block_ids[0]) == 0
        assert len(dispatch3.dump_block_ids[0]) > 0
        assert req_meta.async_loaded is False

    def test_alloc_veto_cancels_async(self, monkeypatch):
        patch_metrics(monkeypatch)
        conn = make_scheduler_connector(load_async=True, lookup_hit_blocks=8)
        request = make_request()
        conn.get_num_new_matched_tokens(request, 0)
        blocks = SimpleNamespace(get_block_ids=lambda: ([],))
        conn.update_state_after_alloc(request, blocks, 0)
        req_meta = conn.requests_meta[request.request_id]
        assert req_meta.do_async_load is False
        assert req_meta.load_async_pending is False
        meta = conn.build_connector_meta(make_scheduler_output())
        assert request.request_id not in meta.request_meta


class TestWorkerSide:
    def test_poll_reports_when_all_tasks_done(self, monkeypatch):
        patch_metrics(monkeypatch)
        conn = make_worker_connector()
        conn._submit_async_load("req-0", [b"h0", b"h1"], [100, 101])
        assert "req-0" in conn._pending_load_reqs

        assert conn.get_finished(set()) == (None, None)
        assert "req-0" in conn._pending_load_reqs

        conn.store.complete(0)
        finished_sending, finished_recving = conn.get_finished(set())
        assert finished_sending is None
        assert finished_recving == {"req-0"}
        assert conn._pending_load_reqs == {}
        assert conn._invalid_block_ids == set()

    def test_poll_failure_reports_invalid_and_recving(self, monkeypatch):
        patch_metrics(monkeypatch)
        conn = make_worker_connector()
        conn._submit_async_load("req-0", [b"h0", b"h1"], [100, 101])
        conn.store.wait_errors.add(0)
        conn.store.complete(0)
        _, finished_recving = conn.get_finished(set())
        assert finished_recving == {"req-0"}
        assert conn._invalid_block_ids == {100, 101}
        assert "req-0" in conn._connector_worker_meta.load_failed_reqs

    def test_submit_failure_reports_next_poll(self, monkeypatch):
        patch_metrics(monkeypatch)
        conn = make_worker_connector()
        conn.store.submit_error = True
        conn._submit_async_load("req-0", [b"h0"], [100])
        assert conn._pending_load_reqs == {}
        _, finished_recving = conn.get_finished(set())
        assert finished_recving == {"req-0"}
        assert conn._invalid_block_ids == {100}

    def test_empty_async_load_reports_next_poll(self, monkeypatch):
        patch_metrics(monkeypatch)
        conn = make_worker_connector()
        conn._submit_async_load("req-0", [], [])
        assert conn._pending_load_reqs == {}
        _, finished_recving = conn.get_finished(set())
        assert finished_recving == {"req-0"}
        assert conn._invalid_block_ids == set()

    def test_aborted_request_tasks_survive_until_done(self, monkeypatch):
        patch_metrics(monkeypatch)
        conn = make_worker_connector()
        conn._submit_async_load("req-0", [b"h0"], [100])
        _, finished_recving = conn.get_finished({"req-0"})
        assert finished_recving is None
        assert "req-0" in conn._pending_load_reqs
        conn.store.complete(0)
        _, finished_recving = conn.get_finished(set())
        assert finished_recving == {"req-0"}

    def test_layerwise_submits_all_layers(self, monkeypatch):
        patch_metrics(monkeypatch)
        conn = make_worker_connector(UCMLayerWiseConnector)
        conn._submit_async_load("req-0", [b"h0", b"h1"], [100, 101])
        pending = conn._pending_load_reqs["req-0"]
        assert len(pending.tasks) == len(conn.layer_ids)
        shard_layers = [shards[0] for _, shards in conn.store.submitted]
        assert shard_layers == conn.layer_ids

        conn.store.complete(0)
        assert conn.get_finished(set()) == (None, None)
        conn.store.complete(1)
        _, finished_recving = conn.get_finished(set())
        assert finished_recving == {"req-0"}

    def test_check_failure_persists_until_all_tasks_done(self, monkeypatch):
        patch_metrics(monkeypatch)
        conn = make_worker_connector(UCMLayerWiseConnector)
        conn._submit_async_load("req-0", [b"h0"], [100])
        conn.store.check_errors.add(0)
        assert conn.get_finished(set()) == (None, None)
        assert conn._pending_load_reqs["req-0"].failed is True

        conn.store.check_errors.clear()
        conn.store.complete(0)
        conn.store.complete(1)
        _, finished_recving = conn.get_finished(set())
        assert finished_recving == {"req-0"}
        assert conn._invalid_block_ids == {100}

    def test_failed_pending_load_drains_all_completed_tasks(self, monkeypatch):
        patch_metrics(monkeypatch)
        conn = make_worker_connector(UCMLayerWiseConnector)
        conn._submit_async_load("req-0", [b"h0"], [100])
        conn._pending_load_reqs["req-0"].failed = True
        conn.store.complete(0)
        conn.store.complete(1)

        _, finished_recving = conn.get_finished(set())

        assert finished_recving == {"req-0"}
        assert conn.store.waited == [0, 1]

    def test_wait_failure_does_not_skip_later_completed_tasks(self, monkeypatch):
        patch_metrics(monkeypatch)
        conn = make_worker_connector(UCMLayerWiseConnector)
        conn._submit_async_load("req-0", [b"h0"], [100])
        conn.store.complete(0)
        conn.store.complete(1)
        conn.store.wait_errors.add(0)

        _, finished_recving = conn.get_finished(set())

        assert finished_recving == {"req-0"}
        assert conn.store.waited == [0, 1]
        assert conn._invalid_block_ids == {100}

    def test_check_failure_drains_task_before_reporting_completion(self, monkeypatch):
        patch_metrics(monkeypatch)
        conn = make_worker_connector()
        conn._submit_async_load("req-0", [b"h0"], [100])
        conn.store.check_errors.add(0)

        _, finished_recving = conn.get_finished(set())

        assert finished_recving == {"req-0"}
        assert conn.store.waited == [0]
        assert conn._invalid_block_ids == {100}


class TestFAWAWorkerSide:
    def test_aborted_pending_load_is_not_reported_as_finished_sending(
        self, monkeypatch
    ):
        patch_metrics(monkeypatch)
        conn = make_fawa_worker_connector()
        task = add_pending_load(conn)

        assert conn.get_finished({"req-0"}) == (None, None)
        assert "req-0" in conn._pending_load_reqs

        conn.store.complete(task.task_id)
        assert conn.get_finished(set()) == (None, {"req-0"})

    def test_completed_aborted_load_is_reported_only_as_finished_recving(
        self, monkeypatch
    ):
        patch_metrics(monkeypatch)
        conn = make_fawa_worker_connector()
        task = add_pending_load(conn)
        conn.store.complete(task.task_id)

        assert conn.get_finished({"req-0"}) == (None, {"req-0"})


class TestWhitelist:
    def test_supported_connectors(self):
        from ucm.integration.vllm.hma_connector import UCMFAWAConnector

        assert UCMDirectConnector._supports_async_load is True
        assert UCMLayerWiseConnector._supports_async_load is True
        assert UCMFAWAConnector._supports_async_load is True

    def test_unsupported_connectors(self):
        from ucm.integration.vllm.blend_connector import UCMBlendConnector
        from ucm.integration.vllm.ucm_connector import (
            UCMCPConnector,
            UCMHMAConnector,
            UCMHybridLinearAttentionConnector,
            UCMLiteConnector,
            UCMMockConnector,
            UCMPDConnector,
        )

        for cls in (
            UCMBlendConnector,
            UCMCPConnector,
            UCMHMAConnector,
            UCMHybridLinearAttentionConnector,
            UCMLiteConnector,
            UCMMockConnector,
            UCMPDConnector,
        ):
            assert cls._supports_async_load is False, cls.__name__


class TestMultiGroupInvalidBlocksPatch:
    def _make_scheduler(self, group_blocks, group_block_sizes):
        groups = [
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=size))
            for size in group_block_sizes
        ]
        return SimpleNamespace(
            kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
            kv_cache_manager=SimpleNamespace(
                get_block_ids=lambda req_id: group_blocks[req_id]
            ),
            block_size=min(group_block_sizes),
        )

    def test_multi_group_truncates_to_min_prefix(self):
        from ucm.integration.vllm.patch.load_failure_patch import (
            _update_multi_group_requests_with_invalid_blocks,
        )

        scheduler = self._make_scheduler(
            {"req-0": ([1, 2, 3, 4], [11, 12])},
            [4, 8],
        )
        request = SimpleNamespace(
            request_id="req-0", num_computed_tokens=16, num_output_placeholders=3
        )
        affected, total_tokens, to_evict = (
            _update_multi_group_requests_with_invalid_blocks(
                scheduler, [request], {3, 12}, {}, evict_blocks=True
            )
        )
        assert affected == {"req-0"}
        assert request.num_computed_tokens == 8
        assert request.num_output_placeholders == 0
        assert total_tokens == 8
        assert to_evict == {3, 4, 12}

    def test_multi_group_unaffected_request(self):
        from ucm.integration.vllm.patch.load_failure_patch import (
            _update_multi_group_requests_with_invalid_blocks,
        )

        scheduler = self._make_scheduler(
            {"req-0": ([1, 2], [11])},
            [4, 8],
        )
        request = SimpleNamespace(
            request_id="req-0", num_computed_tokens=8, num_output_placeholders=1
        )
        affected, total_tokens, _ = _update_multi_group_requests_with_invalid_blocks(
            scheduler, [request], {99}, {}, evict_blocks=False
        )
        assert affected == set()
        assert total_tokens == 0
        assert request.num_computed_tokens == 8

    def test_v018_async_failure_updates_external_computed_tokens(self):
        from ucm.integration.vllm.patch.load_failure_patch import (
            _update_multi_group_requests_with_invalid_blocks,
        )

        scheduler = self._make_scheduler(
            {"req-0": ([1, 2, 3, 4], [11, 12])},
            [4, 8],
        )
        request = SimpleNamespace(
            request_id="req-0",
            status=SimpleNamespace(name="WAITING_FOR_REMOTE_KVS"),
            num_computed_tokens=16,
            num_cached_tokens=0,
            num_external_computed_tokens=16,
            num_output_placeholders=1,
        )

        affected, total_tokens, _ = _update_multi_group_requests_with_invalid_blocks(
            scheduler, [request], {12}, evict_blocks=False
        )

        assert affected == {"req-0"}
        assert total_tokens == 8
        assert request.num_computed_tokens == 8
        assert request.num_external_computed_tokens == 8

    def test_v018_sync_failure_uses_cached_token_count(self):
        from ucm.integration.vllm.patch.load_failure_patch import (
            _update_multi_group_requests_with_invalid_blocks,
        )

        scheduler = self._make_scheduler(
            {"req-0": ([1, 2, 3, 4, 5], [11, 12, 13])},
            [4, 8],
        )
        request = SimpleNamespace(
            request_id="req-0",
            status=SimpleNamespace(name="RUNNING"),
            num_computed_tokens=20,
            num_cached_tokens=16,
            num_external_computed_tokens=8,
            num_output_placeholders=1,
        )

        affected, total_tokens, _ = _update_multi_group_requests_with_invalid_blocks(
            scheduler, [request], {3}, evict_blocks=False
        )

        assert affected == {"req-0"}
        assert total_tokens == 8
        assert request.num_computed_tokens == 8
        assert request.num_external_computed_tokens == 0


class TestLoadFailurePatchRouting:
    def test_generic_patch_is_enabled_for_v018(self):
        from ucm.integration.vllm.patch import apply_patch

        should_apply = getattr(
            apply_patch, "_should_apply_generic_load_failure_patch", None
        )
        assert should_apply is not None
        assert should_apply("0.18.0") is True
        assert should_apply("0.20.2") is True
        assert should_apply("0.11.0") is False
