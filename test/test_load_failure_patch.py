import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


def load_patch_function():
    utils = ModuleType("ucm.integration.vllm.patch.utils")
    utils.patch_or_inject = lambda target, name, replacement: setattr(
        target, name, replacement
    )
    utils.when_imported = lambda module_name: lambda function: function
    module_path = (
        Path(__file__).resolve().parents[1]
        / "ucm/integration/vllm/patch/load_failure_patch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "load_failure_patch_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {utils.__name__: utils}):
        spec.loader.exec_module(module)
    return module.patch_core_sched_scheduler


patch_core_sched_scheduler = load_patch_function()


class KVCacheManager:
    def __init__(self, groups):
        self.groups = groups
        self.num_kv_cache_groups = len(groups)

    def get_block_ids(self, request_id):
        return self.groups


def build_scheduler(groups, fail=False):
    class Scheduler:
        def __init__(self):
            self.kv_cache_manager = KVCacheManager(groups)

        def _update_requests_with_invalid_blocks(
            self, requests, invalid_block_ids, evict_blocks=True
        ):
            (request_blocks,) = self.kv_cache_manager.get_block_ids(
                requests[0].request_id
            )
            if fail:
                raise RuntimeError("load recovery failed")
            return {requests[0].request_id}, len(request_blocks), {request_blocks[-1]}

    patch_core_sched_scheduler(SimpleNamespace(Scheduler=Scheduler))
    return Scheduler()


def test_multi_group_load_failure_recovery_uses_group_zero_and_evicts_all_invalid():
    scheduler = build_scheduler(([10, 11], [20, 21]))
    original_get_block_ids = scheduler.kv_cache_manager.get_block_ids
    request = SimpleNamespace(request_id="request-1", num_output_placeholders=3)

    result = scheduler._update_requests_with_invalid_blocks(
        [request], {11, 20}, evict_blocks=True
    )

    assert result == ({"request-1"}, 2, {11, 20})
    assert request.num_output_placeholders == 0
    assert scheduler.kv_cache_manager.get_block_ids.__func__ is (
        original_get_block_ids.__func__
    )
    assert scheduler.kv_cache_manager.get_block_ids("request-1") == (
        [10, 11],
        [20, 21],
    )


def test_multi_group_load_failure_recovery_restores_get_block_ids_on_error():
    scheduler = build_scheduler(([10], [20]), fail=True)
    original_get_block_ids = scheduler.kv_cache_manager.get_block_ids
    request = SimpleNamespace(request_id="request-1", num_output_placeholders=3)

    with pytest.raises(RuntimeError, match="load recovery failed"):
        scheduler._update_requests_with_invalid_blocks([request], {20})

    assert scheduler.kv_cache_manager.get_block_ids.__func__ is (
        original_get_block_ids.__func__
    )
    assert scheduler.kv_cache_manager.get_block_ids("request-1") == ([10], [20])


def test_single_group_load_failure_recovery_delegates_unchanged():
    scheduler = build_scheduler(([10, 11],))
    request = SimpleNamespace(request_id="request-1", num_output_placeholders=3)

    result = scheduler._update_requests_with_invalid_blocks([request], {10})

    assert result == ({"request-1"}, 2, {11})
    assert request.num_output_placeholders == 0
