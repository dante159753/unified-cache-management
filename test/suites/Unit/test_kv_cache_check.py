import ast
import ctypes
import hashlib
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


class StoreNotFoundError(RuntimeError):
    pass


class StoreUnhealthyError(RuntimeError):
    pass


class DelayedStore:
    def __init__(self, tensor_sizes, name="test"):
        self.config = {"tensor_size_list": tensor_sizes, "unique_id": name}
        self.data = {}
        self.loads = {}
        self.ready = False
        self.error = None
        self.corrupt = False
        self.retain_existing = False

    def dump_data(self, block_ids, shard_indices, ptrs, event_handle):
        for block_id, shard, row in zip(block_ids, shard_indices, ptrs):
            if self.retain_existing and (block_id, shard) in self.data:
                continue
            self.data[block_id, shard] = [
                ctypes.string_at(int(ptr), size) if ptr else bytes(size)
                for ptr, size in zip(row, self.config["tensor_size_list"])
            ]
        return object()

    def load_data(self, block_ids, shard_indices, ptrs):
        task = object()
        self.loads[id(task)] = (block_ids, shard_indices, ptrs.copy())
        return task

    def check(self, task):
        if self.error:
            raise self.error
        return self.ready

    def wait(self, task):
        if self.error:
            raise self.error
        load = self.loads.pop(id(task), None)
        if load is None:
            return
        block_ids, shard_indices, ptrs = load
        for block_id, shard, row in zip(block_ids, shard_indices, ptrs):
            for ptr, data in zip(row, self.data[block_id, shard]):
                if ptr and data:
                    if self.corrupt:
                        data = bytes([data[0] ^ 1]) + data[1:]
                    ctypes.memmove(int(ptr), data, len(data))


@pytest.fixture
def modules(monkeypatch):
    logger_module = ModuleType("ucm.logger")
    logger_module.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "ucm.logger", logger_module)
    errors_module = ModuleType("ucm.store.pipeline.errors")
    errors_module.StoreNotFoundError = StoreNotFoundError
    errors_module.StoreUnhealthyError = StoreUnhealthyError
    monkeypatch.setitem(sys.modules, errors_module.__name__, errors_module)
    store_module = ModuleType("ucm.store.ucmstore_v1")
    store_module.UcmKVStoreBaseV1 = DelayedStore
    monkeypatch.setitem(sys.modules, store_module.__name__, store_module)

    loaded = []
    for name in ("kv_cache_check", "rank_consistency"):
        module_name = f"ucm.integration.vllm.{name}"
        spec = importlib.util.spec_from_file_location(
            module_name, ROOT / "ucm/integration/vllm" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


@pytest.fixture(params=[True, False], ids=["consistency-on", "consistency-off"])
def transfer(modules, request):
    check_module, consistency_module = modules
    kv = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    checker = check_module.KVCacheCheck({"layer": kv})
    manager = consistency_module.RankConsistencyManager(
        is_scheduler=False, use_consistency_manager=request.param
    )
    manager.kv_cache_check = checker
    store = DelayedStore([16])
    return kv, checker, manager, store


def test_dump_hashes_source_before_submission_and_load_checks_after_wait(
    transfer, caplog
):
    kv, checker, manager, store = transfer
    block_id = b"a" * 16
    source = np.array([[kv[1].data_ptr()]], dtype=np.uint64)
    destination = np.array([[kv[3].data_ptr()]], dtype=np.uint64)
    expected = kv[1].clone()
    original_dump = store.dump_data

    def dump_and_reuse_source(*args):
        task = original_dump(*args)
        kv[1].fill_(99)
        return task

    store.dump_data = dump_and_reuse_source
    dump_task = manager.submit_dump(store, {}, [block_id], [0], source, 0)
    manager.wait_dump(dump_task)
    checker.verify_load = Mock(wraps=checker.verify_load)
    task = manager.submit_load(store, {}, [block_id], [0], destination)
    assert not manager.check_load(task)
    store.ready = True
    assert manager.check_load(task)
    checker.verify_load.assert_not_called()
    assert not torch.equal(kv[3], expected)
    manager.wait_load(task)
    checker.verify_load.assert_called_once()
    assert torch.equal(kv[3], expected)
    assert "MD5 mismatch" not in caplog.text


def test_corrupt_load_logs_block_and_both_md5_values(transfer, caplog):
    kv, _, manager, store = transfer
    block_id = b"b" * 16
    source = np.array([[kv[0].data_ptr()]], dtype=np.uint64)
    destination = np.array([[kv[2].data_ptr()]], dtype=np.uint64)
    dump_task = manager.submit_dump(store, {}, [block_id], [5], source, 0)
    manager.wait_dump(dump_task)
    original = kv[0].view(torch.uint8).numpy().tobytes()
    store.corrupt = True
    task = manager.submit_load(
        store, {"load-request": [b"rank-zero-block"]}, [block_id], [5], destination
    )
    assert "MD5 mismatch" not in caplog.text
    manager.wait_load(task)
    assert block_id.hex() in caplog.text
    assert "shard_index=5" in caplog.text
    assert "request_ids=['load-request']" in caplog.text
    assert f"expected_md5={hashlib.md5(original).hexdigest()}" in caplog.text
    changed = bytes([original[0] ^ 1]) + original[1:]
    assert f"actual_md5={hashlib.md5(changed).hexdigest()}" in caplog.text


def test_shared_block_loads_keep_their_request_ids_until_completion(transfer, caplog):
    kv, _, manager, store = transfer
    block_id = b"shared-block"
    source = np.array([[kv[0].data_ptr()]], dtype=np.uint64)
    dump_task = manager.submit_dump(
        store, {"dump-request": {block_id}}, [block_id], [2], source, 0
    )
    manager.wait_dump(dump_task)
    requests = {"first-load": [block_id]}
    first_task = manager.submit_load(
        store,
        requests,
        [block_id],
        [2],
        np.array([[kv[1].data_ptr()]], dtype=np.uint64),
    )
    requests.clear()
    requests["second-load"] = [block_id]
    second_task = manager.submit_load(
        store,
        requests,
        [block_id],
        [2],
        np.array([[kv[2].data_ptr()]], dtype=np.uint64),
    )
    requests.clear()

    store.corrupt = True
    manager.wait_load(second_task)
    store.corrupt = False
    manager.wait_load(first_task)
    errors = [
        record.message for record in caplog.records if "MD5 mismatch" in record.message
    ]
    assert len(errors) == 1
    assert "request_ids=['second-load']" in errors[0]
    assert "first-load" not in errors[0]
    assert "dump-request" not in errors[0]


def test_unknown_blocks_skip_hbm_checksum(transfer, caplog):
    kv, checker, manager, store = transfer
    block_id = b"c" * 16
    store.data[block_id, 0] = [bytes(16)]
    checker._md5 = Mock(side_effect=AssertionError("unexpected HBM read"))
    task = manager.submit_load(
        store, {}, [block_id], [0], np.array([[kv[0].data_ptr()]], dtype=np.uint64)
    )
    manager.wait_load(task)
    checker._md5.assert_not_called()
    assert "MD5 mismatch" not in caplog.text


def test_store_and_shard_records_are_independent(transfer, caplog):
    kv, checker, _, store = transfer
    other_store = DelayedStore([16], name="other")
    block_id = b"d" * 16
    checks = []
    for index, (target_store, shard) in enumerate(
        [(store, 0), (store, 1), (other_store, 0)]
    ):
        ptrs = np.array([[kv[index].data_ptr()]], dtype=np.uint64)
        checker.record_dump(
            target_store, [block_id], [shard], ptrs, request_ids=("dump-request",)
        )
        checks.extend(
            checker.prepare_load(
                target_store, [block_id], [shard], ptrs, request_ids=("load-request",)
            )
        )
    checker.verify_load(checks)
    assert "MD5 mismatch" not in caplog.text
    kv[1, 0] = -1
    checker.verify_load(checks)
    errors = [
        record.message for record in caplog.records if "MD5 mismatch" in record.message
    ]
    assert len(errors) == 1
    assert "shard_index=1 store=test" in errors[0]


def test_offsets_multiple_tensors_and_null_slots_hash_only_transferred_bytes(
    modules, caplog
):
    check_module, _ = modules
    backing = torch.arange(128, dtype=torch.uint8)
    checker = check_module.KVCacheCheck({"layer": (backing[8:40:2], backing[72:100])})
    store = DelayedStore([4, 6, 16, 0])
    ptrs = np.array(
        [[backing[12:].data_ptr(), backing[75:].data_ptr(), 0, 0]],
        dtype=np.uint64,
    )
    checker.record_dump(store, [b"e" * 16], [3], ptrs, request_ids=("dump-request",))
    checks = checker.prepare_load(
        store, [b"e" * 16], [3], ptrs, request_ids=("load-request",)
    )
    expected = bytes(range(12, 16)) + bytes(range(75, 81))
    assert checker._dump_md5[store, b"e" * 16, 3] == {hashlib.md5(expected).hexdigest()}
    backing[11] = 255
    backing[81] = 255
    checker.verify_load(checks)
    assert "MD5 mismatch" not in caplog.text
    backing[15] = 255
    checker.verify_load(checks)
    assert "MD5 mismatch" in caplog.text


@pytest.mark.parametrize("failure_stage", ["submit", "poll", "wait"])
def test_failed_load_does_not_check_partial_data(transfer, failure_stage):
    kv, checker, manager, store = transfer
    ptrs = np.array([[kv[0].data_ptr()]], dtype=np.uint64)
    block_id = b"f" * 16
    dump_task = manager.submit_dump(store, {}, [block_id], [0], ptrs, 0)
    manager.wait_dump(dump_task)
    checker.verify_load = Mock()
    if failure_stage == "submit":
        store.load_data = Mock(side_effect=StoreNotFoundError("missing"))
        with pytest.raises(StoreNotFoundError):
            manager.submit_load(store, {}, [block_id], [0], ptrs)
    else:
        task = manager.submit_load(store, {}, [block_id], [0], ptrs)
        store.error = StoreNotFoundError("missing")
        with pytest.raises(StoreNotFoundError):
            if failure_stage == "poll":
                manager.check_load(task)
            else:
                manager.wait_load(task)
    checker.verify_load.assert_not_called()
    assert not manager._load_checks
    assert not manager._load_task_contexts


def test_load_accepts_both_versions_when_a_dump_is_recorded_during_load(
    transfer, caplog
):
    kv, checker, _, store = transfer
    block_id = b"g" * 16
    first = np.array([[kv[0].data_ptr()]], dtype=np.uint64)
    second = np.array([[kv[1].data_ptr()]], dtype=np.uint64)
    checker.record_dump(store, [block_id], [0], first, request_ids=("first-dump",))
    checks = checker.prepare_load(
        store, [block_id], [0], first, request_ids=("first-load",)
    )
    second_checks = checker.prepare_load(
        store, [block_id], [0], second, request_ids=("second-load",)
    )
    checker.record_dump(store, [block_id], [0], second, request_ids=("second-dump",))
    checker.verify_load(checks)
    checker.verify_load(second_checks)
    assert "MD5 mismatch" not in caplog.text


@pytest.mark.parametrize("retain_existing", [True, False], ids=["dedup", "overwrite"])
def test_repeated_dump_accepts_retained_or_rewritten_bytes_and_detects_corruption(
    transfer, caplog, retain_existing
):
    kv, checker, manager, store = transfer
    store.retain_existing = retain_existing
    block_id = b"redump"
    digests = []
    for row in (0, 1):
        source = np.array([[kv[row].data_ptr()]], dtype=np.uint64)
        digests.append(
            hashlib.md5(kv[row].view(torch.uint8).numpy().tobytes()).hexdigest()
        )
        requests = {f"dump-{row}": {block_id}}
        task = manager.submit_dump(store, requests, [block_id], [3], source, 0)
        requests.clear()
        manager.wait_dump(task)

    assert "MD5 differs between dumps" in caplog.text
    assert "request_ids=['dump-1']" in caplog.text
    assert checker._dump_md5[store, block_id, 3] == set(digests)
    destination = np.array([[kv[2].data_ptr()]], dtype=np.uint64)
    task = manager.submit_load(
        store, {"load-redump": [block_id]}, [block_id], [3], destination
    )
    manager.wait_load(task)
    assert torch.equal(kv[2], kv[0 if retain_existing else 1])
    assert "MD5 mismatch" not in caplog.text

    store.corrupt = True
    task = manager.submit_load(
        store, {"load-corrupt": [block_id]}, [block_id], [3], destination
    )
    manager.wait_load(task)
    errors = [
        record.message for record in caplog.records if record.levelno == logging.ERROR
    ]
    assert len(errors) == 1
    assert "MD5 mismatch" in errors[0]
    assert "request_ids=['load-corrupt']" in errors[0]
    assert all(digest in errors[0] for digest in digests)


def test_identical_redumps_keep_one_candidate_without_conflict_warning(
    transfer, caplog
):
    kv, checker, manager, store = transfer
    block_id = b"same-bytes"
    ptrs = np.array([[kv[0].data_ptr()]], dtype=np.uint64)
    for _ in range(3):
        task = manager.submit_dump(store, {}, [block_id], [0], ptrs, 0)
        manager.wait_dump(task)
    assert len(checker._dump_md5[store, block_id, 0]) == 1
    assert "MD5 differs between dumps" not in caplog.text


@pytest.mark.parametrize("failure_stage", ["submit", "wait", "unhealthy"])
def test_failed_dump_keeps_source_history(transfer, caplog, failure_stage):
    kv, _, manager, store = transfer
    block_id = b"failed-dump"
    source = np.array([[kv[1].data_ptr()]], dtype=np.uint64)
    original = kv[1].view(torch.uint8).numpy().tobytes()
    if failure_stage == "submit":
        store.dump_data = Mock(side_effect=RuntimeError("dump submission failed"))
        with pytest.raises(RuntimeError, match="dump submission failed"):
            manager.submit_dump(store, {}, [block_id], [0], source, 0)
    else:
        task = manager.submit_dump(store, {}, [block_id], [0], source, 0)
        if failure_stage == "unhealthy":
            store.error = StoreUnhealthyError("unhealthy")
            manager.wait_dump(task)
        else:
            store.error = RuntimeError("dump wait failed")
            with pytest.raises(RuntimeError, match="dump wait failed"):
                manager.wait_dump(task)
        store.error = None
    # A failed dump does not establish which bytes the Store retained.
    store.data[block_id, 0] = [original]
    destination = np.array([[kv[2].data_ptr()]], dtype=np.uint64)
    task = manager.submit_load(store, {}, [block_id], [0], destination)
    manager.wait_load(task)
    assert torch.equal(kv[2], kv[1])
    assert "MD5 mismatch" not in caplog.text

    store.corrupt = True
    task = manager.submit_load(
        store, {"after-failed-dump": [block_id]}, [block_id], [0], destination
    )
    manager.wait_load(task)
    assert "MD5 mismatch" in caplog.text
    assert "request_ids=['after-failed-dump']" in caplog.text


def test_load_before_dump_wait_still_checks_every_load(transfer, caplog):
    kv, _, manager, store = transfer
    block_id = b"pending-dump"
    source = np.array([[kv[0].data_ptr()]], dtype=np.uint64)
    destination = np.array([[kv[2].data_ptr()]], dtype=np.uint64)
    dump = manager.submit_dump(store, {}, [block_id], [0], source, 0)
    load = manager.submit_load(store, {}, [block_id], [0], destination)
    manager.wait_load(load)
    assert torch.equal(kv[2], kv[0])
    assert "MD5 mismatch" not in caplog.text

    store.corrupt = True
    load = manager.submit_load(
        store, {"load-pending": [block_id]}, [block_id], [0], destination
    )
    manager.wait_load(load)
    assert "MD5 mismatch" in caplog.text
    assert "request_ids=['load-pending']" in caplog.text
    assert "MD5 check skipped" not in caplog.text
    manager.wait_dump(dump)


def test_overlapping_redump_checks_new_bytes_without_waiting_for_dump(transfer, caplog):
    kv, _, manager, store = transfer
    block_id = b"overlapping-redump"
    first = manager.submit_dump(
        store, {}, [block_id], [0], np.array([[kv[0].data_ptr()]], dtype=np.uint64), 0
    )
    manager.wait_dump(first)
    load = manager.submit_load(
        store, {}, [block_id], [0], np.array([[kv[2].data_ptr()]], dtype=np.uint64)
    )
    second = manager.submit_dump(
        store, {}, [block_id], [0], np.array([[kv[1].data_ptr()]], dtype=np.uint64), 0
    )
    manager.wait_load(load)
    assert torch.equal(kv[2], kv[1])
    assert "MD5 mismatch" not in caplog.text
    assert "MD5 check skipped" not in caplog.text
    manager.wait_dump(second)


def test_checker_is_disabled_by_default(modules):
    _, consistency_module = modules
    manager = consistency_module.RankConsistencyManager(is_scheduler=False)
    assert manager.kv_cache_check is None
    store = Mock(spec=DelayedStore)
    store.dump_data.return_value = object()
    store.load_data.return_value = object()
    ptrs = np.array([[1]], dtype=np.uint64)
    manager.submit_dump(store, {}, [b"h" * 16], [0], ptrs, 0)
    task = manager.submit_load(store, {}, [b"h" * 16], [0], ptrs)
    manager.wait_load(task)
    assert not manager._load_checks


@pytest.mark.parametrize(
    "config", [{}, {"enable_kv_cache_check": False}, {"enable_kv_cache_check": True}]
)
def test_connector_enables_check_after_registering_kv_caches(modules, config):
    check_module, consistency_module = modules
    tree = ast.parse(
        (ROOT / "ucm/integration/vllm/ucm_connector.py").read_text(encoding="utf-8-sig")
    )
    connector_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UCMConnector"
    )
    register = next(
        node
        for node in connector_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "register_kv_caches"
    )
    register.decorator_list = []

    class DirectConnector:
        def __init__(self):
            self._rank_consistency = consistency_module.RankConsistencyManager(
                is_scheduler=False
            )
            self.register_kv_caches = Mock()

    inner = DirectConnector()
    kv_caches = {"layer": torch.zeros(8, dtype=torch.bfloat16)}

    def create_checker(caches):
        inner.register_kv_caches.assert_called_once_with(caches)
        return check_module.KVCacheCheck(caches)

    factory = Mock(side_effect=create_checker)
    namespace = {
        "torch": torch,
        "UCMDirectConnector": DirectConnector,
        "KVCacheCheck": factory,
    }
    exec(
        compile(
            ast.Module(body=[register], type_ignores=[]), "<register-kv-caches>", "exec"
        ),
        namespace,
    )
    namespace["register_kv_caches"](
        SimpleNamespace(connector=inner, launch_config=config), kv_caches
    )
    if config.get("enable_kv_cache_check", False):
        factory.assert_called_once_with(kv_caches)
        assert isinstance(
            inner._rank_consistency.kv_cache_check, check_module.KVCacheCheck
        )
    else:
        factory.assert_not_called()
        assert inner._rank_consistency.kv_cache_check is None
