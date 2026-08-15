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

import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

from ucm.store.allgather import connector
from ucm.store.allgather.connector import AllGatherTask, UcmAllGatherStore, _LoadWindow


def _make_store():
    store = object.__new__(UcmAllGatherStore)
    store._load_condition = threading.Condition()
    return store


def test_load_check_is_nonblocking_while_task_is_pending():
    store = _make_store()
    task = AllGatherTask(operation="load")
    store._wait_queued_load = Mock(side_effect=AssertionError)

    assert store.check(task) is False
    store._wait_queued_load.assert_not_called()


def test_load_wait_uses_progress_completion_condition():
    store = _make_store()
    task = AllGatherTask(operation="load")
    store._load_tasks = [task]

    def complete():
        with store._load_condition:
            task.completed = True
            store._load_tasks.clear()
            store._load_condition.notify_all()

    timer = threading.Timer(0.01, complete)
    timer.start()
    try:
        store.wait(task)
    finally:
        timer.join()

    assert task.completed is True


def test_wait_load_records_collective_and_scatter_device_time(monkeypatch):
    event_times = iter([0.0, 4.0, 5.5])

    class FakeEvent:
        def __init__(self, enable_timing=False):
            assert enable_timing is True
            self.timestamp = None

        def record(self, stream):
            self.timestamp = next(event_times)

        def elapsed_time(self, other):
            return other.timestamp - self.timestamp

    stream = SimpleNamespace(synchronize=Mock())
    monkeypatch.setattr(
        connector.torch,
        "npu",
        SimpleNamespace(Event=FakeEvent, current_stream=lambda: stream),
        raising=False,
    )
    monkeypatch.setattr(connector.dist, "all_gather_into_tensor", Mock())
    update_stats = Mock()
    monkeypatch.setattr(connector.ucmmetrics, "update_stats", update_stats)

    store = _make_store()
    store._collective_enabled = True
    store._world_size = 2
    store._shard_size = 16
    store._tp_group = object()
    store._inner = SimpleNamespace(wait=Mock())
    store._pool = SimpleNamespace(complete_load=Mock(), release_load=Mock())
    store._compact_scatter = Mock()
    store._prime_queued_loads = Mock()
    plan = Mock(collective_blocks=4)
    slot = SimpleNamespace(send_buffer=Mock(), receive_buffer=Mock())
    window = _LoadWindow(plan=plan, owned_shard_indices=[], slot=slot)
    task = AllGatherTask(
        operation="load",
        active_load_windows=deque([window]),
    )

    store._wait_load(task)

    metrics = update_stats.call_args.args[0]
    assert metrics["allgather_load_collective_device_ms"] == 4.0
    assert metrics["allgather_load_scatter_device_ms"] == 1.5
    assert metrics["allgather_load_windows"] == 1.0
    slot.send_buffer.narrow.assert_called_once_with(0, 0, 4 * store._shard_size)
    slot.receive_buffer.narrow.assert_called_once_with(
        0, 0, 4 * store._shard_size * store._world_size
    )


def test_load_plan_crops_each_collective_to_occupied_prefix():
    store = _make_store()
    store._storage_world_size = 4
    store._storage_rank = 0
    store._shard_size = 16
    store._collective_count_crop = True
    store._pool = SimpleNamespace(
        window_blocks=4,
        load_slots=[
            SimpleNamespace(send_buffer=SimpleNamespace(data_ptr=lambda: 1000)),
            SimpleNamespace(send_buffer=SimpleNamespace(data_ptr=lambda: 2000)),
        ],
    )
    block_ids = [
        owner.to_bytes(8, "little") + index.to_bytes(8, "little")
        for owner, count in enumerate((4, 6, 9, 2))
        for index in range(count)
    ]

    plan = store._build_load_plan(block_ids)

    assert [window.collective_blocks for window in plan.windows] == [4, 4, 4]

    sparse = store._build_load_plan(
        [
            owner.to_bytes(8, "little") + owner.to_bytes(8, "little")
            for owner in (0, 1)
        ]
    )
    assert [window.collective_blocks for window in sparse.windows] == [1]

    dense_tail = store._build_load_plan(
        [
            (0).to_bytes(8, "little") + index.to_bytes(8, "little")
            for index in range(7)
        ]
    )
    assert [window.collective_blocks for window in dense_tail.windows] == [4, 4]

    store._collective_count_crop = False
    uncropped = store._build_load_plan(block_ids)
    assert [window.collective_blocks for window in uncropped.windows] == [4, 4, 4]
