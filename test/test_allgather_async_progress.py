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
from unittest.mock import Mock

from ucm.store.allgather.connector import AllGatherTask, UcmAllGatherStore


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
