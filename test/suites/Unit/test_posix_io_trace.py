#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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

import importlib.util
import io
import struct
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[3] / "ucm/store/posix/trace.py"
SPEC = importlib.util.spec_from_file_location("posix_io_trace", MODULE_PATH)
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)


def make_trace(*, endian="<", dropped=0, final=True, tail=b""):
    backend = b"/cache/data0/"
    header = struct.pack(
        endian + TRACE.HEADER_FORMAT,
        b"UCMIO001",
        0x01020304,
        96,
        1_700_000_000_000_000_000,
        1000,
        2,
        42,
        3,
        1,
        0,
    )
    metadata = struct.pack(endian + "I", len(backend)) + backend
    event = struct.pack(
        endian + TRACE.RECORD_FORMAT,
        1100,
        2100,
        123,
        9,
        2,
        8192,
        4096,
        1024,
        bytes.fromhex("abcdef0123456789abcdef0123456789"),
        0,
        0,
        18,
        1,
        2,
    )
    summary = struct.pack(
        endian + TRACE.RECORD_FORMAT,
        2200,
        2200,
        1 + dropped,
        1,
        dropped,
        1,
        0,
        0,
        bytes(16),
        0,
        0,
        0,
        9,
        32768,
    )
    return header + metadata + event + (summary if final else b"") + tail


@pytest.mark.parametrize("endian", ["<", ">"])
def test_decode_fields_and_complete(endian):
    reader = TRACE.TraceReader(io.BytesIO(make_trace(endian=endian)))
    (event,) = list(reader)
    assert event["file"] == "/cache/data0/abc/abcdef0123456789abcdef0123456789"
    assert event["task_id"] == 123
    assert event["shard_index"] == 2
    assert event["offset"] == 8192
    assert event["actual_bytes"] == 1024
    assert event["requested_bytes"] == 4096
    assert event["status"] == "SHORT_IO"
    assert event["async_io"] is True
    assert event["duration_us"] == 1.0
    assert event["start_time"] == "2023-11-14T22:13:20.000000100Z"
    assert reader.summary["complete"] is True


@pytest.mark.parametrize(
    "options", [{"dropped": 1}, {"final": False}, {"tail": b"bad"}]
)
def test_incomplete_evidence(options):
    reader = TRACE.TraceReader(io.BytesIO(make_trace(**options)))
    assert len(list(reader)) == 1
    assert reader.summary["complete"] is False


@pytest.mark.parametrize("length", [0, 12, 64, 68, 72])
def test_truncated_metadata(length):
    with pytest.raises(ValueError):
        TRACE.TraceReader(io.BytesIO(make_trace()[:length]))


def test_incorrect_summary_counts():
    raw = bytearray(make_trace())
    struct.pack_into("<Q", raw, len(raw) - 96 + 24, 99)
    reader = TRACE.TraceReader(io.BytesIO(raw))
    list(reader)
    assert reader.summary["complete"] is False
