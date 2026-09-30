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

"""Decode Posix binary IO traces without importing UCM or device libraries."""

import argparse
import csv
import datetime
import errno
import json
import struct
import sys
from pathlib import Path
from typing import BinaryIO

OPS = (
    "UNKNOWN",
    "READ",
    "WRITE",
    "OPEN",
    "CLOSE",
    "RENAME",
    "REMOVE",
    "SYNC",
    "TIMEOUT",
    "STATS",
)
HEADER_FORMAT = "8sIIQQQQIIQ"
RECORD_FORMAT = "7Qq16sIiIHH"
UNKNOWN_SHARD = (1 << 64) - 1


class TraceReader:
    """Stream records and retain completeness information, including truncated tails."""

    def __init__(self, stream: BinaryIO):
        self.stream = stream
        raw = stream.read(64)
        if len(raw) != 64 or raw[:8] != b"UCMIO001":
            raise ValueError("invalid or truncated Posix IO trace header")
        marker = raw[8:12]
        if marker not in (b"\x04\x03\x02\x01", b"\x01\x02\x03\x04"):
            raise ValueError("unsupported byte order")
        self.endian = "<" if marker[0] == 4 else ">"
        header = struct.unpack(self.endian + HEADER_FORMAT, raw)
        (
            _,
            _,
            record_size,
            self.realtime_ns,
            self.monotonic_ns,
            self.session_id,
            self.pid,
            self.shard_chars,
            count,
            _,
        ) = header
        self.record = struct.Struct(self.endian + RECORD_FORMAT)
        if record_size != self.record.size or self.shard_chars > 5 or count > 65536:
            raise ValueError("unsupported record size or invalid layout")
        self.backends = []
        for _ in range(count):
            raw_length = stream.read(4)
            if len(raw_length) != 4:
                raise ValueError("truncated backend metadata")
            length = struct.unpack(self.endian + "I", raw_length)[0]
            if length > 1024 * 1024:
                raise ValueError("invalid backend path length")
            path = stream.read(length)
            if len(path) != length:
                raise ValueError("truncated backend path")
            self.backends.append(path.decode("utf-8", errors="surrogateescape"))
        self.run_id = f"{self.pid}-{self.realtime_ns}-{self.session_id}"
        self.summary = {
            "run_id": self.run_id,
            "final": False,
            "truncated": False,
            "records": 0,
        }

    def __iter__(self):
        for raw in iter(lambda: self.stream.read(self.record.size), b""):
            if len(raw) != self.record.size:
                self.summary["truncated"] = True
                break
            (
                start,
                end,
                task,
                io_id,
                shard,
                offset,
                requested,
                result,
                block,
                backend,
                error,
                tid,
                op,
                flags,
            ) = self.record.unpack(raw)
            if op == 9:
                self.summary.update(
                    final=bool(flags & 32768),
                    produced=task,
                    written=io_id,
                    dropped=shard,
                    queue_high_water=offset,
                    write_errors=requested,
                )
                continue
            if self.summary["final"]:
                raise ValueError("records after final summary")
            if not 0 < op < len(OPS) or backend >= len(self.backends) or end < start:
                raise ValueError("invalid event record")
            self.summary["records"] += 1
            block_id = block.hex()
            shard_dir = block_id[: self.shard_chars] if self.shard_chars else "data"
            file = f"{self.backends[backend].rstrip('/')}/{shard_dir}/{block_id}"
            wall_ns = self.realtime_ns + start - self.monotonic_ns
            seconds, nanos = divmod(wall_ns, 1_000_000_000)
            timestamp = datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc)
            short_io = op in (1, 2) and error == 0 and result != requested
            event = dict(
                run_id=self.run_id,
                pid=self.pid,
                store_id=self.session_id,
                task_id=task or None,
                io_id=io_id,
                tid=tid,
                op=OPS[op],
                block_id=block_id,
                shard_index=None if shard == UNKNOWN_SHARD else shard,
                backend_index=backend,
                file=file + (".tmp" if flags & 1 else ""),
                rename_to=file if op == 5 else None,
                start_time=timestamp.strftime("%Y-%m-%dT%H:%M:%S") + f".{nanos:09d}Z",
                start_ns=start,
                end_ns=end,
                duration_us=(end - start) / 1000,
                offset=offset,
                requested_bytes=requested,
                actual_bytes=result if op in (1, 2) else None,
                result=result,
                errno=error,
                error_name=errno.errorcode.get(error, "") if error else "",
                status="ERROR" if error else "SHORT_IO" if short_io else "OK",
                async_io=bool(flags & 2),
                submit_failed=bool(flags & 4),
                not_started=bool(flags & 8),
                source="health" if flags & 16 else "gc" if flags & 32 else "transfer",
            )
            if op == 8 and block == bytes(16):
                event["file"] = None
                event["block_id"] = None
            yield event
        self.summary["complete"] = (
            self.summary["final"]
            and not self.summary["truncated"]
            and self.summary.get("dropped") == 0
            and self.summary.get("write_errors") == 0
            and self.summary.get("produced")
            == self.summary.get("written")
            == self.summary["records"]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--block-id")
    parser.add_argument("--file", help="Substring of the full file name")
    args = parser.parse_args()
    writer = None
    complete = True
    for path in args.traces:
        try:
            with path.open("rb") as stream:
                trace = TraceReader(stream)
                for event in trace:
                    if args.task_id is not None and event["task_id"] != args.task_id:
                        continue
                    if args.block_id and event["block_id"] != args.block_id.lower():
                        continue
                    if args.file and args.file not in (event["file"] or ""):
                        continue
                    if args.format == "csv":
                        if writer is None:
                            writer = csv.DictWriter(sys.stdout, fieldnames=event.keys())
                            writer.writeheader()
                        writer.writerow(event)
                    else:
                        print(json.dumps(event, ensure_ascii=True))
                print(
                    json.dumps({"trace": str(path), **trace.summary}), file=sys.stderr
                )
                complete = complete and trace.summary["complete"]
        except (OSError, ValueError) as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            complete = False
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
