import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from .models import Vnpu

SIGTERM = getattr(signal, "SIGTERM", 15)
SIGKILL = getattr(signal, "SIGKILL", 9)
MAX_EVICTION_ROUNDS = 3


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class CommandRunner:
    def run(self, argv: Sequence[str], timeout: int) -> CommandResult:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)


class NpuOperationError(RuntimeError):
    pass


def _table_cells(line: str) -> List[str]:
    return [cell.strip() for cell in line[1:-1].split("|")]


def _normalized_row(cells: Sequence[str]) -> Tuple[str, ...]:
    return tuple(" ".join(cell.lower().split()) for cell in cells)


def _pipe_positions(line: str) -> Tuple[int, ...]:
    return tuple(index for index, character in enumerate(line) if character == "|")


def _separator_geometry(
    line: str, header_line: str
) -> Optional[Tuple[str, Tuple[int, ...]]]:
    if (
        len(line) != len(header_line)
        or len(line) < 5
        or not line.startswith("+")
        or not line.endswith("+")
    ):
        return None
    plus_positions = tuple(
        index for index, character in enumerate(line) if character == "+"
    )
    header_positions = _pipe_positions(header_line)
    if plus_positions not in ((0, len(line) - 1), header_positions):
        return None
    if any(right - left < 4 for left, right in zip(plus_positions, plus_positions[1:])):
        return None
    styles = set(line) - {"+"}
    if len(styles) != 1 or not styles.issubset({"-", "="}):
        return None
    return next(iter(styles)), plus_positions


def _selected_table(
    output: str,
    is_header: Callable[[Sequence[str]], bool],
    error_message: str,
    allow_empty: bool = False,
) -> Tuple[List[str], List[List[str]]]:
    lines = [line for line in output.splitlines() if line.strip()]
    header_position = None
    headers = None
    header_line = None
    for index, line in enumerate(lines):
        if line.startswith("|") and line.endswith("|"):
            cells = _table_cells(line)
            if is_header(cells):
                header_position = index
                headers = cells
                header_line = line
                break
    if header_position is None or headers is None or header_line is None:
        raise NpuOperationError(error_message)

    if header_position == 0 or header_position + 1 >= len(lines):
        raise NpuOperationError(error_message)
    geometry = _separator_geometry(lines[header_position - 1], header_line)
    if geometry is None or (
        _separator_geometry(lines[header_position + 1], header_line) != geometry
    ):
        raise NpuOperationError(error_message)

    rows = []
    normalized_headers = _normalized_row(headers)
    header_pipe_positions = _pipe_positions(header_line)
    position = header_position + 2
    while position < len(lines):
        line = lines[position]
        if (
            len(line) != len(header_line)
            or _pipe_positions(line) != header_pipe_positions
        ):
            raise NpuOperationError(error_message)
        cells = _table_cells(line)
        if len(cells) != len(headers) or _normalized_row(cells) == normalized_headers:
            raise NpuOperationError(error_message)
        rows.append(cells)
        position += 1
        if position >= len(lines) or (
            _separator_geometry(lines[position], header_line) != geometry
        ):
            raise NpuOperationError(error_message)
        position += 1
    if not rows and not allow_empty:
        raise NpuOperationError(error_message)
    return headers, rows


def _is_vnpu_header(cells: Sequence[str]) -> bool:
    return "vnpu id" in [cell.lower() for cell in cells]


def _vnpu_count(output: str) -> int:
    lines = output.splitlines()
    header_position = None
    header_line = None
    for index, line in enumerate(lines):
        if line.startswith("|") and line.endswith("|"):
            cells = _table_cells(line)
            if _is_vnpu_header(cells):
                header_position = index
                header_line = line
                break
    if header_position is None or header_line is None or header_position < 2:
        raise NpuOperationError("could not parse vNPU inventory")

    pattern = re.compile(r"\| Total number of vnpu: (0|[1-9][0-9]*) +\|")
    matches = []
    for line in lines:
        match = pattern.fullmatch(line)
        if match is not None:
            matches.append(match)
    bound_match = pattern.fullmatch(lines[header_position - 2])
    if (
        len(matches) != 1
        or bound_match is None
        or len(lines[header_position - 2]) != len(header_line)
        or len(lines[header_position - 1]) != len(header_line)
    ):
        raise NpuOperationError("could not parse vNPU inventory")
    return int(bound_match.group(1))


def _header_index(headers: Sequence[str], *names: str) -> int:
    normalized = [" ".join(header.lower().split()) for header in headers]
    for name in names:
        if name in normalized:
            return normalized.index(name)
    raise ValueError("missing table column: {}".format(" or ".join(names)))


def parse_info_vnpu(output: str) -> List[Vnpu]:
    expected_count = _vnpu_count(output)
    headers, rows = _selected_table(
        output,
        _is_vnpu_header,
        "could not parse vNPU inventory",
        allow_empty=True,
    )
    if len(rows) != expected_count:
        raise NpuOperationError("could not parse vNPU inventory")
    try:
        id_index = _header_index(headers, "vnpu id")
        template_index = _header_index(headers, "template name")
    except ValueError as error:
        raise NpuOperationError("could not parse vNPU inventory") from error
    vnpus = []
    for row in rows:
        if len(row) != len(headers):
            raise NpuOperationError("could not parse vNPU inventory row")
        try:
            vnpu_id = int(row[id_index])
        except ValueError as error:
            raise NpuOperationError("could not parse vNPU inventory row") from error
        if vnpu_id <= 0 or not row[template_index]:
            raise NpuOperationError("could not parse vNPU inventory row")
        vnpus.append(Vnpu(vnpu_id, row[template_index]))
    return vnpus


def parse_proc_mem(output: str, card_id: int) -> List[int]:
    if output.strip() == "No process in device.":
        return []
    headers, rows = _selected_table(
        output,
        lambda row: any(
            " ".join(cell.lower().split()) in ("process id", "pid") for cell in row
        ),
        "could not parse NPU process inventory",
    )
    try:
        card_index = _header_index(headers, "npu id", "card id")
        pid_index = _header_index(headers, "process id", "pid")
    except ValueError as error:
        raise NpuOperationError("could not parse NPU process inventory") from error
    pids = []
    for row in rows:
        if len(row) != len(headers):
            raise NpuOperationError("could not parse NPU process inventory row")
        try:
            row_card_id = int(row[card_index])
            pid = int(row[pid_index])
        except ValueError as error:
            raise NpuOperationError(
                "could not parse NPU process inventory row"
            ) from error
        if row_card_id < 0 or pid <= 0:
            raise NpuOperationError("could not parse NPU process inventory row")
        if row_card_id == card_id and pid not in pids:
            pids.append(pid)
    return pids


class NpuManager:
    def __init__(
        self,
        command_runner: CommandRunner,
        card_id: int,
        chip_id: int,
        eviction_grace_seconds: int = 30,
        kill: Callable[[int, int], None] = os.kill,
        sleep: Callable[[float], None] = time.sleep,
        stat_fn: Callable[[str], object] = os.stat,
        chmod_fn: Callable[[str, int], None] = os.chmod,
        command_timeout: int = 30,
        device_path: str = None,
    ):
        self.command_runner = command_runner
        self.card_id = card_id
        self.chip_id = chip_id
        self.eviction_grace_seconds = eviction_grace_seconds
        self.kill = kill
        self.sleep = sleep
        self.stat_fn = stat_fn
        self.chmod_fn = chmod_fn
        self.command_timeout = command_timeout
        self.device_path = device_path or "/dev/davinci{}".format(card_id)

    def _run(self, argv: Sequence[str]) -> CommandResult:
        try:
            return self.command_runner.run(argv, self.command_timeout)
        except (OSError, subprocess.SubprocessError) as error:
            raise NpuOperationError("command failed: {}".format(error)) from error

    def _info_argv(self, info_type: str) -> List[str]:
        return [
            "npu-smi",
            "info",
            "-t",
            info_type,
            "-i",
            str(self.card_id),
            "-c",
            str(self.chip_id),
        ]

    def _set_argv(self, operation: str) -> List[str]:
        return [
            "npu-smi",
            "set",
            "-t",
            operation,
            "-i",
            str(self.card_id),
            "-c",
            str(self.chip_id),
        ]

    def list_vnpus(self) -> List[Vnpu]:
        result = self._run(self._info_argv("info-vnpu"))
        self._require_success(result, "list vNPUs")
        return parse_info_vnpu(result.stdout)

    def list_card_pids(self) -> List[int]:
        result = self._run(self._info_argv("proc-mem"))
        self._require_success(result, "list NPU processes")
        return parse_proc_mem(result.stdout, self.card_id)

    def evict_card_pids(self) -> None:
        current = set(self.list_card_pids())
        for _ in range(MAX_EVICTION_ROUNDS):
            if not current:
                break
            terminated = current
            for pid in sorted(terminated):
                self._signal(pid, SIGTERM)
            self.sleep(self.eviction_grace_seconds)
            survivors = set(self.list_card_pids())
            for pid in sorted(terminated.intersection(survivors)):
                self._signal(pid, SIGKILL)
            current = set(self.list_card_pids())
        if current:
            raise NpuOperationError(
                "NPU processes remain because churn did not clear after {} eviction "
                "rounds".format(MAX_EVICTION_ROUNDS)
            )
        for _ in range(3):
            self.sleep(1)
            remaining = self.list_card_pids()
            if remaining:
                raise NpuOperationError(
                    "NPU processes remain after SIGKILL: {}".format(remaining)
                )

    def create_vnpus(
        self,
        count: int,
        template: str,
        on_created: Optional[Callable[[Sequence[Vnpu]], None]] = None,
    ) -> List[Vnpu]:
        known_ids = {vnpu.id for vnpu in self.list_vnpus()}
        created = []
        for _ in range(count):
            result = self._run(self._set_argv("create-vnpu") + ["-f", template])
            self._require_success(result, "create vNPU")
            inventory = self.list_vnpus()
            new_vnpus = [vnpu for vnpu in inventory if vnpu.id not in known_ids]
            if len(new_vnpus) != 1:
                raise NpuOperationError(
                    "create vNPU did not add exactly one inventory ID"
                )
            created.append(new_vnpus[0])
            known_ids = {vnpu.id for vnpu in inventory}
            if on_created is not None:
                on_created(tuple(created))
        return created

    def destroy_vnpus(self, vnpu_ids: Iterable[int]) -> None:
        errors = self._destroy_all(vnpu_ids)
        if errors:
            raise NpuOperationError("; ".join(errors))

    def _destroy_all(self, vnpu_ids: Iterable[int]) -> List[str]:
        ids = tuple(vnpu_ids)
        if not ids:
            return []
        errors = []
        for index, vnpu_id in enumerate(ids):
            remaining_ids = list(ids[index:])
            try:
                card_pids = self.list_card_pids()
            except NpuOperationError as error:
                errors.append(
                    "could not prove NPU card PID inventory empty before destroy: {}; "
                    "remaining vNPU IDs: {}".format(error, remaining_ids)
                )
                break
            if card_pids:
                errors.append(
                    "NPU card processes are active: {}; remaining vNPU IDs: {}".format(
                        card_pids, remaining_ids
                    )
                )
                break
            operation = "destroy vNPU {}".format(vnpu_id)
            try:
                result = self._run(
                    self._set_argv("destroy-vnpu") + ["-v", str(vnpu_id)]
                )
            except NpuOperationError as error:
                errors.append("{} failed: {}".format(operation, error))
                continue
            if result.returncode != 0:
                errors.append(self._error_message(result, operation))
        return errors

    def read_device_mode(self) -> int:
        try:
            return self.stat_fn(self.device_path).st_mode
        except OSError as error:
            raise NpuOperationError(
                "could not read physical NPU mode: {}".format(error)
            ) from error

    def restrict_device(self) -> None:
        try:
            self.chmod_fn(self.device_path, 0)
        except OSError as error:
            raise NpuOperationError(
                "could not restrict physical NPU: {}".format(error)
            ) from error

    def restore_device(self, original_mode: int) -> None:
        try:
            self.chmod_fn(self.device_path, original_mode)
        except OSError as error:
            raise NpuOperationError(
                "could not restore physical NPU: {}".format(error)
            ) from error

    def _signal(self, pid: int, sig: int) -> None:
        try:
            self.kill(pid, sig)
        except ProcessLookupError:
            return
        except OSError as error:
            raise NpuOperationError(
                "could not signal PID {}: {}".format(pid, error)
            ) from error

    @staticmethod
    def _error_message(result: CommandResult, operation: str) -> str:
        detail = result.stderr.strip() or result.stdout.strip()
        if detail:
            return "{} failed: {}".format(operation, detail)
        return "{} failed with exit code {}".format(operation, result.returncode)

    @classmethod
    def _require_success(cls, result: CommandResult, operation: str) -> None:
        if result.returncode != 0:
            raise NpuOperationError(cls._error_message(result, operation))
