import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

from .models import Vnpu

SIGTERM = getattr(signal, "SIGTERM", 15)
SIGKILL = getattr(signal, "SIGKILL", 9)


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


def _table_rows(output: str) -> List[List[str]]:
    rows = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            rows.append([cell.strip() for cell in stripped[1:-1].split("|")])
    return rows


def _header_index(headers: Sequence[str], *names: str) -> int:
    normalized = [" ".join(header.lower().split()) for header in headers]
    for name in names:
        if name in normalized:
            return normalized.index(name)
    raise ValueError("missing table column: {}".format(" or ".join(names)))


def parse_info_vnpu(output: str) -> List[Vnpu]:
    rows = _table_rows(output)
    header_position = next(
        (
            index
            for index, row in enumerate(rows)
            if "vnpu id" in [cell.lower() for cell in row]
        ),
        None,
    )
    if header_position is None:
        if not output.strip():
            return []
        raise NpuOperationError("could not parse vNPU inventory")
    headers = rows[header_position]
    try:
        id_index = _header_index(headers, "vnpu id")
        template_index = _header_index(headers, "template name")
    except ValueError as error:
        raise NpuOperationError("could not parse vNPU inventory") from error
    vnpus = []
    for row in rows[header_position + 1 :]:
        if len(row) <= max(id_index, template_index):
            continue
        try:
            vnpu_id = int(row[id_index])
        except ValueError:
            continue
        if vnpu_id >= 0 and row[template_index]:
            vnpus.append(Vnpu(vnpu_id, row[template_index]))
    return vnpus


def parse_proc_mem(output: str, card_id: int) -> List[int]:
    if "No process in device." in output:
        return []
    rows = _table_rows(output)
    header_position = next(
        (
            index
            for index, row in enumerate(rows)
            if any(
                " ".join(cell.lower().split()) in ("process id", "pid") for cell in row
            )
        ),
        None,
    )
    if header_position is None:
        raise NpuOperationError("could not parse NPU process inventory")
    headers = rows[header_position]
    try:
        card_index = _header_index(headers, "npu id", "card id")
        pid_index = _header_index(headers, "process id", "pid")
    except ValueError as error:
        raise NpuOperationError("could not parse NPU process inventory") from error
    pids = []
    for row in rows[header_position + 1 :]:
        if len(row) <= max(card_index, pid_index):
            continue
        try:
            row_card_id = int(row[card_index])
            pid = int(row[pid_index])
        except ValueError:
            continue
        if row_card_id == card_id and pid > 0 and pid not in pids:
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
        for pid in self.list_card_pids():
            self._signal(pid, SIGTERM)
        self.sleep(self.eviction_grace_seconds)
        for pid in self.list_card_pids():
            self._signal(pid, SIGKILL)
        for _ in range(3):
            self.sleep(1)
            remaining = self.list_card_pids()
            if remaining:
                raise NpuOperationError(
                    "NPU processes remain after SIGKILL: {}".format(remaining)
                )

    def create_vnpus(self, count: int, template: str) -> List[Vnpu]:
        known_ids = {vnpu.id for vnpu in self.list_vnpus()}
        created = []
        try:
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
        except NpuOperationError as error:
            rollback_errors = self._destroy_all(vnpu.id for vnpu in created)
            if rollback_errors:
                raise NpuOperationError(
                    "{}; rollback failed: {}".format(error, "; ".join(rollback_errors))
                ) from error
            raise
        return created

    def destroy_vnpus(self, vnpu_ids: Iterable[int]) -> None:
        errors = self._destroy_all(vnpu_ids)
        if errors:
            raise NpuOperationError("; ".join(errors))

    def _destroy_all(self, vnpu_ids: Iterable[int]) -> List[str]:
        errors = []
        for vnpu_id in vnpu_ids:
            result = self._run(self._set_argv("destroy-vnpu") + ["-v", str(vnpu_id)])
            if result.returncode != 0:
                errors.append(
                    self._error_message(result, "destroy vNPU {}".format(vnpu_id))
                )
        return errors

    def restrict_device(self) -> int:
        try:
            original_mode = self.stat_fn(self.device_path).st_mode
            self.chmod_fn(self.device_path, 0)
        except OSError as error:
            raise NpuOperationError(
                "could not restrict physical NPU: {}".format(error)
            ) from error
        return original_mode

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
