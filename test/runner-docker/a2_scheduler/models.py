import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, FrozenSet, Optional, Tuple

_CREDENTIAL_PATTERN = re.compile(
    r"\b(token|authorization)\b(?:\s*[:=]\s*|\s+)(?:bearer\s+)?[^\s,;]+",
    re.IGNORECASE,
)
_GITHUB_TOKEN_PATTERN = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,})\b"
)
_REPOSITORY = "ModelEngine-Group/unified-cache-management"
_REQUIRED_LABELS = frozenset(("self-hosted", "npu", "a2"))


class Phase(str, Enum):
    WHOLE = "WHOLE"
    RESERVING = "RESERVING"
    SPLITTING = "SPLITTING"
    SERVING = "SERVING"
    DRAINING = "DRAINING"
    RESTORING = "RESTORING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Job:
    id: int
    status: str
    labels: FrozenSet[str] = field(default_factory=frozenset)
    run_id: Optional[int] = None


@dataclass(frozen=True)
class Vnpu:
    id: int
    template: str


@dataclass(frozen=True)
class RunnerSlot:
    name: str
    image: str
    work_dir: Path


@dataclass(frozen=True)
class SchedulerConfig:
    repository: str
    required_labels: FrozenSet[str]
    card_id: int
    chip_id: int
    window_start_minute: int
    window_end_minute: int
    eviction_grace_seconds: int
    vnpu_count: int
    vnpu_template: str
    token_file: Path
    state_file: Path
    lock_file: Path
    runner_slots: Tuple[RunnerSlot, ...]
    environment: Tuple[Tuple[str, str], ...]
    devices: Tuple[str, ...]
    mounts: Tuple[str, ...]


@dataclass(frozen=True)
class SchedulerState:
    phase: Phase = Phase.WHOLE
    vnpu_ids: Tuple[int, ...] = ()
    original_device_mode: Optional[int] = None
    transition_timestamp: Optional[str] = None
    last_error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "phase": self.phase.value,
                "vnpu_ids": list(self.vnpu_ids),
                "original_device_mode": self.original_device_mode,
                "transition_timestamp": self.transition_timestamp,
                "last_error": _redact_credentials(self.last_error),
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, document: str) -> "SchedulerState":
        data = _load_json(document, "state")
        _require_keys(
            data,
            {
                "phase",
                "vnpu_ids",
                "original_device_mode",
                "transition_timestamp",
                "last_error",
            },
            "state",
        )
        phase = _phase(data["phase"])
        vnpu_ids = _integer_tuple(data["vnpu_ids"], "state.vnpu_ids", minimum=0)
        original_device_mode = _optional_integer(
            data["original_device_mode"], "state.original_device_mode", minimum=1
        )
        if phase != Phase.WHOLE and original_device_mode is None:
            raise ValueError(
                "state phase {} requires original_device_mode".format(phase.value)
            )
        transition_timestamp = _optional_string(
            data["transition_timestamp"], "state.transition_timestamp"
        )
        last_error = _optional_string(data["last_error"], "state.last_error")
        return cls(
            phase=phase,
            vnpu_ids=vnpu_ids,
            original_device_mode=original_device_mode,
            transition_timestamp=transition_timestamp,
            last_error=last_error,
        )


def load_config(path: Path) -> SchedulerConfig:
    try:
        document = Path(path).read_text()
    except OSError as error:
        raise ValueError("could not read configuration") from error
    data = _load_json(document, "configuration")
    _require_keys(
        data,
        {
            "repository",
            "required_labels",
            "card_id",
            "chip_id",
            "window_start_minute",
            "window_end_minute",
            "eviction_grace_seconds",
            "vnpu_count",
            "vnpu_template",
            "token_file",
            "state_file",
            "lock_file",
            "runner_slots",
            "environment",
            "devices",
            "mounts",
        },
        "configuration",
    )
    repository = _string(data["repository"], "repository")
    required_labels = frozenset(
        _string_list(data["required_labels"], "required_labels")
    )
    if not required_labels:
        raise ValueError("required_labels must not be empty")
    card_id = _integer(data["card_id"], "card_id", minimum=0)
    chip_id = _integer(data["chip_id"], "chip_id", minimum=0)
    window_start_minute = _minute(data["window_start_minute"], "window_start_minute")
    window_end_minute = _minute(data["window_end_minute"], "window_end_minute")
    eviction_grace_seconds = _integer(
        data["eviction_grace_seconds"], "eviction_grace_seconds", minimum=0
    )
    vnpu_count = _integer(data["vnpu_count"], "vnpu_count", minimum=1)
    vnpu_template = _string(data["vnpu_template"], "vnpu_template")
    _validate_pinned_bindings(
        repository,
        required_labels,
        card_id,
        chip_id,
        window_start_minute,
        window_end_minute,
        eviction_grace_seconds,
        vnpu_count,
        vnpu_template,
    )
    runner_slots = _runner_slots(data["runner_slots"])
    if len(runner_slots) != vnpu_count:
        raise ValueError("runner slot count must match vnpu_count")
    return SchedulerConfig(
        repository=repository,
        required_labels=required_labels,
        card_id=card_id,
        chip_id=chip_id,
        window_start_minute=window_start_minute,
        window_end_minute=window_end_minute,
        eviction_grace_seconds=eviction_grace_seconds,
        vnpu_count=vnpu_count,
        vnpu_template=vnpu_template,
        token_file=_absolute_path(data["token_file"], "token_file"),
        state_file=_absolute_path(data["state_file"], "state_file"),
        lock_file=_absolute_path(data["lock_file"], "lock_file"),
        runner_slots=runner_slots,
        environment=_environment(data["environment"]),
        devices=tuple(_string_list(data["devices"], "devices")),
        mounts=tuple(_string_list(data["mounts"], "mounts")),
    )


def _load_json(document: str, name: str) -> Any:
    try:
        data = json.loads(document)
    except json.JSONDecodeError as error:
        raise ValueError("{} must be valid JSON".format(name)) from error
    if not isinstance(data, dict):
        raise ValueError("{} must be a JSON object".format(name))
    return data


def _require_keys(data: dict, expected: set, name: str) -> None:
    missing = expected.difference(data)
    unexpected = set(data).difference(expected)
    if missing or unexpected:
        raise ValueError(
            "{} has invalid keys: missing={}, unexpected={}".format(
                name, sorted(missing), sorted(unexpected)
            )
        )


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(name))
    return value


def _optional_string(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _string(value, name)


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return value


def _optional_integer(value: Any, name: str, minimum: int) -> Optional[int]:
    if value is None:
        return None
    return _integer(value, name, minimum)


def _minute(value: Any, name: str) -> int:
    minute = _integer(value, name, minimum=0)
    if minute >= 60:
        raise ValueError("{} must be less than 60".format(name))
    return minute


def _string_list(value: Any, name: str) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("{} must be a list".format(name))
    values = tuple(_string(item, name) for item in value)
    if len(values) != len(set(values)):
        raise ValueError("{} must not contain duplicates".format(name))
    return values


def _integer_tuple(value: Any, name: str, minimum: int) -> Tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError("{} must be a list".format(name))
    values = tuple(_integer(item, name, minimum) for item in value)
    if len(values) != len(set(values)):
        raise ValueError("{} must not contain duplicates".format(name))
    return values


def _absolute_path(value: Any, name: str) -> Path:
    path_text = _string(value, name)
    if not PurePosixPath(path_text).is_absolute():
        raise ValueError("{} must be an absolute path".format(name))
    return Path(path_text)


def _runner_slots(value: Any) -> Tuple[RunnerSlot, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("runner_slots must be a non-empty list")
    slots = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError("runner_slots[{}] must be an object".format(index))
        _require_keys(item, {"name", "image", "work_dir"}, "runner slot")
        slots.append(
            RunnerSlot(
                name=_string(item["name"], "runner slot name"),
                image=_string(item["image"], "runner slot image"),
                work_dir=_absolute_path(item["work_dir"], "runner slot work_dir"),
            )
        )
    names = [slot.name for slot in slots]
    if len(names) != len(set(names)):
        raise ValueError("runner names must be unique")
    return tuple(slots)


def _environment(value: Any) -> Tuple[Tuple[str, str], ...]:
    if not isinstance(value, dict):
        raise ValueError("environment must be an object")
    environment = []
    for key, item in value.items():
        key = _string(key, "environment key")
        if "token" in key.lower() or "authorization" in key.lower():
            raise ValueError("environment must not contain credentials")
        item = _string(item, "environment value")
        if _GITHUB_TOKEN_PATTERN.search(item):
            raise ValueError("environment must not contain GitHub token-shaped values")
        environment.append((key, item))
    return tuple(sorted(environment))


def _phase(value: Any) -> Phase:
    try:
        return Phase(_string(value, "state.phase"))
    except ValueError as error:
        raise ValueError("state.phase must be a valid phase") from error


def _redact_credentials(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = _CREDENTIAL_PATTERN.sub(r"\1 [REDACTED]", value)
    return _GITHUB_TOKEN_PATTERN.sub("[REDACTED]", value)


def _validate_pinned_bindings(
    repository: str,
    required_labels: FrozenSet[str],
    card_id: int,
    chip_id: int,
    window_start_minute: int,
    window_end_minute: int,
    eviction_grace_seconds: int,
    vnpu_count: int,
    vnpu_template: str,
) -> None:
    if repository != _REPOSITORY:
        raise ValueError("repository must be {}".format(_REPOSITORY))
    if required_labels != _REQUIRED_LABELS:
        raise ValueError("required_labels must be exactly self-hosted,npu,a2")
    if card_id != 7:
        raise ValueError("card_id must be 7")
    if chip_id != 0:
        raise ValueError("chip_id must be 0")
    if window_start_minute != 0:
        raise ValueError("window_start_minute must be 0")
    if window_end_minute != 20:
        raise ValueError("window_end_minute must be 20")
    if eviction_grace_seconds != 30:
        raise ValueError("eviction_grace_seconds must be 30")
    if vnpu_count != 4:
        raise ValueError("vnpu_count must be 4")
    if vnpu_template != "vir05_1c_16g":
        raise ValueError("vnpu_template must be vir05_1c_16g")
