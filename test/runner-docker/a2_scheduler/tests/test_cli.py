import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from a2_scheduler.cli import (
    AtomicStateStore,
    CliDependencies,
    LinuxFileLock,
    SchedulerCli,
)
from a2_scheduler.host import CommandResult
from a2_scheduler.models import Job, Phase, SchedulerState, Vnpu, load_config

TOKEN = "TOKEN_SENTINEL"


class FakeLock:
    def __init__(self, acquired=True):
        self.acquired = acquired
        self.acquire_calls = []
        self.release_calls = 0

    def acquire(self, blocking=False):
        self.acquire_calls.append(blocking)
        return self.acquired

    def release(self):
        self.release_calls += 1


class FakeCommandRunner:
    def __init__(self):
        self.commands = []
        self.timer_active = False
        self.running_runners = set()

    def run(self, argv, timeout):
        command = tuple(argv)
        self.commands.append(command)
        if command[:2] == ("systemctl", "is-active"):
            if self.timer_active:
                return CommandResult(0, "active\n", "")
            return CommandResult(3, "inactive\n", "")
        if command[:2] == ("docker", "ps"):
            name_filter = next(item for item in command if item.startswith("name=^/"))
            name = name_filter[len("name=^/") : -1]
            output = "{}\n".format(name) if name in self.running_runners else ""
            return CommandResult(0, output, "")
        if command[:2] == ("docker", "top"):
            return CommandResult(
                0,
                "PID COMMAND\n100 /actions-runner/bin/Runner.Listener run\n",
                "",
            )
        raise AssertionError("unexpected command: {!r}".format(command))


class FakeGitHub:
    def __init__(self):
        self.job_snapshots = [[]]
        self.calls = 0

    def list_target_jobs(self):
        index = min(self.calls, len(self.job_snapshots) - 1)
        self.calls += 1
        snapshot = self.job_snapshots[index]
        if isinstance(snapshot, Exception):
            raise snapshot
        return list(snapshot)


class FakeNpu:
    def __init__(self):
        self.inventory = []
        self.card_pids = []
        self.created_ids = []
        self.destroyed_ids = []
        self.mutations = []

    def list_vnpus(self):
        return list(self.inventory)

    def list_card_pids(self):
        return list(self.card_pids)

    def create_vnpus(self, count, template, on_created=None):
        created = []
        for index in range(count):
            vnpu = Vnpu(300 + index, template)
            self.created_ids.append(vnpu.id)
            self.inventory.append(vnpu)
            self.mutations.append("create")
            created.append(vnpu)
            if on_created is not None:
                on_created(tuple(created))
        return created

    def destroy_vnpus(self, vnpu_ids):
        ids = list(vnpu_ids)
        self.destroyed_ids.extend(ids)
        self.mutations.extend("destroy" for _ in ids)
        self.inventory = [vnpu for vnpu in self.inventory if vnpu.id not in ids]


class FakeRunners:
    def __init__(self):
        self.mutations = []


class CheckpointingReconciler:
    def __init__(self, state, checkpoint):
        self.state = state
        self.checkpoint = checkpoint

    def reconcile(self, now):
        state = SchedulerState(
            phase=Phase.ERROR,
            original_device_mode=0o20660,
            transition_timestamp=now.isoformat(),
            last_error="{} {}".format(TOKEN, self.state.last_error or ""),
        )
        self.checkpoint(state)
        return state


class ConfiguredCli:
    def __init__(self, tmp_path):
        self.config_path = tmp_path / "config.json"
        self.token_path = tmp_path / "private" / "github.token"
        self.state_path = tmp_path / "state.json"
        self.lock_path = tmp_path / "scheduler.lock"
        self.token_path.parent.mkdir()
        self.token_path.write_text(TOKEN, encoding="utf-8")
        self.state_path.write_text(SchedulerState().to_json(), encoding="utf-8")

        example_path = Path(__file__).resolve().parents[1] / "config.example.json"
        config = json.loads(example_path.read_text(encoding="utf-8"))
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        parsed_config = replace(
            load_config(example_path),
            token_file=self.token_path,
            state_file=self.state_path,
            lock_file=self.lock_path,
        )

        self.command_runner = FakeCommandRunner()
        self.github = FakeGitHub()
        self.npu = FakeNpu()
        self.runners = FakeRunners()
        self.lock = FakeLock()
        self.reconciler_factory = lambda *args: CheckpointingReconciler(
            args[1], args[5]
        )
        dependencies = CliDependencies(
            command_runner_factory=lambda: self.command_runner,
            github_factory=lambda config, token: self.github,
            npu_factory=lambda config, runner: self.npu,
            runner_factory=lambda config, runner: self.runners,
            reconciler_factory=lambda *args: self.reconciler_factory(*args),
            lock_factory=lambda path: self.lock,
            config_loader=lambda path: parsed_config,
            clock=lambda: datetime(2026, 7, 11, 10, 8, 0),
        )
        self.cli = SchedulerCli(dependencies)

    def run(self, argv):
        return self.cli.run(argv)


@pytest.fixture
def configured_cli(tmp_path):
    return ConfiguredCli(tmp_path)


def test_status_redacts_token_path_contents_and_does_not_mutate(capsys, configured_cli):
    secret_error = "{} {}".format(TOKEN, configured_cli.token_path)
    configured_cli.state_path.write_text(
        json.dumps(
            {
                "phase": "WHOLE",
                "vnpu_ids": [],
                "original_device_mode": None,
                "transition_timestamp": None,
                "last_error": secret_error,
            }
        ),
        encoding="utf-8",
    )
    before = configured_cli.state_path.read_bytes()

    assert (
        configured_cli.run(["status", "--config", str(configured_cli.config_path)]) == 0
    )

    output = capsys.readouterr().out
    assert TOKEN not in output
    assert str(configured_cli.token_path) not in output
    assert '"phase": "WHOLE"' in output
    assert configured_cli.state_path.read_bytes() == before
    assert configured_cli.npu.mutations == []
    assert configured_cli.runners.mutations == []
    assert configured_cli.lock.acquire_calls == []


def test_dry_run_is_read_only(capsys, configured_cli):
    before = configured_cli.state_path.read_bytes()

    assert (
        configured_cli.run(["dry-run", "--config", str(configured_cli.config_path)])
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "dry-run"
    assert output["mutation"] is False
    assert configured_cli.state_path.read_bytes() == before
    assert configured_cli.npu.mutations == []
    assert configured_cli.runners.mutations == []
    assert configured_cli.lock.acquire_calls == []


def test_second_reconcile_cannot_take_lock(configured_cli):
    configured_cli.lock.acquired = False

    assert (
        configured_cli.run(["reconcile", "--config", str(configured_cli.config_path)])
        == 75
    )

    assert configured_cli.lock.acquire_calls == [False]
    assert configured_cli.lock.release_calls == 0
    assert configured_cli.npu.mutations == []


def test_reconcile_wires_atomic_redacted_checkpoint(
    monkeypatch, capsys, configured_cli
):
    replacements = []
    real_replace = __import__("os").replace

    def record_replace(source, destination):
        replacements.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr("a2_scheduler.cli.os.replace", record_replace)

    assert (
        configured_cli.run(["reconcile", "--config", str(configured_cli.config_path)])
        == 1
    )

    persisted = SchedulerState.from_json(
        configured_cli.state_path.read_text(encoding="utf-8")
    )
    output = capsys.readouterr().out
    assert persisted.phase == Phase.ERROR
    assert TOKEN not in (persisted.last_error or "")
    assert TOKEN not in output
    assert replacements and Path(replacements[0][1]) == configured_cli.state_path
    assert configured_cli.lock.release_calls == 1


def test_status_error_redacts_token_and_token_path(capsys, configured_cli):
    configured_cli.github.job_snapshots = [
        RuntimeError("{} {}".format(TOKEN, configured_cli.token_path))
    ]

    assert (
        configured_cli.run(["status", "--config", str(configured_cli.config_path)]) == 1
    )

    output = capsys.readouterr().err
    assert TOKEN not in output
    assert str(configured_cli.token_path) not in output


def test_linux_file_lock_loads_fcntl_only_when_acquired(tmp_path):
    calls = []
    fake_fcntl = SimpleNamespace(LOCK_EX=2, LOCK_NB=4, LOCK_UN=8)
    fake_fcntl.flock = lambda handle, flags: calls.append((handle, flags))
    loads = []
    lock = LinuxFileLock(
        tmp_path / "scheduler.lock",
        fcntl_loader=lambda: loads.append(True) or fake_fcntl,
    )

    assert loads == []
    assert lock.acquire(blocking=False) is True
    assert loads == [True]
    assert calls[0][1] == fake_fcntl.LOCK_EX | fake_fcntl.LOCK_NB
    lock.release()
    assert calls[-1][1] == fake_fcntl.LOCK_UN


def test_linux_file_lock_closes_handle_when_fcntl_load_fails(tmp_path, monkeypatch):
    path = tmp_path / "scheduler.lock"
    opened = []
    real_open = Path.open

    def tracking_open(target, *args, **kwargs):
        handle = real_open(target, *args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", tracking_open)
    lock = LinuxFileLock(
        path,
        fcntl_loader=lambda: (_ for _ in ()).throw(RuntimeError("no fcntl")),
    )

    with pytest.raises(RuntimeError, match="no fcntl"):
        lock.acquire(blocking=False)

    assert opened[0].closed


def test_atomic_state_store_replaces_complete_document(tmp_path):
    path = tmp_path / "state.json"
    store = AtomicStateStore(path)
    store.save(SchedulerState(phase=Phase.WHOLE))

    assert (
        SchedulerState.from_json(path.read_text(encoding="utf-8")).phase == Phase.WHOLE
    )
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_systemd_service_matches_plan():
    path = Path(__file__).resolve().parents[1] / "systemd" / "ucm-a2-scheduler.service"
    content = path.read_text(encoding="utf-8")

    assert "Type=oneshot" in content
    assert "User=root" in content
    assert "WorkingDirectory=/opt/ucm-a2-scheduler" in content
    assert "Environment=PYTHONPATH=/opt/ucm-a2-scheduler" in content
    assert (
        "ExecStart=/usr/bin/python3 -m a2_scheduler.cli reconcile --config "
        "/etc/ucm-a2-scheduler/config.json"
    ) in content


def test_systemd_timer_matches_monotonic_plan():
    path = Path(__file__).resolve().parents[1] / "systemd" / "ucm-a2-scheduler.timer"
    content = path.read_text(encoding="utf-8")

    assert "OnBootSec=30s" in content
    assert "OnUnitActiveSec=30s" in content
    assert "AccuracySec=1s" in content
    assert "Persistent=" not in content


def test_validate_create_rollback_creates_two_then_cleans_up(capsys, configured_cli):
    assert (
        configured_cli.run(
            [
                "validate-create-rollback",
                "--fail-before-create",
                "3",
                "--config",
                str(configured_cli.config_path),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "command": "validate-create-rollback",
        "created": 2,
        "remaining_vnpus": 0,
        "result": "passed",
    }
    assert configured_cli.npu.created_ids == [300, 301]
    assert configured_cli.npu.destroyed_ids == [300, 301]
    assert configured_cli.npu.inventory == []
    assert configured_cli.github.calls == 2
    assert configured_cli.lock.acquire_calls == [False]
    assert configured_cli.lock.release_calls == 1


@pytest.mark.parametrize("unsafe", ["timer", "pids", "vnpus", "runners"])
def test_validate_create_rollback_rejects_unsafe_preconditions(
    capsys, configured_cli, unsafe
):
    if unsafe == "timer":
        configured_cli.command_runner.timer_active = True
    elif unsafe == "pids":
        configured_cli.npu.card_pids = [123]
    elif unsafe == "vnpus":
        configured_cli.npu.inventory = [Vnpu(212, "vir05_1c_16g")]
    else:
        configured_cli.command_runner.running_runners.add("ucm-a2-vnpu-runner-212")

    assert (
        configured_cli.run(
            [
                "validate-create-rollback",
                "--fail-before-create",
                "3",
                "--config",
                str(configured_cli.config_path),
            ]
        )
        == 1
    )

    assert "validation precondition failed" in capsys.readouterr().err
    assert configured_cli.npu.created_ids == []
    assert configured_cli.npu.destroyed_ids == []


def test_validation_policy_change_blocks_cleanup(capsys, configured_cli):
    configured_cli.github.job_snapshots = [
        [],
        [Job(9, "in_progress", frozenset({"a2"}), run_id=10)],
    ]

    assert (
        configured_cli.run(
            [
                "validate-create-rollback",
                "--fail-before-create",
                "3",
                "--config",
                str(configured_cli.config_path),
            ]
        )
        == 1
    )

    assert "cleanup policy guard failed" in capsys.readouterr().err
    assert configured_cli.npu.created_ids == [300, 301]
    assert configured_cli.npu.destroyed_ids == []
