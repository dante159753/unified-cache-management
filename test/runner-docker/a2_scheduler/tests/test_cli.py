import json
import stat
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from a2_scheduler.cli import (
    AtomicStateStore,
    CliDependencies,
    CliError,
    LinuxFileLock,
    SchedulerCli,
    StatePersistenceCleanupError,
    StatePersistenceUncertainError,
)
from a2_scheduler.host import CommandResult, NpuOperationError
from a2_scheduler.models import Job, Phase, SchedulerState, Vnpu, load_config
from a2_scheduler.reconciler import ReconcileOutcome

TOKEN = "TOKEN_SENTINEL"
VALIDATION_RECOVERY_GUIDANCE = (
    "leave the timer disabled; inspect status, vNPU inventory, card PIDs, "
    "runners, and target jobs; destroy only callback-attributed IDs after all "
    "guards are clear"
)


class FakeSecureOs:
    O_NOFOLLOW = 0x01
    O_CREAT = 0x02
    O_RDWR = 0x04
    O_CLOEXEC = 0x08

    def __init__(self):
        self.directory_stat = SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=0)
        self.path_stat = None
        self.fd_stat = SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_uid=0)
        self.open_calls = []
        self.closed = []

    def lstat(self, path):
        if str(path).endswith("ucm-a2-scheduler"):
            return self.directory_stat
        if self.path_stat is None:
            raise FileNotFoundError(str(path))
        return self.path_stat

    def open(self, path, flags, mode):
        self.open_calls.append((path, flags, mode))
        return 42

    def fstat(self, descriptor):
        assert descriptor == 42
        return self.fd_stat

    def close(self, descriptor):
        self.closed.append(descriptor)


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
        self.create_failure = None
        self.destroy_error = None
        self.list_vnpu_calls = 0
        self.external_vnpu_on_call = None
        self.inventory_error_on_call = None
        self.inventory_error = None
        self.inventory_overrides = {}

    def list_vnpus(self):
        self.list_vnpu_calls += 1
        if self.external_vnpu_on_call == self.list_vnpu_calls:
            self.inventory.append(Vnpu(999, "vir05_1c_16g"))
        if self.inventory_error_on_call == self.list_vnpu_calls:
            raise self.inventory_error or NpuOperationError(
                "inventory failed {}".format(TOKEN)
            )
        if self.list_vnpu_calls in self.inventory_overrides:
            return list(self.inventory_overrides[self.list_vnpu_calls])
        return list(self.inventory)

    def list_card_pids(self):
        return list(self.card_pids)

    def create_vnpus(self, count, template, on_created=None):
        created = []
        for index in range(count):
            if self.create_failure == "second_create" and index == 1:
                raise NpuOperationError("second create failed")
            vnpu = Vnpu(300 + index, template)
            self.created_ids.append(vnpu.id)
            self.inventory.append(vnpu)
            self.mutations.append("create")
            created.append(vnpu)
            if self.create_failure == "second_inventory" and index == 1:
                raise NpuOperationError("second inventory failed")
            if self.create_failure == "progress_callback" and index == 0:
                try:
                    on_created(None)
                except Exception as error:
                    raise RuntimeError("progress callback failed") from error
            if on_created is not None:
                on_created(tuple(created))
        return created

    def destroy_vnpus(self, vnpu_ids):
        if self.destroy_error is not None:
            raise self.destroy_error
        ids = list(vnpu_ids)
        self.destroyed_ids.extend(ids)
        self.mutations.extend("destroy" for _ in ids)
        self.inventory = [vnpu for vnpu in self.inventory if vnpu.id not in ids]


class FakeRunners:
    def __init__(self):
        self.mutations = []


class CheckpointingReconciler:
    def __init__(self, state, checkpoint, outcome=None, result=None):
        self.state = state
        self.checkpoint = checkpoint
        self.outcome = outcome or ReconcileOutcome.failed_run("current failure")
        self.result = result

    def reconcile(self, now):
        state = self.result or SchedulerState(
            phase=Phase.WHOLE,
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
    assert persisted.phase == Phase.WHOLE
    assert TOKEN not in (persisted.last_error or "")
    assert TOKEN not in output
    assert replacements and Path(replacements[0][1]) == configured_cli.state_path
    assert configured_cli.lock.release_calls == 1


def test_reconcile_failed_rollback_to_whole_returns_nonzero(capsys, configured_cli):
    result = SchedulerState(phase=Phase.WHOLE, last_error="allocation failed")
    configured_cli.reconciler_factory = lambda *args: CheckpointingReconciler(
        args[1],
        args[5],
        outcome=ReconcileOutcome.failed_run("allocation failed"),
        result=result,
    )

    assert (
        configured_cli.run(["reconcile", "--config", str(configured_cli.config_path)])
        == 1
    )
    assert json.loads(capsys.readouterr().out)["phase"] == "WHOLE"


def test_reconcile_healthy_run_with_stale_error_returns_zero(capsys, configured_cli):
    result = SchedulerState(phase=Phase.WHOLE, last_error="historical failure")
    configured_cli.reconciler_factory = lambda *args: CheckpointingReconciler(
        args[1],
        args[5],
        outcome=ReconcileOutcome.success(),
        result=result,
    )

    assert (
        configured_cli.run(["reconcile", "--config", str(configured_cli.config_path)])
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["last_error"] == "historical failure"
    assert output["outcome"] == {"failed": False, "error": None}


def test_reconcile_reports_current_failure_without_mutating_state(
    capsys, configured_cli
):
    result = SchedulerState(phase=Phase.WHOLE)
    configured_cli.reconciler_factory = lambda *args: CheckpointingReconciler(
        args[1],
        args[5],
        outcome=ReconcileOutcome.failed_run("GitHub HTTP 502 {}".format(TOKEN)),
        result=result,
    )
    assert (
        configured_cli.run(["reconcile", "--config", str(configured_cli.config_path)])
        == 1
    )
    output = capsys.readouterr().out
    assert TOKEN not in output
    document = json.loads(output)
    assert document["last_error"] is None
    assert document["outcome"]["failed"] is True
    assert "GitHub HTTP 502" in document["outcome"]["error"]


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


def test_linux_file_lock_securely_opens_root_owned_runtime_file():
    calls = []
    fake_fcntl = SimpleNamespace(LOCK_EX=2, LOCK_NB=4, LOCK_UN=8)
    fake_fcntl.flock = lambda descriptor, flags: calls.append((descriptor, flags))
    loads = []
    fake_os = FakeSecureOs()
    lock = LinuxFileLock(
        Path("/run/ucm-a2-scheduler/card7.lock"),
        fcntl_loader=lambda: loads.append(True) or fake_fcntl,
        os_module=fake_os,
    )

    assert loads == []
    assert lock.acquire(blocking=False) is True
    assert loads == [True]
    assert fake_os.open_calls == [
        (
            str(Path("/run/ucm-a2-scheduler/card7.lock")),
            fake_os.O_NOFOLLOW | fake_os.O_CREAT | fake_os.O_RDWR | fake_os.O_CLOEXEC,
            0o600,
        )
    ]
    assert calls[0][1] == fake_fcntl.LOCK_EX | fake_fcntl.LOCK_NB
    lock.release()
    assert calls[-1][1] == fake_fcntl.LOCK_UN
    assert fake_os.closed == [42]


@pytest.mark.parametrize(
    "target,mode,uid",
    [
        ("directory", stat.S_IFREG | 0o700, 0),
        ("directory", stat.S_IFDIR | 0o700, 1000),
        ("directory", stat.S_IFDIR | 0o770, 0),
        ("path", stat.S_IFLNK | 0o600, 0),
        ("path", stat.S_IFREG | 0o600, 1000),
        ("path", stat.S_IFREG | 0o640, 0),
        ("fd", stat.S_IFDIR | 0o600, 0),
        ("fd", stat.S_IFREG | 0o600, 1000),
        ("fd", stat.S_IFREG | 0o604, 0),
    ],
)
def test_linux_file_lock_rejects_insecure_metadata_and_closes_fd(target, mode, uid):
    fake_os = FakeSecureOs()
    insecure = SimpleNamespace(st_mode=mode, st_uid=uid)
    if target == "directory":
        fake_os.directory_stat = insecure
    elif target == "path":
        fake_os.path_stat = insecure
    else:
        fake_os.fd_stat = insecure
    lock = LinuxFileLock(
        Path("/run/ucm-a2-scheduler/card7.lock"),
        fcntl_loader=lambda: pytest.fail("fcntl must not load"),
        os_module=fake_os,
    )

    with pytest.raises(CliError, match="lock"):
        lock.acquire(blocking=False)

    assert fake_os.closed == ([42] if target == "fd" else [])


@pytest.mark.parametrize("failure", ["load", "flock", "contention"])
def test_linux_file_lock_closes_fd_on_post_open_failure(failure):
    fake_os = FakeSecureOs()
    fake_fcntl = SimpleNamespace(LOCK_EX=2, LOCK_NB=4, LOCK_UN=8)
    if failure == "load":
        loader = lambda: (_ for _ in ()).throw(RuntimeError("no fcntl"))
    else:
        error_number = 11 if failure == "contention" else 5

        def flock(descriptor, flags):
            raise OSError(error_number, failure)

        fake_fcntl.flock = flock
        loader = lambda: fake_fcntl
    lock = LinuxFileLock(
        Path("/run/ucm-a2-scheduler/card7.lock"),
        fcntl_loader=loader,
        os_module=fake_os,
    )

    if failure == "contention":
        assert lock.acquire(blocking=False) is False
    else:
        with pytest.raises((RuntimeError, OSError)):
            lock.acquire(blocking=False)

    assert fake_os.closed == [42]


def test_atomic_state_store_replaces_complete_document(tmp_path):
    path = tmp_path / "state.json"
    store = AtomicStateStore(path)
    store.save(SchedulerState(phase=Phase.WHOLE))

    assert (
        SchedulerState.from_json(path.read_text(encoding="utf-8")).phase == Phase.WHOLE
    )
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_atomic_state_store_cleans_temp_and_propagates_write_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.json"
    old_state = SchedulerState(phase=Phase.WHOLE, last_error="old")
    path.write_text(old_state.to_json(), encoding="utf-8")
    real_factory = __import__("tempfile").NamedTemporaryFile

    class WriteFailure:
        def __init__(self, handle):
            self.handle = handle
            self.name = handle.name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def write(self, document):
            raise OSError("write failed")

    monkeypatch.setattr(
        "a2_scheduler.cli.tempfile.NamedTemporaryFile",
        lambda **kwargs: WriteFailure(real_factory(**kwargs)),
    )

    with pytest.raises(OSError, match="write failed"):
        AtomicStateStore(path).save(SchedulerState())

    assert path.read_text(encoding="utf-8") == old_state.to_json()
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_atomic_state_store_cleans_temp_and_propagates_file_fsync_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.json"
    old_state = SchedulerState(phase=Phase.WHOLE, last_error="old")
    path.write_text(old_state.to_json(), encoding="utf-8")
    monkeypatch.setattr(
        "a2_scheduler.cli.os.fsync",
        lambda descriptor: (_ for _ in ()).throw(OSError("file fsync failed")),
    )

    with pytest.raises(OSError, match="file fsync failed"):
        AtomicStateStore(path).save(SchedulerState())

    assert path.read_text(encoding="utf-8") == old_state.to_json()
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_atomic_state_store_cleans_temp_and_propagates_replace_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.json"
    old_state = SchedulerState(phase=Phase.WHOLE, last_error="old")
    path.write_text(old_state.to_json(), encoding="utf-8")
    monkeypatch.setattr(
        "a2_scheduler.cli.os.replace",
        lambda source, destination: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        AtomicStateStore(path).save(SchedulerState())

    assert path.read_text(encoding="utf-8") == old_state.to_json()
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_atomic_state_store_reports_post_replace_directory_fsync_uncertainty(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.json"
    old_state = SchedulerState(phase=Phase.WHOLE, last_error="old")
    new_state = SchedulerState(phase=Phase.WHOLE, last_error="new")
    path.write_text(old_state.to_json(), encoding="utf-8")
    store = AtomicStateStore(path)
    monkeypatch.setattr(
        store,
        "_fsync_directory",
        lambda: (_ for _ in ()).throw(OSError("directory fsync failed")),
    )

    with pytest.raises(
        StatePersistenceUncertainError,
        match="replace completed.*durability is uncertain",
    ):
        store.save(new_state)

    assert SchedulerState.from_json(path.read_text(encoding="utf-8")) == new_state
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_atomic_state_store_preserves_primary_when_temp_unlink_also_fails(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.json"
    secret_path = tmp_path / "private" / "github.token"
    leaked_temp_paths = []

    def fail_replace(source, destination):
        raise OSError("replace failed {} {}".format(TOKEN, secret_path))

    def fail_unlink(temporary):
        leaked_temp_paths.append(str(temporary))
        raise OSError("unlink failed {} {}".format(TOKEN, temporary))

    monkeypatch.setattr("a2_scheduler.cli.os.replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(StatePersistenceCleanupError) as captured:
        AtomicStateStore(path).save(SchedulerState())

    error = captured.value
    message = str(error)
    assert error.primary_error_type == "OSError"
    assert error.cleanup_error_type == "OSError"
    assert "state persistence failed before replace" in message
    assert "temporary cleanup also failed" in message
    assert TOKEN not in message
    assert str(secret_path) not in message
    assert leaked_temp_paths and leaked_temp_paths[0] not in message


def test_systemd_service_matches_plan():
    path = Path(__file__).resolve().parents[1] / "systemd" / "ucm-a2-scheduler.service"
    content = path.read_text(encoding="utf-8")

    assert "Type=oneshot" in content
    assert "User=root" in content
    assert "WorkingDirectory=/opt/ucm-a2-scheduler" in content
    assert "Environment=PYTHONPATH=/opt/ucm-a2-scheduler" in content
    assert "RuntimeDirectory=ucm-a2-scheduler" in content
    assert "RuntimeDirectoryMode=0700" in content
    assert (
        "ExecStart=/usr/bin/python3 -m a2_scheduler.cli reconcile --config "
        "/etc/ucm-a2-scheduler/config.json"
    ) in content


def test_systemd_service_preserves_runtime_directory_for_lock_identity():
    path = Path(__file__).resolve().parents[1] / "systemd" / "ucm-a2-scheduler.service"
    lines = path.read_text(encoding="utf-8").splitlines()

    assert lines.count("RuntimeDirectoryPreserve=yes") == 1


def test_config_example_lock_path_matches_systemd_runtime_directory():
    package = Path(__file__).resolve().parents[1]
    config = json.loads((package / "config.example.json").read_text(encoding="utf-8"))
    service_lines = (
        (package / "systemd" / "ucm-a2-scheduler.service")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    runtime_directory = next(
        line.split("=", 1)[1]
        for line in service_lines
        if line.startswith("RuntimeDirectory=")
    )

    assert config["lock_file"] == "/run/{}/card7.lock".format(runtime_directory)


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


@pytest.mark.parametrize(
    "failure,created_ids,destroyed_ids",
    [
        ("second_create", [300], [300]),
        ("second_inventory", [300, 301], []),
        ("progress_callback", [300], []),
    ],
)
def test_validation_unexpected_post_create_failure_cleans_only_attributed_ids(
    capsys, configured_cli, failure, created_ids, destroyed_ids
):
    configured_cli.npu.create_failure = failure

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

    error = capsys.readouterr().err
    assert failure.replace("_", " ") in error
    assert configured_cli.npu.created_ids == created_ids
    assert configured_cli.npu.destroyed_ids == destroyed_ids
    if destroyed_ids != created_ids:
        assert VALIDATION_RECOVERY_GUIDANCE in error


def test_validation_external_vnpu_before_cleanup_is_never_destroyed(
    capsys, configured_cli
):
    configured_cli.npu.external_vnpu_on_call = 2

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

    error = capsys.readouterr().err
    assert VALIDATION_RECOVERY_GUIDANCE in error
    assert TOKEN not in error
    assert str(configured_cli.token_path) not in error
    assert configured_cli.npu.destroyed_ids == []
    assert [vnpu.id for vnpu in configured_cli.npu.inventory] == [300, 301, 999]


def test_validation_inventory_failure_with_external_vnpu_destroys_nothing(
    capsys, configured_cli
):
    configured_cli.npu.external_vnpu_on_call = 2
    configured_cli.npu.inventory_error_on_call = 2
    configured_cli.npu.inventory_error = NpuOperationError(
        "inventory failed {} {}".format(TOKEN, configured_cli.token_path)
    )

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

    error = capsys.readouterr().err
    assert VALIDATION_RECOVERY_GUIDANCE in error
    assert TOKEN not in error
    assert str(configured_cli.token_path) not in error
    assert configured_cli.npu.destroyed_ids == []
    assert [vnpu.id for vnpu in configured_cli.npu.inventory] == [300, 301, 999]


def test_validation_duplicate_initial_inventory_destroys_nothing(
    capsys, configured_cli
):
    duplicate = Vnpu(212, "vir05_1c_16g")
    configured_cli.npu.inventory_overrides[1] = [duplicate, duplicate]

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


def test_validation_duplicate_pre_cleanup_inventory_destroys_nothing(
    capsys, configured_cli
):
    configured_cli.npu.inventory_overrides[2] = [
        Vnpu(300, "vir05_1c_16g"),
        Vnpu(301, "vir05_1c_16g"),
        Vnpu(300, "vir05_1c_16g"),
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

    error = capsys.readouterr().err
    assert "inventory is ambiguous" in error
    assert VALIDATION_RECOVERY_GUIDANCE in error
    assert configured_cli.npu.destroyed_ids == []


def test_validation_malformed_pre_cleanup_inventory_destroys_nothing(
    capsys, configured_cli
):
    configured_cli.npu.inventory_overrides[2] = [
        Vnpu(300, "vir05_1c_16g"),
        Vnpu("301", "vir05_1c_16g"),
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

    error = capsys.readouterr().err
    assert "inventory has invalid shape" in error
    assert VALIDATION_RECOVERY_GUIDANCE in error
    assert configured_cli.npu.destroyed_ids == []


def test_validation_duplicate_post_cleanup_inventory_issues_no_further_destroy(
    capsys, configured_cli
):
    duplicate = Vnpu(300, "vir05_1c_16g")
    configured_cli.npu.inventory_overrides[3] = [duplicate, duplicate]

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

    error = capsys.readouterr().err
    assert "final inventory is ambiguous" in error
    assert VALIDATION_RECOVERY_GUIDANCE in error
    assert configured_cli.npu.destroyed_ids == [300, 301]


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

    error = capsys.readouterr().err
    assert "injected failure before create 3" in error
    assert "cleanup policy guard failed" in error
    assert configured_cli.npu.created_ids == [300, 301]
    assert configured_cli.npu.destroyed_ids == []


def test_validation_pid_guard_failure_preserves_primary_and_cleanup_errors(
    capsys, configured_cli
):
    configured_cli.npu.destroy_error = NpuOperationError(
        "NPU card processes are active"
    )

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

    error = capsys.readouterr().err
    assert "injected failure before create 3" in error
    assert "NPU card processes are active" in error
    assert configured_cli.npu.destroyed_ids == []
    assert [vnpu.id for vnpu in configured_cli.npu.inventory] == [300, 301]
