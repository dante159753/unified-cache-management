import argparse
import errno
import importlib
import json
import os
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .github_client import GitHubClient, GitHubUnavailable
from .host import CommandRunner, NpuManager
from .models import Job, Phase, SchedulerConfig, SchedulerState, load_config
from .reconciler import Reconciler
from .runners import RunnerManager

LOCK_BUSY_EXIT = 75
TIMER_UNIT = "ucm-a2-scheduler.timer"


class CliError(RuntimeError):
    pass


class InjectedCreateFailure(RuntimeError):
    pass


class LinuxFileLock:
    def __init__(
        self,
        path: Path,
        fcntl_loader: Callable[[], object] = lambda: importlib.import_module("fcntl"),
    ) -> None:
        self.path = Path(path)
        self.fcntl_loader = fcntl_loader
        self.handle = None
        self.fcntl = None

    def acquire(self, blocking: bool = False) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+")
        try:
            fcntl = self.fcntl_loader()
        except Exception:
            handle.close()
            raise
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(handle.fileno(), flags)
        except OSError as error:
            handle.close()
            if error.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise
        self.handle = handle
        self.fcntl = fcntl
        return True

    def release(self) -> None:
        if self.handle is None:
            return
        try:
            self.fcntl.flock(self.handle.fileno(), self.fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None
            self.fcntl = None


class AtomicStateStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> SchedulerState:
        try:
            document = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return SchedulerState()
        return SchedulerState.from_json(document)

    def save(self, state: SchedulerState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.path.parent),
                prefix=".{}.".format(self.path.name),
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(state.to_json())
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(self.path))
            temporary = None
            self._fsync_directory()
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    def _fsync_directory(self) -> None:
        if not hasattr(os, "O_DIRECTORY"):
            return
        descriptor = os.open(str(self.path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class CliDependencies:
    def __init__(
        self,
        command_runner_factory: Callable[[], object] = CommandRunner,
        github_factory: Callable[
            [SchedulerConfig, str], object
        ] = lambda config, token: GitHubClient(config.repository, token),
        npu_factory: Callable[
            [SchedulerConfig, object], object
        ] = lambda config, runner: NpuManager(
            runner,
            config.card_id,
            config.chip_id,
            config.eviction_grace_seconds,
        ),
        runner_factory: Callable[
            [SchedulerConfig, object], object
        ] = lambda config, runner: RunnerManager(config, runner),
        reconciler_factory: Callable[
            ..., object
        ] = lambda config, state, github, npu, runners, checkpoint, clock: Reconciler(
            config,
            state,
            github,
            npu,
            runners,
            checkpoint=checkpoint,
            clock=clock,
        ),
        lock_factory: Callable[[Path], object] = LinuxFileLock,
        state_store_factory: Callable[[Path], AtomicStateStore] = AtomicStateStore,
        config_loader: Callable[[Path], SchedulerConfig] = load_config,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.command_runner_factory = command_runner_factory
        self.github_factory = github_factory
        self.npu_factory = npu_factory
        self.runner_factory = runner_factory
        self.reconciler_factory = reconciler_factory
        self.lock_factory = lock_factory
        self.state_store_factory = state_store_factory
        self.config_loader = config_loader
        self.clock = clock


class SchedulerCli:
    def __init__(self, dependencies: Optional[CliDependencies] = None) -> None:
        self.dependencies = dependencies or CliDependencies()
        self.secrets = []

    def run(self, argv: Optional[Sequence[str]] = None) -> int:
        arguments = self._parser().parse_args(argv)
        try:
            config = self.dependencies.config_loader(Path(arguments.config))
            self.secrets = [str(config.token_file)]
            if arguments.command in ("status", "dry-run"):
                return self._observe(arguments.command, config)
            return self._run_locked(arguments, config)
        except Exception as error:
            self._print_error(error)
            return 1

    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="python3 -m a2_scheduler.cli")
        subparsers = parser.add_subparsers(dest="command", required=True)
        for command in ("reconcile", "dry-run", "status"):
            command_parser = subparsers.add_parser(command)
            command_parser.add_argument("--config", required=True)
        validation = subparsers.add_parser("validate-create-rollback")
        validation.add_argument("--fail-before-create", required=True, type=int)
        validation.add_argument("--config", required=True)
        return parser

    def _run_locked(self, arguments, config: SchedulerConfig) -> int:
        lock = self.dependencies.lock_factory(config.lock_file)
        if not lock.acquire(blocking=False):
            self._print_error(CliError("scheduler lock is busy"))
            return LOCK_BUSY_EXIT
        try:
            if arguments.command == "reconcile":
                return self._reconcile(config)
            return self._validate_create_rollback(config, arguments.fail_before_create)
        finally:
            lock.release()

    def _components(self, config: SchedulerConfig):
        token = self._read_token(config.token_file)
        command_runner = self.dependencies.command_runner_factory()
        github = self.dependencies.github_factory(config, token)
        npu = self.dependencies.npu_factory(config, command_runner)
        runners = self.dependencies.runner_factory(config, command_runner)
        return command_runner, github, npu, runners

    def _read_token(self, path: Path) -> str:
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise CliError("unable to read GitHub token") from error
        if not token:
            raise CliError("GitHub token is empty")
        self.secrets.append(token)
        return token

    def _reconcile(self, config: SchedulerConfig) -> int:
        _, github, npu, runners = self._components(config)
        store = self.dependencies.state_store_factory(config.state_file)
        state = store.load()

        def checkpoint(candidate: SchedulerState) -> None:
            store.save(self._redacted_state(candidate))

        reconciler = self.dependencies.reconciler_factory(
            config,
            state,
            github,
            npu,
            runners,
            checkpoint,
            self.dependencies.clock,
        )
        result = reconciler.reconcile(self.dependencies.clock())
        self._print_json(self._state_document(result, "reconcile"))
        return 1 if result.phase == Phase.ERROR else 0

    def _observe(self, command: str, config: SchedulerConfig) -> int:
        command_runner, github, npu, _ = self._components(config)
        state = self.dependencies.state_store_factory(config.state_file).load()
        observed_at = self.dependencies.clock()
        jobs = None
        github_poll = {"status": "ok", "observed_at": observed_at.isoformat()}
        try:
            jobs = github.list_target_jobs()
        except GitHubUnavailable:
            github_poll = {"status": "unknown", "observed_at": None}
        inventory = npu.list_vnpus()
        runners = self._runner_status(config, command_runner)
        document = self._state_document(state, command)
        document.update(
            {
                "mutation": False,
                "window_open": self._window_open(config, observed_at),
                "github_poll": github_poll,
                "target_jobs": self._job_counts(jobs),
                "vnpus": [
                    {"id": vnpu.id, "template": vnpu.template} for vnpu in inventory
                ],
                "runners": runners,
            }
        )
        self._print_json(document)
        return 0

    def _validate_create_rollback(
        self, config: SchedulerConfig, fail_before_create: int
    ) -> int:
        if fail_before_create != 3:
            raise CliError("validation requires --fail-before-create 3")
        command_runner, github, npu, _ = self._components(config)
        self._require_validation_preconditions(config, command_runner, npu, github)
        created = []

        def inject_before_third(discovered) -> None:
            created[:] = list(discovered)
            if len(created) == fail_before_create - 1:
                raise InjectedCreateFailure("injected failure before create 3")

        try:
            npu.create_vnpus(
                config.vnpu_count,
                config.vnpu_template,
                on_created=inject_before_third,
            )
        except InjectedCreateFailure:
            pass
        if len(created) != 2:
            raise CliError("validation did not stop after exactly two creates")
        self._require_cleanup_policy(config, command_runner, github)
        npu.destroy_vnpus([vnpu.id for vnpu in created])
        remaining = npu.list_vnpus()
        if remaining:
            raise CliError("validation cleanup did not restore zero vNPUs")
        self._print_json(
            {
                "command": "validate-create-rollback",
                "created": 2,
                "remaining_vnpus": 0,
                "result": "passed",
            }
        )
        return 0

    def _require_validation_preconditions(
        self, config, command_runner, npu, github
    ) -> None:
        result = command_runner.run(["systemctl", "is-active", TIMER_UNIT], timeout=30)
        if result.returncode != 3 or result.stdout.strip() != "inactive":
            raise CliError("validation precondition failed: timer is not inactive")
        if npu.list_card_pids():
            raise CliError("validation precondition failed: card 7 has processes")
        if npu.list_vnpus():
            raise CliError("validation precondition failed: card 7 has vNPUs")
        if not self._all_runners_stopped(config, command_runner):
            raise CliError("validation precondition failed: scheduler runner is active")
        jobs = github.list_target_jobs()
        if any(job.status == "in_progress" for job in jobs):
            raise CliError("validation precondition failed: target job is active")

    def _require_cleanup_policy(self, config, command_runner, github) -> None:
        jobs = github.list_target_jobs()
        if any(job.status == "in_progress" for job in jobs):
            raise CliError("cleanup policy guard failed: target job is active")
        if not self._all_runners_stopped(config, command_runner):
            raise CliError("cleanup policy guard failed: scheduler runner is active")

    def _all_runners_stopped(self, config, command_runner) -> bool:
        return not any(
            self._runner_is_running(slot.name, command_runner)
            for slot in config.runner_slots
        )

    def _runner_status(self, config, command_runner) -> List[Dict[str, object]]:
        status = []
        for slot in config.runner_slots:
            running = self._runner_is_running(slot.name, command_runner)
            listener = False
            worker = False
            if running:
                result = command_runner.run(
                    ["docker", "top", slot.name, "-eo", "pid,args"], timeout=30
                )
                if result.returncode != 0:
                    raise CliError("unable to inspect scheduler runner processes")
                listener = "Runner.Listener" in result.stdout
                worker = "Runner.Worker" in result.stdout
            status.append(
                {
                    "name": slot.name,
                    "running": running,
                    "listener": listener,
                    "worker": worker,
                }
            )
        return status

    @staticmethod
    def _runner_is_running(name, command_runner) -> bool:
        result = command_runner.run(
            [
                "docker",
                "ps",
                "--filter",
                "name=^/{}$".format(name),
                "--format",
                "{{.Names}}",
            ],
            timeout=30,
        )
        if result.returncode != 0:
            raise CliError("unable to inspect scheduler runner state")
        names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if any(item != name for item in names):
            raise CliError("scheduler runner state is ambiguous")
        return name in names

    @staticmethod
    def _window_open(config: SchedulerConfig, now: datetime) -> bool:
        return config.window_start_minute <= now.minute < config.window_end_minute

    @staticmethod
    def _job_counts(jobs: Optional[Sequence[Job]]):
        if jobs is None:
            return None
        return {
            "queued": sum(job.status == "queued" for job in jobs),
            "in_progress": sum(job.status == "in_progress" for job in jobs),
        }

    @staticmethod
    def _state_document(state: SchedulerState, command: str) -> dict:
        return {
            "command": command,
            "phase": state.phase.value,
            "vnpu_ids": list(state.vnpu_ids),
            "original_device_mode": state.original_device_mode,
            "transition_timestamp": state.transition_timestamp,
            "last_error": state.last_error,
        }

    def _redacted_state(self, state: SchedulerState) -> SchedulerState:
        return replace(state, last_error=self._redact_text(state.last_error))

    def _redact_text(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        redacted = value
        for secret in sorted(set(self.secrets), key=len, reverse=True):
            if secret:
                redacted = redacted.replace(secret, "<redacted>")
        return redacted

    def _redact_value(self, value):
        if isinstance(value, str):
            return self._redact_text(value)
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._redact_value(item) for key, item in value.items()}
        return value

    def _print_json(self, document: dict) -> None:
        print(json.dumps(self._redact_value(document), indent=2, sort_keys=True))

    def _print_error(self, error: Exception) -> None:
        import sys

        document = {"error": self._redact_text(str(error)) or "command failed"}
        print(json.dumps(document, sort_keys=True), file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    return SchedulerCli().run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
