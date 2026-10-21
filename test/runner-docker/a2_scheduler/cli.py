import argparse
import errno
import importlib
import json
import os
import stat
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .github_client import GitHubClient, GitHubUnavailable
from .host import CommandRunner, NpuManager
from .models import Job, SchedulerConfig, SchedulerState, load_config
from .reconciler import Reconciler
from .runners import RunnerManager

LOCK_BUSY_EXIT = 75
TIMER_UNIT = "ucm-a2-scheduler.timer"
VALIDATION_RECOVERY_GUIDANCE = (
    "leave the timer disabled; inspect status, vNPU inventory, card PIDs, "
    "runners, and target jobs; destroy only callback-attributed IDs after all "
    "guards are clear"
)


class CliError(RuntimeError):
    pass


class InjectedCreateFailure(RuntimeError):
    pass


class StatePersistenceUncertainError(RuntimeError):
    pass


class StatePersistenceCleanupError(RuntimeError):
    def __init__(self, primary_error: Exception, cleanup_error: Exception) -> None:
        super().__init__(
            "state persistence failed before replace; temporary cleanup also failed"
        )
        self.primary_error_type = type(primary_error).__name__
        self.cleanup_error_type = type(cleanup_error).__name__


class LinuxFileLock:
    def __init__(
        self,
        path: Path,
        fcntl_loader: Callable[[], object] = lambda: importlib.import_module("fcntl"),
        os_module=os,
    ) -> None:
        self.path = Path(path)
        self.fcntl_loader = fcntl_loader
        self.os = os_module
        self.descriptor = None
        self.fcntl = None

    def acquire(self, blocking: bool = False) -> bool:
        try:
            directory_info = self.os.lstat(str(self.path.parent))
        except OSError as error:
            raise CliError("lock directory is unavailable") from error
        self._validate_metadata(directory_info, directory=True)
        try:
            path_info = self.os.lstat(str(self.path))
        except FileNotFoundError:
            path_info = None
        except OSError as error:
            raise CliError("lock file metadata is unavailable") from error
        if path_info is not None:
            self._validate_metadata(path_info, directory=False)

        try:
            flags = self.os.O_NOFOLLOW | self.os.O_CREAT | self.os.O_RDWR
        except AttributeError as error:
            raise CliError("secure lock open flags are unavailable") from error
        flags |= getattr(self.os, "O_CLOEXEC", 0)
        descriptor = self.os.open(str(self.path), flags, 0o600)
        try:
            self._validate_metadata(self.os.fstat(descriptor), directory=False)
            fcntl = self.fcntl_loader()
            lock_flags = fcntl.LOCK_EX
            if not blocking:
                lock_flags |= fcntl.LOCK_NB
            fcntl.flock(descriptor, lock_flags)
        except OSError as error:
            self.os.close(descriptor)
            if error.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise
        except Exception:
            self.os.close(descriptor)
            raise
        self.descriptor = descriptor
        self.fcntl = fcntl
        return True

    def release(self) -> None:
        if self.descriptor is None:
            return
        try:
            self.fcntl.flock(self.descriptor, self.fcntl.LOCK_UN)
        finally:
            self.os.close(self.descriptor)
            self.descriptor = None
            self.fcntl = None

    @staticmethod
    def _validate_metadata(info, directory: bool) -> None:
        correct_type = (
            stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
        )
        label = "lock directory" if directory else "lock file"
        if not correct_type:
            raise CliError("{} has an invalid type".format(label))
        if info.st_uid != 0:
            raise CliError("{} is not root-owned".format(label))
        if info.st_mode & 0o077:
            raise CliError("{} permits group or other access".format(label))


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
        except Exception as primary_error:
            self._cleanup_temporary(temporary, primary_error)
            raise
        temporary = None
        try:
            self._fsync_directory()
        except Exception as error:
            raise StatePersistenceUncertainError(
                "state replace completed but directory fsync failed; "
                "durability is uncertain"
            ) from error

    @staticmethod
    def _cleanup_temporary(temporary, primary_error: Exception) -> None:
        if temporary is None:
            return
        try:
            temporary.unlink()
        except FileNotFoundError:
            return
        except Exception as cleanup_error:
            raise StatePersistenceCleanupError(primary_error, cleanup_error) from None

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
        document = self._state_document(result, "reconcile")
        document["outcome"] = {
            "failed": reconciler.outcome.failed,
            "error": reconciler.outcome.error,
        }
        self._print_json(document)
        return 1 if reconciler.outcome.failed else 0

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
        attributed_ids = []

        def inject_before_third(discovered) -> None:
            for vnpu in discovered:
                if vnpu.id not in attributed_ids:
                    attributed_ids.append(vnpu.id)
            if len(attributed_ids) == fail_before_create - 1:
                raise InjectedCreateFailure("injected failure before create 3")

        try:
            npu.create_vnpus(
                config.vnpu_count,
                config.vnpu_template,
                on_created=inject_before_third,
            )
        except Exception as error:
            primary_error = error
        else:
            primary_error = CliError("validation fault injection did not occur")

        try:
            cleaned_ids = self._cleanup_validation_vnpus(
                config,
                command_runner,
                github,
                npu,
                attributed_ids,
            )
        except Exception as cleanup_error:
            message = "{}; cleanup failed: {}".format(primary_error, cleanup_error)
            if VALIDATION_RECOVERY_GUIDANCE not in message:
                message = "{}; {}".format(message, VALIDATION_RECOVERY_GUIDANCE)
            raise CliError(message) from primary_error

        if not isinstance(primary_error, InjectedCreateFailure):
            raise CliError(str(primary_error)) from primary_error
        if len(cleaned_ids) != 2:
            raise CliError("validation did not create and clean exactly two vNPUs")
        self._print_json(
            {
                "command": "validate-create-rollback",
                "created": 2,
                "remaining_vnpus": 0,
                "result": "passed",
            }
        )
        return 0

    def _cleanup_validation_vnpus(self, config, command_runner, github, npu, known_ids):
        attributed = tuple(known_ids)
        if len(attributed) != len(set(attributed)):
            raise CliError(
                "validation callback attribution is invalid; {}".format(
                    VALIDATION_RECOVERY_GUIDANCE
                )
            )
        attributed_set = set(attributed)
        try:
            inventory = npu.list_vnpus()
        except Exception as error:
            raise CliError(
                "validation cleanup could not verify inventory; {}".format(
                    VALIDATION_RECOVERY_GUIDANCE
                )
            ) from error
        inventory_ids = self._validated_inventory_ids(
            inventory, len(attributed), "inventory"
        )
        if inventory_ids != attributed_set:
            raise CliError(
                "validation inventory is not exactly the callback-attributed set; "
                "no vNPUs were destroyed; {}".format(VALIDATION_RECOVERY_GUIDANCE)
            )
        if not attributed:
            return ()

        self._require_cleanup_policy(config, command_runner, github)
        npu.destroy_vnpus(tuple(sorted(attributed_set)))
        try:
            final_inventory = npu.list_vnpus()
        except Exception as error:
            raise CliError(
                "validation cleanup could not verify the final inventory; {}".format(
                    VALIDATION_RECOVERY_GUIDANCE
                )
            ) from error
        remaining = self._validated_inventory_ids(final_inventory, 0, "final inventory")
        if remaining:
            raise CliError(
                "validation cleanup did not reach zero vNPUs; {}".format(
                    VALIDATION_RECOVERY_GUIDANCE
                )
            )
        return tuple(sorted(attributed_set))

    def _validated_inventory_ids(self, inventory, expected_count, label):
        raw_ids = []
        for vnpu in inventory:
            vnpu_id = getattr(vnpu, "id", None)
            if type(vnpu_id) is not int or vnpu_id < 0:
                raise CliError(
                    "validation {} has invalid shape; {}".format(
                        label, VALIDATION_RECOVERY_GUIDANCE
                    )
                )
            raw_ids.append(vnpu_id)
        if len(raw_ids) != len(set(raw_ids)):
            raise CliError(
                "validation {} is ambiguous; {}".format(
                    label, VALIDATION_RECOVERY_GUIDANCE
                )
            )
        if len(raw_ids) != expected_count:
            raise CliError(
                "validation {} has an unexpected count; {}".format(
                    label, VALIDATION_RECOVERY_GUIDANCE
                )
            )
        return set(raw_ids)

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
