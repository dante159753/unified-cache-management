import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, List, Optional, Sequence

from .host import CommandResult, CommandRunner
from .models import RunnerSlot, SchedulerConfig, Vnpu

RUNNER_COMMAND = (
    "cd /actions-runner\n"
    'git config --global http.proxy "$http_proxy"\n'
    'git config --global https.proxy "$https_proxy"\n'
    "exec ./run.sh --once"
)


class RunnerOperationError(RuntimeError):
    pass


class RunnerManager:
    def __init__(
        self,
        config: SchedulerConfig,
        command_runner: CommandRunner,
        command_timeout: int = 30,
    ):
        self.config = config
        self.command_runner = command_runner
        self.command_timeout = command_timeout

    def prepare(self, vnpus: Iterable[Vnpu]) -> None:
        discovered = sorted(vnpus, key=lambda vnpu: vnpu.id)
        if len(discovered) != 4:
            raise ValueError("exactly four vNPUs are required")
        if len({vnpu.id for vnpu in discovered}) != 4:
            raise ValueError("vNPU IDs must be unique")

        self.recover_paused()
        try:
            for slot, vnpu in zip(self.config.runner_slots, discovered):
                self._remove_for_prepare(slot.name)
                self._require_success(
                    self._run(self._create_argv(slot, vnpu)),
                    "create runner container",
                )
        except RunnerOperationError as error:
            cleanup_errors = self._cleanup_runner_containers()
            if cleanup_errors:
                raise RunnerOperationError(
                    "{}; cleanup failed: {}".format(error, "; ".join(cleanup_errors))
                ) from error
            raise

    def start_all(self, allow_start: Callable[[], bool] = lambda: True) -> None:
        self.recover_paused()
        self._start([slot.name for slot in self.config.runner_slots], allow_start)

    def restart_exited(self, allow_start: Callable[[], bool] = lambda: True) -> None:
        self.recover_paused()
        exited = []
        for slot in self.config.runner_slots:
            result = self._run(
                ["docker", "inspect", "--format", "{{.State.Running}}", slot.name]
            )
            if result.returncode == 0 and result.stdout.strip().lower() == "true":
                continue
            if result.returncode != 0:
                self._require_success(result, "inspect runner container")
            exited.append(slot.name)
        self._start(exited, allow_start)

    def active_workers(self) -> int:
        self.recover_paused()
        return sum(
            self._top_has_worker(name) for name in self._running_container_names()
        )

    def stop_idle(self) -> bool:
        self.recover_paused()
        paused = []
        all_idle = True
        try:
            for name in self._running_container_names():
                self._pause(name)
                paused.append(name)
                if self._top_has_worker(name):
                    all_idle = False
                    self._unpause(name)
                    paused.remove(name)
                    continue
                self._remove_frozen(name)
                paused.remove(name)
        except RunnerOperationError as error:
            self._raise_after_unpause(error, paused)
        return all_idle

    def stop_all_if_idle(self) -> bool:
        self.recover_paused()
        names = self._running_container_names()
        paused = []
        try:
            for name in names:
                self._pause(name)
                paused.append(name)
            has_worker = any([self._top_has_worker(name) for name in names])
            if has_worker:
                cleanup_errors = self._unpause_all(paused)
                if cleanup_errors:
                    raise RunnerOperationError(
                        "unpause runner containers failed: {}".format(
                            "; ".join(cleanup_errors)
                        )
                    )
                return False
            for name in names:
                self._remove_frozen(name)
                paused.remove(name)
        except RunnerOperationError as error:
            self._raise_after_unpause(error, paused)
        return True

    def recover_paused(self) -> None:
        paused = []
        for slot in self.config.runner_slots:
            state = self._inspect_state(
                slot.name,
                "Paused",
                "inspect paused runner container",
                allow_absent=True,
            )
            if state:
                paused.append(slot.name)
        errors = self._unpause_all(paused)
        if errors:
            raise RunnerOperationError(
                "recover paused runner containers failed: {}".format("; ".join(errors))
            )

    def _running_container_names(self) -> List[str]:
        names = []
        for slot in self.config.runner_slots:
            state = self._inspect_state(
                slot.name,
                "Running",
                "inspect runner container",
                allow_absent=True,
            )
            if state:
                names.append(slot.name)
        return names

    def _top_has_worker(self, name: str) -> bool:
        result = self._run(["docker", "top", name, "-eo", "pid,args"])
        self._require_success(result, "inspect runner processes")
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines or " ".join(lines[0].split()) != "PID COMMAND":
            raise RunnerOperationError("runner process snapshot has invalid header")
        commands = []
        for line in lines[1:]:
            fields = line.split(None, 1)
            if len(fields) != 2:
                raise RunnerOperationError("runner process snapshot has malformed row")
            try:
                pid = int(fields[0])
            except ValueError as error:
                raise RunnerOperationError(
                    "runner process snapshot has malformed row"
                ) from error
            command = fields[1].strip()
            if pid <= 0 or not command:
                raise RunnerOperationError("runner process snapshot has malformed row")
            commands.append(command)
        if not commands:
            raise RunnerOperationError("runner process snapshot has no process rows")
        if not any("Runner.Listener" in command for command in commands):
            raise RunnerOperationError("runner process snapshot has no Runner.Listener")
        return any("Runner.Worker" in command for command in commands)

    def _pause(self, name: str) -> None:
        try:
            result = self._run(["docker", "pause", name])
            self._require_success(result, "pause runner container")
        except RunnerOperationError as error:
            pause_error = RunnerOperationError(
                "pause runner container failed: {}".format(error)
            )
            self._compensate_uncertain_pause(name, pause_error)

    def _unpause(self, name: str) -> None:
        self._require_success(
            self._run(["docker", "unpause", name]), "unpause runner container"
        )

    def _remove_frozen(self, name: str) -> None:
        self._require_success(
            self._run(["docker", "rm", "-f", name]),
            "remove frozen runner container",
        )

    def _compensate_uncertain_pause(
        self, name: str, pause_error: RunnerOperationError
    ) -> None:
        try:
            state = self._inspect_state(
                name,
                "Paused",
                "inspect uncertain paused runner container",
                allow_absent=True,
            )
        except RunnerOperationError as inspect_error:
            try:
                self._unpause(name)
            except RunnerOperationError as unpause_error:
                raise RunnerOperationError(
                    "{}; pause recovery failed: {}; {}".format(
                        pause_error, inspect_error, unpause_error
                    )
                ) from pause_error
            raise pause_error
        if state:
            try:
                self._unpause(name)
            except RunnerOperationError as unpause_error:
                raise RunnerOperationError(
                    "{}; pause recovery failed: {}".format(pause_error, unpause_error)
                ) from pause_error
        raise pause_error

    def _unpause_all(self, names: List[str]) -> List[str]:
        errors = []
        for name in list(names):
            try:
                self._unpause(name)
            except RunnerOperationError as error:
                errors.append("{}: {}".format(name, error))
            else:
                names.remove(name)
        return errors

    def _raise_after_unpause(
        self, error: RunnerOperationError, paused: List[str]
    ) -> None:
        cleanup_errors = self._unpause_all(paused)
        if cleanup_errors:
            raise RunnerOperationError(
                "{}; unpause failed: {}".format(error, "; ".join(cleanup_errors))
            ) from error
        raise error

    def _remove_for_prepare(self, name: str) -> None:
        result = self._run(["docker", "rm", "-f", name])
        if result.returncode == 0 or self._is_exact_absence(result, name):
            return
        self._require_success(result, "remove runner container")

    def _inspect_state(
        self,
        name: str,
        field: str,
        operation: str,
        allow_absent: bool,
    ) -> Optional[bool]:
        result = self._run(
            ["docker", "inspect", "--format", "{{{{.State.{}}}}}".format(field), name]
        )
        if allow_absent and self._is_exact_absence(result, name):
            return None
        self._require_success(result, operation)
        state = result.stdout.strip().lower()
        if state == "true":
            return True
        if state == "false":
            return False
        raise RunnerOperationError("{} returned unknown state".format(operation))

    def _cleanup_runner_containers(self) -> List[str]:
        errors = []
        for slot in self.config.runner_slots:
            try:
                self._remove_for_prepare(slot.name)
            except RunnerOperationError as error:
                errors.append("{}: {}".format(slot.name, error))
        return errors

    @staticmethod
    def _is_exact_absence(result: CommandResult, name: str) -> bool:
        return (
            result.returncode == 1
            and result.stdout == ""
            and result.stderr.rstrip("\r\n")
            == "Error response from daemon: No such container: {}".format(name)
        )

    def _create_argv(self, slot: RunnerSlot, vnpu: Vnpu) -> List[str]:
        host_device = "/dev/vdavinci{}".format(vnpu.id)
        argv = [
            "docker",
            "create",
            "--name",
            slot.name,
            "--network",
            "bridge",
            "--restart=no",
            "--ipc=host",
            "--workdir",
            "/actions-runner",
            "--device",
            "{}:/dev/davinci0".format(host_device),
        ]
        for device in self.config.devices:
            argv.extend(["--device", device])
        argv.extend(["-v", "{}:/actions-runner".format(slot.work_dir)])
        for mount in self.config.mounts:
            argv.extend(["-v", mount])
        for key, value in self.config.environment:
            argv.extend(["-e", "{}={}".format(key, value)])
        argv.extend(
            [
                "-e",
                "A2_E2E_HOST_DEVICE={}".format(host_device),
                "--entrypoint",
                "/bin/bash",
                slot.image,
                "-lc",
                RUNNER_COMMAND,
            ]
        )
        return argv

    def _start(self, names: Sequence[str], allow_start: Callable[[], bool]) -> None:
        if not names:
            return
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self._start_one, name, allow_start) for name in names
            ]
            results = [future.result() for future in futures]
        for result in results:
            self._require_success(result, "start runner container")

    def _start_one(self, name: str, allow_start: Callable[[], bool]) -> CommandResult:
        try:
            allowed = allow_start()
        except Exception as error:
            raise RunnerOperationError("runner start guard failed") from error
        if not allowed:
            raise RunnerOperationError("runner start deadline expired")
        return self._run(["docker", "start", name])

    def _run(self, argv: Sequence[str]) -> CommandResult:
        try:
            return self.command_runner.run(argv, self.command_timeout)
        except (OSError, subprocess.SubprocessError) as error:
            raise RunnerOperationError("runner command failed") from error

    @staticmethod
    def _require_success(result: CommandResult, operation: str) -> None:
        if result.returncode != 0:
            raise RunnerOperationError(
                "{} failed with exit code {}".format(operation, result.returncode)
            )
