import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Sequence

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

    def start_all(self) -> None:
        self._start([slot.name for slot in self.config.runner_slots])

    def restart_exited(self) -> None:
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
        self._start(exited)

    def active_workers(self) -> int:
        return sum(
            self._top_has_worker(name) for name in self._running_container_names()
        )

    def stop_idle(self) -> bool:
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

    def _running_container_names(self) -> List[str]:
        names = []
        for slot in self.config.runner_slots:
            result = self._run(
                ["docker", "inspect", "--format", "{{.State.Running}}", slot.name]
            )
            self._require_success(result, "inspect runner container")
            state = result.stdout.strip().lower()
            if state == "true":
                names.append(slot.name)
            elif state != "false":
                raise RunnerOperationError(
                    "inspect runner container returned unknown state"
                )
        return names

    def _top_has_worker(self, name: str) -> bool:
        result = self._run(["docker", "top", name, "-eo", "args"])
        self._require_success(result, "inspect runner processes")
        return any("Runner.Worker" in line for line in result.stdout.splitlines()[1:])

    def _pause(self, name: str) -> None:
        self._require_success(
            self._run(["docker", "pause", name]), "pause runner container"
        )

    def _unpause(self, name: str) -> None:
        self._require_success(
            self._run(["docker", "unpause", name]), "unpause runner container"
        )

    def _remove_frozen(self, name: str) -> None:
        self._require_success(
            self._run(["docker", "rm", "-f", name]),
            "remove frozen runner container",
        )

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

    def _start(self, names: Sequence[str]) -> None:
        if not names:
            return
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self._run, ["docker", "start", name]) for name in names
            ]
            results = [future.result() for future in futures]
        for result in results:
            self._require_success(result, "start runner container")

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
