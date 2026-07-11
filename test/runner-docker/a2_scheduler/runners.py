import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Sequence

from .host import CommandResult, CommandRunner
from .models import SchedulerConfig, Vnpu

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

        for slot, vnpu in zip(self.config.runner_slots, discovered):
            self._require_success(
                self._run(["docker", "rm", "-f", slot.name]),
                "remove runner container",
            )
            self._require_success(
                self._run(self._create_argv(slot, vnpu)),
                "create runner container",
            )

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
            self._has_process(slot.name, "Runner.Worker")
            for slot in self.config.runner_slots
        )

    def stop_idle(self) -> bool:
        all_idle = True
        for slot in self.config.runner_slots:
            if not self._has_process(slot.name, "Runner.Listener"):
                continue
            if self._has_process(slot.name, "Runner.Worker"):
                all_idle = False
                continue
            self._require_success(
                self._run(["docker", "stop", slot.name]),
                "stop idle runner container",
            )
        return all_idle

    def stop_all_if_idle(self) -> bool:
        if self.active_workers():
            return False
        return self.stop_idle()

    def _create_argv(self, slot, vnpu: Vnpu) -> List[str]:
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

    def _has_process(self, name: str, process_name: str) -> bool:
        result = self._run(["docker", "exec", name, "pgrep", "-af", process_name])
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        self._require_success(result, "inspect runner process")
        return False

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
