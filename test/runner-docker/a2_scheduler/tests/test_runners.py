import threading
from pathlib import Path

import pytest
from a2_scheduler.host import CommandResult
from a2_scheduler.models import Vnpu, load_config
from a2_scheduler.runners import RunnerManager, RunnerOperationError

CONFIG_EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.json"


def ok(stdout=""):
    return CommandResult(0, stdout, "")


class RecordingRunner:
    def __init__(self, running=None, processes=None):
        self.commands = []
        self.running = dict(running or {})
        self.processes = dict(processes or {})
        self._lock = threading.Lock()

    def run(self, argv, timeout):
        argv = list(argv)
        with self._lock:
            self.commands.append((argv, timeout))
        if argv[:2] == ["docker", "inspect"]:
            return ok("true\n" if self.running.get(argv[-1], False) else "false\n")
        if argv[:2] == ["docker", "exec"]:
            process_name = argv[-1]
            present = process_name in self.processes.get(argv[2], set())
            return CommandResult(
                0 if present else 1, process_name if present else "", ""
            )
        return ok()


class BlockingRecordingRunner(RecordingRunner):
    def __init__(self):
        super().__init__()
        self._barrier = threading.Barrier(4)
        self._active = 0
        self.max_concurrency = 0

    def run(self, argv, timeout):
        if list(argv)[:2] != ["docker", "start"]:
            return super().run(argv, timeout)
        with self._lock:
            self.commands.append((list(argv), timeout))
            self._active += 1
            self.max_concurrency = max(self.max_concurrency, self._active)
        try:
            self._barrier.wait(timeout=2)
        finally:
            with self._lock:
                self._active -= 1
        return ok()


@pytest.fixture
def config():
    return load_config(CONFIG_EXAMPLE)


def four_vnpus():
    return [
        Vnpu(313, "vir05_1c_16g"),
        Vnpu(310, "vir05_1c_16g"),
        Vnpu(312, "vir05_1c_16g"),
        Vnpu(311, "vir05_1c_16g"),
    ]


def docker_commands(runner, operation):
    return [argv for argv, _ in runner.commands if argv[:2] == ["docker", operation]]


def test_prepare_maps_sorted_discovered_vnpus_to_four_slots(config):
    runner = RecordingRunner()
    manager = RunnerManager(config, runner)

    manager.prepare(four_vnpus())

    creates = docker_commands(runner, "create")
    assert len(creates) == 4
    for slot, vnpu, command in zip(
        config.runner_slots, sorted(four_vnpus(), key=lambda item: item.id), creates
    ):
        host_device = "/dev/vdavinci{}".format(vnpu.id)
        assert command[command.index("--name") + 1] == slot.name
        assert ["--network", "bridge"] == command[
            command.index("--network") : command.index("--network") + 2
        ]
        assert "--restart=no" in command
        assert "{}:/dev/davinci0".format(host_device) in command
        assert "A2_E2E_HOST_DEVICE={}".format(host_device) in command
        assert "{}:/actions-runner".format(slot.work_dir) in command
        assert command[command.index("--entrypoint") + 1] == "/bin/bash"
        assert command[-3:] == [
            slot.image,
            "-lc",
            "cd /actions-runner\n"
            'git config --global http.proxy "$http_proxy"\n'
            'git config --global https.proxy "$https_proxy"\n'
            "exec ./run.sh --once",
        ]


def test_prepare_preserves_configured_environment_devices_and_mounts(config):
    runner = RecordingRunner()

    RunnerManager(config, runner).prepare(four_vnpus())

    for command in docker_commands(runner, "create"):
        for key, value in config.environment:
            assert "{}={}".format(key, value) in command
        for device in config.devices:
            assert device in command
        for mount in config.mounts:
            assert mount in command


@pytest.mark.parametrize(
    "vnpus",
    [
        [Vnpu(index, "vir05_1c_16g") for index in range(3)],
        [
            Vnpu(310, "vir05_1c_16g"),
            Vnpu(310, "vir05_1c_16g"),
            Vnpu(312, "vir05_1c_16g"),
            Vnpu(313, "vir05_1c_16g"),
        ],
    ],
)
def test_prepare_rejects_invalid_vnpus_before_docker_mutation(config, vnpus):
    runner = RecordingRunner()

    with pytest.raises(ValueError):
        RunnerManager(config, runner).prepare(vnpus)

    assert runner.commands == []


def test_start_all_launches_four_containers_concurrently(config):
    runner = BlockingRecordingRunner()

    RunnerManager(config, runner).start_all()

    assert runner.max_concurrency == 4
    assert sorted(
        command[-1] for command in docker_commands(runner, "start")
    ) == sorted(slot.name for slot in config.runner_slots)


def test_restart_exited_starts_only_non_running_containers(config):
    running = {
        config.runner_slots[0].name: True,
        config.runner_slots[1].name: False,
        config.runner_slots[2].name: True,
        config.runner_slots[3].name: False,
    }
    runner = RecordingRunner(running=running)

    RunnerManager(config, runner).restart_exited()

    assert [command[-1] for command in docker_commands(runner, "start")] == [
        config.runner_slots[1].name,
        config.runner_slots[3].name,
    ]


def test_active_workers_counts_runner_worker_processes(config):
    processes = {
        config.runner_slots[0].name: {"Runner.Listener", "Runner.Worker"},
        config.runner_slots[1].name: {"Runner.Listener"},
        config.runner_slots[2].name: {"Runner.Worker"},
    }

    count = RunnerManager(config, RecordingRunner(processes=processes)).active_workers()

    assert count == 2


def test_stop_idle_never_stops_a_container_with_runner_worker(config):
    processes = {
        config.runner_slots[0].name: {"Runner.Listener", "Runner.Worker"},
        config.runner_slots[1].name: {"Runner.Listener"},
        config.runner_slots[2].name: set(),
        config.runner_slots[3].name: {"Runner.Worker"},
    }
    runner = RecordingRunner(processes=processes)

    RunnerManager(config, runner).stop_idle()

    assert [command[-1] for command in docker_commands(runner, "stop")] == [
        config.runner_slots[1].name
    ]


def test_stop_idle_fails_closed_when_worker_probe_errors(config):
    class FailedWorkerProbeRunner(RecordingRunner):
        def run(self, argv, timeout):
            if list(argv)[-1] == "Runner.Worker":
                with self._lock:
                    self.commands.append((list(argv), timeout))
                return CommandResult(2, "", "pgrep failed")
            return super().run(argv, timeout)

    runner = FailedWorkerProbeRunner(
        processes={config.runner_slots[0].name: {"Runner.Listener"}}
    )

    with pytest.raises(RunnerOperationError, match="inspect runner process"):
        RunnerManager(config, runner).stop_idle()

    assert docker_commands(runner, "stop") == []


def test_stop_all_if_idle_leaves_every_container_running_when_worker_exists(config):
    processes = {
        config.runner_slots[0].name: {"Runner.Listener"},
        config.runner_slots[1].name: {"Runner.Listener", "Runner.Worker"},
        config.runner_slots[2].name: {"Runner.Listener"},
        config.runner_slots[3].name: {"Runner.Listener"},
    }
    runner = RecordingRunner(processes=processes)

    stopped = RunnerManager(config, runner).stop_all_if_idle()

    assert stopped is False
    assert docker_commands(runner, "stop") == []


def test_stop_all_if_idle_stops_all_listeners(config):
    processes = {slot.name: {"Runner.Listener"} for slot in config.runner_slots}
    runner = RecordingRunner(processes=processes)

    stopped = RunnerManager(config, runner).stop_all_if_idle()

    assert stopped is True
    assert [command[-1] for command in docker_commands(runner, "stop")] == [
        slot.name for slot in config.runner_slots
    ]
