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
    def __init__(
        self,
        running=None,
        processes=None,
        race_on_pause=(),
        top_status=None,
        fail_once=None,
    ):
        self.commands = []
        self.running = dict(running or {})
        self.processes = dict(processes or {})
        self.race_on_pause = set(race_on_pause)
        self.top_status = dict(top_status or {})
        self.fail_once = dict(fail_once or {})
        self.failed = set()
        self.paused = set()
        self._lock = threading.Lock()

    def run(self, argv, timeout):
        argv = list(argv)
        with self._lock:
            self.commands.append((argv, timeout))
        operation = argv[1] if len(argv) > 1 else ""
        name = argv[2] if operation == "top" else argv[-1]
        failure_key = (operation, name)
        if failure_key in self.fail_once and failure_key not in self.failed:
            self.failed.add(failure_key)
            return CommandResult(self.fail_once[failure_key], "", "injected failure")
        if argv[:2] == ["docker", "inspect"]:
            return ok("true\n" if self.running.get(argv[-1], False) else "false\n")
        if argv[:2] == ["docker", "pause"]:
            name = argv[-1]
            self.paused.add(name)
            if name in self.race_on_pause:
                self.processes.setdefault(name, set()).add("Runner.Worker --job 99")
            return ok(name)
        if argv[:2] == ["docker", "top"]:
            name = argv[2]
            status = self.top_status.get(name, 0)
            if status:
                return CommandResult(status, "", "top failed")
            output = "COMMAND\n" + "\n".join(sorted(self.processes.get(name, set())))
            return ok(output)
        if argv[:2] == ["docker", "unpause"]:
            if argv[-1] not in self.paused:
                return CommandResult(1, "", "container is not paused")
            self.paused.remove(argv[-1])
            return ok(argv[-1])
        if argv[:3] == ["docker", "rm", "-f"]:
            name = argv[-1]
            self.paused.discard(name)
            self.running[name] = False
            return ok(name)
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


class StatefulPrepareRunner:
    def __init__(
        self,
        names,
        containers=None,
        fail_create_once=None,
        fail_remove_once=None,
        cleanup_remove_failure=None,
    ):
        self.names = tuple(names)
        self.containers = set(self.names if containers is None else containers)
        self.fail_create_once = fail_create_once
        self.fail_remove_once = fail_remove_once
        self.cleanup_remove_failure = cleanup_remove_failure
        self.commands = []
        self._create_failed = False
        self._remove_failed = False
        self._cleanup_started = False
        self._cleanup_failed = False

    def run(self, argv, timeout):
        argv = list(argv)
        self.commands.append((argv, timeout))
        if argv[:2] == ["docker", "create"]:
            name = argv[argv.index("--name") + 1]
            if name == self.fail_create_once and not self._create_failed:
                self._create_failed = True
                self._cleanup_started = True
                return CommandResult(125, "", "create failed")
            self.containers.add(name)
            return ok(name)
        if argv[:3] == ["docker", "rm", "-f"]:
            name = argv[-1]
            if name == self.fail_remove_once and not self._remove_failed:
                self._remove_failed = True
                self._cleanup_started = True
                return CommandResult(1, "", "permission denied")
            if (
                self._cleanup_started
                and name == self.cleanup_remove_failure
                and not self._cleanup_failed
            ):
                self._cleanup_failed = True
                return CommandResult(125, "", "cleanup failed")
            if name not in self.containers:
                return CommandResult(
                    1,
                    "",
                    "Error response from daemon: No such container: {}".format(name),
                )
            self.containers.remove(name)
            return ok(name)
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


def test_prepare_accepts_only_exact_no_such_container_as_idempotent(config):
    names = [slot.name for slot in config.runner_slots]
    runner = StatefulPrepareRunner(names, containers=())

    RunnerManager(config, runner).prepare(four_vnpus())

    assert runner.containers == set(names)


def test_prepare_rejects_other_status_one_and_cleans_all_slots(config):
    names = [slot.name for slot in config.runner_slots]
    runner = StatefulPrepareRunner(names, fail_remove_once=names[0])

    with pytest.raises(RunnerOperationError, match="remove runner container"):
        RunnerManager(config, runner).prepare(four_vnpus())

    assert runner.containers == set()
    assert docker_commands(runner, "create") == []


def test_prepare_failure_cleans_partial_set_and_retry_recreates_all(config):
    names = [slot.name for slot in config.runner_slots]
    runner = StatefulPrepareRunner(names, fail_create_once=names[1])
    manager = RunnerManager(config, runner)

    with pytest.raises(RunnerOperationError, match="create runner container"):
        manager.prepare(four_vnpus())

    assert runner.containers == set()

    manager.prepare(four_vnpus())

    assert runner.containers == set(names)


def test_prepare_aggregates_cleanup_errors_and_attempts_every_slot(config):
    names = [slot.name for slot in config.runner_slots]
    runner = StatefulPrepareRunner(
        names,
        fail_create_once=names[1],
        cleanup_remove_failure=names[2],
    )

    with pytest.raises(RunnerOperationError) as caught:
        RunnerManager(config, runner).prepare(four_vnpus())

    assert "create runner container" in str(caught.value)
    assert "cleanup" in str(caught.value)
    failed_create_index = next(
        index
        for index, (command, _) in enumerate(runner.commands)
        if command[:2] == ["docker", "create"]
        and command[command.index("--name") + 1] == names[1]
    )
    cleanup_names = [
        command[-1]
        for command, _ in runner.commands[failed_create_index + 1 :]
        if command[:3] == ["docker", "rm", "-f"]
    ]
    assert cleanup_names == names


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
    running = {slot.name: True for slot in config.runner_slots}
    processes = {
        config.runner_slots[0].name: {"Runner.Listener", "Runner.Worker --job 1"},
        config.runner_slots[1].name: {"Runner.Listener"},
        config.runner_slots[2].name: {"Runner.Worker --job 2"},
        config.runner_slots[3].name: {"Runner.Listener"},
    }

    runner = RecordingRunner(running=running, processes=processes)
    count = RunnerManager(config, runner).active_workers()

    assert count == 2
    assert docker_commands(runner, "exec") == []
    assert all(
        command[-2:] == ["-eo", "args"] for command in docker_commands(runner, "top")
    )


@pytest.mark.parametrize("status", [1, 2])
def test_active_workers_fails_closed_on_any_nonzero_top_status(config, status):
    name = config.runner_slots[0].name
    runner = RecordingRunner(running={name: True}, top_status={name: status})

    with pytest.raises(RunnerOperationError, match="inspect runner processes"):
        RunnerManager(config, runner).active_workers()


def test_stop_idle_never_stops_a_container_with_runner_worker(config):
    running = {slot.name: True for slot in config.runner_slots}
    processes = {
        config.runner_slots[0].name: {"Runner.Listener", "Runner.Worker --job 1"},
        config.runner_slots[1].name: {"Runner.Listener"},
        config.runner_slots[2].name: {"Runner.Listener"},
        config.runner_slots[3].name: {"Runner.Worker --job 2"},
    }
    runner = RecordingRunner(running=running, processes=processes)

    RunnerManager(config, runner).stop_idle()

    assert [command[-1] for command in docker_commands(runner, "rm")] == [
        config.runner_slots[1].name,
        config.runner_slots[2].name,
    ]
    assert [command[-1] for command in docker_commands(runner, "unpause")] == [
        config.runner_slots[0].name,
        config.runner_slots[3].name,
    ]
    assert runner.paused == set()


def test_stop_idle_sees_worker_that_appears_before_frozen_top(config):
    name = config.runner_slots[0].name
    runner = RecordingRunner(
        running={name: True},
        processes={name: {"Runner.Listener"}},
        race_on_pause={name},
    )

    stopped = RunnerManager(config, runner).stop_idle()

    assert stopped is False
    assert docker_commands(runner, "rm") == []
    assert [command[-1] for command in docker_commands(runner, "unpause")] == [name]
    assert runner.paused == set()


@pytest.mark.parametrize("status", [1, 2])
def test_stop_idle_unpauses_and_fails_closed_on_nonzero_top(config, status):
    name = config.runner_slots[0].name
    runner = RecordingRunner(
        running={name: True},
        processes={name: {"Runner.Listener"}},
        top_status={name: status},
    )

    with pytest.raises(RunnerOperationError, match="inspect runner processes"):
        RunnerManager(config, runner).stop_idle()

    assert docker_commands(runner, "rm") == []
    assert [command[-1] for command in docker_commands(runner, "unpause")] == [name]
    assert runner.paused == set()


def test_stop_all_if_idle_leaves_every_container_running_when_worker_exists(config):
    running = {slot.name: True for slot in config.runner_slots}
    processes = {
        config.runner_slots[0].name: {"Runner.Listener"},
        config.runner_slots[1].name: {"Runner.Listener", "Runner.Worker --job 1"},
        config.runner_slots[2].name: {"Runner.Listener"},
        config.runner_slots[3].name: {"Runner.Listener"},
    }
    runner = RecordingRunner(running=running, processes=processes)

    stopped = RunnerManager(config, runner).stop_all_if_idle()

    assert stopped is False
    assert docker_commands(runner, "rm") == []
    assert runner.paused == set()
    commands = [command for command, _ in runner.commands]
    last_pause = max(
        index for index, command in enumerate(commands) if command[1] == "pause"
    )
    first_top = min(
        index for index, command in enumerate(commands) if command[1] == "top"
    )
    assert last_pause < first_top


def test_stop_all_if_idle_unpauses_all_when_worker_appears_before_top(config):
    running = {slot.name: True for slot in config.runner_slots}
    processes = {slot.name: {"Runner.Listener"} for slot in config.runner_slots}
    raced = config.runner_slots[2].name
    runner = RecordingRunner(
        running=running,
        processes=processes,
        race_on_pause={raced},
    )

    stopped = RunnerManager(config, runner).stop_all_if_idle()

    assert stopped is False
    assert docker_commands(runner, "rm") == []
    assert sorted(
        command[-1] for command in docker_commands(runner, "unpause")
    ) == sorted(running)
    assert runner.paused == set()


def test_stop_all_if_idle_unpauses_all_and_removes_none_on_unknown_top(config):
    running = {slot.name: True for slot in config.runner_slots}
    processes = {slot.name: {"Runner.Listener"} for slot in config.runner_slots}
    failed = config.runner_slots[2].name
    runner = RecordingRunner(
        running=running,
        processes=processes,
        top_status={failed: 1},
    )

    with pytest.raises(RunnerOperationError, match="inspect runner processes"):
        RunnerManager(config, runner).stop_all_if_idle()

    assert docker_commands(runner, "rm") == []
    assert runner.paused == set()


def test_stop_all_unpause_failure_retries_only_container_still_paused(config):
    running = {slot.name: True for slot in config.runner_slots}
    processes = {slot.name: {"Runner.Listener"} for slot in config.runner_slots}
    worker = config.runner_slots[0].name
    failed_unpause = config.runner_slots[1].name
    processes[worker].add("Runner.Worker --job 1")
    runner = RecordingRunner(
        running=running,
        processes=processes,
        fail_once={("unpause", failed_unpause): 125},
    )

    with pytest.raises(RunnerOperationError, match="unpause runner containers"):
        RunnerManager(config, runner).stop_all_if_idle()

    unpaused = [command[-1] for command in docker_commands(runner, "unpause")]
    assert unpaused.count(failed_unpause) == 2
    assert all(unpaused.count(name) == 1 for name in running if name != failed_unpause)
    assert runner.paused == set()
    assert docker_commands(runner, "rm") == []


def test_stop_all_if_idle_stops_all_listeners(config):
    processes = {slot.name: {"Runner.Listener"} for slot in config.runner_slots}
    running = {slot.name: True for slot in config.runner_slots}
    runner = RecordingRunner(running=running, processes=processes)

    stopped = RunnerManager(config, runner).stop_all_if_idle()

    assert stopped is True
    assert [command[-1] for command in docker_commands(runner, "rm")] == [
        slot.name for slot in config.runner_slots
    ]
    assert docker_commands(runner, "unpause") == []
