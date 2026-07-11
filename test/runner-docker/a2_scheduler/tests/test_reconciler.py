from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from a2_scheduler.github_client import GitHubUnavailable
from a2_scheduler.host import NpuOperationError
from a2_scheduler.models import Job, Phase, SchedulerState, Vnpu
from a2_scheduler.runners import RunnerOperationError


class FakeClock:
    def __init__(self, minute):
        self.current = datetime(2026, 7, 11, 10, minute)
        self.sleeps = []

    def __call__(self):
        return self.current

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.current += timedelta(seconds=seconds)


class FakeGitHub:
    def __init__(self, observations, events):
        self.observations = list(observations)
        self.events = events
        self.calls = 0

    def list_target_jobs(self):
        self.events.append("github")
        index = min(self.calls, len(self.observations) - 1)
        self.calls += 1
        observation = self.observations[index]
        if isinstance(observation, Exception):
            raise observation
        return list(observation)


class FakeSignal:
    def __init__(self):
        self.pids = []

    def __call__(self, pid):
        self.pids.append(pid)


class FakeNpu:
    def __init__(self, inventory, card_pids, signal, events):
        self.inventory = list(inventory)
        self.card_pids = list(card_pids)
        self.signal = signal
        self.events = events
        self.created_count = 0
        self.destroyed_ids = []
        self.restricted_modes = []
        self.restored_modes = []
        self.fail = None
        self.partial_create = ()

    @property
    def destroyed_count(self):
        return len(self.destroyed_ids)

    def list_vnpus(self):
        self.events.append("list_vnpus")
        return list(self.inventory)

    def evict_card_pids(self):
        self.events.append("evict")
        if self.fail == "evict":
            raise NpuOperationError("eviction failed")
        for pid in self.card_pids:
            self.signal(pid)
        self.card_pids = []

    def create_vnpus(self, count, template):
        self.events.append("create")
        if self.fail == "create":
            created = [Vnpu(vnpu_id, template) for vnpu_id in self.partial_create]
            self.inventory.extend(created)
            self.created_count += len(created)
            raise NpuOperationError("creation failed")
        start = max([vnpu.id for vnpu in self.inventory] + [300]) + 7
        created = [Vnpu(start + index * 3, template) for index in range(count)]
        self.inventory.extend(created)
        self.created_count += len(created)
        return created

    def destroy_vnpus(self, vnpu_ids):
        ids = tuple(vnpu_ids)
        self.events.append("destroy")
        if self.fail == "destroy":
            raise NpuOperationError("destroy failed")
        self.destroyed_ids.extend(ids)
        self.inventory = [vnpu for vnpu in self.inventory if vnpu.id not in ids]

    def restrict_device(self):
        self.events.append("restrict")
        if self.fail == "restrict":
            raise NpuOperationError("restrict failed")
        mode = 0o20660
        self.restricted_modes.append(mode)
        return mode

    def restore_device(self, mode):
        self.events.append("restore")
        self.restored_modes.append(mode)
        if self.fail == "restore":
            raise NpuOperationError("restore failed")


class FakeRunners:
    def __init__(self, workers, events, minute):
        self.worker_observations = list(workers)
        self.events = events
        self.minute = minute
        self.concurrent_starts = 0
        self.started_after_cutoff = 0
        self.prepared_ids = ()
        self.restart_calls = 0
        self.stop_idle_calls = 0
        self.stop_all_calls = 0
        self.stop_all_results = [True]
        self.fail = None

    def recover_paused(self):
        self.events.append("recover")
        if self.fail == "recover":
            raise RunnerOperationError("recovery failed")

    def prepare(self, vnpus):
        self.events.append("prepare")
        self.prepared_ids = tuple(vnpu.id for vnpu in vnpus)
        if self.fail == "prepare":
            raise RunnerOperationError("prepare failed")

    def start_all(self):
        self.events.append("start")
        if self.minute >= 20:
            self.started_after_cutoff += 4
        if self.fail == "start":
            raise RunnerOperationError("start failed")
        self.concurrent_starts = 4

    def restart_exited(self):
        self.events.append("restart")
        if self.minute >= 20:
            self.started_after_cutoff += 1
        self.restart_calls += 1

    def active_workers(self):
        self.events.append("workers")
        index = min(
            sum(event == "workers" for event in self.events) - 1,
            len(self.worker_observations) - 1,
        )
        return self.worker_observations[index]

    def stop_idle(self):
        self.events.append("stop_idle")
        self.stop_idle_calls += 1
        return False

    def stop_all_if_idle(self):
        self.events.append("stop_all")
        self.stop_all_calls += 1
        result = self.stop_all_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class SchedulerFixture:
    def __init__(
        self,
        phase=Phase.WHOLE,
        minute=8,
        queued=0,
        in_progress=0,
        workers=0,
        card_pids=(),
        github_error=False,
        vnpu_ids=(),
        inventory=None,
        original_mode=None,
        observations=None,
    ):
        self.minute = minute
        self.events = []
        self.clock = FakeClock(minute)
        self.signal = FakeSignal()
        initial_jobs = self.jobs(queued, in_progress)
        if observations is None:
            observations = [
                GitHubUnavailable("unavailable") if github_error else initial_jobs
            ]
        self.github = FakeGitHub(observations, self.events)
        if inventory is None:
            inventory = [Vnpu(vnpu_id, "vir05_1c_16g") for vnpu_id in vnpu_ids]
        self.npu = FakeNpu(inventory, card_pids, self.signal, self.events)
        worker_observations = workers if isinstance(workers, list) else [workers]
        self.runners = FakeRunners(worker_observations, self.events, minute)
        self.config = SimpleNamespace(
            window_start_minute=0,
            window_end_minute=20,
            vnpu_count=4,
            vnpu_template="vir05_1c_16g",
        )
        self.state = SchedulerState(
            phase=phase,
            vnpu_ids=tuple(vnpu_ids),
            original_device_mode=original_mode,
        )

    @staticmethod
    def jobs(queued, in_progress=0):
        jobs = [Job(index, "queued") for index in range(queued)]
        jobs.extend(Job(100 + index, "in_progress") for index in range(in_progress))
        return jobs

    def reconcile(self):
        from a2_scheduler.reconciler import Reconciler

        reconciler = Reconciler(
            self.config,
            self.state,
            self.github,
            self.npu,
            self.runners,
            clock=self.clock,
            sleep=self.clock.sleep,
        )
        self.state = reconciler.reconcile(self.clock())
        return self.state

    @property
    def killed_pids(self):
        return self.signal.pids


def test_whole_with_queue_inside_window_allocates_and_starts_four():
    fixture = SchedulerFixture(phase=Phase.WHOLE, minute=8, queued=4, card_pids=[123])

    state = fixture.reconcile()

    assert state.phase == Phase.SERVING
    assert state.original_device_mode == 0o20660
    assert state.vnpu_ids == (307, 310, 313, 316)
    assert fixture.killed_pids == [123]
    assert fixture.npu.created_count == 4
    assert fixture.runners.prepared_ids == state.vnpu_ids
    assert fixture.runners.concurrent_starts == 4
    assert fixture.events == [
        "recover",
        "github",
        "restrict",
        "evict",
        "create",
        "list_vnpus",
        "prepare",
        "start",
    ]


def test_whole_with_api_failure_does_not_preempt():
    fixture = SchedulerFixture(
        phase=Phase.WHOLE,
        minute=8,
        github_error=True,
        card_pids=[123],
    )

    state = fixture.reconcile()

    assert state.phase == Phase.WHOLE
    assert fixture.killed_pids == []
    assert fixture.events == ["recover", "github"]


def test_cutoff_leaves_queued_jobs_for_next_window():
    fixture = SchedulerFixture(
        phase=Phase.SERVING,
        minute=20,
        queued=3,
        workers=0,
        vnpu_ids=(41, 44, 47, 50),
        original_mode=0o20660,
    )

    state = fixture.reconcile()

    assert state.phase == Phase.WHOLE
    assert fixture.runners.started_after_cutoff == 0
    assert fixture.npu.destroyed_count == 4


def test_early_restore_requires_two_complete_empty_observations():
    fixture = SchedulerFixture(
        phase=Phase.SERVING,
        observations=[[], []],
        workers=[0, 0],
        vnpu_ids=(71, 74, 77, 80),
        original_mode=0o20640,
    )

    state = fixture.reconcile()

    assert state.phase == Phase.WHOLE
    assert fixture.github.calls == 2
    assert fixture.clock.sleeps == [5]
    assert fixture.runners.stop_all_calls == 1
    assert fixture.npu.restored_modes == [0o20640]


@pytest.mark.parametrize(
    "second_jobs,second_workers",
    [([Job(9, "queued")], 0), ([Job(10, "in_progress")], 0), ([], 1)],
)
def test_early_restore_is_cancelled_when_second_observation_is_not_empty(
    second_jobs, second_workers
):
    fixture = SchedulerFixture(
        phase=Phase.SERVING,
        observations=[[], second_jobs],
        workers=[0, second_workers],
        vnpu_ids=(71, 74, 77, 80),
        original_mode=0o20640,
    )

    state = fixture.reconcile()

    assert state.phase == Phase.SERVING
    assert fixture.npu.destroyed_count == 0


def test_cutoff_drains_idle_listeners_but_keeps_active_worker_allocation():
    fixture = SchedulerFixture(
        phase=Phase.SERVING,
        minute=20,
        queued=6,
        workers=1,
        vnpu_ids=(11, 12, 13, 14),
        original_mode=0o20660,
    )

    state = fixture.reconcile()

    assert state.phase == Phase.DRAINING
    assert fixture.runners.stop_idle_calls == 1
    assert fixture.runners.restart_calls == 0
    assert fixture.npu.destroyed_count == 0


def test_cutoff_known_in_progress_job_blocks_restore():
    fixture = SchedulerFixture(
        phase=Phase.DRAINING,
        minute=37,
        in_progress=1,
        workers=0,
        vnpu_ids=(11, 12, 13, 14),
        original_mode=0o20660,
    )

    state = fixture.reconcile()

    assert state.phase == Phase.DRAINING
    assert fixture.npu.destroyed_count == 0


def test_api_failure_while_serving_retains_allocation():
    fixture = SchedulerFixture(
        phase=Phase.SERVING,
        minute=9,
        github_error=True,
        workers=0,
        vnpu_ids=(11, 12, 13, 14),
        original_mode=0o20660,
    )

    state = fixture.reconcile()

    assert state.phase == Phase.SERVING
    assert fixture.runners.restart_calls == 0
    assert fixture.npu.destroyed_count == 0


def test_api_failure_after_cutoff_uses_atomic_local_idle_evidence():
    fixture = SchedulerFixture(
        phase=Phase.SERVING,
        minute=20,
        github_error=True,
        workers=0,
        vnpu_ids=(21, 24, 27, 30),
        original_mode=0o20660,
    )

    state = fixture.reconcile()

    assert state.phase == Phase.WHOLE
    assert fixture.runners.stop_all_calls == 1
    assert fixture.npu.destroyed_ids == [21, 24, 27, 30]


def test_scheduler_restart_uses_actual_vnpu_ids():
    actual = [Vnpu(212, "vir05_1c_16g"), Vnpu(219, "vir05_1c_16g")]
    actual.extend([Vnpu(225, "vir05_1c_16g"), Vnpu(231, "vir05_1c_16g")])
    fixture = SchedulerFixture(
        phase=Phase.SPLITTING,
        minute=8,
        queued=1,
        inventory=actual,
        original_mode=0o20660,
    )

    state = fixture.reconcile()

    assert state.phase == Phase.SERVING
    assert state.vnpu_ids == (212, 219, 225, 231)
    assert fixture.runners.prepared_ids == state.vnpu_ids
    assert fixture.npu.created_count == 0


def test_serving_restart_refreshes_stale_ids_when_github_is_unknown():
    actual = [Vnpu(212, "vir05_1c_16g"), Vnpu(219, "vir05_1c_16g")]
    actual.extend([Vnpu(225, "vir05_1c_16g"), Vnpu(231, "vir05_1c_16g")])
    fixture = SchedulerFixture(
        phase=Phase.SERVING,
        minute=8,
        github_error=True,
        vnpu_ids=(1, 2, 3, 4),
        inventory=actual,
        original_mode=0o20660,
    )

    state = fixture.reconcile()

    assert state.phase == Phase.SERVING
    assert state.vnpu_ids == (212, 219, 225, 231)
    assert fixture.npu.destroyed_count == 0


def test_serving_with_missing_vnpus_enters_error_without_lifecycle_progress():
    fixture = SchedulerFixture(
        phase=Phase.SERVING,
        minute=8,
        github_error=True,
        vnpu_ids=(1, 2, 3, 4),
        inventory=[],
        original_mode=0o20660,
    )

    state = fixture.reconcile()

    assert state.phase == Phase.ERROR
    assert state.vnpu_ids == ()
    assert fixture.runners.restart_calls == 0
    assert fixture.npu.destroyed_count == 0


def test_partial_split_failure_rolls_back_discovered_ids_and_device_mode():
    fixture = SchedulerFixture(phase=Phase.WHOLE, queued=1)
    fixture.npu.fail = "create"
    fixture.npu.partial_create = (401, 407)

    state = fixture.reconcile()

    assert state.phase == Phase.WHOLE
    assert fixture.npu.destroyed_ids == [401, 407]
    assert fixture.npu.restored_modes == [0o20660]
    assert fixture.runners.concurrent_starts == 0


def test_partial_split_failure_does_not_destroy_when_job_became_in_progress():
    fixture = SchedulerFixture(
        phase=Phase.WHOLE,
        observations=[
            SchedulerFixture.jobs(1),
            SchedulerFixture.jobs(0, in_progress=1),
        ],
    )
    fixture.npu.fail = "create"
    fixture.npu.partial_create = (401, 407)

    state = fixture.reconcile()

    assert state.phase == Phase.ERROR
    assert state.vnpu_ids == (401, 407)
    assert fixture.npu.destroyed_count == 0
    assert fixture.npu.restored_modes == []


@pytest.mark.parametrize("failure", ["prepare", "start"])
def test_runner_setup_failure_rolls_back_allocation(failure):
    fixture = SchedulerFixture(phase=Phase.WHOLE, queued=1)
    fixture.runners.fail = failure

    state = fixture.reconcile()

    assert state.phase == Phase.WHOLE
    assert fixture.runners.stop_all_calls == 1
    assert fixture.npu.destroyed_count == 4
    assert fixture.npu.restored_modes == [0o20660]


def test_atomic_worker_discovery_blocks_rollback_destruction():
    fixture = SchedulerFixture(phase=Phase.WHOLE, queued=1)
    fixture.runners.fail = "start"
    fixture.runners.stop_all_results = [False]

    state = fixture.reconcile()

    assert state.phase == Phase.ERROR
    assert len(state.vnpu_ids) == 4
    assert fixture.npu.destroyed_count == 0


def test_restore_failure_enters_error_and_preserves_original_mode():
    fixture = SchedulerFixture(
        phase=Phase.RESTORING,
        queued=0,
        workers=0,
        vnpu_ids=(51, 55, 59, 63),
        original_mode=0o20640,
    )
    fixture.npu.fail = "restore"

    state = fixture.reconcile()

    assert state.phase == Phase.ERROR
    assert state.vnpu_ids == ()
    assert state.original_device_mode == 0o20640


def test_error_retries_restore_idempotently():
    fixture = SchedulerFixture(
        phase=Phase.ERROR,
        queued=0,
        workers=0,
        vnpu_ids=(),
        inventory=[],
        original_mode=0o20640,
    )

    state = fixture.reconcile()

    assert state == SchedulerState(phase=Phase.WHOLE)
    assert fixture.npu.destroyed_count == 0
    assert fixture.npu.restored_modes == [0o20640]


def test_queued_work_restarts_exited_runners_only_inside_window():
    fixture = SchedulerFixture(
        phase=Phase.SERVING,
        queued=1,
        workers=0,
        vnpu_ids=(11, 12, 13, 14),
        original_mode=0o20660,
    )

    state = fixture.reconcile()

    assert state.phase == Phase.SERVING
    assert fixture.runners.restart_calls == 1


@pytest.mark.parametrize("phase", list(Phase))
def test_pause_recovery_failure_is_error_and_blocks_all_progress(phase):
    ids = () if phase == Phase.WHOLE else (11, 12, 13, 14)
    fixture = SchedulerFixture(
        phase=phase,
        minute=20 if phase == Phase.DRAINING else 8,
        queued=1,
        vnpu_ids=ids,
        original_mode=0o20660 if ids else None,
    )
    fixture.runners.fail = "recover"

    state = fixture.reconcile()

    assert state.phase == Phase.ERROR
    assert state.vnpu_ids == ids
    assert fixture.events == ["recover"]


def test_atomic_idle_check_refusing_shutdown_never_destroys_vnpus():
    fixture = SchedulerFixture(
        phase=Phase.DRAINING,
        minute=45,
        queued=3,
        workers=0,
        vnpu_ids=(11, 12, 13, 14),
        original_mode=0o20660,
    )
    fixture.runners.stop_all_results = [False]

    state = fixture.reconcile()

    assert state.phase == Phase.DRAINING
    assert fixture.npu.destroyed_count == 0


def test_restoring_never_destroys_with_in_progress_target_job():
    fixture = SchedulerFixture(
        phase=Phase.RESTORING,
        in_progress=1,
        workers=0,
        vnpu_ids=(11, 12, 13, 14),
        original_mode=0o20660,
    )

    state = fixture.reconcile()

    assert state.phase == Phase.RESTORING
    assert fixture.npu.destroyed_count == 0
