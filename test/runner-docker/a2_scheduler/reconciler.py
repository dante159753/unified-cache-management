import time
from dataclasses import replace
from datetime import datetime
from typing import Callable, List, Optional, Sequence, Tuple

from .github_client import GitHubUnavailable
from .host import NpuOperationError
from .models import Job, Phase, SchedulerConfig, SchedulerState, Vnpu
from .runners import RunnerOperationError


class StateCheckpointError(RuntimeError):
    def __init__(self, message: str, state: SchedulerState) -> None:
        super().__init__(message)
        self.state = state


class Reconciler:
    def __init__(
        self,
        config: SchedulerConfig,
        state: SchedulerState,
        github,
        npu,
        runners,
        checkpoint: Callable[[SchedulerState], None],
        clock: Callable[[], datetime] = datetime.now,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.state = state
        self.github = github
        self.npu = npu
        self.runners = runners
        self.checkpoint = checkpoint
        self.clock = clock
        self.sleep = sleep
        self.now = clock()

    def reconcile(self, now: datetime) -> SchedulerState:
        self.now = now
        try:
            self.runners.recover_paused()
        except RunnerOperationError as error:
            return self._error(error)

        try:
            if self.state.phase != Phase.WHOLE:
                refresh_error = self._refresh_actual_ids()
                if refresh_error is not None:
                    return refresh_error

            after_cutoff = self._after_cutoff(now)
            if self.state.phase == Phase.WHOLE:
                return self._reconcile_whole(not after_cutoff)
            if after_cutoff:
                return self._drain_or_restore()
            if self.state.phase in (Phase.RESERVING, Phase.SPLITTING):
                return self._continue_allocation()
            if self.state.phase == Phase.SERVING:
                return self._serve()
            if self.state.phase == Phase.DRAINING:
                return self._drain_or_restore()
            return self._reconcile_toward_whole()
        except StateCheckpointError as error:
            return self._error(
                error,
                vnpu_ids=error.state.vnpu_ids,
                original_device_mode=error.state.original_device_mode,
                persist=False,
            )

    def _refresh_actual_ids(self) -> Optional[SchedulerState]:
        try:
            inventory = self.npu.list_vnpus()
        except NpuOperationError as error:
            return self._error(error)
        ids = self._ids(inventory)
        if ids != self.state.vnpu_ids:
            self._checkpoint(replace(self.state, vnpu_ids=ids))
        if self.state.phase == Phase.SERVING:
            try:
                self._require_exact_inventory(inventory)
            except ValueError as error:
                return self._error(error)
        return None

    def _reconcile_whole(self, allow_allocation: bool) -> SchedulerState:
        try:
            inventory = self.npu.list_vnpus()
            workers = self.runners.active_workers()
        except (NpuOperationError, RunnerOperationError) as error:
            return self._error(error)
        jobs = self._jobs()
        ids = self._ids(inventory)

        if ids != self.state.vnpu_ids:
            self._checkpoint(replace(self.state, vnpu_ids=ids))
        if workers:
            return self._error(RuntimeError("local runner worker is active"))
        if self._in_progress(jobs or ()):
            if ids or self.state.original_device_mode is not None:
                return self._error(RuntimeError("target job is in progress"))
            return self.state

        stale_allocation = bool(ids) or self.state.original_device_mode is not None
        if stale_allocation:
            if jobs is None and allow_allocation:
                return self._error(GitHubUnavailable("GitHub state is unknown"))
            if self.state.original_device_mode is None:
                return self._error(RuntimeError("original device mode is unknown"))
            return self._restore_allocation()

        if not allow_allocation or jobs is None or not self._queued(jobs):
            return self.state
        return self._allocate_when_queued()

    def _after_cutoff(self, now: datetime) -> bool:
        minute = now.minute
        start = self.config.window_start_minute
        end = self.config.window_end_minute
        return not start <= minute < end

    def _allocate_when_queued(self) -> SchedulerState:
        try:
            original_mode = self.npu.restrict_device()
        except NpuOperationError as error:
            return self._whole_with_error(error)

        reserving = self._transition(
            Phase.RESERVING,
            vnpu_ids=(),
            original_mode=original_mode,
        )
        try:
            self._checkpoint(reserving)
        except StateCheckpointError as error:
            try:
                self.npu.restore_device(original_mode)
            except NpuOperationError as restore_error:
                error = StateCheckpointError(
                    "{}; mode recovery failed: {}".format(error, restore_error),
                    error.state,
                )
            return self._error(error, persist=False)

        try:
            self.npu.evict_card_pids()
        except NpuOperationError as error:
            return self._restore_mode_only(original_mode, error)
        return self._split_and_start()

    def _continue_allocation(self) -> SchedulerState:
        try:
            inventory = self._validated_inventory(allow_empty=True)
        except (NpuOperationError, ValueError) as error:
            return self._rollback(error)
        if inventory:
            return self._prepare_and_start(inventory)
        if self.state.original_device_mode is None:
            return self._error(ValueError("original device mode is unknown"))
        try:
            self.npu.evict_card_pids()
        except NpuOperationError as error:
            return self._rollback(error)
        return self._split_and_start()

    def _split_and_start(self) -> SchedulerState:
        try:
            self.npu.create_vnpus(
                self.config.vnpu_count,
                self.config.vnpu_template,
            )
            inventory = self._validated_inventory(allow_empty=False)
        except (NpuOperationError, ValueError) as error:
            return self._rollback(error)
        return self._prepare_and_start(inventory)

    def _prepare_and_start(self, inventory: Sequence[Vnpu]) -> SchedulerState:
        ids = self._ids(inventory)
        self._checkpoint(self._transition(Phase.SPLITTING, vnpu_ids=ids))
        try:
            self.runners.prepare(inventory)
        except (RunnerOperationError, ValueError) as error:
            return self._rollback(error)

        self._checkpoint(self._transition(Phase.SERVING, vnpu_ids=ids))
        current = self.clock()
        self.now = current
        if self._after_cutoff(current):
            return self._rollback(RuntimeError("allocation cutoff reached"))
        try:
            self.runners.start_all()
        except RunnerOperationError as error:
            return self._rollback(error)
        return self.state

    def _serve(self) -> SchedulerState:
        jobs = self._jobs()
        if jobs is None:
            return self.state
        if self._queued(jobs):
            current = self.clock()
            self.now = current
            if self._after_cutoff(current):
                return self._drain_or_restore()
            try:
                self.runners.restart_exited()
            except RunnerOperationError as error:
                return self._error(error)
            return self.state
        if self._in_progress(jobs):
            return self.state
        try:
            if self.runners.active_workers() != 0:
                return self.state
        except RunnerOperationError as error:
            return self._error(error)

        self.sleep(5)
        second_jobs = self._jobs()
        if (
            second_jobs is None
            or self._queued(second_jobs)
            or self._in_progress(second_jobs)
        ):
            return self.state
        try:
            if self.runners.active_workers() != 0:
                return self.state
        except RunnerOperationError as error:
            return self._error(error)
        return self._restore_allocation()

    def _drain_or_restore(self) -> SchedulerState:
        jobs = self._jobs()
        try:
            workers = self.runners.active_workers()
        except RunnerOperationError as error:
            return self._error(error)
        if workers or (jobs is not None and self._in_progress(jobs)):
            self._checkpoint(self._transition(Phase.DRAINING))
            try:
                self.runners.stop_idle()
            except RunnerOperationError as error:
                return self._error(error)
            return self.state

        self._checkpoint(self._transition(Phase.RESTORING))
        try:
            all_idle = self.runners.stop_all_if_idle()
        except RunnerOperationError as error:
            return self._error(error)
        if not all_idle:
            self._checkpoint(self._transition(Phase.DRAINING))
            return self.state
        return self._restore_allocation(runners_stopped=True, initial_checkpoint=False)

    def _reconcile_toward_whole(self) -> SchedulerState:
        jobs = self._jobs()
        if jobs is None:
            return self.state
        if self._in_progress(jobs):
            return self.state
        try:
            if self.runners.active_workers() != 0:
                return self.state
        except RunnerOperationError as error:
            return self._error(error)
        return self._restore_allocation()

    def _rollback(self, cause: Exception) -> SchedulerState:
        jobs = self._jobs()
        if jobs is None and not self._after_cutoff(self.now):
            return self._error(
                self._combined(cause, GitHubUnavailable("GitHub state is unknown")),
                vnpu_ids=self._observed_ids(),
            )
        if self._in_progress(jobs or ()):
            return self._error(
                self._combined(cause, RuntimeError("target job is in progress")),
                vnpu_ids=self._observed_ids(),
            )
        return self._restore_allocation(cause=cause)

    def _restore_allocation(
        self,
        cause: Optional[Exception] = None,
        runners_stopped: bool = False,
        initial_checkpoint: bool = True,
    ) -> SchedulerState:
        try:
            inventory = self.npu.list_vnpus()
        except NpuOperationError as error:
            return self._error(self._combined(cause, error))
        ids = self._ids(inventory)
        original_mode = self.state.original_device_mode
        if ids and original_mode is None:
            return self._error(
                self._combined(cause, RuntimeError("original device mode is unknown")),
                vnpu_ids=ids,
            )

        restoring = self._transition(Phase.RESTORING, vnpu_ids=ids)
        if initial_checkpoint or restoring != self.state:
            self._checkpoint(restoring)

        if not runners_stopped:
            try:
                if not self.runners.stop_all_if_idle():
                    return self._error(cause or RuntimeError("runner worker is active"))
            except RunnerOperationError as error:
                return self._error(self._combined(cause, error))

        if ids:
            try:
                self.npu.destroy_vnpus(ids)
                remaining = self.npu.list_vnpus()
            except NpuOperationError as error:
                return self._error(self._combined(cause, error), vnpu_ids=ids)
            if remaining:
                return self._error(
                    self._combined(cause, RuntimeError("vNPU inventory is not empty")),
                    vnpu_ids=self._ids(remaining),
                )
            self._checkpoint(self._transition(Phase.RESTORING, vnpu_ids=()))

        if original_mode is not None:
            try:
                self.npu.restore_device(original_mode)
            except NpuOperationError as error:
                return self._error(
                    self._combined(cause, error),
                    vnpu_ids=(),
                    original_device_mode=original_mode,
                )

        if cause is None:
            return self._checkpoint(SchedulerState(phase=Phase.WHOLE))
        return self._checkpoint(self._whole_with_error(cause))

    def _restore_mode_only(
        self, original_mode: int, cause: Exception
    ) -> SchedulerState:
        try:
            self.npu.restore_device(original_mode)
        except NpuOperationError as error:
            return self._error(
                self._combined(cause, error),
                vnpu_ids=(),
                original_device_mode=original_mode,
            )
        return self._checkpoint(self._whole_with_error(cause))

    def _validated_inventory(self, allow_empty: bool) -> List[Vnpu]:
        inventory = sorted(self.npu.list_vnpus(), key=lambda vnpu: vnpu.id)
        if allow_empty and not inventory:
            return inventory
        self._require_exact_inventory(inventory)
        return inventory

    def _require_exact_inventory(self, inventory: Sequence[Vnpu]) -> None:
        if len(inventory) != self.config.vnpu_count:
            raise ValueError("vNPU inventory must contain exactly four devices")
        if any(vnpu.template != self.config.vnpu_template for vnpu in inventory):
            raise ValueError("vNPU inventory has an unexpected template")
        if len({vnpu.id for vnpu in inventory}) != len(inventory):
            raise ValueError("vNPU inventory IDs must be unique")

    def _jobs(self) -> Optional[List[Job]]:
        try:
            return self.github.list_target_jobs()
        except GitHubUnavailable:
            return None

    @staticmethod
    def _queued(jobs: Sequence[Job]) -> bool:
        return any(job.status == "queued" for job in jobs)

    @staticmethod
    def _in_progress(jobs: Sequence[Job]) -> bool:
        return any(job.status == "in_progress" for job in jobs)

    @staticmethod
    def _ids(vnpus: Sequence[Vnpu]) -> Tuple[int, ...]:
        return tuple(sorted(vnpu.id for vnpu in vnpus))

    def _observed_ids(self) -> Tuple[int, ...]:
        try:
            return self._ids(self.npu.list_vnpus())
        except NpuOperationError:
            return self.state.vnpu_ids

    def _checkpoint(self, state: SchedulerState) -> SchedulerState:
        try:
            self.checkpoint(state)
        except Exception as error:
            raise StateCheckpointError(
                "state checkpoint failed: {}".format(error), state
            ) from error
        self.state = state
        return state

    def _transition(
        self,
        phase: Phase,
        vnpu_ids: Optional[Tuple[int, ...]] = None,
        original_mode: Optional[int] = None,
    ) -> SchedulerState:
        return SchedulerState(
            phase=phase,
            vnpu_ids=self.state.vnpu_ids if vnpu_ids is None else vnpu_ids,
            original_device_mode=(
                self.state.original_device_mode
                if original_mode is None
                else original_mode
            ),
            transition_timestamp=self.now.isoformat(),
        )

    def _error(
        self,
        error: Exception,
        vnpu_ids: Optional[Tuple[int, ...]] = None,
        original_device_mode: Optional[int] = None,
        persist: bool = True,
    ) -> SchedulerState:
        state = SchedulerState(
            phase=Phase.ERROR,
            vnpu_ids=self.state.vnpu_ids if vnpu_ids is None else vnpu_ids,
            original_device_mode=(
                self.state.original_device_mode
                if original_device_mode is None
                else original_device_mode
            ),
            transition_timestamp=self.now.isoformat(),
            last_error=str(error),
        )
        if not persist:
            return state
        try:
            return self._checkpoint(state)
        except StateCheckpointError:
            return state

    def _whole_with_error(self, error: Exception) -> SchedulerState:
        return SchedulerState(
            phase=Phase.WHOLE,
            transition_timestamp=self.now.isoformat(),
            last_error=str(error),
        )

    @staticmethod
    def _combined(first: Optional[Exception], second: Exception) -> Exception:
        if first is None:
            return second
        return RuntimeError("{}; rollback failed: {}".format(first, second))
