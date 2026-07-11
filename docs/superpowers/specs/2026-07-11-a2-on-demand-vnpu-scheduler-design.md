# A2 On-Demand vNPU Scheduler Design

## Status

Design sections confirmed by the user on 2026-07-11. The written specification is pending final user review.

## Context

A2 physical NPU 7 is currently split into four `vir05_1c_16g` vNPUs for UCM GitHub Actions runners. The split consumes all 20 AI cores and all 64 GB of HBM on the physical card even when no CI job is running. As a result, card 7 cannot serve ordinary full-card workloads during most of the hour, while CI needs the four-way split only during short bursts.

The current vNPUs are 212, 213, 214, and 215. Runner containers exist for 212, 214, and 215. Runner 213 is no longer registered with GitHub. Existing runner containers use Docker bridge networking, the A2 GitHub proxy, the host Docker socket, model mounts, and one vNPU per runner.

## Goals

- Keep physical card 7 whole whenever no matching GitHub job needs it.
- Open an allocation window during minutes `00` through `19` of every hour.
- During the window, detect queued jobs for `ModelEngine-Group/unified-cache-management` whose labels include `self-hosted`, `npu`, and `a2`.
- Give CI priority over an existing full-card workload on card 7.
- Run up to four A2 CI jobs concurrently on four `vir05_1c_16g` vNPUs.
- Stop accepting new jobs at minute `20`, allow already-started jobs to finish, and then restore the physical card.
- Restore the physical card as soon as the matching queue and active job set become empty, including before minute `20`.
- Prevent scheduler re-entry, partial allocation, stale device binding, and accidental destruction of an active CI workload.
- Require no UCM workflow changes.

## Non-Goals

- Scheduling cards 0 through 6.
- Preempting workloads outside physical card 7.
- Managing or stopping the container that owns a preempted host PID.
- Providing hard isolation against a cooperating user with unrestricted host root access. The scheduler prevents normal races, but a root process can bypass file permissions and ignore the scheduler lock.
- Replacing GitHub Actions with Slurm, Kubernetes, or another cluster scheduler.

## Selected Approach

Use an A2 host-side state reconciler invoked by a systemd timer every 30 seconds. It reads GitHub Actions state with a repository-scoped fine-grained PAT, reconciles that desired state with the actual NPU, vNPU, process, and runner state, and performs one idempotent transition while holding a global lock.

The alternative of adding a dispatcher job to the workflow was rejected because it couples resource management to PR workflow logic and requires stale-request cleanup. Unconditionally splitting card 7 for the first 20 minutes of every hour was rejected because it wastes the card and may preempt a full-card workload when no CI job exists.

## Components

### Scheduler

Install a Python standard-library-only reconciler on A2. It depends on `npu-smi`, Docker, systemd, and the existing A2 GitHub proxy.

The scheduler owns:

- GitHub run and job discovery.
- The hourly allocation window.
- The card 7 host-process eviction sequence.
- vNPU creation, discovery, validation, and destruction.
- Concurrent runner startup and single-job runner recycling.
- State recovery after scheduler or host service restart.
- Structured event logging.

### systemd Units

- `ucm-a2-scheduler.timer` invokes the reconciler every 30 seconds.
- `ucm-a2-scheduler.service` runs one reconciliation and exits.
- The service is non-overlapping because the reconciler also holds a global `flock`.
- A failed reconciliation does not trigger uncontrolled systemd restart loops; the next timer event retries from observed state.

### Configuration And State

Use these ownership boundaries:

- Root-owned scheduler configuration under `/etc/ucm-a2-scheduler/`.
- A root-only GitHub token file with mode `0600`. The token must not appear in process arguments, runner environments, state files, or logs.
- Persistent atomic state under `/var/lib/ucm-a2-scheduler/state.json`.
- A transient lock under `/run/lock/ucm-a2-card7.lock`.
- Logs in journald under `ucm-a2-scheduler.service`.

Configuration includes the repository, required labels, physical card ID 7, window start and end minutes, eviction grace period, vNPU count and template, runner definitions, proxy settings, and device/mount settings.

## GitHub Job Discovery

The scheduler queries both queued and in-progress workflow runs. A target A2 job can remain queued while other jobs in the same workflow make the overall run `in_progress`, so querying only queued runs is insufficient.

For recent queued and in-progress runs, the scheduler lists jobs and selects a job when:

- Its repository is `ModelEngine-Group/unified-cache-management`.
- Its labels are a superset of `self-hosted`, `npu`, and `a2`.
- Its status is `queued` or `in_progress`.

The scheduler uses conditional requests where supported and stays below the authenticated REST API limit. API errors are represented as an unknown state, never as an empty queue.

The fine-grained PAT created for this design is pending organization approval, but it was verified on 2026-07-11 to authenticate as `dante159753`, receive a 5000-request hourly API limit, and read the UCM Actions runs endpoint. Approval is therefore not a rollout blocker for this public repository, although approval remains desirable.

## Allocation Window

- The allocation window is `[HH:00:00, HH:20:00)` in the A2 host timezone.
- A matching queued job arriving at any point in the window can trigger allocation.
- At or after minute `20`, the scheduler cannot start or restart a runner listener.
- Jobs accepted before minute `20` may run past the end of the window.
- If no target job is queued and no target job or local worker is active, the physical card is restored after two consistent observations approximately five seconds apart.

## State Machine

### `WHOLE`

Expected state:

- Card 7 has no vNPUs.
- All four scheduler-managed runner containers are stopped.
- The original access mode of `/dev/davinci7` is restored.

Within the allocation window, a matching queued job transitions the scheduler to `RESERVING`. Outside the window, no allocation occurs.

### `RESERVING`

The scheduler:

1. Acquires the global card 7 lock.
2. Atomically records the transition and the original `/dev/davinci7` mode.
3. Temporarily restricts new ordinary opens of `/dev/davinci7`.
4. Uses `npu-smi` to identify host PIDs using physical card 7.
5. Sends `SIGTERM` to those host PIDs without stopping or deleting their containers.
6. Waits up to 30 seconds.
7. Sends `SIGKILL` to any remaining card 7 PIDs.
8. Requires three consecutive clean card samples before proceeding.

If the card cannot be cleared, the scheduler restores access permissions and returns to `WHOLE` with an error event. It does not attempt a partial split.

### `SPLITTING`

The scheduler creates four `vir05_1c_16g` vNPUs sequentially. Sequential creation avoids concurrent driver-control operations. After creation, it queries `npu-smi info -t info-vnpu -i 7 -c 0` and requires exactly four healthy vNPUs with the requested template.

vNPU IDs are discovered after every split. The current IDs 212 through 215 are expected but are not treated as stable. The four actual IDs are sorted and mapped to four logical runner slots. Runner container device bindings and `A2_E2E_HOST_DEVICE` values are generated from this mapping.

If any creation or validation step fails, the scheduler destroys only the vNPUs created by the current transition, verifies that card 7 has no vNPUs, restores physical-device access, and returns to `WHOLE`.

### `SERVING`

Only after all four vNPUs validate successfully does the scheduler start all four runner containers concurrently. Each runner:

- Uses Docker bridge networking, not host networking.
- Uses the existing container-facing GitHub proxy at `http://172.17.0.1:7891`.
- Mounts `/var/run/docker.sock` and the existing model and Ascend runtime paths.
- Binds one discovered host vNPU to container `/dev/davinci0`.
- Exposes the same discovered host vNPU path through `A2_E2E_HOST_DEVICE` for the inner Docker test container.
- Runs one GitHub job and then exits.
- Uses Docker restart policy `no`; only the scheduler may start it.

During the allocation window, an exited runner is restarted only when matching queued work remains. Four available jobs therefore run concurrently, and a backlog larger than four proceeds in waves until minute `20`.

When queued jobs, in-progress target jobs, and local runner workers are all absent for the required two observations, the scheduler transitions immediately to `RESTORING`.

### `DRAINING`

At minute `20`, the scheduler stops runner listeners that have not accepted a job. It does not signal an active `Runner.Worker` or its child processes. Completed single-job runners are not restarted.

Queued jobs do not keep the allocation alive in `DRAINING`; they remain queued for the next hourly window. When no in-progress target job and no local worker remains, the scheduler transitions to `RESTORING` even if matching queued jobs still exist.

### `RESTORING`

The scheduler:

1. Confirms there are no active target jobs or local runner workers.
2. Stops any remaining idle runner listener.
3. Verifies no CI-owned NPU process remains.
4. Destroys the four discovered vNPUs.
5. Verifies the vNPU count is zero.
6. Restores the original `/dev/davinci7` access mode.
7. Verifies that physical card 7 can be queried as a whole card.
8. Clears allocation state and returns to `WHOLE`.

### `ERROR`

An unrecoverable mismatch enters `ERROR`. No new runner is started in this state. Future timer executions first attempt to reconcile toward a safe whole-card state. Logs retain the exact failed command, observed processes, vNPU inventory, and previous state without including credentials.

## Concurrency And Collision Prevention

- A global `flock` prevents timer overlap, manual invocation overlap, and concurrent state-file writes.
- State writes use a temporary file, `fsync`, and atomic rename.
- Physical-device permissions are restricted before eviction and restored from recorded state, not from a hard-coded mode.
- Card clearance requires repeated clean observations before the first create operation.
- All four vNPUs must exist before any runner starts.
- Actual vNPU IDs are discovered rather than assumed.
- Runner startup is parallel; vNPU control operations are sequential.
- The scheduler never destroys vNPUs while a matching GitHub job or local `Runner.Worker` remains active.
- A shared root user can bypass the device permission barrier. Absolute enforcement against such a process requires a cluster-wide resource manager and is outside this design.

## Runner Registration

Runner identities are persistent, while listener processes are on demand. The implementation must verify that runner version 2.335.1 supports the selected one-job invocation before replacing the current startup commands.

Runner 213 has been deleted from GitHub and must be registered once before rollout. This requires a short-lived runner registration token or an administrator performing the registration. The Actions-read PAT is deliberately insufficient for runner administration and must not be broadened for this purpose.

## Failure Semantics

- GitHub API failure in `WHOLE`: do nothing to card 7.
- GitHub API failure in `SERVING`: retain the current allocation because unknown is not empty.
- GitHub API failure at or after minute `20`: use two consistent local observations before stopping listeners with no `Runner.Worker`, allow active workers to finish, and restore when local activity reaches zero. Remote queued jobs wait for the next window.
- Host PID eviction failure: abort before splitting and restore device access.
- Partial vNPU creation: destroy current-transition vNPUs and restore the whole card.
- Runner startup failure: stop all scheduler-managed runners and restore the whole card if no job was accepted.
- Scheduler restart: reconstruct state from the physical card, vNPU inventory, container state, `Runner.Worker` processes, persisted state, and GitHub job state before acting.
- Restore failure: enter `ERROR`, refuse new jobs, and retry restoration on later timer events.

## Observability

Each reconciliation logs structured fields for:

- Previous and next state.
- GitHub run ID and job ID.
- Queue and active-job counts.
- Evicted host PID and signal.
- Physical card ID, vNPU ID, and template.
- Runner name, logical slot, and bound host device.
- Transition duration and command failure.

The token and Authorization header are always redacted. A status command reports the current state, window status, last successful GitHub poll, actual vNPU inventory, runner listener and worker status, and last error.

## Validation Plan

Rollout is accepted only after these scenarios pass:

1. No queued job during the window leaves card 7 whole and runners stopped.
2. A queued job arriving at minute `08` triggers host-PID eviction, four-vNPU creation, and concurrent runner startup.
3. Four jobs run concurrently and use four distinct vNPUs.
4. More than four jobs proceed in waves before minute `20`.
5. Queue and active-job counts reaching zero restore the physical card after the two-observation safety check.
6. Jobs active at minute `20` finish successfully while idle runners stop, no new job starts, queued backlog remains remote, and the card is restored after active work finishes.
7. Scheduler restart in `RESERVING`, `SERVING`, `DRAINING`, and `RESTORING` reconstructs the correct state without killing an active CI worker.
8. Injected failure on the third vNPU creation destroys the first two and restores the physical card.
9. GitHub API or proxy failure does not preempt a whole-card workload or treat unknown queue state as empty.
10. Final `npu-smi info -t info-vnpu -i 7 -c 0` reports zero vNPUs, and a controlled full-card memory allocation and compute probe succeeds on card 7.

## Rollout And Rollback

Rollout begins by stopping the three current persistent runner containers and restoring card 7 to whole-card mode. Runner 213 is registered, all four runner definitions are converted to scheduler ownership, and the timer is enabled only after dry-run discovery and fault-injection tests pass.

Rollback disables the timer, waits for or explicitly drains active CI jobs, stops scheduler-managed runner listeners, destroys scheduler-created vNPUs, restores `/dev/davinci7` access, and verifies a whole-card compute probe. The previous persistent-runner container definitions remain backed up but disabled so they cannot restart unexpectedly.
