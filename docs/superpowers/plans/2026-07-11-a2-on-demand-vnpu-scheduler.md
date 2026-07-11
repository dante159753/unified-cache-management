# A2 On-Demand vNPU Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy an A2 host scheduler that gives UCM GitHub Actions four concurrent vNPUs during the first 20 minutes of each hour and otherwise restores physical NPU 7 for whole-card use.

**Architecture:** A Python 3.9-compatible, standard-library-only reconciler runs every 30 seconds from systemd. Pure parsers and state transitions are unit tested locally, while narrow command adapters control GitHub REST queries, `npu-smi`, host signals, Docker runner containers, and atomic state persistence on A2.

**Tech Stack:** Python 3.9, pytest, urllib, dataclasses, subprocess, `npu-smi` 25.5.2, Docker 24.0.9, systemd 249, GitHub Actions REST API.

## Global Constraints

- Keep `D:\code\unified-cache-management` untouched; all repository edits remain in `D:\code\unified-cache-management\.worktrees\a2-on-demand-vnpu-scheduler`.
- Physical card ID is `7`, chip ID is `0`, and the allocation window is `[HH:00:00, HH:20:00)` in A2 local time.
- Create exactly four `vir05_1c_16g` vNPUs and discover their actual IDs after creation.
- Start four runner containers concurrently only after all four vNPUs validate.
- Match only `ModelEngine-Group/unified-cache-management` jobs whose labels include `self-hosted`, `npu`, and `a2`.
- CI preempts card 7 host PIDs with `SIGTERM`, a 30-second grace period, and then `SIGKILL`; do not stop or delete the owning workload container.
- At minute `20`, stop accepting new jobs, let active workers finish, and leave queued backlog for the next window.
- When the queue and active set clear inside the window, confirm twice five seconds apart and restore the whole card immediately.
- Use Docker bridge networking and proxy `http://172.17.0.1:7891`; never use host networking for runners.
- Never print, log, commit, or place the GitHub token in runner environments or process arguments.
- Do not destroy a vNPU while a matching GitHub job or local `Runner.Worker` is active.
- Add code comments only for non-obvious safety invariants or driver constraints.
- Keep all commits local until the user separately asks to push; publish to `wyz` and target `origin/develop` when authorized.

---

## File Map

- Create `test/runner-docker/a2_scheduler/__init__.py`: package version and public exports.
- Create `test/runner-docker/a2_scheduler/models.py`: configuration, job, vNPU, runner, phase, and persisted-state types.
- Create `test/runner-docker/a2_scheduler/github_client.py`: authenticated REST client and target-job filtering.
- Create `test/runner-docker/a2_scheduler/host.py`: subprocess adapter, `npu-smi` parsers, PID eviction, and physical/vNPU operations.
- Create `test/runner-docker/a2_scheduler/runners.py`: deterministic Docker container definitions and listener/worker inspection.
- Create `test/runner-docker/a2_scheduler/reconciler.py`: allocation-window policy and state-machine transitions.
- Create `test/runner-docker/a2_scheduler/cli.py`: `reconcile`, `status`, and `dry-run` entry points with locking and atomic state IO.
- Create `test/runner-docker/a2_scheduler/config.example.json`: complete deployable configuration without credentials.
- Create `test/runner-docker/a2_scheduler/systemd/ucm-a2-scheduler.service`: one-shot reconciler service.
- Create `test/runner-docker/a2_scheduler/systemd/ucm-a2-scheduler.timer`: 30-second timer.
- Create `test/runner-docker/a2_scheduler/tests/conftest.py`: add `test/runner-docker` to the local Windows test import path and provide shared fake adapters.
- Create `test/runner-docker/a2_scheduler/tests/`: focused tests for every module and integration-style state transitions using fakes.
- Modify `test/runner-docker/README.md`: A2 scheduler build, deploy, status, rollback, and registration instructions.

---

### Task 1: Define Configuration And Persisted State

**Files:**
- Create: `test/runner-docker/a2_scheduler/__init__.py`
- Create: `test/runner-docker/a2_scheduler/models.py`
- Create: `test/runner-docker/a2_scheduler/config.example.json`
- Create: `test/runner-docker/a2_scheduler/tests/conftest.py`
- Test: `test/runner-docker/a2_scheduler/tests/test_models.py`

**Interfaces:**
- Produces: `Phase`, `Job`, `Vnpu`, `RunnerSlot`, `SchedulerConfig`, `SchedulerState`, and `load_config(path)`.
- Consumes: JSON configuration and JSON state documents only.

- [ ] **Step 1: Write failing model/config tests**

Define `valid_config_dict()` in the test module. It returns the complete example configuration loaded from `config.example.json`; each test deep-copies it before mutation so tests cannot share state.

In `conftest.py`, add the package parent before test collection:

```python
import sys
from pathlib import Path


PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))
```

```python
def test_load_config_has_four_runner_slots(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(valid_config_dict()))
    config = load_config(path)
    assert config.card_id == 7
    assert config.window_end_minute == 20
    assert [slot.name for slot in config.runner_slots] == [
        "ucm-a2-vnpu-runner-212",
        "ucm-a2-vnpu-runner-213",
        "ucm-a2-vnpu-runner-214",
        "ucm-a2-vnpu-runner-215",
    ]


def test_config_rejects_duplicate_runner_names(tmp_path):
    data = valid_config_dict()
    data["runner_slots"][1]["name"] = data["runner_slots"][0]["name"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="runner names must be unique"):
        load_config(path)
```

- [ ] **Step 2: Run tests and verify the expected import failure**

Run: `python -m pytest test/runner-docker/a2_scheduler/tests/test_models.py -q`

Expected: FAIL because `a2_scheduler.models` does not exist.

- [ ] **Step 3: Implement typed models and strict validation**

Use Python 3.9-compatible dataclasses and enums. `SchedulerState.to_json()` must serialize only phase, vNPU IDs, original device mode, transition timestamp, and last error; it must never serialize credentials.

```python
class Phase(str, Enum):
    WHOLE = "WHOLE"
    RESERVING = "RESERVING"
    SPLITTING = "SPLITTING"
    SERVING = "SERVING"
    DRAINING = "DRAINING"
    RESTORING = "RESTORING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class SchedulerConfig:
    repository: str
    required_labels: FrozenSet[str]
    card_id: int
    chip_id: int
    window_start_minute: int
    window_end_minute: int
    eviction_grace_seconds: int
    vnpu_count: int
    vnpu_template: str
    token_file: Path
    state_file: Path
    lock_file: Path
    runner_slots: Tuple[RunnerSlot, ...]
```

The example config must contain the four current configured runner image names, work directories under `/data/github-runners/ucm-a2/`, proxy variables, common Ascend devices, Docker socket, and host runtime mounts.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest test/runner-docker/a2_scheduler/tests/test_models.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add test/runner-docker/a2_scheduler
git commit -m "feat: define A2 scheduler configuration"
```

---

### Task 2: Implement GitHub Job Discovery

**Files:**
- Create: `test/runner-docker/a2_scheduler/github_client.py`
- Test: `test/runner-docker/a2_scheduler/tests/test_github_client.py`

**Interfaces:**
- Consumes: `SchedulerConfig.repository`, `SchedulerConfig.required_labels`, and token text read by the CLI.
- Produces: `GitHubClient.list_target_jobs() -> List[Job]` and `GitHubUnavailable`.

- [ ] **Step 1: Write failing REST/filter tests with a fake opener**

Define `TOKEN = "test-token"`, `REPOSITORY = "ModelEngine-Group/unified-cache-management"`, and a `FakeOpener` whose `open(request, timeout)` returns context-manager responses keyed by request URL. `ErrorOpener(status)` raises `urllib.error.HTTPError` for every request.

```python
def test_finds_queued_a2_job_inside_in_progress_run():
    opener = FakeOpener(
        runs={"queued": [], "in_progress": [{"id": 42}]},
        jobs={42: [{
            "id": 99,
            "name": "test-e2e-pc-a2",
            "status": "queued",
            "labels": ["self-hosted", "npu", "a2"],
        }]},
    )
    jobs = GitHubClient("ModelEngine-Group/unified-cache-management", TOKEN, opener).list_target_jobs()
    assert [(job.id, job.status) for job in jobs] == [(99, "queued")]


def test_api_error_is_unknown_not_empty():
    client = GitHubClient(REPOSITORY, TOKEN, ErrorOpener(503))
    with pytest.raises(GitHubUnavailable):
        client.list_target_jobs()
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest test/runner-docker/a2_scheduler/tests/test_github_client.py -q`

Expected: FAIL because `GitHubClient` does not exist.

- [ ] **Step 3: Implement the urllib client**

Query both of these endpoints with `per_page=100` and the versioned GitHub headers:

```text
GET /repos/ModelEngine-Group/unified-cache-management/actions/runs?status=queued&per_page=100
GET /repos/ModelEngine-Group/unified-cache-management/actions/runs?status=in_progress&per_page=100
GET /repos/ModelEngine-Group/unified-cache-management/actions/runs/{run_id}/jobs?per_page=100
```

Deduplicate run IDs, select only `queued` and `in_progress` jobs, and require the configured labels as a subset. Convert HTTP, timeout, JSON, and pagination failures to `GitHubUnavailable`; do not return an empty list for those failures.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest test/runner-docker/a2_scheduler/tests/test_github_client.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add test/runner-docker/a2_scheduler/github_client.py test/runner-docker/a2_scheduler/tests/test_github_client.py
git commit -m "feat: discover queued A2 GitHub jobs"
```

---

### Task 3: Implement Host NPU Control And Rollback

**Files:**
- Create: `test/runner-docker/a2_scheduler/host.py`
- Test: `test/runner-docker/a2_scheduler/tests/test_host.py`

**Interfaces:**
- Consumes: `CommandRunner.run(argv, timeout) -> CommandResult`, card/chip IDs, vNPU template, and injectable `kill`/`sleep` callables.
- Produces: `NpuManager.list_vnpus()`, `list_card_pids()`, `evict_card_pids()`, `create_vnpus()`, `destroy_vnpus()`, `restrict_device()`, and `restore_device()`.

- [ ] **Step 1: Write failing parser and rollback tests**

Define `load_fixture(name)` to read from `tests/fixtures/`, and define `ScriptedRunner` with `run(argv, timeout)` that consumes ordered `CommandResult` values while recording create and destroy IDs. The `ok()` and `failed()` helpers construct successful and failed `CommandResult` objects.

```python
def test_parse_vnpu_inventory_discovers_actual_ids():
    output = load_fixture("info-vnpu-four.txt")
    assert [v.id for v in parse_info_vnpu(output)] == [212, 213, 214, 215]


def test_partial_create_rolls_back_discovered_ids():
    runner = ScriptedRunner([
        ok("Create vnpu success"),
        ok("Create vnpu success"),
        failed("create failed"),
        ok("destroy 212"),
        ok("destroy 213"),
    ])
    manager = NpuManager(runner, card_id=7, chip_id=0)
    with pytest.raises(NpuOperationError):
        manager.create_vnpus(4, "vir05_1c_16g")
    assert runner.destroyed_ids == [212, 213]
```

Include fixtures for zero vNPUs, four vNPUs, no process, and card 7 process rows.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest test/runner-docker/a2_scheduler/tests/test_host.py -q`

Expected: FAIL because host adapters do not exist.

- [ ] **Step 3: Implement command execution and NPU operations**

Use argument arrays only, never `shell=True`. The concrete commands are:

```text
npu-smi info -t info-vnpu -i 7 -c 0
npu-smi info -t proc-mem -i 7 -c 0
npu-smi set -t create-vnpu -i 7 -c 0 -f vir05_1c_16g
npu-smi set -t destroy-vnpu -i 7 -c 0 -v $vnpu_id
```

After each create, query inventory and compute the newly added ID from set difference. On failure, destroy only IDs created by the current call. For eviction, signal only positive PIDs returned for card 7, wait 30 seconds after `SIGTERM`, signal remaining PIDs with `SIGKILL`, then require three clean samples one second apart.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest test/runner-docker/a2_scheduler/tests/test_host.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add test/runner-docker/a2_scheduler/host.py test/runner-docker/a2_scheduler/tests/test_host.py
git commit -m "feat: manage A2 physical and virtual NPUs"
```

---

### Task 4: Implement Four-Runner Docker Lifecycle

**Files:**
- Create: `test/runner-docker/a2_scheduler/runners.py`
- Test: `test/runner-docker/a2_scheduler/tests/test_runners.py`

**Interfaces:**
- Consumes: four `RunnerSlot` values, four actual `Vnpu` values, and `CommandRunner`.
- Produces: `RunnerManager.prepare(vnpus)`, `start_all()`, `restart_exited()`, `stop_idle()`, `active_workers()`, and `stop_all_if_idle()`.

- [ ] **Step 1: Write failing deterministic command tests**

Define `RecordingRunner` as a thread-safe command collector and `BlockingRecordingRunner` as the same adapter with a barrier around `docker start`; `max_concurrency` records the largest number of simultaneous `run()` calls.

```python
def test_prepare_maps_discovered_vnpus_to_four_slots():
    manager = RunnerManager(config, RecordingRunner())
    manager.prepare([Vnpu(310, "vir05_1c_16g"), Vnpu(311, "vir05_1c_16g"), Vnpu(312, "vir05_1c_16g"), Vnpu(313, "vir05_1c_16g")])
    commands = manager.command_runner.commands
    assert any("/dev/vdavinci310:/dev/davinci0" in command for command in commands)
    assert any("A2_E2E_HOST_DEVICE=/dev/vdavinci313" in command for command in commands)
    assert all("--network=host" not in command for command in commands)
    assert all("--restart=no" in command for command in commands)


def test_start_all_launches_four_containers_concurrently():
    runner = BlockingRecordingRunner()
    manager = RunnerManager(config, runner)
    manager.start_all()
    assert runner.max_concurrency == 4
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest test/runner-docker/a2_scheduler/tests/test_runners.py -q`

Expected: FAIL because `RunnerManager` does not exist.

- [ ] **Step 3: Implement container preparation and single-job startup**

For each logical slot, recreate the stopped container with its configured image and work directory. Preserve the current Docker socket, Docker CLI, Ascend device/runtime, model, update, and proxy mounts. Use this listener command:

```bash
cd /actions-runner
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$https_proxy"
exec ./run.sh --once
```

Detect active work with `docker exec CONTAINER pgrep -af Runner.Worker`. An idle listener has `Runner.Listener` and no `Runner.Worker`; an exited one has no running container. Use a four-worker `ThreadPoolExecutor` only for `docker start` calls.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest test/runner-docker/a2_scheduler/tests/test_runners.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add test/runner-docker/a2_scheduler/runners.py test/runner-docker/a2_scheduler/tests/test_runners.py
git commit -m "feat: manage four on-demand A2 runners"
```

---

### Task 5: Implement The Reconciler State Machine

**Files:**
- Create: `test/runner-docker/a2_scheduler/reconciler.py`
- Test: `test/runner-docker/a2_scheduler/tests/test_reconciler.py`

**Interfaces:**
- Consumes: `GitHubClient`, `NpuManager`, `RunnerManager`, `SchedulerState`, clock, and sleeper.
- Produces: `Reconciler.reconcile(now) -> SchedulerState`.

- [ ] **Step 1: Write failing transition tests**

Define `SchedulerFixture` in the test module with fake GitHub, NPU, runner, signal, clock, and sleep adapters. Constructor fields set the observed minute, queue, workers, card PIDs, and API error; `reconcile()` runs one real `Reconciler` and exposes all side effects through the fakes.

```python
def test_whole_with_queue_inside_window_allocates_and_starts_four():
    fixture = SchedulerFixture(phase=Phase.WHOLE, minute=8, queued=4)
    state = fixture.reconcile()
    assert state.phase == Phase.SERVING
    assert fixture.npu.created_count == 4
    assert fixture.runners.concurrent_starts == 4


def test_whole_with_api_failure_does_not_preempt():
    fixture = SchedulerFixture(phase=Phase.WHOLE, minute=8, github_error=True, card_pids=[123])
    state = fixture.reconcile()
    assert state.phase == Phase.WHOLE
    assert fixture.killed_pids == []


def test_cutoff_leaves_queued_jobs_for_next_window():
    fixture = SchedulerFixture(phase=Phase.SERVING, minute=20, queued=3, workers=0)
    state = fixture.reconcile()
    assert state.phase == Phase.WHOLE
    assert fixture.runners.started_after_cutoff == 0
    assert fixture.npu.destroyed_count == 4
```

Add tests for early empty restoration after two checks, active worker drain, API failure while serving, scheduler restart with actual vNPUs, partial split rollback, and restore failure entering `ERROR`.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest test/runner-docker/a2_scheduler/tests/test_reconciler.py -q`

Expected: FAIL because `Reconciler` does not exist.

- [ ] **Step 3: Implement minimal idempotent transitions**

The policy order in every reconciliation is:

```python
if after_cutoff:
    return drain_or_restore(snapshot)
if state.phase == Phase.WHOLE:
    return allocate_only_when_queue_is_known_nonempty(snapshot)
if state.phase in {Phase.RESERVING, Phase.SPLITTING}:
    return recover_or_continue_transition(snapshot)
if state.phase == Phase.SERVING:
    return serve_restart_or_restore(snapshot)
if state.phase in {Phase.RESTORING, Phase.ERROR}:
    return reconcile_toward_whole(snapshot)
```

No transition may infer empty work from `GitHubUnavailable`. Before early restoration, query twice with a five-second interval and require no queued jobs, no in-progress target jobs, and no local workers both times.

- [ ] **Step 4: Run focused and aggregate tests**

Run: `python -m pytest test/runner-docker/a2_scheduler/tests -q`

Expected: PASS with no skipped scheduler tests.

- [ ] **Step 5: Commit**

```bash
git add test/runner-docker/a2_scheduler/reconciler.py test/runner-docker/a2_scheduler/tests/test_reconciler.py
git commit -m "feat: reconcile A2 CI allocation windows"
```

---

### Task 6: Add CLI, Locking, systemd Units, And Operator Docs

**Files:**
- Create: `test/runner-docker/a2_scheduler/cli.py`
- Create: `test/runner-docker/a2_scheduler/systemd/ucm-a2-scheduler.service`
- Create: `test/runner-docker/a2_scheduler/systemd/ucm-a2-scheduler.timer`
- Create: `test/runner-docker/a2_scheduler/tests/test_cli.py`
- Modify: `test/runner-docker/README.md`

**Interfaces:**
- Consumes: all earlier modules.
- Produces: `python3 -m a2_scheduler.cli reconcile|dry-run|status --config PATH`.

- [ ] **Step 1: Write failing CLI tests**

Define `configured_cli` as a pytest fixture that writes a valid config, token, and state to `tmp_path` and injects fake adapters. Define `FakeLock` with `acquire(blocking=False)` and `release()` so lock-contention tests run on Windows. Production `LinuxFileLock` imports `fcntl` lazily and uses `LOCK_EX | LOCK_NB`; verify that concrete adapter on A2.

```python
def test_status_redacts_token_path_contents(capsys, configured_cli):
    configured_cli.run(["status", "--config", str(configured_cli.config_path)])
    output = capsys.readouterr().out
    assert "TOKEN_SENTINEL" not in output
    assert '"phase": "WHOLE"' in output


def test_second_reconcile_cannot_take_lock(configured_cli):
    with hold_lock(configured_cli.lock_path):
        assert configured_cli.run(["reconcile", "--config", str(configured_cli.config_path)]) == 75
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest test/runner-docker/a2_scheduler/tests/test_cli.py -q`

Expected: FAIL because the CLI does not exist.

- [ ] **Step 3: Implement CLI, atomic state, and unit files**

The service command is:

```ini
[Service]
Type=oneshot
User=root
WorkingDirectory=/opt/ucm-a2-scheduler
Environment=PYTHONPATH=/opt/ucm-a2-scheduler
ExecStart=/usr/bin/python3 -m a2_scheduler.cli reconcile --config /etc/ucm-a2-scheduler/config.json
```

The timer uses `OnBootSec=30s`, `OnUnitActiveSec=30s`, and `AccuracySec=1s`. Do not set `Persistent=true`, because persistence applies to calendar timers rather than this monotonic timer. The README must include dry-run, status, journal, manual reconcile, disable, and whole-card rollback commands.

Add an explicit operator-only command:

```text
python3 -m a2_scheduler.cli validate-create-rollback --fail-before-create 3 --config /etc/ucm-a2-scheduler/config.json
```

It is allowed only when the timer is inactive, card 7 has no processes or vNPUs, and all scheduler runners are stopped. It performs two real creates, injects an exception immediately before the third create, destroys the two discovered IDs, and verifies a zero-vNPU result.

- [ ] **Step 4: Run all local verification**

Run:

```bash
python -m pytest test/runner-docker/a2_scheduler/tests -q
python -m compileall -q test/runner-docker/a2_scheduler
git diff --check
```

Expected: all tests pass, compileall exits 0, and `git diff --check` emits no output.

- [ ] **Step 5: Commit**

```bash
git add test/runner-docker/a2_scheduler test/runner-docker/README.md
git commit -m "feat: package A2 scheduler for systemd"
```

---

### Task 7: Deploy In Dry-Run Mode On A2

**Files:**
- Deploy package to: `/opt/ucm-a2-scheduler/a2_scheduler/`
- Deploy config to: `/etc/ucm-a2-scheduler/config.json`
- Deploy token to: `/etc/ucm-a2-scheduler/github.token`
- Deploy units to: `/etc/systemd/system/ucm-a2-scheduler.service`
- Deploy units to: `/etc/systemd/system/ucm-a2-scheduler.timer`

**Interfaces:**
- Consumes: committed package and the fine-grained PAT currently held by the user.
- Produces: a disabled timer and a working read-only `dry-run`/`status` command on A2.

- [ ] **Step 1: Capture pre-deploy evidence**

Run on A2:

```bash
date
npu-smi info
npu-smi info -t info-vnpu -i 7 -c 0
docker ps -a --filter name=ucm-a2-vnpu-runner --format '{{.Names}}|{{.Status}}|{{.Image}}'
systemctl is-active aliyun-proxy-tunnel
curl -fsS -o /dev/null -w '%{http_code}\n' -x http://127.0.0.1:7898 https://github.com/
```

Expected: the proxy tunnel is active and GitHub returns 200. Record, but do not alter, any active worker.

- [ ] **Step 2: Upload package, config, and token securely**

Use Paramiko SFTP. Read the token from the local clipboard into memory, validate its GitHub prefix, upload directly to `/etc/ucm-a2-scheduler/github.token`, set owner root and mode `0600`, clear the in-process variable, and never print token content.

- [ ] **Step 3: Validate dry-run and status**

Run on A2:

```bash
PYTHONPATH=/opt/ucm-a2-scheduler python3 -m a2_scheduler.cli dry-run --config /etc/ucm-a2-scheduler/config.json
PYTHONPATH=/opt/ucm-a2-scheduler python3 -m a2_scheduler.cli status --config /etc/ucm-a2-scheduler/config.json
systemd-analyze verify /etc/systemd/system/ucm-a2-scheduler.service /etc/systemd/system/ucm-a2-scheduler.timer
```

Expected: GitHub state and actual four-vNPU inventory are reported, no PID is signaled, no runner is restarted, no token appears, and systemd verification exits 0.

- [ ] **Step 4: Keep the timer disabled**

Run: `systemctl disable --now ucm-a2-scheduler.timer`

Expected: inactive timer while rollout prerequisites are completed.

---

### Task 8: Restore The Fourth Runner And Cut Over Ownership

**Files:**
- Update A2 image: `ucm-actions-runner-configured:213-http-proxy-20260603174448`
- Preserve runner work directory: `/data/github-runners/ucm-a2/ucm-a2-vnpu-runner-213`

**Interfaces:**
- Consumes: one short-lived repository runner registration token supplied at rollout time.
- Produces: four valid persistent runner identities whose listener processes are scheduler-owned.

- [ ] **Step 1: Verify no active A2 job before cutover**

Check GitHub target jobs and `Runner.Worker` in 212, 214, and 215. Do not stop a container with an active worker.

- [ ] **Step 2: Re-register runner 213 once**

Start an isolated configuration container from `ucm-actions-runner:a2-arm64`, configure the repository URL, runner name `ucm-a2-vnpu-runner-213`, and labels `self-hosted,npu,a2`, then commit the configured image back to `ucm-actions-runner-configured:213-http-proxy-20260603174448`. Keep the registration token only in process memory and remove it immediately after `config.sh` completes.

- [ ] **Step 3: Validate one-job listener support before ownership change**

Start one configured runner with `./run.sh --once` while no target job is queued, confirm its log reaches the listening state, then terminate the idle listener. Expected: no configuration deletion and the runner remains registered offline.

- [ ] **Step 4: Stop persistent listeners and restore the initial whole-card baseline**

Stop 212, 214, and 215 only after worker count is zero. Query and destroy the actual current card 7 vNPU IDs, expected initially to be 212 through 215, verify the vNPU count is zero, and run a controlled card 7 memory/compute probe. Preserve old container inspect JSON under `/var/lib/ucm-a2-scheduler/backup/` for rollback.

- [ ] **Step 5: Enable scheduler ownership**

Run:

```bash
systemctl daemon-reload
systemctl enable --now ucm-a2-scheduler.timer
systemctl start ucm-a2-scheduler.service
systemctl status --no-pager ucm-a2-scheduler.timer
journalctl -u ucm-a2-scheduler.service -n 100 --no-pager
```

Expected: timer active, state `WHOLE`, zero vNPUs, and four runner containers stopped while no target job is queued.

---

### Task 9: Validate Preemption, Four-Way Concurrency, Drain, And Rollback

**Files:**
- Preserve A2 logs under: `/home/yanzhao/ucm-a2-scheduler-validation/`

**Interfaces:**
- Consumes: live scheduler, four valid runners, and controlled UCM PR jobs.
- Produces: timestamped evidence for all acceptance criteria and a proven rollback command sequence.

- [ ] **Step 1: Validate no-queue whole-card behavior**

During minutes `00` through `19` with no target queue, capture two timer cycles. Expected: state remains `WHOLE`, zero vNPUs, and no runner listener.

- [ ] **Step 2: Validate CI preemption**

Start a controlled card 7 process that allocates NPU memory and can exit cleanly on `SIGTERM`. Queue one A2 job during the window. Expected: scheduler logs `SIGTERM`, waits for exit without stopping its container, creates four vNPUs, and starts four listeners concurrently.

- [ ] **Step 3: Validate four parallel jobs**

Queue four UCM PR runs with the A2 PC E2E job. Capture GitHub job IDs, runner names, `A2_E2E_HOST_DEVICE`, and `npu-smi` process inventory. Expected: four jobs overlap in time and map to four distinct actual vNPU IDs.

- [ ] **Step 4: Validate backlog waves and cutoff**

Queue more than four jobs before minute `20`. Expected: completed one-job runners restart only before cutoff. At minute `20`, idle listeners stop, active jobs finish, remaining queued jobs stay remote, and card 7 returns to whole-card mode.

- [ ] **Step 5: Validate fault rollback**

Run `validate-create-rollback --fail-before-create 3` while the timer is inactive. Expected: the two created test vNPUs are destroyed, runners never start, state returns to `WHOLE`, and the physical compute probe succeeds.

- [ ] **Step 6: Validate operational rollback**

Disable the timer, drain active workers, stop listeners, destroy scheduler-created vNPUs, restore `/dev/davinci7` mode, and rerun the full-card probe. Then re-enable the timer and confirm `WHOLE` state.

- [ ] **Step 7: Run final repository and host verification**

Run locally:

```bash
python -m pytest test/runner-docker/a2_scheduler/tests -q
python -m compileall -q test/runner-docker/a2_scheduler
git status --short
```

Run on A2:

```bash
systemctl is-active ucm-a2-scheduler.timer
PYTHONPATH=/opt/ucm-a2-scheduler python3 -m a2_scheduler.cli status --config /etc/ucm-a2-scheduler/config.json
journalctl -u ucm-a2-scheduler.service --since today --no-pager
npu-smi info -t info-vnpu -i 7 -c 0
```

Expected: local tests pass, timer is active, status matches actual hardware, logs contain no credentials, and the final hardware state matches whether target jobs are active.

- [ ] **Step 8: Commit validation documentation**

```bash
git add test/runner-docker/README.md
git commit -m "docs: document A2 scheduler validation"
```
