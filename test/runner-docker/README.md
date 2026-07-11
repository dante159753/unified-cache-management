# build runner image:
`docker build -t vllm-openai-pipeline:0.11.0 .` to build default image without proxy

`docker build --build-arg PROXY_SERVER="http://172.80.0.1:7890" -t vllm-openai-pipeline:0.11.0 .` to build image with default vllm version and custom proxy server

`docker build --build-arg VLLM_VERSION="v0.9.2" --build-arg PROXY_SERVER="http://172.80.0.1:7890" -t vllm-openai-pipeline:0.9.2-proxy .` to build image with custom vllm version and custom proxy server

# initialize lock files for gpu allocation

``` bash
LOCK_DIR="${1:-/home/yanzhao/pipeline_results/gpu_locks}"
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "ERROR: No GPUs detected via nvidia-smi" >&2
    exit 1
fi

echo "Detected ${GPU_COUNT} GPU(s)"
echo "Initializing lock files in: ${LOCK_DIR}"

mkdir -p "${LOCK_DIR}"
chmod 777 "${LOCK_DIR}"

for i in $(seq 0 $((GPU_COUNT - 1))); do
    LOCK_FILE="${LOCK_DIR}/gpu_${i}.lock"
    touch "${LOCK_FILE}"
    chmod 666 "${LOCK_FILE}"
    echo "  Created: ${LOCK_FILE}"
done

echo "Done. ${GPU_COUNT} lock file(s) ready."
```

# use runner:
## start container:
```
docker run \
        -itd \
        --gpus all \
        --network=host \
        --ipc=host \
        --cap-add IPC_LOCK \
        -e http_proxy="http://127.0.0.1:7890" \
        -e https_proxy="http://127.0.0.1:7890" \
        -v /home/models:/home/models \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v /home/yanzhao/pipeline_results:/workspace/test_results \
        --name pipeline-host-docker-a \
        vllm-openai-pipeline:0.9.2-proxy
```

## attach container
`docker exec -it pipeline-host-docker-a /bin/bash`

## configure github runner
get connect token from github, run 

`./config.sh --url https://github.com/ModelEngine-Group/unified-cache-management --token ***`

## start runner
`sudo -E ./runner-svc.sh install`

`-E` here keeps the proxy setting env var.

## check runner log
`sudo ./runner-svc.sh status`

or go to `./_diag` to check more log

# A2 on-demand vNPU scheduler

Install the `a2_scheduler` package under `/opt/ucm-a2-scheduler`, place the
root-owned configuration at `/etc/ucm-a2-scheduler/config.json`, and keep the
configured GitHub token file mode at `0600`. Install the service and timer from
`a2_scheduler/systemd/`, then reload systemd. Do not enable the timer until the
read-only checks and operator validation below have passed.

Set `lock_file` to `/run/ucm-a2-scheduler/card7.lock`. The service creates
`/run/ucm-a2-scheduler` as a root-owned runtime directory with mode `0700`; the
lock refuses symlinks, non-root ownership, non-file/non-directory objects, and
any group/other permissions.

## Read-only checks

Neither command acquires the scheduler lock, writes state, changes device
permissions, invokes a Docker lifecycle command, or creates/destroys a vNPU.

```bash
cd /opt/ucm-a2-scheduler
sudo /usr/bin/python3 -m a2_scheduler.cli dry-run --config /etc/ucm-a2-scheduler/config.json
sudo /usr/bin/python3 -m a2_scheduler.cli status --config /etc/ucm-a2-scheduler/config.json
```

Inspect timer/service state and recent logs with:

```bash
sudo systemctl status ucm-a2-scheduler.timer ucm-a2-scheduler.service
sudo journalctl -u ucm-a2-scheduler.service -n 200 --no-pager
sudo journalctl -u ucm-a2-scheduler.service -f
```

## Manual reconcile

One reconciliation uses the same nonblocking card lock as the timer service.
Lock contention exits with status `75` without mutating scheduler state or host
resources.

```bash
sudo install -d -o root -g root -m 0700 /run/ucm-a2-scheduler
cd /opt/ucm-a2-scheduler
sudo /usr/bin/python3 -m a2_scheduler.cli reconcile --config /etc/ucm-a2-scheduler/config.json
```

The exit status describes the current invocation, not the persisted phase or
historical `last_error`. A failed allocation that rolls back safely to `WHOLE`
still exits nonzero; a healthy run carrying old diagnostic text exits zero.
If the command reports that state replacement completed but directory `fsync`
failed, the visible state file is new but its crash durability is uncertain.
Treat that invocation as failed, inspect status and host state, and do not retry
mutating work until they agree.

## Create/rollback validation

This operator-only validation performs real host actions. Run it only before
enabling the timer. The command itself requires the timer to report `inactive`,
card 7 to have no PIDs or vNPUs, and every scheduler runner to be stopped. It
creates exactly two vNPUs, injects failure before the third create, applies the
GitHub/runner cleanup policy and per-destroy PID guard, destroys only the two
discovered IDs, and requires a zero-vNPU result.

```bash
cd /opt/ucm-a2-scheduler
sudo /usr/bin/python3 -m a2_scheduler.cli validate-create-rollback --fail-before-create 3 --config /etc/ucm-a2-scheduler/config.json
```

Any failure after the first create enters the same guarded cleanup path.
Cleanup re-discovers IDs, re-checks GitHub policy and stopped runners before
each batch, and relies on a fresh card-PID guard before every destroy command.
If policy, runner, PID, inventory, or destroy checks fail, one or more IDs can
remain. The error preserves both the primary validation failure and cleanup
failure. Keep the timer disabled and inspect before recovery:

```bash
sudo systemctl disable --now ucm-a2-scheduler.timer
sudo /usr/bin/python3 -m a2_scheduler.cli status --config /etc/ucm-a2-scheduler/config.json
sudo npu-smi info -t info-vnpu -i 7 -c 0
sudo npu-smi info -t proc-mem -i 7 -c 0
sudo docker ps --format '{{.Names}}' | grep '^ucm-a2-vnpu-runner-' || true
```

Only after target jobs, scheduler runners, and card 7 PIDs are all absent,
destroy each positively identified validation-created ID and verify zero:

```bash
sudo npu-smi set -t destroy-vnpu -i 7 -c 0 -v <ID>
sudo npu-smi info -t info-vnpu -i 7 -c 0
```

## Disable and restore the whole card

First disable future timer starts and wait for target GitHub jobs and local
`Runner.Worker` processes to finish. Confirm `target_jobs.in_progress` is zero
and every reported runner has `worker: false` before the manual reconcile. The
reconciler then applies its runner-idle, GitHub policy, checkpoint, and card-PID
guards while restoring the recorded physical-device mode.

```bash
sudo systemctl disable --now ucm-a2-scheduler.timer
cd /opt/ucm-a2-scheduler
sudo /usr/bin/python3 -m a2_scheduler.cli status --config /etc/ucm-a2-scheduler/config.json
sudo /usr/bin/python3 -m a2_scheduler.cli reconcile --config /etc/ucm-a2-scheduler/config.json
sudo /usr/bin/python3 -m a2_scheduler.cli status --config /etc/ucm-a2-scheduler/config.json
sudo npu-smi info -t info-vnpu -i 7 -c 0
sudo npu-smi info -t proc-mem -i 7 -c 0
```

Rollback is complete only when status reports `phase: WHOLE`, the vNPU
inventory is empty, and card 7 has no scheduler runner process. Leave the timer
disabled until the cause of the rollback is resolved.
