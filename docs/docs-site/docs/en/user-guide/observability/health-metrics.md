# Store health and circuit breaking

A responding inference server can still have an unusable cache backend. Store
health tells you whether a Pipeline stage is accepting new cache operations.
It does not establish that a particular KV block exists, that a model answer is
correct, or that a deployment meets its latency target.

## What happens when a backend fails

Pipeline wraps each loaded stage in a `HealthBreakerStore` when health checking
is enabled. The wrapper starts enabled and keeps two independent windows. By
default, two failures among eight active probes block the stage. Passive detection
uses 60 one-second buckets and blocks when the window contains at least 10
eligible task results and their failure ratio is strictly greater than 1%.

Only active probes can restore service. After a trip, a fresh full window must
contain only successes, and a successful probe must start after the cooldown
expires. The initial cooldown is 60 seconds. A new trip within 3600 seconds of
recovery doubles the previous cooldown, up to 3600 seconds. At least 3600 seconds of
healthy uptime resets it to 60 seconds on the next trip. Probes continue during
cooldown; failures while blocked do not extend the current deadline.

| Operation while blocked | Result |
| --- | --- |
| `Lookup` | A miss for every requested block |
| `LookupOnPrefix` / `LookupOnReverse` | No hit (`-1`) |
| `Prefetch` | No new prefetch is submitted |
| `Load` / `Dump` | `StoreUnhealthy` is returned |
| `Check` / `Wait` for an existing task | Forwarded to the underlying Store |

Blocking new work does not cancel an already-submitted transfer. Whether a cache
miss leads to recomputation or an error is decided by the integration and request
path; the breaker itself does not retry inference requests.

The base `StoreV1.CheckHealth()` returns success. Posix and Mooncake override it
with backend operations. Pipeline assembly enables passive detection only for
Posix and Mooncake. Cache and other stages do not participate; downstream failures
are handled by the downstream stage's breaker.

Passive detection counts only the first `Wait` result of each accepted task.
`Load` / `Dump` submission results and `Check` polling do not enter either count
or the passive failure metrics. `NotFound` (including misses and short reads),
`StoreUnhealthy`, `InvalidParam`, `DuplicateKey`, and `Unsupported` enter
neither numerator nor denominator. Other failures, including `Retry`, timeouts,
and I/O errors, count as failed tasks. The ratio is total failures divided by total
eligible results across live buckets. Results enter buckets when observed using
a monotonic clock, with one-second boundary precision. Probe I/O is excluded.
Recovery clears passive history. Results from a `Wait` or probe that spans a
breaker state transition are ignored. Tasks are not tracked at submission;
a `Wait` started after recovery belongs to the current window. Existing tasks
still need to be waited on and released.

Bucket counters atomically update their timestamp and count. Bucket reuse needs
no rotation lock and ignores older updates. Recovery resets can still overlap
in-flight updates, so counts near recovery are approximate.

With passive detection enabled, eligible failed `Wait` results are returned as
`StoreUnhealthy` after local accounting, even before the breaker reaches its trip
threshold. The returned message includes the original status. An outer breaker
excludes this status from its own window if intermediate stores preserve it.

## Set the probe policy

`store_health` belongs inside `ucm_connector_config`. For example, a
`Cache|Posix` configuration can explicitly set the default health policy:

```yaml
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: /mnt/ucm-cache
      store_health:
        enabled: true
        health_check_interval_s: 10
        health_check_timeout_s: 3
        health_window_size: 8
        failure_threshold: 2
        passive_enabled: true
        passive_window_s: 60
        passive_min_samples: 10
        passive_failure_ratio: 0.01
        initial_cooldown_s: 60
        max_cooldown_s: 3600
        backoff_factor: 2
        stable_reset_after_s: 3600
enable_metrics: true
```

Keep the model, mount and cache settings appropriate to your deployment; see
[Pipeline Store](../../developer-guide/cache-configuration/pipeline.md).

| Field | Meaning |
| --- | --- |
| `enabled` | Create the health wrapper and probe thread; default `true` |
| `health_check_interval_s` | Target probe interval; default 10 seconds |
| `health_check_timeout_s` | Probe execution deadline; default 3 seconds |
| `health_window_size` | Number of recent results retained; default 8 |
| `failure_threshold` | Failures needed to block new operations; default 2 |
| `passive_enabled` | Enable passive detection for Posix and Mooncake; default `true`; excludes Cache and other stages |
| `passive_window_s` | Positive integer number of one-second buckets; default 60 |
| `passive_min_samples` | Positive minimum eligible task count before evaluating the ratio; default 10 |
| `passive_failure_ratio` | Strict trip threshold in `[0, 1)`; default 0.01 |
| `initial_cooldown_s` / `max_cooldown_s` | Initial / maximum cooldown; defaults 60 / 3600 seconds |
| `backoff_factor` | Cooldown multiplier after an early repeat trip, at least 1; default 2 |
| `stable_reset_after_s` | Healthy uptime after recovery that resets backoff; default 3600 seconds |

Windows, probe interval, probe timeout, and stable uptime must be
positive. The active failure threshold cannot exceed its window, and probe timeout
must be shorter than the interval. Cooldowns can be zero; the maximum cannot be
less than the initial cooldown. The first probe is delayed by
one interval plus random jitter of up to another interval. The initial enabled
state is published before that first probe; it is not evidence of a successful I/O.

Setting `enabled: false` removes this protection. It does not fix a backend failure
or turn off the backend's own error handling.

## Understand what each probe exercises

**Posix** checks every path selected by its storage layout. It creates a temporary
file, writes 4096 bytes, reads and compares them, and removes the file. Buffered
I/O also calls `Sync`; Direct I/O uses the configured direct-open flag. A failed
open, transfer, sync, comparison or removal makes the probe fail. For an NFS-backed
path this exercises the mounted filesystem from that UCM process.

**Mooncake** writes an eight-byte test value under a dedicated key, retrieves and
compares it, then removes it. The implementation uses the real client on the
transfer path and the RPC client on the scheduler path. This checks the configured
client path; it is not a test of every model's KV layout or every remote replica.

Health checks produce their own small objects. Do not count those files or keys as
proof that a request saved reusable KV data.

## Inspect state before aggregating

The default vLLM connector export uses these metrics:

| Backend | State Gauge | Probe Counters |
| --- | --- | --- |
| Posix | `ucm:posix_store_health` | `ucm:posix_healthy_count_total`, `ucm:posix_unhealthy_count_total` |
| Mooncake | `ucm:mooncake_store_health` | `ucm:mooncake_healthy_count_total`, `ucm:mooncake_unhealthy_count_total` |

The Gauge is 1 while the wrapper accepts work and 0 while it is blocked. Counters
record probe outcomes, including timeouts, rather than breaker transitions.
A single successful probe need not change a blocked Gauge back to 1.

`ucm:posix_passive_failures_total` and `ucm:mooncake_passive_failures_total` count
eligible failures observed by passive detection, including late failures from
older tasks. They are separate from active probe counters and do not represent
the current passive window. A passive trip immediately updates the native health
Gauge; export still depends on connector statistics collection.

Start with individual series and their labels:

```promql
ucm:posix_store_health{job="vllm"}
```

Then inspect failures over a recent window:

```promql
increase(ucm:posix_unhealthy_count_total{job="vllm"}[5m])
```

These queries assume your scrape job is named `vllm`. Preserve `instance`, model,
engine and `worker_rank` labels when locating a failure. Scheduler observations use
`worker_rank="scheduler"`. Multiple processes can probe the same backend, and the
exported labels do not identify every physical mount or underlying pipeline
object; series counts are not counts of failed disks.

The native probe thread and Prometheus scrape run on different schedules. In the
vLLM connector path, native statistics reach the exporter through
`get_kv_connector_stats()`. Check that collection is advancing before interpreting
a repeated value or a missing series as the current backend state.

## Investigate and confirm recovery

On each `health_check_interval_s` cycle (10 seconds by default), the background
thread logs `Store passive health window` only if the current window has failures.
It includes the stage, window duration, eligible result count, failure count,
failure ratio, minimum samples, and threshold, even below the minimum sample count.
Expired buckets are excluded using the current time, including during idle periods.
Logging stops when failures expire or recovery clears the window.

1. Check the scrape target and identify the affected process from the labels.
2. Find `Store health check` failures and `transitioned to UNHEALTHY` in its log;
   transition logs include the stage identifier, `source=active_probe` or
   `source=passive_io`, cooldown in milliseconds, and breaker generation.
3. For Posix, inspect that process's mount, permissions, available capacity and
   read/write/remove errors. For Mooncake, inspect its configured client and
   metadata/master connectivity and the reported operation error.
4. Restore the failed dependency, then wait for a full active success window and
   the cooldown. Confirm `transitioned to HEALTHY` and the corresponding Gauge update.
5. Separately repeat the [external-cache verification](../quick_start/index.md#vllm-verify-the-service-and-external-cache).
   Recovery of a probe does not prove recovery of a particular request's cache.

For metric units and export paths, see [Metrics reference](metrics-reference.md).
The policy and operation behavior are defined in
[`StoreHealthConfig`](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/ucm/store/pipeline/cc/store_health_config.h)
and [`HealthBreakerStore`](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/ucm/store/pipeline/cc/health_breaker_store.cc).
