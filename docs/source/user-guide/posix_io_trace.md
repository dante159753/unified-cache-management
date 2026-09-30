# Posix file IO tracing

Posix IO tracing records KV data file operations for both `aio` and `psync`, including layerwise transfers. It is disabled by default and is independent of `UCM_LOG_LEVEL`, log rate limiting, and Lite Connector trace mode.

## Configuration

Place these options inside the store connector configuration, alongside `storage_backends`:

```yaml
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: /data/ucm-cache
      posix_io_engine: aio
      io_direct: true
      posix_io_trace_enable: true
      posix_io_trace_dir: /local_nvme/ucm-io-trace
      posix_io_trace_buffer_mb: 64
      posix_io_trace_flush_ms: 10
use_layerwise: true
```

The directory is required when enabled and is created automatically. An unusable trace destination fails store setup. Each store instance writes a separate `posix-<pid>-<realtime_ns>-<store_id>.bin` file. Startup logs include `Posix IO tracing is enabled` and its path. Use a local trace disk when possible to avoid contention with the KV backend. Files grow for the duration of the run; there is no rotation.

The buffer setting is the total record-ring budget **per Posix store instance**, not per thread. Queues are rounded down to powers of two within this budget. Metadata and the writer batch add a small fixed overhead. Defaults are 64 MiB and 10 ms; the valid buffer range is 1–4096 MiB and the flush interval is 1–60000 ms. Too small a buffer for the configured worker count fails setup. Longer flush intervals can overflow the completion thread's ring under high-IOPS AIO workloads even when other rings have spare space; check `dropped` and shorten the interval or increase the buffer if needed.

## Export and filter

Run the standalone decoder from a source checkout; it uses only the Python standard library:

```bash
python ucm/store/posix/trace.py /local_nvme/ucm-io-trace/*.bin > io.jsonl
python ucm/store/posix/trace.py /local_nvme/ucm-io-trace/*.bin --task-id 812
python ucm/store/posix/trace.py /local_nvme/ucm-io-trace/*.bin --block-id abcdef0123456789abcdef0123456789
python ucm/store/posix/trace.py /local_nvme/ucm-io-trace/*.bin --file abcdef --format csv > io.csv
```

Each event includes:

| Field | Meaning |
| --- | --- |
| `run_id`, `pid`, `store_id` | Distinguish processes, restarts and multiple Posix stores |
| `task_id` | Posix `TransTask.id`; not a Cache task ID or inference request ID |
| `io_id`, `tid` | Unique event ID within this trace and the thread that recorded completion |
| `op` | `OPEN`, `READ`, `WRITE`, `CLOSE`, `RENAME`, `REMOVE`, `SYNC`, `TIMEOUT` |
| `block_id`, `shard_index` | UCM block and shard; file-level operations have no shard |
| `file`, `rename_to` | Full file name at operation time, including `.tmp`; rename destination |
| `start_time`, `start_ns`, `end_ns`, `duration_us` | UTC time calibrated at startup, monotonic timestamps and elapsed time |
| `offset`, `requested_bytes`, `actual_bytes` | Byte offset, requested size and actual read/write result |
| `result`, `errno`, `error_name`, `status` | Raw syscall result, captured errno and `OK`/`ERROR`/`SHORT_IO` |
| `async_io`, `submit_failed`, `not_started` | AIO operation, submission failure, or cancelled queued open |
| `source` | `transfer`, `health`, or `gc`; health/GC operations have no transfer task ID |

Records are grouped by producer when drained and are **not globally time ordered**. Sort by `start_ns` when reconstructing a timeline. For overlapping operations compare both start and end times. The `file` field describes the path used to open the descriptor; later renames do not change this recorded name. Use `RENAME` events to track that transition.

For synchronous operations timing surrounds the syscall. AIO timing starts in `ReadAsync`/`WriteAsync` before preparing/tracking/submitting the request, includes `io_submit` retries, and ends when completion or cancellation is observed. It is application-observed latency, not device service time. A successful WRITE completion does not imply the `.tmp` file has been committed; inspect the separate RENAME result. It also does not imply power-loss durability.

TIMEOUT is a task-level notification, not physical IO completion. A timed-out, uncancellable AIO may still complete later. Queued opens removed during cancellation are marked `not_started=true`. Only attempted operations and explicit queued-open cancellation are recorded: work skipped because its parent task already failed does not invent an IO event. Pending operations abandoned by engine teardown do not acquire a fictitious completion record.

## Completeness and cost

After the first registration of each producer thread, the hot path records timestamps and fixed-size 96-byte events in preallocated SPSC rings. Each actual producer gets its own ring, including cancellation callers. Path reconstruction, time formatting and JSON/CSV conversion happen offline. A background writer batches writes without per-event logging or `fsync`. Disabled tracing creates no buffers, writer or files and does not read clocks.

The writer periodically appends counters and drains queued records during normal teardown, after the store's IO/GC threads stop. The final ordinary log reports `produced`, `written`, `dropped`, `queue_high_water`, and `write_errors`. queue_high_water is an upper estimate of the maximum occupancy of any one producer ring. A full queue drops events instead of blocking inference. The number of producer slots is bounded by configured worker counts plus 32 spare slots; unusually high caller thread churn can exhaust slots and also increments `dropped`. Slow/full trace disks may therefore produce incomplete evidence without blocking IO workers.

The decoder writes a summary to stderr even when filters are used. Exit code 0 requires a final footer, zero drops/write errors, and matching event counts. Exit code 2 indicates incomplete or invalid evidence (including a running trace or crash-truncated tail). A complete trace means all **recorded completion events** were persisted; it does not assert all submitted tasks succeeded or all pending IO completed. No `fsync` is performed, so this is diagnostic evidence, not a durable audit log. A writer failure emits an ordinary ERROR and leaves the trace incomplete.

This trace covers KV files, health-probe data files, and GC deletion of KV files. Directory setup, lookup `access`/`stat`, GC lock/heartbeat metadata, and the trace file's own IO are outside its scope. It records no KV tensor contents or checksums. Combine block/shard IDs with connector KV checksum diagnostics to investigate data corruption; request-to-Cache-to-Posix task linkage is not added by this feature.

Measure throughput, latency, CPU and `dropped` with tracing off/on for the target workload. The impact depends on operation size, concurrency and trace storage throughput.
