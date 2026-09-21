# 存储健康检查与熔断

推理服务能够响应，并不代表缓存后端可用。Store 健康状态说明 Pipeline 阶段是否接收新的缓存操作，不能证明某个 KV 块存在、模型回答正确，或部署达到了延迟目标。

## 后端故障时会发生什么

启用健康检查后，Pipeline 为每个加载的阶段包装 `HealthBreakerStore`。包装器初始允许操作，主动探测和被动 I/O 使用独立窗口：主动窗口默认保留八次探测结果，其中两次失败即熔断；被动窗口默认保留 60 个一秒桶，窗口内至少有 10 个有效任务结果且失败比例严格大于 1% 时熔断。

恢复只能由主动探测触发：熔断后重新积累完整的成功探测窗口，冷却期结束后再启动一次成功探测才允许新操作。首次冷却 60 秒；恢复后不足 3600 秒再次熔断，冷却时间乘以 2，最多 3600 秒；稳定运行至少 3600 秒后再次熔断则回到 60 秒。冷却期间继续主动探测，失败不会反复延长本次冷却。

| 被阻止期间的操作 | 结果 |
| --- | --- |
| `Lookup` | 返回所有请求块均未命中 |
| `LookupOnPrefix` / `LookupOnReverse` | 返回未命中（`-1`） |
| `Prefetch` | 不提交新的预取 |
| `Load` / `Dump` | 返回 `StoreUnhealthy` |
| 已有任务的 `Check` / `Wait` | 继续转发给底层 Store |

阻止新操作不会取消已经提交的传输。缓存未命中后重计算还是返回错误，由引擎集成和请求路径决定；熔断器本身不会重试推理请求。

基类 `StoreV1.CheckHealth()` 直接返回成功，Posix 和 Mooncake 覆盖该方法执行真实探测。Pipeline 组装时只为 Posix 和 Mooncake 开启被动检测，Cache 等阶段不参与；下游故障由下游阶段的熔断器处理。

被动统计按任务计数：提交成功的任务只在首次 `Wait` 返回最终状态时计一次。`Load` / `Dump` 的提交结果和 `Check` 轮询均不进入被动窗口或被动失败指标。`NotFound`（含缓存缺失和短读）、`StoreUnhealthy`、`InvalidParam`、`DuplicateKey`、`Unsupported` 不进入分子或分母，其余失败（含 `Retry`、超时和 I/O 错误）计为失败。比例为有效桶的失败总数除以任务总数；按结果被观察到的单调时钟时间入桶，边界精度为一秒。主动探测产生的 I/O 不进入被动窗口。恢复时清空被动窗口；如果一次 `Wait` 或主动探测执行期间发生状态切换，其结果不进入新窗口。提交时不跟踪任务代次，恢复后才开始的 `Wait` 按当前窗口统计；已有任务仍须完成等待和资源释放。

各桶通过原子操作同时更新时间戳和计数，复用时无需轮转锁，并忽略旧时间戳的迟到更新。恢复清零仍可能与在途更新重叠，允许恢复边界附近少量统计误差。

启用被动检测时，有效的 `Wait` 失败在本层统计后统一返回 `StoreUnhealthy`，未达到熔断阈值时也会转换；返回消息保留原始状态。只要中间 store 保留该状态码，上层熔断器就不会重复计数。

## 设置探测策略

`store_health` 放在 `ucm_connector_config` 内。下面的`Cache|Posix` 示例显式填写了默认健康策略：

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

模型、挂载和缓存参数仍须符合实际部署，详见[存储流水线](../../developer-guide/cache-configuration/pipeline.md)。

| 字段 | 含义 |
| --- | --- |
| `enabled` | 创建健康包装器和探测线程；默认 `true` |
| `health_check_interval_s` | 目标探测间隔；默认 10 秒 |
| `health_check_timeout_s` | 探测执行期限；默认 3 秒 |
| `health_window_size` | 保留的最近结果数量；默认 8 |
| `failure_threshold` | 阻止新操作所需的失败次数；默认 2 |
| `passive_enabled` | 启用 Posix、Mooncake 的被动检测；默认 `true`，Cache 等阶段不参与 |
| `passive_window_s` | 一秒桶数量，正整数；默认 60 |
| `passive_min_samples` | 被动窗口触发判定所需的最少有效任务数，正整数；默认 10 |
| `passive_failure_ratio` | 被动失败比例阈值，范围 `[0, 1)`，严格大于时熔断；默认 0.01 |
| `initial_cooldown_s` / `max_cooldown_s` | 首次 / 最大冷却时间；默认 60 / 3600 秒 |
| `backoff_factor` | 短期再次熔断的冷却倍数，至少 1；默认 2 |
| `stable_reset_after_s` | 恢复后持续正常运行多久重置退避；默认 3600 秒 |

窗口、探测间隔、探测超时和稳定期必须为正数，主动失败阈值不能大于窗口长度，探测超时必须短于探测间隔。冷却时间允许为零，最大值不能小于初始值。首次探测会等待一个间隔，再加不超过一个间隔的随机延迟。首次探测前就会发布初始允许状态，因此初始状态不能作为 I/O 成功的证据。

设置 `enabled: false` 会撤掉这层保护，不会修复后端故障，也不会关闭后端自身的错误处理。

## 理解探测实际检查了什么

**Posix** 遍历存储布局选定的健康检查路径，创建临时文件、写入 4096 字节、读回比较并删除。Buffered I/O 还会调用 `Sync`，Direct I/O 则使用已配置的直接打开标志。打开、传输、同步、比较或删除失败都会使探测失败。路径使用 NFS 时，检查的是该 UCM 进程实际看到的文件系统挂载。

**Mooncake** 在专用 key 下写入八字节测试值，读回比较后删除。传输路径使用实际客户端，scheduler 路径使用 RPC 客户端。这检查的是已配置的客户端通路，不会验证所有模型的 KV 布局或每个远端副本。

健康检查会产生自己的小文件或 key，不能把这些对象当成请求已保存可复用 KV 数据的证据。

## 先检查单条状态，再考虑聚合

默认 vLLM connector 导出以下指标：

| 后端 | 状态 Gauge | 探测 Counter |
| --- | --- | --- |
| Posix | `ucm:posix_store_health` | `ucm:posix_healthy_count_total`、`ucm:posix_unhealthy_count_total` |
| Mooncake | `ucm:mooncake_store_health` | `ucm:mooncake_healthy_count_total`、`ucm:mooncake_unhealthy_count_total` |

Gauge 为 1 表示包装器允许操作，为 0 表示阻止操作。Counter 记录探测成功、失败及超时，不记录状态迁移次数。一次成功探测不一定会使被阻止的 Gauge 恢复为 1。

`ucm:posix_passive_failures_total` 和 `ucm:mooncake_passive_failures_total` 单独累计被动检测观察到的有效失败（包括迟到的旧任务失败），不与主动探测 Counter 混算，也不等于当前被动窗口的失败数量。被动熔断会立即更新原生健康 Gauge，其导出仍取决于 connector 的统计采集节奏。

先查看保留完整标签的各条序列：

```promql
ucm:posix_store_health{job="vllm"}
```

再查看最近窗口中的失败：

```promql
increase(ucm:posix_unhealthy_count_total{job="vllm"}[5m])
```

这些查询假设抓取任务名为 `vllm`。定位问题时保留 `instance`、模型、engine 和 `worker_rank` 标签。Scheduler 的标签值为 `worker_rank="scheduler"`。多个进程可能探测同一个后端，导出的标签也不能标识每个物理挂载或底层流水线对象，因此序列数量不是故障磁盘数量。

原生探测线程和 Prometheus 抓取采用不同的调度节奏。在 vLLM connector 路径中，原生统计通过 `get_kv_connector_stats()` 进入导出器。把重复数值或缺失序列解释为后端当前状态之前，应先确认统计采集仍在推进。

## 定位故障并确认恢复

后台线程按 `health_check_interval_s`（默认 10 秒）的节奏检查被动窗口。窗口内有失败时，打印 `Store passive health window`，包含阶段、窗口秒数、有效结果数、失败数、失败比例、最小样本数和阈值；不足最小样本数时也会打印。没有失败则不打印。统计按当前时间排除过期桶，无新 I/O 时旧失败也会自然过期；恢复清空窗口后停止打印。

1. 检查抓取目标，根据标签找到对应进程。
2. 在其日志中查找 `Store health check` 失败和 `transitioned to UNHEALTHY`；迁移日志给出阶段标识、`source=active_probe` 或 `source=passive_io`、冷却毫秒数和状态代次。
3. Posix 检查该进程的挂载、权限、可用容量及读写删除错误；Mooncake 检查客户端、metadata/master 连接和具体操作错误。
4. 修复依赖后，等待主动成功窗口满足要求且冷却结束，并确认 `transitioned to HEALTHY` 及对应 Gauge 更新。
5. 单独重做[外部缓存验证](../quick_start/index.md#vllm-verify-the-service-and-external-cache)。探测恢复不能证明某个请求的缓存恢复。

指标单位和导出路径见[指标参考](metrics-reference.md)。策略和操作行为分别定义在
[`StoreHealthConfig`](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/ucm/store/pipeline/cc/store_health_config.h)
和 [`HealthBreakerStore`](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/ucm/store/pipeline/cc/health_breaker_store.cc)。
