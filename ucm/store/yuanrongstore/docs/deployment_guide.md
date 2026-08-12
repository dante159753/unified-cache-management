# UCM-YuanRongStore 部署指导

本文档描述 vLLM-Ascend + UCM + YuanRong的部署视图与部署流程。YuanRongStore 是 UCM 的一种 Store 实现，通过 YuanRong DataSystem 的异构传输（`HeteroClient::MSetD2H/MGetH2D`）与共享内存能力，完成 Device 与 Host Cache 之间的 KV Cache 传输，可选串联 Posix 作为持久化后端。

本文基于vllm-ascend：0.23.0rc1进行部署

## 1. 部署视图

### 1.1 组件拓扑

```mermaid
flowchart TB
    %% 样式统一美化定义
    classDef clientBox fill:#e6f7ff,stroke:#1890ff,stroke-width:1.2px
    classDef vllmBox fill:#f0f8ff,stroke:#2385bb,stroke-width:1.2px
    classDef yrBox fill:#f6ffed,stroke:#389e0d,stroke-width:1.2px
    classDef hardwareBox fill:#fff7e6,stroke:#fa8c16,stroke-width:1.2px
    classDef component fill:#ffffff,stroke:#444,stroke-width:1px
    classDef linkText font-size:11px

    subgraph Client["🖥️ 客户端"]
        REQ["OpenAI API / offline_inference.py"]:::component
    end
    class Client clientBox

    subgraph Node["⚙️ 推理节点（Ascend NPU 服务器）"]
        direction TB

        subgraph VLLM["📦 vLLM-Ascend 容器"]
            direction TB
            VS["vllm serve<br/>(Scheduler + Worker)"]:::component
            UC["UCMConnector<br/>ucm.integration.vllm.ucm_connector"]:::component
            UPS["UcmPipelineStore"]:::component
            YS["YuanRongStore<br/>libyuanrongstore.so"]:::component
            HC["HeteroClient / KVClient<br/>yr.datasystem SDK"]:::component
            PS["PosixStore（可选）"]:::component

            VS --> UC
            UC --> UPS
            UPS --> YS
            YS --> HC
            YS -. store_backend .-> PS
        end
        class VLLM vllmBox

        subgraph YR["🗄️ YuanRong DataSystem"]
            direction TB
            YW["YuanRong Worker<br/>host:port (默认 18481)"]:::component
            YHC["YuanRong Host Cache<br/>共享内存"]:::component
            YW --- YHC
        end
        class YR yrBox

        NPU["🧠 NPU Device KV Cache<br/>davinci 0..N"]:::hardwareBox
        DISK["💾 本地磁盘 / NFS（/mnt/ucm）"]:::hardwareBox

        HC <-->|"D2H / H2D<br/>Remote H2D"| YW
        HC --- YHC
        PS <--> DISK
        VS --- NPU
    end

    REQ -->|"HTTP /v1/... :server_port"| VS
```

### 1.2 组件职责

| 组件                                               | 部署形态                                                  | 职责                                                    |
| -------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| vLLM-Ascend                                        | 容器/进程                                                 | 推理引擎，通过 `--kv-transfer-config` 挂载 UCMConnector |
| UCM (UCMConnector + PipelineStore + YuanRongStore) | 随 vLLM 进程加载的 Python 包 + `.so`                      | KV Cache 的 Lookup/Load/Dump 调度                       |
| YuanRong DataSystem (Worker + SDK)                 | 独立部署的 Worker 进程；SDK 以 pip 包形式在 vLLM 容器安装 | 异构 D2H/H2D 传输与 Host 共享内存缓存                   |
| PosixStore（可选）                                 | 随 UCM 加载                                               | 本地磁盘/NFS 持久化与冷恢复                             |

数据路径：

```text
YuanRong:       Device <-> YuanRong Host Cache
YuanRong|Posix: Device <-> YuanRong Host Cache <-> Posix
```

## 2. 部署前置条件

1. **Ascend 环境**：已安装 CANN toolkit（`/usr/local/Ascend`），`ascend-toolkit/set_env.sh` 与 `nnal/atb/set_env.sh` 可正常 source；NPU 驱动就绪（`/dev/davinci*`、`/dev/davinci_manager` 等设备节点可见）。
2. 【HDK】≥ 25.5.0
3. 【CANN版本】≥ 9.1.0
4. **vLLM-Ascend 镜像**：`quay.io/ascend/vllm-ascend:0.23.0rc1-a3`
5. **存储目录**：Posix 后端目录（如 `/mnt/ucm`）已挂载且可读写；多机场景使用 NFS 共享。

## 3. 部署流程

### 3.1 配置内存大页

根据Hugepagesize大小选择配置方法，在每个宿主机执行对应命令，分配500G大页内存

>  查看页大小：`grep Huge /proc/meminfo`；

**Hugepagesize 2M**

```bash
# 分配256000页共500G
echo 256000 > /proc/sys/vm/nr_hugepages
```

**Hugepagesize 512M**

```sh
# 分配1000页共500G
echo 1000 > /proc/sys/vm/nr_hugepages
```

执行`grep Huge /proc/meminfo`查看是否配置成功

### 3.2 安装并启动 etcd

>   示例为单实例部署，etcd 只需在节点0安装和启动，其他节点无需操作。

#### 安装 etcd

Yuanrong 服务启动脚本依赖 `etcd` 和 `etcdctl`。至少在节点0安装，其他节点的元戎worker连接同一个 etcd。

```bash
ETCD_VERSION="v3.5.12"
if [ "$(uname -m)" = "aarch64" ]; then
  ETCD_ARCH="linux-arm64"
else
  ETCD_ARCH="linux-amd64"
fi
wget https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-${ETCD_ARCH}.tar.gz
tar -xvf etcd-${ETCD_VERSION}-${ETCD_ARCH}.tar.gz
cd etcd-${ETCD_VERSION}-${ETCD_ARCH}
cp etcd etcdctl /usr/local/bin/
```

> 若已有安装包，则跳过 wget 下载步骤，直接执行以下命令：
>
> ```bash
> tar -xvf etcd-v3.5.12-linux-arm64.tar.gz
> cd etcd-v3.5.12-linux-arm64
> cp etcd etcdctl /usr/local/bin/
> ```

验证安装

```bash
etcd --version
etcdctl version
```

#### 启动 etcd

创建启动脚本 `run_etcd.sh`，在节点0启动 etcd：

```sh
#!/bin/bash

export ETCD_IP="<节点IP>"
export ETCD_PORT=2379
export ETCD_PEER_PORT=2380

etcd \
  --name etcd-single \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:${ETCD_PORT} \
  --advertise-client-urls http://${ETCD_IP}:${ETCD_PORT} \
  --listen-peer-urls http://0.0.0.0:${ETCD_PEER_PORT} \
  --initial-advertise-peer-urls http://${ETCD_IP}:${ETCD_PEER_PORT} \
  --initial-cluster etcd-single=http://${ETCD_IP}:${ETCD_PEER_PORT} \
  > /tmp/etcd.log 2>&1 &

sleep 3

etcdctl --endpoints "${ETCD_IP}:${ETCD_PORT}" put key "value"
etcdctl --endpoints "${ETCD_IP}:${ETCD_PORT}" get key

echo "ETCD start finished, log dir: /tmp/etcd.log"
```

验证：

```sh
# 方式1，预期输出：{"health":"true","reason":""}
etcdctl --endpoints "${ETCD_IP}:${ETCD_PORT}" endpoint health

# 方式2，预期输出：100.100.xxx.xxx:2379 is healthy: successfully committed proposal: took = 1.43913ms
curl -L http://${ETCD_IP}:${ETCD_PORT}/health
```

### 3.3 安装ucm

```sh
pip install wrapt-1.17.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

pip install uc_manager-0.5.0-cp312-cp312-linux_aarch64.whl --force-reinstall --no-deps
```

### 3.4 配置 UCM Connector

新建文件`ucm_yuanrong_config.yaml`

**推荐配置一：**

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "YuanRong|Posix"		# 纯YuanRong不带Posix 时改为 "YuanRong"
      yuanrong_host: "127.0.0.1"				  # 指向 3.3 启动的 Worker
      yuanrong_port: 18483
      yuanrong_enable_remote_h2d: true		# 开启rh2d
      yuanrong_timeout_ms: 20000
      yuanrong_waiting_queue_depth: 8192
      yuanrong_load_worker_count: 4
      # Posix cold recovery pipeline batch size.
      yuanrong_dump_prerequisite_worker_count: 2
      yuanrong_recovery_batch_size: 32
      yuanrong_h2d_stream_count: 1
      yuanrong_backfill_worker_count: 4
      yuanrong_backfill_queue_depth: 128
      yuanrong_posix_max_inflight_gb: 1

      #share_buffer_enable: true
      storage_backends: "/mnt/ucm"        # Posix 后端目录
      posix_io_engine: "psync"			  # aio 仅 io_direct=true
      io_direct: false					  # 为true时，元戎worker启动参数需要带`memory_alignment=4096`

enable_event_sync: true
use_layerwise: false
```



**推荐配置二：**

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "YuanRong|Posix"		# 纯YuanRong不带Posix 时改为 "YuanRong"
      yuanrong_host: "127.0.0.1"			# 指向 3.3 启动的 Worker
      yuanrong_port: 18483
      yuanrong_enable_remote_h2d: true		# 开启rh2d
      yuanrong_timeout_ms: 20000
      yuanrong_waiting_queue_depth: 8192
      yuanrong_load_worker_count: 4
      # Posix cold recovery pipeline batch size.
      yuanrong_dump_prerequisite_worker_count: 2
      yuanrong_recovery_batch_size: 32
      yuanrong_h2d_stream_count: 1
      yuanrong_backfill_worker_count: 4
      yuanrong_backfill_queue_depth: 128
      yuanrong_posix_max_inflight_gb: 1

      #share_buffer_enable: true
      storage_backends: "/mnt/ucm"      # Posix 后端目录
      posix_io_engine: "aio"			# aio 仅 io_direct=true
      io_direct: true					# 为true时，元戎worker启动参数需要带`memory_alignment=4096`

enable_event_sync: true
use_layerwise: false
```





### 3.4 安装并启动 Yuanrong 服务

#### 安装元戎

```
pip install openyuanrong_datasystem-0.9.2-cp312-cp312-manylinux_2_35_aarch64.whl --force-reinstall
```

#### 启动元戎

所有节点都需要启动 Yuanrong Worker，并连接到节点 0 上的 etcd，根据3.4中`ucm_connectors`不同的配置选择不同的参数。

> 注意：ucm中io_direct: true时如下脚本需要带上--memory_alignment 4096，为false时去除

创建 `run_yr_worker.sh`，修改`HOST_IP`、`ETCD_IP`、`SHM_SIZE`：

```bash
#!/bin/bash

export HOST_IP="<当前节点IP>"
export ETCD_IP="<ETCD节点IP>"
export WORKER_PORT=18481
export ETCD_PORT=2379
export SHM_SIZE=512000	# 共享内存大小，500G

dscli start -t 600 -w \
  --worker_address ${HOST_IP}:${WORKER_PORT} \
  --etcd_address ${ETCD_IP}:${ETCD_PORT} \
  --shared_memory_size_mb ${SHM_SIZE} \
  --node_timeout_s 20 \
  --node_dead_timeout_s 30 \
  --log_only_write_info_file false \
  --rpc_thread_num 64 \
  --sc_regular_socket_num 0 \
  --sc_stream_socket_num 0 \
  --oc_thread_num 64 \
  --arena_per_tenant 1 \
  --enable_huge_tlb true \
  --spill_directory "/spill_ssd_dir" \
  --spill_size_limit 100000000000 \
  --remote_get_policy MEMORY_ONLY \
  --enable_memory_rebalance true \
  --enable_worker_worker_batch_get true \
  --remote_h2d_device_ids "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" \
  --remote_h2d_link_type "HCCS" \
  --remote_h2d_hccs_buffer_pool "4:8" \
  --memory_alignment 4096
```

内存大页相关参数：

-   arena_per_tenant： arena数量
-   enable_huge_tlb：开启大页

SSD相关参数：

-   spill_directory: 本地SSD目录
-   spill_size_limit：本地SSD目录大小上限，单位Byte，此处为100G
-   remote_get_policy MEMORY_ONLY: 不读取远端SSD策略

内存rebalance参数：

-   enable_memory_rebalance：开启rebalance功能，向空闲节点溢出数据

RH2D相关参数：

-   enable_worker_worker_batch_get: 批量获取

-   remote_h2d_device_ids：设置npu卡资源

RH2D over HCCS需新增：

-   remote_h2d_link_type "HCCS"
-   remote_h2d_hccs_buffer_pool "4:8"

io_direct相关参数（ucm的io_direct配置为true时需配置）

-   memory_alignment： 内存对齐

### 3.6 添加补丁

添加补丁到VLLM-ascend

```
cd /vllm-workspace/vllm-ascend
git am /workspace/0001-delete-yuanrong_backend-param-enable_exclusive_conne.patch
```
### 3.7 升级CANN

升级cann到9.1.0

```
Ascend-cann-A3-ops_9.1.0~weekly.20260708.01_linux-aarch64.run
Ascend-cann-toolkit_9.1.0~weekly.20260708.01_linux-aarch64.run
```

### 3.8 拉起服务

```sh
export LD_LIBRARY_PATH=/usr/local/python3.12.13/lib/python3.12/site-packages/yr/datasystem/lib:$LD_LIBRARY_PATH
export DATASYSTEM_LOG_ONLY_WRITE_INFO_FILE=false
export DS_RH2D_LINK_TYPE=HCCS   # RH2D开启HCCS时需要配置

.... 	# 	其他配置省略

vllm serve /tmp/GLM-5.1-W8A8 \
... 	# 	其他配置省略
  --kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/workspace/ucm_yuanrong_config.yaml"}
}'
```
