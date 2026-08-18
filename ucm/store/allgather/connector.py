import copy
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from ucm.integration.vllm.device import create_device
from ucm.logger import init_logger
from ucm.shared.metrics import ucmmetrics
from ucm.store.pipeline.connector import UcmPipelineStore
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1

logger = init_logger(__name__)

_MAX_AIV_CORES = 40
_COPY_CHUNK_BYTES = 32 * 1024
_DESCRIPTOR_BYTES = 3 * np.dtype(np.int64).itemsize
_CHUNK_LAYOUT_FIELDS = 4
_DEFAULT_HCCL_BUFFER_MB = 8
_COLLECTIVE_MODES = {
    "auto": 0,
    "default": 0,
    "host": 1,
    "host_ts": 1,
    "aicpu": 2,
    "ai_cpu": 2,
    "aicpu_ts": 2,
    "ai_cpu_ts": 2,
    "aiv": 3,
}


def _collective_mode(value: object) -> tuple[str, int]:
    normalized = str(value).strip().lower()
    try:
        mode = _COLLECTIVE_MODES[normalized]
    except KeyError as error:
        raise ValueError(
            "allgather_collective_mode must be auto, host, aicpu_ts, or aiv"
        ) from error
    canonical = {0: "auto", 1: "host", 2: "aicpu_ts", 3: "aiv"}[mode]
    return canonical, mode


@dataclass
class _BufferSlot:
    index: int
    send_buffer: torch.Tensor
    in_use: bool = field(default=False, init=False)


@dataclass
class _CopySlot(_BufferSlot):
    descriptor_buffer: torch.Tensor
    offset_buffer: torch.Tensor
    host_descriptors: np.ndarray
    host_offsets: np.ndarray
    host_descriptor_tensor: torch.Tensor
    host_offset_tensor: torch.Tensor


@dataclass
class _LoadSlot(_BufferSlot):
    receive_buffer: torch.Tensor = None
    completion_event: object = None
    reclaimable: bool = False
    sequence: int = 0


@dataclass
class _DumpSlot(_CopySlot):
    pass


@dataclass
class _LoadWindowPlan:
    row_count: int
    metadata_offset: int
    collective_blocks: int
    owned_block_ids: List[bytes]
    owned_source_rows: List[int]
    owned_addrs_by_slot: List[np.ndarray]


@dataclass
class _LoadPlan:
    source_row_count: int
    row_order: np.ndarray
    route_rows: np.ndarray
    windows: List[_LoadWindowPlan]


@dataclass
class _LoadMetadataArena:
    destination_buffer: torch.Tensor
    route_buffer: torch.Tensor
    destination_capacity: int
    route_capacity: int
    in_use: bool = False


@dataclass
class _PreparedLoadBatch:
    arena: _LoadMetadataArena
    host_destination_tensor: torch.Tensor
    host_route_tensor: torch.Tensor
    metadata_event: object = None
    released: bool = False
    active_tasks: int = 0


@dataclass
class PreparedLoadRequest:
    batch: _PreparedLoadBatch
    plan: _LoadPlan
    destination_offset: int
    route_offset: int
    layer_count: int
    store_id: int


@dataclass
class _LoadWindow:
    plan: _LoadWindowPlan
    owned_shard_indices: List[int]
    slot: Optional[_LoadSlot] = None
    inner_task: Optional[Task] = None
    error: Optional[BaseException] = None
    completed: bool = False


@dataclass
class _DumpWindow:
    slot: _DumpSlot
    inner_task: Task
    event_handle: int
    completed: bool = False


@dataclass
class AllGatherTask(Task):
    operation: str
    inner_task: Optional[Task] = None
    load_windows: List[_LoadWindow] = field(default_factory=list)
    active_load_windows: deque = field(default_factory=deque)
    next_load_window: int = 0
    load_error: Optional[BaseException] = None
    destination_offset: int = 0
    route_offset: int = 0
    destination_buffer: Optional[torch.Tensor] = None
    route_buffer: Optional[torch.Tensor] = None
    host_destination_tensor: Optional[torch.Tensor] = None
    host_route_tensor: Optional[torch.Tensor] = None
    prepared_batch: Optional[_PreparedLoadBatch] = None
    metadata_event: object = None
    dump_windows: List[_DumpWindow] = field(default_factory=list)
    passthrough: bool = False
    completed: bool = False
    terminal_error: Optional[BaseException] = None


class _LoadProgressManager:
    def __init__(
        self,
        device_id: int,
        group: object = None,
        registry_key: object = None,
        owns_group: bool = False,
    ) -> None:
        self.group = group
        self.registry_key = registry_key
        self._owns_group = owns_group
        self._device_id = device_id
        self._condition = threading.Condition()
        self._queue = deque()
        self._connectors = set()
        self._stop = False
        self._error = None
        self._thread = None

    @property
    def error(self) -> Optional[BaseException]:
        with self._condition:
            return self._error

    def register(self, connector: "UcmAllGatherStore") -> None:
        with self._condition:
            self._connectors.add(connector)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._loop,
                    name=f"ucm-allgather-load-{self._device_id}",
                    daemon=True,
                )
                self._thread.start()

    def submit(self, connector: "UcmAllGatherStore", task: AllGatherTask) -> None:
        with self._condition:
            if self._error is not None:
                raise self._error
            self._queue.append((connector, task))
            self._condition.notify()

    def release(self, connector: "UcmAllGatherStore") -> None:
        thread = None
        with self._condition:
            self._connectors.remove(connector)
            if not self._connectors:
                self._stop = True
                self._condition.notify_all()
                thread = self._thread
        if thread is None:
            return
        thread.join()
        if self._owns_group and self.group is not None:
            dist.destroy_process_group(self.group)
        if self.registry_key is not None:
            with _SHARED_PROGRESS_MANAGERS_LOCK:
                current = _SHARED_PROGRESS_MANAGERS.get(self.registry_key)
                if current is self:
                    del _SHARED_PROGRESS_MANAGERS[self.registry_key]

    def _loop(self) -> None:
        try:
            torch.npu.set_device(self._device_id)
            progress_stream = torch.npu.Stream()
            with torch.npu.stream(progress_stream):
                while True:
                    with self._condition:
                        while not self._queue and not self._stop:
                            self._condition.wait()
                        if self._stop:
                            return
                        connector, task = self._queue.popleft()
                    connector._finish_load_task(task)
        except BaseException as error:
            logger.exception("allgather load progressor failed")
            with self._condition:
                self._error = error
                self._queue.clear()
                connectors = list(self._connectors)
                self._condition.notify_all()
            for connector in connectors:
                connector._abort_load_tasks(error)


_SHARED_PROGRESS_MANAGERS = {}
_SHARED_PROGRESS_MANAGERS_LOCK = threading.Lock()


def _acquire_shared_progress_manager(
    device_id: int, hccl_buffer_mb: int, collective_mode: int
) -> _LoadProgressManager:
    from vllm.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    backend = getattr(tp_group, "backend", None)
    if backend is None:
        backend = dist.get_backend(tp_group.device_group)
    group_ranks = tuple(tuple(ranks) for ranks in tp_group.group_ranks)
    key = (device_id, str(backend), group_ranks, hccl_buffer_mb, collective_mode)
    with _SHARED_PROGRESS_MANAGERS_LOCK:
        manager = _SHARED_PROGRESS_MANAGERS.get(key)
        if manager is not None:
            return manager
        local_group = None
        pg_options = None
        if "hccl" in str(backend).lower():
            import torch_npu

            pg_options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
            pg_options.hccl_config = {
                "hccl_buffer_size": hccl_buffer_mb,
                "hccl_op_expansion_mode": collective_mode,
            }
        for ranks in group_ranks:
            group = dist.new_group(list(ranks), backend=backend, pg_options=pg_options)
            if tp_group.rank in ranks:
                local_group = group
        dist.barrier(group=local_group)
        manager = _LoadProgressManager(
            device_id,
            group=local_group,
            registry_key=key,
            owns_group=True,
        )
        _SHARED_PROGRESS_MANAGERS[key] = manager
        return manager


class _FusedBufferPool:
    def __init__(
        self,
        device_id: int,
        world_size: int,
        shard_size: int,
        chunk_layout: np.ndarray,
        requested_window_blocks: int,
        load_slot_count: int,
        dump_slot_count: int,
        capacity_mb: int,
    ) -> None:
        if requested_window_blocks <= 0:
            raise ValueError("allgather_window_blocks_per_rank must be positive")
        if load_slot_count <= 0 or dump_slot_count <= 0:
            raise ValueError("allgather load and dump slot counts must be positive")
        if capacity_mb <= 0:
            raise ValueError("allgather_fused_buffer_capacity_mb must be positive")

        capacity_bytes = capacity_mb * 1024 * 1024
        load_payload_copies = world_size + 1 if world_size > 1 else 1
        payload_bytes_per_block = shard_size * (
            load_slot_count * load_payload_copies + dump_slot_count
        )
        chunks_per_block = len(chunk_layout)
        dump_descriptor_bytes_per_block = (
            chunks_per_block * _DESCRIPTOR_BYTES * dump_slot_count
        )
        fixed_bytes = (
            dump_slot_count * (_MAX_AIV_CORES + 1) * np.dtype(np.int32).itemsize
            + chunk_layout.size * np.dtype(np.int64).itemsize
        )
        metadata_bytes_per_block = dump_descriptor_bytes_per_block
        bytes_per_block = payload_bytes_per_block + metadata_bytes_per_block
        maximum_window_blocks = (capacity_bytes - fixed_bytes) // bytes_per_block
        if maximum_window_blocks < 1:
            minimum_mb = (fixed_bytes + bytes_per_block + 1024 * 1024 - 1) // (
                1024 * 1024
            )
            raise ValueError(
                "allgather fused buffer capacity is too small: "
                f"capacity_mb={capacity_mb}, minimum_mb={minimum_mb}"
            )

        self.window_blocks = min(requested_window_blocks, maximum_window_blocks)
        self.load_slot_count = load_slot_count
        self.dump_slot_count = dump_slot_count
        self._load_sequence = 0
        self._device = f"npu:{device_id}"
        self.chunk_layout = torch.empty(
            chunk_layout.shape, dtype=torch.int64, device=self._device
        )
        self.chunk_layout.copy_(torch.from_numpy(chunk_layout))

        dump_descriptors = self.window_blocks * chunks_per_block
        self.load_slots = [
            self._make_load_slot(index, shard_size, world_size)
            for index in range(load_slot_count)
        ]
        self.dump_slots = [
            self._make_dump_slot(index, shard_size, dump_descriptors)
            for index in range(dump_slot_count)
        ]

        payload_bytes = self.window_blocks * payload_bytes_per_block
        metadata_bytes = self.window_blocks * metadata_bytes_per_block
        allocated_bytes = payload_bytes + metadata_bytes + fixed_bytes
        logger.info(
            "UcmAllGatherStore fused NPU pool: "
            f"requested_window_blocks_per_rank={requested_window_blocks}, "
            f"window_blocks_per_rank={self.window_blocks}, "
            f"load_slots={load_slot_count}, dump_slots={dump_slot_count}, "
            f"payload_bytes={payload_bytes}, metadata_bytes={metadata_bytes}, "
            f"allocated_bytes={allocated_bytes}, capacity_bytes={capacity_bytes}"
        )

    def _make_copy_slot(
        self, index: int, payload_bytes: int, descriptor_count: int, zero: bool
    ) -> dict:
        allocator = torch.zeros if zero else torch.empty
        send_buffer = allocator(payload_bytes, dtype=torch.uint8, device=self._device)
        descriptor_buffer = torch.empty(
            (descriptor_count, 3), dtype=torch.int64, device=self._device
        )
        offset_buffer = torch.empty(
            _MAX_AIV_CORES + 1, dtype=torch.int32, device=self._device
        )
        host_descriptors = np.empty((descriptor_count, 3), dtype=np.int64)
        host_offsets = np.empty(_MAX_AIV_CORES + 1, dtype=np.int32)
        return {
            "index": index,
            "send_buffer": send_buffer,
            "descriptor_buffer": descriptor_buffer,
            "offset_buffer": offset_buffer,
            "host_descriptors": host_descriptors,
            "host_offsets": host_offsets,
            "host_descriptor_tensor": torch.from_numpy(host_descriptors),
            "host_offset_tensor": torch.from_numpy(host_offsets),
        }

    def _make_load_slot(
        self, index: int, shard_size: int, world_size: int
    ) -> _LoadSlot:
        payload_bytes = self.window_blocks * shard_size
        send_buffer = torch.empty(payload_bytes, dtype=torch.uint8, device=self._device)
        receive_buffer = send_buffer
        if world_size > 1:
            receive_buffer = torch.empty(
                world_size * payload_bytes, dtype=torch.uint8, device=self._device
            )
        return _LoadSlot(
            index=index,
            send_buffer=send_buffer,
            receive_buffer=receive_buffer,
            completion_event=torch.npu.Event(),
        )

    def _make_dump_slot(
        self, index: int, shard_size: int, descriptor_count: int
    ) -> _DumpSlot:
        payload_bytes = self.window_blocks * shard_size
        return _DumpSlot(
            **self._make_copy_slot(index, payload_bytes, descriptor_count, True)
        )

    def try_acquire_load(self) -> Optional[_LoadSlot]:
        for slot in self.load_slots:
            if slot.in_use and slot.reclaimable and slot.completion_event.query():
                slot.in_use = False
                slot.reclaimable = False
        available = next((slot for slot in self.load_slots if not slot.in_use), None)
        if available is None:
            reclaimable = [slot for slot in self.load_slots if slot.reclaimable]
            if not reclaimable:
                return None
            available = min(reclaimable, key=lambda slot: slot.sequence)
            wait_start = time.perf_counter()
            available.completion_event.synchronize()
            ucmmetrics.update_stats(
                {
                    "allgather_load_slot_reclaim_wait_ms": (
                        time.perf_counter() - wait_start
                    )
                    * 1000
                }
            )
            available.in_use = False
            available.reclaimable = False
        self._load_sequence += 1
        available.sequence = self._load_sequence
        available.in_use = True
        return available

    @staticmethod
    def complete_load(slot: _LoadSlot) -> None:
        slot.completion_event.record(torch.npu.current_stream())
        slot.reclaimable = True

    @staticmethod
    def release_load(slot: _LoadSlot) -> None:
        slot.in_use = False
        slot.reclaimable = False

    def try_acquire_dump(self) -> Optional[_DumpSlot]:
        slot = next((slot for slot in self.dump_slots if not slot.in_use), None)
        if slot is not None:
            slot.in_use = True
        return slot

    @staticmethod
    def release_dump(slot: _DumpSlot) -> None:
        slot.in_use = False


class UcmAllGatherStore(UcmKVStoreBaseV1):
    def __init__(self, config: Dict[str, object]) -> None:
        super().__init__(config)
        injected_group = config.get("allgather_process_group")
        inner_config = copy.deepcopy(
            {
                key: value
                for key, value in config.items()
                if key != "allgather_process_group"
            }
        )
        self._worker = int(config.get("device_id", -1)) >= 0
        self._rank = int(config.get("allgather_rank", 0))
        self._world_size = int(config.get("allgather_world_size", 1))
        self._replicated_data = bool(config.get("allgather_replicated_data", False))
        self._scatter_only = bool(config.get("allgather_scatter_only", False))
        self._collective_enabled = (
            self._replicated_data and not self._scatter_only and self._world_size > 1
        )
        self._collective_count_crop = bool(
            config.get("allgather_collective_count_crop", True)
        )
        self._skip_load_collective = bool(
            config.get("allgather_load_skip_collective", False)
        )
        self._skip_load_scatter = bool(config.get("allgather_load_skip_scatter", False))
        self._load_storage_rank = self._rank if self._collective_enabled else 0
        self._load_storage_world_size = (
            self._world_size if self._collective_enabled else 1
        )
        self._dump_storage_rank = self._rank if self._replicated_data else 0
        self._dump_storage_world_size = self._world_size if self._replicated_data else 1
        self._tensor_sizes = [int(size) for size in config.get("tensor_size_list", [])]
        self._shard_size = int(config.get("shard_size", 0))
        self._logical_size = sum(self._tensor_sizes)
        self._device_id = int(config.get("device_id", -1))
        self._hccl_buffer_mb = 0
        self._collective_mode_name, self._collective_mode = _collective_mode(
            config.get("allgather_collective_mode", "host")
        )
        self._device = create_device() if self._worker else None
        self._copy_op = None
        self._tp_group = None
        self._load_progress_manager = None
        self._pool = None
        self._active_dump_windows = deque()
        self._load_tasks = deque()
        self._unprimed_load_tasks = deque()
        self._load_condition = threading.Condition()
        self._load_progress_error = None
        self._load_metadata_arenas = []
        self._closed = False

        if self._worker:
            if (
                self._world_size <= 0
                or self._rank < 0
                or self._rank >= self._world_size
            ):
                raise ValueError(
                    f"invalid all-gather rank {self._rank}/{self._world_size}"
                )
            if not self._tensor_sizes or self._shard_size < self._logical_size:
                raise ValueError(
                    "tensor_size_list must be non-empty and fit within shard_size"
                )
            library_dir = Path(__file__).parent
            for library in (
                "libucm_segmented_copy_kernels.so",
                "libucm_compact_scatter_kernels.so",
            ):
                torch.ops.load_library(str(library_dir / library))
            from ucm.store.allgather import ucm_segmented_copy

            self._copy_op = ucm_segmented_copy
            if self._collective_enabled:
                hccl_buffer_mb = int(
                    config.get("allgather_hccl_buffer_mb", _DEFAULT_HCCL_BUFFER_MB)
                )
                if hccl_buffer_mb <= 0:
                    raise ValueError("allgather_hccl_buffer_mb must be positive")
                self._hccl_buffer_mb = hccl_buffer_mb
                if injected_group is None:
                    self._load_progress_manager = _acquire_shared_progress_manager(
                        self._device_id, hccl_buffer_mb, self._collective_mode
                    )
                else:
                    self._load_progress_manager = _LoadProgressManager(
                        self._device_id, group=injected_group
                    )
                self._tp_group = self._load_progress_manager.group
            else:
                self._load_progress_manager = _LoadProgressManager(self._device_id)

            chunk_layout = []
            source_offset = 0
            for tensor_index, size in enumerate(self._tensor_sizes):
                for chunk_offset in range(0, size, _COPY_CHUNK_BYTES):
                    chunk_layout.append(
                        (
                            tensor_index,
                            source_offset + chunk_offset,
                            chunk_offset,
                            min(size - chunk_offset, _COPY_CHUNK_BYTES),
                        )
                    )
                source_offset += size
            chunk_layout = np.asarray(chunk_layout, dtype=np.int64).reshape(
                -1, _CHUNK_LAYOUT_FIELDS
            )
            layerwise = bool(config.get("allgather_layerwise", False))
            default_capacity_mb = 256 if layerwise else 2048
            self._pool = _FusedBufferPool(
                device_id=self._device_id,
                world_size=self._load_storage_world_size,
                shard_size=self._shard_size,
                chunk_layout=chunk_layout,
                requested_window_blocks=int(
                    config.get("allgather_window_blocks_per_rank", 4)
                ),
                load_slot_count=int(config.get("allgather_load_slots", 2)),
                dump_slot_count=int(config.get("allgather_dump_slots", 2)),
                capacity_mb=int(
                    config.get(
                        "allgather_fused_buffer_capacity_mb", default_capacity_mb
                    )
                ),
            )
            inner_config["tensor_size_list"] = [self._shard_size]
            if not self._scatter_only:
                inner_config["share_buffer_enable"] = False
                inner_config["local_rank_size"] = 1
            logger.info(
                "UcmAllGatherStore initialized: "
                f"rank={self._rank}, world_size={self._world_size}, "
                f"shard_size={self._shard_size}, layerwise={layerwise}, "
                f"mode={'scatter-only' if self._scatter_only else ('allgather' if self._collective_enabled else 'local-coalesced')}, "
                f"shared_cache={bool(inner_config.get('share_buffer_enable', False))}, "
                f"hccl_buffer_mb={self._hccl_buffer_mb}, "
                f"collective_mode={self._collective_mode_name}, "
                f"skip_load_collective={self._skip_load_collective}, "
                f"skip_load_scatter={self._skip_load_scatter}"
            )

        self._inner = UcmPipelineStore(inner_config)
        if self._worker:
            self._load_progress_manager.register(self)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        error = None
        if self._worker:
            with self._load_condition:
                while self._load_tasks:
                    self._load_condition.wait()
            for window in list(self._active_dump_windows):
                try:
                    self._finish_dump_window(window)
                except BaseException as current_error:
                    if error is None:
                        error = current_error
            try:
                torch.npu.synchronize(self._device_id)
            except BaseException as current_error:
                if error is None:
                    error = current_error
            try:
                self._load_progress_manager.release(self)
            except BaseException as current_error:
                if error is None:
                    error = current_error
        try:
            self._inner.close()
        except BaseException as current_error:
            if error is None:
                error = current_error
        self._active_dump_windows.clear()
        self._load_tasks.clear()
        self._unprimed_load_tasks.clear()
        self._load_metadata_arenas.clear()
        self._pool = None
        self._copy_op = None
        self._load_progress_manager = None
        self._tp_group = None
        self._device = None
        if error is not None:
            raise error

    def cc_store(self) -> int:
        return self._inner.cc_store()

    @staticmethod
    def _owner(block_id: bytes, world_size: int) -> int:
        if not block_id:
            raise ValueError("block id must not be empty")
        value = int.from_bytes(block_id[:8], byteorder="little", signed=False)
        return value % world_size

    @staticmethod
    def _normalize_addrs(addrs: List[List[int]] | np.ndarray) -> np.ndarray:
        result = np.asarray(addrs, dtype=np.uint64)
        if result.ndim != 2:
            raise ValueError(f"addresses must be two-dimensional, got {result.shape}")
        return np.ascontiguousarray(result)

    def _tensor_addrs(self, tensors: List[List[torch.Tensor]]) -> np.ndarray:
        return np.asarray(
            [[tensor.data_ptr() for tensor in row] for row in tensors],
            dtype=np.uint64,
        )

    def _partition(
        self, block_ids: List[bytes], storage_world_size: int
    ) -> tuple[List[int], List[int]]:
        owners = [self._owner(block_id, storage_world_size) for block_id in block_ids]
        counts = [0] * storage_world_size
        owner_slots = []
        for owner in owners:
            owner_slots.append(counts[owner])
            counts[owner] += 1
        return owners, owner_slots

    def owns(self, block_id: bytes) -> bool:
        if not self._worker:
            return True
        return (
            self._owner(block_id, self._dump_storage_world_size)
            == self._dump_storage_rank
        )

    @staticmethod
    def _balanced_descriptors(
        descriptors: List[tuple[int, int, int]],
    ) -> tuple[List[tuple[int, int, int]], List[int]]:
        chunks: List[tuple[int, int, int]] = []
        for src, dst, size in descriptors:
            for offset in range(0, size, _COPY_CHUNK_BYTES):
                current = min(size - offset, _COPY_CHUNK_BYTES)
                chunks.append((src + offset, dst + offset, current))
        if not chunks:
            return [], []

        used_cores = min(_MAX_AIV_CORES, len(chunks))
        bins: List[List[tuple[int, int, int]]] = [[] for _ in range(used_cores)]
        loads = [0] * used_cores
        for item in sorted(chunks, key=lambda row: row[2], reverse=True):
            target = min(range(used_cores), key=loads.__getitem__)
            bins[target].append(item)
            loads[target] += item[2]

        ordered: List[tuple[int, int, int]] = []
        offsets = [0]
        for items in bins:
            ordered.extend(items)
            offsets.append(len(ordered))
        return ordered, offsets

    def _segmented_copy(
        self, descriptors: List[tuple[int, int, int]], slot: _CopySlot
    ) -> None:
        ordered, offsets = self._balanced_descriptors(descriptors)
        if not ordered:
            return
        if len(ordered) > slot.descriptor_buffer.shape[0]:
            raise RuntimeError(
                "segmented-copy descriptor capacity exceeded: "
                f"required={len(ordered)}, capacity={slot.descriptor_buffer.shape[0]}"
            )
        descriptor_count = len(ordered)
        offset_count = len(offsets)
        slot.host_descriptors[:descriptor_count] = ordered
        slot.host_offsets[:offset_count] = offsets
        slot.descriptor_buffer[:descriptor_count].copy_(
            slot.host_descriptor_tensor[:descriptor_count]
        )
        slot.offset_buffer[:offset_count].copy_(slot.host_offset_tensor[:offset_count])
        self._copy_op.copy(
            slot.descriptor_buffer[:descriptor_count],
            slot.offset_buffer[:offset_count],
            offset_count - 1,
        )

    def _pack_descriptors(
        self,
        addrs: np.ndarray,
        owners: List[int],
        owner_slots: List[int],
        base: int,
    ) -> List[tuple[int, int, int]]:
        descriptors = []
        for row, (owner, slot) in enumerate(zip(owners, owner_slots)):
            if owner != self._dump_storage_rank:
                continue
            offset = 0
            for address, size in zip(addrs[row], self._tensor_sizes):
                if int(address) != 0 and size != 0:
                    destination = base + slot * self._shard_size + offset
                    descriptors.append((int(address), destination, size))
                offset += size
        return descriptors

    def _prepare_load_metadata(
        self, task: AllGatherTask, plan: _LoadPlan, addrs: np.ndarray
    ) -> None:
        host_destinations = np.ascontiguousarray(addrs[plan.row_order], dtype=np.int64)
        host_routes = plan.route_rows
        task.host_destination_tensor = torch.from_numpy(host_destinations)
        task.host_route_tensor = torch.from_numpy(host_routes)
        task.destination_buffer = torch.empty(
            host_destinations.shape, dtype=torch.int64, device=f"npu:{self._device_id}"
        )
        task.route_buffer = torch.empty(
            host_routes.shape, dtype=torch.int32, device=f"npu:{self._device_id}"
        )
        task.destination_buffer.copy_(task.host_destination_tensor)
        task.route_buffer.copy_(task.host_route_tensor)
        task.metadata_event = torch.npu.Event()
        task.metadata_event.record(torch.npu.current_stream())

    def _compact_scatter(
        self, task: AllGatherTask, window: _LoadWindow, slot: _LoadSlot
    ) -> None:
        plan = window.plan
        self._copy_op.scatter(
            slot.receive_buffer,
            task.destination_buffer.narrow(
                0, task.destination_offset + plan.metadata_offset, plan.row_count
            ),
            task.route_buffer.narrow(
                0, task.route_offset + plan.metadata_offset, plan.row_count
            ),
            self._pool.chunk_layout,
            plan.row_count,
            plan.collective_blocks * self._shard_size,
            self._shard_size,
        )

    def _window_rows(self, owner_slots: List[int]) -> List[tuple[int, List[int]]]:
        if not owner_slots:
            return []
        window_blocks = self._pool.window_blocks
        window_count = max(owner_slots) // window_blocks + 1
        return [
            (
                window_index * window_blocks,
                [
                    row
                    for row, owner_slot in enumerate(owner_slots)
                    if owner_slot // window_blocks == window_index
                ],
            )
            for window_index in range(window_count)
        ]

    def lookup(self, block_ids: List[bytes]) -> List[bool]:
        return self._inner.lookup(block_ids)

    def lookup_on_prefix(self, block_ids: List[bytes]) -> int:
        return self._inner.lookup_on_prefix(block_ids)

    def prefetch(self, block_ids: List[bytes]) -> None:
        if not self._worker:
            self._inner.prefetch(block_ids)
            return
        if not self._collective_enabled:
            self._inner.prefetch(block_ids)
            return
        owned = [
            block_id
            for block_id in block_ids
            if self._owner(block_id, self._load_storage_world_size)
            == self._load_storage_rank
        ]
        if owned:
            self._inner.prefetch(owned)

    def load(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_tensor: List[List[torch.Tensor]],
    ) -> Task:
        return self.load_data(block_ids, shard_index, self._tensor_addrs(dst_tensor))

    def dump(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_tensor: List[List[torch.Tensor]],
    ) -> Task:
        return self.dump_data(block_ids, shard_index, self._tensor_addrs(src_tensor))

    def _build_load_plan(self, block_ids: List[bytes]) -> _LoadPlan:
        owners, owner_slots = self._partition(block_ids, self._load_storage_world_size)
        windows = []
        row_order = []
        route_rows = []
        metadata_offset = 0
        window_rows = list(self._window_rows(owner_slots))
        allow_crop = self._collective_count_crop and len(window_rows) == 1
        for start_slot, rows in window_rows:
            window_owners = [owners[row] for row in rows]
            window_slots = [owner_slots[row] - start_slot for row in rows]
            collective_blocks = max(window_slots) + 1
            # Dense collectives retain higher link bandwidth than a slightly shorter tail.
            if not allow_crop or collective_blocks * 2 > self._pool.window_blocks:
                collective_blocks = self._pool.window_blocks
            owned_positions = [
                index
                for index, owner in enumerate(window_owners)
                if owner == self._load_storage_rank
            ]
            owned_source_rows = [rows[index] for index in owned_positions]
            owned_addrs_by_slot = []
            for slot in self._pool.load_slots:
                base = slot.send_buffer.data_ptr()
                owned_addrs_by_slot.append(
                    np.asarray(
                        [
                            [base + window_slots[index] * self._shard_size]
                            for index in owned_positions
                        ],
                        dtype=np.uint64,
                    ).reshape(-1, 1)
                )
            windows.append(
                _LoadWindowPlan(
                    row_count=len(rows),
                    metadata_offset=metadata_offset,
                    collective_blocks=collective_blocks,
                    owned_block_ids=[block_ids[row] for row in owned_source_rows],
                    owned_source_rows=owned_source_rows,
                    owned_addrs_by_slot=owned_addrs_by_slot,
                )
            )
            row_order.extend(rows)
            route_rows.extend(zip(window_owners, window_slots))
            metadata_offset += len(rows)
        return _LoadPlan(
            source_row_count=len(block_ids),
            row_order=np.asarray(row_order, dtype=np.intp),
            route_rows=np.asarray(route_rows, dtype=np.int32).reshape(-1, 2),
            windows=windows,
        )

    @staticmethod
    def _metadata_capacity(required: int) -> int:
        return 1 if required <= 1 else 1 << (required - 1).bit_length()

    def _acquire_load_metadata_arena(
        self, destination_rows: int, route_rows: int
    ) -> _LoadMetadataArena:
        with self._load_condition:
            candidates = [
                arena
                for arena in self._load_metadata_arenas
                if not arena.in_use
                and arena.destination_capacity >= destination_rows
                and arena.route_capacity >= route_rows
            ]
            if candidates:
                arena = min(
                    candidates,
                    key=lambda item: item.destination_capacity + item.route_capacity,
                )
                arena.in_use = True
                return arena

        destination_capacity = self._metadata_capacity(destination_rows)
        route_capacity = self._metadata_capacity(route_rows)
        arena = _LoadMetadataArena(
            destination_buffer=torch.empty(
                (destination_capacity, len(self._tensor_sizes)),
                dtype=torch.int64,
                device=f"npu:{self._device_id}",
            ),
            route_buffer=torch.empty(
                (route_capacity, 2),
                dtype=torch.int32,
                device=f"npu:{self._device_id}",
            ),
            destination_capacity=destination_capacity,
            route_capacity=route_capacity,
            in_use=True,
        )
        with self._load_condition:
            self._load_metadata_arenas.append(arena)
        return arena

    def prepare_load_batch(
        self, requests: List[tuple[List[bytes], np.ndarray]]
    ) -> tuple[_PreparedLoadBatch, List[PreparedLoadRequest]]:
        if not self._worker:
            raise RuntimeError("prepared all-gather loads require a worker store")
        prepared_inputs = []
        destination_rows = 0
        route_rows = 0
        for block_ids, all_layer_addrs in requests:
            addrs = np.asarray(all_layer_addrs, dtype=np.uint64)
            expected_tail = (len(block_ids), len(self._tensor_sizes))
            if addrs.ndim != 3 or addrs.shape[1:] != expected_tail:
                raise ValueError(
                    f"prepared load address shape {addrs.shape} does not match "
                    f"(layers, {expected_tail[0]}, {expected_tail[1]})"
                )
            plan = self._build_load_plan(block_ids)
            prepared_inputs.append((plan, addrs))
            destination_rows += addrs.shape[0] * plan.source_row_count
            route_rows += plan.source_row_count
        if not prepared_inputs:
            raise ValueError("prepared load batch must not be empty")

        host_destinations = np.empty(
            (destination_rows, len(self._tensor_sizes)), dtype=np.int64
        )
        host_routes = np.empty((route_rows, 2), dtype=np.int32)
        request_specs = []
        destination_offset = 0
        route_offset = 0
        for plan, addrs in prepared_inputs:
            layer_count = addrs.shape[0]
            row_count = plan.source_row_count
            host_routes[route_offset : route_offset + row_count] = plan.route_rows
            request_destination_rows = layer_count * row_count
            host_destinations[
                destination_offset : destination_offset + request_destination_rows
            ] = addrs[:, plan.row_order, :].reshape(
                request_destination_rows, len(self._tensor_sizes)
            )
            request_specs.append((plan, destination_offset, route_offset, layer_count))
            destination_offset += request_destination_rows
            route_offset += row_count

        arena = self._acquire_load_metadata_arena(destination_rows, route_rows)
        host_destination_tensor = torch.from_numpy(host_destinations)
        host_route_tensor = torch.from_numpy(host_routes)
        try:
            arena.destination_buffer[:destination_rows].copy_(host_destination_tensor)
            arena.route_buffer[:route_rows].copy_(host_route_tensor)
        except BaseException:
            arena.in_use = False
            raise
        batch = _PreparedLoadBatch(
            arena=arena,
            host_destination_tensor=host_destination_tensor,
            host_route_tensor=host_route_tensor,
            metadata_event=torch.npu.Event(),
        )
        batch.metadata_event.record(torch.npu.current_stream())
        prepared_requests = [
            PreparedLoadRequest(
                batch=batch,
                plan=plan,
                destination_offset=request_destination_offset,
                route_offset=request_route_offset,
                layer_count=layer_count,
                store_id=id(self),
            )
            for (
                plan,
                request_destination_offset,
                request_route_offset,
                layer_count,
            ) in request_specs
        ]
        return batch, prepared_requests

    def release_prepared_load_batch(self, batch: _PreparedLoadBatch) -> None:
        with self._load_condition:
            if batch.released:
                return
            batch.released = True
            if batch.active_tasks == 0:
                batch.arena.in_use = False

    @staticmethod
    def _build_load_windows(
        plan: _LoadPlan, shard_index: List[int]
    ) -> List[_LoadWindow]:
        if len(shard_index) != plan.source_row_count:
            raise ValueError(
                f"shard index count {len(shard_index)} does not match "
                f"block count {plan.source_row_count}"
            )
        return [
            _LoadWindow(
                plan=window,
                owned_shard_indices=[
                    shard_index[row] for row in window.owned_source_rows
                ],
            )
            for window in plan.windows
        ]

    def _submit_load_window(self, window: _LoadWindow, slot: _LoadSlot) -> None:
        window.slot = slot
        plan = window.plan
        if not plan.owned_block_ids:
            return
        try:
            window.inner_task = self._inner.load_data(
                plan.owned_block_ids,
                window.owned_shard_indices,
                plan.owned_addrs_by_slot[slot.index],
            )
        except BaseException as error:
            window.error = error

    def _prime_load_task(self, task: AllGatherTask) -> None:
        while (
            task.next_load_window < len(task.load_windows)
            and len(task.active_load_windows) < self._pool.load_slot_count
        ):
            slot = self._pool.try_acquire_load()
            if slot is None:
                break
            window = task.load_windows[task.next_load_window]
            self._submit_load_window(window, slot)
            task.active_load_windows.append(window)
            task.next_load_window += 1

    def _prime_queued_loads(self) -> None:
        while self._unprimed_load_tasks:
            task = self._unprimed_load_tasks[0]
            self._prime_load_task(task)
            if task.next_load_window < len(task.load_windows):
                break
            self._unprimed_load_tasks.popleft()

    def _queue_load_task(self, task: AllGatherTask) -> AllGatherTask:
        manager_error = self._load_progress_manager.error
        with self._load_condition:
            progress_error = self._load_progress_error or manager_error
            if progress_error is not None:
                task.terminal_error = progress_error
                task.completed = True
                self._release_prepared_task(task)
                return task
            self._load_tasks.append(task)
            self._unprimed_load_tasks.append(task)
            queue_depth = len(self._load_tasks)
        try:
            self._load_progress_manager.submit(self, task)
        except BaseException as error:
            self._abort_load_tasks(error)
        ucmmetrics.update_stats({"allgather_load_task_queue_depth": float(queue_depth)})
        return task

    def _remove_unprimed_load_task(self, task: AllGatherTask) -> None:
        for index, pending in enumerate(self._unprimed_load_tasks):
            if pending is task:
                del self._unprimed_load_tasks[index]
                break

    @staticmethod
    def _release_prepared_task(task: AllGatherTask) -> None:
        if task.prepared_batch is None:
            return
        task.prepared_batch.active_tasks -= 1
        if task.prepared_batch.active_tasks == 0 and task.prepared_batch.released:
            task.prepared_batch.arena.in_use = False

    def load_data_prepared(
        self,
        prepared: PreparedLoadRequest,
        shard_index: List[int],
        layer_index: int,
    ) -> Task:
        if prepared.store_id != id(self):
            raise ValueError("prepared load belongs to a different store")
        if layer_index < 0 or layer_index >= prepared.layer_count:
            raise IndexError(
                f"prepared load layer {layer_index} is outside "
                f"[0, {prepared.layer_count})"
            )
        task = AllGatherTask(
            operation="load",
            load_windows=self._build_load_windows(prepared.plan, shard_index),
            destination_offset=(
                prepared.destination_offset
                + layer_index * prepared.plan.source_row_count
            ),
            route_offset=prepared.route_offset,
            destination_buffer=prepared.batch.arena.destination_buffer,
            route_buffer=prepared.batch.arena.route_buffer,
            prepared_batch=prepared.batch,
            metadata_event=prepared.batch.metadata_event,
        )
        if not task.load_windows:
            with self._load_condition:
                if prepared.batch.released:
                    raise RuntimeError("prepared load batch has already been released")
                task.completed = True
            return task
        with self._load_condition:
            if prepared.batch.released:
                raise RuntimeError("prepared load batch has already been released")
            prepared.batch.active_tasks += 1
        return self._queue_load_task(task)

    def load_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_addr: List[List[int]] | np.ndarray,
    ) -> Task:
        if not self._worker:
            return AllGatherTask(
                operation="load",
                inner_task=self._inner.load_data(block_ids, shard_index, dst_addr),
                passthrough=True,
            )
        addrs = self._normalize_addrs(dst_addr)
        if addrs.shape != (len(block_ids), len(self._tensor_sizes)):
            raise ValueError(
                f"load address shape {addrs.shape} does not match "
                f"({len(block_ids)}, {len(self._tensor_sizes)})"
            )
        plan = self._build_load_plan(block_ids)
        task = AllGatherTask(
            operation="load", load_windows=self._build_load_windows(plan, shard_index)
        )
        if not task.load_windows:
            task.completed = True
            return task
        try:
            self._prepare_load_metadata(task, plan, addrs)
        except BaseException as error:
            task.load_error = error
        return self._queue_load_task(task)

    def _acquire_dump_slot(self) -> _DumpSlot:
        slot = self._pool.try_acquire_dump()
        if slot is not None:
            return slot
        self._finish_dump_window(self._active_dump_windows[0])
        slot = self._pool.try_acquire_dump()
        if slot is None:
            raise RuntimeError("failed to reclaim an allgather dump slot")
        return slot

    def _finish_dump_window(self, window: _DumpWindow) -> None:
        if window.completed:
            return
        try:
            self._inner.wait(window.inner_task)
        finally:
            if window.event_handle:
                self._device.destroy_event_handle(window.event_handle)
                window.event_handle = 0
            self._pool.release_dump(window.slot)
            window.completed = True
            try:
                self._active_dump_windows.remove(window)
            except ValueError:
                pass

    def dump_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_addr: List[List[int]] | np.ndarray,
        prerequisite_handle: int = 0,
    ) -> Task:
        if not self._worker:
            return AllGatherTask(
                operation="dump",
                inner_task=self._inner.dump_data(
                    block_ids, shard_index, src_addr, prerequisite_handle
                ),
                passthrough=True,
            )
        addrs = self._normalize_addrs(src_addr)
        if addrs.shape != (len(block_ids), len(self._tensor_sizes)):
            raise ValueError(
                f"dump address shape {addrs.shape} does not match "
                f"({len(block_ids)}, {len(self._tensor_sizes)})"
            )
        owners, owner_slots = self._partition(block_ids, self._dump_storage_world_size)
        task = AllGatherTask(operation="dump")
        waited_prerequisite = False
        try:
            for start_slot, rows in self._window_rows(owner_slots):
                window_owners = [owners[row] for row in rows]
                owned_rows = [
                    index
                    for index, owner in enumerate(window_owners)
                    if owner == self._dump_storage_rank
                ]
                if not owned_rows:
                    continue
                window_slots = [owner_slots[row] - start_slot for row in rows]
                slot = self._acquire_dump_slot()
                event_handle = 0
                try:
                    if prerequisite_handle and not waited_prerequisite:
                        self._copy_op.wait_event(prerequisite_handle)
                        waited_prerequisite = True
                    window_addrs = np.ascontiguousarray(addrs[rows])
                    self._segmented_copy(
                        self._pack_descriptors(
                            window_addrs,
                            window_owners,
                            window_slots,
                            slot.send_buffer.data_ptr(),
                        ),
                        slot,
                    )
                    event_handle = self._device.get_event_handle()
                    if event_handle == 0:
                        raise RuntimeError(
                            "failed to record allgather dump completion event"
                        )
                    inner_task = self._inner.dump_data(
                        [block_ids[rows[row]] for row in owned_rows],
                        [shard_index[rows[row]] for row in owned_rows],
                        np.asarray(
                            [
                                [
                                    slot.send_buffer.data_ptr()
                                    + window_slots[row] * self._shard_size
                                ]
                                for row in owned_rows
                            ],
                            dtype=np.uint64,
                        ),
                        event_handle,
                    )
                except BaseException:
                    if event_handle:
                        self._device.destroy_event_handle(event_handle)
                    self._pool.release_dump(slot)
                    raise
                window = _DumpWindow(slot, inner_task, event_handle)
                task.dump_windows.append(window)
                self._active_dump_windows.append(window)
        except BaseException:
            for window in task.dump_windows:
                self._finish_dump_window(window)
            raise
        task.completed = not task.dump_windows
        return task

    def _drain_load_windows(self, windows: deque) -> None:
        while windows:
            window = windows.popleft()
            try:
                if window.inner_task is not None:
                    self._inner.wait(window.inner_task)
            except BaseException:
                pass
            if window.slot is not None:
                self._pool.release_load(window.slot)
            window.completed = True

    def _wait_load(self, task: AllGatherTask) -> None:
        inner_wait_ms = 0.0
        collective_submit_ms = 0.0
        scatter_submit_ms = 0.0
        collective_device_ms = 0.0
        scatter_device_ms = 0.0
        device_timings = []
        window_count = 0
        if task.metadata_event is not None:
            torch.npu.current_stream().wait_event(task.metadata_event)
        while task.active_load_windows:
            window = task.active_load_windows.popleft()
            window_count += 1
            local_error = window.error
            if local_error is None and window.inner_task is not None:
                try:
                    wait_start = time.perf_counter()
                    self._inner.wait(window.inner_task)
                    inner_wait_ms += (time.perf_counter() - wait_start) * 1000
                except BaseException as error:
                    inner_wait_ms += (time.perf_counter() - wait_start) * 1000
                    local_error = error
            if task.load_error is None and local_error is not None:
                task.load_error = local_error

            slot = window.slot
            try:
                collective_start_event = None
                collective_done_event = None
                scatter_done_event = None
                stream = torch.npu.current_stream()
                if self._collective_enabled and not self._skip_load_collective:
                    collective_start_event = torch.npu.Event(enable_timing=True)
                    collective_done_event = torch.npu.Event(enable_timing=True)
                    collective_start_event.record(stream)
                    collective_bytes = window.plan.collective_blocks * self._shard_size
                    collective_start = time.perf_counter()
                    dist.all_gather_into_tensor(
                        slot.receive_buffer.narrow(
                            0, 0, collective_bytes * self._world_size
                        ),
                        slot.send_buffer.narrow(0, 0, collective_bytes),
                        group=self._tp_group,
                    )
                    collective_submit_ms += (
                        time.perf_counter() - collective_start
                    ) * 1000
                    collective_done_event.record(stream)
                if task.load_error is None and not self._skip_load_scatter:
                    scatter_done_event = torch.npu.Event(enable_timing=True)
                    if collective_done_event is None:
                        collective_done_event = torch.npu.Event(enable_timing=True)
                        collective_done_event.record(stream)
                    scatter_start = time.perf_counter()
                    self._compact_scatter(task, window, slot)
                    scatter_submit_ms += (time.perf_counter() - scatter_start) * 1000
                    scatter_done_event.record(stream)
                device_timings.append(
                    (
                        collective_start_event,
                        collective_done_event,
                        scatter_done_event,
                    )
                )
            except BaseException:
                self._pool.release_load(slot)
                self._drain_load_windows(task.active_load_windows)
                raise
            self._pool.complete_load(slot)
            window.completed = True
            with self._load_condition:
                self._prime_queued_loads()

        sync_start = time.perf_counter()
        torch.npu.current_stream().synchronize()
        sync_ms = (time.perf_counter() - sync_start) * 1000
        try:
            for collective_start, collective_done, scatter_done in device_timings:
                if collective_start is not None:
                    collective_device_ms += collective_start.elapsed_time(
                        collective_done
                    )
                if scatter_done is not None:
                    scatter_device_ms += collective_done.elapsed_time(scatter_done)
        except BaseException as error:
            logger.warning("Failed to collect AllGather device timings: %s", error)
            collective_device_ms = 0.0
            scatter_device_ms = 0.0
        ucmmetrics.update_stats(
            {
                "allgather_load_inner_wait_ms": inner_wait_ms,
                "allgather_load_collective_submit_ms": collective_submit_ms,
                "allgather_load_scatter_submit_ms": scatter_submit_ms,
                "allgather_load_collective_device_ms": collective_device_ms,
                "allgather_load_scatter_device_ms": scatter_device_ms,
                "allgather_load_sync_ms": sync_ms,
                "allgather_load_windows": float(window_count),
            }
        )
        if task.load_error is not None:
            raise task.load_error

    def _finish_load_task(self, task: AllGatherTask) -> None:
        with self._load_condition:
            if not self._load_tasks or self._load_tasks[0] is not task:
                raise RuntimeError("allgather load progress order mismatch")
            self._prime_queued_loads()
        try:
            self._wait_load(task)
        except BaseException as error:
            self._drain_load_windows(task.active_load_windows)
            with self._load_condition:
                self._remove_unprimed_load_task(task)
            task.terminal_error = error
        with self._load_condition:
            task.completed = True
            self._load_tasks.popleft()
            self._release_prepared_task(task)
            self._prime_queued_loads()
            self._load_condition.notify_all()

    def _abort_load_tasks(self, error: BaseException) -> None:
        with self._load_condition:
            self._load_progress_error = error
            tasks = list(self._load_tasks)
            self._load_tasks.clear()
            self._unprimed_load_tasks.clear()
        for task in tasks:
            self._drain_load_windows(task.active_load_windows)
            task.terminal_error = error
            task.completed = True
            self._release_prepared_task(task)
        with self._load_condition:
            self._load_condition.notify_all()

    def _wait_queued_load(self, task: AllGatherTask) -> None:
        with self._load_condition:
            while not task.completed:
                if not any(pending is task for pending in self._load_tasks):
                    raise RuntimeError("allgather load task is not queued")
                self._load_condition.wait()
        if task.terminal_error is not None:
            raise task.terminal_error

    def wait(self, task: Task) -> None:
        if not isinstance(task, AllGatherTask):
            raise TypeError(f"unexpected task type: {type(task).__name__}")
        if task.completed:
            if task.terminal_error is not None:
                raise task.terminal_error
            return
        if task.passthrough:
            self._inner.wait(task.inner_task)
        elif task.operation == "load":
            self._wait_queued_load(task)
        else:
            for window in task.dump_windows:
                self._finish_dump_window(window)
        if task.operation != "load" or task.passthrough:
            task.completed = True

    def check(self, task: Task) -> bool:
        if not isinstance(task, AllGatherTask):
            raise TypeError(f"unexpected task type: {type(task).__name__}")
        if task.completed:
            return True
        if task.passthrough:
            return self._inner.check(task.inner_task)
        if task.operation == "load":
            return task.completed
        for window in task.dump_windows:
            if window.completed:
                continue
            if not self._inner.check(window.inner_task):
                return False
            self._finish_dump_window(window)
        task.completed = True
        return True
