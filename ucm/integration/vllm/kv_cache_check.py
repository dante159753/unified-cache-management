from __future__ import annotations

import hashlib
from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from ucm.logger import init_logger

if TYPE_CHECKING:
    from ucm.store.ucmstore_v1 import UcmKVStoreBaseV1

logger = init_logger(__name__)


@dataclass(frozen=True)
class KVCacheLoadCheck:
    block_id: bytes
    shard_index: int
    store_name: str
    ptrs: tuple[int, ...]
    tensor_sizes: tuple[int, ...]
    expected_md5: str


class KVCacheCheck:
    """Keep worker-local MD5 records of the HBM bytes submitted for dumping."""

    def __init__(
        self,
        kv_caches: dict[
            str, torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]
        ],
    ) -> None:
        self._dump_md5: dict[tuple[UcmKVStoreBaseV1, bytes, int], str] = {}
        self._buffers: dict[int, torch.Tensor] = {}
        tensors = list(kv_caches.values())
        while tensors:
            tensor = tensors.pop()
            if isinstance(tensor, (tuple, list)):
                tensors.extend(tensor)
                continue
            storage = tensor.untyped_storage()
            base_ptr = storage.data_ptr()
            if base_ptr in self._buffers:
                continue
            # Transfer pointers address raw storage, not logical tensor elements.
            self._buffers[base_ptr] = torch.empty(
                0, dtype=torch.uint8, device=tensor.device
            ).set_(storage, 0, (storage.nbytes(),), (1,))
        self._base_ptrs = sorted(self._buffers)
        logger.warning(
            "KV cache MD5 checking is enabled; HBM-to-CPU copies synchronize "
            "inference and dump records remain in this worker's memory."
        )

    def _md5(self, ptrs: tuple[int, ...], tensor_sizes: tuple[int, ...]) -> str:
        digest = hashlib.md5()
        for ptr, size in zip(ptrs, tensor_sizes, strict=True):
            # Layerwise layouts can contain null slots for absent tensor components.
            if ptr == 0 or size == 0:
                continue
            index = bisect_right(self._base_ptrs, ptr) - 1
            if index < 0:
                raise ValueError(f"KV checksum address {ptr} is not registered")
            base_ptr = self._base_ptrs[index]
            buffer = self._buffers[base_ptr]
            offset = ptr - base_ptr
            if size < 0 or offset + size > buffer.numel():
                raise ValueError(
                    f"KV checksum range exceeds registered storage: ptr={ptr}, "
                    f"size={size}, storage_bytes={buffer.numel()}"
                )
            digest.update(buffer[offset : offset + size].cpu().numpy().tobytes())
        return digest.hexdigest()

    def record_dump(
        self,
        store: UcmKVStoreBaseV1,
        block_ids: list[bytes],
        shard_indices: list[int],
        ptrs: np.ndarray,
    ) -> None:
        """Hash the source bytes before submitting a dump, replacing older records."""
        tensor_sizes = tuple(int(size) for size in store.config["tensor_size_list"])
        for block_id, shard_index, row in zip(
            block_ids, shard_indices, ptrs, strict=True
        ):
            addresses = tuple(int(ptr) for ptr in row)
            self._dump_md5[store, block_id, shard_index] = self._md5(
                addresses, tensor_sizes
            )

    def prepare_load(
        self,
        store: UcmKVStoreBaseV1,
        block_ids: list[bytes],
        shard_indices: list[int],
        ptrs: np.ndarray,
    ) -> list[KVCacheLoadCheck]:
        """Snapshot known dump digests and destination addresses without reading HBM."""
        checks = []
        tensor_sizes = tuple(int(size) for size in store.config["tensor_size_list"])
        store_name = str(store.config.get("unique_id", type(store).__name__))
        for block_id, shard_index, row in zip(
            block_ids, shard_indices, ptrs, strict=True
        ):
            expected_md5 = self._dump_md5.get((store, block_id, shard_index))
            if expected_md5 is not None:
                checks.append(
                    KVCacheLoadCheck(
                        block_id,
                        shard_index,
                        store_name,
                        tuple(int(ptr) for ptr in row),
                        tensor_sizes,
                        expected_md5,
                    )
                )
        return checks

    def verify_load(self, checks: list[KVCacheLoadCheck]) -> None:
        """Read completed load destinations and log content mismatches."""
        for check in checks:
            actual_md5 = self._md5(check.ptrs, check.tensor_sizes)
            if actual_md5 != check.expected_md5:
                logger.error(
                    "KV cache MD5 mismatch: ucm_block_id=%s shard_index=%d "
                    "store=%s expected_md5=%s actual_md5=%s",
                    check.block_id.hex(),
                    check.shard_index,
                    check.store_name,
                    check.expected_md5,
                    actual_md5,
                )
