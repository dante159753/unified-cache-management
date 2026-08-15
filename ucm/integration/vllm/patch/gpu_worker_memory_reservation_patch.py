from __future__ import annotations

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger
from ucm.store.allgather.reservation import calculate_vllm_reservation

logger = init_logger(__name__)


@when_imported("vllm.v1.worker.gpu_worker")
def patch_gpu_worker_memory_reservation(mod):
    worker = mod.GPUWorker
    if getattr(worker, "_ucm_allgather_reservation_patched", False):
        return
    original = worker.determine_available_memory

    def determine_available_memory(self) -> int:
        raw_available = int(original(self))
        reservation = int(calculate_vllm_reservation(self))
        if reservation == 0:
            return raw_available
        available = raw_available - reservation
        if available <= 0:
            raise RuntimeError(
                "UCM AllGather reservation leaves no memory for KV cache: "
                f"raw_available_bytes={raw_available}, "
                f"reservation_bytes={reservation}"
            )
        self.available_kv_cache_memory_bytes = available
        logger.info(
            "Reserve UCM AllGather CUDA memory before KV cache allocation: "
            f"raw_available_bytes={raw_available}, "
            f"reservation_bytes={reservation}, "
            f"available_kv_cache_bytes={available}"
        )
        return available

    worker.determine_available_memory = determine_available_memory
    worker._ucm_allgather_reservation_patched = True
