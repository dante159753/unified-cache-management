from types import SimpleNamespace

import pytest

from ucm.integration.vllm.patch import gpu_worker_memory_reservation_patch


@pytest.mark.parametrize("worker_name", ["GPUWorker", "Worker"])
def test_patch_supports_gpu_worker_class_names(monkeypatch, worker_name):
    class Worker:
        def determine_available_memory(self):
            return 1024

    module = SimpleNamespace(**{worker_name: Worker})
    monkeypatch.setattr(
        gpu_worker_memory_reservation_patch,
        "calculate_vllm_reservation",
        lambda _: 256,
    )

    gpu_worker_memory_reservation_patch.patch_gpu_worker_memory_reservation(module)
    gpu_worker_memory_reservation_patch.patch_gpu_worker_memory_reservation(module)

    worker = Worker()
    assert worker.determine_available_memory() == 768
    assert worker.available_kv_cache_memory_bytes == 768


def test_patch_rejects_unknown_gpu_worker_class():
    with pytest.raises(AttributeError, match="neither GPUWorker nor Worker"):
        gpu_worker_memory_reservation_patch.patch_gpu_worker_memory_reservation(
            SimpleNamespace()
        )
