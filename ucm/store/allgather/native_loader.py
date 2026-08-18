from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType

from ucm.store.native_loader import _preload_library

_LIBRARY_DIR = Path(__file__).resolve().parent
_KERNEL_LIBRARIES = (
    _LIBRARY_DIR / "libucm_segmented_copy_kernels.so",
    _LIBRARY_DIR / "libucm_compact_scatter_kernels.so",
)


def preload_allgather_kernels(*, required: bool = False) -> None:
    require_all = (
        required
        or find_spec("torch_npu") is not None
        or any(path.is_file() for path in _KERNEL_LIBRARIES)
    )
    for path in _KERNEL_LIBRARIES:
        _preload_library(path, required=require_all)


def _load_module(name: str, *, kernels_required: bool) -> ModuleType:
    preload_allgather_kernels(required=kernels_required)
    try:
        return import_module(f"ucm.store.allgather.{name}")
    except ImportError as error:
        if any(path.name in str(error) for path in _KERNEL_LIBRARIES):
            expected = ", ".join(str(path) for path in _KERNEL_LIBRARIES)
            raise ImportError(
                f"AllGather native kernels are not installed; expected {expected}"
            ) from error
        raise


def load_allgather_runtime() -> ModuleType:
    return _load_module("ucm_allgather_runtime", kernels_required=False)


def load_segmented_copy() -> ModuleType:
    return _load_module("ucm_segmented_copy", kernels_required=True)
