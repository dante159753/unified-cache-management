import ctypes
import os
from importlib import import_module
from pathlib import Path
from types import ModuleType

_library_dir = Path(__file__).resolve().parent
_kernel_libraries = (
    _library_dir / "libucm_segmented_copy_kernels.so",
    _library_dir / "libucm_compact_scatter_kernels.so",
)
_preloaded_libraries: dict[Path, ctypes.CDLL] = {}


def _preload_kernels() -> None:
    if os.name != "posix":
        return
    for path in _kernel_libraries:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in _preloaded_libraries:
            continue
        _preloaded_libraries[resolved] = ctypes.CDLL(
            str(resolved),
            mode=getattr(os, "RTLD_NOW", 0) | getattr(os, "RTLD_GLOBAL", 0),
        )


def __getattr__(name: str) -> ModuleType:
    if name not in {"ucm_allgather_runtime", "ucm_segmented_copy"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _preload_kernels()
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
