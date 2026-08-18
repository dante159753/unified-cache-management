import ctypes
import os
import threading
from pathlib import Path

_preloaded_libraries: dict[Path, ctypes.CDLL] = {}
_preload_lock = threading.Lock()


def _preload_library(path: Path, *, required: bool = True) -> None:
    if os.name != "posix":
        return
    resolved = path.resolve()
    if not resolved.is_file():
        if required:
            raise FileNotFoundError(
                f"required UCM shared library is missing: {resolved}"
            )
        return
    with _preload_lock:
        if resolved in _preloaded_libraries:
            return
        try:
            library = ctypes.CDLL(
                str(resolved),
                mode=getattr(os, "RTLD_NOW", 0) | getattr(os, "RTLD_GLOBAL", 0),
            )
        except OSError as error:
            raise OSError(
                f"failed to preload UCM shared library {resolved}: {error}"
            ) from error
        _preloaded_libraries[resolved] = library
