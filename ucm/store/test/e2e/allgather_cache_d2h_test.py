import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _allgather_cache_store_benchmark import run  # noqa: E402

if __name__ == "__main__":
    run("d2h")
