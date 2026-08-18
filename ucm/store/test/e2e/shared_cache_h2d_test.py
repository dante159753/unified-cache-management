import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _allgather_cache_store_benchmark import run  # noqa: E402

if __name__ == "__main__":
    run("h2d", "shared-cache")
