import argparse
import ast
import glob
import gzip
import json
import os
import re
from pathlib import Path

match_pattern = re.compile(
    r"timestamp:\s*(?P<timestamp>\d+(?:\.\d+)?),.*?"
    r"input_length:\s*(?P<input_length>\d+),\s*"
    r"output_length:\s*(?P<output_length>\d+),\s*"
    r"ucm_block_ids:\s*(?P<ucm_block_ids>\[.*?\])"
)

DEFAULT_OUTPUT_LOG = Path("./output.jsonl")


def open_log_file(file_path):
    if str(file_path).endswith(".gz"):
        return gzip.open(file_path, "rt", encoding="utf-8", errors="ignore")
    return open(file_path, "r", encoding="utf-8", errors="ignore")


def parse_trace_line(line):
    match = match_pattern.search(line)
    if not match:
        return None

    clean_hash_ids = ast.literal_eval(match.group("ucm_block_ids"))
    return {
        "timestamp": float(match.group("timestamp")),
        "input_length": int(match.group("input_length")),
        "output_length": int(match.group("output_length")),
        "hash_ids": clean_hash_ids,
    }


def process_file(file_path, out_f):
    count = 0
    with open_log_file(file_path) as f:
        for line in f:
            data = parse_trace_line(line)
            if data is not None:
                out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
                count += 1
    return count


def resolve_target(target):
    path = Path(target)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = []
        for pattern in ("ucm.log*", "ucm_*.log.gz"):
            files.extend(path.glob(pattern))
        files = sorted(set(files))
        if not files:
            raise FileNotFoundError(f"No log files found in: {target}")
        return files

    matches = [Path(match) for match in sorted(glob.glob(target))]
    if matches:
        return [match for match in matches if match.is_file()]
    raise FileNotFoundError(f"Log files not found: {target}")


def resolve_input_files(targets):
    files = []
    seen = set()
    for target in targets:
        for file_path in resolve_target(target):
            resolved = file_path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(file_path)
    return files


def write_aggregate(files, output_log):
    output_log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(files)} log file(s) into {output_log}... ")
    total = 0
    with open(output_log, "w", encoding="utf-8") as out_f:
        for file_path in files:
            print(f"  -> {os.path.basename(file_path)}")
            count = process_file(file_path, out_f)
            total += count
            print(f"{count} records extracted from {file_path}")
    print(f"Done! Total: {total} records -> {output_log}")


def build_output_path(file_path, output_dir, used_paths):
    output_dir = output_dir or file_path.parent
    base_name = f"{file_path.name}.jsonl"
    output_path = output_dir / base_name
    index = 1
    while output_path in used_paths:
        output_path = output_dir / f"{file_path.name}.{index}.jsonl"
        index += 1
    used_paths.add(output_path)
    return output_path


def write_per_file(files, output_dir):
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(files)} log file(s) in per-file mode... ")
    total = 0
    used_paths = set()
    for file_path in files:
        output_path = build_output_path(file_path, output_dir, used_paths)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out_f:
            count = process_file(file_path, out_f)
        total += count
        print(f"{count} records extracted from {file_path} -> {output_path}")
    print(f"Done! Total: {total} records")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract trace replay jsonl records from UCM log files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Log file, directory, or multiple log files/directories.",
    )
    parser.add_argument(
        "--mode",
        choices=["aggregate", "per-file"],
        default="per-file",
        help="Write all records to one file or one output file per input log.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_LOG,
        help="Aggregate output jsonl path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used by --mode per-file. Defaults to each log file's directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        files = resolve_input_files(args.inputs)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    if args.mode == "aggregate":
        write_aggregate(files, args.output)
    else:
        write_per_file(files, args.output_dir)


if __name__ == "__main__":
    main()
