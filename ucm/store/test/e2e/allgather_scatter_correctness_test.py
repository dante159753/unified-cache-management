from __future__ import annotations

import argparse

import torch
import torch_npu  # noqa: F401

from ucm.store.allgather.native_loader import load_segmented_copy

ucm_segmented_copy = load_segmented_copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=11)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--window-blocks", type=int, default=4)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    tensor_sizes = (131072, 16384, 256)
    shard_size = 151552
    rank_stride = args.window_blocks * shard_size
    receive = torch.zeros(
        args.world_size * rank_stride, dtype=torch.uint8, device=device
    )

    owner_counts = [0] * args.world_size
    routes: list[list[int]] = []
    expected: list[list[torch.Tensor]] = []
    destinations: list[list[torch.Tensor]] = []
    for row in range(args.rows):
        owner = row % args.world_size
        owner_slot = owner_counts[owner]
        owner_counts[owner] += 1
        if owner_slot >= args.window_blocks:
            raise ValueError("rows exceed one scatter window")
        routes.append([owner, owner_slot])
        base = owner * rank_stride + owner_slot * shard_size
        row_expected: list[torch.Tensor] = []
        row_destinations: list[torch.Tensor] = []
        offset = 0
        for tensor_index, size in enumerate(tensor_sizes):
            value = (row * len(tensor_sizes) + tensor_index + 1) % 251
            receive[base + offset : base + offset + size].fill_(value)
            row_expected.append(
                torch.full((size,), value, dtype=torch.uint8, device=device)
            )
            row_destinations.append(torch.zeros(size, dtype=torch.uint8, device=device))
            offset += size
        expected.append(row_expected)
        destinations.append(row_destinations)

    destination_addresses = torch.tensor(
        [[tensor.data_ptr() for tensor in row] for row in destinations],
        dtype=torch.int64,
        device=device,
    )
    route_tensor = torch.tensor(routes, dtype=torch.int32, device=device)
    chunks: list[list[int]] = []
    source_offset = 0
    for tensor_index, size in enumerate(tensor_sizes):
        for offset in range(0, size, 32 * 1024):
            chunks.append(
                [
                    tensor_index,
                    source_offset + offset,
                    offset,
                    min(32 * 1024, size - offset),
                ]
            )
        source_offset += size
    chunk_tensor = torch.tensor(chunks, dtype=torch.int64, device=device)

    ucm_segmented_copy.scatter(
        receive,
        destination_addresses,
        route_tensor,
        chunk_tensor,
        args.rows,
        rank_stride,
        shard_size,
    )
    torch.npu.synchronize()
    for row in range(args.rows):
        for tensor_index in range(len(tensor_sizes)):
            torch.testing.assert_close(
                destinations[row][tensor_index], expected[row][tensor_index]
            )
    print("ALLGATHER_SCATTER_CORRECTNESS_PASS")


if __name__ == "__main__":
    main()
