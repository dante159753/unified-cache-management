"""Split DeepSeek V4/V4.1 FAWA store rows into one shard per vllm-ascend layer tuple.

vllm-ascend groups the DeepSeek V4 and V4.1 KV caches into layer tuples. Every
cache layer belongs to exactly one tuple, and every tuple has the same set of
cache kinds. Storing one shard per tuple gives every shard of a store the same
layout, so a shard can be loaded or dumped as soon as the decoder layers that
own it are reached.

This module only depends on numpy so that it can be unit tested without vLLM.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StoreRowColumn:
    """One device segment of a FAWA store row.

    A store row repeats each KV cache group's tensor views once per tail block,
    group by group, in the same order as the store's ``tensor_size_list``.
    """

    group_id: int
    tail_block_index: int
    layer_name: str
    view_index_in_layer: int
    size_bytes: int


def describe_store_row_columns(
    group_ids: Iterable[int],
    group_layouts: Mapping[int, object],
    group_metas: Mapping[int, object],
) -> list[StoreRowColumn]:
    """Describe every column of a store row built from these KV cache groups.

    ``group_layouts`` holds ``KVCacheGroupLayout`` objects and ``group_metas``
    holds ``KVCacheGroupMeta`` objects. Groups without a layout or without a
    stored tail are not part of the row.
    """
    columns: list[StoreRowColumn] = []
    for group_id in group_ids:
        layout = group_layouts.get(group_id)
        meta = group_metas[group_id]
        if layout is not None and meta.tail_tokens:
            segment_tokens = meta.tail_tokens // meta.tail_blocks
            segment_sizes = layout.segment_tensor_size_list(
                segment_tokens, meta.token_block_size
            )
            layer_names = [view["name"] for view in layout.view_meta]
            view_indices_in_layer = _count_views_within_layer(layer_names)
            for tail_block_index in range(meta.tail_blocks):
                for layer_name, view_index, size_bytes in zip(
                    layer_names, view_indices_in_layer, segment_sizes
                ):
                    columns.append(
                        StoreRowColumn(
                            group_id=group_id,
                            tail_block_index=tail_block_index,
                            layer_name=layer_name,
                            view_index_in_layer=view_index,
                            size_bytes=int(size_bytes),
                        )
                    )
    return columns


def _count_views_within_layer(layer_names: Sequence[str]) -> list[int]:
    views_seen_by_layer: dict[str, int] = {}
    view_indices = []
    for layer_name in layer_names:
        view_index = views_seen_by_layer.get(layer_name, 0)
        view_indices.append(view_index)
        views_seen_by_layer[layer_name] = view_index + 1
    return view_indices


def assign_layer_tuple_slots(kv_cache_tensors: Iterable[object]) -> dict[str, int]:
    """Return the layer tuple slot of every KV cache layer.

    vllm-ascend describes the tuples with ``KVCacheTensor`` in two ways:

    - DeepSeek V4 puts all tuples in one buffer. ``layers[i]`` starts at byte
      ``offset + i * layer_stride``, and the regular descriptors share the
      tuple size as ``layer_stride``, so the slot is that start byte divided
      by the tuple size. MTP descriptors use ``layer_stride == 0`` and put
      their slot in ``offset``.
    - DeepSeek V4.1 gives each tuple its own buffer. Every descriptor has
      ``layer_stride == 0`` and lists all layers of one tuple, so the slot is
      the descriptor index.
    """
    kv_cache_tensors = list(kv_cache_tensors)
    tuple_sizes = {
        tensor.layer_stride for tensor in kv_cache_tensors if tensor.layer_stride > 0
    }
    slot_by_layer: dict[str, int] = {}

    def place(layer_name: str, slot: int) -> None:
        if slot_by_layer.setdefault(layer_name, slot) != slot:
            raise ValueError(
                f"KV cache layer {layer_name} is placed in two layer tuples: "
                f"{slot_by_layer[layer_name]} and {slot}."
            )

    if len(tuple_sizes) == 1:
        tuple_size = tuple_sizes.pop()
        for tensor in kv_cache_tensors:
            for position, layer_name in enumerate(tensor.layers):
                place(
                    layer_name,
                    (tensor.offset + position * tensor.layer_stride) // tuple_size,
                )
    elif not tuple_sizes:
        for slot, tensor in enumerate(kv_cache_tensors):
            for layer_name in tensor.layers:
                place(layer_name, slot)
    else:
        raise ValueError(
            "Layer tuple placement needs one shared positive layer_stride, "
            f"got {sorted(tuple_sizes)}."
        )
    return slot_by_layer


def split_layer_name(layer_name: str) -> tuple[str, str]:
    """Split a cache name at the layer number.

    ``model.layers.6.self_attn.swa_cache`` gives ``("model.layers.6",
    "self_attn.swa_cache")``, and ``mtp.0.self_attn.swa_cache`` gives
    ``("mtp.0", "self_attn.swa_cache")``. The first part names the decoder
    layer, the second part names the cache kind inside it.
    """
    name_parts = layer_name.split(".")
    for index, name_part in enumerate(name_parts):
        if name_part.isdigit():
            return (
                ".".join(name_parts[: index + 1]),
                ".".join(name_parts[index + 1 :]),
            )
    return layer_name, layer_name


def is_draft_layer_name(layer_name: str) -> bool:
    """Whether a cache belongs to an MTP or DSpark draft layer."""
    return ".mtp." in f".{layer_name}"


@dataclass(frozen=True)
class LayerTupleShardPlan:
    """How one FAWA store row is split into equally sized shards.

    Shard ``i`` holds the layers of layer tuple ``tuple_slots[i]``. Segment
    ``k`` of a shard copies ``shard_tensor_sizes[k]`` bytes starting
    ``shard_column_offsets[i, k]`` bytes into row column ``shard_columns[i,
    k]``. A column of ``-1`` is padding: that tuple has no data there, so the
    segment keeps its bytes in the shard but has no device memory behind it.
    """

    row_width: int
    tuple_slots: tuple[int, ...]
    shard_columns: np.ndarray
    shard_column_offsets: np.ndarray
    shard_tensor_sizes: tuple[int, ...]
    shard_layer_names: tuple[frozenset[str], ...]

    @property
    def num_shards(self) -> int:
        return len(self.tuple_slots)

    @property
    def shard_payload_bytes(self) -> int:
        return sum(self.shard_tensor_sizes)

    def gather_shard_ptrs(self, row_ptrs: np.ndarray, shard_index: int) -> np.ndarray:
        """Pick the device pointers of one shard out of full store rows.

        Padding segments get a null pointer. The stores skip null segments when
        they copy, but still keep their bytes so that every shard has the same
        layout.
        """
        if row_ptrs.ndim == 2 and row_ptrs.shape[1] == self.row_width:
            columns = self.shard_columns[shard_index]
            offsets = self.shard_column_offsets[shard_index]
            has_device_memory = columns >= 0
            shard_ptrs = np.zeros((row_ptrs.shape[0], columns.size), dtype=np.uint64)
            shard_ptrs[:, has_device_memory] = (
                row_ptrs[:, columns[has_device_memory]] + offsets[has_device_memory]
            )
            return shard_ptrs
        else:
            raise ValueError(
                f"Store rows have shape {row_ptrs.shape}, expected "
                f"(num_keys, {self.row_width})."
            )


def build_layer_tuple_shard_plan(
    row_columns: Sequence[StoreRowColumn],
    tuple_slot_by_layer: Mapping[str, int],
) -> LayerTupleShardPlan:
    """Group the columns of a store row by layer tuple.

    Columns of different tuples line up when they have the same KV cache
    group, tail block, cache kind and view, for example "tail block 2 of the
    odd-layer SWA cache". Every shard gets the same segments for each such
    position, in the order the positions first appear in the row.

    DeepSeek V4.1 stores whole tuple pages, and tuples can have different page
    sizes. Such a position is cut at every page size that occurs: with pages
    of 100 and 120 bytes it becomes a 100-byte and a 20-byte segment, and the
    100-byte page leaves the second segment as padding. A copy never reads or
    writes past the end of a page.
    """
    sizes_by_position: dict[tuple, set[int]] = {}
    column_by_slot_and_position: dict[int, dict[tuple, int]] = {}
    for column_index, column in enumerate(row_columns):
        if column.layer_name in tuple_slot_by_layer:
            slot = tuple_slot_by_layer[column.layer_name]
        else:
            raise ValueError(
                f"KV cache layer {column.layer_name} has no layer tuple placement."
            )
        position = (
            column.group_id,
            column.tail_block_index,
            split_layer_name(column.layer_name)[1],
            column.view_index_in_layer,
        )
        sizes_by_position.setdefault(position, set()).add(column.size_bytes)
        columns_in_slot = column_by_slot_and_position.setdefault(slot, {})
        if position in columns_in_slot:
            other_layer = row_columns[columns_in_slot[position]].layer_name
            raise ValueError(
                f"Layers {other_layer} and {column.layer_name} fill the same "
                f"segment {position} of layer tuple {slot}."
            )
        columns_in_slot[position] = column_index

    # Each segment is (position, start byte in the column, length).
    segments = []
    for position, sizes in sizes_by_position.items():
        segment_start = 0
        for segment_end in sorted(sizes):
            segments.append((position, segment_start, segment_end - segment_start))
            segment_start = segment_end

    tuple_slots = tuple(sorted(column_by_slot_and_position))
    shard_columns = np.full((len(tuple_slots), len(segments)), -1, dtype=np.int64)
    shard_column_offsets = np.zeros((len(tuple_slots), len(segments)), dtype=np.uint64)
    shard_layer_names = []
    for shard_index, slot in enumerate(tuple_slots):
        columns_in_slot = column_by_slot_and_position[slot]
        for segment_index, (position, segment_start, _) in enumerate(segments):
            column_index = columns_in_slot.get(position, -1)
            if (
                column_index >= 0
                and row_columns[column_index].size_bytes > segment_start
            ):
                shard_columns[shard_index, segment_index] = column_index
                shard_column_offsets[shard_index, segment_index] = segment_start
        shard_layer_names.append(
            frozenset(
                row_columns[index].layer_name for index in columns_in_slot.values()
            )
        )
    return LayerTupleShardPlan(
        row_width=len(row_columns),
        tuple_slots=tuple_slots,
        shard_columns=shard_columns,
        shard_column_offsets=shard_column_offsets,
        shard_tensor_sizes=tuple(length for _, _, length in segments),
        shard_layer_names=tuple(shard_layer_names),
    )
