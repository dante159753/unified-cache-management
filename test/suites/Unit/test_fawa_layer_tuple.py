"""CPU tests for splitting DeepSeek V4 FAWA store rows by layer tuple."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "ucm/integration/vllm/fawa_layer_tuple.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fawa_layer_tuple", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


layer_tuple = _load_module()

# DeepSeek V4 Flash geometry seen on Ascend with vllm-ascend 0.30.0 and DSpark:
# C4 layers are 2..42 (even), C128 layers are 3..41 (odd), layers 0 and 1 only
# have a SWA cache, and three draft layers sit in trailing tuples.
TUPLE_BYTES = 1000
C4_LAYERS = list(range(2, 43, 2))
C128_LAYERS = list(range(3, 42, 2))
EVEN_SWA_LAYERS = list(range(0, 43, 2))
ODD_SWA_LAYERS = list(range(1, 42, 2))
ATTN_PAGE, INDEXER_K, INDEXER_SCALE, STATE_PAGE, INDEXER_STATE = (
    32768,
    4096,
    64,
    32768,
    4160,
)


def cache(layer, name):
    return f"model.layers.{layer}.self_attn.{name}"


def descriptor(layer_names, page_offset):
    return NS(layers=layer_names, offset=page_offset, layer_stride=TUPLE_BYTES)


def draft_descriptor(layer_name, slot):
    return NS(layers=[layer_name], offset=slot * TUPLE_BYTES + 100, layer_stride=0)


def kv_cache_tensors():
    return [
        descriptor([cache(i, "indexer.k_cache") for i in C4_LAYERS], 0),
        descriptor([cache(i, "attn") for i in C4_LAYERS], 100),
        descriptor([cache(i, "attn") for i in C128_LAYERS], 100),
        descriptor([cache(i, "swa_cache") for i in EVEN_SWA_LAYERS], 100),
        descriptor([cache(i, "swa_cache") for i in ODD_SWA_LAYERS], 100),
        descriptor([cache(i, "indexer.compressor.state_cache") for i in C4_LAYERS], 0),
        descriptor([cache(i, "compressor.state_cache") for i in C4_LAYERS], 100),
        descriptor([cache(i, "compressor.state_cache") for i in C128_LAYERS], 100),
        draft_descriptor("mtp.1.self_attn.swa_cache", 22),
        draft_descriptor("mtp.0.self_attn.swa_cache", 23),
        draft_descriptor("mtp.2.self_attn.swa_cache", 24),
    ]


class FakeGroupLayout:
    """Stands in for KVCacheGroupLayout: views and their per-block sizes."""

    def __init__(self, views):
        self.view_meta = [{"name": layer_name} for layer_name, _ in views]
        self.view_block_sizes = [size for _, size in views]

    def segment_tensor_size_list(self, logical_tokens, group_token_block_size):
        return [
            size * logical_tokens // group_token_block_size
            for size in self.view_block_sizes
        ]


def group_meta(token_block_size, tail_blocks, tail_tokens):
    return NS(
        token_block_size=token_block_size,
        tail_blocks=tail_blocks,
        tail_tokens=tail_tokens,
    )


GROUP_METAS = {
    0: group_meta(128, 1, 128),
    1: group_meta(4096, 1, 128),
    2: group_meta(32, 4, 128),
    3: group_meta(32, 4, 128),
    4: group_meta(2, 2, 4),
    5: group_meta(8, 1, 0),
}
FA_GROUP_IDS = [0, 1]
WA_GROUP_IDS = [2, 3, 4, 5]


def group_layouts():
    c4_views = []
    for layer in C4_LAYERS:
        c4_views += [
            (cache(layer, "attn"), ATTN_PAGE),
            (cache(layer, "indexer.k_cache"), INDEXER_K),
            (cache(layer, "indexer.k_cache"), INDEXER_SCALE),
        ]
    c4_state_views = []
    for layer in C4_LAYERS:
        c4_state_views += [
            (cache(layer, "compressor.state_cache"), STATE_PAGE),
            (cache(layer, "indexer.compressor.state_cache"), INDEXER_STATE),
        ]
    return {
        0: FakeGroupLayout(c4_views),
        1: FakeGroupLayout([(cache(i, "attn"), ATTN_PAGE) for i in C128_LAYERS]),
        2: FakeGroupLayout(
            [("mtp.0.self_attn.swa_cache", ATTN_PAGE)]
            + [(cache(i, "swa_cache"), ATTN_PAGE) for i in EVEN_SWA_LAYERS]
        ),
        3: FakeGroupLayout(
            [(cache(i, "swa_cache"), ATTN_PAGE) for i in ODD_SWA_LAYERS]
            + [
                ("mtp.1.self_attn.swa_cache", ATTN_PAGE),
                ("mtp.2.self_attn.swa_cache", ATTN_PAGE),
            ]
        ),
        4: FakeGroupLayout(c4_state_views),
        5: FakeGroupLayout(
            [(cache(i, "compressor.state_cache"), STATE_PAGE) for i in C128_LAYERS]
        ),
    }


def build_plan(group_ids):
    columns = layer_tuple.describe_store_row_columns(
        group_ids, group_layouts(), GROUP_METAS
    )
    slots = layer_tuple.assign_layer_tuple_slots(kv_cache_tensors())
    return columns, layer_tuple.build_layer_tuple_shard_plan(columns, slots)


def test_slots_follow_descriptor_offsets_including_draft_layers():
    slots = layer_tuple.assign_layer_tuple_slots(kv_cache_tensors())

    assert slots[cache(2, "indexer.k_cache")] == 0
    assert slots[cache(2, "attn")] == 0
    assert slots[cache(3, "attn")] == 0
    assert slots[cache(0, "swa_cache")] == 0
    assert slots[cache(1, "swa_cache")] == 0
    assert slots[cache(42, "attn")] == 20
    assert slots[cache(42, "swa_cache")] == 21
    assert slots["mtp.1.self_attn.swa_cache"] == 22
    assert slots["mtp.0.self_attn.swa_cache"] == 23
    assert slots["mtp.2.self_attn.swa_cache"] == 24


def test_row_columns_repeat_views_once_per_tail_block():
    columns = layer_tuple.describe_store_row_columns(
        [4, 5], group_layouts(), GROUP_METAS
    )

    # Group 5 keeps no tail, so only group 4 is stored: 2 blocks x 42 views.
    assert len(columns) == 2 * 2 * len(C4_LAYERS)
    first_block = columns[: 2 * len(C4_LAYERS)]
    assert {column.tail_block_index for column in first_block} == {0}
    assert columns[0].layer_name == cache(2, "compressor.state_cache")
    assert columns[0].size_bytes == STATE_PAGE
    assert columns[2 * len(C4_LAYERS)].tail_block_index == 1


def test_fa_store_has_one_equal_shard_per_c4_tuple():
    columns, plan = build_plan(FA_GROUP_IDS)

    assert plan.tuple_slots == tuple(range(21))
    assert plan.shard_tensor_sizes == (ATTN_PAGE, INDEXER_K, INDEXER_SCALE, 1024)
    assert plan.shard_payload_bytes * plan.num_shards >= sum(
        column.size_bytes for column in columns
    )
    assert plan.shard_layer_names[0] == {
        cache(2, "attn"),
        cache(2, "indexer.k_cache"),
        cache(3, "attn"),
    }
    # Layer 43 does not exist, so the last tuple pads the C128 segment.
    assert plan.shard_columns[-1, -1] == -1
    assert (plan.shard_columns[:-1] >= 0).all()


def test_wa_store_pads_tuples_without_every_window_cache():
    _, plan = build_plan(WA_GROUP_IDS)

    assert plan.tuple_slots == tuple(range(25))
    # 4 even SWA blocks, 4 odd SWA blocks, 2 blocks x 2 C4 state views.
    assert len(plan.shard_tensor_sizes) == 12
    assert (plan.shard_columns[0] >= 0).all()
    assert plan.shard_layer_names[21] == {cache(42, "swa_cache")}
    assert (plan.shard_columns[21] >= 0).sum() == 4
    for draft_slot, draft_layer in ((22, "mtp.1"), (23, "mtp.0"), (24, "mtp.2")):
        assert plan.shard_layer_names[draft_slot] == {
            f"{draft_layer}.self_attn.swa_cache"
        }
        assert (plan.shard_columns[draft_slot] >= 0).sum() == 4


def test_gather_shard_ptrs_picks_columns_and_nulls_padding():
    columns, plan = build_plan(FA_GROUP_IDS)
    num_keys = 2
    row_ptrs = (
        np.arange(len(columns), dtype=np.uint64)[None, :] * 10
        + np.arange(1, num_keys + 1, dtype=np.uint64)[:, None]
    )

    last_shard = plan.gather_shard_ptrs(row_ptrs, plan.num_shards - 1)

    assert last_shard.shape == (num_keys, 4)
    assert last_shard.dtype == np.uint64
    expected_columns = plan.shard_columns[-1, :3]
    assert (last_shard[:, :3] == row_ptrs[:, expected_columns]).all()
    assert (last_shard[:, 3] == 0).all()
    assert columns[expected_columns[0]].layer_name == cache(42, "attn")


def test_gather_shard_ptrs_rejects_rows_of_another_store():
    _, plan = build_plan(FA_GROUP_IDS)

    with pytest.raises(ValueError, match="expected"):
        plan.gather_shard_ptrs(np.zeros((1, plan.row_width + 1), np.uint64), 0)


def test_each_shard_of_a_store_moves_the_same_bytes():
    for group_ids in (FA_GROUP_IDS, WA_GROUP_IDS):
        columns, plan = build_plan(group_ids)
        for shard_columns in plan.shard_columns:
            for position, column_index in enumerate(shard_columns):
                if column_index >= 0:
                    assert (
                        columns[column_index].size_bytes
                        == plan.shard_tensor_sizes[position]
                    )


def test_split_layer_name_separates_decoder_layer_and_cache_kind():
    assert layer_tuple.split_layer_name(cache(6, "swa_cache")) == (
        "model.layers.6",
        "self_attn.swa_cache",
    )
    assert layer_tuple.split_layer_name("mtp.0.self_attn.swa_cache") == (
        "mtp.0",
        "self_attn.swa_cache",
    )
    # Attention hook names map to the same decoder layer as its caches.
    assert layer_tuple.split_layer_name(cache(6, "v41_attn"))[0] == "model.layers.6"
    assert layer_tuple.split_layer_name("no_number") == ("no_number", "no_number")


def test_draft_layer_names_are_detected():
    assert layer_tuple.is_draft_layer_name("mtp.0.self_attn.swa_cache")
    assert layer_tuple.is_draft_layer_name("model.mtp.1.self_attn.swa_cache")
    assert not layer_tuple.is_draft_layer_name(cache(3, "swa_cache"))


def test_layers_without_placement_are_rejected():
    columns = layer_tuple.describe_store_row_columns(
        FA_GROUP_IDS, group_layouts(), GROUP_METAS
    )
    slots = layer_tuple.assign_layer_tuple_slots(kv_cache_tensors())
    del slots[cache(4, "attn")]

    with pytest.raises(ValueError, match="no layer tuple placement"):
        layer_tuple.build_layer_tuple_shard_plan(columns, slots)


def test_two_layers_in_the_same_tuple_segment_are_rejected():
    columns = layer_tuple.describe_store_row_columns(
        FA_GROUP_IDS, group_layouts(), GROUP_METAS
    )
    slots = layer_tuple.assign_layer_tuple_slots(kv_cache_tensors())
    slots[cache(4, "attn")] = slots[cache(2, "attn")]

    with pytest.raises(ValueError, match="fill the same segment"):
        layer_tuple.build_layer_tuple_shard_plan(columns, slots)


def test_v41_descriptors_give_one_tuple_each():
    tensors = [
        NS(
            layers=[cache(2, "long_kv_cache"), cache(2, "indexer.k_cache"), "s0"],
            offset=0,
            layer_stride=0,
        ),
        NS(layers=[cache(4, "long_kv_cache"), "s1"], offset=0, layer_stride=0),
    ]

    slots = layer_tuple.assign_layer_tuple_slots(tensors)

    assert slots == {
        cache(2, "long_kv_cache"): 0,
        cache(2, "indexer.k_cache"): 0,
        "s0": 0,
        cache(4, "long_kv_cache"): 1,
        "s1": 1,
    }


def test_v41_whole_pages_of_different_sizes_are_cut_into_segments():
    # DeepSeek V4.1 stores one whole tuple page per owner layer, and tuples
    # can have different page sizes.
    page_sizes = {2: 120, 4: 100, 6: 120}
    columns = [
        layer_tuple.StoreRowColumn(0, 0, cache(layer, "long_kv_cache"), 0, size)
        for layer, size in page_sizes.items()
    ]
    slots = {
        cache(layer, "long_kv_cache"): slot for slot, layer in enumerate(page_sizes)
    }

    plan = layer_tuple.build_layer_tuple_shard_plan(columns, slots)

    assert plan.shard_tensor_sizes == (100, 20)
    assert plan.shard_columns.tolist() == [[0, 0], [1, -1], [2, 2]]
    assert plan.shard_column_offsets.tolist() == [[0, 100], [0, 0], [0, 100]]
    row_ptrs = np.array([[1000, 2000, 3000]], dtype=np.uint64)
    assert plan.gather_shard_ptrs(row_ptrs, 0).tolist() == [[1000, 1100]]
    # The 100-byte page never copies past its own end.
    assert plan.gather_shard_ptrs(row_ptrs, 1).tolist() == [[2000, 0]]


def test_v4_shards_need_no_cuts():
    for group_ids in (FA_GROUP_IDS, WA_GROUP_IDS):
        _, plan = build_plan(group_ids)
        assert not plan.shard_column_offsets.any()


def test_descriptors_with_different_tuple_sizes_are_rejected():
    tensors = kv_cache_tensors()
    tensors[0].layer_stride = TUPLE_BYTES * 2

    with pytest.raises(ValueError, match="one shared positive layer_stride"):
        layer_tuple.assign_layer_tuple_slots(tensors)


def test_layer_placed_in_two_tuples_is_rejected():
    tensors = kv_cache_tensors()
    tensors.append(draft_descriptor(cache(2, "attn"), 5))

    with pytest.raises(ValueError, match="two layer tuples"):
        layer_tuple.assign_layer_tuple_slots(tensors)
