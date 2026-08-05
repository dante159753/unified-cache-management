from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported


def _update_multi_group_requests_with_invalid_blocks(
    self, requests, invalid_block_ids, num_scheduled_tokens=None, evict_blocks=True
):
    affected_req_ids = set()
    total_affected_tokens = 0
    blocks_to_evict = set()
    kv_cache_groups = self.kv_cache_config.kv_cache_groups
    for request in requests:
        req_id = request.request_id
        group_block_ids = self.kv_cache_manager.get_block_ids(req_id)
        req_num_computed_tokens = request.num_computed_tokens
        if num_scheduled_tokens is not None:
            req_num_computed_tokens -= num_scheduled_tokens.get(req_id, 0)
        elif getattr(getattr(request, "status", None), "name", None) != (
            "WAITING_FOR_REMOTE_KVS"
        ):
            req_num_computed_tokens = getattr(
                request, "num_cached_tokens", req_num_computed_tokens
            )
        valid_tokens = req_num_computed_tokens
        is_affected = False
        for group, block_ids in zip(kv_cache_groups, group_block_ids):
            group_block_size = group.kv_cache_spec.block_size
            num_computed_blocks = (
                req_num_computed_tokens + group_block_size - 1
            ) // group_block_size
            for idx, block_id in zip(range(num_computed_blocks), block_ids):
                if block_id not in invalid_block_ids:
                    continue
                is_affected = True
                valid_tokens = min(valid_tokens, idx * group_block_size)
                if evict_blocks:
                    blocks_to_evict.update(block_ids[idx:])
                break
        if not is_affected:
            continue
        block_size = getattr(self, "block_size", 0)
        if block_size:
            valid_tokens -= valid_tokens % block_size
        num_affected_tokens = req_num_computed_tokens - valid_tokens
        total_affected_tokens += num_affected_tokens
        request.num_computed_tokens = valid_tokens
        if hasattr(request, "num_external_computed_tokens"):
            request.num_external_computed_tokens -= num_affected_tokens
        if hasattr(request, "num_output_placeholders"):
            request.num_output_placeholders = 0
        affected_req_ids.add(req_id)
    return affected_req_ids, total_affected_tokens, blocks_to_evict


@when_imported("vllm.v1.core.sched.scheduler")
def patch_core_sched_scheduler(mod):
    """Wrap Scheduler._update_requests_with_invalid_blocks for KV load-failure
    recovery. Delegates to each version's own implementation, then applies
    UCM-specific post-processing. Hybrid configs (more than one KV cache
    group) take a UCM reimplementation instead: the upstream implementation
    unpacks a single-group block table and raises on hybrid models."""

    original = getattr(mod.Scheduler, "_update_requests_with_invalid_blocks", None)

    if original is not None:

        def wrapped_update(self, requests, *args, **kwargs):
            kv_cache_groups = getattr(
                getattr(self, "kv_cache_config", None), "kv_cache_groups", None
            )
            if kv_cache_groups is not None and len(kv_cache_groups) > 1:
                return _update_multi_group_requests_with_invalid_blocks(
                    self, requests, *args, **kwargs
                )

            # Delegate to the version-specific implementation
            result = original(self, requests, *args, **kwargs)

            if result:
                affected_req_ids = result[0]
                for request in requests:
                    if request.request_id in affected_req_ids:
                        request.num_output_placeholders = 0

            return result

        patch_or_inject(
            mod.Scheduler,
            "_update_requests_with_invalid_blocks",
            wrapped_update,
        )
