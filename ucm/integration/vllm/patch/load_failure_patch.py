from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported


@when_imported("vllm.v1.core.sched.scheduler")
def patch_core_sched_scheduler(mod):
    """Wrap Scheduler._update_requests_with_invalid_blocks for KV load-failure
    recovery."""

    original = getattr(mod.Scheduler, "_update_requests_with_invalid_blocks", None)

    if original is not None:

        def wrapped_update(self, requests, invalid_block_ids, *args, **kwargs):
            num_groups = getattr(self.kv_cache_manager, "num_kv_cache_groups", 1)

            if num_groups > 1:
                original_get_block_ids = self.kv_cache_manager.get_block_ids

                def single_group_get_block_ids(req_id):
                    groups = original_get_block_ids(req_id)
                    return (groups[0],) if len(groups) > 1 else groups

                self.kv_cache_manager.get_block_ids = single_group_get_block_ids
                try:
                    result = original(
                        self, requests, invalid_block_ids, *args, **kwargs
                    )
                finally:
                    self.kv_cache_manager.get_block_ids = original_get_block_ids

                if result and len(result) >= 3:
                    blocks_to_evict = set(result[2])
                    blocks_to_evict.update(invalid_block_ids)
                    result = (result[0], result[1], blocks_to_evict)
            else:
                result = original(self, requests, invalid_block_ids, *args, **kwargs)

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
