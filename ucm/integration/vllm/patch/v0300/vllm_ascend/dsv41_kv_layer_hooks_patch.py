"""Call the KV connector layer hooks from vllm-ascend's DeepSeek V4.1 attention.

vllm-ascend (0.30.0 and later) runs DeepSeek V4.1 attention through
``AscendDSAV41Impl.forward``. Unlike the V4 attention, it never calls
``wait_for_layer_load`` or ``save_kv_layer``, so the FAWA layerwise connector
cannot overlap its loads and dumps with the forward pass.

This patch calls the same vllm-ascend helpers that the V4 attention uses,
before and after each layer's attention. It only does so when a UCM connector
is active: other connectors have never received these calls for V4.1 and may
not handle the V4.1 layer names.
"""

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def _ucm_connector_is_active() -> bool:
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
    )

    if has_kv_transfer_group():
        connector_cls = type(get_kv_transfer_group())
        # vllm-ascend registers its own UCMConnectorV1 wrapper around UCM.
        return (
            connector_cls.__module__.startswith("ucm.")
            or connector_cls.__name__ == "UCMConnectorV1"
        )
    else:
        return False


@when_imported("vllm_ascend.attention.dsa_v41")
def patch_dsv41_kv_layer_hooks(mod):
    from vllm_ascend.attention.utils import (
        maybe_save_kv_layer_to_connector,
        wait_for_kv_layer_from_connector,
    )

    impl_cls = mod.AscendDSAV41Impl
    original_forward = impl_cls.forward
    if getattr(original_forward, "_ucm_kv_layer_hooks", False):
        logger.debug("DeepSeek V4.1 KV layer hooks are already patched.")
    else:

        def forward_with_kv_layer_hooks(
            self, attn, positions, hidden_states, output=None
        ):
            if _ucm_connector_is_active():
                layer_name = attn.v41_layer_name
                wait_for_kv_layer_from_connector(layer_name)
                result = original_forward(self, attn, positions, hidden_states, output)
                # UCM reads the KV caches through its own registered pointers,
                # so it does not need the layer's cache tensors here.
                maybe_save_kv_layer_to_connector(layer_name, [])
                return result
            else:
                return original_forward(self, attn, positions, hidden_states, output)

        forward_with_kv_layer_hooks._ucm_kv_layer_hooks = True
        impl_cls.forward = forward_with_kv_layer_hooks
        logger.info("UCM patched DeepSeek V4.1 attention with KV layer hooks.")
