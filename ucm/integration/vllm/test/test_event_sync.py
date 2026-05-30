import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


class DummyLogger:
    def debug(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def install_stubs():
    def module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    ndarray = type("ndarray", (), {})
    module(
        "numpy",
        ndarray=ndarray,
        uint64=int,
        array=lambda data, dtype=None: data,
        asarray=lambda data, dtype=None: data,
        ascontiguousarray=lambda data: data,
        frombuffer=lambda data, dtype=None: data,
    )
    module(
        "torch",
        Tensor=type("Tensor", (), {}),
        dtype=type("dtype", (), {}),
        cuda=types.SimpleNamespace(
            current_stream=lambda: types.SimpleNamespace(synchronize=lambda: None),
            Event=lambda enable_timing=False: types.SimpleNamespace(
                cuda_event=1,
                record=lambda stream: None,
            ),
            get_device_properties=lambda local_rank: types.SimpleNamespace(
                pci_domain_id=0,
                pci_bus_id=0,
                pci_device_id=0,
            ),
            device_count=lambda: 1,
        ),
        npu=types.SimpleNamespace(
            current_stream=lambda: types.SimpleNamespace(
                npu_stream=1,
                synchronize=lambda: None,
            ),
            device=lambda name: name,
            device_count=lambda: 1,
        ),
    )

    ucm = module("ucm")
    ucm.__path__ = []
    integration = module("ucm.integration")
    integration.__path__ = []
    vllm_integration = module("ucm.integration.vllm")
    vllm_integration.__path__ = []

    module("ucm.logger", init_logger=lambda name: DummyLogger())
    module("ucm.observability", PrometheusStatsLogger=object)
    module(
        "ucm.shared.metrics",
        ucmmetrics=types.SimpleNamespace(update_stats=lambda *args, **kwargs: None),
    )
    module(
        "ucm.store.factory_v1",
        UcmConnectorFactoryV1=types.SimpleNamespace(
            Create=lambda *args, **kwargs: None
        ),
    )
    module("ucm.store.ucmstore_v1", Task=object, UcmKVStoreBaseV1=object)
    module("ucm.utils", Config=object)
    module("ucm.sparse.state", has_ucm_sparse=lambda: False)

    vllm = module("vllm")
    vllm.__path__ = []
    module("vllm.config", VllmConfig=object)
    module(
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        KVConnectorBase_V1=type("KVConnectorBase_V1", (), {}),
        KVConnectorMetadata=type("KVConnectorMetadata", (), {}),
        KVConnectorRole=types.SimpleNamespace(SCHEDULER="scheduler", WORKER="worker"),
        KVConnectorWorkerMetadata=type("KVConnectorWorkerMetadata", (), {}),
        SupportsHMA=type("SupportsHMA", (), {}),
    )
    module(
        "vllm.distributed.parallel_state",
        get_world_group=lambda: types.SimpleNamespace(rank=0, local_rank=0),
    )
    module(
        "vllm.model_executor.models.utils",
        extract_layer_index=lambda name: (
            int(name.rsplit(".", 1)[-1]) if name.rsplit(".", 1)[-1].isdigit() else 0
        ),
    )
    module(
        "vllm.platforms",
        current_platform=types.SimpleNamespace(
            is_cuda_alike=lambda: False,
            device_type="npu",
        ),
    )
    module("vllm.v1.core.sched.output", SchedulerOutput=object)
    module(
        "vllm.v1.kv_cache_interface",
        FullAttentionSpec=type("FullAttentionSpec", (), {}),
        KVCacheConfig=object,
        KVCacheSpec=object,
        MambaSpec=type("MambaSpec", (), {}),
        SlidingWindowSpec=type("SlidingWindowSpec", (), {}),
        UniformTypeKVCacheSpecs=object,
    )
    module("vllm.v1.outputs", KVConnectorOutput=object)


def load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


install_stubs()
device_module = load_module(
    "ucm.integration.vllm.device", "ucm/integration/vllm/device.py"
)
connector_module = load_module(
    "ucm.integration.vllm.ucm_connector", "ucm/integration/vllm/ucm_connector.py"
)


class EventHandleTest(unittest.TestCase):
    def test_cuda_existing_event_handle_is_borrowed(self):
        dev = device_module.CudaDevice()
        event = types.SimpleNamespace(cuda_event=11)

        self.assertEqual(dev.get_event_handle_from_event(event), 11)
        self.assertIs(dev.borrowed_events[0], event)
        dev.destroy_event_handles()
        self.assertEqual(dev.borrowed_events, [])

    def test_npu_existing_event_handle_from_npu_event_is_borrowed(self):
        dev = device_module.NpuDevice()
        event = types.SimpleNamespace(npu_event=22)

        self.assertEqual(dev.get_event_handle_from_event(event), 22)
        self.assertIs(dev.borrowed_events[0], event)

    def test_npu_existing_event_handle_from_as_parameter_is_borrowed(self):
        dev = device_module.NpuDevice()
        event = types.SimpleNamespace(
            npu_event=None,
            _as_parameter_=types.SimpleNamespace(value=33),
        )

        self.assertEqual(dev.get_event_handle_from_event(event), 33)
        self.assertIs(dev.borrowed_events[0], event)

    def test_invalid_existing_event_handle_returns_zero(self):
        dev = device_module.NpuDevice()
        event = types.SimpleNamespace(npu_event=0)

        self.assertEqual(dev.get_event_handle_from_event(event), 0)
        self.assertEqual(dev.borrowed_events, [])


class FakeDevice:
    def __init__(self, existing_handle=0, fallback_handle=0):
        self.existing_handle = existing_handle
        self.fallback_handle = fallback_handle
        self.existing_event = None
        self.fallback_called = False
        self.synchronize_called = False

    def get_event_handle_from_event(self, event):
        self.existing_event = event
        return self.existing_handle

    def get_event_handle(self):
        self.fallback_called = True
        return self.fallback_handle

    def synchronize(self):
        self.synchronize_called = True


class ConnectorEventSelectionTest(unittest.TestCase):
    def make_connector(self, device, enable_event_sync=True, is_mla=False):
        connector = object.__new__(connector_module.UCMDirectConnector)
        connector.device = device
        connector.enable_event_sync = enable_event_sync
        connector.is_mla = is_mla
        return connector

    def test_selects_direct_metadata_event(self):
        event = object()
        connector = self.make_connector(FakeDevice())
        metadata = types.SimpleNamespace(reshape_cache_event=event)

        selected_event, event_source = connector._get_reshape_cache_event(
            "layer.0", metadata
        )
        self.assertIs(selected_event, event)
        self.assertEqual(event_source, "direct")

    def test_non_mla_selects_direct_metadata_event_even_if_layer_event_exists(self):
        direct_event = object()
        layer_event = object()
        connector = self.make_connector(FakeDevice())

        class LayerMetadata:
            reshape_cache_event = layer_event

        class MappingMetadata:
            reshape_cache_event = direct_event

            def __getitem__(self, key):
                if key == "layer.0":
                    return LayerMetadata()
                raise KeyError(key)

        selected_event, event_source = connector._get_reshape_cache_event(
            "layer.0", MappingMetadata()
        )
        self.assertIs(selected_event, direct_event)
        self.assertEqual(event_source, "direct")

    def test_mla_selects_layer_metadata_event(self):
        direct_event = object()
        layer_event = object()
        connector = self.make_connector(FakeDevice(), is_mla=True)

        class LayerMetadata:
            reshape_cache_event = layer_event

        class MappingMetadata:
            reshape_cache_event = direct_event

            def __getitem__(self, key):
                if key == "layer.0":
                    return LayerMetadata()
                raise KeyError(key)

        selected_event, event_source = connector._get_reshape_cache_event(
            "layer.0", MappingMetadata()
        )
        self.assertIs(selected_event, layer_event)
        self.assertEqual(event_source, "mla_layer")

    def test_missing_metadata_event_uses_current_stream_fallback(self):
        fake_device = FakeDevice(fallback_handle=77)
        connector = self.make_connector(fake_device)

        self.assertEqual(connector._get_dump_event_handle("layer.0", object()), 77)
        self.assertTrue(fake_device.fallback_called)
        self.assertFalse(fake_device.synchronize_called)

    def test_existing_metadata_event_uses_existing_handle(self):
        event = object()
        fake_device = FakeDevice(existing_handle=55, fallback_handle=77)
        connector = self.make_connector(fake_device)
        metadata = types.SimpleNamespace(reshape_cache_event=event)

        self.assertEqual(connector._get_dump_event_handle("layer.0", metadata), 55)
        self.assertIs(fake_device.existing_event, event)
        self.assertFalse(fake_device.fallback_called)

    def test_event_sync_disabled_synchronizes_and_returns_zero(self):
        fake_device = FakeDevice(existing_handle=55, fallback_handle=77)
        connector = self.make_connector(fake_device, enable_event_sync=False)

        self.assertEqual(connector._get_dump_event_handle("layer.0", object()), 0)
        self.assertTrue(fake_device.synchronize_called)
        self.assertFalse(fake_device.fallback_called)


class FakeStore:
    def __init__(self):
        self.dump_calls = []

    def dump_data(self, block_ids, shard_indexs, layer_ptrs, event_handle):
        self.dump_calls.append((block_ids, shard_indexs, layer_ptrs, event_handle))
        return "task"


class FakeLayout:
    def extract_block_addrs(self, block_ids, layer_first=False):
        self.block_ids = block_ids
        self.layer_first = layer_first
        return [[[123]]]


class LayerwiseSaveTest(unittest.TestCase):
    def test_save_kv_layer_passes_reshape_event_handle_to_dump_data(self):
        event = object()
        fake_device = FakeDevice(existing_handle=88)
        connector = object.__new__(connector_module.UCMLayerWiseConnector)
        connector._connector_metadata = True
        connector.is_mla = False
        connector.tp_rank = 0
        connector.tp_size = 1
        connector.is_save = False
        connector.dump_total_ptrs = None
        connector.dump_tasks = {}
        connector.layer_name_to_id = {"layer.0": 0}
        connector.first_layer_id = 0
        connector.kv_cache_layout = FakeLayout()
        connector.store = FakeStore()
        connector.device = fake_device
        connector.enable_event_sync = True

        metadata = types.SimpleNamespace(
            request_meta={
                "request": types.SimpleNamespace(dump_block_ids=([b"key"], [1]))
            },
            reshape_cache_event=event,
        )
        connector._get_connector_metadata = lambda: metadata

        connector_module.UCMLayerWiseConnector.save_kv_layer(
            connector, "layer.0", None, metadata
        )

        self.assertEqual(connector.store.dump_calls[0][3], 88)
        self.assertEqual(connector.dump_tasks["layer.0"], "task")


if __name__ == "__main__":
    unittest.main()
