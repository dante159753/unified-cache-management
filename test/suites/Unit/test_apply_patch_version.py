import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch


def _load_apply_patch_module():
    logger = MagicMock()
    ucm_module = ModuleType("ucm")
    ucm_module.__path__ = []
    logger_module = ModuleType("ucm.logger")
    logger_module.init_logger = MagicMock(return_value=logger)
    module_path = (
        Path(__file__).parents[3]
        / "ucm"
        / "integration"
        / "vllm"
        / "patch"
        / "apply_patch.py"
    )
    spec = importlib.util.spec_from_file_location("ucm_apply_patch_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"ucm": ucm_module, "ucm.logger": logger_module},
    ):
        spec.loader.exec_module(module)
    return module, logger


def test_release_version_keeps_vllm_ascend_patch_version():
    module, logger = _load_apply_patch_module()
    with patch.object(
        module,
        "_read_vllm_ascend_version_raw",
        return_value="0.19.1rc2",
    ):
        assert module.get_vllm_ascend_patch_version("0.26.0") == "0.19.1"
    logger.warning.assert_not_called()


def test_development_version_uses_matching_vllm_patch_version():
    module, logger = _load_apply_patch_module()
    with patch.object(
        module,
        "_read_vllm_ascend_version_raw",
        return_value="0.19.1rc2.dev1475",
    ):
        assert module.get_vllm_ascend_patch_version("0.26.0+empty") == "0.26.0"
    logger.warning.assert_called_once()


def test_development_version_keeps_supported_matching_version():
    module, logger = _load_apply_patch_module()
    with patch.object(
        module,
        "_read_vllm_ascend_version_raw",
        return_value="0.26.0.dev12",
    ):
        assert module.get_vllm_ascend_patch_version("0.26.0") == "0.26.0"
    logger.warning.assert_not_called()


def test_development_version_does_not_select_unsupported_vllm_version():
    module, logger = _load_apply_patch_module()
    with patch.object(
        module,
        "_read_vllm_ascend_version_raw",
        return_value="0.19.1rc2.dev1475",
    ):
        assert module.get_vllm_ascend_patch_version("0.29.0") == "0.19.1"
    logger.warning.assert_not_called()
