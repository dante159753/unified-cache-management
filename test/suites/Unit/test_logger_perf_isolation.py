import logging
import os
import subprocess
import sys
import textwrap
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def run_python(code: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


class LoggerPerfIsolationTest(unittest.TestCase):
    def test_ucm_logger_import_does_not_require_ucmlogger_extension(self):
        result = run_python(
            """
            import importlib.abc
            import sys

            class BlockUcmlogger(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "ucm.shared.infra.ucmlogger":
                        raise ImportError("blocked ucmlogger extension")
                    return None

            sys.meta_path.insert(0, BlockUcmlogger())
            from ucm.logger import init_logger
            logger = init_logger("ucm.test")
            assert logger.isEnabledFor(50) is False
            """
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_noop_logger_methods_do_not_emit_logging_records(self):
        from ucm.logger import init_logger

        logger = init_logger("ucm.test")
        records = []

        class CaptureHandler(logging.Handler):
            def emit(self, record):
                records.append(record)

        root_logger = logging.getLogger()
        handler = CaptureHandler()
        old_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(handler)
        try:
            logger.debug("debug %s", "message")
            logger.info("info %s", "message")
            logger.warning("warning %s", "message")
            logger.error("error %s", "message")
            logger.critical("critical %s", "message")
            logger.log(logging.INFO, "log %s", "message")
            logger.info_once("info once %s", "message")
            logger.warning_once("warning once %s", "message")
            logger.debug_once("debug once %s", "message")
            logger.info_limit("info limit %s", "message")
            logger.warning_limit("warning limit %s", "message")
            logger.debug_limit("debug limit %s", "message")
            logger.exception("exception %s", "message")
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(old_level)

        self.assertFalse(logger.isEnabledFor(logging.CRITICAL))
        self.assertEqual(records, [])

    def test_import_ucm_does_not_patch_vllm_logger(self):
        result = run_python(
            """
            import sys
            import types

            fake_ucmlogger = types.ModuleType("ucm.shared.infra.ucmlogger")

            class Level:
                DEBUG = object()
                INFO = object()
                WARNING = object()
                ERROR = object()
                CRITICAL = object()

            fake_ucmlogger.Level = Level
            fake_ucmlogger.setup = lambda *args, **kwargs: None
            fake_ucmlogger.flush = lambda *args, **kwargs: None
            fake_ucmlogger.isEnabledFor = lambda *args, **kwargs: False
            fake_ucmlogger.log = lambda *args, **kwargs: None
            fake_ucmlogger.log_rate_limit = lambda *args, **kwargs: None
            sys.modules["ucm.shared.infra.ucmlogger"] = fake_ucmlogger

            vllm = types.ModuleType("vllm")
            vllm_logger = types.ModuleType("vllm.logger")

            def original_init_logger(name=None):
                return ("original", name)

            vllm_logger.init_logger = original_init_logger
            vllm.logger = vllm_logger
            sys.modules["vllm"] = vllm
            sys.modules["vllm.logger"] = vllm_logger

            import ucm  # noqa: F401

            assert vllm_logger.init_logger is original_init_logger
            """
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
