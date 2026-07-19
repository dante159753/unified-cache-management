import re
from pathlib import Path
from typing import Optional
import unittest


class YuanRongDumpQueueSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "ucm"
            / "store"
            / "yuanrongstore"
            / "cc"
            / "dump_queue.cc"
        )
        load_source_path = source_path.with_name("load_queue.cc")
        backfill_source_path = source_path.with_name("backfill_queue.cc")
        store_source_path = source_path.with_name("yuanrong_store.cc")
        cls.source = source_path.read_text()
        cls.load_source = load_source_path.read_text()
        cls.store_source = store_source_path.read_text()
        cls.backfill_source = (
            backfill_source_path.read_text() if backfill_source_path.exists() else ""
        )

    def _function_body(self, name: str, source: Optional[str] = None) -> str:
        source = self.source if source is None else source
        match = re.search(rf"{re.escape(name)}\([^)]*\)\s*\{{", source)
        self.assertIsNotNone(match, f"{name} not found")
        start = match.end()
        depth = 1
        index = start
        while index < len(source) and depth:
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
            index += 1
        self.assertEqual(depth, 0, f"{name} body is not balanced")
        return source[start : index - 1]

    def test_prerequisite_stream_is_created_in_worker_thread(self):
        setup_body = self._function_body("DumpQueue::Setup")
        worker_body = self._function_body("DumpQueue::WorkerStage")

        self.assertNotIn("MakeSharedStream", setup_body)
        self.assertIn("device.Setup(config_.deviceId)", worker_body)
        self.assertIn("MakeSharedStream", worker_body)

    def test_mset_d2h_overwrites_existing_yuanrong_objects(self):
        dump_one_body = self._function_body("DumpQueue::DumpOne")

        self.assertIn("setParam.existence = datasystem::ExistenceOpt::NONE", dump_one_body)
        self.assertNotIn("setParam.existence = datasystem::ExistenceOpt::NX", dump_one_body)

    def test_yuanrong_dump_uses_payload_address_after_metadata_header(self):
        dump_one_body = self._function_body("DumpQueue::DumpOne")
        load_recover_body = self._function_body(
            "BackfillQueue::RunOne", self.backfill_source
        )

        self.assertNotIn("GetSize() != static_cast<int64_t>(config_.objectSize)", dump_one_body)
        self.assertNotIn("size != static_cast<int64_t>(config_.objectSize)", load_recover_body)
        self.assertIn("GetYuanRongPayloadAddress", dump_one_body)
        self.assertIn("payloadAddress", dump_one_body)
        self.assertIn("YuanRongComposedObjectSize", load_recover_body)
        self.assertIn("InitYuanRongComposedBuffer", load_recover_body)

    def test_posix_dump_validates_yuanrong_payload_layout_before_using_buffer(self):
        dump_one_body = self._function_body("DumpQueue::DumpOne")

        self.assertIn("GetMetaInfo(keys, false", dump_one_body)
        self.assertIn("ValidateYuanRongBlobSizes", dump_one_body)
        self.assertLess(
            dump_one_body.index("ValidateYuanRongBlobSizes"),
            dump_one_body.index("kvClient_->Get"),
        )

    def test_dump_persists_only_mset_confirmed_local_keys(self):
        dump_one_body = self._function_body("DumpQueue::DumpOne")

        self.assertIn(
            "MSetD2H(keys, blobLists, setParam, &localSetKeys)",
            dump_one_body,
        )
        self.assertIn(
            "SelectYuanRongObjectsByKeys(localSetKeys, keys, task->desc)",
            dump_one_body,
        )
        self.assertIn("persisting the confirmed keys only", dump_one_body)
        self.assertNotIn("heteroClient_->Exist", dump_one_body)
        self.assertNotIn("postExist", dump_one_body)

    def test_mset_error_without_confirmed_local_key_fails_dump(self):
        dump_one_body = self._function_body("DumpQueue::DumpOne")

        empty_pos = dump_one_body.index("if (localSetKeys.empty())")
        error_pos = dump_one_body.index("if (dumpStatus.IsError())", empty_pos)
        selection_pos = dump_one_body.index("SelectYuanRongObjectsByKeys")
        partial_pos = dump_one_body.index("partially failed", selection_pos)
        self.assertLess(empty_pos, error_pos)
        self.assertLess(error_pos, selection_pos)
        self.assertLess(selection_pos, partial_pos)

    def test_partial_kv_get_persists_only_latched_buffers(self):
        dump_one_body = self._function_body("DumpQueue::DumpOne")

        self.assertIn("if (!buffer)", dump_one_body)
        self.assertIn("lockedBuffers.push_back(std::move(buffer))", dump_one_body)
        self.assertIn("if (backendTask.empty())", dump_one_body)
        self.assertIn("std::move(lockedBuffers)", dump_one_body)
        self.assertIn("persisting readable buffers only", dump_one_body)

    def test_backend_submit_logs_time_since_d2h_completion(self):
        dump_one_body = self._function_body("DumpQueue::DumpOne")

        self.assertIn("d2h_to_backend_submit={:.3f}ms", dump_one_body)
        self.assertIn("(backendSubmitEnd - publishEnd) * 1e3", dump_one_body)
        self.assertIn("local_set_keys={}", dump_one_body)

    def test_yuanrong_load_probes_miss_with_zero_timeout(self):
        load_one_body = self._function_body("LoadQueue::LoadOne", self.load_source)

        self.assertIn("constexpr int32_t mgetTimeoutMs = 0", load_one_body)
        self.assertIn(
            "MGetH2D(keys, blobLists, failedKeys, mgetTimeoutMs)", load_one_body
        )
        self.assertNotIn("firstGetTimeoutMs", load_one_body)
        self.assertNotIn("missTimeoutMs", load_one_body)

    def test_yuanrong_posix_miss_h2d_precedes_async_backfill(self):
        recover_body = self._function_body(
            "LoadQueue::RecoverFromBackend", self.load_source
        )
        finalize_body = self._function_body(
            "LoadQueue::FinalizeHostBatch", self.load_source
        )

        self.assertIn("config_.recoveryBatchSize", recover_body)
        self.assertIn("PrepareHostBatch", recover_body)
        self.assertIn("FinalizeHostBatch", recover_body)
        wait_pos = finalize_body.index("backend_->Wait")
        h2d_pos = finalize_body.index("HostToDeviceScatterAsync")
        sync_pos = finalize_body.index("stream.Synchronize")
        backfill_pos = finalize_body.index("backfillQueue_.Submit")
        self.assertLess(wait_pos, h2d_pos)
        self.assertLess(h2d_pos, sync_pos)
        self.assertLess(sync_pos, backfill_pos)
        self.assertNotIn("MGetH2D", finalize_body)

    def test_yuanrong_async_backfill_does_not_fail_front_load(self):
        run_body = self._function_body("BackfillQueue::RunOne", self.backfill_source)

        self.assertIn("kvClient_->MCreate", run_body)
        self.assertIn("kvClient_->MSet", run_body)
        self.assertIn("InitYuanRongComposedBuffer", run_body)
        self.assertNotIn("failureSet_", run_body)

    def test_invalid_queue_config_identifies_the_parameter_and_values(self):
        check_body = self._function_body("CheckConfig", self.store_source)

        self.assertNotIn("invalid YuanRong queue depth", check_body)
        self.assertIn("yuanrong_waiting_queue_depth({})", check_body)
        self.assertIn("yuanrong_load_worker_count", check_body)
        self.assertIn("yuanrong_recovery_batch_size", check_body)
        self.assertIn("yuanrong_host_buffer_count({})", check_body)
        self.assertIn("config.hostBufferCount, config.recoveryBatchSize", check_body)
        self.assertIn("yuanrong_h2d_stream_count", check_body)
        self.assertIn("yuanrong_backfill_worker_count", check_body)
        self.assertIn("yuanrong_backfill_queue_depth", check_body)
        self.assertIn("yuanrong_reaper_queue_depth({})", check_body)

    def test_yuanrong_posix_accepts_aio_only_with_direct_io(self):
        check_body = self._function_body("CheckConfig", self.store_source)

        self.assertIn('config.posixIoEngine != "aio"', check_body)
        self.assertIn('config.posixIoEngine == "aio"', check_body)
        self.assertIn("!config.ioDirect", check_body)
        self.assertIn("posix_io_engine=aio requires io_direct=true", check_body)
        self.assertNotIn("return Status::Unsupported()", check_body)


if __name__ == "__main__":
    unittest.main()
