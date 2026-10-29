from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import h5py
import numpy as np

import echoChamber as app


class FaultEventRecordingTests(unittest.TestCase):
    def make_recorder(self, directory: str, *, batch_s: float = 0.02) -> app.H5Recorder:
        config = replace(
            app.AppConfig(record_dir=Path(directory), processing_block_ms=5.0,
                          recorder_batch_s=batch_s, recorder_flush_s=batch_s),
        )
        recorder = app.H5Recorder(config, {"configuration": app.serializable_config(config)})
        recorder.start("fault-events")
        return recorder

    @staticmethod
    def record_block(config: app.AppConfig, sample_index: int,
                     pulse_ids: tuple[str, ...] = ()) -> app.RecordBlock:
        count = config.chunk_size
        events = [{
            "pulse_id": pulse_id,
            "status": "rendered_for_ao",
            "generation": 1,
            "source_onset_sample_index": sample_index + 10,
            "target_onset_sample_index": sample_index + 10 + config.ao_lead_samples,
        } for pulse_id in pulse_ids]
        return app.RecordBlock(
            sample_index=sample_index,
            ao_target_sample_index=sample_index + config.ao_lead_samples,
            ai=np.zeros((2, count)),
            raw_esn=np.zeros((1, count)),
            model_esn=np.zeros((1, count)),
            pulse_threshold=np.zeros((1, count)),
            pulse_peak=np.zeros((1, count)),
            pulse_fired=np.zeros((1, count)),
            safe_ao=np.zeros((1, count)),
            actual_stim=None,
            armed=False,
            mode_value=0.0,
            ai_electrode_mV=np.zeros((2, count)),
            controller_id="rule",
            config_generation=1,
            controller_diagnostics={
                "ready": True, "calibrated": True, "state": "ready",
                "latest_sample_index": sample_index + count - 1,
                "pulse_events": events,
            },
        )

    @staticmethod
    def fault_zero(recorder: app.H5Recorder, start: int, *, count: int = 100,
                   generation: int = 4, controller: str = "rule",
                   reason: str = "faulted AO refill committed as safe zero") -> None:
        app.record_sparse_ao_outcome(
            recorder, sample_index=start, generation=generation,
            stage="zero_substituted", controller=controller, waveform=None,
            reason=reason, confirmed_zero_sample_count=count,
        )

    def test_event_only_flush_coalesces_without_creating_signal_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = self.make_recorder(directory, batch_s=0.02)
            path = recorder._data_path
            assert path is not None
            flushed_by_writer = threading.Event()
            original_flush = recorder._flush_action_stages

            def observe_flush():
                original_flush()
                if threading.current_thread() is recorder._thread:
                    flushed_by_writer.set()

            recorder._flush_action_stages = observe_flush
            for start in range(0, 2_000, 100):
                self.fault_zero(recorder, start)
            self.assertTrue(flushed_by_writer.wait(2.0), "event-only queue timeout did not flush AO rows")
            recorder.stop_recording(timeout=3.0)
            recorder.close()

            with h5py.File(path, "r") as recording:
                rows = recording["events/action_stages"][:]
                self.assertEqual(len(rows), 1)
                self.assertEqual(int(rows[0]["sample_index"]), 0)
                self.assertEqual(int(rows[0]["range_end_sample_index"]), 2_000)
                self.assertEqual(int(rows[0]["generation"]), 4)
                self.assertEqual(rows[0]["controller"].decode().rstrip("\x00"), "rule")
                self.assertEqual(
                    rows[0]["reason"].decode().rstrip("\x00"),
                    "faulted AO refill committed as safe zero",
                )
                self.assertEqual(int(recording.attrs["committed_samples"]), 0)
                self.assertEqual(int(recording.attrs["complete"]), 1)
                self.assertEqual(recording["signals/ai_raw_V"].shape, (2, 0))
                self.assertEqual(recording["signals/ao_command_V"].shape, (1, 0))

    def test_simulated_scheduler_reports_only_committed_fault_zero_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                app.AppConfig(record_dir=Path(directory), processing_block_ms=5.0,
                              recorder_batch_s=0.02, recorder_flush_s=0.02),
            )
            recorder = app.H5Recorder(config, {"configuration": app.serializable_config(config)})
            path = recorder.start("simulated-fault-zero")
            state = app.RuntimeState(
                esn_ready=False, start_paused=False, electrode_labels=("CA3", "EC"), ctx_index=1,
                pulse_threshold_std=3.0, pulse_window_sec=10.0, controller="rule",
                controller_ready={"esn": False, "rule": True, "pid": True},
            )
            state.fault("test latched safe fault")
            scheduler = app.SimulatedAoScheduler(config, state, recorder)
            flushed_by_writer = threading.Event()
            original_flush = recorder._flush_action_stages

            def observe_flush():
                original_flush()
                if threading.current_thread() is recorder._thread:
                    flushed_by_writer.set()

            recorder._flush_action_stages = observe_flush
            try:
                generated = [scheduler.advance() for _ in range(config.ao_lead_chunks + 5)]
                self.assertTrue(all(not np.any(block.waveform_v) for block in generated))
                self.assertEqual(scheduler.stats_snapshot()["generated_sample_count"],
                                 (config.ao_lead_chunks + 5) * config.chunk_size)
                self.assertTrue(flushed_by_writer.wait(2.0))
                recorder.stop_recording(timeout=3.0)
            finally:
                recorder.close()

            with h5py.File(path, "r") as recording:
                rows = recording["events/action_stages"][:]
                zero_rows = [row for row in rows
                             if row["stage"].decode().rstrip("\x00") == "zero_substituted"]
                self.assertTrue(zero_rows)
                extents = [int(row["range_end_sample_index"]) - int(row["sample_index"])
                           for row in zero_rows]
                self.assertTrue(all(extent > 0 and extent % config.chunk_size == 0
                                    for extent in extents))
                self.assertTrue(all(int(row["sample_index"]) >= config.ao_lead_samples
                                    for row in zero_rows))
                self.assertTrue(all(int(row["generation"]) == state.snapshot().generation
                                    for row in zero_rows))
                self.assertEqual(int(recording.attrs["committed_samples"]), 0)
                self.assertEqual(recording["signals/ai_raw_V"].shape, (2, 0))
                self.assertEqual(recording["signals/ao_command_V"].shape, (1, 0))

    def test_ranges_preserve_order_and_split_on_events_gaps_and_metadata_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = self.make_recorder(directory, batch_s=0.5)
            path = recorder._data_path
            assert path is not None

            self.fault_zero(recorder, 100)
            self.fault_zero(recorder, 200)
            recorder.record_action_stage(
                sample_index=900, generation=4, stage="drain_acknowledged", controller="rule",
            )
            self.fault_zero(recorder, 400)
            # Discontinuous AO indices cannot be merged across the missing span.
            self.fault_zero(recorder, 700)
            event = SimpleNamespace(
                pulse_id="pulse-x", generation=5, source_onset_sample_index=750,
                latest_sample_index=799, target_onset_sample_index=2_750,
            )
            app.record_sparse_ao_outcome(
                recorder, sample_index=2_700, generation=5, stage="cancelled",
                controller="pid", events=(event,), reason="reserved pulse cancelled",
            )
            self.fault_zero(recorder, 800)
            self.fault_zero(recorder, 900, generation=5, controller="pid")
            self.fault_zero(recorder, 1_000, generation=5, controller="pid",
                            reason="faulted refill was written as safe zero")
            self.fault_zero(recorder, 1_100, generation=5, controller="pid",
                            reason="faulted AO refill committed as safe zero")

            recorder.stop_recording(timeout=3.0)
            recorder.close()
            with h5py.File(path, "r") as recording:
                rows = recording["events/action_stages"][:]
                self.assertEqual(
                    [row["stage"].decode().rstrip("\x00") for row in rows],
                    ["zero_substituted", "drain_acknowledged", "zero_substituted",
                     "zero_substituted", "cancelled", "zero_substituted", "zero_substituted",
                     "zero_substituted", "zero_substituted"],
                )
                np.testing.assert_array_equal(
                    rows["sample_index"], [100, 900, 400, 700, 2_750, 800, 900, 1_000, 1_100],
                )
                np.testing.assert_array_equal(
                    rows["range_end_sample_index"],
                    [300, -1, 500, 800, -1, 900, 1_000, 1_100, 1_200],
                )
                self.assertEqual(rows[4]["request_id"].decode().rstrip("\x00"), "pulse-x")
                self.assertEqual(int(rows[5]["generation"]), 4)
                self.assertEqual(int(rows[6]["generation"]), 5)

    def test_unconfirmed_or_partial_zero_extent_never_becomes_a_full_chunk_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = self.make_recorder(directory, batch_s=0.5)
            path = recorder._data_path
            assert path is not None

            # A missing waveform without an explicit accepted count is not proof.
            app.record_sparse_ao_outcome(
                recorder, sample_index=100, generation=1, stage="zero_substituted",
                controller="rule", waveform=None,
                reason="faulted AO refill committed as safe zero",
            )
            # A known partial accepted prefix carries only that exact extent.
            self.fault_zero(recorder, 200, count=20)
            # A caller cannot turn an ambiguous write reason into a full range.
            self.fault_zero(recorder, 400, count=100, reason="NI write remainder state unknown")
            recorder.stop_recording(timeout=3.0)
            recorder.close()

            with h5py.File(path, "r") as recording:
                rows = recording["events/action_stages"][:]
                self.assertEqual(len(rows), 3)
                np.testing.assert_array_equal(rows["sample_index"], [100, 200, 400])
                np.testing.assert_array_equal(rows["range_end_sample_index"], [-1, 220, -1])
                self.assertEqual(int(recording.attrs["committed_samples"]), 0)

    def test_overflow_latches_first_error_without_waiting_for_hdf_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = self.make_recorder(directory, batch_s=0.5)
            path = recorder._data_path
            assert path is not None
            recorder.MAX_PENDING_ACTION_ROWS = 0
            returned = threading.Event()

            def submit_overflowing_event() -> None:
                recorder.record_action_stage(
                    sample_index=10, generation=1, stage="drain_acknowledged",
                    controller="rule",
                )
                returned.set()

            with recorder._lock:
                caller = threading.Thread(target=submit_overflowing_event)
                caller.start()
                self.assertTrue(returned.wait(0.5), "AO event admission waited on the HDF5 file lock")
            caller.join(timeout=1.0)
            first_error = recorder.error
            self.assertIn("pending action-stage limit", first_error or "")
            self.assertFalse(recorder._accepting)

            with self.assertRaisesRegex(RuntimeError, "pending action-stage limit"):
                recorder.close()
            with h5py.File(path, "r") as recording:
                self.assertEqual(int(recording.attrs["complete"]), 0)
                self.assertEqual(int(recording.attrs["committed_samples"]), 0)
                self.assertEqual(recording["signals/ai_raw_V"].shape, (2, 0))

    def test_writer_pulse_row_overflow_leaves_second_batch_uncommitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                app.AppConfig(record_dir=Path(directory), processing_block_ms=5.0,
                              recorder_batch_s=0.015, recorder_flush_s=0.015),
            )
            recorder = app.H5Recorder(config, {"configuration": app.serializable_config(config)})
            path = recorder.start("writer-pulse-overflow")
            recorder.MAX_PENDING_ACTION_ROWS = 1
            first_batch_prepared = threading.Event()
            original_batch_prepare = recorder._append_block_record_batch

            def observe_first_batch(batch, start):
                result = original_batch_prepare(batch, start)
                if start == 0:
                    first_batch_prepared.set()
                return result

            recorder._append_block_record_batch = observe_first_batch
            first = [
                self.record_block(config, 0),
                self.record_block(config, 100, ("first-batch-pulse",)),
                self.record_block(config, 200),
            ]
            for block in first:
                recorder.submit(block)
            self.assertTrue(first_batch_prepared.wait(2.0))
            deadline = time.monotonic() + 2.0
            while recorder._committed_samples != 300 and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertEqual(recorder._committed_samples, 300)

            recorder.submit(self.record_block(config, 300, ("second-batch-pulse-a",)))
            recorder.submit(self.record_block(config, 400, ("second-batch-pulse-b",)))
            deadline = time.monotonic() + 2.0
            while recorder.error is None and time.monotonic() < deadline:
                time.sleep(0.001)
            first_error = recorder.error
            self.assertIn("pending action-stage limit", first_error or "")
            self.assertEqual(recorder._committed_samples, 300)

            with self.assertRaisesRegex(RuntimeError, "pending action-stage limit"):
                recorder.close()
            self.assertFalse(recorder._thread.is_alive())
            self.assertEqual(recorder.items.unfinished_tasks, 0)
            self.assertEqual(recorder.error, first_error)
            with h5py.File(path, "r") as recording:
                self.assertEqual(int(recording.attrs["complete"]), 0)
                self.assertEqual(int(recording.attrs["committed_samples"]), 300)
                self.assertEqual(recording["signals/ai_raw_V"].shape[1], 500)

    def test_ao_overflow_before_commit_decision_preserves_previous_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                app.AppConfig(record_dir=Path(directory), processing_block_ms=5.0,
                              recorder_batch_s=0.015, recorder_flush_s=0.015),
            )
            recorder = app.H5Recorder(config, {"configuration": app.serializable_config(config)})
            path = recorder.start("ao-overflow-before-commit")
            recorder.MAX_PENDING_ACTION_ROWS = 1
            second_flush_entered, release_second_flush = threading.Event(), threading.Event()
            flush_number = 0
            original_flush = recorder._flush_action_stages

            def pause_second_flush():
                nonlocal flush_number
                original_flush()
                if threading.current_thread() is recorder._thread:
                    flush_number += 1
                    if flush_number == 2:
                        second_flush_entered.set()
                        if not release_second_flush.wait(2.0):
                            raise TimeoutError("test did not release second-batch metadata flush")

            recorder._flush_action_stages = pause_second_flush
            for start in (0, 100, 200, 300, 400, 500):
                recorder.submit(self.record_block(config, start))
            self.assertTrue(second_flush_entered.wait(2.0))
            self.assertEqual(recorder._committed_samples, 300)

            # The writer is paused before its boundary decision while holding
            # the file lock. AO overflow must latch independently of that lock.
            recorder.MAX_PENDING_ACTION_ROWS = 0
            overflow_returned = threading.Event()

            def overflow_from_ao():
                recorder.record_action_stage(
                    sample_index=700, generation=1, stage="drain_acknowledged",
                    controller="rule",
                )
                overflow_returned.set()

            ao_caller = threading.Thread(target=overflow_from_ao)
            ao_caller.start()
            self.assertTrue(overflow_returned.wait(0.5), "AO overflow waited on the HDF5 writer")
            ao_caller.join(timeout=1.0)
            first_error = recorder.error
            self.assertIn("pending action-stage limit", first_error or "")
            release_second_flush.set()

            deadline = time.monotonic() + 2.0
            while recorder.items.unfinished_tasks and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertEqual(recorder._committed_samples, 300)
            with self.assertRaisesRegex(RuntimeError, "pending action-stage limit"):
                recorder.close()
            self.assertEqual(recorder.error, first_error)
            self.assertFalse(recorder._thread.is_alive())
            self.assertEqual(recorder.items.unfinished_tasks, 0)
            with h5py.File(path, "r") as recording:
                self.assertEqual(int(recording.attrs["complete"]), 0)
                self.assertEqual(int(recording.attrs["committed_samples"]), 300)
                self.assertEqual(recording["signals/ai_raw_V"].shape[1], 600)

    def test_periodic_event_flush_serializes_with_final_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = self.make_recorder(directory, batch_s=0.02)
            path = recorder._data_path
            assert path is not None
            writer_entered, release_writer = threading.Event(), threading.Event()
            close_finished = threading.Event()
            original_flush = recorder._flush_action_stages

            def blocked_flush():
                if threading.current_thread() is recorder._thread and not writer_entered.is_set():
                    writer_entered.set()
                    if not release_writer.wait(2.0):
                        raise TimeoutError("test did not release event-only writer flush")
                original_flush()

            recorder._flush_action_stages = blocked_flush
            self.fault_zero(recorder, 1_000)
            self.assertTrue(writer_entered.wait(2.0))
            close_errors: list[BaseException] = []

            def close_recorder() -> None:
                try:
                    recorder.close()
                except BaseException as exc:
                    close_errors.append(exc)
                finally:
                    close_finished.set()

            closer = threading.Thread(target=close_recorder)
            closer.start()
            time.sleep(0.02)
            release_writer.set()
            self.assertTrue(close_finished.wait(3.0))
            closer.join(timeout=1.0)
            self.assertEqual(close_errors, [])
            self.assertFalse(recorder._thread.is_alive())

            with h5py.File(path, "r") as recording:
                rows = recording["events/action_stages"][:]
                self.assertEqual(len(rows), 1)
                self.assertEqual(int(rows[0]["sample_index"]), 1_000)
                self.assertEqual(int(rows[0]["range_end_sample_index"]), 1_100)
                self.assertEqual(int(recording.attrs["complete"]), 1)
                self.assertEqual(int(recording.attrs["committed_samples"]), 0)


if __name__ == "__main__":
    unittest.main()
