from __future__ import annotations

import tempfile
import queue
import threading
import time
import unittest
from pathlib import Path
from dataclasses import replace

import h5py
import numpy as np

import echoChamber as app


def _blocks(config: app.AppConfig) -> list[app.RecordBlock]:
    starts = (100, 200, 300, 415, 515)
    modes = (0.0, 0.0, 1.0, 1.0, 0.0)
    controllers = ("rule", "rule", "pid", "pid", "esn")
    generations = (1, 1, 2, 2, 3)
    blocks = []
    for index, (start, mode, controller, generation) in enumerate(
            zip(starts, modes, controllers, generations, strict=True)):
        count = config.chunk_size
        ai = np.vstack((
            np.full(count, 0.01 * (index + 1)),
            np.full(count, -0.02 * (index + 1)),
        ))
        event = None
        if index in (1, 4):
            event = {
                "pulse_id": f"p{index}", "status": "rendered_for_ao",
                "generation": generation, "source_onset_sample_index": start + 90,
                "target_onset_sample_index": start + 90 + config.ao_lead_samples,
            }
        diagnostics = {
            "ready": True, "calibrated": True, "state": "ready",
            "envelope_mv": float(index + 0.25), "baseline_mean_mv": 0.5,
            "baseline_std_mv": 0.1, "z_score": float(index), "target_z": 2.0,
            "error": float(2 - index), "output_rate_hz": 0.25 * index,
            "latest_sample_index": start + count - 1,
            "feature_valid": index != 2, "feature_recovery_samples": 20,
            "feature_recovery_remaining_samples": 5 if index == 2 else 0,
            "stimulated_sample_count": 2 if index == 2 else 0,
            "feature_invalid_sample_count": 7 if index == 2 else 0,
            "reason": "stimulus_feature_recovery" if index == 2 else "",
            "pulse_events": [event] if event else [],
        }
        blocks.append(app.RecordBlock(
            sample_index=start,
            ao_target_sample_index=start + config.ao_lead_samples,
            ai=ai,
            raw_esn=np.full((1, count), 0.1 + index, dtype=np.float64),
            model_esn=np.full((1, count), 1.1 + index, dtype=np.float64),
            pulse_threshold=np.full((1, count), 2.1 + index, dtype=np.float64),
            pulse_peak=np.full((1, count), 3.1 + index, dtype=np.float64),
            pulse_fired=np.full((1, count), float(index in (1, 4)), dtype=np.float64),
            safe_ao=np.full((1, count), 0.01 * index, dtype=np.float64),
            actual_stim=None if index == 2 else np.full((1, count), 0.05 * index),
            armed=mode > 0.0,
            mode_value=mode,
            ai_electrode_mV=ai * (1000.0 / np.asarray(config.amplifier_gain)[:, None]),
            controller_id=controller,
            config_generation=generation,
            controller_diagnostics=diagnostics,
        ))
    return blocks


class H5RecorderBatchingTests(unittest.TestCase):
    def test_batch_and_partial_final_batch_preserve_exact_rows_and_event_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                app.AppConfig(record_dir=Path(directory), processing_block_ms=5.0,
                              recorder_batch_s=0.015, recorder_flush_s=0.015,
                              amplifier_gain=(10.0, 20.0), actual_stim_monitor_channel="ai2"),
            )
            recorder = app.H5Recorder(config, {"configuration": app.serializable_config(config)})
            path = recorder.start("batch-exact-rows")
            original_append = recorder._append_records
            append_counts: dict[str, list[int]] = {}
            batch_number = 0
            gap_batch_numbers: list[int] = []
            first_batch_metadata_started = threading.Event()
            release_first_batch = threading.Event()

            def counted_append(dataset, rows):
                nonlocal batch_number
                append_counts.setdefault(dataset.name, []).append(len(rows))
                if dataset.name == "/diagnostics/blocks":
                    batch_number += 1
                    if not first_batch_metadata_started.is_set():
                        first_batch_metadata_started.set()
                        if not release_first_batch.wait(2.0):
                            raise TimeoutError("test did not release the first recorder batch")
                if dataset.name == "/events/sample_discontinuities":
                    gap_batch_numbers.append(batch_number + 1)
                original_append(dataset, rows)

            recorder._append_records = counted_append
            recorder.record_action_stage(
                sample_index=7_777, generation=9, stage="reserved", controller="pid",
                request_id="external-first", source_sample_index=700,
                latest_sample_index=799, target_sample_index=2_700,
                range_end_sample_index=2_800,
            )
            values = _blocks(config)
            for block in values[:3]:
                recorder.submit(block)
            self.assertTrue(first_batch_metadata_started.wait(2.0))
            for block in values[3:]:
                recorder.submit(block)
            release_first_batch.set()
            recorder.stop_recording(timeout=5.0)
            recorder.close()

            # The recorder batches the first three blocks, then writes the
            # final two as a partial batch. Every metadata dataset gets one
            # append per batch in which it has rows.
            self.assertEqual(append_counts["/diagnostics/blocks"], [3, 2])
            self.assertEqual(append_counts["/diagnostics/controller_blocks"], [3, 2])
            self.assertEqual(append_counts["/events/mode_changes"], [2, 1])
            self.assertEqual(append_counts["/events/sample_discontinuities"], [1])
            self.assertEqual(gap_batch_numbers, [2])
            self.assertEqual(append_counts["/events/pulses"], [1, 1])

            with h5py.File(path, "r") as file:
                self.assertEqual(int(file.attrs["first_ai_sample_index"]), 100)
                self.assertEqual(int(file.attrs["committed_samples"]), 5 * config.chunk_size)
                self.assertEqual(int(file.attrs["complete"]), 1)
                self.assertEqual(int(file["signals/ai_raw_V"].shape[1]), 500)
                self.assertEqual(int(file["signals/electrode_mV"].shape[1]), 500)
                self.assertEqual(int(file["signals/ao_command_V"].shape[1]), 500)
                self.assertEqual(int(file["signals/actual_stim_monitor_V"].shape[1]), 500)

                expected_ai = np.hstack([block.ai for block in values]).astype(np.float32)
                expected_electrode = np.hstack([block.ai_electrode_mV for block in values]).astype(np.float32)
                expected_ao = np.hstack([block.safe_ao for block in values]).astype(np.float32)
                expected_monitor = np.hstack([
                    block.actual_stim if block.actual_stim is not None
                    else np.full((1, config.chunk_size), np.nan)
                    for block in values
                ]).astype(np.float32)
                np.testing.assert_array_equal(file["signals/ai_raw_V"][:], expected_ai)
                np.testing.assert_array_equal(file["signals/electrode_mV"][:], expected_electrode)
                np.testing.assert_array_equal(file["signals/ao_command_V"][:], expected_ao)
                np.testing.assert_array_equal(file["signals/actual_stim_monitor_V"][:], expected_monitor)

                diagnostics = file["diagnostics/blocks"][:]
                np.testing.assert_array_equal(diagnostics["sample_offset"], [0, 100, 200, 300, 400])
                np.testing.assert_array_equal(diagnostics["ai_sample_index"], [100, 200, 300, 415, 515])
                np.testing.assert_array_equal(
                    diagnostics["ao_target_sample_index"],
                    [start + config.ao_lead_samples for start in (100, 200, 300, 415, 515)],
                )
                np.testing.assert_array_equal(diagnostics["sample_count"], [100] * 5)
                np.testing.assert_allclose(diagnostics["raw_esn_mean_V"], [0.1, 1.1, 2.1, 3.1, 4.1])

                controllers = file["diagnostics/controller_blocks"][:]
                np.testing.assert_array_equal(controllers["sample_offset"], [0, 100, 200, 300, 400])
                np.testing.assert_array_equal(controllers["config_generation"], [1, 1, 2, 2, 3])
                self.assertEqual([row.decode().rstrip("\x00") for row in controllers["controller_id"]],
                                 ["rule", "rule", "pid", "pid", "esn"])
                self.assertEqual(int(controllers[2]["feature_invalid_sample_count"]), 7)

                modes = file["events/mode_changes"][:]
                np.testing.assert_array_equal(modes["sample_offset"], [0, 200, 400])
                np.testing.assert_array_equal(modes["mode"], [0, 1, 0])
                gaps = file["events/sample_discontinuities"][:]
                self.assertEqual(len(gaps), 1)
                self.assertEqual(tuple(gaps[0]), (300, 400, 415))
                pulses = file["events/pulses"][:]
                np.testing.assert_array_equal(pulses["sample_offset"], [100, 400])
                np.testing.assert_array_equal(pulses["ai_sample_index"], [200, 515])
                action_rows = file["events/action_stages"][:]
                self.assertEqual([row.decode().rstrip("\x00") for row in action_rows["request_id"]],
                                 ["external-first", "p1", "p4"])
                np.testing.assert_array_equal(action_rows["sample_index"], [7_777, 2_290, 2_605])

    def test_metadata_and_one_shot_action_flush_failures_leave_incomplete_file_and_stop_owner(self) -> None:
        for failure_point in ("controller_metadata", "action_flush"):
            with self.subTest(failure_point=failure_point), tempfile.TemporaryDirectory() as directory:
                config = app.AppConfig(record_dir=Path(directory), processing_block_ms=5.0,
                                       recorder_batch_s=0.005, recorder_flush_s=0.005)
                recorder = app.H5Recorder(config, {"configuration": app.serializable_config(config)})
                path = recorder.start(f"failure-{failure_point}")
                recorder.record_action_stage(
                    sample_index=100, generation=1, stage="reserved", controller="pid",
                    request_id="failure-test", target_sample_index=100,
                )
                failure_seen = threading.Event()
                if failure_point == "controller_metadata":
                    original_append = recorder._append_records

                    def fail_metadata(dataset, rows):
                        if dataset.name == "/diagnostics/controller_blocks":
                            failure_seen.set()
                            raise OSError("injected controller metadata write failure")
                        original_append(dataset, rows)

                    recorder._append_records = fail_metadata
                else:
                    original_flush = recorder._flush_action_stages
                    flush_calls = 0

                    def fail_action_flush_once():
                        nonlocal flush_calls
                        flush_calls += 1
                        if flush_calls == 1:
                            failure_seen.set()
                            raise OSError("injected action-stage flush failure")
                        original_flush()

                    recorder._flush_action_stages = fail_action_flush_once

                block = _blocks(config)[0]
                recorder.submit(block)
                recorder.submit(replace(block, sample_index=200, ao_target_sample_index=200 + config.ao_lead_samples))
                self.assertTrue(failure_seen.wait(2.0))
                deadline = time.monotonic() + 2.0
                while recorder.error is None and time.monotonic() < deadline:
                    time.sleep(0.001)
                self.assertIsNotNone(recorder.error)

                with self.assertRaisesRegex(RuntimeError, "recording writer failed"):
                    recorder.close()
                self.assertFalse(recorder._thread.is_alive())
                self.assertEqual(recorder.items.unfinished_tasks, 0)
                self.assertIn("injected", recorder.error)
                if failure_point == "action_flush":
                    self.assertEqual(flush_calls, 1)
                with h5py.File(path, "r") as file:
                    self.assertEqual(int(file.attrs["complete"]), 0)
                    self.assertEqual(int(file.attrs["committed_samples"]), 0)

    def test_second_batch_write_failure_preserves_prior_committed_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = app.AppConfig(record_dir=Path(directory), processing_block_ms=5.0,
                                   recorder_batch_s=0.015, recorder_flush_s=0.015)
            recorder = app.H5Recorder(config, {"configuration": app.serializable_config(config)})
            path = recorder.start("second-batch-failure")
            blocks = _blocks(config)
            first_batch_prepared = threading.Event()
            original_batch = recorder._append_block_record_batch

            def observe_first_batch(batch, start):
                result = original_batch(batch, start)
                if start == 0:
                    first_batch_prepared.set()
                return result

            recorder._append_block_record_batch = observe_first_batch
            original_append = recorder._append_records

            def fail_second_batch_metadata(dataset, rows):
                if (dataset.name == "/diagnostics/controller_blocks"
                        and int(rows["sample_offset"][0]) >= 300):
                    raise OSError("injected second-batch metadata failure")
                original_append(dataset, rows)

            recorder._append_records = fail_second_batch_metadata
            for block in blocks[:3]:
                recorder.submit(block)
            self.assertTrue(first_batch_prepared.wait(2.0))
            with recorder._lock:
                self.assertEqual(recorder._committed_samples, 300)
            for block in blocks[3:]:
                recorder.submit(block)

            with self.assertRaisesRegex(RuntimeError, "injected second-batch metadata failure"):
                recorder.close()
            self.assertFalse(recorder._thread.is_alive())
            self.assertEqual(recorder.items.unfinished_tasks, 0)
            with h5py.File(path, "r") as file:
                self.assertEqual(int(file.attrs["complete"]), 0)
                self.assertEqual(int(file.attrs["committed_samples"]), 300)
                expected_prefix = np.hstack([block.ai for block in blocks[:3]]).astype(np.float32)
                np.testing.assert_array_equal(file["signals/ai_raw_V"][:, :300], expected_prefix)
                # The failed second batch may have extended signal datasets,
                # but readers must stop at the last fully committed boundary.
                self.assertEqual(int(file["signals/ai_raw_V"].shape[1]), 500)

    def test_close_preserves_stop_error_across_full_queue_join_timeout_and_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = app.AppConfig(record_dir=Path(directory), processing_block_ms=5.0,
                                   logger_queue_blocks=1, recorder_batch_s=0.005,
                                   recorder_flush_s=0.005)
            recorder = app.H5Recorder(config, {"configuration": app.serializable_config(config)})
            path = recorder.start("close-cleanup-errors")
            first_entered, first_release = threading.Event(), threading.Event()
            second_entered, second_release = threading.Event(), threading.Event()
            original_prepare = recorder._append_block_records

            def block_owner(item, offset):
                if offset == 0:
                    first_entered.set()
                    if not first_release.wait(3.0):
                        raise TimeoutError("first writer block was not released")
                elif offset == config.chunk_size:
                    second_entered.set()
                    if not second_release.wait(3.0):
                        raise TimeoutError("second writer block was not released")
                return original_prepare(item, offset)

            recorder._append_block_records = block_owner
            recorder.submit(_blocks(config)[0])
            self.assertTrue(first_entered.wait(2.0))
            recorder.submit(replace(_blocks(config)[1], sample_index=200))

            original_stop = recorder.stop_recording

            def short_stop(timeout=10.0):
                return original_stop(timeout=0.01)

            recorder.stop_recording = short_stop
            original_put = recorder.items.put
            forced_full = threading.Event()

            def fail_first_sentinel_put(item, block=True, timeout=None):
                if item is None and not forced_full.is_set():
                    forced_full.set()
                    first_release.set()
                    raise queue.Full
                return original_put(item, block=block, timeout=timeout)

            recorder.items.put = fail_first_sentinel_put
            original_join = recorder._thread.join

            def immediate_join(timeout=None):
                # The first close attempt must report a timeout while the real
                # owner is still blocked on the second record block.
                return None

            recorder._thread.join = immediate_join
            with self.assertRaises(TimeoutError) as raised:
                recorder.close()
            self.assertTrue(forced_full.is_set())
            self.assertTrue(second_entered.wait(2.0))
            self.assertTrue(any("writer did not stop" in note for note in raised.exception.__notes__))
            self.assertTrue(recorder._thread.is_alive())
            self.assertTrue(recorder._shutdown_sentinel_sent)

            unfinished_before_retry = recorder.items.unfinished_tasks
            with self.assertRaises(TimeoutError):
                recorder.close()
            self.assertEqual(recorder.items.unfinished_tasks, unfinished_before_retry)
            self.assertEqual(unfinished_before_retry, 2)
            self.assertTrue(recorder._thread.is_alive())

            second_release.set()
            original_join(timeout=3.0)
            self.assertFalse(recorder._thread.is_alive())
            self.assertEqual(recorder.items.unfinished_tasks, 0)

            # A close after the owner exits must not enqueue another sentinel.
            recorder._thread.join = original_join
            recorder.close()
            self.assertEqual(recorder.items.unfinished_tasks, 0)
            recorder.close()
            self.assertEqual(recorder.items.unfinished_tasks, 0)
            with h5py.File(path, "r") as file:
                self.assertEqual(int(file.attrs["complete"]), 1)
                self.assertEqual(int(file.attrs["committed_samples"]), 200)


if __name__ == "__main__":
    unittest.main()
