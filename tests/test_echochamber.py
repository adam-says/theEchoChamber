from __future__ import annotations

import json
import asyncio
import threading
import tempfile
import time
import unittest
from pathlib import Path
from dataclasses import replace

import h5py
import numpy as np

import echoChamber as app
from readEchoChamberData import load_recording, plot_recording, required_display_rows


class EchoChamberRecorderTests(unittest.TestCase):
    @staticmethod
    def make_block(start: int, count: int, value: float, mode: float, fired: bool = False) -> app.RecordBlock:
        pair = np.full((2, count), value, dtype=np.float64)
        single = np.full((1, count), value, dtype=np.float64)
        return app.RecordBlock(
            sample_index=start,
            ao_target_sample_index=start + count,
            ai=pair,
            raw_esn=single,
            model_esn=single + 1.0,
            pulse_threshold=single + 2.0,
            pulse_peak=single + 3.0,
            pulse_fired=np.full((1, count), float(fired)),
            safe_ao=single + 4.0,
            actual_stim=None,
            armed=mode > 1,
            mode_value=mode,
        )

    def test_compact_v5_layout_scaling_and_sparse_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = app.AppConfig(
                record_dir=Path(directory), processing_block_ms=10.0,
                recorder_batch_s=0.02, recorder_flush_s=0.02,
                amplifier_gain=(10.0, 20.0),
            )
            recorder = app.H5Recorder(
                config, {"application": "test", "configuration": app.serializable_config(config)}
            )
            path = recorder.start("compact")
            recorder.submit(self.make_block(100, config.chunk_size, 1.0, 2.0))
            recorder.submit(self.make_block(100 + config.chunk_size + 5, config.chunk_size, 2.0, 3.0, True))
            recorder.stop_recording()
            recorder.close()

            with h5py.File(path, "r") as recording:
                self.assertEqual(recording.attrs["format_tag"], "echoChamber_H5_v5")
                self.assertEqual(int(recording.attrs["schema_version"]), 5)
                self.assertNotIn("data", recording)
                ai = recording["signals/ai_raw_V"]
                self.assertEqual(ai.dtype, np.dtype("float32"))
                self.assertEqual(ai.compression, "gzip")
                self.assertTrue(ai.shuffle)
                self.assertEqual(recording["events/mode_changes"].shape[0], 2)
                self.assertEqual(recording["events/pulses"].shape[0], 1)
                self.assertEqual(recording["events/sample_discontinuities"].shape[0], 1)
                self.assertEqual(recording["diagnostics/blocks"].shape[0], 2)
                blocks = recording["diagnostics/blocks"][:]
                np.testing.assert_array_equal(blocks["armed"], [1, 1])
                np.testing.assert_array_equal(
                    blocks["ao_target_sample_index"],
                    [100 + config.chunk_size, 100 + 2 * config.chunk_size + 5],
                )
                metadata = json.loads(recording.attrs["metadata_json"])
                self.assertEqual(metadata["format_tag"], "echoChamber_H5_v5")

            loaded = load_recording(path)
            self.assertEqual(
                loaded.rows,
                ("ai_sample_index", "CA3_electrode_mV", "Cortex_electrode_mV",
                 "safe_ao_command_V", "mode"),
            )
            np.testing.assert_allclose(loaded.data[1, :config.chunk_size], 100.0)
            np.testing.assert_allclose(loaded.data[2, :config.chunk_size], 50.0)
            np.testing.assert_array_equal(
                loaded.data[4],
                np.r_[np.full(config.chunk_size, 2.0), np.full(config.chunk_size, 3.0)],
            )
            self.assertEqual(loaded.data[0, config.chunk_size], 100 + config.chunk_size + 5)

    def test_v5_controller_rows_distinguish_stimulation_and_feature_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = app.AppConfig(record_dir=Path(directory), processing_block_ms=5.0,
                                   recorder_batch_s=0.005, recorder_flush_s=0.005)
            recorder = app.H5Recorder(config, {"configuration": app.serializable_config(config)})
            path = recorder.start("feature-recovery")
            shared = {"ready": True, "calibrated": True, "envelope_mv": np.nan,
                      "baseline_mean_mv": 2.0, "baseline_std_mv": 0.5,
                      "z_score": np.nan, "target_z": 3.0, "error": 0.0,
                      "output_rate_hz": 0.0, "state": "ready"}
            stimulated = dict(shared, feature_valid=False, feature_recovery_samples=20,
                              feature_recovery_remaining_samples=0,
                              stimulated_sample_count=5, feature_invalid_sample_count=25,
                              reason="stimulus_feature_recovery")
            recovery_done = dict(shared, feature_valid=True, feature_recovery_samples=20,
                                 feature_recovery_remaining_samples=0,
                                 stimulated_sample_count=0, feature_invalid_sample_count=0,
                                 reason="")
            first = replace(self.make_block(0, config.chunk_size, 0.1, 2.0),
                            controller_id="pid", controller_diagnostics=stimulated)
            second = replace(self.make_block(config.chunk_size, config.chunk_size, 0.1, 2.0),
                             controller_id="pid", controller_diagnostics=recovery_done)
            recorder.submit(first)
            recorder.submit(second)
            recorder.stop_recording()
            recorder.close()

            with h5py.File(path, "r") as recording:
                rows = recording["diagnostics/controller_blocks"][:]
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["feature_valid"], 0)
                self.assertEqual(rows[0]["feature_eligible_sample_count"], config.chunk_size - 25)
                self.assertEqual(rows[0]["stimulated_sample_count"], 5)
                self.assertEqual(rows[0]["feature_invalid_sample_count"], 25)
                self.assertEqual(rows[0]["feature_recovery_samples"], 20)
                self.assertEqual(rows[0]["feature_recovery_reason"].decode().rstrip("\x00"),
                                 "stimulus_feature_recovery")
                self.assertEqual(rows[1]["feature_valid"], 1)
                self.assertEqual(rows[1]["feature_eligible_sample_count"], config.chunk_size)
                # Baseline remains frozen across the known feature-invalid block.
                np.testing.assert_allclose(rows["baseline_mean_mV"], [2.0, 2.0])
                np.testing.assert_allclose(rows["baseline_std_mV"], [0.5, 0.5])

    def test_blocked_hdf5_writer_does_not_block_submit_and_stop_drains_admitted_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = app.AppConfig(record_dir=Path(directory), processing_block_ms=5.0,
                                   recorder_batch_s=0.005, recorder_flush_s=0.005)
            recorder = app.H5Recorder(config, {"configuration": app.serializable_config(config)})
            path = recorder.start("writer-lock")
            writer_entered, writer_release = threading.Event(), threading.Event()
            original = recorder._append_block_records

            def blocked_append(item, offset):
                writer_entered.set()
                if not writer_release.wait(5.0):
                    raise TimeoutError("test writer was not released")
                original(item, offset)

            recorder._append_block_records = blocked_append
            recorder.submit(self.make_block(0, config.chunk_size, 1.0, 1.0))
            self.assertTrue(writer_entered.wait(2.0))

            started = time.perf_counter()
            recorder.submit(self.make_block(config.chunk_size, config.chunk_size, 2.0, 1.0))
            self.assertLess(time.perf_counter() - started, 0.05)
            self.assertEqual(recorder.items.qsize(), 1)
            self.assertLess(recorder.submit_wait_snapshot()["max_wait_us"], 50_000)

            stop_errors = []
            stop_thread = threading.Thread(target=lambda: self._stop_capture(recorder, stop_errors))
            stop_thread.start()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                with recorder._admission_lock:
                    if not recorder._accepting:
                        break
                time.sleep(0.001)
            recorder.record_action_stage(sample_index=999, generation=0, stage="late",
                                         controller="rule", reason="outside recording window")
            with recorder._action_stage_lock:
                self.assertEqual(recorder._pending_action_stages, [])
            writer_release.set()
            stop_thread.join(5.0)
            self.assertFalse(stop_thread.is_alive())
            self.assertEqual(stop_errors, [])
            recorder.close()
            with h5py.File(path, "r") as result:
                self.assertEqual(result["signals/ai_raw_V"].shape[1], config.chunk_size * 2)
                self.assertEqual(int(result.attrs["committed_samples"]), config.chunk_size * 2)

    def test_v5_plot_resolves_cortex_role_without_renaming_ec_region(self) -> None:
        import matplotlib
        matplotlib.use("Agg", force=True)
        with tempfile.TemporaryDirectory() as directory:
            config = replace(app.AppConfig(record_dir=Path(directory), processing_block_ms=5.0),
                             electrode_labels=("EC", "CA3"), ctx_index=0)
            recorder = app.H5Recorder(config, {"configuration": app.serializable_config(config)})
            path = recorder.start("ec-role")
            recorder.submit(self.make_block(0, config.chunk_size, 0.1, 1.0))
            recorder.stop_recording()
            recorder.close()

            loaded = load_recording(path)
            self.assertEqual(loaded.metadata["channel_roles"], ["Cortex", "CA3"])
            self.assertEqual(loaded.metadata["channel_regions"], ["EC", "CA3"])
            ca3, cortex, _, _ = required_display_rows(
                loaded.rows, loaded.metadata["channel_roles"], loaded.metadata["channel_labels"],
                loaded.metadata["channel_regions"],
            )
            self.assertEqual(ca3[0][0], loaded.rows.index("CA3_electrode_mV"))
            self.assertEqual(cortex[0][0], loaded.rows.index("EC_electrode_mV"))
            output = Path(directory) / "ec-role.png"
            plot_recording(loaded, save=output, show=False)
            self.assertTrue(output.is_file())

    def test_generic_cortex_region_is_not_inferred_to_be_ec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(app.AppConfig(record_dir=Path(directory), processing_block_ms=5.0),
                             electrode_labels=("Cortex", "CA3"), ctx_index=0)
            recorder = app.H5Recorder(config, {"configuration": app.serializable_config(config)})
            path = recorder.start("generic-cortex-region")
            recorder.submit(self.make_block(0, config.chunk_size, 0.1, 1.0))
            recorder.stop_recording()
            recorder.close()

            with h5py.File(path, "r") as recording:
                regions = [value.decode() if isinstance(value, bytes) else str(value)
                           for value in recording["signals/electrode_mV"].attrs["channel_regions"]]
                self.assertEqual(regions, ["Cortex", "CA3"])

    def test_fixed_replay_mapping_websocket_rejects_before_state_or_h5_change(self) -> None:
        class OneCommandSocket:
            def __init__(self, payload):
                self.payload = payload
                self.sent = []

            async def __aiter__(self):
                yield self.payload

            async def send(self, value):
                self.sent.append(value)

        with tempfile.TemporaryDirectory() as directory:
            config = replace(app.AppConfig(record_dir=Path(directory), processing_block_ms=5.0),
                             electrode_labels=("CA3", "EC"), ctx_index=1,
                             source_channel_regions=("CA3", "EC"))
            state = app.RuntimeState(
                esn_ready=False, start_paused=True, electrode_labels=config.electrode_labels,
                ctx_index=config.ctx_index, pulse_threshold_std=3.0, pulse_window_sec=10.0,
                controller="rule", controller_ready={"esn": False, "rule": True, "pid": True},
                input_mapping_locked=True,
            )
            recorder = app.H5Recorder(config, {"configuration": app.serializable_config(config)})
            socket = OneCommandSocket(json.dumps({"command": "set_cortex_channel", "channel": "ai0"}))
            asyncio.run(app.websocket_handler(socket, state, recorder, app.UiHub(1), asyncio.Event(), config))
            result = json.loads(socket.sent[0])
            self.assertFalse(result["ok"])
            self.assertIn("fixed by the calibrated mV input source", result["error"])
            self.assertEqual(state.snapshot().electrode_labels, ("CA3", "EC"))
            self.assertEqual(state.snapshot().ctx_index, 1)
            self.assertEqual(recorder.config.electrode_labels, ("CA3", "EC"))
            self.assertEqual(recorder.metadata["configuration"]["electrode_labels"], ("CA3", "EC"))

            path = recorder.start("mapping-unchanged")
            recorder.stop_recording()
            recorder.close()
            with h5py.File(path, "r") as recording:
                labels = [value.decode() if isinstance(value, bytes) else str(value)
                          for value in recording["signals/electrode_mV"].attrs["channel_labels"]]
                regions = [value.decode() if isinstance(value, bytes) else str(value)
                           for value in recording["signals/electrode_mV"].attrs["channel_regions"]]
                self.assertEqual(labels, ["CA3", "EC"])
                self.assertEqual(regions, ["CA3", "EC"])

    def test_action_stages_are_sparse_but_keep_pulse_and_drain_lifecycle(self) -> None:
        from controllers import PulseEvent

        with tempfile.TemporaryDirectory() as directory:
            config = app.AppConfig(record_dir=Path(directory), processing_block_ms=5.0,
                                   recorder_batch_s=0.005, recorder_flush_s=0.005)
            recorder = app.H5Recorder(config, {"configuration": app.serializable_config(config)})
            path = recorder.start("sparse-actions")
            state = app.RuntimeState(
                esn_ready=False, start_paused=False, electrode_labels=("CA3", "Cortex"), ctx_index=1,
                pulse_threshold_std=3.0, pulse_window_sec=10.0, controller="rule",
                controller_ready={"esn": False, "rule": True, "pid": True},
            )
            scheduler = app.SimulatedAoScheduler(config, state, recorder)
            for _ in range(20):
                scheduler.advance()
            with recorder._action_stage_lock:
                self.assertEqual(recorder._pending_action_stages, [])

            def submit_pulse(pulse_id: str, target_block: int, generation: int) -> int:
                target_onset = target_block * config.chunk_size + 80
                source_onset = target_onset - config.ao_lead_samples
                event = PulseEvent(pulse_id, source_onset, source_onset, target_onset,
                                   "rendered_for_ao", generation, 20, source_onset + 99)
                command = np.zeros((1, config.chunk_size), dtype=np.float64)
                command[0, 80:100] = 0.2
                diagnostics = {
                    "ready": True, "calibrated": True, "state": "ready",
                    "action_kind": "pulse_events", "latest_sample_index": source_onset + 99,
                    "pulse_events": [app.dataclasses.asdict(event)],
                }
                block = replace(
                    self.make_block(source_onset, config.chunk_size, 0.0, 2.0),
                    safe_ao=command, controller_id="rule", config_generation=generation,
                    controller_diagnostics=diagnostics,
                )
                recorder.submit(block)
                self.assertTrue(scheduler.submit(target_block, command[0], generation, (event,)))
                return target_onset

            first_target = scheduler._next_write_block
            submit_pulse("pulse-commit", first_target, state.snapshot().generation)
            for _ in range(12):
                scheduler.advance()
            second_target = scheduler._next_write_block + 40
            submit_pulse("pulse-cancel", second_target, state.snapshot().generation)
            state.set_mode("closed-loop")
            for _ in range(100):
                scheduler.advance()
                if not state.snapshot().transition_pending:
                    break
            self.assertFalse(state.snapshot().transition_pending)
            state.fault("recorded fault zero substitution")
            for _ in range(12):
                scheduler.advance()

            recorder.stop_recording()
            recorder.close()
            with h5py.File(path, "r") as recording:
                rows = recording["events/action_stages"][:]
                stages = [(row["request_id"].decode().rstrip("\x00"),
                           row["stage"].decode().rstrip("\x00")) for row in rows]
                self.assertIn(("pulse-commit", "rendered_for_ao"), stages)
                self.assertIn(("pulse-commit", "ao_committed"), stages)
                self.assertIn(("pulse-cancel", "rendered_for_ao"), stages)
                self.assertIn(("pulse-cancel", "cancelled"), stages)
                self.assertTrue(any(stage == "drain_acknowledged" for _, stage in stages))
                self.assertTrue(any(stage == "zero_substituted" for _, stage in stages))
                self.assertFalse(any(stage == "simulated_generated" for _, stage in stages))
                self.assertFalse(any(stage == "ao_committed" and not request_id
                                     for request_id, stage in stages))

    def test_recording_started_while_paused_excludes_dummy_nan_frames(self) -> None:
        class Core(app.ProcessingCore):
            def _publish_ui(self, *args, **kwargs):
                pass

        with tempfile.TemporaryDirectory() as directory:
            config = app.AppConfig(record_dir=Path(directory), processing_block_ms=5.0)
            state = app.RuntimeState(
                esn_ready=False, start_paused=True, electrode_labels=("CA3", "EC"), ctx_index=1,
                pulse_threshold_std=3.0, pulse_window_sec=10.0, controller="rule",
                controller_ready={"esn": False, "rule": True, "pid": True},
            )
            state.set_recording(True)
            recorder = app.H5Recorder(config, {"configuration": app.serializable_config(config)})
            path = recorder.start("paused-start")
            core = Core(config, state, app.EsnRuntime(None, config), recorder, object(), None, app.Telemetry())
            core.process(np.full((2, config.chunk_size), np.nan), sample_index=0,
                         input_is_electrode_mV=True)
            self.assertEqual(recorder.backlog, 0)

            state.set_acquiring(True)
            core.process(np.zeros((2, config.chunk_size)), sample_index=0,
                         input_is_electrode_mV=True)
            recorder.stop_recording()
            recorder.close()

            loaded = load_recording(path)
            self.assertEqual(loaded.data.shape[1], config.chunk_size)
            self.assertTrue(np.all(np.isfinite(loaded.data[1:3])))
            self.assertEqual(loaded.data[0, 0], 0.0)

    @staticmethod
    def _stop_capture(recorder, errors) -> None:
        try:
            recorder.stop_recording(timeout=5.0)
        except Exception as exc:
            errors.append(exc)


if __name__ == "__main__":
    unittest.main()
