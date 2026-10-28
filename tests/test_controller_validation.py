from __future__ import annotations

import tempfile
import unittest
import csv
from dataclasses import replace
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

import echoChamber as app
from benchmarkControllers import (
    RunMetrics, _browser_connection_url, acceptance_failures, file_sha256,
    parse_ui_packet, run_replay_case,
)
from simulation import PlantSettings, ResponsiveSyntheticPlant, run_responsive_case
from simulation import (
    TRACKING_PID_SETTINGS, TRACKING_PLANT_SETTINGS, _summarize_target_tracking,
)
import tools.ni_staged_controller_test as ni_stage
import tools.validate_controllers as validate_cli


class ResponsivePlantTests(unittest.TestCase):
    def test_effect_starts_after_generated_command_and_delay(self) -> None:
        plant = ResponsiveSyntheticPlant(PlantSettings(
            effect_delay_ms=10.0, disturbance_start_s=100.0,
            disturbance_stop_s=101.0, noise_sd_mv=0.0,
        ))
        plant.next_block(0)
        waveform = np.full(100, 0.5)
        waveform[:10] = 0.0
        command = app.GeneratedAoBlock(0, waveform, 0, "ao_committed")
        plant.observe_generated_ao(command)
        plant.next_block(100)
        self.assertEqual(plant.reduction_history_mv[-1], 0.0)
        plant.next_block(200)
        self.assertTrue(np.all(plant.last_reduction_samples_mv[:10] == 0.0))
        self.assertGreater(plant.last_reduction_samples_mv[10], 0.0)
        self.assertEqual(plant.command_effect_pulse_count, 1)

    def test_zero_substituted_output_has_no_plant_effect(self) -> None:
        plant = ResponsiveSyntheticPlant(PlantSettings(
            effect_delay_ms=0.0, disturbance_start_s=100.0,
            disturbance_stop_s=101.0, noise_sd_mv=0.0,
        ))
        plant.observe_generated_ao(app.GeneratedAoBlock(
            0, np.zeros(100), 0, "zero_substituted",
        ))
        for first in range(0, 1_000, 100):
            plant.next_block(first)
        self.assertEqual(plant.generated_nonzero_samples, 0)
        self.assertEqual(plant.command_effect_pulse_count, 0)
        self.assertEqual(plant.reduction_mv, 0.0)

    def test_tracking_metrics_are_unavailable_when_runtime_fault_leaves_a_gap(self) -> None:
        settings = TRACKING_PLANT_SETTINGS
        trace = []
        for index in range(3_200):
            time_s = index * 0.005
            if 12.5 <= time_s < 13.0:
                trace.append((time_s, 2.0, 0.5))
            else:
                trace.append((time_s, float("nan"), float("nan")))
        result = _summarize_target_tracking(trace, settings, target_z=1.0, max_rate_hz=2.0)
        self.assertFalse(result["window_complete"])
        self.assertLess(result["coverage_fraction"], 0.95)
        self.assertTrue(np.isnan(result["iae_z_seconds"]))

    def test_actual_core_pid_is_bounded_and_changes_observed_activity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            active, _ = run_responsive_case(
                "pid", stim_enabled=True, duration_s=16.0,
                recording_dir=Path(directory) / "active",
            )
            quiet, _ = run_responsive_case(
                "pid", stim_enabled=False, duration_s=16.0,
                recording_dir=Path(directory) / "quiet",
            )
        self.assertLessEqual(active["metrics"]["max_requested_rate_hz"], 2.0)
        self.assertGreater(active["generated_nonzero_ao_command_samples"], 0)
        self.assertEqual(quiet["generated_nonzero_ao_command_samples"], 0)
        self.assertEqual(quiet["plant_effect_pulse_count"], 0)
        self.assertGreater(active["plant_effect_pulse_count"], 0)
        self.assertFalse(active["plant_setpoint_reachable_by_max_reduction"])
        self.assertGreater(active["activity"]["disturbance_ca3_block_rms_mv"], 0)
        self.assertLess(active["activity"]["disturbance_ca3_block_rms_mv"],
                        quiet["activity"]["disturbance_ca3_block_rms_mv"])

    def test_over_lead_stall_faults_without_committing_or_simulating_missing_pulse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, _ = run_responsive_case(
                "pid", stim_enabled=True, duration_s=16.0,
                recording_dir=Path(directory) / "stall", inject_stall_ms=150.0,
                stall_on_first_pulse=True,
            )
        metrics = result["metrics"]
        self.assertEqual(metrics["injected_stall_count"], 1)
        self.assertGreater(metrics["requested_pulse_events"], 0)
        self.assertGreater(metrics["ao_active_demand_misses"], 0)
        self.assertIsNotNone(metrics["runtime_fault"])
        self.assertEqual(metrics["pulse_commit_ids"], [])
        self.assertEqual(result["plant_effect_pulse_count"], 0)

    def test_reachable_pid_case_tracks_target_better_than_no_stimulation(self) -> None:
        settings = TRACKING_PLANT_SETTINGS
        with tempfile.TemporaryDirectory() as directory:
            active, _ = run_responsive_case(
                "pid", stim_enabled=True, duration_s=20.0,
                plant_settings=settings, pid_settings=TRACKING_PID_SETTINGS,
                recording_dir=Path(directory) / "track-active",
            )
            quiet, _ = run_responsive_case(
                "pid", stim_enabled=False, duration_s=20.0,
                plant_settings=settings, pid_settings=TRACKING_PID_SETTINGS,
                recording_dir=Path(directory) / "track-quiet",
            )
        active_tracking = active["controller_tracking"]
        quiet_tracking = quiet["controller_tracking"]
        self.assertTrue(active["plant_setpoint_reachable_by_max_reduction"])
        self.assertTrue(active_tracking["rate_within_bounds"])
        self.assertGreater(active["metrics"]["requested_rate_blocks"], 0)
        self.assertGreater(active["plant_effect_pulse_count"], 0)
        self.assertGreater(active_tracking["rate_std_hz"], 0.0)
        self.assertLess(active_tracking["saturation_fraction"], 1.0)
        self.assertLess(active_tracking["iae_z_seconds"], quiet_tracking["iae_z_seconds"])


class ActualCoreBenchmarkTests(unittest.TestCase):
    def test_short_benchmark_uses_recorder_ui_and_shared_ao_scheduler(self) -> None:
        trace = Path(app.__file__).resolve().parent / "sample" / "test_AI0_CA3_AI1_CTX.h5"
        with tempfile.TemporaryDirectory() as directory:
            for controller in ("rule", "pid", "esn"):
                with self.subTest(controller=controller):
                    result, environment = run_replay_case(
                        trace, controller, limit_seconds=0.25, paced=True,
                        recording_dir=Path(directory) / controller,
                    )
                    self.assertEqual(result.processed_blocks, 50)
                    self.assertTrue(result.recording_active)
                    self.assertTrue(result.recording_completed)
                    self.assertEqual(result.recording_committed_samples, 5_000)
                    self.assertIn("recorder_submit", result.component_timings_ms)
                    self.assertIn("controller_rule_step" if controller == "rule" else f"controller_{controller}_step",
                                  result.component_timings_ms)
                    self.assertGreater(result.ui_packets_submitted, 0)
                    self.assertGreater(result.ui_packets_consumed, 0)
                    self.assertGreater(environment["ao_scheduler"]["committed_blocks"], 30)
                    self.assertTrue(result.ao_drain_complete)
                    self.assertFalse(result.physical_pulse_claims)
                    self.assertGreaterEqual(result.process_thread_cpu_max_ms, 0.0)
                    self.assertGreaterEqual(result.process_cpu_max_ms, 0.0)
                    recording = Path(environment["recording_file"])
                    self.assertTrue(recording.is_file())
                    with h5py.File(recording, "r") as h5:
                        self.assertEqual(int(h5.attrs["committed_samples"]), 5_000)
                        self.assertEqual(int(h5.attrs["complete"]), 1)
                        self.assertIn("signals/ao_command_V", h5)
                        self.assertIn("events/action_stages", h5)
                    self.assertEqual(file_sha256(trace),
                                     "40c7fbb780ff97979dbc6507bb3613b186d87c528942769faf6f7a8769ff1dd1")

    def test_paced_replay_refuses_to_repeat_a_finite_source_implicitly(self) -> None:
        trace = Path(app.__file__).resolve().parent / "sample" / "test_AI0_CA3_AI1_CTX.h5"
        with self.assertRaisesRegex(ValueError, "finite source ends"):
            run_replay_case(trace, "rule", limit_seconds=10_000.0, paced=True)


class NiStageDryRunTests(unittest.TestCase):
    def test_ao_waveform_dry_run_never_imports_ni_daqmx(self) -> None:
        args = ni_stage.build_parser().parse_args(["--stage", "ao-waveform"])
        ni_stage.validate_args(args)
        output = StringIO()
        real_import = __import__

        def reject_ni_import(name, *pargs, **kwargs):
            if name == "nidaqmx":
                raise AssertionError("dry-run imported NI-DAQmx")
            return real_import(name, *pargs, **kwargs)

        with patch("builtins.__import__", side_effect=reject_ni_import), redirect_stdout(output):
            self.assertEqual(ni_stage.main(["--stage", "ao-waveform"]), 0)
        document = __import__("json").loads(output.getvalue())
        self.assertFalse(document["hardware_accessed"])
        self.assertTrue(document["dry_run"])
        self.assertEqual(document["stage"], "ao-waveform")

    def test_staged_test_rejects_nonreference_timing_and_large_amplitude(self) -> None:
        for args in (
            ni_stage.build_parser().parse_args(["--stage", "ao-zero", "--sample-rate", "10000"]),
            ni_stage.build_parser().parse_args(["--stage", "ao-waveform", "--amplitude-v", "0.2"]),
        ):
            with self.assertRaises(ValueError):
                ni_stage.validate_args(args)


class UiWireFormatTests(unittest.TestCase):
    def test_browser_wire_parser_rejects_python_nan_extension(self) -> None:
        self.assertEqual(parse_ui_packet('{"type":"update","value":1}')['type'], "update")
        with self.assertRaises(ValueError):
            parse_ui_packet('{"type":"update","value":NaN}')

    def test_external_browser_url_targets_actual_ui_websocket_port(self) -> None:
        self.assertEqual(_browser_connection_url("http://127.0.0.1:8000", 43123),
                         "http://127.0.0.1:8000/index.html?wsPort=43123")
        self.assertEqual(_browser_connection_url("http://127.0.0.1:8000/index.html", 43123),
                         "http://127.0.0.1:8000/index.html?wsPort=43123")


class BenchmarkGateTests(unittest.TestCase):
    def test_gate_fails_known_invalid_latency_summary(self) -> None:
        invalid = RunMetrics(
            case_id="known-invalid", source_name="fixture", controller="rule",
            sample_rate_hz=20_000, block_samples=100, block_ms=5.0,
            ao_lead_ms=100.0, deadline_ms=5.0, processed_blocks=10,
            process_p99_ms=2.75, process_max_ms=5.1, deadline_exceeded_blocks=1,
        )
        failures = acceptance_failures(invalid)
        self.assertIn("process p99 is not below 2.5 ms", failures)
        self.assertIn("maximum process time reached or exceeded the block deadline", failures)
        self.assertIn("one or more blocks exceeded the processing deadline", failures)

    def test_matrix_cli_writes_failure_artifacts_then_returns_nonzero(self) -> None:
        invalid = RunMetrics(
            case_id="gate-fixture", source_name="fixture", controller="rule",
            sample_rate_hz=20_000, block_samples=100, block_ms=5.0,
            ao_lead_ms=100.0, deadline_ms=5.0, processed_blocks=1,
            process_p99_ms=3.0, process_max_ms=5.2, deadline_exceeded_blocks=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(validate_cli, "run_replay_case", return_value=(invalid, {})):
                code = validate_cli.main([
                    "--matrix", "--trace", "synthetic-fixture.h5", "--controller", "rule",
                    "--duration-seconds", "0.25", "--output-dir", directory,
                ])
            json_path = next(Path(directory).glob("controller_matrix_*.json"))
            csv_path = json_path.with_suffix(".csv")
            result = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(code, 1)
        self.assertFalse(result["acceptance"][0]["passed"])
        self.assertTrue(any("process p99" in item for item in result["acceptance"][0]["failures"]))
        self.assertEqual(rows[0]["acceptance_passed"], "False")


if __name__ == "__main__":
    unittest.main()
