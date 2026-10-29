from __future__ import annotations

import tempfile
import unittest
import csv
import time
import asyncio
import gc
import threading
from dataclasses import asdict, replace
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

import echoChamber as app
import benchmarkControllers as benchmark
from benchmarkControllers import (
    RunMetrics, _BrowserPresenceTracker, _CountingUiHub, _Harness, _browser_connection_url,
    _record_pulse_event_metrics, acceptance_failures, evaluate_case_acceptance,
    evaluate_collection_acceptance, file_sha256,
    json_safe, parse_ui_packet, run_replay_case,
)
from simulation import (PlantSettings, ResponsiveSyntheticPlant, _summarize_capacity_reference,
                        run_feedback_comparison, run_responsive_case)
from simulation import (
    TRACKING_PLANT_SETTINGS, _summarize_target_tracking,
    plant_rate_capacity,
)
import tools.ni_staged_controller_test as ni_stage
import tools.validate_controllers as validate_cli
from controllers import ControllerAction, ControllerDiagnostics, StepResult


class ResponsivePlantTests(unittest.TestCase):
    def test_strict_json_sanitizes_nested_numpy_nonfinite_tracking_metrics(self) -> None:
        payload = {
            "tracking": {
                "final_window_moving_mean_error_z": np.float32(np.nan),
                "array": np.asarray([1.0, np.inf, np.nan], dtype=np.float32),
                "trace": [{"z": np.float64(0.25), "error_z": np.float32(-np.inf)}],
            }
        }
        serialized = json.dumps(json_safe(payload), allow_nan=False)
        parsed = json.loads(serialized)
        self.assertIsNone(parsed["tracking"]["final_window_moving_mean_error_z"])
        self.assertEqual(parsed["tracking"]["array"], [1.0, None, None])
        self.assertIsNone(parsed["tracking"]["trace"][0]["error_z"])

    def test_rate_limited_plant_capacity_includes_recovery_and_rate(self) -> None:
        settings = PlantSettings(
            disturbance_amplitude_mv=0.003, amplitude_reduction_per_pulse_mv=0.0005,
            effect_recovery_s=5.0, max_amplitude_reduction_mv=0.020,
        )
        capacity = plant_rate_capacity(
            settings, max_rate_hz=2.0, effective_max_rate_hz=1.2708,
            measured_generated_rate_hz=1.1, target_correction_mv=0.0026,
        )
        self.assertAlmostEqual(capacity["requested_rate_mean_capacity_mv"], 0.005)
        self.assertAlmostEqual(capacity["effective_rate_mean_capacity_mv"], 0.003177)
        self.assertAlmostEqual(capacity["disturbance_amplitude_mv"], 0.003)
        self.assertTrue(capacity["requested_rate_reaches_full_disturbance"])
        self.assertTrue(capacity["measured_generated_rate_reaches_target"])
        self.assertFalse(plant_rate_capacity(settings, max_rate_hz=0.0)["requested_rate_reaches_full_disturbance"])

    def test_capacity_reference_uses_actual_steady_reduction_not_regulated_usage(self) -> None:
        settings = replace(
            TRACKING_PLANT_SETTINGS, disturbance_start_s=12.0,
            disturbance_stop_s=42.0, effect_recovery_s=5.0,
        )
        plant = ResponsiveSyntheticPlant(settings)
        plant.reduction_history_mv = [0.003] * 12_800
        plant.command_effect_sample_indices = [
            int(t * settings.sample_rate_hz) for t in np.arange(22.4, 42.0, 0.8)
        ]
        metrics = {
            "source_samples": 1_280_000, "source_index_gap_samples": 0,
            "skipped_source_samples": 0, "max_requested_rate_hz": 2.0,
        }
        reference = _summarize_capacity_reference(
            plant, metrics, duration_s=64.0, lead_ms=100.0, target_z=1.0,
            target_correction_mv=0.0026, max_rate_hz=2.0,
            feature_diagnostics=[{"feature_recovery_samples": 3_638}],
            scheduler_stats={"committed_blocks": 12_820},
        )
        self.assertEqual(reference["steady_window_start_s"], 22.0)
        self.assertEqual(reference["steady_window_stop_s"], 42.0)
        self.assertAlmostEqual(reference["generated_effect_rate_hz_in_steady_window"], 1.25)
        self.assertAlmostEqual(reference["steady_mean_reduction_mv"], 0.003)
        self.assertGreater(reference["steady_reduction_margin_mv"], 0)
        self.assertTrue(reference["capacity_reference_reaches_target"])
        self.assertTrue(reference["max_request_limit_reached"])
        self.assertEqual(reference["scheduler"]["committed_blocks"], 12_820)

        gapped = _summarize_capacity_reference(
            plant, {**metrics, "source_index_gap_samples": 100}, duration_s=64.0,
            lead_ms=100.0, target_z=1.0, target_correction_mv=0.0026,
            max_rate_hz=2.0, feature_diagnostics=[{"feature_recovery_samples": 3_638}],
            scheduler_stats={},
        )
        self.assertFalse(gapped["capacity_reference_reaches_target"])
        below_limit = _summarize_capacity_reference(
            plant, {**metrics, "max_requested_rate_hz": 1.9}, duration_s=64.0,
            lead_ms=100.0, target_z=1.0, target_correction_mv=0.0026,
            max_rate_hz=2.0, feature_diagnostics=[{"feature_recovery_samples": 3_638}],
            scheduler_stats={},
        )
        self.assertTrue(below_limit["steady_reduction_margin_mv"] > 0)
        self.assertFalse(below_limit["max_request_limit_reached"])
        self.assertFalse(below_limit["capacity_reference_reaches_target"])

    def test_feedback_acceptance_uses_max_request_reference_not_actual_pid_usage(self) -> None:
        tracking = {"iae_z_seconds": 10.0, "window_complete": True, "target_z": 1.0}
        active = {
            "activity": {"disturbance_ca3_block_rms_mv": 0.08},
            "controller_tracking": tracking, "tracking_gate_passed": True,
            "plant_recovery": {"bounded_recovery": True}, "plant_rate_capacity": {},
            "regulated_controller_output_usage": {"generated_effect_rate_hz_in_disturbance": 0.1},
            "generated_nonzero_ao_command_samples": 100,
            "plant_effect_pulse_count": 1, "metrics": {"max_requested_rate_hz": 2.0},
        }
        disabled = {
            "activity": {"disturbance_ca3_block_rms_mv": 0.09},
            "controller_tracking": {**tracking, "iae_z_seconds": 20.0},
            "tracking_gate_passed": False, "plant_recovery": {"bounded_recovery": True},
            "plant_rate_capacity": {}, "regulated_controller_output_usage": {},
            "generated_nonzero_ao_command_samples": 0, "plant_effect_pulse_count": 0,
            "metrics": {},
        }
        reference = {"capacity_reference_reaches_target": True, "steady_mean_reduction_mv": 0.003}
        with patch("simulation.run_responsive_case", side_effect=[(active, {}), (disabled, {})]), \
             patch("simulation.run_capacity_reference_case", return_value=reference):
            result = run_feedback_comparison(
                "pid", duration_s=64.0, plant_settings=TRACKING_PLANT_SETTINGS,
                pid_settings={"target_z": 1.0, "max_rate_hz": 2.0},
                include_capacity_reference=True,
            )
        self.assertEqual(result["regulated_controller_output_usage"]["generated_effect_rate_hz_in_disturbance"], 0.1)
        self.assertEqual(result["plant_rate_capacity"], reference)
        self.assertTrue(result["simulation_acceptance_passed"])

    def test_tracking_coverage_excludes_only_explicit_feature_masks(self) -> None:
        settings = PlantSettings(disturbance_start_s=0.0, disturbance_stop_s=5.0)
        trace = [(i * 0.005, 1.0, 0.2) for i in range(1_000)]
        diagnostics = [{"feature_valid": True, "feature_invalid_sample_count": 0,
                        "stimulated_sample_count": 0, "feature_recovery_samples": 200}
                       for _ in trace]
        # Empty startup diagnostics occur before the target window and must not
        # contaminate the target-window feature-invalid count.
        diagnostics[0]["feature_valid"] = None
        diagnostics[1]["feature_valid"] = None
        # The two blocks are known feature masks: one pulse and one post-mask guard.
        diagnostics[500] = {"feature_valid": False, "feature_invalid_sample_count": 100,
                            "stimulated_sample_count": 100, "feature_recovery_samples": 200}
        diagnostics[501] = {"feature_valid": False, "feature_invalid_sample_count": 100,
                            "stimulated_sample_count": 0, "feature_recovery_samples": 200}
        summary = _summarize_target_tracking(
            trace, settings, target_z=1.0, max_rate_hz=2.0,
            feature_diagnostics=diagnostics, source_duration_s=5.0,
            source_samples_seen=100_000,
        )
        self.assertEqual(summary["known_feature_invalid_blocks"], 2)
        self.assertEqual(summary["expected_eligible_blocks"], 898)
        self.assertEqual(summary["eligible_observation_blocks"], 898)
        self.assertEqual(summary["stimulated_mask_samples"], 100)
        self.assertEqual(summary["feature_guard_samples"], 100)
        self.assertAlmostEqual(summary["masked_coverage_fraction"], 200 / 90_000)
        self.assertTrue(summary["full_source_coverage"])

        diagnostics[501]["feature_recovery_samples"] = 0
        unexplained = _summarize_target_tracking(
            trace, settings, target_z=1.0, max_rate_hz=2.0,
            feature_diagnostics=diagnostics, source_duration_s=5.0,
            source_samples_seen=100_000,
        )
        self.assertGreater(unexplained["unexplained_feature_invalid_samples"], 0)
        self.assertFalse(unexplained["tracking_response_gate_passed"])

        gapped = _summarize_target_tracking(
            trace, settings, target_z=1.0, max_rate_hz=2.0,
            feature_diagnostics=diagnostics, source_duration_s=5.0,
            source_samples_seen=100_000, source_gap_samples=100,
        )
        self.assertFalse(gapped["full_source_coverage"])
        self.assertFalse(gapped["tracking_response_gate_passed"])

    def test_moving_means_keep_source_time_grid_across_known_masks(self) -> None:
        settings = PlantSettings(disturbance_start_s=0.0, disturbance_stop_s=5.0)
        trace = [(i * 0.005, 1.2, 0.2) for i in range(1_000)]
        diagnostics = [{"feature_valid": True, "feature_invalid_sample_count": 0,
                        "stimulated_sample_count": 0, "feature_recovery_samples": 200}
                       for _ in trace]
        for index in range(500, 1_000, 50):
            diagnostics[index] = {
                "feature_valid": False, "feature_invalid_sample_count": 100,
                "stimulated_sample_count": 100, "feature_recovery_samples": 200,
            }
        result = _summarize_target_tracking(
            trace, settings, target_z=1.0, max_rate_hz=2.0,
            feature_diagnostics=diagnostics, source_duration_s=5.0,
            source_samples_seen=100_000,
        )
        self.assertEqual(result["eligible_observation_blocks"], 890)
        self.assertGreater(result["complete_elapsed_moving_window_count"], 0)
        self.assertAlmostEqual(result["minimum_moving_window_eligible_coverage"], 1.0)
        self.assertTrue(result["tracking_response_gate_passed"])

    def test_settling_requires_staying_in_band_and_rejects_large_ripple(self) -> None:
        settings = PlantSettings(disturbance_start_s=0.0, disturbance_stop_s=5.0)
        trace = []
        for index in range(1_000):
            t = index * 0.005
            z = 3.0 if t < 1.0 else 1.2
            trace.append((t, z, 0.5))
        settled = _summarize_target_tracking(trace, settings, target_z=1.0, max_rate_hz=2.0)
        self.assertTrue(settled["settled_within_band"])
        self.assertTrue(settled["tracking_response_gate_passed"])
        self.assertLessEqual(settled["final_window_moving_mean_abs_error_z"], 0.5)

        transient = [
            (t, 3.0 if t < 1.0 or t >= 2.0 else 1.1, 0.5)
            for t in np.arange(0, 5.0, 0.005)
        ]
        not_settled = _summarize_target_tracking(transient, settings, target_z=1.0, max_rate_hz=2.0)
        self.assertFalse(not_settled["settled_within_band"])
        self.assertFalse(not_settled["tracking_response_gate_passed"])

        alternating = [
            (t, 5.0 if index % 2 == 0 else -3.0, 0.5)
            for index, t in enumerate(np.arange(0, 5.0, 0.005))
        ]
        ripple = _summarize_target_tracking(alternating, settings, target_z=1.0, max_rate_hz=2.0)
        self.assertTrue(ripple["settled_within_band"])
        self.assertFalse(ripple["raw_ripple_within_bounds"])
        self.assertGreater(ripple["raw_error_std_z"], ripple["raw_error_std_limit_z"])
        self.assertFalse(ripple["tracking_response_gate_passed"])

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
        # A short regulated run reports actual pulse usage separately; it is
        # not presented as a maximum-capacity reference measurement.
        self.assertIsNone(active["plant_rate_capacity"]["measured_generated_rate_hz"])
        self.assertIsNone(active["plant_rate_capacity"]["measured_generated_rate_reaches_target"])
        self.assertGreater(
            active["regulated_controller_output_usage"]["generated_effect_rate_hz_in_disturbance"], 0.0)
        self.assertEqual(
            quiet["regulated_controller_output_usage"]["generated_effect_rate_hz_in_disturbance"], 0.0)
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

class ActualCoreBenchmarkTests(unittest.TestCase):
    def test_benchmark_exports_float64_actuator_phase_high_water_separately_from_current_phase(self) -> None:
        actuator = app.PulseActuator(refractory_s=0.0)
        actuator.arm()
        actuator.process(
            ControllerAction(kind="pulse_rate", requested_rate_hz=2.0),
            first_sample_index=0, sample_count=10_000, sample_rate_hz=20_000,
            armed=True, validity_mask=np.ones(10_000, dtype=bool), amplitude_v=0.25,
            generation=0,
        )
        # A full-cycle crossing wraps the live phase to zero. The diagnostic
        # must retain the pre-modulo peak from the actual float64 actuator.
        self.assertEqual(actuator.rate_phase, 0.0)
        self.assertGreaterEqual(actuator.rate_phase_high_water, 1.0 - 1e-12)
        actuator.process(
            ControllerAction(kind="pulse_rate", requested_rate_hz=2.0),
            first_sample_index=10_000, sample_count=1_000, sample_rate_hz=20_000,
            armed=True, validity_mask=np.ones(1_000, dtype=bool), amplitude_v=0.25,
            generation=0,
        )
        self.assertAlmostEqual(actuator.rate_phase, 0.1)
        actuator.process(
            ControllerAction(kind="pulse_rate", requested_rate_hz=0.0),
            first_sample_index=11_000, sample_count=100, sample_rate_hz=20_000,
            armed=True, validity_mask=np.ones(100, dtype=bool), amplitude_v=0.25,
            generation=0,
        )

        metrics = RunMetrics(
            case_id="phase-fixture", source_name="fixture", controller="pid",
            sample_rate_hz=20_000, block_samples=100, block_ms=5.0,
            ao_lead_ms=100.0, deadline_ms=5.0,
        )
        phase = benchmark._capture_actuator_rate_phase(metrics, actuator)
        exported = json.loads(json.dumps(benchmark.json_safe(asdict(metrics)), allow_nan=False))

        self.assertGreaterEqual(phase["reset_count"], 1)
        self.assertGreater(phase["high_water"], phase["final"])
        self.assertEqual(phase["final"], 0.0)
        self.assertEqual(exported["actuator_rate_phase_high_water"], phase["high_water"])
        self.assertEqual(exported["actuator_rate_phase_reset_count"], phase["reset_count"])
        self.assertEqual(exported["actuator_rate_phase_final"], phase["final"])

    def test_canonical_replay_labels_and_roles_survive_into_recording_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "canonical_ec.h5"
            with h5py.File(trace, "w") as h5:
                h5.attrs.update(schema="echo-chamber-replay", schema_version=1,
                                sample_rate_hz=20_000, committed_samples=5_000)
                dataset = h5.create_dataset(
                    "signals/electrode_mV",
                    data=np.vstack((0.05 * np.sin(2 * np.pi * 150 * np.arange(5_000) / 20_000),
                                    0.03 * np.sin(2 * np.pi * 80 * np.arange(5_000) / 20_000))),
                )
                text = h5py.string_dtype(encoding="utf-8")
                dataset.attrs.update(unit="mV")
                dataset.attrs["channel_labels"] = np.asarray(["CA3", "EC"], dtype=text)
                dataset.attrs["channel_roles"] = np.asarray(["CA3", "Cortex"], dtype=text)
                dataset.attrs["channel_regions"] = np.asarray(["CA3", "EC"], dtype=text)
            recording_dir = Path(directory) / "out"
            result, environment = run_replay_case(
                trace, "rule", limit_seconds=0.25, paced=True, recording_dir=recording_dir,
            )
            self.assertTrue(result.recording_completed)
            self.assertEqual(tuple(environment["config"]["electrode_labels"]), ("CA3", "EC"))
            self.assertEqual(environment["config"]["ctx_index"], 1)
            self.assertEqual(environment["ui_channel_mapping"],
                             {"channels": ["CA3", "EC"], "cortex_ai": "ai1"})
            with h5py.File(environment["recording_file"], "r") as recorded:
                channel_data = recorded["signals/electrode_mV"]
                self.assertEqual([x.decode() if isinstance(x, bytes) else str(x)
                                  for x in channel_data.attrs["channel_labels"]], ["CA3", "EC"])
                self.assertEqual([x.decode() if isinstance(x, bytes) else str(x)
                                  for x in channel_data.attrs["channel_roles"]], ["CA3", "Cortex"])
                self.assertEqual([x.decode() if isinstance(x, bytes) else str(x)
                                  for x in channel_data.attrs["channel_regions"]], ["CA3", "EC"])
                metadata = json.loads(recorded.attrs["metadata_json"])
                self.assertEqual(tuple(metadata["configuration"]["electrode_labels"]), ("CA3", "EC"))
                self.assertEqual(metadata["configuration"]["ctx_index"], 1)

    def test_reversed_ec_channel_order_is_preserved_in_websocket_packet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "reversed_ec.h5"
            with h5py.File(trace, "w") as h5:
                h5.attrs.update(schema="echo-chamber-replay", schema_version=1,
                                sample_rate_hz=20_000, committed_samples=5_000)
                dataset = h5.create_dataset(
                    "signals/electrode_mV",
                    data=np.vstack((0.03 * np.sin(2 * np.pi * 80 * np.arange(5_000) / 20_000),
                                    0.05 * np.sin(2 * np.pi * 150 * np.arange(5_000) / 20_000))),
                )
                text = h5py.string_dtype(encoding="utf-8")
                dataset.attrs.update(unit="mV")
                dataset.attrs["channel_labels"] = np.asarray(["EC", "CA3"], dtype=text)
                dataset.attrs["channel_roles"] = np.asarray(["Cortex", "CA3"], dtype=text)
                dataset.attrs["channel_regions"] = np.asarray(["EC", "CA3"], dtype=text)
            result, environment = run_replay_case(
                trace, "rule", limit_seconds=0.25, paced=True,
                recording_dir=Path(directory) / "out",
            )
        self.assertTrue(result.recording_completed)
        self.assertEqual(tuple(environment["config"]["electrode_labels"]), ("EC", "CA3"))
        self.assertEqual(environment["config"]["ctx_index"], 0)
        self.assertEqual(environment["ui_channel_mapping"],
                         {"channels": ["EC", "CA3"], "cortex_ai": "ai0"})

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
                    self.assertEqual(result.recorder_backpressure_policy, "disabled_for_paced")
                    self.assertEqual(result.recorder_backpressure_wait_count, 0)
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

    def test_unpaced_backpressure_keeps_queue_bounded_without_advancing_source_clock(self) -> None:
        trace = Path(app.__file__).resolve().parent / "sample" / "test_AI0_CA3_AI1_CTX.h5"
        real_config = benchmark._make_config
        real_append = app.H5Recorder._append_block_records

        def small_queue_config(*args, **kwargs):
            return replace(real_config(*args, **kwargs), logger_queue_blocks=8, recorder_batch_s=0.005)

        def slow_append(recorder, item, offset):
            time.sleep(0.01)
            return real_append(recorder, item, offset)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(benchmark, "_make_config", side_effect=small_queue_config):
                with patch.object(app.H5Recorder, "_append_block_records", slow_append):
                    result, environment = run_replay_case(
                        trace, "rule", limit_seconds=0.25, paced=False,
                        recording_dir=Path(directory) / "recording",
                    )
        self.assertEqual(result.processed_blocks, 50)
        self.assertEqual(result.source_index_gap_samples, 0)
        self.assertEqual(result.recording_committed_samples, 5_000)
        self.assertTrue(result.recording_completed)
        self.assertEqual(result.recorder_backpressure_policy, "bounded_queue_wait_between_blocks")
        self.assertGreater(result.recorder_backpressure_wait_count, 0)
        self.assertGreater(result.recorder_backpressure_wait_ms, 0.0)
        self.assertLessEqual(result.recorder_backpressure_max_backlog, 8)
        self.assertEqual(result.ao_generated_sample_count, 7_400)
        self.assertEqual(environment["config"]["logger_queue_blocks"], 8)


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
    @staticmethod
    def _healthy_case(*, seconds: float, blocks: int, samples: int) -> RunMetrics:
        return RunMetrics(
            case_id="coverage-fixture", source_name="fixture", controller="rule",
            sample_rate_hz=20_000, block_samples=100, block_ms=5.0,
            ao_lead_ms=100.0, deadline_ms=5.0,
            source_samples=samples, processed_blocks=blocks, duration_source_s=seconds,
            process_p99_ms=1.0, process_max_ms=2.0,
            ao_drain_complete=True, recording_completed=True,
            recording_committed_samples=samples,
            ui_packets_submitted=1, ui_packets_consumed=1,
        )

    def test_twenty_second_matrix_prefix_does_not_pass_full_replay_acceptance(self) -> None:
        diagnostic = evaluate_case_acceptance(
            self._healthy_case(seconds=20.0, blocks=4_000, samples=400_000), "matrix")
        self.assertEqual(diagnostic["scope"], "diagnostic_prefix")
        self.assertTrue(diagnostic["runtime_checks_passed"])
        self.assertIsNone(diagnostic["full_acceptance_passed"])
        self.assertEqual(diagnostic["full_acceptance_status"], "NOT EVALUATED")
        self.assertFalse(diagnostic["coverage"]["complete"])
        self.assertEqual(diagnostic["coverage"]["required"], {
            "duration_seconds": 1800.0, "processed_blocks": 360_000,
            "source_samples": 36_000_000, "sample_rate_hz": 20_000.0,
            "block_samples": 100, "block_ms": 5.0, "deadline_ms": 5.0,
        })
        self.assertEqual(diagnostic["coverage"]["observed"]["duration_seconds"], 20.0)

    def test_full_matrix_and_paced_cases_require_exact_reference_coverage(self) -> None:
        complete = self._healthy_case(seconds=1800.0, blocks=360_000, samples=36_000_000)
        matrix = evaluate_case_acceptance(complete, "matrix")
        paced = evaluate_case_acceptance(complete, "paced")
        self.assertEqual(matrix["scope"], "full_replay_case")
        self.assertEqual(paced["scope"], "paced_30min_case")
        self.assertTrue(matrix["coverage"]["complete"])
        self.assertTrue(matrix["full_acceptance_passed"])
        self.assertTrue(paced["full_acceptance_passed"])

    def test_duration_coverage_tolerance_is_one_sample_period(self) -> None:
        one_sample_short = self._healthy_case(
            seconds=1800.0 - 1 / 20_000, blocks=360_000, samples=36_000_000)
        two_samples_short = replace(
            one_sample_short, duration_source_s=1800.0 - 2 / 20_000)

        accepted = evaluate_case_acceptance(
            one_sample_short, "matrix", requested_scope="full_replay_case")
        rejected = evaluate_case_acceptance(
            two_samples_short, "matrix", requested_scope="full_replay_case")

        self.assertTrue(accepted["coverage"]["complete"])
        self.assertTrue(accepted["full_acceptance_passed"])
        self.assertFalse(rejected["coverage"]["complete"])
        self.assertFalse(rejected["full_acceptance_passed"])

    def test_full_duration_with_invalid_sampling_configuration_fails_full_scope(self) -> None:
        malformed = self._healthy_case(seconds=1800.0, blocks=360_000, samples=36_000_000)
        malformed.sample_rate_hz = 10_000

        result = evaluate_case_acceptance(
            malformed, "matrix", requested_scope="full_replay_case")

        self.assertEqual(result["scope"], "full_replay_case")
        self.assertFalse(result["coverage"]["complete"])
        self.assertIn("full replay coverage sample_rate_hz", " ".join(result["coverage"]["failures"]))
        self.assertFalse(result["full_acceptance_passed"])
        self.assertEqual(result["full_acceptance_status"], "FAIL")

    def test_latched_runtime_fault_fails_runtime_and_full_acceptance(self) -> None:
        faulty = self._healthy_case(seconds=1800.0, blocks=360_000, samples=36_000_000)
        faulty.runtime_fault = "safety trip after final source block"

        failures = acceptance_failures(faulty)
        result = evaluate_case_acceptance(
            faulty, "matrix", requested_scope="full_replay_case")

        self.assertTrue(any("runtime fault was latched" in failure for failure in failures))
        self.assertFalse(result["runtime_checks_passed"])
        self.assertFalse(result["full_acceptance_passed"])

    def test_collection_gate_requires_nine_matrix_or_three_paced_cases(self) -> None:
        matrix_cases = [
            replace(self._healthy_case(seconds=1800.0, blocks=360_000, samples=36_000_000),
                    case_id=f"{trace}-{controller}", source_name=trace, controller=controller)
            for trace in ("trace-a", "trace-b", "trace-c")
            for controller in ("rule", "pid", "esn")
        ]
        complete = evaluate_collection_acceptance(
            "matrix", matrix_cases, [], requested_scope="full_replay_case", finalized=True)
        self.assertEqual(complete["status"], "PASS")
        self.assertTrue(complete["full_acceptance_passed"])
        self.assertEqual(complete["required"]["case_count"], 9)
        self.assertEqual(complete["observed"]["completed_case_count"], 9)
        self.assertEqual(complete["observed"]["source_count"], 3)
        self.assertEqual(complete["observed"]["unique_source_controller_pairs"], 9)

        malformed_matrix = list(matrix_cases)
        malformed_matrix[0] = replace(malformed_matrix[0], sample_rate_hz=10_000)
        malformed_collection = evaluate_collection_acceptance(
            "matrix", malformed_matrix, [], requested_scope="full_replay_case", finalized=True)
        self.assertEqual(malformed_collection["status"], "FAIL")
        self.assertFalse(malformed_collection["full_acceptance_passed"])
        self.assertEqual(malformed_collection["observed"]["failed_full_acceptance_case_ids"],
                         [malformed_matrix[0].case_id])

        subset = evaluate_collection_acceptance(
            "matrix", matrix_cases[:3], [], requested_scope="full_replay_case", finalized=True)
        self.assertEqual(subset["status"], "INCOMPLETE")
        self.assertFalse(subset["full_acceptance_passed"])

        paced_cases = [
            replace(self._healthy_case(seconds=1800.0, blocks=360_000, samples=36_000_000),
                    case_id=f"paced-{controller}", source_name="trace-a", controller=controller)
            for controller in ("rule", "pid", "esn")
        ]
        paced = evaluate_collection_acceptance(
            "paced", paced_cases, [], requested_scope="paced_30min_case", finalized=True)
        self.assertEqual(paced["status"], "PASS")
        self.assertEqual(paced["required"]["case_count"], 3)

        malformed_paced_cases = list(paced_cases)
        malformed_paced_cases[1] = replace(malformed_paced_cases[1], deadline_ms=6.0)
        malformed_paced = evaluate_collection_acceptance(
            "paced", malformed_paced_cases, [], requested_scope="paced_30min_case",
            finalized=True)
        self.assertEqual(malformed_paced["status"], "FAIL")
        self.assertFalse(malformed_paced["full_acceptance_passed"])
        self.assertEqual(malformed_paced["observed"]["failed_full_acceptance_case_ids"],
                         [malformed_paced_cases[1].case_id])

        diagnostic_collection = evaluate_collection_acceptance(
            "matrix", [self._healthy_case(seconds=20.0, blocks=4_000, samples=400_000)], [],
            requested_scope="diagnostic_prefix", finalized=True)
        self.assertEqual(diagnostic_collection["status"], "NOT EVALUATED")
        self.assertIsNone(diagnostic_collection["full_acceptance_passed"])

    def test_short_matrix_cli_records_runtime_pass_and_null_full_acceptance(self) -> None:
        diagnostic = self._healthy_case(seconds=20.0, blocks=4_000, samples=400_000)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(validate_cli, "run_replay_case", return_value=(diagnostic, {})):
                code = validate_cli.main([
                    "--matrix", "--trace", "synthetic-fixture.h5", "--controller", "rule",
                    "--duration-seconds", "20", "--output-dir", directory,
                ])
            aggregate = next(Path(directory).glob("controller_matrix_*.json"))
            checkpoint = next(Path(directory).glob("controller_case_*.json"))
            csv_path = aggregate.with_suffix(".csv")
            result = json.loads(aggregate.read_text(encoding="utf-8"))
            case_result = json.loads(checkpoint.read_text(encoding="utf-8"))
            with csv_path.open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
        self.assertEqual(code, 0)
        for acceptance in (result["acceptance"][0], case_result["acceptance"]):
            self.assertEqual(acceptance["scope"], "diagnostic_prefix")
            self.assertTrue(acceptance["runtime_checks_passed"])
            self.assertIsNone(acceptance["full_acceptance_passed"])
            self.assertEqual(acceptance["full_acceptance_status"], "NOT EVALUATED")
        self.assertEqual(row["acceptance_scope"], "diagnostic_prefix")
        self.assertEqual(row["runtime_checks_passed"], "True")
        self.assertEqual(row["full_acceptance_status"], "NOT EVALUATED")
        self.assertEqual(row["full_acceptance_passed"], "")
        self.assertEqual(row["required_blocks"], "360000")
        self.assertEqual(row["observed_blocks"], "4000")
        self.assertEqual(row["required_samples"], "36000000")
        self.assertEqual(row["required_sample_rate_hz"], "20000.0")
        self.assertEqual(row["required_block_samples"], "100")
        self.assertEqual(result["collection_acceptance"]["status"], "NOT EVALUATED")
        self.assertIsNone(result["collection_acceptance"]["full_acceptance_passed"])
        self.assertEqual(row["collection_status"], "NOT EVALUATED")

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
        self.assertFalse(result["acceptance"][0]["runtime_checks_passed"])
        self.assertIsNone(result["acceptance"][0]["full_acceptance_passed"])
        self.assertTrue(any("process p99" in item for item in result["acceptance"][0]["failures"]))
        self.assertEqual(rows[0]["runtime_checks_passed"], "False")
        self.assertEqual(rows[0]["full_acceptance_passed"], "")

    def test_matrix_exception_writes_machine_readable_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(validate_cli, "run_replay_case",
                              side_effect=RuntimeError("recording queue overflow")):
                code = validate_cli.main([
                    "--matrix", "--trace", "missing-fixture.h5", "--controller", "rule",
                    "--duration-seconds", "0.25", "--output-dir", directory,
                ])
            json_path = next(Path(directory).glob("controller_matrix_*.json"))
            csv_path = json_path.with_suffix(".csv")
            result = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(code, 1)
        self.assertEqual(len(result["case_failures"]), 1)
        self.assertEqual(result["case_failures"][0]["message"], "recording queue overflow")
        self.assertFalse(result["acceptance"][0]["runtime_checks_passed"])
        self.assertIsNone(result["acceptance"][0]["full_acceptance_passed"])
        self.assertEqual(rows[0]["case_status"], "failed")
        self.assertEqual(rows[0]["failure_type"], "RuntimeError")

    def test_completed_case_checkpoint_survives_later_case_exception(self) -> None:
        completed = RunMetrics(
            case_id="first-case", source_name="fixture", controller="rule",
            sample_rate_hz=20_000, block_samples=100, block_ms=5.0,
            ao_lead_ms=100.0, deadline_ms=5.0, processed_blocks=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(validate_cli, "run_replay_case",
                              side_effect=[(completed, {"source_sha256": "fixture-hash"}),
                                            RuntimeError("second case failed")]):
                code = validate_cli.main([
                    "--matrix", "--trace", "synthetic-fixture.h5", "--controller", "rule",
                    "--controller", "pid", "--duration-seconds", "0.25", "--output-dir", directory,
                ])
            checkpoints = sorted(Path(directory).glob("controller_case_*.json"))
            self.assertEqual(len(checkpoints), 2)
            documents = [json.loads(path.read_text(encoding="utf-8")) for path in checkpoints]
            rows_by_status = {document["case_status"]: document for document in documents}
            self.assertEqual(code, 1)
            self.assertEqual(rows_by_status["completed"]["case"]["case_id"], "first-case")
            self.assertEqual(rows_by_status["completed"]["environment"]["source_sha256"], "fixture-hash")
            self.assertFalse(rows_by_status["completed"]["matrix_complete"])
            self.assertEqual(rows_by_status["failed"]["failure"]["message"], "second case failed")
            self.assertFalse(rows_by_status["failed"]["matrix_complete"])
            for checkpoint in checkpoints:
                with checkpoint.with_suffix(".csv").open(encoding="utf-8", newline="") as stream:
                    csv_rows = list(csv.DictReader(stream))
                self.assertEqual(len(csv_rows), 1)
                self.assertEqual(csv_rows[0]["matrix_complete"], "False")


class RuntimeHealthMetricsTests(unittest.TestCase):
    def test_harness_routes_typed_pulse_ranges_and_counts_commits_idempotently(self) -> None:
        class RecorderCapture:
            def __init__(self):
                self.blocks = []
                self.stages = []

            def submit(self, block):
                self.blocks.append(block)

            def record_action_stage(self, **stage):
                self.stages.append(stage)

        class QuietCore(app.ProcessingCore):
            def _publish_ui(self, *args, **kwargs):
                pass

        class TwoPulseController:
            def reset(self, keep_calibration: bool = False):
                pass

            def drain(self):
                pass

            def step(self, inp):
                requests = {800: (880,), 10_800: (10_880,)}.get(inp.first_sample_index, ())
                return StepResult(
                    ControllerAction(kind="pulse_events", pulse_sample_indices=requests),
                    ControllerDiagnostics(
                        "rule", True, True, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0,
                        "ready", inp.generation,
                    ),
                )

        config = app.AppConfig(processing_block_ms=5.0, ao_target_lead_ms=100.0)
        config.validate()
        state = app.RuntimeState(
            esn_ready=False, start_paused=False, electrode_labels=("CA3", "EC"), ctx_index=1,
            pulse_threshold_std=3.0, pulse_window_sec=10.0, controller="rule",
            controller_ready={"esn": False, "rule": True, "pid": True},
        )
        state.set_stim("threshold_pulse", 0.5)
        state.set_mode("closed-loop")
        state.acknowledge_drain(state.snapshot().generation)
        state.update_controller_calibration("rule", True)
        state.set_armed(True)
        state.set_recording(True)
        recorder = RecorderCapture()
        core = QuietCore(config, state, app.EsnRuntime(None, config), recorder,
                         object(), None, app.Telemetry())
        core.controllers["rule"] = TwoPulseController()

        harness = _Harness.__new__(_Harness)
        harness.state, harness.recorder, harness.core = state, recorder, core
        harness.pulse_commit_samples = 0
        harness._pulse_commit_ids = set()
        harness._pulse_commit_ranges = benchmark._CommittedAoRangeCounter()
        harness._pulse_commit_lock = threading.Lock()
        scheduler = harness._create_scheduler(config)
        admitted_shapes = []

        def process_source(source_index: int, *, expected_accepted: bool = True):
            _, safe, _, reason = core.process(
                np.zeros((2, config.chunk_size), dtype=np.float64), sample_index=source_index,
                ao_target_sample_index=source_index + config.ao_lead_samples,
                input_is_electrode_mV=True, input_roles=("CA3", "Cortex"),
                input_labels=("CA3", "EC"), input_ctx_index=1,
            )
            self.assertIsNone(reason)
            admitted_shapes.extend((source_index, plan.pulse_id, plan.waveform_v.size,
                                    plan.target_onset_sample_index)
                                   for plan in core.last_admitted_pulse_plans)
            target = source_index // config.chunk_size + config.ao_lead_chunks
            accepted = scheduler.submit(target, safe.reshape(-1), core.last_processed_generation,
                                        core.last_pulse_events, core.last_admitted_pulse_plans)
            self.assertEqual(accepted, expected_accepted)

        for source_index in range(0, 10_800, config.chunk_size):
            process_source(source_index)
            scheduler.advance()

        # The first pulse has committed at its original target. Source block
        # 900 has also sent its cumulative local-account ACK through the harness
        # binding and the scheduler has serviced that queue.
        self.assertEqual(scheduler.stats_snapshot()["pulse_committed_samples"], 100)
        self.assertGreater(harness.pulse_commit_samples, 0)
        self.assertEqual(len(harness._pulse_commit_ids), 1)
        self.assertTrue(scheduler._local_account_queue.empty())

        process_source(10_800)
        state.fault("deterministic cancellation after second pulse admission")
        scheduler.advance()
        source_index_at_eof = core.sample_index
        recorder_blocks_at_eof = len(recorder.blocks)
        self.assertFalse(any(stage.get("stage") == "ao_cancelled"
                             and stage.get("request_id") == "g2-p2"
                             for stage in recorder.stages))
        harness._flush_final_core_notifications()
        eof_cancel_rows = [stage for stage in recorder.stages
                           if stage.get("stage") == "ao_cancelled"
                           and stage.get("request_id") == "g2-p2"]
        self.assertTrue(eof_cancel_rows)
        self.assertEqual(core.sample_index, source_index_at_eof)
        self.assertEqual(len(recorder.blocks), recorder_blocks_at_eof)

        by_pulse = {}
        for stage in recorder.stages:
            request_id = stage.get("request_id")
            if request_id:
                by_pulse.setdefault(request_id, []).append(stage)
        self.assertEqual(len(by_pulse), 2)
        all_stages = [stage for values in by_pulse.values() for stage in values
                      if "range_end_sample_index" in stage]
        statuses = {stage["stage"] for stage in all_stages}
        self.assertIn("ao_reserved", statuses)
        self.assertIn("ao_commit", statuses)
        self.assertIn("ao_cancelled", statuses)
        committed = [stage for stage in all_stages if stage["stage"] == "ao_commit"]
        self.assertEqual(admitted_shapes, [(800, "g2-p1", 100, 2_880),
                                           (10_800, "g2-p2", 100, 12_880)])
        self.assertEqual(sum(stage["range_end_sample_index"] - stage["sample_index"]
                             for stage in committed), 100,
                         (committed, admitted_shapes, scheduler.stats_snapshot()))
        committed_ranges = {(stage["sample_index"], stage["range_end_sample_index"])
                            for stage in committed}
        self.assertEqual(committed_ranges, {(2_880, 2_900), (2_900, 2_980)})
        reserved_ranges = {(stage["sample_index"], stage["range_end_sample_index"])
                           for stage in all_stages if stage["stage"] == "ao_reserved"}
        self.assertEqual(reserved_ranges, committed_ranges)
        cancelled = [stage for stage in all_stages if stage["stage"] == "ao_cancelled"]
        self.assertEqual({(stage["sample_index"], stage["range_end_sample_index"])
                          for stage in cancelled}, {(12_880, 12_900), (12_900, 12_980)})
        self.assertTrue(all(stage["sample_index"] < stage["range_end_sample_index"]
                            for stage in all_stages))
        self.assertTrue(all(stage["controller"] == "rule" for stage in all_stages))
        self.assertEqual(harness.pulse_commit_samples, 100,
                         (scheduler.stats_snapshot(), recorder.stages, core.last_pulse_events,
                          admitted_shapes))

    def test_committed_ao_range_counter_is_idempotent_and_bounded(self) -> None:
        counter = benchmark._CommittedAoRangeCounter()
        self.assertEqual(counter.add("pulse", 100, 140), 40)
        self.assertEqual(counter.add("pulse", 100, 140), 0)
        self.assertEqual(counter.add("pulse", 120, 150), 10)
        self.assertEqual(counter.add("pulse", 110, 140), 0)
        for index in range(513):
            counter.add(f"p{index}", index * 100, index * 100 + 1)
        self.assertEqual(counter.add("pulse", 100, 140), 0)
        self.assertEqual(counter._committed_through, 51_201)

    def test_proposed_pulses_include_rejections_and_gate_categorizes_all(self) -> None:
        metrics = RunMetrics(
            case_id="pulse-categories", source_name="fixture", controller="pid",
            sample_rate_hz=20_000, block_samples=100, block_ms=5.0,
            ao_lead_ms=100.0, deadline_ms=5.0,
        )
        seen: set[str] = set()
        Event = type("Event", (), {})
        accepted, rejected = Event(), Event()
        accepted.pulse_id, accepted.status = "p1", "rendered_for_ao"
        rejected.pulse_id, rejected.status = "p2", "rejected_refractory"
        _record_pulse_event_metrics(metrics, seen, [accepted, rejected, accepted])
        self.assertEqual(metrics.requested_pulse_events, 2)
        self.assertEqual(metrics.pulse_request_ids, ["p1", "p2"])
        self.assertEqual(metrics.rendered_pulse_events, 1)
        self.assertEqual(metrics.rejected_pulse_events, 1)
        self.assertEqual(metrics.pulse_rejected_ids, ["p2"])
        self.assertFalse(any("categorized as rendered or rejected" in failure
                             for failure in acceptance_failures(metrics)))

        unknown = Event()
        unknown.pulse_id, unknown.status = "p3", "new_status"
        _record_pulse_event_metrics(metrics, seen, [unknown])
        self.assertEqual(metrics.unclassified_pulse_events, 1)
        self.assertTrue(any("categorized as rendered or rejected" in failure
                            for failure in acceptance_failures(metrics)))

    def test_recorder_backpressure_times_out_with_queue_diagnostics(self) -> None:
        class StalledRecorder:
            backlog = 6
            error = None

        harness = _Harness.__new__(_Harness)
        harness.paced = False
        harness.recorder = StalledRecorder()
        harness.config = type("Config", (), {"logger_queue_blocks": 8})()
        harness.progress_path = None
        harness.metrics = type("Metrics", (), {
            "recorder_backpressure_threshold_blocks": 6,
            "recorder_backpressure_wait_count": 0,
            "recorder_backpressure_wait_ms": 0.0,
            "recorder_backpressure_max_backlog": 0,
            "recorder_backpressure_policy": "bounded_queue_wait_between_blocks",
        })()

        with patch.object(_Harness, "UNPACED_RECORDER_NO_PROGRESS_TIMEOUT_S", 0.01):
            with self.assertRaisesRegex(
                TimeoutError,
                r"source_sample_index=500, queue_depth=6, queue_capacity_blocks=8, threshold_blocks=6",
            ):
                harness._wait_for_unpaced_recorder_capacity(500)
        self.assertEqual(harness.metrics.recorder_backpressure_wait_count, 1)
        self.assertGreater(harness.metrics.recorder_backpressure_wait_ms, 0.0)
        self.assertEqual(harness.metrics.recorder_backpressure_max_backlog, 6)

    def test_browser_disconnect_then_reconnect_still_fails_presence_gate(self) -> None:
        tracker = _BrowserPresenceTracker()
        tracker.observe(True, 100)
        tracker.observe(False, 200)
        tracker.observe(True, 300)
        presence = tracker.snapshot(400)

        summary = RunMetrics(
            case_id="browser-reconnect", source_name="fixture", controller="rule",
            sample_rate_hz=20_000, block_samples=100, block_ms=5.0,
            ao_lead_ms=100.0, deadline_ms=5.0, processed_blocks=1,
            browser_ui_required=True, browser_connected=presence["ever_connected"],
            browser_disconnect_count=presence["disconnect_count"],
            browser_disconnected_duration_ms=presence["disconnected_duration_ms"],
            browser_connected_at_finish=presence["connected_at_finish"],
            browser_disconnect_open_at_finish=presence["disconnect_open_at_finish"],
            ao_drain_complete=True, recording_completed=True,
            recording_committed_samples=100, source_samples=100,
            ui_packets_submitted=1, ui_packets_consumed=1,
        )
        failures = acceptance_failures(summary)
        self.assertTrue(summary.browser_connected_at_finish)
        self.assertEqual(summary.browser_disconnect_count, 1)
        self.assertAlmostEqual(summary.browser_disconnected_duration_ms, 0.0001)
        self.assertIn("required browser UI disconnected during the run", failures)

    def test_hub_counts_external_page_separately_from_internal_consumer(self) -> None:
        tracker = _BrowserPresenceTracker()
        hub = _CountingUiHub(2, tracker)
        internal = hub.subscribe()
        self.assertFalse(tracker.snapshot()["ever_connected"])
        browser = hub.subscribe()
        self.assertTrue(tracker.snapshot()["ever_connected"])
        hub.unsubscribe(browser)
        hub.subscribe()
        status = tracker.snapshot()
        self.assertEqual(status["disconnect_count"], 1)
        self.assertTrue(status["connected_at_finish"])

    def test_recorder_close_failure_does_not_skip_other_abort_cleanup(self) -> None:
        harness = _Harness.__new__(_Harness)
        calls: list[str] = []
        harness._stop_ao_clock = lambda: calls.append("ao")
        harness._ao_clock_thread = None
        harness._pending_ao_shutdown_error = None
        harness.state = type("State", (), {
            "set_armed": lambda self, value: calls.append("disarm"),
            "set_recording": lambda self, value: calls.append("recording"),
        })()

        class Recorder:
            def stop_recording(self) -> None:
                calls.append("stop-recorder")

            def close(self) -> None:
                calls.append("close-recorder")
                raise RuntimeError("writer did not stop")

        harness.recorder = Recorder()
        harness.core = type("Core", (), {
            "drain_ao_notifications": lambda self: calls.append("drain-core"),
            "close_ui": lambda self: calls.append("core-ui"),
        })()
        harness._run_on_loop = lambda awaitable: calls.append("ui-transport")
        harness._stop_ui_transport = lambda: "mock transport shutdown"
        harness.loop = asyncio.new_event_loop()
        harness.loop_thread = threading.Thread(target=harness.loop.run_forever, daemon=True)
        harness.loop_thread.start()
        harness.metrics = type("Metrics", (), {"case_id": "cleanup-test"})()
        harness.progress_path = None
        harness._gc_callback = lambda *args: None
        gc.callbacks.append(harness._gc_callback)
        harness._gc_callback_registered = True

        try:
            cleanup_errors = harness.abort()
        finally:
            if harness._gc_callback in gc.callbacks:
                gc.callbacks.remove(harness._gc_callback)
            if not harness.loop.is_closed():
                harness.loop.call_soon_threadsafe(harness.loop.stop)
                harness.loop_thread.join(timeout=1)
                harness.loop.close()

        self.assertIn("close recorder: RuntimeError: writer did not stop", cleanup_errors)
        self.assertEqual(calls, [
            "disarm", "ao", "drain-core", "recording", "stop-recorder",
            "close-recorder", "core-ui", "ui-transport",
        ])
        self.assertFalse(harness.loop_thread.is_alive())
        self.assertTrue(harness.loop.is_closed())
        self.assertNotIn(harness._gc_callback, gc.callbacks)


if __name__ == "__main__":
    unittest.main()
