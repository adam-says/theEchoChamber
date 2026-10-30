from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
import sys

import numpy as np
from scipy.signal import butter, sosfilt, welch
from controllers import ControllerAction, PulseActuator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import echoChamber as app
from tools.run_neural_mock import (
    ConditionResult,
    _HeadlessCore,
    _UnavailableEsn,
    _compare_pair,
    _core_feature_block,
)

from mock_neural import (
    ElectrodeArtifactParameters,
    NeuralMassParameters,
    SyntheticElectrodeObservation,
    TwoPopulationNeuralMass,
)


def actuator_pulse_trace(sample_rate_hz: int, total_samples: int, onset_sample: int) -> np.ndarray:
    """Render one pulse through the shared actuator's default AO waveform."""
    trace = np.zeros(total_samples, dtype=np.float64)
    actuator = PulseActuator(pulse_duration_ms=5.0, pulse_frequency_hz=100.0,
                             refractory_s=0.5)
    actuator.arm()
    result = actuator.process(
        ControllerAction(kind="pulse_events", pulse_sample_indices=(onset_sample,)),
        first_sample_index=onset_sample, sample_count=100, sample_rate_hz=sample_rate_hz,
        armed=True, validity_mask=np.ones(100, dtype=bool), amplitude_v=0.5, generation=1,
    )
    trace[onset_sample:onset_sample + result.waveform_v.size] = result.waveform_v
    return trace


class TwoPopulationNeuralMassTests(unittest.TestCase):
    def test_seeded_ei_state_generates_ripple_band_content_without_carrier(self) -> None:
        parameters = NeuralMassParameters()
        model = TwoPopulationNeuralMass(parameters, seed=7)
        rows = []
        for first in range(0, 4 * parameters.sample_rate_hz, 1_000):
            lfp, states = model.step_block(first, parameters.baseline_drive,
                                           np.zeros(1_000, dtype=np.float64))
            rows.append(lfp)
            self.assertLessEqual(np.max(np.abs(states["e_activity_deviation"])), 1.0)
            self.assertLessEqual(np.max(np.abs(states["i_activity_deviation"])), 1.0)
        output = np.concatenate(rows)
        settled = output[parameters.sample_rate_hz:]
        frequencies, power = welch(settled, fs=parameters.sample_rate_hz, nperseg=20_000)
        band = (frequencies >= 100.0) & (frequencies <= 250.0)
        peak_hz = float(frequencies[band][np.argmax(power[band])])
        non_dc = frequencies > 0.0
        global_peak_hz = float(frequencies[non_dc][np.argmax(power[non_dc])])
        ripple_fraction = float(np.sum(power[band]) / np.sum(power[non_dc]))
        self.assertGreaterEqual(peak_hz, 170.0)
        self.assertLessEqual(peak_hz, 230.0)
        self.assertGreaterEqual(global_peak_hz, 170.0)
        self.assertLessEqual(global_peak_hz, 230.0)
        self.assertGreater(ripple_fraction, 0.75)
        self.assertGreater(float(np.sqrt(np.mean(output * output))), 0.01)

    def test_state_and_output_are_invariant_to_block_partition(self) -> None:
        parameters = NeuralMassParameters()
        count = 20_000
        rng = np.random.default_rng(41)
        drives = 0.05 + rng.normal(0.0, 0.01, count)
        ao = actuator_pulse_trace(parameters.sample_rate_hz, count, 8_000)

        def run(block_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            model = TwoPopulationNeuralMass(parameters, seed=13)
            signal = []
            e_state = []
            effect = []
            for first in range(0, count, block_size):
                last = first + block_size
                lfp, states = model.step_block(first, drives[first:last], ao[first:last])
                signal.append(lfp)
                e_state.append(states["e_activity_deviation"])
                effect.append(states["inhibitory_effect_state"])
            return np.concatenate(signal), np.concatenate(e_state), np.concatenate(effect)

        one_block = run(count)
        small_blocks = run(100)
        for expected, actual in zip(one_block, small_blocks):
            np.testing.assert_array_equal(expected, actual)

    def test_only_generated_ao_changes_state_after_declared_delay(self) -> None:
        parameters = NeuralMassParameters(ao_effect_delay_ms=20.0)
        count = 80_000
        ao = actuator_pulse_trace(parameters.sample_rate_hz, count, 40_000)
        pulse_start = 40_000

        def run(generated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            model = TwoPopulationNeuralMass(parameters, seed=29)
            signal = []
            effects = []
            for first in range(0, count, 1_000):
                lfp, state = model.step_block(first, 0.05, generated[first:first + 1_000])
                signal.append(lfp)
                effects.append(state["inhibitory_effect_state"])
            return np.concatenate(signal), np.concatenate(effects)

        baseline = run(np.zeros(count, dtype=np.float64))
        stimulated = run(ao)
        delay_samples = round(parameters.ao_effect_delay_ms * parameters.sample_rate_hz / 1_000)
        generated_first_nonzero = int(np.flatnonzero(ao)[0])
        first_effect = int(np.flatnonzero(stimulated[1] > 0.0)[0])
        self.assertEqual(first_effect - generated_first_nonzero, delay_samples)
        np.testing.assert_array_equal(
            baseline[0][:pulse_start + delay_samples],
            stimulated[0][:pulse_start + delay_samples],
        )
        self.assertTrue(np.any(stimulated[1][pulse_start + delay_samples:] > 0.0))
        self.assertTrue(np.any(stimulated[0][pulse_start + delay_samples:]
                               != baseline[0][pulse_start + delay_samples:]))
        self.assertEqual(float(np.max(baseline[1])), 0.0)

    def test_half_step_converges_with_the_same_per_sample_noise_and_ao(self) -> None:
        parameters = NeuralMassParameters()
        count = 40_000
        drive_noise = np.random.default_rng(23).normal(0.0, 0.01, count)
        ao = actuator_pulse_trace(parameters.sample_rate_hz, count, 20_000)

        def run(substeps: int) -> np.ndarray:
            model = TwoPopulationNeuralMass(
                NeuralMassParameters(integration_substeps=substeps), seed=31
            )
            values = []
            for first in range(0, count, 100):
                end = first + 100
                lfp, _ = model.step_block(first, 0.05 + drive_noise[first:end], ao[first:end])
                values.append(lfp)
            return np.concatenate(values)

        base_dt = run(1)
        half_dt = run(2)
        relative_rms_difference = float(
            np.sqrt(np.mean((base_dt - half_dt) ** 2)) / np.sqrt(np.mean(base_dt ** 2))
        )
        self.assertLess(relative_rms_difference, 0.05)

    def test_generated_pulses_can_reduce_disturbed_state_derived_rms(self) -> None:
        parameters = NeuralMassParameters()
        sample_rate = parameters.sample_rate_hz
        count = 16 * sample_rate
        ao = actuator_pulse_trace(sample_rate, count, 10 * sample_rate)
        band_sos = butter(4, (100.0, 250.0), btype="bandpass", fs=sample_rate, output="sos")

        def run(generated_ao: np.ndarray) -> np.ndarray:
            model = TwoPopulationNeuralMass(parameters, seed=7)
            rows = []
            for first in range(0, count, 1_000):
                disturbance = 0.30 if 8 * sample_rate <= first < 14 * sample_rate else 0.05
                lfp, _ = model.step_block(first, disturbance, generated_ao[first:first + 1_000])
                rows.append(lfp)
            return sosfilt(band_sos, np.concatenate(rows))

        disabled = run(np.zeros(count, dtype=np.float64))
        matched = run(ao)

        def rms(values: np.ndarray, first_s: int, last_s: int) -> float:
            window = values[first_s * sample_rate:last_s * sample_rate]
            return float(np.sqrt(np.mean(window * window)))

        off_disturbance_rms = rms(disabled, 12, 14)
        pulse_disturbance_rms = rms(matched, 12, 14)
        self.assertGreater(off_disturbance_rms, 0.0)
        self.assertLess(pulse_disturbance_rms, off_disturbance_rms)

    def test_electrode_artifact_and_oracle_mask_are_separate_from_true_lfp(self) -> None:
        parameters = ElectrodeArtifactParameters(blanking_ms=25.0)
        observation = SyntheticElectrodeObservation(parameters)
        true_lfp = np.full(1_000, 0.2, dtype=np.float64)
        generated_ao = np.zeros(1_000, dtype=np.float64)
        generated_ao[100:200] = 0.5
        row = observation.observe(0, true_lfp, generated_ao)

        self.assertEqual(row["true_lfp_mV"][0], 0.2)
        self.assertEqual(row["electrode_mV"][0], 0.2)
        self.assertAlmostEqual(row["electrode_mV"][100],
                               row["true_lfp_mV"][100] + row["artifact_mV"][100])
        self.assertTrue(np.all(row["artifact_oracle_mask"][100:700]))
        self.assertFalse(np.any(row["artifact_oracle_mask"][700:]))
        self.assertFalse(np.array_equal(row["electrode_mV"], row["true_lfp_mV"]))

    def test_nonfinite_or_unbounded_ao_is_rejected_by_observation(self) -> None:
        truth = np.zeros(4, dtype=np.float64)
        for bad_ao in (np.asarray([0.0, np.nan, 0.0, 0.0]),
                       np.asarray([0.0, np.inf, 0.0, 0.0]),
                       np.asarray([0.0, 11.0, 0.0, 0.0])):
            with self.subTest(bad_ao=bad_ao):
                with self.assertRaises(ValueError):
                    SyntheticElectrodeObservation().observe(0, truth, bad_ao)

    def test_nonfinite_generated_io_and_discontinuous_indices_are_rejected(self) -> None:
        model = TwoPopulationNeuralMass(seed=1)
        with self.assertRaises(ValueError):
            model.step_block(0, 0.0, np.asarray([np.nan]))
        model.step_block(0, 0.0, np.zeros(10))
        with self.assertRaises(ValueError):
            model.step_block(11, 0.0, np.zeros(10))


class RuntimeClosureTests(unittest.TestCase):
    def test_cancelled_before_commit_never_reaches_model_or_sensor(self) -> None:
        settings = {"rule": app.controller_settings_for("rule"),
                    "pid": app.controller_settings_for("pid")}
        config = app.AppConfig(
            sample_rate=20_000, processing_block_ms=5.0, ao_target_lead_ms=100.0,
            controller="rule", controller_settings=settings, open_browser=False,
        )
        config.validate()
        state = app.RuntimeState(
            esn_ready=False, start_paused=False, electrode_labels=("CA3", "Cortex"),
            ctx_index=1, pulse_threshold_std=config.pulse_threshold_std,
            pulse_window_sec=config.pulse_window_sec, pulse_polarity=config.pulse_polarity,
            controller="rule", controller_ready={"esn": False, "rule": True, "pid": True},
            controller_settings=settings, input_mapping_locked=True,
        )
        scheduler = app.SimulatedAoScheduler(config, state)
        actuator = PulseActuator(pulse_duration_ms=5.0, pulse_frequency_hz=100.0,
                                 refractory_s=0.5)
        actuator.arm()
        proposal = actuator.process(
            ControllerAction(kind="pulse_events", pulse_sample_indices=(0,)),
            first_sample_index=0, sample_count=config.chunk_size,
            sample_rate_hz=config.sample_rate, armed=True,
            validity_mask=np.ones(config.chunk_size, dtype=bool),
            amplitude_v=0.5, generation=state.snapshot().generation,
            scheduler_lead_samples=config.ao_lead_samples, latest_sample_index=99,
            controller_id="rule",
        )
        safety = app.StimulusSafetyAdapter(config.safety, config.sample_rate)
        safe, safety_reason = safety.process(proposal.waveform_v)
        self.assertIsNone(safety_reason)
        self.assertTrue(proposal.pulse_plans)
        admitted_plan = proposal.pulse_plans[0]
        actuator.admit_safe_plan(admitted_plan.pulse_id, safe.reshape(-1))
        safe_plan = app.SafetyAdmittedPulsePlan(
            admitted_plan.pulse_id, admitted_plan.generation,
            admitted_plan.source_onset_sample_index,
            admitted_plan.target_onset_sample_index,
            admitted_plan.latest_sample_index, safe.reshape(-1).copy())
        self.assertTrue(scheduler.submit(
            config.ao_lead_chunks, safe.reshape(-1), state.snapshot().generation,
            proposal.pulse_events, (safe_plan,)))

        # A recoverable switch happens before the first AO refill can commit
        # this queued proposal. Only scheduler-generated blocks enter the model.
        state.begin_recoverable_drain()
        candidate_model = TwoPopulationNeuralMass(seed=53)
        zero_model = TwoPopulationNeuralMass(seed=53)
        observation = SyntheticElectrodeObservation()
        for first in range(0, 3_000, config.chunk_size):
            generated = scheduler.advance()
            self.assertEqual(generated.sample_index, first)
            self.assertFalse(np.any(generated.waveform_v))
            candidate_lfp, candidate_state = candidate_model.step_block(
                first, 0.05, generated.waveform_v)
            zero_lfp, zero_state = zero_model.step_block(
                first, 0.05, np.zeros(config.chunk_size, dtype=np.float64))
            observed = observation.observe(first, candidate_lfp, generated.waveform_v)
            np.testing.assert_array_equal(candidate_lfp, zero_lfp)
            np.testing.assert_array_equal(candidate_state["inhibitory_effect_state"],
                                          zero_state["inhibitory_effect_state"])
            self.assertFalse(np.any(observed["artifact_oracle_mask"]))
        self.assertEqual(scheduler.stats["pulse_committed_samples"], 0)
        self.assertGreater(scheduler.stats["cancelled_blocks"], 0)
        self.assertEqual(scheduler.stats["late_commands"], 0)

    def test_actual_core_skips_feature_mask_during_drain_then_recomputes(self) -> None:
        settings = {
            "rule": app.controller_settings_for(
                "rule", {"upper_z": 1.0, "lower_z": 0.5, "band_hz": (100.0, 250.0)}),
            "pid": app.controller_settings_for("pid"),
        }
        config = app.AppConfig(
            sample_rate=20_000, processing_block_ms=5.0, ao_target_lead_ms=100.0,
            controller="rule", controller_settings=settings, open_browser=False,
        )
        config.validate()
        state = app.RuntimeState(
            esn_ready=False, start_paused=False, electrode_labels=("CA3", "Cortex"),
            ctx_index=1, pulse_threshold_std=config.pulse_threshold_std,
            pulse_window_sec=config.pulse_window_sec, pulse_polarity=config.pulse_polarity,
            controller="rule", controller_ready={"esn": False, "rule": True, "pid": True},
            controller_settings=settings, input_mapping_locked=True,
        )
        loop = asyncio.new_event_loop()
        core = _HeadlessCore(
            config, state, _UnavailableEsn(), None,
            app.UiHub(config.ui_queue_packets), loop, app.Telemetry())
        scheduler = app.SimulatedAoScheduler(config, state)
        core.set_local_account_sink(scheduler.enqueue_local_account)
        state.set_mode("closed-loop")
        saw_intentional_skip = False
        saw_computed_mask = False
        try:
            for first in range(0, 5_000, config.chunk_size):
                generated = scheduler.advance()
                self.assertEqual(generated.sample_index, first)
                before = state.snapshot()
                _, safe, _, _ = core.process(
                    np.zeros((2, config.chunk_size), dtype=np.float64),
                    sample_index=first,
                    ao_target_sample_index=first + config.ao_lead_samples,
                    input_is_electrode_mV=True,
                    input_roles=("CA3", "Cortex"),
                    input_labels=("CA3", "Cortex"), input_ctx_index=1,
                )
                mask, computed, reason = _core_feature_block(core, before, config.chunk_size)
                if before.transition_pending:
                    saw_intentional_skip = True
                    self.assertFalse(computed)
                    self.assertEqual(reason, "transition_pending")
                    self.assertFalse(np.any(mask))
                elif computed:
                    saw_computed_mask = True
                    self.assertEqual(mask.size, config.chunk_size)
                accepted = scheduler.submit(
                    first // config.chunk_size + config.ao_lead_chunks,
                    safe[0], core.last_processed_generation,
                    core.last_pulse_events, core.last_admitted_pulse_plans,
                )
                self.assertTrue(accepted)
                if computed:
                    break
            self.assertTrue(saw_intentional_skip)
            self.assertTrue(saw_computed_mask)
            state.begin_recoverable_drain()
            with core._actuator_lock:
                core.actuator.disarm()
            for _ in range(config.ao_lead_chunks + 8):
                snapshot = state.snapshot()
                if (not snapshot.transition_pending
                        and scheduler.drain_barrier.pending_generation is None
                        and scheduler.stats_snapshot()["queue_depth"] == 0):
                    break
                scheduler.advance()
            self.assertFalse(state.snapshot().transition_pending)
            self.assertIsNone(scheduler.drain_barrier.pending_generation)
            self.assertEqual(scheduler.stats_snapshot()["queue_depth"], 0)
            self.assertIsNone(state.snapshot().fault)
        finally:
            loop.close()

    def test_artifact_only_pair_change_cannot_satisfy_neural_response_gate(self) -> None:
        samples = 1_000
        block_starts = np.arange(0, samples, 100, dtype=np.uint64)
        baseline = np.zeros(samples, dtype=np.float32)
        observed_off = baseline.copy()
        observed_rule = baseline.copy()
        observed_rule[100:200] = 1.0  # sensor artifact only
        z_off = np.zeros(10, dtype=np.float32)
        z_rule = z_off.copy()
        z_rule[8:] = 1.0  # a later valid measurement difference without latent response
        common = {
            "controller_source_block_start": block_starts,
            "controller_last_sample_feature_valid": np.ones(10, dtype=bool),
            "controller_z_score": z_off,
            "true_lfp_mV": baseline,
            "observed_ca3_mV": observed_off,
            "generated_ao_V": np.zeros(samples, dtype=np.float32),
        }
        off = ConditionResult("off", Path("off.h5"), Path("off.npz"),
                              {"h5_electrode_samples": samples, "chunk_size": 100},
                              dict(common))
        rule_arrays = dict(common)
        rule_arrays.update({"observed_ca3_mV": observed_rule,
                            "generated_ao_V": np.r_[np.zeros(100), np.full(100, 0.5),
                                                       np.zeros(800)].astype(np.float32),
                            "controller_z_score": z_rule})
        rule = ConditionResult(
            "rule", Path("rule.h5"), Path("rule.npz"),
            {"h5_electrode_samples": samples, "chunk_size": 100,
             "first_neural_effect_sample_index": 500,
             "configured_neural_effect_delay_samples": 400}, rule_arrays)
        paired = _compare_pair(off, rule)
        self.assertIsNotNone(paired["first_observed_ca3_divergence_sample_index"])
        self.assertIsNotNone(paired["first_later_valid_rule_measurement_change_sample_index"])
        self.assertFalse(paired["latent_divergence_after_neural_effect"])
        self.assertFalse(
            paired["no_latent_divergence_before_neural_effect"]
            and paired["latent_divergence_after_neural_effect"])


if __name__ == "__main__":
    unittest.main()
