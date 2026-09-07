from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.signal import lfilter
from scipy.io import savemat

import echoChamber as app
from esn.artifact import load_inference_artifact
from esn.inference import StatefulESN
from esn.preprocessing import StreamingFIR
from esn.train import pair_recordings
from esn import train as training
from esn_bridge import EchoChamberEsnBridge
from tests.test_esn_bridge import FakeOriginalStreamer


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "esn_legacy_compat_v1.npz"
CORRECTED_ARTIFACT = ROOT / "artifacts" / "esn_corrected_v1.npz"


class InferenceArtifactTests(unittest.TestCase):
    def test_numeric_artifact_loads_without_pickle(self) -> None:
        artifact = load_inference_artifact(ARTIFACT)
        self.assertEqual(artifact.W.shape, (10, 10))
        self.assertFalse(artifact.manifest["production_eligible"])

    def test_corrected_artifact_is_the_validated_runtime_default(self) -> None:
        artifact = load_inference_artifact(CORRECTED_ARTIFACT)
        self.assertTrue(artifact.manifest["production_eligible"])
        self.assertTrue(artifact.manifest["validation"]["passed"])
        self.assertEqual(app.ESN_ARTIFACT, CORRECTED_ARTIFACT)

    def test_runtime_rejects_pickle_and_checksum_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be .npz"):
            load_inference_artifact(ROOT / "esn_artifact.pkl")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.npz"
            manifest = target.with_suffix(".json")
            target.write_bytes(ARTIFACT.read_bytes() + b"tampered")
            manifest.write_text(
                (ARTIFACT.with_suffix(".json")).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_inference_artifact(target)

    def test_numpy_recurrence_is_chunk_invariant(self) -> None:
        artifact = load_inference_artifact(ARTIFACT)
        values = np.random.default_rng(12).normal(size=(1000, 1))
        complete = StatefulESN(artifact, "numpy").run(values)
        chunked_engine = StatefulESN(artifact, "numpy")
        chunked = np.vstack((chunked_engine.run(values[:137]), chunked_engine.run(values[137:])))
        np.testing.assert_array_equal(complete, chunked)


class PreprocessingTests(unittest.TestCase):
    def test_streaming_fir_matches_zero_state_offline_filter(self) -> None:
        coefficients = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
        values = np.r_[1.0, np.zeros(19)]
        expected = lfilter(coefficients, [1.0], values)
        streaming = StreamingFIR(coefficients)
        actual = np.r_[streaming.process(values[:7]), streaming.process(values[7:])]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[0], coefficients[0])


class TriggerAndSafetyTests(unittest.TestCase):
    @staticmethod
    def _warmed_bridge(polarity: str) -> EchoChamberEsnBridge:
        bridge = EchoChamberEsnBridge(
            FakeOriginalStreamer(),
            runtime_chunk_size=400,
            sample_rate=20_000,
            passthrough_dc_block_hz=0.0,
            pulse_threshold_std=1.0,
            pulse_window_sec=1.0,
            pulse_polarity=polarity,
        )
        bridge.configure(stim_mode="threshold_pulse", stim_gain=1.0)
        baseline = np.zeros((2, 400), dtype=np.float64)
        for _ in range(50):
            bridge.process(baseline, ctx_index=1)
        return bridge

    def test_negative_polarity_and_cross_block_pulse_continuation(self) -> None:
        bridge = self._warmed_bridge("negative")
        event = np.zeros((2, 400), dtype=np.float64)
        event[1, 390] = -1.0
        first = bridge.process(event, ctx_index=1)
        _, threshold, peak, fired = bridge.diagnostics(400)
        self.assertTrue(fired)
        self.assertGreater(peak, threshold)
        self.assertTrue(np.any(first[0, 391:] != 0))
        second = bridge.process(np.zeros_like(event), ctx_index=1)
        self.assertTrue(np.any(second[0, :90] != 0))
        self.assertTrue(np.all(second[0, 90:] == 0))

    def test_positive_polarity_ignores_negative_event(self) -> None:
        bridge = self._warmed_bridge("positive")
        event = np.zeros((2, 400), dtype=np.float64)
        event[1, 100] = -1.0
        command = bridge.process(event, ctx_index=1)
        self.assertFalse(bridge.diagnostics(400)[3])
        np.testing.assert_array_equal(command, np.zeros_like(command))

    def test_runtime_requires_explicit_arming(self) -> None:
        state = app.RuntimeState(
            esn_ready=True,
            start_paused=False,
            electrode_labels=("CA3", "Cortex"),
            ctx_index=1,
            pulse_threshold_std=3.0,
            pulse_window_sec=10.0,
        )
        state.set_mode("closed-loop")
        with self.assertRaisesRegex(ValueError, "non-zero"):
            state.set_armed(True)
        state.set_stim("threshold_pulse", 1.0)
        state.set_armed(True)
        self.assertTrue(state.snapshot().armed)
        state.fault("test")
        self.assertFalse(state.snapshot().armed)

    def test_inert_duty_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "active_threshold"):
            app.SafetyConfig(max_command_v=1.0, active_threshold_v=3.5).validate()


class TrainingManifestTests(unittest.TestCase):
    def test_missing_validation_fs_uses_explicit_model_rate_assumption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.mat"
            savemat(path, {"data": np.arange(20, dtype=np.float64)})
            values, sample_rate = training._load_mat_signal(
                path, missing_fs_hz=2_000
            )
            np.testing.assert_array_equal(values, np.arange(20, dtype=np.float64))
            self.assertEqual(sample_rate, 2_000)

    def test_pairing_is_by_electrode_rank_within_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ctx = root / "CTX"
            ca3 = root / "CA3"
            ctx.mkdir()
            ca3.mkdir()
            for name in ("20200226_slice04_01_CTRL1_17.mat", "20200226_slice04_01_CTRL1_16.mat"):
                (ctx / name).touch()
            for name in ("20200226_slice04_01_CTRL1_65.mat", "20200226_slice04_01_CTRL1_27.mat"):
                (ca3 / name).touch()
            pairs = pair_recordings(ctx, ca3, role="training")
            self.assertEqual(
                [pair.recording_id for pair in pairs],
                [
                    "training__20200226_slice04_01__ctx16__ca327",
                    "training__20200226_slice04_01__ctx17__ca365",
                ],
            )

    def test_legacy_validation_uses_legacy_scalers_in_physical_units(self) -> None:
        artifact = load_inference_artifact(ARTIFACT)
        pair = training.RecordingPair("held-out", Path("ctx.mat"), Path("ca3.mat"))
        ctx = np.linspace(-2.0, 2.0, 50)
        ca3 = np.linspace(0.5, -0.5, 50)
        corrected_input = training.AffineScaler.from_range(-20.0, 20.0)
        corrected_target = training.AffineScaler.from_range(-10.0, 10.0)
        legacy_input = ctx * artifact.input_scale[0] + artifact.input_offset[0]
        legacy_scaled = training._predict_scaled(
            legacy_input, artifact, artifact.Wout, artifact.readout_bias
        )
        legacy_physical = (
            legacy_scaled - artifact.target_offset[0]
        ) / artifact.target_scale[0]
        expected = training._metrics(ca3, legacy_physical, 0)
        with mock.patch.object(training, "_pair_values", return_value=(ctx, ca3)):
            result = training.validate_readout(
                [pair], artifact, corrected_input, corrected_target,
                artifact.Wout, artifact.readout_bias, washout_seconds=0.0,
            )
        actual = result["recordings"][0]["legacy_readout"]
        self.assertAlmostEqual(actual["pearson_r"], expected["pearson_r"])
        self.assertAlmostEqual(actual["nrmse"], expected["nrmse"])


class DeploymentReadinessTests(unittest.TestCase):
    def test_unvalidated_hardware_profile_is_rejected(self) -> None:
        profile = ROOT / "hardware_profiles" / "example_unvalidated.json"
        with self.assertRaisesRegex(ValueError, "validated or explicitly marked"):
            app.load_hardware_profile(profile)

    def test_confirmed_pending_test_profile_is_accepted(self) -> None:
        profile = app.load_hardware_profile(
            ROOT / "hardware_profiles" / "lab_pending_test.json"
        )
        self.assertEqual(profile["validation_status"], "confirmed_pending_test")

    def test_default_real_configuration_is_ready_with_pending_profile(self) -> None:
        config = app.config_from_args(app.build_parser().parse_args([]))
        config.validate()
        self.assertEqual(config.hardware_profile, app.DEFAULT_HARDWARE_PROFILE)
        self.assertEqual(config.model_artifact, CORRECTED_ARTIFACT)

    def test_default_processing_block_is_five_milliseconds(self) -> None:
        self.assertEqual(app.AppConfig().processing_block_ms, 5.0)

    def test_real_hardware_allows_logged_cli_calibration_override(self) -> None:
        profile = {
            "format_tag": "echo-chamber-hardware-profile-v1",
            "validated_for_real_hardware": True,
            "electrode_labels": ["CA3", "Cortex"],
            "ctx_index": 1,
            "amplifier_gain": [10.0, 10.0],
            "ao_command_gain_v_per_esn_unit": 0.5,
            "safety": {
                "max_command_v": 5.0,
                "max_slew_v_per_s": 2000.0,
                "max_abs_area_v_s": 0.01,
                "area_window_s": 1.0,
                "max_active_fraction": 0.1,
                "active_threshold_v": 3.5,
                "max_consecutive_active_s": 0.05,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            args = app.build_parser().parse_args(
                ["--hardware-profile", str(path), "--amplifier-gain", "20", "20"]
            )
            config = app.config_from_args(args)
            self.assertEqual(config.amplifier_gain, (20.0, 20.0))


if __name__ == "__main__":
    unittest.main()
