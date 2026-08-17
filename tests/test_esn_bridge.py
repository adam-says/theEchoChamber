from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from esn_bridge import EchoChamberEsnBridge


@dataclass
class FakePulseConfig:
    min_interval_sec: float = 0.5
    pulse_duration_ms: float = 5.0
    pulse_freq_hz: float = 100.0
    waveform: str = "sine"


class FakeOriginalStreamer:
    """Strict 100-sample stand-in for the collaborator ESN interface."""

    chunk_size = 100
    fs_in = 20_000
    fs_train = 2_000
    decim_q = 10
    stim_clip_v = (-10.0, 10.0)

    def __init__(self) -> None:
        self.pulse_cfg = FakePulseConfig()
        self.calls = 0
        self.resets = 0
        self.configured: dict[str, object] = {}

    def configure(self, **kwargs: object) -> None:
        self.configured.update(kwargs)

    def reset(self) -> None:
        self.resets += 1

    def process_chunk(self, values: np.ndarray, *, ctx_index: int) -> np.ndarray:
        if values.shape[1] != self.chunk_size:
            raise ValueError("fake ESN accepts exactly 100 samples")
        self.calls += 1
        return values[ctx_index:ctx_index + 1].copy()


class EchoChamberEsnBridgeTests(unittest.TestCase):
    def make_bridge(self, **kwargs: object) -> tuple[FakeOriginalStreamer, EchoChamberEsnBridge]:
        streamer = FakeOriginalStreamer()
        bridge = EchoChamberEsnBridge(
            streamer,
            runtime_chunk_size=400,
            sample_rate=20_000,
            passthrough_dc_block_hz=0.0,
            **kwargs,
        )
        return streamer, bridge

    def test_adapts_one_runtime_block_to_four_artifact_calls(self) -> None:
        streamer, bridge = self.make_bridge()
        values = np.vstack((np.linspace(-0.5, 0.5, 400), np.linspace(-1.0, 1.0, 400)))
        result = bridge.process(values, ctx_index=1)
        np.testing.assert_array_equal(result, values[1:2])
        self.assertEqual(streamer.calls, 4)
        self.assertEqual(streamer.configured, {"stim_mode": "passthrough", "stim_gain": 1.0})
        self.assertEqual(streamer.stim_clip_v, (-float("inf"), float("inf")))
        self.assertEqual(bridge.stim_clip_v, (-10.0, 10.0))

    def test_off_mode_preserves_prediction_diagnostics_but_returns_zero(self) -> None:
        _, bridge = self.make_bridge()
        bridge.configure(stim_mode="off")
        values = np.vstack((np.zeros(400), np.linspace(-1.0, 1.0, 400)))
        result = bridge.process(values, ctx_index=1)
        model, _, peak, fired = bridge.diagnostics(400)
        np.testing.assert_array_equal(result, np.zeros((1, 400)))
        np.testing.assert_allclose(model, values[1:2])
        self.assertTrue(np.isfinite(peak))
        self.assertFalse(fired)

    def test_explicit_ao_command_gain_scales_passthrough_to_volts(self) -> None:
        _, bridge = self.make_bridge(ao_command_gain_v_per_esn_unit=0.25)
        values = np.vstack((np.zeros(400), np.full(400, 2.0)))
        np.testing.assert_allclose(bridge.process(values, ctx_index=1), 0.5)

    def test_threshold_pulse_is_application_owned(self) -> None:
        _, bridge = self.make_bridge(pulse_threshold_std=1.0, pulse_window_sec=1.0)
        bridge.configure(stim_mode="threshold_pulse", stim_gain=0.5)
        baseline = np.zeros((2, 400), dtype=np.float64)
        for _ in range(50):  # one second at the bridge's 2 kHz decision rate
            bridge.process(baseline, ctx_index=1)
        event = baseline.copy()
        event[1, 100] = 1.0
        command = bridge.process(event, ctx_index=1)
        _, threshold, peak, fired = bridge.diagnostics(400)
        self.assertTrue(fired)
        self.assertGreater(peak, threshold)
        self.assertGreater(np.max(np.abs(command)), 0.0)

    def test_reset_clears_application_and_esn_state(self) -> None:
        streamer, bridge = self.make_bridge()
        bridge.process(np.ones((2, 400)), ctx_index=1)
        bridge.reset()
        model, threshold, peak, fired = bridge.diagnostics(400)
        self.assertEqual(streamer.resets, 1)
        self.assertTrue(np.all(np.isnan(model)))
        self.assertTrue(np.isnan(threshold))
        self.assertTrue(np.isnan(peak))
        self.assertFalse(fired)

    def test_rejects_incompatible_runtime_chunk(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a multiple"):
            EchoChamberEsnBridge(
                FakeOriginalStreamer(), runtime_chunk_size=250, sample_rate=20_000
            )

    def test_rejects_sample_rate_different_from_artifact(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit resampling is required"):
            EchoChamberEsnBridge(
                FakeOriginalStreamer(), runtime_chunk_size=100, sample_rate=10_000
            )


if __name__ == "__main__":
    unittest.main()
