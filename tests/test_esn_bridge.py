from __future__ import annotations

import math
import unittest
from collections import deque
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


def _old_threshold_step(history: deque[float], samples_since_pulse: int, previous: float | None,
                        bridge: EchoChamberEsnBridge, conditioned: np.ndarray):
    """Shadow oracle for the pre-optimization rolling threshold implementation."""
    values = np.asarray(conditioned[0, ::bridge.decim_q], dtype=np.float64)
    if bridge.pulse_polarity == "positive":
        metric = values
    elif bridge.pulse_polarity == "negative":
        metric = -values
    else:
        metric = np.abs(values)
    warmup = max(5, int(min(1.0, bridge.pulse_window_sec) * bridge.model_rate))
    threshold = float("nan")
    fired = False
    trigger_sample = None
    samples_since_pulse += values.size
    if len(history) >= warmup:
        recent = np.fromiter(history, dtype=np.float64)
        threshold = float(np.mean(recent) + bridge.pulse_threshold_std * (np.std(recent) + 1e-12))
        crossings = np.flatnonzero(
            (metric > threshold)
            & np.r_[previous is None or previous <= threshold, metric[:-1] <= threshold]
        )
        interval = int(bridge.pulse_min_interval_sec * bridge.model_rate)
        if crossings.size and samples_since_pulse >= interval:
            fired = True
            trigger_sample = int(crossings[0]) * bridge.decim_q
            samples_since_pulse = 0
    previous = float(metric[-1]) if metric.size else previous
    if not fired:
        history.extend(float(value) for value in metric)
    return threshold, float(np.max(metric)), fired, trigger_sample, samples_since_pulse, previous


def _metric_block(values: np.ndarray, decim_q: int) -> np.ndarray:
    block = np.zeros((1, values.size * decim_q), dtype=np.float64)
    block[0, ::decim_q] = values
    return block


class IncrementalPulseStatsTests(unittest.TestCase):
    def make_bridge(self, **kwargs: object) -> tuple[FakeOriginalStreamer, EchoChamberEsnBridge]:
        return EchoChamberEsnBridgeTests().make_bridge(**kwargs)

    def _compare_stream(self, bridge: EchoChamberEsnBridge, blocks: list[np.ndarray],
                        *, resize_after: dict[int, float] | None = None) -> None:
        history: deque[float] = deque(bridge._recent, maxlen=bridge._recent.maxlen)
        samples_since_pulse = bridge._samples_since_pulse
        previous = bridge._previous_pulse_metric
        for block_index, conditioned in enumerate(blocks):
            if resize_after and block_index in resize_after:
                seconds = resize_after[block_index]
                bridge.configure(pulse_window_sec=seconds)
                history = deque(history, maxlen=max(1, int(seconds * bridge.model_rate)))
            expected = _old_threshold_step(history, samples_since_pulse, previous, bridge, conditioned)
            actual = bridge._update_pulse_state(conditioned)
            threshold, peak, fired, trigger = actual
            threshold_atol = (max(2e-12, 4 * abs(float(np.spacing(expected[0]))))
                              if np.isfinite(expected[0]) else 2e-12)
            np.testing.assert_allclose(threshold, expected[0], rtol=0, atol=threshold_atol, equal_nan=True)
            self.assertEqual(peak, expected[1])
            self.assertEqual(fired, expected[2])
            self.assertEqual(trigger, expected[3])
            samples_since_pulse, previous = expected[4:]
            self.assertEqual(bridge._samples_since_pulse, samples_since_pulse)
            self.assertEqual(bridge._previous_pulse_metric, previous)
            self.assertEqual(len(bridge._recent), len(history))
            np.testing.assert_array_equal(np.fromiter(bridge._recent, dtype=np.float64),
                                          np.fromiter(history, dtype=np.float64))
            self.assertEqual(bridge._recent_count, len(bridge._recent))
            if history:
                recent_values = np.fromiter(history, dtype=np.float64)
                # Compare centered moments against an anchored high-accuracy
                # reference. np.std(high_offset_values) recomputes deviations
                # from its rounded full-scale np.mean and is the separate old
                # threshold oracle checked above, not a true centered variance.
                anchor = float(recent_values[-1])
                centered = [float(value) - anchor for value in recent_values]
                centered_mean = math.fsum(centered) / len(centered)
                running_mean = bridge._recent_origin + bridge._recent_mean_delta
                centered_std = math.sqrt(math.fsum(
                    (value - centered_mean) ** 2 for value in centered
                ) / len(centered))
                # The public old-threshold comparator above checks mean-derived
                # decisions. Here the internal invariant specifically checks
                # variance against a centered reference, independent of the
                # old full-scale NumPy mean rounding.
                self.assertTrue(np.isfinite(running_mean))
                np.testing.assert_allclose(np.sqrt(max(0.0, bridge._recent_m2 / len(history))),
                                           centered_std, rtol=2e-10, atol=2e-12)

    @staticmethod
    def _assert_peak_queue_matches_window(test: unittest.TestCase,
                                          bridge: EchoChamberEsnBridge) -> None:
        expected: list[float] = []
        for value in bridge._recent:
            magnitude = abs(float(value))
            while expected and expected[-1] <= magnitude:
                expected.pop()
            expected.append(magnitude)
        actual = [magnitude for _, magnitude in bridge._recent_peak_abs_values]
        test.assertEqual(actual, expected)
        sequences = [sequence for sequence, _ in bridge._recent_peak_abs_values]
        test.assertEqual(sequences, sorted(sequences))

    def test_moment_only_rebase_preserves_peak_queue_sequence_and_future_expiry(self) -> None:
        _, bridge = EchoChamberEsnBridgeTests().make_bridge(pulse_window_sec=1.0)
        _, full_rebuild_reference = EchoChamberEsnBridgeTests().make_bridge(pulse_window_sec=1.0)
        bridge._recent = deque(maxlen=5)
        for value in (5.0, 1.0, 5.0, 0.0, 0.0, 2.0):
            bridge._append_recent(value)

        live_queue = tuple(bridge._recent_peak_abs_values)
        live_sequence = bridge._recent_sequence
        live_highwater = bridge._recent_scale_highwater
        rebuilds_before = bridge.rolling_stat_rebuild_count
        bridge._rebuild_recent_moments()

        self.assertEqual(tuple(bridge._recent_peak_abs_values), live_queue)
        self.assertEqual(bridge._recent_sequence, live_sequence)
        self.assertEqual(bridge._recent_scale_highwater, live_highwater)
        self.assertEqual(bridge.rolling_stat_rebuild_count, rebuilds_before + 1)
        self._assert_peak_queue_matches_window(self, bridge)

        # Compare subsequent equal-peak and zero-tail expiry against the former
        # full-rebuild behavior; both queues must remain exact as old maxima leave.
        full_rebuild_reference._recent = deque(bridge._recent, maxlen=5)
        full_rebuild_reference._rebuild_recent_statistics()
        for value in (0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0):
            bridge._append_recent(value)
            full_rebuild_reference._append_recent(value)
            np.testing.assert_array_equal(np.fromiter(bridge._recent, dtype=np.float64),
                                          np.fromiter(full_rebuild_reference._recent, dtype=np.float64))
            self._assert_peak_queue_matches_window(self, bridge)
            self._assert_peak_queue_matches_window(self, full_rebuild_reference)
            self.assertEqual(
                [magnitude for _, magnitude in bridge._recent_peak_abs_values],
                [magnitude for _, magnitude in full_rebuild_reference._recent_peak_abs_values],
            )
            self.assertEqual(bridge._recent_scale_highwater,
                             full_rebuild_reference._recent_scale_highwater)

    def test_scale_contraction_rebases_moments_and_resets_only_live_highwater(self) -> None:
        for values, expected_peak in (
            ((0.0, 0.002, 0.002, 0.001, 0.0), 0.002),
            ((0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
        ):
            with self.subTest(expected_peak=expected_peak):
                _, bridge = EchoChamberEsnBridgeTests().make_bridge(pulse_window_sec=1.0)
                bridge._recent = deque(maxlen=5)
                for value in values:
                    bridge._append_recent(value)
                queue_before = tuple(bridge._recent_peak_abs_values)
                sequence_before = bridge._recent_sequence
                bridge._recent_scale_highwater = 100.0
                rebuilds_before = bridge.rolling_stat_rebuild_count

                self.assertTrue(bridge._maybe_rebuild_after_scale_contraction(warmup=5))

                self.assertEqual(bridge.rolling_stat_rebuild_count, rebuilds_before + 1)
                self.assertEqual(tuple(bridge._recent_peak_abs_values), queue_before)
                self.assertEqual(bridge._recent_sequence, sequence_before)
                self.assertEqual(bridge._recent_scale_highwater, expected_peak)
                self._assert_peak_queue_matches_window(self, bridge)
                np.testing.assert_allclose(
                    bridge._recent_origin + bridge._recent_mean_delta,
                    np.mean(np.asarray(values, dtype=np.float64)), rtol=0, atol=1e-15,
                )

    def test_drift_rebase_preserves_updated_sequence_and_resets_highwater(self) -> None:
        _, bridge = EchoChamberEsnBridgeTests().make_bridge(pulse_window_sec=1.0)
        bridge._recent = deque(maxlen=4)
        for value in (1e9, 1e9, 1e9, 1e9):
            bridge._append_recent(value)
        bridge._rebuild_recent_statistics()
        bridge._recent_origin = 0.0
        bridge._recent_mean_delta = 1e9
        bridge._recent_m2 = 0.0
        bridge._recent_scale_highwater = 1e12
        rebuilds_before = bridge.rolling_stat_rebuild_count

        bridge._append_recent(1e9)

        self.assertEqual(bridge.rolling_stat_rebuild_count, rebuilds_before + 1)
        self.assertEqual(bridge._recent_sequence, 5)
        self.assertEqual(bridge._recent_scale_highwater,
                         bridge._recent_peak_abs_values[0][1])
        self._assert_peak_queue_matches_window(self, bridge)
        self.assertAlmostEqual(bridge._recent_origin + bridge._recent_mean_delta,
                               float(np.mean(np.fromiter(bridge._recent, dtype=np.float64))))

    def test_configured_window_resize_keeps_full_peak_rebuild_semantics(self) -> None:
        _, bridge = EchoChamberEsnBridgeTests().make_bridge(pulse_window_sec=1.0)
        bridge._recent = deque(maxlen=5)
        for value in (3.0, 3.0, 0.0, 1.0, 0.0, 2.0):
            bridge._append_recent(value)

        bridge.configure(pulse_window_sec=2.0)

        self.assertEqual(bridge._recent_sequence, len(bridge._recent))
        self.assertEqual(bridge._recent_scale_highwater,
                         max(abs(float(value)) for value in bridge._recent))
        self._assert_peak_queue_matches_window(self, bridge)

    def test_incremental_stats_match_old_numpy_path_through_wrap_resize_and_reset(self) -> None:
        rng = np.random.default_rng(2918)
        for polarity in ("absolute", "positive", "negative"):
            with self.subTest(polarity=polarity):
                _, bridge = EchoChamberEsnBridgeTests().make_bridge(pulse_threshold_std=1.5,
                                                                    pulse_window_sec=1.0)
                bridge.configure(stim_mode="threshold_pulse", pulse_polarity=polarity)
                values = 0.1 + rng.normal(0.0, .02, 40 * 360)
                for start in range(300, values.size, 900):
                    values[start:start + 40] += .18
                blocks = [_metric_block(values[i:i + 40], bridge.decim_q)
                          for i in range(0, values.size, 40)]
                self._compare_stream(bridge, blocks[:100], resize_after={70: 2.0})
                bridge.reset()
                self._compare_stream(bridge, blocks[100:], resize_after={160: 1.0})

    def test_centered_statistics_handle_flat_and_high_offset_low_noise_windows(self) -> None:
        rng = np.random.default_rng(2901)
        for base, noise in ((0.0, 0.0), (1e12, .002)):
            with self.subTest(base=base, noise=noise):
                _, bridge = EchoChamberEsnBridgeTests().make_bridge(pulse_threshold_std=3.0,
                                                                    pulse_window_sec=1.0)
                bridge.pulse_min_interval_sec = .5
                values = base + rng.normal(0.0, noise, 40 * 180)
                blocks = [_metric_block(values[i:i + 40], bridge.decim_q)
                          for i in range(0, values.size, 40)]
                self._compare_stream(bridge, blocks)
                self.assertTrue(np.isfinite(bridge._recent_mean_delta))
                self.assertTrue(np.isfinite(bridge._recent_m2))
                if base == 0.0:
                    self.assertLess(bridge.exact_threshold_fallback_count, len(blocks) // 10)
                else:
                    # The original NumPy statistic is intentionally used at
                    # extreme offsets where centered-moment uncertainty can
                    # consume the four-ULP diagnostic budget.
                    self.assertGreater(bridge.exact_threshold_fallback_count, 0)

    def test_recenter_after_large_offset_leaves_window_and_flat_tail_arrives(self) -> None:
        rng = np.random.default_rng(2902)
        _, bridge = EchoChamberEsnBridgeTests().make_bridge(pulse_threshold_std=20.0,
                                                            pulse_window_sec=1.0)
        high = 1e12 + rng.normal(0.0, .002, 40 * 80)
        low_tail = 3.0 + rng.normal(0.0, 1e-4, 40 * 180)
        blocks = [_metric_block(values, bridge.decim_q)
                  for values in (high[i:i + 40] for i in range(0, high.size, 40))]
        blocks.extend(_metric_block(values, bridge.decim_q)
                      for values in (low_tail[i:i + 40] for i in range(0, low_tail.size, 40)))
        before = bridge.rolling_stat_rebuild_count
        self._compare_stream(bridge, blocks)
        rebuilds = bridge.rolling_stat_rebuild_count - before
        self.assertGreaterEqual(rebuilds, 1)
        self.assertLessEqual(rebuilds, 3)  # level-change repair is rare, not per-block O(window)
        self.assertAlmostEqual(bridge._recent_origin + bridge._recent_mean_delta,
                               float(np.mean(low_tail[-2000:])), delta=1e-10)
        self.assertAlmostEqual(np.sqrt(max(0.0, bridge._recent_m2 / bridge._recent_count)),
                               float(np.std(low_tail[-2000:])), delta=1e-10)

    def test_startup_excursion_residual_is_repaired_after_scale_contracts(self) -> None:
        _, bridge = EchoChamberEsnBridgeTests().make_bridge(
            pulse_threshold_std=3.0, pulse_window_sec=10.0
        )
        window_samples = int(bridge.pulse_window_sec * bridge.model_rate)
        startup = np.tile(np.array([0.0, 2_000.0]), window_samples // 2)
        bridge._recent.extend(float(value) for value in startup)
        bridge._rebuild_recent_statistics()
        initial_rebuilds = bridge.rolling_stat_rebuild_count

        # A large startup excursion leaves the window, followed by a small,
        # noisy tail like the failing real recording. Compare each threshold
        # to the original NumPy oracle while the transient contracts.
        rng = np.random.default_rng(9182)
        tail = 4e-4 + rng.normal(0.0, 2e-4, window_samples + 500)
        blocks = [_metric_block(tail[start:start + 100], bridge.decim_q)
                  for start in range(0, tail.size, 100)]
        history = deque(bridge._recent, maxlen=bridge._recent.maxlen)
        samples_since_pulse = bridge._samples_since_pulse
        previous = bridge._previous_pulse_metric
        for block in blocks:
            expected = _old_threshold_step(history, samples_since_pulse, previous, bridge, block)
            actual = bridge._update_pulse_state(block)
            tolerance = max(2e-12, 4 * abs(float(np.spacing(expected[0]))))
            np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=tolerance)
            self.assertEqual(actual[2:4], expected[2:4])
            samples_since_pulse, previous = expected[4:]

        self.assertGreater(bridge.rolling_stat_rebuild_count, initial_rebuilds)
        self.assertLess(bridge._recent_scale_highwater, 1e-2)
        self.assertLess(bridge._recent_peak_abs_values[0][1], 1e-2)

    def test_near_threshold_uses_exact_old_statistic_for_decision(self) -> None:
        rng = np.random.default_rng(118)
        _, bridge = EchoChamberEsnBridgeTests().make_bridge(pulse_threshold_std=1.0,
                                                            pulse_window_sec=1.0)
        bridge.pulse_min_interval_sec = 0.0
        history = rng.normal(0.0, 1.0, 2000)
        bridge._recent.extend(float(value) for value in history)
        bridge._rebuild_recent_statistics()
        bridge._samples_since_pulse = 10000
        exact_threshold = float(np.mean(history) + bridge.pulse_threshold_std * (np.std(history) + 1e-12))
        bridge._previous_pulse_metric = exact_threshold - 1.0
        block = _metric_block(np.r_[np.nextafter(exact_threshold, np.inf), np.zeros(39)], bridge.decim_q)
        exact_fallbacks_before = bridge.exact_threshold_fallback_count
        threshold, _, fired, trigger = bridge._update_pulse_state(block)
        self.assertEqual(bridge.exact_threshold_fallback_count, exact_fallbacks_before + 1)
        self.assertEqual(threshold, exact_threshold)
        self.assertTrue(fired)
        self.assertEqual(trigger, 0)

    def test_threshold_adjacent_decisions_after_long_rolling_level_shift(self) -> None:
        """Probe original threshold boundaries after 20k-sample eviction drift."""
        window_samples = 20_000  # 10 s at the bridge's 2 kHz decision rate
        values_per_block = 40
        checkpoints = (4_000, 12_000, 20_000, 40_000, 80_000, 100_000)
        excess_heuristic_cases = 0
        rng = np.random.default_rng(8_139)

        for polarity, signal_sign in (("positive", 1.0), ("negative", -1.0)):
            with self.subTest(polarity=polarity):
                _, bridge = EchoChamberEsnBridgeTests().make_bridge(
                    pulse_threshold_std=3.0, pulse_window_sec=10.0
                )
                bridge.configure(stim_mode="threshold_pulse", pulse_polarity=polarity)
                # Keep every stream block un-fired while allowing the actual
                # bounded deque and add/remove moments to advance naturally.
                bridge.pulse_min_interval_sec = 1_000_000.0
                bridge._samples_since_pulse = 0
                high = signal_sign * (1e12 + rng.normal(0.0, 0.002, window_samples + 4_000))
                tail = signal_sign * (7.0 + rng.normal(0.0, 1e-4, checkpoints[-1]))

                def probe_threshold_boundaries(source_bridge: EchoChamberEsnBridge) -> None:
                    nonlocal excess_heuristic_cases
                    history_values = np.fromiter(source_bridge._recent, dtype=np.float64)
                    exact_threshold = float(
                        np.mean(history_values)
                        + source_bridge.pulse_threshold_std * (np.std(history_values) + 1e-12)
                    )
                    running_mean = source_bridge._recent_origin + source_bridge._recent_mean_delta
                    running_std = float(np.sqrt(max(
                        0.0, source_bridge._recent_m2 / source_bridge._recent_count
                    )))
                    running_threshold = float(
                        running_mean + source_bridge.pulse_threshold_std * (running_std + 1e-12)
                    )
                    candidates = (
                        np.nextafter(exact_threshold, -np.inf),
                        exact_threshold,
                        np.nextafter(exact_threshold, np.inf),
                    )
                    metric_floor = min(7.0, exact_threshold - max(1.0, 10.0 * np.std(history_values)))
                    for candidate in candidates:
                        # Copy only the threshold state, avoiding mutation of
                        # the warmed source state by event exclusion/refractory.
                        _, probe = EchoChamberEsnBridgeTests().make_bridge(
                            pulse_threshold_std=source_bridge.pulse_threshold_std,
                            pulse_window_sec=source_bridge.pulse_window_sec,
                            pulse_polarity=polarity,
                        )
                        probe.configure(stim_mode="threshold_pulse", pulse_polarity=polarity)
                        probe._recent = deque(source_bridge._recent, maxlen=source_bridge._recent.maxlen)
                        probe._recent_count = source_bridge._recent_count
                        probe._recent_origin = source_bridge._recent_origin
                        probe._recent_mean_delta = source_bridge._recent_mean_delta
                        probe._recent_m2 = source_bridge._recent_m2
                        probe.pulse_min_interval_sec = 0.0
                        probe._samples_since_pulse = 10**9
                        probe._previous_pulse_metric = exact_threshold - max(
                            1.0, 4.0 * abs(float(np.spacing(exact_threshold)))
                        )

                        metric = np.full(values_per_block, metric_floor, dtype=np.float64)
                        metric[0] = candidate
                        raw = metric * signal_sign  # negative polarity maps back to a negative input
                        block = _metric_block(raw, probe.decim_q)
                        centered_peak = float(np.max(np.abs(metric - source_bridge._recent_origin)))
                        heuristic = (
                            16.0 * np.finfo(np.float64).eps * max(
                                1.0, abs(source_bridge._recent_mean_delta), running_std, centered_peak
                            )
                            + 2.0 * abs(float(np.spacing(running_threshold)))
                        )
                        old_history = deque(probe._recent, maxlen=probe._recent.maxlen)
                        expected = _old_threshold_step(
                            old_history, probe._samples_since_pulse, probe._previous_pulse_metric,
                            probe, block,
                        )
                        actual = probe._update_pulse_state(block)
                        if abs(running_threshold - exact_threshold) > heuristic:
                            excess_heuristic_cases += 1
                        diagnostic = (
                            f"heuristic-excess boundary probes seen: {excess_heuristic_cases}"
                        )
                        self.assertEqual(actual[2], expected[2], diagnostic)
                        self.assertEqual(actual[3], expected[3], diagnostic)
                        self.assertEqual(actual[2], bool(candidate > exact_threshold), diagnostic)
                        self.assertEqual(actual[3], 0 if candidate > exact_threshold else None,
                                         diagnostic)

                # Establish a large-offset baseline, step to a low-level tail,
                # then move through multiple full 20k-sample window turnovers.
                for start in range(0, high.size, values_per_block):
                    block = _metric_block(high[start:start + values_per_block], bridge.decim_q)
                    _, _, fired, _ = bridge._update_pulse_state(block)
                    self.assertFalse(fired)
                tail_seen = 0
                checkpoint_set = set(checkpoints)
                for start in range(0, tail.size, values_per_block):
                    block = _metric_block(tail[start:start + values_per_block], bridge.decim_q)
                    _, _, fired, _ = bridge._update_pulse_state(block)
                    self.assertFalse(fired)
                    tail_seen += values_per_block
                    if tail_seen in checkpoint_set:
                        probe_threshold_boundaries(bridge)

                self.assertEqual(bridge._recent_count, window_samples)
                self.assertEqual(len(bridge._recent), window_samples)

        # Keep this count available during diagnosis, but do not require drift
        # to exceed the heuristic: a numerically improved implementation should
        # still pass as long as every threshold-adjacent action matches the oracle.

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
