"""Application boundary between Echo Chamber and numeric ESN inference.

This module adapts DAQ-sized blocks to the artifact's preferred chunk size and
owns all stimulation mapping and diagnostics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np


StimMode = Literal["off", "passthrough", "threshold_pulse"]
PulsePolarity = Literal["absolute", "positive", "negative"]


@dataclass(frozen=True)
class BridgeDiagnostics:
    model_output: np.ndarray
    pulse_threshold: float
    pulse_peak: float
    pulse_fired: bool
    trigger_sample: Optional[int] = None
    conditioned_output: Optional[np.ndarray] = None
    passthrough_output: Optional[np.ndarray] = None


class EchoChamberEsnBridge:
    """Adapt an unmodified ESN streamer to the application's runtime contract."""

    def __init__(
        self,
        streamer: Any,
        *,
        runtime_chunk_size: int,
        sample_rate: int,
        passthrough_dc_block_hz: float = 0.5,
        pulse_threshold_std: float = 3.0,
        pulse_window_sec: float = 10.0,
        pulse_polarity: PulsePolarity = "absolute",
        ao_command_gain_v_per_esn_unit: float = 1.0,
    ) -> None:
        self.streamer = streamer
        self.runtime_chunk_size = int(runtime_chunk_size)
        self.sample_rate = int(sample_rate)
        self.artifact_sample_rate = int(getattr(streamer, "fs_in", self.sample_rate))
        self.preferred_chunk_size = int(getattr(streamer, "chunk_size", 0))
        self.model_rate = int(getattr(streamer, "fs_train", 2_000))
        self.decim_q = int(getattr(streamer, "decim_q", self.sample_rate // self.model_rate))
        self.ao_command_gain_v_per_esn_unit = float(ao_command_gain_v_per_esn_unit)
        if self.runtime_chunk_size <= 0 or self.preferred_chunk_size <= 0:
            raise ValueError("runtime and artifact chunk sizes must be positive")
        if self.sample_rate != self.artifact_sample_rate:
            raise ValueError(
                f"DAQ sample_rate={self.sample_rate} does not match artifact "
                f"fs_in={self.artifact_sample_rate}; explicit resampling is required"
            )
        if self.runtime_chunk_size % self.preferred_chunk_size:
            raise ValueError(
                f"runtime chunk_size={self.runtime_chunk_size} must be a multiple of "
                f"artifact chunk_size={self.preferred_chunk_size}"
            )
        if self.sample_rate <= 0 or self.model_rate <= 0 or self.sample_rate % self.model_rate:
            raise ValueError("sample rate must be an integer multiple of the ESN model rate")
        if self.decim_q != self.sample_rate // self.model_rate:
            raise ValueError("artifact decimation factor is inconsistent with its sample rates")
        if not np.isfinite(self.ao_command_gain_v_per_esn_unit) or self.ao_command_gain_v_per_esn_unit <= 0:
            raise ValueError("AO command gain must be finite and positive")

        self.stim_mode: StimMode = "passthrough"
        self.stim_gain = 1.0
        self.passthrough_dc_block_hz = 0.0
        self.pulse_threshold_std = 1.0
        self.pulse_window_sec = 10.0
        self.pulse_polarity: PulsePolarity = "absolute"
        pulse_cfg = getattr(streamer, "pulse_cfg", None)
        self.pulse_min_interval_sec = float(getattr(pulse_cfg, "min_interval_sec", 0.5))
        self.pulse_duration_ms = float(getattr(pulse_cfg, "pulse_duration_ms", 5.0))
        self.pulse_freq_hz = float(getattr(pulse_cfg, "pulse_freq_hz", 100.0))
        self.pulse_waveform = str(getattr(pulse_cfg, "waveform", "sine"))
        self.stim_clip_v = tuple(getattr(streamer, "stim_clip_v", (-10.0, 10.0)))

        self._recent: deque[float] = deque()
        self._recent_count = 0
        self._recent_origin = 0.0
        self._recent_mean_delta = 0.0
        self._recent_m2 = 0.0
        # Live absolute-value peak and since-rebuild high-water support a
        # bounded conditioning check even when the fixed statistical origin
        # belongs to a departed startup transient.
        self._recent_peak_abs_values: deque[tuple[int, float]] = deque()
        self._recent_sequence = 0
        self._recent_scale_highwater = 0.0
        self._recent_scale_contraction_factor = 0.25
        self._threshold_exact_fallbacks = 0
        self._rolling_stat_rebuilds = 0
        self._samples_since_pulse = 10**9
        self._dc_previous_input: Optional[float] = None
        self._dc_previous_output = 0.0
        self._previous_pulse_metric: Optional[float] = None
        self._pending_pulse = np.empty(0, dtype=np.float64)
        self._diagnostics = BridgeDiagnostics(np.full((1, self.runtime_chunk_size), np.nan),
                                              float("nan"), float("nan"), False)

        # The original streamer already contains stimulation modes. Keep it in
        # its identity-like passthrough configuration and perform all choices
        # after its returned prediction crosses this bridge.
        self.streamer.configure(stim_mode="passthrough", stim_gain=1.0)
        # The original passthrough mapper clips before returning. This artifact
        # predicts around 21.8, so its default +/-10 clip would collapse the
        # prediction to a constant 10 before the application can DC-block it.
        # Preserve that configured range above for the final bridge command,
        # but make the prediction boundary itself non-clipping.
        if hasattr(self.streamer, "stim_clip_v"):
            self.streamer.stim_clip_v = (-float("inf"), float("inf"))
        self.configure(
            stim_mode="passthrough",
            stim_gain=1.0,
            passthrough_dc_block_hz=passthrough_dc_block_hz,
            pulse_threshold_std=pulse_threshold_std,
            pulse_window_sec=pulse_window_sec,
            pulse_polarity=pulse_polarity,
        )

    @classmethod
    def load(
        cls,
        artifact: Path | str,
        **kwargs: Any,
    ) -> "EchoChamberEsnBridge":
        from esn import load_artifact

        backend = str(kwargs.pop("esn_backend", "auto"))
        return cls(load_artifact(str(artifact), backend=backend), **kwargs)

    def configure(
        self,
        *,
        stim_mode: Optional[StimMode] = None,
        stim_gain: Optional[float] = None,
        passthrough_dc_block_hz: Optional[float] = None,
        pulse_threshold_std: Optional[float] = None,
        pulse_window_sec: Optional[float] = None,
        pulse_polarity: Optional[PulsePolarity] = None,
    ) -> None:
        if stim_mode is not None:
            if stim_mode not in {"off", "passthrough", "threshold_pulse"}:
                raise ValueError(f"invalid stimulation mode: {stim_mode}")
            self.stim_mode = stim_mode
        if stim_gain is not None:
            if not np.isfinite(stim_gain) or stim_gain < 0:
                raise ValueError("stimulation gain must be finite and non-negative")
            self.stim_gain = float(stim_gain)
        if passthrough_dc_block_hz is not None:
            if not np.isfinite(passthrough_dc_block_hz) or passthrough_dc_block_hz < 0:
                raise ValueError("DC-block frequency must be finite and non-negative")
            self.passthrough_dc_block_hz = float(passthrough_dc_block_hz)
        if pulse_threshold_std is not None:
            if not np.isfinite(pulse_threshold_std) or pulse_threshold_std <= 0:
                raise ValueError("pulse threshold must be finite and positive")
            self.pulse_threshold_std = float(pulse_threshold_std)
        if pulse_window_sec is not None:
            if not np.isfinite(pulse_window_sec) or pulse_window_sec < 1:
                raise ValueError("pulse window must be finite and at least one second")
            self.pulse_window_sec = float(pulse_window_sec)
            maxlen = max(1, int(self.pulse_window_sec * self.model_rate))
            self._recent = deque(self._recent, maxlen=maxlen)
            self._rebuild_recent_statistics()
        if pulse_polarity is not None:
            if pulse_polarity not in {"absolute", "positive", "negative"}:
                raise ValueError(f"invalid pulse polarity: {pulse_polarity}")
            self.pulse_polarity = pulse_polarity

    def reset(self) -> None:
        self.streamer.reset()
        self._recent.clear()
        self._reset_recent_statistics()
        self._threshold_exact_fallbacks = 0
        self._samples_since_pulse = 10**9
        self._dc_previous_input = None
        self._dc_previous_output = 0.0
        self._previous_pulse_metric = None
        self._pending_pulse = np.empty(0, dtype=np.float64)
        self._diagnostics = BridgeDiagnostics(np.full((1, self.runtime_chunk_size), np.nan),
                                              float("nan"), float("nan"), False)

    def process(self, ai_chunk: np.ndarray, *, ctx_index: int, render_legacy: bool = True) -> np.ndarray:
        data = np.asarray(ai_chunk, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.runtime_chunk_size:
            raise ValueError(
                f"expected AI shape (channels, {self.runtime_chunk_size}); got {data.shape}"
            )
        predictions: list[np.ndarray] = []
        for start in range(0, self.runtime_chunk_size, self.preferred_chunk_size):
            piece = data[:, start:start + self.preferred_chunk_size]
            prediction = np.asarray(
                self.streamer.process_chunk(piece, ctx_index=ctx_index), dtype=np.float64
            )
            expected = (1, self.preferred_chunk_size)
            if prediction.shape != expected or not np.all(np.isfinite(prediction)):
                raise ValueError(f"ESN returned invalid output {prediction.shape}; expected {expected}")
            predictions.append(prediction)
        model_output = np.concatenate(predictions, axis=1)

        conditioned = self._condition_model_output(model_output)
        threshold, peak, fired, trigger_sample = self._update_pulse_state(conditioned)
        passthrough = self._passthrough_command(conditioned)
        if not render_legacy:
            command = np.zeros_like(model_output)
        elif self.stim_mode == "off":
            command = np.zeros_like(model_output)
        elif self.stim_mode == "threshold_pulse":
            command = self._pulse_command(model_output.shape[1], trigger_sample if fired else None)
        else:
            command = self._passthrough_command(conditioned)

        self._diagnostics = BridgeDiagnostics(model_output.copy(), threshold, peak, fired,
                                              trigger_sample, conditioned.copy(), passthrough.copy())
        return command

    def diagnostics(self, samples: int) -> tuple[np.ndarray, float, float, bool]:
        diagnostics = self._diagnostics
        model = diagnostics.model_output
        if model.shape != (1, samples):
            model = np.full((1, samples), np.nan)
        return model.copy(), diagnostics.pulse_threshold, diagnostics.pulse_peak, diagnostics.pulse_fired

    def detailed_diagnostics(self) -> BridgeDiagnostics:
        """Return exact threshold crossing and conditioned ESN output for shared-actuator adapters."""
        item = self._diagnostics
        return BridgeDiagnostics(item.model_output.copy(), item.pulse_threshold, item.pulse_peak,
                                 item.pulse_fired, item.trigger_sample,
                                 None if item.conditioned_output is None else item.conditioned_output.copy(),
                                 None if item.passthrough_output is None else item.passthrough_output.copy())

    def _update_pulse_state(
        self, conditioned_output: np.ndarray
    ) -> tuple[float, float, bool, Optional[int]]:
        values = np.asarray(conditioned_output[0, ::self.decim_q], dtype=np.float64)
        if self.pulse_polarity == "positive":
            metric = values
        elif self.pulse_polarity == "negative":
            metric = -values
        else:
            metric = np.abs(values)
        peak = float(np.max(metric))
        warmup = max(5, int(min(1.0, self.pulse_window_sec) * self.model_rate))
        self._maybe_rebuild_after_scale_contraction(warmup)
        threshold = float("nan")
        fired = False
        trigger_index: Optional[int] = None
        self._samples_since_pulse += values.size
        interval = int(self.pulse_min_interval_sec * self.model_rate)
        interval_ready = self._samples_since_pulse >= interval
        if self._recent_count >= warmup:
            mean = self._recent_origin + self._recent_mean_delta
            std = float(np.sqrt(max(0.0, self._recent_m2 / self._recent_count)))
            threshold = float(mean + self.pulse_threshold_std * (std + 1e-12))
            previous = self._previous_pulse_metric
            ambiguous_crossing = (
                interval_ready
                and self._ambiguous_crossing_needs_exact_threshold(metric, previous, threshold)
            )
            # At extreme absolute offsets, the centered running moments and the
            # original full-scale NumPy calculation can differ by more than the
            # threshold diagnostic's four-ULP budget. Use the exact legacy
            # statistic only when the estimated floating-point uncertainty is
            # already a material fraction of that budget.
            if ambiguous_crossing or self._threshold_uncertainty_needs_exact_statistic(metric, threshold):
                threshold = self._exact_recent_threshold()
                self._threshold_exact_fallbacks += 1
            crossings = np.flatnonzero(
                (metric > threshold)
                & np.r_[previous is None or previous <= threshold, metric[:-1] <= threshold]
            )
            if crossings.size and interval_ready:
                fired = True
                trigger_index = int(crossings[0])
                self._samples_since_pulse = 0
        self._previous_pulse_metric = float(metric[-1]) if metric.size else self._previous_pulse_metric
        # Detected-event blocks are excluded so the event cannot inflate its
        # own adaptive threshold and suppress subsequent genuine events.
        if not fired:
            for value in metric:
                self._append_recent(float(value))
        trigger_sample = trigger_index * self.decim_q if trigger_index is not None else None
        return threshold, peak, fired, trigger_sample

    def _reset_recent_statistics(self) -> None:
        self._recent_count = 0
        self._recent_origin = 0.0
        self._recent_mean_delta = 0.0
        self._recent_m2 = 0.0
        self._recent_peak_abs_values.clear()
        self._recent_sequence = 0
        self._recent_scale_highwater = 0.0

    def _rebuild_recent_statistics(self) -> None:
        """Rebase rolling moments and peak scale with a centered two-pass sum."""
        self._rolling_stat_rebuilds += 1
        self._reset_recent_statistics()
        if not self._recent:
            return
        values = np.fromiter(self._recent, dtype=np.float64, count=len(self._recent))
        self._recent_origin = float(values[0])
        centered = values - self._recent_origin
        self._recent_mean_delta = float(np.mean(centered))
        deviations = centered - self._recent_mean_delta
        self._recent_m2 = float(np.sum(deviations * deviations, dtype=np.float64))
        self._recent_count = int(values.size)
        self._recent_sequence = int(values.size)
        for sequence, value in enumerate(values):
            magnitude = abs(float(value))
            self._recent_scale_highwater = max(self._recent_scale_highwater, magnitude)
            while (self._recent_peak_abs_values
                   and self._recent_peak_abs_values[-1][1] <= magnitude):
                self._recent_peak_abs_values.pop()
            self._recent_peak_abs_values.append((sequence, magnitude))

    def _add_recent(self, value: float) -> None:
        if self._recent_count == 0:
            self._recent_origin = value
            self._recent_mean_delta = 0.0
            self._recent_m2 = 0.0
            self._recent_count = 1
            return
        centered = value - self._recent_origin
        count = self._recent_count + 1
        delta = centered - self._recent_mean_delta
        mean = self._recent_mean_delta + delta / count
        self._recent_m2 += delta * (centered - mean)
        self._recent_mean_delta = mean
        self._recent_count = count

    def _remove_recent(self, value: float) -> None:
        if self._recent_count <= 1:
            self._reset_recent_statistics()
            return
        centered = value - self._recent_origin
        count = self._recent_count
        mean = self._recent_mean_delta + (self._recent_mean_delta - centered) / (count - 1)
        self._recent_m2 -= (centered - self._recent_mean_delta) * (centered - mean)
        self._recent_count = count - 1
        self._recent_mean_delta = mean
        # Welford removal can leave a tiny negative roundoff residue for a flat
        # window. It represents zero variance, never negative variance.
        if self._recent_m2 < 0.0:
            self._recent_m2 = 0.0

    def _append_recent(self, value: float) -> None:
        maxlen = self._recent.maxlen
        if maxlen is not None and len(self._recent) == maxlen:
            expired = self._recent.popleft()
            self._remove_recent(expired)
            expired_sequence = self._recent_sequence - len(self._recent) - 1
            if (self._recent_peak_abs_values
                    and self._recent_peak_abs_values[0][0] == expired_sequence):
                self._recent_peak_abs_values.popleft()
        self._recent.append(value)
        if self._recent_count == 0:
            self._recent_origin = value
        self._add_recent(value)
        magnitude = abs(float(value))
        self._recent_scale_highwater = max(self._recent_scale_highwater, magnitude)
        while (self._recent_peak_abs_values
               and self._recent_peak_abs_values[-1][1] <= magnitude):
            self._recent_peak_abs_values.pop()
        self._recent_peak_abs_values.append((self._recent_sequence, magnitude))
        self._recent_sequence += 1
        # If the original center is far outside the current signal's variation
        # (for example after a large level shift has fully crossed the window),
        # rebase once. This path is unusual and keeps Welford deltas well-scaled.
        std = float(np.sqrt(max(0.0, self._recent_m2 / max(1, self._recent_count))))
        if abs(self._recent_mean_delta) > 1e6 * max(std, 1e-12):
            self._rebuild_recent_statistics()

    def _maybe_rebuild_after_scale_contraction(self, warmup: int) -> bool:
        """Empirically rebase once when the live scale contracts materially."""
        if self._recent_count < warmup or not self._recent_peak_abs_values:
            return False
        live_peak = self._recent_peak_abs_values[0][1]
        highwater = self._recent_scale_highwater
        if highwater <= 0.0:
            return False
        if live_peak == 0.0 or live_peak <= self._recent_scale_contraction_factor * highwater:
            self._rebuild_recent_statistics()
            return True
        return False

    @property
    def rolling_stat_rebuild_count(self) -> int:
        """Number of O(window) initialization, resize, or numerical rebase passes."""
        return self._rolling_stat_rebuilds

    @property
    def exact_threshold_fallback_count(self) -> int:
        """Count exact NumPy fallbacks for ambiguous crossings or roundoff budget."""
        return self._threshold_exact_fallbacks

    def _exact_recent_threshold(self) -> float:
        history = np.fromiter(self._recent, dtype=np.float64)
        return float(np.mean(history) + self.pulse_threshold_std * (np.std(history) + 1e-12))

    def _threshold_uncertainty_needs_exact_statistic(
        self, metric: np.ndarray, threshold: float
    ) -> bool:
        """Fall back if estimated roundoff consumes half the four-ULP budget."""
        if not metric.size:
            return False
        std = float(np.sqrt(max(0.0, self._recent_m2 / max(1, self._recent_count))))
        centered_peak = max(
            float(np.max(np.abs(metric - self._recent_origin))),
            self._recent_scale_highwater,
        )
        uncertainty = (
            16.0 * np.finfo(np.float64).eps
            * max(1.0, abs(self._recent_mean_delta), std, centered_peak)
            + 2.0 * abs(float(np.spacing(threshold)))
        )
        threshold_budget = max(
            2e-12, 4.0 * abs(float(np.spacing(threshold)))
        )
        return uncertainty >= 0.5 * threshold_budget

    def _ambiguous_crossing_needs_exact_threshold(
        self, metric: np.ndarray, previous: Optional[float], threshold: float
    ) -> bool:
        """Use the old NumPy calculation only when rounding could alter a pulse decision."""
        if not metric.size:
            return False
        # Centered Welford keeps statistical error tied to signal variation;
        # this check narrows the legacy NumPy calculation to possible crossings.
        mean_delta = self._recent_mean_delta
        std = float(np.sqrt(max(0.0, self._recent_m2 / max(1, self._recent_count))))
        centered_peak = float(np.max(np.abs(metric - self._recent_origin)))
        roundoff = 16.0 * np.finfo(np.float64).eps * max(1.0, abs(mean_delta), std, centered_peak)
        roundoff += 2.0 * abs(float(np.spacing(threshold)))
        # Constant windows have an exactly known variance of zero; avoid
        # repeatedly scanning a high-offset flat window whose threshold rounds
        # back to its common value.
        if self._recent_m2 == 0.0 and np.all(metric == self._recent_origin):
            return False
        current = np.asarray(metric, dtype=np.float64)
        prior = np.empty(current.size, dtype=np.float64)
        if previous is None:
            prior[0] = -np.inf
        else:
            prior[0] = previous
        if current.size > 1:
            prior[1:] = current[:-1]
        possible = (current > threshold - roundoff) & (prior <= threshold + roundoff)
        robust = (current > threshold + roundoff) & (prior <= threshold - roundoff)
        return bool(np.any(possible & ~robust))

    def _condition_model_output(self, model_output: np.ndarray) -> np.ndarray:
        source = model_output.reshape(-1)
        if self.passthrough_dc_block_hz > 0:
            dt_s = 1.0 / self.sample_rate
            rc_s = 1.0 / (2.0 * np.pi * self.passthrough_dc_block_hz)
            alpha = rc_s / (rc_s + dt_s)
            blocked = np.empty_like(source)
            previous_input = self._dc_previous_input
            previous_output = self._dc_previous_output
            for index, value in enumerate(source):
                if previous_input is None:
                    previous_input = float(value)
                    blocked[index] = 0.0
                    continue
                previous_output = alpha * (previous_output + float(value) - previous_input)
                blocked[index] = previous_output
                previous_input = float(value)
            self._dc_previous_input = previous_input
            self._dc_previous_output = previous_output
            source = blocked
        return source.reshape(1, -1)

    def _passthrough_command(self, conditioned_output: np.ndarray) -> np.ndarray:
        source = conditioned_output.reshape(-1)
        command = (source.reshape(1, -1) * self.stim_gain
                   * self.ao_command_gain_v_per_esn_unit)
        return np.clip(command, float(self.stim_clip_v[0]), float(self.stim_clip_v[1]))

    def _pulse_command(self, samples: int, trigger_sample: Optional[int]) -> np.ndarray:
        command = np.zeros((1, samples), dtype=np.float64)
        continuation = min(samples, self._pending_pulse.size)
        if continuation:
            command[0, :continuation] = self._pending_pulse[:continuation]
            self._pending_pulse = self._pending_pulse[continuation:].copy()
        if trigger_sample is not None:
            # CONFIRMED PENDING TEST: preserve the collaborator-selected 5 ms,
            # 100 Hz sine half-cycle. Verify its measured polarity, amplitude,
            # and duration on the NI rig before interpreting biological runs.
            duration = max(1, int(self.pulse_duration_ms * self.sample_rate / 1_000.0))
            time_s = np.arange(duration, dtype=np.float64) / self.sample_rate
            phase = 2.0 * np.pi * self.pulse_freq_hz * time_s
            wave = np.sign(np.sin(phase)) if self.pulse_waveform == "square" else np.sin(phase)
            wave = wave * self.stim_gain * self.ao_command_gain_v_per_esn_unit
            start = min(samples, max(0, int(trigger_sample)))
            available = samples - start
            copied = min(available, duration)
            if copied:
                command[0, start:start + copied] = wave[:copied]
            self._pending_pulse = wave[copied:].copy()
        return np.clip(command, float(self.stim_clip_v[0]), float(self.stim_clip_v[1]))
