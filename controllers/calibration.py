"""Chunk-invariant causal bandpass RMS envelope and frozen baseline."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfilt


class CausalEnvelope:
    def __init__(self, sample_rate_hz: float, band_hz: tuple[float, float] = (100.0, 250.0),
                 smooth_ms: float = 10.0, settle_s: float = 1.0, calibration_s: float = 10.0):
        values = (sample_rate_hz, *band_hz, smooth_ms, settle_s, calibration_s)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("sample rate, band, smoothing, settling, and calibration settings must be finite")
        if (sample_rate_hz <= 0 or len(band_hz) != 2 or band_hz[0] <= 0
                or band_hz[0] >= band_hz[1] or band_hz[1] >= sample_rate_hz / 2):
            raise ValueError("band must satisfy 0 < low < high < Nyquist")
        if smooth_ms <= 0 or settle_s < 0 or calibration_s <= 0:
            raise ValueError("smoothing must be positive, settling non-negative, calibration positive")
        self.fs = float(sample_rate_hz)
        self.band_hz = band_hz
        self.window = max(1, round(self.fs * smooth_ms / 1000))
        self.settle_samples = round(self.fs * settle_s)
        self.calibration_samples = round(self.fs * calibration_s)
        self.sos = butter(4, band_hz, btype="bandpass", fs=self.fs, output="sos")
        self._smooth_b = np.full(self.window, 1.0 / self.window)
        self._smooth_a = np.asarray([1.0])
        # Bound post-mask filter-state error from the dominant SOS pole decay.
        # 1e-8 is 100x below the baseline rejection floor (relative 1e-6).
        # Include the complete moving-RMS support so a contaminated squared
        # sample cannot remain in the envelope after the IIR state settles.
        poles = np.concatenate([np.roots(section[3:6]) for section in self.sos])
        dominant_radius = float(np.max(np.abs(poles)))
        if not 0.0 < dominant_radius < 1.0:
            raise ValueError("bandpass filter poles must lie inside the unit circle")
        self.recovery_attenuation = 1e-8
        self.filter_settle_samples = int(np.ceil(
            np.log(self.recovery_attenuation) / np.log(dominant_radius)))
        self.feature_recovery_samples = self.filter_settle_samples + self.window - 1
        self.reset()

    def reset(self, *, keep_baseline: bool = False) -> None:
        baseline = getattr(self, "baseline", None) if keep_baseline else None
        self.zi = np.zeros((self.sos.shape[0], 2), dtype=np.float64)
        self.smooth_zi = np.zeros(self.window - 1, dtype=np.float64)
        self.valid_seen = 0
        self.baseline_values: list[np.ndarray] = []
        self.baseline_count = 0
        self.baseline = baseline
        self.calibration_reason = "ready" if baseline is not None else "calibrating"
        self.last_index: int | None = None
        self.last_valid_input: float | None = None
        # A reset discards filter and RMS state. If calibration is retained,
        # its baseline is not permission to consume the new filter startup
        # transient as a valid feature.
        self.feature_recovery_remaining_samples = self.feature_recovery_samples if baseline is not None else 0
        self.feature_valid_mask = np.zeros(0, dtype=bool)
        self.last_stimulated_sample_count = 0
        self.last_feature_invalid_sample_count = 0

    def _gap_reset(self, reason: str = "calibration restarted after an acquisition gap") -> None:
        self.reset(keep_baseline=False)
        self.calibration_reason = reason

    def process(self, samples_mv: np.ndarray, first_sample_index: int, validity_mask: np.ndarray,
                stimulated_mask: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(samples_mv, dtype=np.float64).reshape(-1)
        declared_valid = np.asarray(validity_mask, dtype=bool).reshape(-1)
        finite = np.isfinite(x)
        stim = np.zeros(x.size, dtype=bool) if stimulated_mask is None else np.asarray(stimulated_mask, dtype=bool).reshape(-1)
        if x.size != declared_valid.size or x.size != stim.size:
            raise ValueError("sample and mask lengths must match")
        # A finite programmed-stimulation sample is masked even when the DAQ validity
        # mask also excludes it. Non-finite input always denotes a true acquisition fault.
        valid = finite & (declared_valid | stim)
        baseline_eligible = valid & declared_valid & ~stim
        if x.size and self.last_index is not None and first_sample_index != self.last_index + 1:
            self._gap_reset()
        env = np.full(x.shape, np.nan)
        feature_valid = np.zeros(x.size, dtype=bool)
        self.last_stimulated_sample_count = int(np.count_nonzero(stim))
        # Reset exactly once per true invalid run. Masked stimulation spans use a
        # held last-valid input to advance filter time without admitting artifact values.
        i = 0
        while i < x.size:
            if not valid[i]:
                self._gap_reset("calibration restarted after invalid or non-finite input")
                while i < x.size and not valid[i]:
                    i += 1
                continue
            is_stim = bool(stim[i] or not declared_valid[i])
            j = i + 1
            while j < x.size and valid[j] and bool(stim[j] or not declared_valid[j]) == is_stim:
                j += 1
            if is_stim:
                held = 0.0 if self.last_valid_input is None else self.last_valid_input
                filter_input = np.full(j - i, held, dtype=np.float64)
            else:
                filter_input = x[i:j]
                self.last_valid_input = float(filter_input[-1])
            band, self.zi = sosfilt(self.sos, filter_input, zi=self.zi)
            squared = band * band
            smooth, self.smooth_zi = lfilter(self._smooth_b, self._smooth_a, squared, zi=self.smooth_zi)
            envelope = np.sqrt(np.maximum(smooth, 0.0))
            if is_stim:
                # The held value is only a clock-advance input. Neither it nor
                # any settling samples after it are valid controller features.
                self.feature_recovery_remaining_samples = self.feature_recovery_samples
                unmasked = np.empty(0, dtype=np.int64)
            else:
                invalid_recovery = min(self.feature_recovery_remaining_samples, j - i)
                self.feature_recovery_remaining_samples -= invalid_recovery
                usable_start = i + invalid_recovery
                feature_valid[usable_start:j] = True
                env[usable_start:j] = envelope[invalid_recovery:]
                unmasked = np.flatnonzero(baseline_eligible[i:j])[invalid_recovery:]
            settle_left = max(0, self.settle_samples - self.valid_seen)
            usable = unmasked[min(settle_left, unmasked.size):]
            if self.baseline is None and usable.size:
                values = env[i + usable]
                need = self.calibration_samples - self.baseline_count
                values = values[:need]
                if values.size:
                    self.baseline_values.append(values.copy())
                    self.baseline_count += values.size
                    if self.baseline_count >= self.calibration_samples:
                        vals = np.concatenate(self.baseline_values)
                        mean, std = float(np.mean(vals)), float(np.std(vals))
                        floor = max(1e-12, abs(mean) * 1e-6)
                        if not np.isfinite(mean) or not np.isfinite(std) or std <= floor:
                            self.calibration_reason = "baseline rejected: envelope is flat or non-finite"
                            self.baseline_values.clear()
                            self.baseline_count = 0
                        else:
                            self.baseline = (mean, std)
                            self.calibration_reason = "ready"
                            self.baseline_values.clear()
            self.valid_seen += int(unmasked.size)
            i = j
        self.feature_valid_mask = feature_valid
        self.last_feature_invalid_sample_count = int(x.size - np.count_nonzero(feature_valid))
        self.last_index = first_sample_index + x.size - 1 if x.size else self.last_index
        return env
