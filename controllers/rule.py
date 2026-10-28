from __future__ import annotations

import numpy as np

from .calibration import CausalEnvelope
from .types import ControllerAction, ControllerDiagnostics, ControllerInput, StepResult, ca3_channel_index


class RuleController:
    """Upper 3 SD trigger with lower 1.5 SD re-arm hysteresis."""
    name = "rule"

    def __init__(self, sample_rate_hz: float, *, upper_z: float = 3.0, lower_z: float = 1.5,
                 band_hz: tuple[float, float] = (100.0, 250.0), envelope: CausalEnvelope | None = None):
        if not np.isfinite(upper_z) or not np.isfinite(lower_z):
            raise ValueError("rule thresholds must be finite")
        if lower_z < 0 or upper_z <= lower_z:
            raise ValueError("rule thresholds must satisfy 0 <= lower < upper")
        self.envelope = envelope or CausalEnvelope(sample_rate_hz, band_hz)
        if self.envelope.fs != float(sample_rate_hz):
            raise ValueError("provided envelope sample rate does not match rule sample rate")
        self.upper_z, self.lower_z = float(upper_z), float(lower_z)
        self._armed = True
        self._last_generation: int | None = None

    def reset(self, *, keep_calibration: bool = False) -> None:
        self.envelope.reset(keep_baseline=keep_calibration)
        self._armed = True
        self._last_generation = None

    def drain(self) -> None:
        self._armed = True

    def step(self, data: ControllerInput) -> StepResult:
        if (data.samples_mv.ndim != 2 or data.samples_mv.shape[0] != len(data.roles)
                or data.samples_mv.shape[0] != len(data.channel_labels)):
            raise ValueError("samples must have shape (channels, samples), matching roles")
        if data.sample_rate_hz != self.envelope.fs or data.validity_mask.size != data.samples_mv.shape[1]:
            raise ValueError("input sample rate or validity mask does not match the configured feature")
        channel = ca3_channel_index(data.roles)
        source = (data.region, channel, data.channel_labels[channel], data.roles[channel], float(data.sample_rate_hz))
        stim = np.zeros(data.samples_mv.shape[1], dtype=bool) if data.stimulated_mask is None else np.asarray(data.stimulated_mask, bool)
        selected = np.asarray(data.samples_mv[channel], dtype=np.float64)
        broken = (getattr(self, "_source", None) is not None and source != self._source)
        broken |= self.envelope.last_index is not None and data.first_sample_index != self.envelope.last_index + 1
        broken |= np.any((~data.validity_mask) & (~stim)) or np.any(~np.isfinite(selected))
        if broken:
            self.reset(keep_calibration=False)
        self._source = source
        was_ready = self.envelope.baseline is not None
        baseline_at_start = self.envelope.baseline
        self._last_generation = data.generation
        env = self.envelope.process(selected, data.first_sample_index,
                                    data.validity_mask, stim)
        mean, std = baseline_at_start or (float("nan"), float("nan"))
        z = (env - mean) / std if np.isfinite(std) else np.full(env.shape, np.nan)
        diag_mean, diag_std = self.envelope.baseline or (float("nan"), float("nan"))
        events: list[int] = []
        if was_ready:
            for j, score in enumerate(z):
                if not np.isfinite(score) or stim is not None and stim[j]:
                    continue
                if self._armed and score >= self.upper_z:
                    events.append(data.first_sample_index + j)
                    self._armed = False
                elif not self._armed and score <= self.lower_z:
                    self._armed = True
        ready = self.envelope.baseline is not None
        # Calibration may finish partway through this block; do not backdate control to earlier samples.
        action = ControllerAction(kind="pulse_events", pulse_sample_indices=tuple(events)) if events and was_ready else ControllerAction()
        diag = ControllerDiagnostics(self.name, ready, ready,
            float(env[-1]) if env.size and np.isfinite(env[-1]) else float("nan"), diag_mean, diag_std,
            float(z[-1]) if z.size and np.isfinite(z[-1]) else float("nan"), self.upper_z,
            float(z[-1] - self.upper_z) if z.size and np.isfinite(z[-1]) else float("nan"),
            0.0, "armed" if self._armed else "latched", data.generation,
            self.envelope.calibration_reason if not ready else "")
        return StepResult(action, diag)
