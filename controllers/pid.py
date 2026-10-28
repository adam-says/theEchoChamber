from __future__ import annotations

import numpy as np

from .calibration import CausalEnvelope
from .types import ControllerAction, ControllerDiagnostics, ControllerInput, StepResult, ca3_channel_index


class PIDController:
    """Bounded pulse-rate PID; positive error can increase inhibitory output."""
    name = "pid"

    def __init__(self, sample_rate_hz: float, *, kp: float = 0.0, ki: float = 0.0, kd: float = 0.0,
                 target_z: float = 0.0, direction: float = 1.0, max_rate_hz: float = 2.0,
                 update_ms: float = 5.0, derivative_tau_s: float = 0.020,
                 band_hz: tuple[float, float] = (100.0, 250.0), envelope: CausalEnvelope | None = None):
        numeric = (sample_rate_hz, kp, ki, kd, target_z, direction, max_rate_hz,
                   update_ms, derivative_tau_s)
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("PID settings must all be finite")
        if min(kp, ki, kd) < 0:
            raise ValueError("PID gains must be non-negative")
        if not 0 <= max_rate_hz <= 2:
            raise ValueError("maximum pulse rate must be in [0, 2] Hz")
        if update_ms <= 0 or derivative_tau_s < 0:
            raise ValueError("update interval must be positive and derivative time constant non-negative")
        update_sample_count = sample_rate_hz * update_ms / 1000.0
        if update_sample_count < 1 or not np.isclose(update_sample_count, round(update_sample_count), atol=1e-9):
            raise ValueError("update interval must resolve to a positive integer sample count")
        if direction not in (-1, 1):
            raise ValueError("PID direction must be -1 or 1")
        self.envelope = envelope or CausalEnvelope(sample_rate_hz, band_hz)
        if self.envelope.fs != float(sample_rate_hz):
            raise ValueError("provided envelope sample rate does not match PID sample rate")
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.target_z, self.direction, self.max_rate_hz = float(target_z), float(direction), float(max_rate_hz)
        self.update_samples = int(round(update_sample_count))
        self.dt = self.update_samples / sample_rate_hz
        self.derivative_alpha = self.dt / (derivative_tau_s + self.dt)
        self.reset()

    def reset(self, *, keep_calibration: bool = False) -> None:
        self.envelope.reset(keep_baseline=keep_calibration)
        self.integral = 0.0
        self.derivative = 0.0
        self.previous_z: float | None = None
        self.rate = 0.0
        self._last_update_index: int | None = None
        self._last_generation: int | None = None

    def drain(self) -> None:
        self.rate = 0.0

    def _compute_output(self, measured: float) -> float:
        """Apply filtered derivative-on-measurement and conditional-integral antiwindup."""
        if self.previous_z is not None:
            raw = (measured - self.previous_z) / self.dt
            self.derivative += self.derivative_alpha * (raw - self.derivative)
        error = self.direction * (measured - self.target_z)
        candidate_i = self.integral + error * self.dt
        # Positive Kd makes a rising measurement increase inhibitory output when direction is positive.
        unconstrained = self.kp * error + self.ki * candidate_i + self.kd * self.direction * self.derivative
        output = float(np.clip(unconstrained, 0.0, self.max_rate_hz))
        drives_saturation = (unconstrained > self.max_rate_hz and error > 0) or (unconstrained < 0 and error < 0)
        if not drives_saturation:
            self.integral = candidate_i
        self.previous_z = measured
        return output

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
        env = self.envelope.process(selected, data.first_sample_index, data.validity_mask, stim)
        mean, std = baseline_at_start or (float("nan"), float("nan"))
        z = (env - mean) / std if np.isfinite(std) else np.full(env.shape, np.nan)
        diag_mean, diag_std = self.envelope.baseline or (float("nan"), float("nan"))
        valid = data.validity_mask & np.isfinite(z)
        valid &= ~stim
        feedback_blocked = data.feedback is not None and (not data.feedback.accepted or not data.feedback.armed)
        if feedback_blocked:
            self.rate = 0.0
            self.derivative = 0.0
            self.previous_z = None
        elif not np.all(data.validity_mask):
            # A partial gap makes this block unusable for feedback; hold the integrator.
            self.rate = 0.0
            self.derivative = 0.0
            self.previous_z = None
        elif not was_ready:
            self.rate = 0.0
        elif not np.any(valid):
            if self.envelope.last_feature_invalid_sample_count:
                # Do not schedule pulses from an interval with no valid feature.
                # The shared actuator resets fractional phase at rate zero; it
                # never repays this inhibited time as catch-up pulses.
                self.rate = 0.0
                self.derivative = 0.0
                self.previous_z = None
            else:
                self.rate = 0.0
        else:
            # Update only at the configured 5 ms grid; the final value is held between updates.
            for j in range(z.size):
                index = data.first_sample_index + int(j)
                if stim[j] or not self.envelope.feature_valid_mask[j]:
                    self.derivative = 0.0
                    self.previous_z = None
                    continue
                if not valid[j]:
                    self.derivative = 0.0
                    self.previous_z = None
                    continue
                if (index + 1) % self.update_samples:
                    continue
                measured = float(z[j])
                if data.feedback is not None and (not data.feedback.accepted or not data.feedback.armed):
                    self.rate = 0.0
                    self.derivative = 0.0
                    self.previous_z = None
                    self._last_update_index = index
                    continue
                self.rate = self._compute_output(measured)
                self._last_update_index = index
        ready = self.envelope.baseline is not None
        if (not ready or not was_ready or not np.all(data.validity_mask)
                or self.envelope.last_feature_invalid_sample_count):
            self.rate = 0.0
        action = ControllerAction(kind="pulse_rate", requested_rate_hz=self.rate) if ready else ControllerAction()
        lastz = float(z[-1]) if z.size and np.isfinite(z[-1]) else float("nan")
        diag = ControllerDiagnostics(self.name, ready, ready,
            float(env[-1]) if env.size and np.isfinite(env[-1]) else float("nan"), diag_mean, diag_std, lastz,
            self.target_z, lastz - self.target_z if np.isfinite(lastz) else float("nan"), self.rate,
            "active" if ready else "calibrating", data.generation,
            ("actuator rejected or disarmed" if feedback_blocked
             else self.envelope.calibration_reason if not ready
             else "calibration completed this block" if not was_ready
             else "stimulus_feature_recovery" if self.envelope.last_feature_invalid_sample_count else ""),
            bool(np.all(self.envelope.feature_valid_mask)), self.envelope.feature_recovery_samples,
            self.envelope.feature_recovery_remaining_samples, self.envelope.last_stimulated_sample_count,
            self.envelope.last_feature_invalid_sample_count)
        return StepResult(action, diag)
