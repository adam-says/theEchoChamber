"""Shared pulse waveform renderer with explicit AO-command commit lifecycle."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import count

import numpy as np

from .types import ControllerAction


@dataclass(frozen=True)
class PulseEvent:
    pulse_id: str
    requested_sample_index: int
    source_onset_sample_index: int
    target_onset_sample_index: int
    status: str
    generation: int
    rendered_sample_count: int = 0
    latest_sample_index: int | None = None

    @property
    def onset_sample_index(self) -> int | None:
        return self.target_onset_sample_index


@dataclass(frozen=True)
class PulsePlan:
    """Complete safe-waveform proposal in acquisition and AO target coordinates.

    ``waveform_v[0]`` belongs at ``target_onset_sample_index``. A scheduler can
    stage the whole plan at proposal time and retain only its original-index
    tail after a generation switch. The plan is not evidence of AO commitment
    or physical delivery, and must pass the runtime safety admission boundary.
    """

    pulse_id: str
    generation: int
    source_onset_sample_index: int
    target_onset_sample_index: int
    latest_sample_index: int
    waveform_v: np.ndarray


@dataclass(frozen=True)
class ActuatorResult:
    waveform_v: np.ndarray
    pulse_events: tuple[PulseEvent, ...]
    requested_rate_hz: float
    rendered_count: int
    reason: str = ""
    pulse_plans: tuple[PulsePlan, ...] = ()


@dataclass
class _Pulse:
    pulse_id: str
    requested_index: int
    onset_index: int
    target_onset_index: int
    generation: int
    waveform: np.ndarray
    render_cursor: int = 0
    committed_samples: int = 0


class PulseActuator:
    def __init__(self, *, pulse_duration_ms: float = 5.0, pulse_frequency_hz: float = 100.0,
                 refractory_s: float = 0.5, waveform: str = "sine"):
        self.pulse_duration_ms = float(pulse_duration_ms)
        self.pulse_frequency_hz = float(pulse_frequency_hz)
        self.refractory_s = float(refractory_s)
        if not np.all(np.isfinite((self.pulse_duration_ms, self.pulse_frequency_hz, self.refractory_s))):
            raise ValueError("pulse duration, frequency, and refractory interval must be finite")
        if (waveform not in {"sine", "square"} or self.pulse_duration_ms <= 0
                or self.pulse_frequency_hz <= 0 or self.refractory_s < 0):
            raise ValueError("invalid pulse waveform settings")
        self.waveform = waveform
        self.armed = False
        self.rate_phase = 0.0
        self.last_onset: int | None = None
        self._ids = count(1)
        self._pulses: dict[str, _Pulse] = {}

    def arm(self) -> None:
        self.armed = True

    def disarm(self) -> None:
        self.armed = False
        self.cancel_uncommitted()

    def reset(self, *, preserve_refractory: bool = True) -> None:
        self.rate_phase = 0.0
        if not preserve_refractory:
            self.last_onset = None
        self.cancel_uncommitted()

    def cancel_uncommitted(self) -> tuple[str, ...]:
        """Cancel staged pulses with no AO commit; preserve committed pulses for exact tail completion."""
        canceled = tuple(key for key, pulse in self._pulses.items() if pulse.committed_samples == 0)
        for key in canceled:
            del self._pulses[key]
        self.rate_phase = 0.0
        return canceled

    def committed_pulse_ids(self) -> tuple[str, ...]:
        """IDs whose staged AO commands contain at least one confirmed committed sample."""
        return tuple(key for key, pulse in self._pulses.items() if 0 < pulse.committed_samples < pulse.waveform.size)

    def commit(self, pulse_id: str, committed_samples: int) -> None:
        """Record samples confirmed committed by the AO scheduler; this is not physical-emission evidence."""
        pulse = self._pulses.get(pulse_id)
        if pulse is None:
            raise KeyError(f"unknown or canceled pulse id: {pulse_id}")
        if committed_samples < 0:
            raise ValueError("committed_samples must be non-negative")
        pulse.committed_samples = min(pulse.waveform.size, pulse.committed_samples + committed_samples)
        if pulse.committed_samples >= pulse.waveform.size:
            self._pulses.pop(pulse_id, None)

    def _render(self, pulse: _Pulse, first: int, count_samples: int, out: np.ndarray,
                pulse_offset_start: int = 0) -> int:
        begin = max(first, pulse.onset_index + pulse_offset_start)
        offset = begin - first
        pulse_offset = begin - pulse.onset_index
        if offset >= count_samples or pulse_offset >= pulse.waveform.size:
            return 0
        n = min(count_samples - offset, pulse.waveform.size - pulse_offset)
        out[offset:offset + n] += pulse.waveform[pulse_offset:pulse_offset + n]
        pulse.render_cursor = max(pulse.render_cursor, pulse_offset + n)
        return n

    def process(self, action: ControllerAction, *, first_sample_index: int, sample_count: int,
                sample_rate_hz: float, armed: bool, validity_mask: np.ndarray,
                amplitude_v: float = 0.0, generation: int, scheduler_lead_samples: int = 0,
                latest_sample_index: int | None = None) -> ActuatorResult:
        valid = np.asarray(validity_mask, dtype=bool).reshape(-1)
        if valid.size != sample_count or sample_count < 0 or sample_rate_hz <= 0:
            raise ValueError("invalid actuator block timing or validity mask")
        if not np.isfinite(amplitude_v):
            raise ValueError("amplitude must be finite")
        end = first_sample_index + sample_count
        latest_used = end - 1 if latest_sample_index is None else int(latest_sample_index)
        out = np.zeros(sample_count, dtype=np.float64)
        can_output = bool(armed and self.armed and np.all(valid))
        # A partly committed pulse must finish even after a switch/disarm. An uncommitted stage is canceled by lifecycle.
        events: list[PulseEvent] = []
        for pulse_id, pulse in tuple(self._pulses.items()):
            n = self._render(pulse, first_sample_index, sample_count, out,
                             pulse_offset_start=pulse.render_cursor)
            if n:
                events.append(PulseEvent(pulse_id, pulse.requested_index, pulse.onset_index,
                                         pulse.target_onset_index, "committed_tail_rendered", pulse.generation, n,
                                         latest_used))
        if action.kind == "analog":
            if action.analog_v is None or np.asarray(action.analog_v).size != sample_count:
                raise ValueError("analog action must contain one voltage per sample")
            if can_output:
                out += np.asarray(action.analog_v, dtype=np.float64).reshape(-1)
            else:
                self.cancel_uncommitted()
            return ActuatorResult(out, tuple(events), 0.0, 0, "" if can_output else "analog output disarmed or invalid")
        if not can_output or action.kind not in {"pulse_rate", "pulse_events"}:
            self.rate_phase = 0.0
            if not armed or not self.armed or not np.all(valid):
                self.cancel_uncommitted()
            return ActuatorResult(out, tuple(events), float(action.requested_rate_hz), 0,
                                  "disarmed, invalid feedback, or no pulse request")
        rate = float(action.requested_rate_hz) if action.kind == "pulse_rate" else 0.0
        if action.kind == "pulse_rate" and not np.isfinite(rate):
            self.rate_phase = 0.0
            raise ValueError("pulse rate must be finite")
        if action.kind == "pulse_rate" and not 0 <= rate <= 2:
            self.rate_phase = 0.0
            raise ValueError("pulse rate request must be in [0, 2] Hz")
        if action.kind == "pulse_rate":
            if rate == 0:
                self.rate_phase = 0.0
                requests: list[int] = []
            else:
                increments = rate / sample_rate_hz
                phase_before = self.rate_phase
                phase = phase_before + increments * np.arange(1, sample_count + 1, dtype=np.float64)
                crossings = np.flatnonzero(np.floor(phase) > np.floor(phase - increments))
                requests = [first_sample_index + int(i) for i in crossings]
                self.rate_phase = float(phase[-1] % 1.0) if sample_count else phase_before
        else:
            requests = [int(i) for i in action.pulse_sample_indices]
        min_interval = int(round(self.refractory_s * sample_rate_hz))
        pulse_n = max(1, int(round(self.pulse_duration_ms * sample_rate_hz / 1000.0)))
        t = np.arange(pulse_n, dtype=np.float64) / sample_rate_hz
        unit_wave = np.sin(2 * np.pi * self.pulse_frequency_hz * t)
        if self.waveform == "square":
            unit_wave = np.sign(unit_wave)
        rendered = 0
        plans: list[PulsePlan] = []
        for requested in requests:
            pulse_id = f"g{generation}-p{next(self._ids)}"
            if requested < first_sample_index:
                events.append(PulseEvent(pulse_id, requested, requested, requested, "rejected_expired", generation,
                                         latest_sample_index=latest_used))
                self.rate_phase = 0.0
                continue
            if requested >= end:
                events.append(PulseEvent(pulse_id, requested, requested, requested, "rejected_future", generation,
                                         latest_sample_index=latest_used))
                self.rate_phase = 0.0
                continue
            if self.last_onset is not None and requested - self.last_onset < min_interval:
                events.append(PulseEvent(pulse_id, requested, requested, requested, "rejected_refractory", generation,
                                         latest_sample_index=latest_used))
                self.rate_phase = 0.0
                continue
            pulse = _Pulse(pulse_id, requested, requested, requested + scheduler_lead_samples,
                           generation, unit_wave * float(amplitude_v))
            self._pulses[pulse_id] = pulse
            plan_wave = pulse.waveform.copy()
            plan_wave.setflags(write=False)
            plans.append(PulsePlan(pulse_id, generation, requested, requested + scheduler_lead_samples,
                                   latest_used, plan_wave))
            n = self._render(pulse, first_sample_index, sample_count, out)
            rendered += 1
            self.last_onset = requested
            events.append(PulseEvent(pulse_id, requested, requested, requested + scheduler_lead_samples,
                                     "rendered_for_ao", generation, n, latest_used))
        return ActuatorResult(out, tuple(events), rate, rendered, pulse_plans=tuple(plans))
