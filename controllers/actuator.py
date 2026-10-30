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
    controller_id: str | None = None
    rendered_source_sample_index: int | None = None

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
class PulseAoOutcome:
    """Scheduler-authoritative status for an absolute, half-open AO range."""

    pulse_id: str
    generation: int
    start_sample_index: int
    end_sample_index: int
    status: str
    reason: str = ""
    source_onset_sample_index: int | None = None
    target_onset_sample_index: int | None = None
    latest_sample_index: int | None = None
    controller_id: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"reserved", "committed", "cancelled", "write_unknown"}:
            raise ValueError(f"invalid AO pulse outcome: {self.status}")
        if self.start_sample_index < 0 or self.end_sample_index < self.start_sample_index:
            raise ValueError("invalid AO pulse outcome range")


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
    accounted_samples: int = 0
    fault_accounting_limit: int | None = None
    ao_status: np.ndarray | None = None
    admitted_safe: bool = False
    controller_id: str | None = None
    latest_sample_index: int | None = None


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
        self.rate_phase_high_water = 0.0
        self.rate_phase_reset_count = 0
        self.last_onset: int | None = None
        self._ids = count(1)
        self._pulses: dict[str, _Pulse] = {}
        self.max_pending_pulses = 64

    def arm(self) -> None:
        self.armed = True

    def disarm(self) -> None:
        self.armed = False
        self.cancel_uncommitted()

    def reset(self, *, preserve_refractory: bool = True) -> None:
        self._reset_rate_phase()
        if not preserve_refractory:
            self.last_onset = None
        self.cancel_uncommitted()

    def _reset_rate_phase(self) -> None:
        if self.rate_phase != 0.0:
            self.rate_phase_reset_count += 1
        self.rate_phase = 0.0

    def cancel_uncommitted(self) -> tuple[str, ...]:
        """Return candidate IDs while retaining state until scheduler cancellation is authoritative."""
        canceled = tuple(key for key, pulse in self._pulses.items() if pulse.committed_samples == 0)
        # A scheduler reservation can be racing this source-thread call. Keep
        # admitted state until an authoritative terminal outcome; this return
        # value is advisory and never proves cancellation.
        self._reset_rate_phase()
        return canceled

    def committed_pulse_ids(self) -> tuple[str, ...]:
        """IDs whose staged AO commands contain at least one confirmed committed sample."""
        return tuple(key for key, pulse in self._pulses.items() if 0 < pulse.committed_samples < pulse.waveform.size)

    def pulse_origin(self, pulse_id: str) -> dict[str, int | str | None] | None:
        pulse = self._pulses.get(str(pulse_id))
        if pulse is None:
            return None
        return {
            "controller_id": pulse.controller_id,
            "generation": int(pulse.generation),
            "source_onset_sample_index": int(pulse.onset_index),
            "target_onset_sample_index": int(pulse.target_onset_index),
            "latest_sample_index": pulse.latest_sample_index,
        }

    def commit(self, pulse_id: str, committed_samples: int) -> None:
        """Record samples confirmed committed by the AO scheduler; this is not physical-emission evidence."""
        pulse = self._pulses.get(pulse_id)
        if pulse is None:
            raise KeyError(f"unknown or canceled pulse id: {pulse_id}")
        if committed_samples < 0:
            raise ValueError("committed_samples must be non-negative")
        pulse.committed_samples = min(pulse.waveform.size, pulse.committed_samples + committed_samples)
        if pulse.ao_status is None:
            pulse.ao_status = np.zeros(pulse.waveform.size, dtype=np.uint8)
        pulse.ao_status[:pulse.committed_samples] = 2
        if pulse.fault_accounting_limit is not None:
            # A successful AO write may have been in flight when the fault was
            # latched. Its newly confirmed prefix becomes irrevocable history.
            pulse.fault_accounting_limit = self._possible_prefix(pulse)
        self._retire_accounted_pulses()

    def apply_ao_outcome(self, outcome: PulseAoOutcome) -> None:
        """Merge scheduler truth monotonically; duplicate or reordered ranges are safe."""
        pulse = self._pulses.get(outcome.pulse_id)
        if pulse is None:
            return
        if pulse.ao_status is None:
            pulse.ao_status = np.zeros(pulse.waveform.size, dtype=np.uint8)
        lo = max(0, int(outcome.start_sample_index) - pulse.target_onset_index)
        hi = min(pulse.waveform.size, int(outcome.end_sample_index) - pulse.target_onset_index)
        if hi <= lo:
            return
        status = {"reserved": 1, "committed": 2, "cancelled": 3, "write_unknown": 4}[outcome.status]
        target = pulse.ao_status[lo:hi]
        if status == 1:
            target[target == 0] = 1
        elif status == 2:
            # A successful write is authoritative even if reservation callback
            # delivery was delayed or reordered.
            target[:] = 2
        elif status == 3:
            # Cancellation is terminal only for samples not known committed.
            target[(target != 2) & (target != 4)] = 3
        else:
            target[(target != 2) & (target != 3)] = 4
        if pulse.fault_accounting_limit is not None:
            pulse.fault_accounting_limit = self._possible_prefix(pulse)
        pulse.committed_samples = int(np.count_nonzero(pulse.ao_status == 2))
        self._retire_accounted_pulses()

    @staticmethod
    def _possible_prefix(pulse: _Pulse) -> int:
        if pulse.ao_status is None:
            return 0
        possible = ((pulse.ao_status == 1) | (pulse.ao_status == 2) | (pulse.ao_status == 4)
                    | ((pulse.ao_status == 0) if pulse.admitted_safe else False))
        missing = np.flatnonzero(~possible)
        return int(missing[0]) if missing.size else int(possible.size)

    def admit_safe_plan(self, pulse_id: str, waveform_v: np.ndarray) -> None:
        """Replace the local timeline with the complete waveform admitted by safety."""
        pulse = self._pulses.get(pulse_id)
        if pulse is None:
            return
        safe = np.asarray(waveform_v, dtype=np.float64).reshape(-1)
        if safe.size != pulse.waveform.size or not np.all(np.isfinite(safe)):
            raise ValueError("admitted pulse waveform must be finite and match the rendered plan")
        # The current source block has already passed through the live safety
        # adapter; its admitted values may differ from the raw proposal. Future
        # local accounting and AO ownership use this full safe plan.
        pulse.waveform = safe.copy()
        pulse.admitted_safe = True

    def account_source_interval(self, first_sample_index: int, sample_count: int, *,
                                committed_only: bool = False,
                                possible_only: bool = False) -> tuple[tuple[int, np.ndarray, str], ...]:
        """Claim admitted pulse slices overlapping a source interval exactly once.

        Returned positions are absolute source indices. On a fault, only the
        irrevocably committed prefix is eligible; this method never creates an
        AO request or changes scheduler ownership.
        """
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        start, end = int(first_sample_index), int(first_sample_index) + int(sample_count)
        pieces: list[tuple[int, np.ndarray, str]] = []
        for pulse_id, pulse in tuple(self._pulses.items()):
            limit = (self._possible_prefix(pulse)
                     if possible_only or committed_only else pulse.waveform.size)
            if pulse.fault_accounting_limit is not None:
                limit = min(limit, pulse.fault_accounting_limit)
            lo = max(start, pulse.onset_index + pulse.accounted_samples,
                      pulse.onset_index)
            hi = min(end, pulse.onset_index + limit)
            if lo >= hi:
                continue
            first_offset = lo - pulse.onset_index
            last_offset = hi - pulse.onset_index
            pieces.append((lo, pulse.waveform[first_offset:last_offset].copy(), pulse_id))
            pulse.accounted_samples = last_offset
            if pulse.render_cursor < last_offset:
                pulse.render_cursor = last_offset
        pieces.sort(key=lambda item: item[0])
        # The caller still needs immutable pulse origin metadata to record the
        # accounting-only slice. Retirement is deferred until after that
        # metadata has been captured (normally by mark_source_accounted).
        return tuple(pieces)

    def retire_accounted(self) -> None:
        """Retire resolved plans after their source-accounting metadata is captured."""
        self._retire_accounted_pulses()

    def mark_source_accounted(self, first_sample_index: int, sample_count: int, *,
                              committed_only: bool = False) -> tuple[tuple[str, int], ...]:
        """Advance local safety-accounting watermarks after the live adapter ran."""
        start, end = int(first_sample_index), int(first_sample_index) + int(sample_count)
        changed: list[tuple[str, int]] = []
        for pulse in self._pulses.values():
            previous = pulse.accounted_samples
            limit = min(pulse.render_cursor, pulse.waveform.size)
            if committed_only:
                limit = min(limit, self._possible_prefix(pulse))
            lo = max(start, pulse.onset_index + pulse.accounted_samples, pulse.onset_index)
            hi = min(end, pulse.onset_index + limit)
            if lo < hi:
                pulse.accounted_samples = hi - pulse.onset_index
                if pulse.accounted_samples > previous:
                    changed.append((pulse.pulse_id, pulse.onset_index + pulse.accounted_samples))
        self._retire_accounted_pulses()
        return tuple(changed)

    def advance_cancelled_accounting(self, first_sample_index: int,
                                    sample_count: int) -> tuple[tuple[str, int], ...]:
        """Acknowledge terminal canceled source slices as zero history, without rendering them."""
        end = int(first_sample_index) + int(sample_count)
        changed: list[tuple[str, int]] = []
        for pulse in self._pulses.values():
            if pulse.ao_status is None:
                continue
            offset = pulse.accounted_samples
            while offset < pulse.waveform.size and pulse.ao_status[offset] == 3:
                offset += 1
            terminal_end = min(end, pulse.onset_index + offset)
            if terminal_end > pulse.onset_index + pulse.accounted_samples:
                pulse.accounted_samples = terminal_end - pulse.onset_index
                changed.append((pulse.pulse_id, terminal_end))
        self._retire_accounted_pulses()
        return tuple(changed)

    def limit_accounting_to_committed(self) -> tuple[tuple[str, int, int], ...]:
        """Prevent continuation after fault/gap and return cumulative terminal watermarks."""
        finalized: list[tuple[str, int, int]] = []
        for pulse in self._pulses.values():
            pulse.fault_accounting_limit = self._possible_prefix(pulse)
            finalized.append((pulse.pulse_id, pulse.onset_index + pulse.accounted_samples,
                              pulse.onset_index + pulse.fault_accounting_limit))
        self._retire_accounted_pulses()
        return tuple(finalized)

    def _retire_accounted_pulses(self) -> None:
        for pulse_id, pulse in tuple(self._pulses.items()):
            limit = (pulse.fault_accounting_limit if pulse.fault_accounting_limit is not None
                     else self._possible_prefix(pulse))
            terminal = (pulse.ao_status is not None
                        and bool(np.all((pulse.ao_status == 2) | (pulse.ao_status == 3))))
            if pulse.accounted_samples >= limit and terminal:
                self._pulses.pop(pulse_id, None)

    def _render(self, pulse: _Pulse, first: int, count_samples: int, out: np.ndarray,
                pulse_offset_start: int = 0, pulse_limit: int | None = None) -> int:
        limit = pulse.waveform.size if pulse_limit is None else min(
            pulse.waveform.size, max(0, int(pulse_limit)))
        begin = max(first, pulse.onset_index + pulse_offset_start)
        offset = begin - first
        pulse_offset = begin - pulse.onset_index
        if offset >= count_samples or pulse_offset >= limit:
            return 0
        n = min(count_samples - offset, limit - pulse_offset)
        out[offset:offset + n] += pulse.waveform[pulse_offset:pulse_offset + n]
        pulse.render_cursor = max(pulse.render_cursor, pulse_offset + n)
        return n

    def process(self, action: ControllerAction, *, first_sample_index: int, sample_count: int,
                sample_rate_hz: float, armed: bool, validity_mask: np.ndarray,
                amplitude_v: float = 0.0, generation: int, scheduler_lead_samples: int = 0,
                latest_sample_index: int | None = None,
                controller_id: str | None = None) -> ActuatorResult:
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
            possible = (pulse.ao_status is not None and np.any(
                (pulse.ao_status == 1) | (pulse.ao_status == 2) | (pulse.ao_status == 4)))
            if not can_output and pulse.committed_samples <= 0 and not possible:
                continue
            rendered_start = max(first_sample_index, pulse.onset_index + pulse.render_cursor)
            n = self._render(pulse, first_sample_index, sample_count, out,
                             pulse_offset_start=pulse.render_cursor,
                             pulse_limit=pulse.fault_accounting_limit)
            if n:
                events.append(PulseEvent(pulse_id, pulse.requested_index, pulse.onset_index,
                                         pulse.target_onset_index, "committed_tail_rendered", pulse.generation, n,
                                         latest_used, pulse.controller_id, rendered_start))
        self._retire_accounted_pulses()
        if action.kind == "analog":
            if action.analog_v is None or np.asarray(action.analog_v).size != sample_count:
                raise ValueError("analog action must contain one voltage per sample")
            if can_output:
                out += np.asarray(action.analog_v, dtype=np.float64).reshape(-1)
            else:
                self.cancel_uncommitted()
            return ActuatorResult(out, tuple(events), 0.0, 0, "" if can_output else "analog output disarmed or invalid")
        if not can_output or action.kind not in {"pulse_rate", "pulse_events"}:
            self._reset_rate_phase()
            if not armed or not self.armed or not np.all(valid):
                self.cancel_uncommitted()
            return ActuatorResult(out, tuple(events), float(action.requested_rate_hz), 0,
                                  "disarmed, invalid feedback, or no pulse request")
        rate = float(action.requested_rate_hz) if action.kind == "pulse_rate" else 0.0
        if action.kind == "pulse_rate" and not np.isfinite(rate):
            self._reset_rate_phase()
            raise ValueError("pulse rate must be finite")
        if action.kind == "pulse_rate" and not 0 <= rate <= 2:
            self._reset_rate_phase()
            raise ValueError("pulse rate request must be in [0, 2] Hz")
        if action.kind == "pulse_rate":
            if rate == 0:
                self._reset_rate_phase()
                requests: list[int] = []
            else:
                increments = rate / sample_rate_hz
                phase_before = self.rate_phase
                phase = phase_before + increments * np.arange(1, sample_count + 1, dtype=np.float64)
                if phase.size:
                    self.rate_phase_high_water = max(self.rate_phase_high_water, float(np.max(phase)))
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
            if len(self._pulses) >= self.max_pending_pulses:
                events.append(PulseEvent(pulse_id, requested, requested, requested,
                                         "rejected_capacity", generation,
                                         latest_sample_index=latest_used, controller_id=controller_id))
                self._reset_rate_phase()
                continue
            if requested < first_sample_index:
                events.append(PulseEvent(pulse_id, requested, requested, requested, "rejected_expired", generation,
                                         latest_sample_index=latest_used, controller_id=controller_id))
                self._reset_rate_phase()
                continue
            if requested >= end:
                events.append(PulseEvent(pulse_id, requested, requested, requested, "rejected_future", generation,
                                         latest_sample_index=latest_used, controller_id=controller_id))
                self._reset_rate_phase()
                continue
            if self.last_onset is not None and requested - self.last_onset < min_interval:
                events.append(PulseEvent(pulse_id, requested, requested, requested, "rejected_refractory", generation,
                                         latest_sample_index=latest_used, controller_id=controller_id))
                self._reset_rate_phase()
                continue
            pulse = _Pulse(pulse_id, requested, requested, requested + scheduler_lead_samples,
                           generation, unit_wave * float(amplitude_v))
            pulse.controller_id = controller_id
            pulse.latest_sample_index = latest_used
            pulse.ao_status = np.zeros(pulse.waveform.size, dtype=np.uint8)
            self._pulses[pulse_id] = pulse
            plan_wave = pulse.waveform.copy()
            plan_wave.setflags(write=False)
            plans.append(PulsePlan(pulse_id, generation, requested, requested + scheduler_lead_samples,
                                   latest_used, plan_wave))
            n = self._render(pulse, first_sample_index, sample_count, out)
            rendered += 1
            self.last_onset = requested
            events.append(PulseEvent(pulse_id, requested, requested, requested + scheduler_lead_samples,
                                     "rendered_for_ao", generation, n, latest_used, controller_id))
        return ActuatorResult(out, tuple(events), rate, rendered, pulse_plans=tuple(plans))
