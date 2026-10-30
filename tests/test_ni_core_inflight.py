"""Actual-core regressions for AO writes that finish after source advances.

The fake device advances its sample counter on a wall-clock thread, while its
writer can pause a selected refill.  It deliberately models only the NI API
surface used by ScheduledAo; it does not generate analog output.
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

import echoChamber as app
from controllers import ControllerAction, ControllerDiagnostics, PulseActuator, StepResult
from controllers import PulseAoOutcome


class _NoUiCore(app.ProcessingCore):
    def _publish_ui(self, *args, **kwargs):
        pass


class _PulseController:
    def __init__(self, onset: int):
        self.onset = onset

    def reset(self, keep_calibration=False):
        pass

    def drain(self):
        pass

    def step(self, inp):
        request = (self.onset,) if inp.first_sample_index <= self.onset < inp.first_sample_index + 100 else ()
        return StepResult(
            ControllerAction(kind="pulse_events", pulse_sample_indices=request),
            ControllerDiagnostics("rule", True, True, 0.0, 0.0, 1.0, 0.0,
                                  3.0, 0.0, 0.0, "ready", inp.generation),
        )


@dataclass
class _Recorder:
    blocks: list[app.RecordBlock]
    stages: list[dict]

    def submit(self, block):
        self.blocks.append(block)

    def record_action_stage(self, **stage):
        self.stages.append(stage)


class _FakeNiTask:
    def __init__(self, name, generated):
        self.name = name
        self.generated = generated
        self.out_stream = self
        self.timing = type("Timing", (), {"cfg_samp_clk_timing": lambda *a, **k: None})()
        self.ao_channels = type("AoChannels", (), {"add_ao_voltage_chan": lambda *a, **k: None})()
        trigger = type("Trigger", (), {"cfg_dig_edge_start_trig": lambda *a, **k: None})()
        self.triggers = type("Triggers", (), {"start_trigger": trigger})()
        self.started = False

    @property
    def total_samp_per_chan_generated(self):
        if self.started:
            self.generated[0] += 100
        return self.generated[0]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def start(self):
        self.started = True

    def write(self, *args, **kwargs):
        pass


class _BlockingNiWriter:
    def __init__(self, pause_call: int, fail_call: int | None = None,
                 written_count: int | None = None):
        self.pause_call = pause_call
        self.fail_call = fail_call
        self.written_count = written_count
        self.calls = 0
        self.blocked = threading.Event()
        self.release = threading.Event()
        self.writes: list[np.ndarray] = []
        self.lock = threading.Lock()

    def write_many_sample(self, values, timeout=None):
        array = np.asarray(values, dtype=np.float64).copy()
        with self.lock:
            self.calls += 1
            call = self.calls
        if call == self.pause_call:
            self.blocked.set()
            if not self.release.wait(8.0):
                raise TimeoutError(f"test did not release blocked NI write call {call}")
        with self.lock:
            self.writes.append(array)
        if call == self.fail_call:
            error = OSError("fake NI write returned an ambiguous post-transfer error")
            if self.written_count is not None:
                error.samps_per_chan_written = int(self.written_count)
            raise error
        return int(array.size)


def _fixture(onset: int, pause_call: int, *, fail_call: int | None = None,
             written_count: int | None = None, defer_outcomes: bool = False):
    config = app.AppConfig(processing_block_ms=5.0, ao_target_lead_ms=100.0)
    config.validate()
    state = app.RuntimeState(
        esn_ready=False, start_paused=False, electrode_labels=("CA3", "EC"), ctx_index=1,
        pulse_threshold_std=3.0, pulse_window_sec=10.0, controller="rule",
        controller_ready={"esn": False, "rule": True, "pid": True},
    )
    state.set_stim("threshold_pulse", 0.5)
    state.set_mode("closed-loop")
    state.acknowledge_drain(state.snapshot().generation)
    state.update_controller_calibration("rule", True)
    state.set_armed(True)
    state.set_recording(True)
    recorder = _Recorder([], [])
    core = _NoUiCore(config, state, app.EsnRuntime(None, config), recorder,
                     object(), None, app.Telemetry())
    core.controllers["rule"] = _PulseController(onset)
    callbacks: list[tuple[str, int]] = []
    pulse_outcomes: list[app.PulseAoOutcome] = []

    def commit(pulse_id, count):
        callbacks.append((pulse_id, count))
        core.commit_actuator_event(pulse_id, count)

    def publish_outcome(outcome):
        pulse_outcomes.append(outcome)
        if not defer_outcomes:
            core.enqueue_ao_outcome(outcome)

    scheduler = app.ScheduledAo(config, state, "/Dev1/ai/StartTrigger",
                                recorder=recorder, commit_event=commit,
                                ao_outcome=core.record_ao_outcome,
                                pulse_outcome=publish_outcome)
    core.set_local_account_sink(scheduler.enqueue_local_account)
    generated = [0]
    writer = _BlockingNiWriter(pause_call, fail_call=fail_call, written_count=written_count)
    return config, state, recorder, core, scheduler, callbacks, pulse_outcomes, writer, generated


def _process_submit(config, core, scheduler, source_index: int):
    _, safe, _, reason = core.process(
        np.zeros((2, config.chunk_size), dtype=np.float64), sample_index=source_index,
        ao_target_sample_index=source_index + config.ao_lead_samples,
        input_is_electrode_mV=True, input_roles=("CA3", "Cortex"),
        input_labels=("CA3", "EC"), input_ctx_index=1,
    )
    assert reason is None
    target = source_index // config.chunk_size + config.ao_lead_chunks
    accepted = scheduler.submit(target, safe.reshape(-1), core.last_processed_generation,
                                core.last_pulse_events, core.last_admitted_pulse_plans)
    return safe.reshape(-1), accepted


def _stage_until(config, core, scheduler, last_source: int):
    for source_index in range(0, last_source + 1, config.chunk_size):
        _, accepted = _process_submit(config, core, scheduler, source_index)
        assert accepted, f"could not stage source block {source_index}"


def _start_scheduler(config, state, scheduler, generated, writer, monkeypatch):
    monkeypatch.setattr(app.nidaqmx, "Task",
                        lambda name: _FakeNiTask(name, generated))
    monkeypatch.setattr(app, "AnalogSingleChannelWriter", lambda stream, auto_start=False: writer)
    scheduler.start()
    scheduler.notify_clock_started()


def _wait_for(predicate, message: str, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    assert predicate(), message


def _assert_safety_equal(actual, expected):
    assert actual.previous_v == expected.previous_v
    assert tuple(actual.area_history) == tuple(expected.area_history)
    assert tuple(actual.active_history) == tuple(expected.active_history)
    assert actual.area_sum_v_s == expected.area_sum_v_s
    assert actual.area_sum_compensation == expected.area_sum_compensation
    assert actual.active_count == expected.active_count
    assert actual.consecutive_active == expected.consecutive_active


def _full_pulse_reference(config, plan, end_sample_index: int):
    safety = app.StimulusSafetyAdapter(config.safety, config.sample_rate)
    onset = int(plan.source_onset_sample_index)
    plan_end = onset + int(plan.waveform_v.size)
    timeline_end = max(plan_end, int(end_sample_index))
    for block_start in range(0, timeline_end, config.chunk_size):
        block = np.zeros(config.chunk_size)
        overlap_start = max(onset, block_start)
        overlap_end = min(plan_end, block_start + config.chunk_size)
        if overlap_end > overlap_start:
            block[overlap_start - block_start:overlap_end - block_start] = (
                plan.waveform_v[overlap_start - onset:overlap_end - onset]
            )
        safety.process(block)
    return safety


def _accounted_prefix_reference(config, plan, accounted_samples: int):
    safety = app.StimulusSafetyAdapter(config.safety, config.sample_rate)
    onset = int(plan.source_onset_sample_index)
    end = onset + int(accounted_samples)
    for block_start in range(0, 1_100, config.chunk_size):
        block = np.zeros(config.chunk_size)
        overlap_start = max(onset, block_start)
        overlap_end = min(end, block_start + config.chunk_size)
        if overlap_end > overlap_start:
            block[overlap_start - block_start:overlap_end - block_start] = (
                plan.waveform_v[overlap_start - onset:overlap_end - onset]
            )
        safety.process(block)
    return safety


def test_core_source_tail_during_reserved_full_pulse_write_is_not_lost_or_rewritten(
        monkeypatch):
    config, state, recorder, core, scheduler, callbacks, pulse_outcomes, writer, generated = _fixture(880, 2)
    try:
        _stage_until(config, core, scheduler, 800)
        plan = core.last_admitted_pulse_plans[0]
        assert plan.target_onset_sample_index == 2_880
        assert plan.waveform_v.size == 100
        _start_scheduler(config, state, scheduler, generated, writer, monkeypatch)
        assert writer.blocked.wait(3.0), "first scheduled refill did not reach blocked writer"

        state.set_armed(False)
        local_tail, accepted = _process_submit(config, core, scheduler, 900)
        recorded = next(block for block in recorder.blocks if block.sample_index == 900)
        reserved_ranges = {(value.pulse_id, value.start_sample_index, value.end_sample_index)
                           for value in pulse_outcomes if value.status == "reserved"}
        expected_ranges = {(plan.pulse_id, 2_880, 2_900), (plan.pulse_id, 2_900, 2_980)}
        problems = []
        if any(value.pulse_id == plan.pulse_id
               and (value.source_onset_sample_index != 880
                    or value.target_onset_sample_index != 2_880)
               for value in pulse_outcomes):
            problems.append("typed scheduler outcome lost the original source/target onset coordinates")
        if not expected_ranges.issubset(reserved_ranges):
            problems.append(f"blocked writer did not expose exact reserved pulse ranges: {reserved_ranges}")
        if np.any(local_tail) or np.any(recorded.safe_ao):
            problems.append("unresolved in-flight output escaped as a new local command while disarmed")
        if not accepted:
            problems.append("source block 900's zero result was rejected instead of recognized as non-authorizing")
        if callbacks:
            problems.append(f"legacy observer callback ran despite typed outcome ownership: {callbacks}")
        pending_reference = _full_pulse_reference(config, plan, 1_000)
        try:
            _assert_safety_equal(core.safety, pending_reference)
        except AssertionError:
            problems.append("pending possible-output waveform was not conservatively included in safety history")
        if not any(stage.get("stage") == "possible_accounted"
                   and stage.get("request_id") == plan.pulse_id for stage in recorder.stages):
            problems.append("source accounting did not record separate possible-output provenance")
        next_source, next_accepted = _process_submit(config, core, scheduler, 1_000)
        next_recorded = next(block for block in recorder.blocks if block.sample_index == 1_000)
        if np.any(next_source) or np.any(next_recorded.safe_ao):
            problems.append("disarmed source block1000 emitted a new command while the reserved write was pending")
        if not next_accepted:
            problems.append("zero source block1000 was rejected during the pending drain")

        # The first refill owns blocks 20..29; target 29 contains the final 80
        # pulse samples. The duplicate source submission may reconcile that
        # exact reservation, but may not replace it or cause a second write.
        writer.release.set()
        _wait_for(lambda: scheduler._next_block >= 30, "reserved refill never completed")
        _wait_for(lambda: len([value for value in pulse_outcomes
                              if value.pulse_id == plan.pulse_id and value.status == "committed"]) >= 2,
                  "successful driver write did not publish both pulse intersections")
        after_commit = _process_core_only(config, core, 1_100)
        if np.any(after_commit):
            problems.append("late write confirmation authorized new source output")
        expected = _full_pulse_reference(config, plan, 1_200)
        try:
            _assert_safety_equal(core.safety, expected)
        except AssertionError:
            problems.append("live safety history did not account for the complete possible-output plan exactly once")

        writes = tuple(writer.writes)
        assert len(writes) >= 2
        refill = writes[1]
        target_28_offset = (28 - 20) * config.chunk_size
        target_29_offset = (29 - 20) * config.chunk_size
        target_28 = refill[target_28_offset:target_28_offset + config.chunk_size]
        target_29 = refill[target_29_offset:target_29_offset + config.chunk_size]
        expected_28 = np.zeros(config.chunk_size)
        expected_28[80:] = plan.waveform_v[:20]
        expected_29 = np.zeros(config.chunk_size)
        expected_29[:80] = plan.waveform_v[20:]
        if not np.array_equal(target_28, expected_28):
            problems.append("the blocked batch changed the original target-28 pulse segment")
        if not np.array_equal(target_29, expected_29):
            problems.append("the blocked batch changed or lost the original target-29 pulse tail")
        if scheduler.error is not None:
            problems.append(f"AO scheduler errored: {scheduler.error!r}")
        stats = scheduler.stats.snapshot()
        committed_ranges = {(value.pulse_id, value.start_sample_index, value.end_sample_index)
                            for value in pulse_outcomes if value.status == "committed"}
        if not expected_ranges.issubset(committed_ranges):
            problems.append(f"successful write did not confirm exact reserved ranges: {pulse_outcomes!r}")
        observed_transitions = Counter((value.status, value.pulse_id,
                                        value.start_sample_index, value.end_sample_index)
                                       for value in pulse_outcomes)
        for pulse_id, start, end in sorted(expected_ranges):
            if observed_transitions[("reserved", pulse_id, start, end)] != 1:
                problems.append(f"expected one reserved outcome for {(pulse_id, start, end)}: {observed_transitions}")
            if observed_transitions[("committed", pulse_id, start, end)] != 1:
                problems.append(f"expected one committed outcome for {(pulse_id, start, end)}: {observed_transitions}")
        if stats["late_commands"] != 0 or stats["fault_rejections"] != 0:
            problems.append(f"reserved pulse was misclassified as late/stale: {stats}")
        if callbacks:
            problems.append(f"legacy observer callbacks duplicated typed commit outcomes: {callbacks}")
        if len(writer.writes[1]) != 1_000:
            problems.append(f"first refill size changed from ten blocks: {len(writer.writes[1])}")
        if any(np.any(values) for values in writer.writes[2:]):
            problems.append("a later writer call emitted a duplicate nonzero pulse")
        assert not problems, "\n".join(problems)
    finally:
        writer.release.set()
        scheduler.stop()


def test_fault_during_inflight_suffix_keeps_possible_tail_history_but_zeroes_output(
        monkeypatch):
    # The 100-sample plan begins at source sample 980, hence target sample
    # 2980: 20 samples are in block 29 and 80 in block 30. Refill call 2 owns
    # blocks 20..29; call 3 naturally owns blocks 30..39.
    config, state, recorder, core, scheduler, callbacks, pulse_outcomes, writer, generated = _fixture(980, 3)
    try:
        _stage_until(config, core, scheduler, 900)
        plan = core.last_admitted_pulse_plans[0]
        assert plan.source_onset_sample_index == 980
        assert plan.target_onset_sample_index == 2_980
        assert plan.target_onset_sample_index % config.chunk_size == 80
        _start_scheduler(config, state, scheduler, generated, writer, monkeypatch)
        assert writer.blocked.wait(3.0), (
            f"second scheduled refill did not reach blocked writer: calls={writer.calls}, "
            f"generated={generated[0]}, stats={scheduler.stats.snapshot()}, error={scheduler.error!r}"
        )
        _wait_for(lambda: scheduler._committed_pulse_samples.get(plan.pulse_id, 0) == 20,
                  "first 20 pulse samples were not committed before suffix refill")
        assert writer.calls == 3

        state.fault("fault while safe pulse suffix write was in flight")
        safe, accepted = _process_submit(config, core, scheduler, 1_000)
        assert not np.any(safe)
        source_900_before = next(block for block in recorder.blocks if block.sample_index == 900)
        source_900_copy = source_900_before.safe_ao.copy()
        assert not any(block.sample_index == 1_000 for block in recorder.blocks)
        assert state.snapshot().fault == "fault while safe pulse suffix write was in flight"
        assert any(value.pulse_id == plan.pulse_id and value.generation == plan.generation
                   and value.start_sample_index == 3_000 and value.end_sample_index == 3_080
                   and value.status == "reserved"
                   and value.source_onset_sample_index == 980
                   and value.target_onset_sample_index == 2_980
                   for value in pulse_outcomes)
        possible_reference = _full_pulse_reference(config, plan, 1_100)
        _assert_safety_equal(core.safety, possible_reference)
        assert any(stage.get("stage") == "possible_accounted"
                   and stage.get("request_id") == plan.pulse_id for stage in recorder.stages)

        # The 80 samples are already in the NI writer call. A later success
        # confirms possible output; it cannot authorize a nonzero source return
        # or cause the source safety interval to be replayed/rewritten.
        writer.release.set()
        _wait_for(lambda: scheduler._committed_pulse_samples.get(plan.pulse_id, 0) == 100,
                  "successful in-flight suffix was not acknowledged")
        _process_submit(config, core, scheduler, 1_100)
        expected = _full_pulse_reference(config, plan, 1_200)
        try:
            _assert_safety_equal(core.safety, expected)
        except AssertionError:
            problems.append("successful outcome confirmation replayed or lost already-accounted source samples")
        assert np.array_equal(source_900_before.safe_ao, source_900_copy)
        assert state.snapshot().fault == "fault while safe pulse suffix write was in flight"
        assert not core._feedback.accepted
        assert any(value.pulse_id == plan.pulse_id and value.start_sample_index == 3_000
                   and value.end_sample_index == 3_080 and value.status == "committed"
                   for value in pulse_outcomes)
        committed_ranges = {(value.start_sample_index, value.end_sample_index)
                            for value in pulse_outcomes
                            if value.pulse_id == plan.pulse_id and value.status == "committed"}
        assert committed_ranges == {(2_980, 3_000), (3_000, 3_080)}
        assert sum(end - start for start, end in committed_ranges) == 100
        assert scheduler.error is None
        # Existing event history must make the uncertain interval distinguishable
        # from a post-write confirmation without writing it into source output.
        assert callbacks == []
        assert np.any(writer.writes[2])
    finally:
        writer.release.set()
        scheduler.stop()


def test_fault_before_any_pulse_commit_excludes_uncommitted_source_suffix(monkeypatch):
    config, state, recorder, core, scheduler, callbacks, pulse_outcomes, writer, generated = _fixture(980, 3)
    _stage_until(config, core, scheduler, 900)
    plan = core.last_admitted_pulse_plans[0]
    assert plan.source_onset_sample_index == 980
    assert scheduler._committed_pulse_samples.get(plan.pulse_id, 0) == 0
    assert np.any(scheduler.commands[30][0])

    state.fault("fault before pulse target was reserved or committed")
    try:
        _start_scheduler(config, state, scheduler, generated, writer, monkeypatch)
        _wait_for(lambda: any(value.pulse_id == plan.pulse_id
                              and value.start_sample_index == 3_000
                              and value.end_sample_index == 3_080
                              and value.status == "cancelled" for value in pulse_outcomes),
                  "scheduler did not authoritatively cancel the not-yet-reserved suffix")
        # Source block 900 already passed its first 20 samples through live
        # safety; its not-yet-processed suffix is definitively cancelled before
        # block1000 is accounted.
        safe = _process_core_only(config, core, 1_000)
        expected = _accounted_prefix_reference(config, plan, accounted_samples=20)

        assert not np.any(safe)
        _assert_safety_equal(core.safety, expected)
        assert scheduler._committed_pulse_samples.get(plan.pulse_id, 0) == 0
        assert callbacks == []
        assert state.snapshot().fault == "fault before pulse target was reserved or committed"
        assert not core._feedback.accepted
    finally:
        writer.release.set()
        scheduler.stop()


def test_failed_inflight_write_is_unknown_not_committed_or_cancelled(monkeypatch):
    config, state, recorder, core, scheduler, callbacks, pulse_outcomes, writer, generated = (
        _fixture(980, 3, fail_call=3)
    )
    try:
        _stage_until(config, core, scheduler, 900)
        plan = core.last_admitted_pulse_plans[0]
        _start_scheduler(config, state, scheduler, generated, writer, monkeypatch)
        assert writer.blocked.wait(3.0), "suffix write did not reach the injected failure point"
        _wait_for(lambda: scheduler._committed_pulse_samples.get(plan.pulse_id, 0) == 20,
                  "first 20 pulse samples were not confirmed before the ambiguous write")

        state.fault("fault while suffix writer outcome is unknown")
        safe = _process_core_only(config, core, 1_000)
        assert not np.any(safe)
        _assert_safety_equal(core.safety, _full_pulse_reference(config, plan, 1_100))

        writer.release.set()
        _wait_for(lambda: scheduler.error is not None, "injected NI write error was not surfaced")
        unknown = [value for value in pulse_outcomes
                   if value.pulse_id == plan.pulse_id and value.start_sample_index == 3_000
                   and value.end_sample_index == 3_080]
        assert any(value.status == "write_unknown" for value in unknown)
        assert all(value.source_onset_sample_index == 980
                   and value.target_onset_sample_index == 2_980 for value in unknown)
        assert not any(value.status == "cancelled" for value in unknown)

        safe_after_error = _process_core_only(config, core, 1_100)
        assert not np.any(safe_after_error)
        _assert_safety_equal(core.safety, _full_pulse_reference(config, plan, 1_200))
        assert scheduler._committed_pulse_samples.get(plan.pulse_id, 0) == 20
        assert any(stage.get("stage") == "ao_write_unknown"
                   and stage.get("request_id") == plan.pulse_id for stage in recorder.stages)
        assert writer.writes[2].size == config.chunk_size
        assert np.array_equal(writer.writes[2][:80], plan.waveform_v[20:])
        assert np.all(writer.writes[2][80:] == 0.0)
        assert callbacks == []
        assert state.snapshot().fault is not None
        assert "OSError" in state.snapshot().fault or "unknown" in state.snapshot().fault.lower()
    finally:
        writer.release.set()
        scheduler.stop()


def test_partial_ni_write_commits_only_reported_prefix_and_leaves_unknown_suffix(monkeypatch):
    config, state, recorder, core, scheduler, callbacks, pulse_outcomes, writer, generated = (
        _fixture(980, 3, fail_call=3, written_count=50)
    )
    try:
        _stage_until(config, core, scheduler, 900)
        plan = core.last_admitted_pulse_plans[0]
        _start_scheduler(config, state, scheduler, generated, writer, monkeypatch)
        assert writer.blocked.wait(3.0), "suffix write did not reach the injected partial result"
        _wait_for(lambda: scheduler._committed_pulse_samples.get(plan.pulse_id, 0) == 20,
                  "first pulse segment was not committed before partial suffix write")

        state.fault("fault while NI writer returns a confirmed prefix")
        safe = _process_core_only(config, core, 1_000)
        assert not np.any(safe)
        _assert_safety_equal(core.safety, _full_pulse_reference(config, plan, 1_100))

        writer.release.set()
        _wait_for(lambda: scheduler.error is not None, "partial write error was not surfaced")
        prefix = [value for value in pulse_outcomes if value.pulse_id == plan.pulse_id
                  and value.start_sample_index == 3_000 and value.end_sample_index == 3_050]
        suffix = [value for value in pulse_outcomes if value.pulse_id == plan.pulse_id
                  and value.start_sample_index == 3_050 and value.end_sample_index == 3_080]
        assert any(value.status == "committed" for value in prefix)
        assert any(value.status == "write_unknown" for value in suffix)
        assert not any(value.status == "cancelled" for value in prefix + suffix)

        safe_after_error = _process_core_only(config, core, 1_100)
        assert not np.any(safe_after_error)
        _assert_safety_equal(core.safety, _full_pulse_reference(config, plan, 1_200))
        assert scheduler._committed_pulse_samples.get(plan.pulse_id, 0) == 70
        assert any(stage.get("stage") == "ao_commit" and stage.get("request_id") == plan.pulse_id
                   for stage in recorder.stages)
        assert any(stage.get("stage") == "ao_write_unknown" and stage.get("request_id") == plan.pulse_id
                   for stage in recorder.stages)
        assert writer.writes[2].size == config.chunk_size
        assert callbacks == []
    finally:
        writer.release.set()
        scheduler.stop()


def test_reordered_duplicate_range_outcomes_are_idempotent_in_actual_core():
    config, state, _, core, _, _, _, _, _ = _fixture(880, 2)
    _stage_until(config, core, _NoopSubmitScheduler(config, state), 800)
    plan = core.last_admitted_pulse_plans[0]
    state.set_armed(False)

    # A driver completion can reach the core before an earlier queued
    # reservation notification. Duplicate completion deliveries are also
    # possible across retrying event transports. Neither may downgrade or
    # account the same source interval twice.
    outcomes = (
        PulseAoOutcome(plan.pulse_id, plan.generation, 2_880, 2_900, "committed", "test completion"),
        PulseAoOutcome(plan.pulse_id, plan.generation, 2_900, 2_980, "committed", "test completion"),
        PulseAoOutcome(plan.pulse_id, plan.generation, 2_880, 2_900, "reserved", "delayed reservation"),
        PulseAoOutcome(plan.pulse_id, plan.generation, 2_900, 2_980, "reserved", "delayed reservation"),
        PulseAoOutcome(plan.pulse_id, plan.generation, 2_900, 2_980, "committed", "duplicate completion"),
    )
    for outcome in outcomes:
        core.enqueue_ao_outcome(outcome)

    safe = _process_core_only(config, core, 900)
    _assert_safety_equal(core.safety, _full_pulse_reference(config, plan, 1_000))
    assert not np.any(safe)
    assert not core._feedback.accepted
    assert not state.snapshot().fault


def test_admitted_safe_plan_is_possible_before_reservation_notification(monkeypatch):
    """A blocked writer may reserve a plan before its outcome reaches the core."""
    config, state, recorder, core, scheduler, callbacks, pulse_outcomes, writer, generated = (
        _fixture(880, 2, defer_outcomes=True)
    )
    try:
        _stage_until(config, core, scheduler, 800)
        plan = core.last_admitted_pulse_plans[0]
        _start_scheduler(config, state, scheduler, generated, writer, monkeypatch)
        assert writer.blocked.wait(3.0), "first scheduled refill did not reach blocked writer"

        # The callback has been captured by the harness but deliberately not
        # enqueued to ProcessingCore. This models a delayed/batched owner event.
        reserved = [outcome for outcome in pulse_outcomes
                    if outcome.pulse_id == plan.pulse_id and outcome.status == "reserved"]
        assert {(outcome.start_sample_index, outcome.end_sample_index) for outcome in reserved} == {
            (2_880, 2_900), (2_900, 2_980)
        }
        assert core._pulse_ao_outcomes.empty()

        state.set_armed(False)
        safe, accepted = _process_submit(config, core, scheduler, 900)
        recorded = next(block for block in recorder.blocks if block.sample_index == 900)
        expected = _full_pulse_reference(config, plan, 1_000)
        problems = []
        if np.any(safe) or np.any(recorded.safe_ao):
            problems.append("unresolved plan was returned as new output while disarmed")
        try:
            _assert_safety_equal(core.safety, expected)
        except AssertionError:
            problems.append("admitted safe plan tail was not conservatively accounted before reservation notification")
        if not any(stage.get("stage") == "possible_accounted"
                   and stage.get("request_id") == plan.pulse_id for stage in recorder.stages):
            problems.append("unresolved plan lacks separate possible_accounted provenance")
        if not accepted:
            problems.append("zero submission for the accounting-only source block was rejected")

        # Let the blocked driver write succeed while its reserved and committed
        # callbacks are still withheld from the core. Then deliver both outcomes
        # only after source 900 has advanced the safety timeline.
        writer.release.set()
        _wait_for(lambda: scheduler._committed_pulse_samples.get(plan.pulse_id, 0) == 100,
                  "blocked full pulse write did not complete")
        _wait_for(lambda: sum(outcome.pulse_id == plan.pulse_id
                              and outcome.status == "committed" for outcome in pulse_outcomes) == 2,
                  "successful write did not publish both committed ranges")
        for outcome in pulse_outcomes:
            core.enqueue_ao_outcome(outcome)
        safe_later = _process_core_only(config, core, 1_000)
        if np.any(safe_later):
            problems.append("late outcome confirmation authorized a new source command")
        try:
            _assert_safety_equal(core.safety, _full_pulse_reference(config, plan, 1_100))
        except AssertionError:
            problems.append("late reservation/commit confirmation replayed or lost source safety history")
        assert not state.snapshot().fault
        assert not core._feedback.accepted
        assert not core._actuator_commits.qsize()
        assert callbacks == []
        assert not problems, "\n".join(problems)
    finally:
        writer.release.set()
        scheduler.stop()


def test_authoritatively_cancelled_suffixes_do_not_exhaust_pending_plan_capacity():
    """Resolved canceled AO suffixes need no future source output accounting."""
    actuator = PulseActuator(pulse_duration_ms=5.0, pulse_frequency_hz=100.0,
                             refractory_s=0.0, waveform="sine")
    actuator.arm()
    validity = np.ones(100, dtype=bool)
    plans_seen = []

    # Each source block accounts the first 20 samples of a pulse. The AO owner
    # confirms that prefix and authoritatively cancels the remaining 80 before
    # the next source interval is processed. The canceled suffix must not keep
    # consuming one of the bounded 64 admitted-plan slots.
    for index in range(70):
        source_start = index * 100
        result = actuator.process(
            ControllerAction(kind="pulse_events", pulse_sample_indices=(source_start,)),
            first_sample_index=source_start, sample_count=100, sample_rate_hz=20_000.0,
            armed=True, validity_mask=validity, amplitude_v=0.5, generation=index,
            scheduler_lead_samples=2_000,
        )
        assert len(result.pulse_plans) == 1, f"plan {index} was rejected: {result.pulse_events!r}"
        plan = result.pulse_plans[0]
        plans_seen.append(plan.pulse_id)
        actuator.apply_ao_outcome(PulseAoOutcome(
            plan.pulse_id, index, plan.target_onset_sample_index,
            plan.target_onset_sample_index + 20, "committed",
            "confirmed prefix", plan.source_onset_sample_index,
            plan.target_onset_sample_index,
        ))
        actuator.apply_ao_outcome(PulseAoOutcome(
            plan.pulse_id, index, plan.target_onset_sample_index + 20,
            plan.target_onset_sample_index + 100, "cancelled",
            "uncommitted suffix canceled", plan.source_onset_sample_index,
            plan.target_onset_sample_index,
        ))
        actuator.mark_source_accounted(source_start, 20, committed_only=True)

    assert len(plans_seen) == 70
    assert len(set(plans_seen)) == 70
    # The subsequent source interval contains no output for a canceled suffix;
    # advancing it may age zero history but must not be required to free plans.
    assert len(actuator._pulses) == 0


def test_h5_ao_outcome_preserves_original_origin_and_both_absolute_range_bounds(tmp_path):
    """Outcome provenance survives a controller switch and a nonempty reason."""
    import h5py

    config = app.AppConfig(record_dir=tmp_path, processing_block_ms=5.0)
    config.validate()
    state = app.RuntimeState(
        esn_ready=False, start_paused=False, electrode_labels=("CA3", "EC"), ctx_index=1,
        pulse_threshold_std=3.0, pulse_window_sec=10.0, controller="rule",
        controller_ready={"esn": False, "rule": True, "pid": True},
    )
    state.set_stim("threshold_pulse", 0.5)
    state.set_mode("closed-loop")
    state.acknowledge_drain(state.snapshot().generation)
    state.update_controller_calibration("rule", True)
    state.set_armed(True)
    state.set_recording(True)
    recorder = app.H5Recorder(config, {"test": "pulse-origin-provenance"})
    path = recorder.start("pulse-origin")
    core = _NoUiCore(config, state, app.EsnRuntime(None, config), recorder,
                     object(), None, app.Telemetry())
    core.controllers["rule"] = _PulseController(80)
    try:
        _, first_safe, _, reason = core.process(
            np.zeros((2, config.chunk_size), dtype=np.float64), sample_index=0,
            ao_target_sample_index=config.ao_lead_samples,
            input_is_electrode_mV=True, input_roles=("CA3", "Cortex"),
            input_labels=("CA3", "EC"), input_ctx_index=1,
        )
        assert reason is None
        plan = core.last_admitted_pulse_plans[0]
        assert (plan.generation, plan.source_onset_sample_index,
                plan.target_onset_sample_index, plan.latest_sample_index) == (
                    state.snapshot().generation, 80, 2_080, 99)
        core.enqueue_ao_outcome(PulseAoOutcome(
            plan.pulse_id, plan.generation, plan.target_onset_sample_index,
            plan.target_onset_sample_index + plan.waveform_v.size, "reserved",
            "writer accepted the original plan", plan.source_onset_sample_index,
            plan.target_onset_sample_index, controller_id="rule",
            latest_sample_index=plan.latest_sample_index,
        ))
        assert state.set_controller("pid")
        _, second_safe, _, reason = core.process(
            np.zeros((2, config.chunk_size), dtype=np.float64), sample_index=100,
            ao_target_sample_index=100 + config.ao_lead_samples,
            input_is_electrode_mV=True, input_roles=("CA3", "Cortex"),
            input_labels=("CA3", "EC"), input_ctx_index=1,
        )
        assert reason is None
        assert np.any(first_safe) and not np.any(second_safe)
    finally:
        recorder.close()

    with h5py.File(path, "r") as recording:
        rows = recording["events/action_stages"][:]
    reserved = [row for row in rows
                if row["stage"].decode().rstrip("\x00") == "ao_reserved"]
    possible = [row for row in rows
                if row["stage"].decode().rstrip("\x00") == "possible_accounted"]
    assert len(reserved) == 1, rows
    assert len(possible) == 1, rows
    row = reserved[0]
    fields = set(rows.dtype.names or ())
    problems = []
    if "range_end_sample_index" not in fields:
        problems.append("sparse action-stage schema has no absolute AO range end field")
    if int(row["sample_index"]) != 2_080:
        problems.append("sample_index is not the absolute AO range start")
    if "range_end_sample_index" in fields and int(row["range_end_sample_index"]) != 2_180:
        problems.append("AO range end is not the exclusive absolute sample 2180")
    if int(row["generation"]) != plan.generation:
        problems.append("AO outcome was relabeled with the current controller generation")
    if row["controller"].decode().rstrip("\x00") != "rule":
        problems.append("AO outcome was relabeled with the current controller instead of its origin")
    if int(row["source_sample_index"]) != 80:
        problems.append("source onset was not preserved as a source coordinate")
    if int(row["target_sample_index"]) != 2_080:
        problems.append("AO target onset was not preserved separately from source onset")
    if int(row["latest_sample_index"]) != 99:
        problems.append("latest source sample used by the pulse proposal was not preserved")
    if row["reason"].decode().rstrip("\x00") != "writer accepted the original plan":
        problems.append("nonempty outcome reason was lost while encoding AO range")
    possible_row = possible[0]
    if int(possible_row["generation"]) != plan.generation:
        problems.append("possible-accounted provenance used the switched controller generation")
    if possible_row["controller"].decode().rstrip("\x00") != "rule":
        problems.append("possible-accounted provenance used the switched controller identity")
    if int(possible_row["source_sample_index"]) != 80:
        problems.append("possible-accounted source onset was mislabeled as its source interval")
    if int(possible_row["target_sample_index"]) != 2_080:
        problems.append("possible-accounted target onset was recomputed from current lead")
    if int(possible_row["sample_index"]) != 2_100:
        problems.append("possible-accounted sample_index is not AO range start for the source tail")
    if "range_end_sample_index" in fields and int(possible_row["range_end_sample_index"]) != 2_180:
        problems.append("possible-accounted AO range end does not match the admitted tail")
    assert not problems, "\n".join(problems)


class _NoopSubmitScheduler:
    """Fixture-only sink to exercise the normal core submit API without a thread."""

    def __init__(self, config, state):
        self.config = config
        self.state = state

    def submit(self, target_block, command, generation, events=(), plans=()):
        return True


def _process_core_only(config, core, source_index: int):
    _, safe, _, reason = core.process(
        np.zeros((2, config.chunk_size), dtype=np.float64), sample_index=source_index,
        ao_target_sample_index=source_index + config.ao_lead_samples,
        input_is_electrode_mV=True, input_roles=("CA3", "Cortex"),
        input_labels=("CA3", "EC"), input_ctx_index=1,
    )
    assert reason is None
    return safe.reshape(-1)
