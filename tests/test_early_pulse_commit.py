"""Focused regression reproduction for early AO commitment vs source rendering."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import echoChamber as app
from controllers import ControllerAction, ControllerDiagnostics, StepResult


class _NoUiCore(app.ProcessingCore):
    def _publish_ui(self, *args, **kwargs):
        pass


class _SinglePulseController:
    """Emit one deterministic request at source sample 880."""

    def reset(self, keep_calibration: bool = False) -> None:
        pass

    def drain(self) -> None:
        pass

    def step(self, inp):
        requests = (880,) if inp.first_sample_index == 800 else ()
        return StepResult(
            ControllerAction(kind="pulse_events", pulse_sample_indices=requests),
            ControllerDiagnostics(
                "rule", True, True, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0,
                "ready", inp.generation,
            ),
        )


@dataclass
class _RecorderCapture:
    blocks: list[app.RecordBlock]
    stages: list[dict] | None = None

    def __post_init__(self):
        if self.stages is None:
            self.stages = []

    def submit(self, block: app.RecordBlock) -> None:
        self.blocks.append(block)

    def record_action_stage(self, **stage) -> None:
        self.stages.append(stage)


def _make_core_and_scheduler(*, fault_on_first_partial_commit: bool = False):
    config = app.AppConfig(processing_block_ms=5.0, ao_target_lead_ms=100.0)
    config.validate()
    state = app.RuntimeState(
        esn_ready=False, start_paused=False,
        electrode_labels=("CA3", "EC"), ctx_index=1,
        pulse_threshold_std=3.0, pulse_window_sec=10.0,
        controller="rule", controller_ready={"esn": False, "rule": True, "pid": True},
    )
    state.set_stim("threshold_pulse", 0.5)
    state.set_mode("closed-loop")
    state.acknowledge_drain(state.snapshot().generation)
    state.update_controller_calibration("rule", True)
    state.set_armed(True)
    state.set_recording(True)

    recorder = _RecorderCapture([])
    core = _NoUiCore(
        config, state, app.EsnRuntime(None, config), recorder, object(), None, app.Telemetry(),
    )
    core.controllers["rule"] = _SinglePulseController()
    commit_callbacks: list[tuple[str, int]] = []

    def commit(pulse_id: str, samples: int) -> None:
        commit_callbacks.append((pulse_id, samples))
        core.commit_actuator_event(pulse_id, samples)

    def pulse_outcome(outcome: app.PulseAoOutcome) -> None:
        core.enqueue_ao_outcome(outcome)
        if (fault_on_first_partial_commit and outcome.status == "committed"
                and outcome.start_sample_index == 2_880
                and outcome.end_sample_index == 2_900
                and state.snapshot().fault is None):
            state.fault("fault injected after the first 20 pulse samples committed")

    scheduler = app.SimulatedAoScheduler(
        config, state, recorder=recorder, commit_event=commit,
        ao_outcome=core.record_ao_outcome,
        pulse_outcome=pulse_outcome,
    )
    core.set_local_account_sink(scheduler.enqueue_local_account)
    return config, state, recorder, core, scheduler, commit_callbacks


def _process_and_submit(config, core, scheduler, source_index: int):
    block = np.zeros((2, config.chunk_size), dtype=np.float64)
    _, safe, _, reason = core.process(
        block, sample_index=source_index,
        ao_target_sample_index=source_index + config.ao_lead_samples,
        input_is_electrode_mV=True,
        input_roles=("CA3", "Cortex"), input_labels=("CA3", "EC"), input_ctx_index=1,
    )
    assert reason is None
    target_block = source_index // config.chunk_size + config.ao_lead_chunks
    accepted = scheduler.submit(
        target_block, safe.reshape(-1), core.last_processed_generation,
        core.last_pulse_events, core.last_admitted_pulse_plans,
    )
    return safe.reshape(-1), accepted


def _advance_sources_through_early_commit(config, core, scheduler):
    """Submit source blocks through 800; commit both pulse targets before 900."""
    for source_index in range(0, 800, config.chunk_size):
        _, accepted = _process_and_submit(config, core, scheduler, source_index)
        assert accepted
        scheduler.advance()

    first_segment, accepted = _process_and_submit(config, core, scheduler, 800)
    assert accepted
    plans = core.last_admitted_pulse_plans
    assert len(plans) == 1
    plan = plans[0]
    assert plan.source_onset_sample_index == 880
    assert plan.target_onset_sample_index == 2_880
    assert plan.waveform_v.size == 100
    assert plan.target_onset_sample_index + plan.waveform_v.size - 1 == 2_979
    scheduler.advance()
    scheduler.advance()

    assert scheduler.generated_sample_count == 1_000
    assert scheduler.stats_snapshot()["pulse_committed_samples"] == 100
    committed = scheduler._committed_output.get(29)
    assert committed is not None
    assert np.array_equal(committed[0][:80], plan.waveform_v[20:])
    assert np.all(committed[0][80:] == 0.0)
    assert any(event.pulse_id == plan.pulse_id for event in committed[2])
    return first_segment, plan


def _reference_after_contiguous_tail(config, plan):
    reference = app.StimulusSafetyAdapter(config.safety, config.sample_rate)
    for _ in range(8):
        reference.process(np.zeros(config.chunk_size))
    first = np.zeros(config.chunk_size)
    first[80:] = plan.waveform_v[:20]
    tail = np.zeros(config.chunk_size)
    tail[:80] = plan.waveform_v[20:]
    reference.process(first)
    reference.process(tail)
    return reference


def _reference_after_first_segment(config, plan):
    reference = app.StimulusSafetyAdapter(config.safety, config.sample_rate)
    for _ in range(8):
        reference.process(np.zeros(config.chunk_size))
    first = np.zeros(config.chunk_size)
    first[80:] = plan.waveform_v[:20]
    reference.process(first)
    return reference


def _assert_same_safety_history(actual, expected):
    assert actual.previous_v == expected.previous_v
    assert np.array_equal(tuple(actual.area_history), tuple(expected.area_history))
    assert np.array_equal(tuple(actual.active_history), tuple(expected.active_history))
    assert actual.area_sum_v_s == expected.area_sum_v_s
    assert actual.area_sum_compensation == expected.area_sum_compensation
    assert actual.active_count == expected.active_count
    assert actual.consecutive_active == expected.consecutive_active


def _process_core_block(config, core, source_index: int):
    _, safe, _, reason = core.process(
        np.zeros((2, config.chunk_size), dtype=np.float64), sample_index=source_index,
        ao_target_sample_index=source_index + config.ao_lead_samples,
        input_is_electrode_mV=True,
        input_roles=("CA3", "Cortex"), input_labels=("CA3", "EC"), input_ctx_index=1,
    )
    assert reason is None
    return safe


def _observe_safety_inputs(core):
    original = core.safety.process
    observation = {"sample_counts": [], "nonzero_values": []}

    def observed(samples):
        values = np.asarray(samples, dtype=np.float64).reshape(-1)
        observation["sample_counts"].append(values.size)
        observation["nonzero_values"].extend(values[values != 0.0].tolist())
        return original(samples)

    core.safety.process = observed
    return observation


def test_actual_core_preserves_source_tail_after_early_target_commit():
    """The plan tail outlives scheduler commitment until the source pass renders it."""
    config, state, recorder, core, scheduler, commit_callbacks = _make_core_and_scheduler()
    first_segment, plan = _advance_sources_through_early_commit(config, core, scheduler)
    committed_blocks_before_tail = scheduler.stats_snapshot()["committed_blocks"]
    callbacks_before_tail = tuple(commit_callbacks)

    tail_segment, accepted = _process_and_submit(config, core, scheduler, 900)
    recorded_tail = next(block for block in recorder.blocks if block.sample_index == 900)
    tail_event = next((event for event in core.last_pulse_events
                       if event.status == "committed_tail_rendered"), None)

    expected_first_raw = np.zeros(config.chunk_size)
    expected_first_raw[80:] = plan.waveform_v[:20]
    expected_tail_raw = np.zeros(config.chunk_size)
    expected_tail_raw[:80] = plan.waveform_v[20:]
    reference = app.StimulusSafetyAdapter(config.safety, config.sample_rate)
    for _ in range(8):
        reference.process(np.zeros(config.chunk_size))
    expected_first_safe, first_reason = reference.process(expected_first_raw)
    expected_tail_safe, tail_reason = reference.process(expected_tail_raw)

    problems = []
    if first_reason or tail_reason:
        problems.append(f"reference waveform unexpectedly failed safety: {first_reason or tail_reason}")
    if not np.array_equal(first_segment, expected_first_safe.reshape(-1)):
        problems.append("source block 800 local output differs from the first admitted waveform segment")
    if not np.array_equal(tail_segment, expected_tail_safe.reshape(-1)):
        problems.append("source block 900 local output omitted or changed the admitted pulse tail")
    if not np.array_equal(tail_segment[:80], plan.waveform_v[20:]):
        problems.append("source block 900 local safe tail does not equal the admitted safe plan slice")
    if np.any(tail_segment[80:] != 0.0):
        problems.append("source block 900 local output extended beyond the admitted pulse tail")
    if not np.array_equal(recorded_tail.safe_ao, expected_tail_safe):
        problems.append("recorded source block 900 safe_ao differs from its complete admitted tail")
    if not np.array_equal(np.concatenate((expected_first_safe.reshape(-1),
                                          expected_tail_safe.reshape(-1)))[80:180], plan.waveform_v):
        problems.append("reference safety output does not reconstruct the admitted 100-sample plan")
    if not accepted:
        problems.append("source block 900 target was rejected after its pulse tail had already been committed")
    if (tail_event is None or tail_event.pulse_id != plan.pulse_id
            or tail_event.target_onset_sample_index != 2_880
            or tail_event.rendered_sample_count != 80):
        problems.append("source block 900 tail was not attributed to the exact committed pulse identity and target")
    committed_target = scheduler._committed_output.get(29)
    if committed_target is None or not np.array_equal(committed_target[0], tail_segment):
        problems.append("source block 900 reconciliation does not match the complete already-owned target block")
    stats = scheduler.stats_snapshot()
    if stats["late_commands"] != 0 or stats["rejected_stale"] != 0 or stats["rejected_fault"] != 0:
        problems.append(f"already-owned pulse target was misclassified: {stats}")
    if stats["committed_blocks"] != committed_blocks_before_tail:
        problems.append("reconciling a committed target incorrectly committed another AO block")
    if tuple(commit_callbacks) != callbacks_before_tail:
        problems.append("reconciling a committed target repeated its AO commit callback")
    if stats["pulse_committed_samples"] != 100:
        problems.append("reconciling a committed target changed the 100 committed pulse samples")
    if not np.array_equal(tuple(core.safety.area_history), tuple(reference.area_history)):
        problems.append("live safety area history does not include the complete source pulse tail")
    if not np.array_equal(tuple(core.safety.active_history), tuple(reference.active_history)):
        problems.append("live safety active-duty history does not include the complete source pulse tail")
    if core.safety.previous_v != reference.previous_v:
        problems.append("live safety previous-voltage state differs after the source pulse tail")
    if (core.safety.area_sum_v_s != reference.area_sum_v_s
            or core.safety.area_sum_compensation != reference.area_sum_compensation
            or core.safety.active_count != reference.active_count
            or core.safety.consecutive_active != reference.consecutive_active):
        problems.append("live rolling safety counters differ after the source pulse tail")

    # Keep the admitted target waveform in the device-indexed output timeline too.
    for source_index in range(1_000, 2_800, config.chunk_size):
        _, next_accepted = _process_and_submit(config, core, scheduler, source_index)
        assert next_accepted
        scheduler.advance()
    generated_28 = scheduler.advance()
    generated_29 = scheduler.advance()
    expected_28 = np.zeros(config.chunk_size)
    expected_28[80:] = plan.waveform_v[:20]
    expected_29 = np.zeros(config.chunk_size)
    expected_29[:80] = plan.waveform_v[20:]
    if generated_28.sample_index != 2_800 or not np.array_equal(generated_28.waveform_v, expected_28):
        problems.append("generated target block 28 differs from the first planned pulse segment")
    if generated_29.sample_index != 2_900 or not np.array_equal(generated_29.waveform_v, expected_29):
        problems.append("generated target block 29 differs from the planned pulse tail")

    assert not problems, "\n".join(problems)


def test_disarm_after_early_commit_accounts_source_tail_and_emits_owned_output_once():
    config, state, recorder, core, scheduler, commit_callbacks = _make_core_and_scheduler()
    _, plan = _advance_sources_through_early_commit(config, core, scheduler)
    state.set_armed(False)
    committed_blocks_before_tail = scheduler.stats_snapshot()["committed_blocks"]
    callbacks_before_tail = tuple(commit_callbacks)
    _, accepted = _process_and_submit(config, core, scheduler, 900)
    recorded_tail = next(block for block in recorder.blocks if block.sample_index == 900)
    reference = _reference_after_contiguous_tail(config, plan)
    stats_after_tail = scheduler.stats_snapshot()
    callbacks_after_tail = tuple(commit_callbacks)
    possible_tail = [stage for stage in recorder.stages
                     if stage.get("stage") == "possible_accounted"
                     and stage.get("request_id") == plan.pulse_id]

    # Continue the AO device clock while disarmed. Blocks 28 and 29 were already
    # committed and cannot be recalled; later targets must not recreate the pulse.
    generated = []
    for _ in range(18):
        generated.append(scheduler.advance())
    generated.extend((scheduler.advance(), scheduler.advance()))
    by_index = {block.sample_index: block for block in generated}
    expected_28 = np.zeros(config.chunk_size)
    expected_28[80:] = plan.waveform_v[:20]
    expected_29 = np.zeros(config.chunk_size)
    expected_29[:80] = plan.waveform_v[20:]

    # The committed tail advances the safety history as possible output, but
    # disarm forbids returning or recording a new command on this source block.
    assert np.all(recorded_tail.safe_ao == 0.0)
    assert not np.any(recorded_tail.safe_ao)
    assert not core.last_pulse_events
    assert len(possible_tail) == 1
    assert possible_tail[0]["controller"] == "rule"
    assert possible_tail[0]["generation"] == plan.generation
    assert possible_tail[0]["source_sample_index"] == 880
    assert possible_tail[0]["latest_sample_index"] == plan.latest_sample_index
    assert possible_tail[0]["target_sample_index"] == 2_880
    assert possible_tail[0]["sample_index"] == 2_900
    assert possible_tail[0]["range_end_sample_index"] == 2_980
    assert possible_tail[0]["reason"] == "source-accounted range [900, 980)"
    _assert_same_safety_history(core.safety, reference)
    assert accepted
    assert stats_after_tail["committed_blocks"] == committed_blocks_before_tail
    assert callbacks_after_tail == callbacks_before_tail
    assert stats_after_tail["late_commands"] == 0
    assert stats_after_tail["rejected_stale"] == 0
    assert stats_after_tail["rejected_fault"] == 0
    assert stats_after_tail["pulse_committed_samples"] == 100
    assert np.array_equal(by_index[2_800].waveform_v, expected_28)
    assert np.array_equal(by_index[2_900].waveform_v, expected_29)
    assert (np.count_nonzero(by_index[2_800].waveform_v)
            + np.count_nonzero(by_index[2_900].waveform_v)
            == np.count_nonzero(plan.waveform_v))
    assert scheduler.stats_snapshot()["pulse_tail_misses"] == 0
    assert state.snapshot().fault is None


def test_delayed_source_tail_reconciles_after_generated_output_and_drain_barrier():
    config, state, recorder, core, scheduler, commit_callbacks = _make_core_and_scheduler()
    _, plan = _advance_sources_through_early_commit(config, core, scheduler)
    state.set_armed(False)

    generated_pulse_blocks = {}
    stop_sample = plan.target_onset_sample_index + plan.waveform_v.size
    for _ in range(100):
        if not state.snapshot().transition_pending and scheduler.generated_sample_count > stop_sample:
            break
        block = scheduler.advance()
        if block.sample_index in {2_800, 2_900}:
            generated_pulse_blocks[block.sample_index] = block.waveform_v
    assert not state.snapshot().transition_pending
    assert scheduler.generated_sample_count > stop_sample

    expected_28 = np.zeros(config.chunk_size)
    expected_28[80:] = plan.waveform_v[:20]
    expected_29 = np.zeros(config.chunk_size)
    expected_29[:80] = plan.waveform_v[20:]
    assert np.array_equal(generated_pulse_blocks[2_800], expected_28)
    assert np.array_equal(generated_pulse_blocks[2_900], expected_29)

    before = scheduler.stats_snapshot()
    callbacks_before = tuple(commit_callbacks)
    observed = _observe_safety_inputs(core)
    tail_segment, accepted = _process_and_submit(config, core, scheduler, 900)
    recorded_tail = next(block for block in recorder.blocks if block.sample_index == 900)
    reference = _reference_after_contiguous_tail(config, plan)

    assert not np.any(tail_segment)
    assert np.all(recorded_tail.safe_ao == 0.0)
    assert np.array_equal(np.asarray(observed["nonzero_values"]),
                          plan.waveform_v[20:][plan.waveform_v[20:] != 0.0])
    _assert_same_safety_history(core.safety, reference)
    possible_tail = [stage for stage in recorder.stages
                     if stage.get("stage") == "possible_accounted"
                     and stage.get("request_id") == plan.pulse_id]
    assert len(possible_tail) == 1
    assert possible_tail[0]["generation"] == plan.generation
    assert possible_tail[0]["controller"] == "rule"
    assert possible_tail[0]["source_sample_index"] == 880
    assert possible_tail[0]["latest_sample_index"] == plan.latest_sample_index
    assert possible_tail[0]["target_sample_index"] == 2_880
    assert possible_tail[0]["sample_index"] == 2_900
    assert possible_tail[0]["range_end_sample_index"] == 2_980

    # Exact ownership reconciliation acknowledges delayed local accounting,
    # but cannot issue a second write or turn disarmed feedback into acceptance.
    after = scheduler.stats_snapshot()
    assert accepted, f"already-generated identical pulse tail was rejected: {after}"
    assert after["committed_blocks"] == before["committed_blocks"]
    assert tuple(commit_callbacks) == callbacks_before
    assert after["pulse_committed_samples"] == before["pulse_committed_samples"] == 100
    assert after["late_commands"] == before["late_commands"] == 0
    assert after["rejected_stale"] == before["rejected_stale"] == 0
    assert after["rejected_fault"] == before["rejected_fault"] == 0
    assert after["active_demand_misses"] == before["active_demand_misses"] == 0
    assert not core._feedback.accepted
    assert state.snapshot().fault is None


def test_committed_target_reconciliation_rejects_unowned_or_conflicting_content():
    config, state, _, core, scheduler, callbacks = _make_core_and_scheduler()
    _, plan = _advance_sources_through_early_commit(config, core, scheduler)
    original, generation, events, _ = scheduler._committed_output[29]
    original = original.copy()
    committed_count = scheduler.stats_snapshot()["committed_blocks"]
    callbacks_before = tuple(callbacks)

    # A zero block without the admitted pulse identity is not an idempotent
    # reconciliation, and a matching identity cannot replace owned samples.
    assert not scheduler.submit(29, np.zeros(config.chunk_size), generation)
    assert not scheduler.submit(29, np.zeros(config.chunk_size), generation, events)

    committed_after = scheduler._committed_output[29][0]
    assert np.array_equal(committed_after, original)
    assert any(event.pulse_id == plan.pulse_id for event in events)
    assert scheduler.stats_snapshot()["committed_blocks"] == committed_count
    assert tuple(callbacks) == callbacks_before
    assert state.snapshot().fault is None


def test_forward_source_gap_settles_committed_tail_once_then_retires_it():
    config, _, recorder, core, scheduler, _ = _make_core_and_scheduler()
    _, plan = _advance_sources_through_early_commit(config, core, scheduler)
    observed = _observe_safety_inputs(core)

    _process_core_block(config, core, 1_000)  # gap [900, 1000) crosses the source pulse tail
    _process_core_block(config, core, 1_100)  # a subsequent valid source block must not replay it
    reference = _reference_after_contiguous_tail(config, plan)
    reference.process(np.zeros(config.chunk_size))
    reference.process(np.zeros(config.chunk_size))

    expected_tail = plan.waveform_v[20:]
    assert np.array_equal(np.asarray(observed["nonzero_values"]), expected_tail[expected_tail != 0.0])
    assert sum(observed["sample_counts"]) <= (
        int(config.safety.area_window_s * config.sample_rate) + plan.waveform_v.size
        + 2 * config.chunk_size
    )
    _assert_same_safety_history(core.safety, reference)
    assert not core.actuator._pulses
    assert np.all(next(block for block in recorder.blocks if block.sample_index == 1_000).safe_ao == 0.0)
    assert np.all(next(block for block in recorder.blocks if block.sample_index == 1_100).safe_ao == 0.0)


def test_very_large_forward_gap_and_rewind_do_not_retain_or_double_account_tail():
    config, _, _, core, scheduler, _ = _make_core_and_scheduler()
    _, plan = _advance_sources_through_early_commit(config, core, scheduler)
    observed = _observe_safety_inputs(core)
    far_source_index = 100_000_000

    _process_core_block(config, core, far_source_index)
    nonzero_after_forward_gap = tuple(observed["nonzero_values"])
    calls_after_forward_gap = len(observed["sample_counts"])
    assert not any(core.safety.area_history)
    assert not any(core.safety.active_history)
    assert core.safety.previous_v == 0.0
    try:
        _process_core_block(config, core, far_source_index - config.chunk_size)
    except ValueError:
        # Rejecting an out-of-order source block is acceptable if it leaves
        # already-settled pulse/safety history untouched.
        pass

    expected_tail = plan.waveform_v[20:]
    assert np.array_equal(np.asarray(nonzero_after_forward_gap), expected_tail[expected_tail != 0.0])
    assert tuple(observed["nonzero_values"]) == nonzero_after_forward_gap
    assert len(observed["sample_counts"]) <= calls_after_forward_gap + 1
    assert sum(observed["sample_counts"]) <= (
        int(config.safety.area_window_s * config.sample_rate) + plan.waveform_v.size
        + 2 * config.chunk_size
    )
    assert not any(core.safety.area_history)
    assert not any(core.safety.active_history)
    assert core.safety.previous_v == 0.0
    assert not core.actuator._pulses


def test_fault_after_full_early_commit_accounts_only_committed_residual_without_reauthorizing():
    config, state, recorder, core, scheduler, commit_callbacks = _make_core_and_scheduler()
    _, plan = _advance_sources_through_early_commit(config, core, scheduler)
    assert scheduler.stats_snapshot()["pulse_committed_samples"] == 100
    assert core._feedback.accepted
    committed_before = scheduler.stats_snapshot()["committed_blocks"]
    callbacks_before = tuple(commit_callbacks)
    observed = _observe_safety_inputs(core)

    state.fault("fault after the complete pulse was irreversibly committed")
    safe = _process_core_block(config, core, 900)
    reference = _reference_after_contiguous_tail(config, plan)

    expected_tail = plan.waveform_v[20:]
    assert np.array_equal(np.asarray(observed["nonzero_values"]), expected_tail[expected_tail != 0.0])
    assert sum(observed["sample_counts"]) <= config.chunk_size + expected_tail.size
    _assert_same_safety_history(core.safety, reference)
    # Accounting the physical residual must never authorize a second AO output.
    assert np.all(safe == 0.0)
    # A latched fault stops acquisition, so there is no new RecordBlock. The
    # returned source command is still zero and the provenance stage is retained.
    assert not any(block.sample_index == 900 for block in recorder.blocks)
    possible_tail = [stage for stage in recorder.stages
                     if stage.get("stage") == "possible_accounted"
                     and stage.get("request_id") == plan.pulse_id]
    assert len(possible_tail) == 1
    assert possible_tail[0]["generation"] == plan.generation
    assert possible_tail[0]["controller"] == "rule"
    assert possible_tail[0]["source_sample_index"] == 880
    assert possible_tail[0]["target_sample_index"] == 2_880
    assert possible_tail[0]["sample_index"] == 2_900
    assert possible_tail[0]["range_end_sample_index"] == 2_980
    assert scheduler.stats_snapshot()["committed_blocks"] == committed_before
    assert tuple(commit_callbacks) == callbacks_before
    assert scheduler.stats_snapshot()["pulse_committed_samples"] == 100
    assert not core._feedback.accepted
    assert state.snapshot().fault == "fault after the complete pulse was irreversibly committed"


def test_fault_after_partial_commit_does_not_account_uncommitted_plan_suffix():
    config, state, _, core, scheduler, commit_callbacks = _make_core_and_scheduler(
        fault_on_first_partial_commit=True,
    )
    for source_index in range(0, 800, config.chunk_size):
        _, accepted = _process_and_submit(config, core, scheduler, source_index)
        assert accepted
        scheduler.advance()

    _, accepted = _process_and_submit(config, core, scheduler, 800)
    assert accepted
    plan = core.last_admitted_pulse_plans[0]
    # The first tick advances the output clock without reaching the refill
    # watermark. The second triggers a batched refill; the commit callback
    # faults immediately after target 28's first 20 pulse samples, so target 29
    # in that same refill must be replaced with zero.
    scheduler.advance()
    assert scheduler.stats_snapshot()["pulse_committed_samples"] == 0
    scheduler.advance()
    assert scheduler.stats_snapshot()["pulse_committed_samples"] == 20
    assert scheduler._committed_output[28][0][-20:].tolist() == plan.waveform_v[:20].tolist()
    assert np.all(scheduler._committed_output[29][0] == 0.0)
    assert state.snapshot().fault == "fault injected after the first 20 pulse samples committed"
    committed_before = scheduler.stats_snapshot()["committed_blocks"]
    callbacks_before = tuple(commit_callbacks)
    observed = _observe_safety_inputs(core)

    safe = _process_core_block(config, core, 900)
    reference = _reference_after_first_segment(config, plan)
    reference.process(np.zeros(config.chunk_size))

    assert observed["nonzero_values"] == []
    _assert_same_safety_history(core.safety, reference)
    assert np.all(safe == 0.0)
    assert scheduler.stats_snapshot()["pulse_committed_samples"] == 20
    assert not core._feedback.accepted
    assert state.snapshot().fault == "fault injected after the first 20 pulse samples committed"
    assert scheduler.stats_snapshot()["committed_blocks"] == committed_before
    assert tuple(commit_callbacks) == callbacks_before

    # Continue the virtual AO clock: target 28's committed first 20 samples
    # remain immutable, while the uncommitted target-29 suffix is canceled.
    generated = {}
    while scheduler.generated_sample_count <= 2_900:
        block = scheduler.advance()
        if block.sample_index in {2_800, 2_900}:
            generated[block.sample_index] = block.waveform_v
    expected_28 = np.zeros(config.chunk_size)
    expected_28[-20:] = plan.waveform_v[:20]
    assert np.array_equal(generated[2_800], expected_28)
    assert np.all(generated[2_900] == 0.0)
    assert tuple(commit_callbacks) == callbacks_before
    assert scheduler.stats_snapshot()["pulse_committed_samples"] == 20
    assert state.snapshot().fault == "fault injected after the first 20 pulse samples committed"
