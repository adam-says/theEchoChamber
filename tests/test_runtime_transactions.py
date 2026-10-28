from __future__ import annotations

from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from unittest import mock

import numpy as np
import pytest

import echoChamber as app


def make_state(controller: str = "rule") -> app.RuntimeState:
    return app.RuntimeState(
        esn_ready=False, start_paused=False, electrode_labels=("CA3", "EC"), ctx_index=1,
        pulse_threshold_std=3.0, pulse_window_sec=10.0, controller=controller,
        controller_ready={"esn": False, "rule": True, "pid": True},
    )


def make_config() -> app.AppConfig:
    return app.AppConfig(processing_block_ms=5.0, ao_target_lead_ms=100.0)


def test_ui_diagnostics_serialize_as_strict_json_during_calibration():
    import json

    value = app.json_safe({"z_score": float("nan"), "rate": np.float64(np.inf), "x": 1.5})
    assert value == {"z_score": None, "rate": None, "x": 1.5}
    packet = json.dumps(value, allow_nan=False)
    reject_nonstandard_constant = lambda token: (_ for _ in ()).throw(ValueError(token))
    assert json.loads(packet, parse_constant=reject_nonstandard_constant) == value
    with pytest.raises(ValueError, match="NaN"):
        json.loads('{"ai":[NaN]}', parse_constant=reject_nonstandard_constant)


def event(pulse_id: str, *, onset: int, count: int, generation: int = 0):
    return SimpleNamespace(
        pulse_id=pulse_id, requested_sample_index=onset, source_onset_sample_index=onset,
        target_onset_sample_index=onset, latest_sample_index=onset,
        status="rendered_for_ao", generation=generation, rendered_sample_count=count,
    )


class FakeNiTask:
    def __init__(self, name, counter):
        self.name = name
        self.counter = counter
        self.out_stream = self
        self.timing = SimpleNamespace(cfg_samp_clk_timing=lambda *a, **kw: None)
        self.ao_channels = SimpleNamespace(add_ao_voltage_chan=lambda *a, **kw: None)
        self.triggers = SimpleNamespace(start_trigger=SimpleNamespace(cfg_dig_edge_start_trig=lambda *a, **kw: None))
        self.started = False

    @property
    def total_samp_per_chan_generated(self):
        if self.started:
            self.counter[0] += 100
        return self.counter[0]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def start(self):
        self.started = True

    def write(self, *args, **kwargs):
        pass


class FakeNiWriter:
    def __init__(self, *, pause=False, fail=False):
        self.calls = 0
        self.pause = pause
        self.fail = fail
        self.reserved = threading.Event()
        self.release = threading.Event()
        self.writes = []

    def write_many_sample(self, values, timeout=None):
        self.calls += 1
        if self.calls == 2 and self.pause:
            self.reserved.set()
            if not self.release.wait(5.0):
                raise TimeoutError("fake NI writer was not released")
        if self.calls == 2 and self.fail:
            raise OSError("injected fake NI write failure")
        self.writes.append(np.asarray(values, dtype=np.float64).copy())


def test_reserved_refill_is_not_removed_until_successful_writer_commit():
    state = make_state()
    config = make_config()
    scheduler = app.ScheduledAo(config, state, "/Dev1/ai/StartTrigger")
    target = config.ao_lead_chunks
    command = np.zeros(config.chunk_size)
    command[-1] = 0.5
    scheduler.submit(target, command, generation=0)

    reserved = scheduler._take_contiguous(target, 1, 0)
    assert len(reserved) == 1
    assert target in scheduler.commands  # a paused/failed writer cannot lose this command

    state.set_mode("closed-loop")
    assert state.snapshot().transition_pending
    assert target in scheduler.commands

    # Model the writer returning successfully, then publish the command's committed span.
    scheduler.commands.pop(target)
    scheduler.drain_barrier.committed(target * config.chunk_size, [command], config.chunk_size)
    scheduler.drain_barrier.begin(state.snapshot().generation, generated_sample_count=0)
    assert not scheduler.drain_barrier.drained(state.snapshot().generation, 0)

    zeros = np.zeros(config.chunk_size)
    scheduler.drain_barrier.committed((target + 1) * config.chunk_size, [zeros], config.chunk_size)
    assert not scheduler.drain_barrier.drained(state.snapshot().generation, target * config.chunk_size)
    assert scheduler.drain_barrier.drained(state.snapshot().generation, (target + 1) * config.chunk_size)


def test_generation_switch_preserves_only_partially_committed_tail_at_same_target():
    lifecycle = app.AoCommandLifecycle()
    tail = np.zeros(100)
    tail[:25] = 0.25
    whole = np.full(100, 0.5)
    tail_event = event("partial", onset=2100, count=25)
    whole_event = event("whole", onset=2200, count=0)
    lifecycle.submit(21, tail, 0, (tail_event,))
    lifecycle.submit(22, whole, 0, (whole_event,))
    lifecycle.committed((tail_event,), 20_000)

    cancelled, preserved = lifecycle.switch(0)
    assert [target for target, _ in cancelled] == [22]
    assert [target for target, _ in preserved] == [21]
    np.testing.assert_array_equal(lifecycle.commands[21][0], tail)
    assert lifecycle.eligible(lifecycle.commands[21], generation=1, transition_pending=True)
    assert not lifecycle.eligible((whole, 0, (whole_event,)), generation=1, transition_pending=True)


def test_transition_block_cannot_relabel_previous_block_pulse_events():
    class Core(app.ProcessingCore):
        def _publish_ui(self, *args, **kwargs):
            pass

    config = app.dataclasses.replace(make_config(), controller="rule")
    state = make_state()
    core = Core(config, state, SimpleNamespace(ready=False, streamer=None, reset=lambda: None), object(), object(), None,
                app.Telemetry())
    core._last_pulse_events = (event("prior-generation", onset=2100, count=100),)
    state.set_mode("closed-loop")
    current_generation = state.snapshot().generation
    core.process(np.zeros((2, config.chunk_size)), sample_index=0, input_is_electrode_mV=True)
    assert core.last_processed_generation == current_generation
    assert core.last_pulse_events == ()


def test_simulated_ao_uses_real_lead_and_waits_on_generated_counter():
    config = make_config()
    state = make_state()
    scheduler = app.SimulatedAoScheduler(config, state)
    target = config.ao_lead_chunks
    command = np.zeros(config.chunk_size)
    command[50:] = 0.5
    pulse = event("p1", onset=target * config.chunk_size + 50, count=50)
    assert scheduler.submit(target, command, 0, (pulse,))
    tail = np.zeros(config.chunk_size)
    tail[:50] = 0.5
    tail_event = event("p1", onset=target * config.chunk_size + 50, count=50)
    assert scheduler.submit(target + 1, tail, 0, (tail_event,))

    first = scheduler.advance()
    assert first.sample_index == 0
    assert not np.any(first.waveform_v)
    assert scheduler.generated_sample_count == config.chunk_size
    assert target not in scheduler._committed_output  # still above NI's refill threshold
    while scheduler.generated_sample_count < config.ao_lead_samples - config.chunk_size:
        scheduler.advance()
    np.testing.assert_allclose(scheduler._committed_output[target][0], command)
    np.testing.assert_allclose(scheduler._committed_output[target + 1][0], tail)
    assert scheduler._committed_output[target + 1][3] == "ao_committed"
    state.set_mode("closed-loop")
    scheduler.advance()  # committed commands retain their original generation and sample positions
    assert state.snapshot().transition_pending
    generated = None
    while scheduler.generated_sample_count < 3_000:
        generated = scheduler.advance()
    assert generated is not None and generated.sample_index == 2_900
    assert generated.generation == 0  # future generation-1 refills cannot relabel this old output
    assert all(scheduler._committed_output[target][1] == 1 for target in range(40, 50))
    while scheduler.generated_sample_count < pulse.target_onset_sample_index + 100:
        scheduler.advance()
    assert not state.snapshot().transition_pending


def test_simulated_fault_cancels_pending_commands_and_suppresses_staged_full_pulse():
    config, state = make_config(), make_state()
    state.set_stim("threshold_pulse", 0.5)
    state.update_controller_calibration("rule", True)
    state.set_mode("closed-loop")
    state.acknowledge_drain(state.snapshot().generation)
    state.set_armed(True)
    outcomes = []
    commits = []
    scheduler = app.SimulatedAoScheduler(
        config, state, ao_outcome=lambda generation, accepted, reason:
        outcomes.append((generation, accepted, reason)),
        commit_event=lambda pulse_id, count: commits.append((pulse_id, count)),
    )
    generation = state.snapshot().generation
    target = 50
    waveform = np.sin(np.linspace(0.0, np.pi, 100, endpoint=False)) * 0.25
    plan = app.SafetyAdmittedPulsePlan("fault-pulse", generation, 80, target * 100 + 80, 99, waveform)
    first = np.zeros(config.chunk_size)
    first[80:] = waveform[:20]
    tail = np.zeros(config.chunk_size)
    tail[:80] = waveform[20:]
    assert scheduler.submit(target, first, generation,
                            (event("fault-pulse", onset=80, count=20, generation=generation),), (plan,))
    assert scheduler.submit(target + 1, tail, generation,
                            (event("fault-pulse", onset=80, count=80, generation=generation),))

    # An AO host stall crosses the safe deadline before any refill service.
    generated = scheduler.advance_elapsed(0.25)
    assert state.snapshot().fault is not None
    assert target not in scheduler.commands and target + 1 not in scheduler.commands
    assert all(not np.any(block.waveform_v) for block in generated)
    assert not scheduler._committed_output.get(target, (np.zeros(100),))[0].any()
    assert not scheduler._committed_output.get(target + 1, (np.zeros(100),))[0].any()
    assert scheduler.stats_snapshot()["pulse_committed_samples"] == 0
    assert commits == []
    assert outcomes and all(not accepted for _, accepted, _ in outcomes)

    after_fault = [scheduler.advance() for _ in range(30)]
    assert all(not np.any(block.waveform_v) for block in after_fault)
    assert not scheduler._committed_pulse_samples


def test_simulated_fault_preserves_committed_residual_and_cancels_uncommitted_suffix():
    config, state = make_config(), make_state()
    scheduler = app.SimulatedAoScheduler(config, state)
    committed_target, pending_target = 50, 60
    nonzero = np.full(config.chunk_size, 0.25)
    pending = np.full(config.chunk_size, 0.4)
    assert scheduler.submit(committed_target, nonzero, state.snapshot().generation)
    assert scheduler.submit(pending_target, pending, state.snapshot().generation)
    while committed_target not in scheduler._committed_output:
        scheduler.advance()
        assert scheduler.generated_sample_count <= committed_target * config.chunk_size
    assert np.any(scheduler._committed_output[committed_target][0])
    assert scheduler.generated_sample_count <= committed_target * config.chunk_size

    state.fault("fault after virtual AO staging")
    residual = None
    while scheduler.generated_sample_count <= committed_target * config.chunk_size:
        residual = scheduler.advance()
    assert residual is not None and residual.sample_index == committed_target * config.chunk_size
    np.testing.assert_array_equal(residual.waveform_v, nonzero)

    assert pending_target not in scheduler.commands
    if pending_target in scheduler._committed_output:
        assert not np.any(scheduler._committed_output[pending_target][0])
    later = None
    while scheduler.generated_sample_count <= pending_target * config.chunk_size:
        later = scheduler.advance()
    assert later is not None and later.sample_index == pending_target * config.chunk_size
    assert not np.any(later.waveform_v)


def test_fake_ni_fault_cancels_pending_full_pulse_without_commit_or_nonzero_write():
    config, state, writer = make_config(), make_state(), FakeNiWriter()
    commits = []
    outcomes = []
    counter = [0]
    scheduler = app.ScheduledAo(
        config, state, "/Dev1/ai/StartTrigger",
        commit_event=lambda pulse_id, count: commits.append((pulse_id, count)),
        ao_outcome=lambda generation, accepted, reason: outcomes.append((generation, accepted, reason)),
    )
    target = config.ao_lead_chunks
    waveform = np.sin(np.linspace(0, np.pi, 100, endpoint=False)) * 0.25
    command = np.zeros(config.chunk_size)
    command[80:] = waveform[:20]
    plan = app.SafetyAdmittedPulsePlan("ni-fault-pulse", 0, 80, target * 100 + 80, 99, waveform)
    assert scheduler.submit(target, command, 0, (event("ni-fault-pulse", onset=80, count=20),), (plan,))
    state.fault("injected latched fault")
    task_patch = mock.patch.object(app.nidaqmx, "Task", side_effect=lambda name: FakeNiTask(name, counter))
    writer_patch = mock.patch.object(app, "AnalogSingleChannelWriter", return_value=writer)
    with task_patch, writer_patch:
        scheduler.start()
        scheduler.notify_clock_started()
        try:
            deadline = time.monotonic() + 3.0
            while len(writer.writes) < 2 and time.monotonic() < deadline:
                time.sleep(0.005)
            assert len(writer.writes) >= 2
            assert all(not np.any(values) for values in writer.writes)
            assert scheduler.commands == {}
            assert commits == []
            assert not scheduler._committed_pulse_samples
            assert outcomes and all(not accepted for _, accepted, _ in outcomes)
            assert scheduler.stats.snapshot()["cancelled_blocks"] >= 1
        finally:
            scheduler.stop()


def test_simulated_ao_elapsed_clock_advances_without_a_submission():
    config = make_config()
    state = make_state()
    scheduler = app.SimulatedAoScheduler(config, state)
    assert scheduler.advance_elapsed(0.0025) == ()
    assert scheduler.generated_sample_count == 0
    generated = scheduler.advance_elapsed(0.0075)
    assert len(generated) == 2
    assert [block.sample_index for block in generated] == [0, 100]
    assert scheduler.generated_sample_count == 200


def test_safe_full_pulse_plan_reserves_original_tail_before_generation_switch():
    config = make_config()
    state = make_state()
    scheduler = app.SimulatedAoScheduler(config, state)
    target = config.ao_lead_chunks
    plan_wave = np.sin(np.linspace(0, np.pi, 100, endpoint=False)) * 0.25
    plan = app.SafetyAdmittedPulsePlan(
        "planned", 0, 80, target * config.chunk_size + 80, 99, plan_wave,
    )
    first_command = np.zeros(config.chunk_size)
    first_command[80:] = plan_wave[:20]
    proposal = event("planned", onset=80, count=20)
    assert scheduler.submit(target, first_command, 0, (proposal,), (plan,))
    assert scheduler.commands[target + 1][2][0].rendered_sample_count == 80
    np.testing.assert_array_equal(scheduler.commands[target + 1][0][:80], plan_wave[20:])

    generated = [scheduler.advance()
                 for _ in range(config.ao_lead_samples // config.chunk_size // 2)]
    # This switch happens before any later ProcessingCore block supplies a tail.
    state.set_mode("closed-loop")
    while scheduler.generated_sample_count < plan.target_onset_sample_index + 100:
        generated.append(scheduler.advance())
    by_index = {block.sample_index: block for block in generated}
    first_block = plan.target_onset_sample_index // config.chunk_size
    first = by_index[first_block * config.chunk_size].waveform_v
    second = by_index[(first_block + 1) * config.chunk_size].waveform_v
    np.testing.assert_allclose(first[80:], plan_wave[:20])
    np.testing.assert_allclose(second[:80], plan_wave[20:])
    assert scheduler.lifecycle.committed_pulse_samples["planned"] == 100
    assert scheduler._pulse_exclusive_ends["planned"] == plan.target_onset_sample_index + 100


def test_simulated_ao_faults_and_counts_a_missed_active_refill():
    config = make_config()
    state = make_state()
    state.set_stim("threshold_pulse", 0.25)
    state.set_mode("closed-loop")
    state.acknowledge_drain(state.snapshot().generation)
    state.update_controller_calibration("rule", True)
    state.set_armed(True)
    scheduler = app.SimulatedAoScheduler(config, state)
    generation = state.snapshot().generation
    generated = scheduler.advance_elapsed(0.1)
    assert len(generated) == 20
    assert state.snapshot().fault is not None
    stats = scheduler.stats_snapshot()
    assert stats["active_demand_misses"] == 1
    assert stats["zero_substitutions"] == config.ao_lead_chunks


def test_half_lead_refill_keeps_future_source_frontier_writable_while_unarmed():
    config = make_config()
    state = make_state()
    state.set_stim("threshold_pulse", 0.25)
    state.set_mode("closed-loop")
    state.acknowledge_drain(state.snapshot().generation)
    state.update_controller_calibration("rule", True)
    generation = state.snapshot().generation
    scheduler = app.SimulatedAoScheduler(config, state)
    for source_block in range(9):
        assert scheduler.submit(source_block + config.ao_lead_chunks,
                                np.zeros(config.chunk_size), generation)

    generated = scheduler.advance_elapsed(0.050)
    assert len(generated) == 10
    assert scheduler._next_write_block == config.ao_lead_chunks + 9
    assert scheduler.stats_snapshot()["committed_blocks"] == 9
    assert scheduler.stats_snapshot()["zero_substitutions"] == 0
    assert scheduler.stats_snapshot()["active_demand_misses"] == 0
    assert scheduler.submit(config.ao_lead_chunks + 9, np.zeros(config.chunk_size), generation)


def test_simulated_ao_defers_future_missing_block_until_source_submission():
    config = make_config()
    state = make_state()
    state.set_stim("threshold_pulse", 0.25)
    state.set_mode("closed-loop")
    state.acknowledge_drain(state.snapshot().generation)
    state.update_controller_calibration("rule", True)
    state.set_armed(True)
    scheduler = app.SimulatedAoScheduler(config, state)
    generation = state.snapshot().generation
    pulse_wave = np.linspace(0.1, 0.2, config.chunk_size)
    generated = []

    for source_block in range(40):
        # At each boundary the AO clock advances before the producer finishes
        # processing the just-acquired source block, as it can on NI hardware.
        generated.extend(scheduler.advance_elapsed(config.chunk_size / config.sample_rate))
        target = source_block + config.ao_lead_chunks
        command = np.zeros(config.chunk_size)
        events = ()
        plans = ()
        if source_block == 19:
            command[:] = pulse_wave
            onset = target * config.chunk_size
            events = (SimpleNamespace(
                pulse_id="boundary-pulse", requested_sample_index=source_block * config.chunk_size,
                source_onset_sample_index=source_block * config.chunk_size,
                target_onset_sample_index=onset, latest_sample_index=(source_block + 1) * config.chunk_size - 1,
                status="rendered_for_ao", generation=generation, rendered_sample_count=config.chunk_size,
            ),)
            plans = (app.SafetyAdmittedPulsePlan(
                "boundary-pulse", generation, source_block * config.chunk_size, onset,
                (source_block + 1) * config.chunk_size - 1, pulse_wave,
            ),)
        assert scheduler.submit(target, command, state.snapshot().generation, events, plans)

    pulse_block = next(block for block in generated
                       if block.sample_index == (19 + config.ao_lead_chunks) * config.chunk_size)
    np.testing.assert_array_equal(pulse_block.waveform_v, pulse_wave)
    stats = scheduler.stats_snapshot()
    assert stats["active_demand_misses"] == 0
    assert stats["late_commands"] == 0
    assert stats["pulse_committed_samples"] == config.chunk_size


def test_elapsed_ao_host_stall_does_not_retroactively_refill_elapsed_targets():
    config = make_config()
    state = make_state()
    state.set_stim("threshold_pulse", 0.25)
    state.set_mode("closed-loop")
    state.acknowledge_drain(state.snapshot().generation)
    state.update_controller_calibration("rule", True)
    state.set_armed(True)
    scheduler = app.SimulatedAoScheduler(config, state)
    target = config.ao_lead_chunks
    pulse = np.full(config.chunk_size, 0.25)
    assert scheduler.submit(target, pulse, state.snapshot().generation)

    # The hardware consumes its initially committed 100 ms of zeros while
    # the AO host receives no scheduling time for the full 250 ms interval.
    generated = scheduler.advance_elapsed(0.250)
    assert len(generated) == 50
    assert all(not np.any(block.waveform_v) for block in generated)
    assert scheduler.generated_sample_count == 5_000
    assert scheduler.stats_snapshot()["late_commands"] >= 1
    assert scheduler.stats_snapshot()["active_demand_misses"] == 1
    assert scheduler.stats_snapshot()["pulse_committed_samples"] == 0
    assert state.snapshot().fault is not None


def test_host_stall_faults_if_partly_committed_pulse_tail_expires_at_original_target():
    config = make_config()
    state = make_state()
    scheduler = app.SimulatedAoScheduler(config, state)
    generation = state.snapshot().generation
    target_block = config.ao_lead_chunks
    target_onset = target_block * config.chunk_size + 80
    wave = np.linspace(0.1, 0.2, 100)
    first = np.zeros(config.chunk_size)
    first[80:] = wave[:20]
    event_value = SimpleNamespace(
        pulse_id="partial-tail", requested_sample_index=0, source_onset_sample_index=0,
        target_onset_sample_index=target_onset, latest_sample_index=99,
        status="rendered_for_ao", generation=generation, rendered_sample_count=20,
    )
    plan = app.SafetyAdmittedPulsePlan("partial-tail", generation, 0, target_onset, 99, wave)
    assert scheduler.submit(target_block, first, generation, (event_value,), (plan,))

    # Simulate the AO writer having committed the first 20 pulse samples while
    # the remaining original-index tail is still an uncommitted host command.
    scheduler.commands.pop(target_block)
    scheduler._committed_output[target_block] = (first.copy(), generation, (event_value,), "ao_committed")
    scheduler.lifecycle.committed((event_value,), config.sample_rate)
    scheduler._next_write_block = target_block + 1
    scheduler.drain_barrier.preserve_until(target_onset + wave.size)
    state.set_mode("closed-loop")

    generated = scheduler.advance_elapsed(0.250)
    block = next(value for value in generated if value.sample_index == target_block * config.chunk_size)
    np.testing.assert_array_equal(block.waveform_v[80:], wave[:20])
    assert scheduler.stats_snapshot()["pulse_tail_misses"] == 1
    assert state.snapshot().fault is not None
    assert scheduler.lifecycle.committed_pulse_samples["partial-tail"] == 20


def test_mock_replay_cursor_does_not_skip_to_ao_sample_clock():
    config = make_config()
    mock_daq = object.__new__(app.MockDaq)
    mock_daq.config = config
    mock_daq.state = make_state()
    mock_daq.replay_index = 0
    mock_daq.mock_sample_index = 50_000  # independent AO clock is far ahead
    mock_daq.replay = SimpleNamespace(
        sample_count=2 * config.chunk_size,
        electrode_mV=np.vstack((np.arange(2 * config.chunk_size),
                                -np.arange(2 * config.chunk_size))).astype(np.float64),
        raw_V=None,
    )

    electrode_mv, raw_v, valid_count = mock_daq._next_input()
    assert valid_count == config.chunk_size
    assert mock_daq.replay_index == config.chunk_size
    np.testing.assert_array_equal(electrode_mv[:, :config.chunk_size], mock_daq.replay.electrode_mV[:, :config.chunk_size])
    assert np.all(np.isnan(raw_v))


def test_incremental_safety_history_matches_reference_and_resets_on_trip():
    config = app.SafetyConfig(
        max_command_v=1.0, max_slew_v_per_s=2_000.0, max_abs_area_v_s=0.004,
        area_window_s=0.02, max_active_fraction=0.15, active_threshold_v=0.5,
        max_consecutive_active_s=0.005,
    )
    optimized = app.StimulusSafetyAdapter(config, 20_000)
    area = deque(maxlen=400)
    active = deque(maxlen=400)
    previous = 0.0
    consecutive = 0
    rng = np.random.default_rng(142)
    blocks = [rng.normal(0.0, 0.3, 100) for _ in range(120)]
    blocks[20][:] = 0.9
    blocks[70][:] = -0.8

    for raw in blocks:
        signal = np.clip(raw, -config.max_command_v, config.max_command_v)
        reference = np.empty_like(signal)
        for index, value in enumerate(signal):
            delta = float(value) - previous
            previous += min(config.max_slew_v_per_s / 20_000,
                            max(-config.max_slew_v_per_s / 20_000, delta))
            reference[index] = previous
        for value in reference:
            area.append(float(value) / 20_000)
            flag = int(abs(float(value)) >= config.active_threshold_v)
            active.append(flag)
            consecutive = consecutive + 1 if flag else 0
        reason = None
        if abs(sum(area)) > config.max_abs_area_v_s:
            reason = "net command area limit exceeded"
        elif len(active) == active.maxlen and sum(active) / len(active) > config.max_active_fraction:
            reason = "stimulation duty-cycle limit exceeded"
        elif consecutive > int(config.max_consecutive_active_s * 20_000):
            reason = "continuous stimulation limit exceeded"
        if reason is not None:
            reference[:] = 0.0
            area.clear()
            active.clear()
            previous = 0.0
            consecutive = 0

        safe, optimized_reason = optimized.process(raw)
        assert optimized_reason == reason
        np.testing.assert_allclose(safe.reshape(-1), reference, rtol=0, atol=1e-14)
        assert len(optimized.area_history) == len(area)
        assert len(optimized.active_history) == len(active)
        assert optimized.consecutive_active == consecutive
        assert optimized.active_count == sum(active)
        np.testing.assert_allclose(optimized.area_sum_v_s, sum(area), rtol=0, atol=1e-14)


def test_full_pulse_safety_admission_does_not_advance_live_history():
    safety = app.StimulusSafetyAdapter(make_config().safety, 20_000)
    safety.process(np.zeros(100))
    before = (safety.previous_v, tuple(safety.area_history), tuple(safety.active_history),
              safety.area_sum_v_s, safety.active_count, safety.consecutive_active)
    safe_tail, reason = safety.admit_pulse_tail(np.full(100, 0.01))
    after = (safety.previous_v, tuple(safety.area_history), tuple(safety.active_history),
             safety.area_sum_v_s, safety.active_count, safety.consecutive_active)
    assert reason is None
    assert safe_tail.shape == (1, 100)
    assert before == after


def test_rejected_complete_pulse_plan_stages_neither_first_segment_nor_tail():
    from controllers import ControllerAction, ControllerDiagnostics, PulseEvent, PulsePlan, StepResult

    class Core(app.ProcessingCore):
        def _publish_ui(self, *args, **kwargs):
            pass

    class Controller:
        def reset(self, keep_calibration=False):
            pass

        def drain(self):
            pass

        def step(self, inp):
            return StepResult(
                ControllerAction(),
                ControllerDiagnostics("rule", True, True, 0.0, 0.0, 1.0, 0.0,
                                      3.0, 0.0, 0.0, "ready", inp.generation),
            )

    class EventTelemetry(app.Telemetry):
        def trip(self):
            pass

    config = app.AppConfig(
        processing_block_ms=5.0, ao_target_lead_ms=100.0,
        safety=app.SafetyConfig(
            max_command_v=1.0, max_slew_v_per_s=20_000.0, max_abs_area_v_s=0.0002,
            area_window_s=1.0, max_active_fraction=1.0, active_threshold_v=2.0,
            max_consecutive_active_s=1.0,
        ),
    )
    state = make_state()
    state.set_stim("threshold_pulse", 0.5)
    state.set_mode("closed-loop")
    state.acknowledge_drain(state.snapshot().generation)
    state.update_controller_calibration("rule", True)
    state.set_armed(True)
    core = Core(config, state, app.EsnRuntime(None, config), object(), object(), None,
                EventTelemetry())
    core.controllers["rule"] = Controller()
    waveform = np.sin(np.linspace(0.0, np.pi, 100, endpoint=False)) * 0.5
    source_onset = 0
    target_onset = config.ao_lead_samples
    event_value = PulseEvent("unsafe-plan", source_onset, source_onset, target_onset,
                             "rendered_for_ao", state.snapshot().generation, 20, 99)
    plan = PulsePlan("unsafe-plan", state.snapshot().generation, source_onset, target_onset, 99, waveform)

    def render_proposal(*args, **kwargs):
        current = np.zeros(config.chunk_size)
        current[:20] = waveform[:20]
        return SimpleNamespace(waveform_v=current, pulse_events=(event_value,), requested_rate_hz=1.0,
                               rendered_count=20, reason="rendered", pulse_plans=(plan,))

    core.actuator.process = render_proposal
    _, safe, _, reason = core.process(np.zeros((2, config.chunk_size)), sample_index=0,
                                      ao_target_sample_index=target_onset,
                                      input_is_electrode_mV=True)
    assert reason is not None and "full pulse plan failed safety admission" in reason
    assert not np.any(safe)
    assert core.last_admitted_pulse_plans == ()
    assert core.last_pulse_events[0].status == "rejected_safety"
    assert core.last_pulse_events[0].rendered_sample_count == 0

    scheduler = app.SimulatedAoScheduler(config, state)
    target_block = target_onset // config.chunk_size
    assert not scheduler.submit(target_block, safe.reshape(-1), state.snapshot().generation,
                                core.last_pulse_events, core.last_admitted_pulse_plans)
    assert target_block not in scheduler.commands
    assert scheduler.stats_snapshot()["pulse_committed_samples"] == 0


def _start_fake_ni_scheduler(writer, state, config, committed):
    counter = [0]
    task_factory = lambda name: FakeNiTask(name, counter)
    scheduler = app.ScheduledAo(config, state, "/Dev1/ai/StartTrigger",
                                commit_event=lambda pulse_id, count: committed.append((pulse_id, count)))
    return scheduler, mock.patch.object(app.nidaqmx, "Task", side_effect=task_factory), \
        mock.patch.object(app, "AnalogSingleChannelWriter", return_value=writer)


def _submit_fake_committed_tail(scheduler, config):
    target = config.ao_lead_chunks
    wave = np.sin(np.linspace(0, np.pi, 100, endpoint=False)) * 0.25
    command = np.zeros(config.chunk_size)
    command[80:] = wave[:20]
    plan = app.SafetyAdmittedPulsePlan("ni-pulse", 0, 80, target*config.chunk_size + 80,
                                       99, wave)
    assert scheduler.submit(target, command, 0, (event("ni-pulse", onset=80, count=20),), (plan,))


def test_fake_ni_writer_switch_waits_for_reserved_write_and_drain_ack():
    config, state, writer = make_config(), make_state(), FakeNiWriter(pause=True)
    committed = []
    scheduler, task_patch, writer_patch = _start_fake_ni_scheduler(writer, state, config, committed)
    with task_patch, writer_patch:
        _submit_fake_committed_tail(scheduler, config)
        scheduler.start()
        scheduler.notify_clock_started()
        try:
            assert writer.reserved.wait(3.0)
            target = config.ao_lead_chunks
            assert target in scheduler.commands  # reservation remains live while writer is paused
            state.set_mode("closed-loop")
            generation = state.snapshot().generation
            assert state.snapshot().transition_pending
            assert not state.snapshot().armed
            writer.release.set()
            deadline = time.monotonic() + 5.0
            while state.snapshot().transition_pending and time.monotonic() < deadline:
                time.sleep(0.005)
            assert not state.snapshot().transition_pending
            assert scheduler.error is None
            assert ("ni-pulse", 20) in committed
            assert ("ni-pulse", 80) in committed
            assert scheduler._pulse_exclusive_ends["ni-pulse"] == 2_180
            assert state.snapshot().generation == generation
        finally:
            writer.release.set()
            scheduler.stop()


def test_fake_ni_write_failure_never_commits_pulse_or_acknowledges_switch():
    config, state, writer = make_config(), make_state(), FakeNiWriter(pause=True, fail=True)
    committed = []
    scheduler, task_patch, writer_patch = _start_fake_ni_scheduler(writer, state, config, committed)
    with task_patch, writer_patch:
        _submit_fake_committed_tail(scheduler, config)
        scheduler.start()
        scheduler.notify_clock_started()
        try:
            assert writer.reserved.wait(3.0)
            state.set_mode("closed-loop")
            writer.release.set()
            deadline = time.monotonic() + 3.0
            while scheduler.error is None and time.monotonic() < deadline:
                time.sleep(0.005)
            assert isinstance(scheduler.error, OSError)
            assert committed == []
            assert not scheduler._committed_pulse_samples
            assert state.snapshot().transition_pending
        finally:
            writer.release.set()
            scheduler.stop()


def test_uncalibrated_controller_can_compute_but_cannot_arm_until_after_drain():
    state = make_state("rule")
    state.set_stim("threshold_pulse", 0.5)
    state.set_mode("closed-loop")
    state.acknowledge_drain(state.snapshot().generation)
    with pytest.raises(ValueError, match="baseline calibration"):
        state.set_armed(True)
    state.update_controller_calibration("rule", True)
    state.set_armed(True)
    assert state.snapshot().armed

    # Any complete setting edit disarms and starts a new drain transaction.
    prepared = app.build_controller("rule", 20_000, {"lower_z": 1.2, "upper_z": 3.2, "band_hz": (100, 250)})
    state.set_controller_settings("rule", {"lower_z": 1.2, "upper_z": 3.2, "band_hz": (100, 250)}, prepared)
    assert state.snapshot().transition_pending
    assert not state.snapshot().armed
    with pytest.raises(ValueError, match="drain"):
        state.set_armed(True)


def test_threshold_pulse_amplitude_is_limited_for_esn_as_well_as_rule_pid():
    state = app.RuntimeState(esn_ready=True, start_paused=False, electrode_labels=("CA3", "EC"),
        ctx_index=1, pulse_threshold_std=3, pulse_window_sec=10,
        controller_ready={"esn": True, "rule": True, "pid": True}, pulse_max_amplitude_v=0.25)
    state.set_mode("closed-loop")
    state.set_stim("threshold_pulse", 0.5, max_gain=10)
    state.acknowledge_drain(state.snapshot().generation)
    with pytest.raises(ValueError, match="configured safety"):
        state.set_armed(True)


@pytest.mark.parametrize("controller", ["esn", "rule", "pid"])
def test_mock_cli_starts_and_stops_with_each_selected_controller(controller):
    app_dir = Path(app.__file__).resolve().parent
    with tempfile.TemporaryDirectory() as directory:
        env = os.environ.copy()
        temp = str(Path(directory) / "tmp")
        Path(temp).mkdir()
        env["TEMP"] = temp
        env["TMP"] = temp
        result = subprocess.run(
            [sys.executable, str(app_dir / "echoChamber.py"), "--mock", "--controller", controller,
             "--start-paused", "--no-browser", "--port", "0", "--run-seconds", "0.2",
             "--record-dir", str(Path(directory) / "recordings")],
            cwd=app_dir, env=env, text=True, capture_output=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Mock replay acquisition started" in result.stderr
        assert "Shutdown complete" in result.stderr


def test_mock_cli_keeps_rule_available_without_esn_artifact():
    app_dir = Path(app.__file__).resolve().parent
    with tempfile.TemporaryDirectory() as directory:
        env = os.environ.copy()
        temp = str(Path(directory) / "tmp")
        Path(temp).mkdir()
        env["TEMP"] = temp
        env["TMP"] = temp
        result = subprocess.run(
            [sys.executable, str(app_dir / "echoChamber.py"), "--mock", "--controller", "rule",
             "--model-artifact", str(Path(directory) / "missing.pkl"), "--start-paused",
             "--no-browser", "--port", "0", "--run-seconds", "0.2",
             "--record-dir", str(Path(directory) / "recordings")],
            cwd=app_dir, env=env, text=True, capture_output=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert '"controller": "rule"' in result.stderr
        assert "Shutdown complete" in result.stderr
