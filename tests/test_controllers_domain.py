import numpy as np
import pytest

from controllers import (ActuatorFeedback, ControllerAction, ControllerInput, CausalEnvelope,
                         EsnController, PIDController, PulseActuator, RuleController)
from esn_bridge import EchoChamberEsnBridge


def make_input(x, start, fs, *, valid=None, generation=1, feedback=None, ctx_index=0):
    x = np.asarray(x, dtype=float)
    n = x.size
    return ControllerInput(np.vstack((x, np.zeros(n))), start, fs, ctx_index,
        np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool),
        np.zeros(n, dtype=bool), ("CA3", "EC"), ("CA3", "EC"), start + n - 1,
        "CA3", generation, feedback)


def blocks(signal, block):
    for start in range(0, len(signal), block):
        yield start, signal[start:start + block]


def test_rule_calibrates_then_triggers_with_hysteresis():
    fs = 1000
    rng = np.random.default_rng(41)
    amplitude = np.r_[np.full(11500, .1), np.full(1000, 1.0), np.full(1000, .1), np.full(1000, 1.0), np.full(1500, .1)]
    signal = amplitude * rng.normal(size=16000)
    c = RuleController(fs)
    actions = []
    diag = None
    for start, data in blocks(signal, 100):
        result = c.step(make_input(data, start, fs))
        actions.extend(result.action.pulse_sample_indices)
        diag = result.diagnostics
    assert diag is not None and diag.calibrated
    assert any(11500 <= event < 12500 for event in actions)
    assert any(13500 <= event < 14500 for event in actions)
    assert diag.baseline_mean_mv > 0


def test_baseline_is_frozen_after_calibration():
    fs = 1000
    rng = np.random.default_rng(7)
    c = RuleController(fs)
    baseline = None
    next_index = 0
    for start, x in blocks(rng.normal(scale=.1, size=12000), 100):
        d = c.step(make_input(x, start, fs)).diagnostics
        next_index = start + x.size
        if d.calibrated:
            baseline = (d.baseline_mean_mv, d.baseline_std_mv)
            break
    assert baseline is not None
    for start, x in blocks(rng.normal(scale=2, size=2000), 100):
        d = c.step(make_input(x, next_index + start, fs)).diagnostics
    assert (d.baseline_mean_mv, d.baseline_std_mv) == baseline


def test_pid_nonzero_synthetic_preset_saturates_and_zero_gain_is_safe():
    fs = 1000
    rng = np.random.default_rng(8)
    x = rng.normal(scale=.1, size=13000)
    c = PIDController(fs, kp=.4, ki=.1, kd=.02, target_z=1)
    for start, data in blocks(x, 100):
        result = c.step(make_input(data, start, fs))
    rates = []
    for start in range(13000, 14000, 100):
        result = c.step(make_input(rng.normal(scale=3, size=100), start, fs))
        rates.append(result.action.requested_rate_hz)
    assert result.diagnostics.calibrated
    assert result.action.kind == "pulse_rate"
    assert any(0 < rate <= 2 for rate in rates)
    safe = PIDController(fs)
    # Seed calibration using same long unstimulated valid signal.
    for start, data in blocks(x, 100):
        safe_result = safe.step(make_input(data, start, fs))
    for start in range(13000, 14000, 100):
        safe_result = safe.step(make_input(rng.normal(scale=3, size=100), start, fs))
    assert safe_result.action.requested_rate_hz == 0


def test_actuator_boundaries_zero_invalid_refractory_and_no_catchup():
    fs = 1000
    actuator = PulseActuator()
    actuator.arm()
    valid = np.ones(100, dtype=bool)
    rate = ControllerAction(kind="pulse_rate", requested_rate_hz=2.0)
    emitted = 0
    for start in range(0, 500, 100):
        emitted += actuator.process(rate, first_sample_index=start, sample_count=100,
            sample_rate_hz=fs, armed=True, validity_mask=valid, amplitude_v=0.5, generation=1).rendered_count
    assert emitted == 1
    assert actuator.process(ControllerAction(kind="pulse_rate", requested_rate_hz=0),
        first_sample_index=500, sample_count=100, sample_rate_hz=fs, armed=True,
        validity_mask=valid, generation=1).rendered_count == 0
    invalid = valid.copy(); invalid[30] = False
    assert actuator.process(rate, first_sample_index=600, sample_count=100, sample_rate_hz=fs,
        armed=True, validity_mask=invalid, generation=1).rendered_count == 0
    # A long elapsed interval must not turn fractional phase into multiple catch-up pulses.
    result = actuator.process(rate, first_sample_index=2000, sample_count=1000, sample_rate_hz=fs,
        armed=True, validity_mask=np.ones(1000, bool), amplitude_v=0.5, generation=1)
    assert result.rendered_count == 2
    onsets = [event.source_onset_sample_index for event in result.pulse_events if event.status == "rendered_for_ao"]
    assert len(onsets) == 2 and onsets[1] - onsets[0] >= 500
    assert np.max(np.abs(result.waveform_v)) <= .5


def test_actuator_keeps_refractory_across_reset_and_cancels_rate_phase_on_disarm():
    fs = 1000
    actuator = PulseActuator(); actuator.arm()
    valid = np.ones(100, bool)
    event = ControllerAction(kind="pulse_events", pulse_sample_indices=(0,))
    assert actuator.process(event, first_sample_index=0, sample_count=100, sample_rate_hz=fs,
        armed=True, validity_mask=valid, amplitude_v=.2, generation=1).rendered_count == 1
    actuator.reset(preserve_refractory=True)
    result = actuator.process(ControllerAction(kind="pulse_events", pulse_sample_indices=(100,)),
        first_sample_index=100, sample_count=100, sample_rate_hz=fs, armed=True,
        validity_mask=valid, amplitude_v=.2, generation=2)
    assert result.pulse_events[0].status == "rejected_refractory"
    actuator.disarm()
    assert actuator.rate_phase == 0


def test_envelope_chunk_invariance_freezes_masked_baseline_and_invalidates_gap():
    fs = 1000
    rng = np.random.default_rng(902)
    signal = rng.normal(scale=.2, size=14000)
    stim = np.zeros(signal.size, bool); stim[3000:5000] = True
    whole = CausalEnvelope(fs); chunked = CausalEnvelope(fs)
    all_env = whole.process(signal, 0, np.ones(signal.size, bool), stim)
    pieces = []
    for start in range(0, signal.size, 137):
        end = min(signal.size, start + 137)
        pieces.append(chunked.process(signal[start:end], start, np.ones(end-start, bool), stim[start:end]))
    np.testing.assert_allclose(np.concatenate(pieces), all_env, rtol=0, atol=1e-13, equal_nan=True)
    assert whole.baseline == chunked.baseline
    frozen = whole.baseline
    whole.process(rng.normal(scale=10, size=100), signal.size, np.ones(100, bool), np.ones(100, bool))
    assert whole.baseline == frozen
    whole.process(np.ones(100), signal.size + 102, np.ones(100, bool))
    assert whole.baseline is None


def test_masked_large_artifact_is_held_out_of_filter_and_pid_recovery():
    fs = 1000
    rng = np.random.default_rng(77)
    pre = rng.normal(scale=.2, size=100)
    artifact = np.full(50, 1e9)
    held = np.full(50, pre[-1])
    post = rng.normal(scale=.2, size=100)
    actual = CausalEnvelope(fs); reference = CausalEnvelope(fs)
    actual.baseline = reference.baseline = (.1, .02)
    a = np.r_[actual.process(pre, 0, np.ones(100, bool)),
        actual.process(artifact, 100, np.zeros(50, bool), np.ones(50, bool)),
        actual.process(post, 150, np.ones(100, bool))]
    b = np.r_[reference.process(pre, 0, np.ones(100, bool)),
        reference.process(held, 100, np.zeros(50, bool), np.ones(50, bool)),
        reference.process(post, 150, np.ones(100, bool))]
    np.testing.assert_allclose(a, b, atol=0, rtol=0, equal_nan=True)
    assert actual.baseline == reference.baseline == (.1, .02)

    controllers = [PIDController(fs, kp=.1, ki=.2, kd=.01), PIDController(fs, kp=.1, ki=.2, kd=.01)]
    for pid in controllers:
        pid.envelope.baseline = (.1, .02)
        pid.integral = .25
        pid.previous_z = 0.0
        pid.derivative = 1.0
    for pid, values in zip(controllers, (artifact, held)):
        inp = make_input(values, 100, fs, valid=np.zeros(50, bool))
        inp = ControllerInput(inp.samples_mv, 100, fs, 0, inp.validity_mask, np.ones(50, bool),
            inp.channel_labels, inp.roles, 149, "CA3", 1)
        pid.step(inp)
    for pid in controllers:
        assert pid.integral == .25 and pid.derivative == 0 and pid.rate == 0
    results = [pid.step(make_input(post, 150, fs)) for pid in controllers]
    assert results[0].action.requested_rate_hz == results[1].action.requested_rate_hz


def test_flat_and_nonfinite_calibration_stays_unready_with_reason():
    env = CausalEnvelope(1000)
    result = env.process(np.zeros(12000), 0, np.ones(12000, bool))
    assert np.all(np.isfinite(result))
    assert env.baseline is None
    assert "flat" in env.calibration_reason
    nonfinite = CausalEnvelope(1000)
    values = np.ones(100); values[50] = np.nan
    nonfinite.process(values, 0, np.ones(100, bool))
    assert nonfinite.baseline is None
    assert "non-finite" in nonfinite.calibration_reason


def test_source_key_includes_channel_identity_and_calibration_block_is_not_controlled():
    fs = 1000
    c = RuleController(fs)
    c.envelope.baseline = (.1, .02)
    c._source = ("CA3", 0, "adc-0", "CA3", float(fs))
    data = make_input(np.ones(100), 0, fs)
    changed = ControllerInput(data.samples_mv, data.first_sample_index, fs, data.ctx_index,
        data.validity_mask, data.stimulated_mask, ("changed-adc", "EC"), data.roles,
        data.latest_sample_index, data.region, data.generation)
    result = c.step(changed)
    assert not result.diagnostics.calibrated
    assert result.action.kind == "none"


def test_calibration_completion_block_cannot_generate_backdated_rule_event():
    fs = 1000
    env = CausalEnvelope(fs, settle_s=0, calibration_s=.01)
    c = RuleController(fs, envelope=env)
    first = c.step(make_input(np.random.default_rng(11).normal(size=100), 0, fs))
    assert first.diagnostics.calibrated
    assert first.action.kind == "none"
    assert np.isnan(first.diagnostics.z_score)


def test_acquisition_index_gap_resets_baseline_and_suppresses_both_controllers():
    fs = 1000
    rule = RuleController(fs)
    rule.envelope.baseline = (.1, .02)
    rule.envelope.last_index = 99
    rule._source = ("CA3", 0, "CA3", "CA3", float(fs))
    high = np.random.default_rng(12).normal(scale=10, size=100)
    result = rule.step(make_input(high, 200, fs))
    assert result.action.kind == "none"
    assert not result.diagnostics.calibrated

    pid = PIDController(fs, kp=1, ki=1, kd=1)
    pid.envelope.baseline = (.1, .02)
    pid.envelope.last_index = 99
    pid._source = ("CA3", 0, "CA3", "CA3", float(fs))
    pid.integral = 2.0; pid.derivative = 3.0; pid.previous_z = 4.0
    result = pid.step(make_input(high, 200, fs))
    assert result.action.kind == "none"
    assert not result.diagnostics.calibrated
    assert pid.integral == 0 and pid.derivative == 0 and pid.previous_z is None


def test_pid_derivative_sign_saturation_rejection_and_masked_recovery():
    fs = 1000
    pid = PIDController(fs, kp=0, ki=0, kd=.01)
    pid.previous_z = 0.0
    assert pid._compute_output(1.0) > 0  # rising activity commands more inhibition
    saturated = PIDController(fs, kp=5, ki=1, kd=0)
    assert saturated._compute_output(10.0) == 2.0
    assert saturated.integral == 0.0  # conditional antiwindup at upper bound

    pid.envelope.baseline = (.1, .02)
    pid.integral = .4; pid.derivative = 10.0; pid.previous_z = 1.0
    stim_input = make_input(np.ones(5), 0, fs)
    stim_input = ControllerInput(stim_input.samples_mv, 0, fs, 0, stim_input.validity_mask,
        np.ones(5, bool), stim_input.channel_labels, stim_input.roles, 4, "CA3", 1)
    pid.step(stim_input)
    assert pid.integral == .4 and pid.derivative == 0 and pid.previous_z is None
    follow = make_input(np.ones(1), 5, fs)
    pid.step(follow)
    assert pid.derivative == 0  # derivative is reseeded after the masked interval

    pid.integral = .25
    rejected = make_input(np.ones(5), 6, fs, feedback=ActuatorFeedback(False, "rejected"))
    pid.step(rejected)
    assert pid.integral == .25 and pid.rate == 0
    pid.integral = .3
    unarmed = make_input(np.ones(5), 11, fs, feedback=ActuatorFeedback(True, armed=False))
    pid.step(unarmed)
    assert pid.integral == .3 and pid.rate == 0


def test_pid_updates_only_on_configured_sample_time_grid():
    pid = PIDController(1000, kp=1, target_z=-10, update_ms=5)
    pid.envelope.baseline = (.1, .02)
    first = make_input(np.zeros(4), 0, 1000)
    result = pid.step(first)
    assert result.action.requested_rate_hz == 0
    on_grid = make_input(np.zeros(1), 4, 1000)
    result = pid.step(on_grid)
    assert result.action.requested_rate_hz > 0


@pytest.mark.parametrize("kwargs", [
    {"sample_rate_hz": float("nan")},
    {"sample_rate_hz": 1000, "band_hz": (0, 250)},
    {"sample_rate_hz": 1000, "band_hz": (100, 500)},
    {"sample_rate_hz": 1000, "band_hz": (300, 100)},
    {"sample_rate_hz": 1000, "smooth_ms": 0},
    {"sample_rate_hz": 1000, "settle_s": -1},
    {"sample_rate_hz": 1000, "calibration_s": 0},
    {"sample_rate_hz": 1000, "calibration_s": float("inf")},
])
def test_envelope_rejects_invalid_feature_configuration(kwargs):
    kwargs.setdefault("sample_rate_hz", 1000)
    with pytest.raises(ValueError):
        CausalEnvelope(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"kp": float("nan")}, {"ki": float("inf")}, {"kd": -1},
    {"target_z": float("nan")}, {"direction": 0}, {"max_rate_hz": 2.01},
    {"max_rate_hz": float("nan")}, {"update_ms": 0}, {"update_ms": 1e-4},
    {"update_ms": float("inf")}, {"derivative_tau_s": -1},
    {"derivative_tau_s": float("nan")},
])
def test_pid_rejects_invalid_configuration_before_runtime(kwargs):
    with pytest.raises(ValueError):
        PIDController(1000, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"upper_z": float("nan")}, {"lower_z": float("inf")},
    {"lower_z": -0.1}, {"upper_z": 1, "lower_z": 1},
])
def test_rule_rejects_invalid_threshold_configuration(kwargs):
    with pytest.raises(ValueError):
        RuleController(1000, **kwargs)


def test_ca3_selection_is_case_insensitive_and_independent_of_channel_order():
    fs = 1000
    c = RuleController(fs)
    c.envelope.baseline = (.1, .02)
    x = np.random.default_rng(33).normal(size=100)
    samples = np.vstack((np.zeros(100), x))
    data = ControllerInput(samples, 0, fs, 0, np.ones(100, bool), np.zeros(100, bool),
        ("cortex-input", "ca3-input"), ("cortex", "ca3"), 99, "CA3", 1)
    c.step(data)
    assert c._source[1] == 1
    assert c._source[2] == "ca3-input"
    duplicate = ControllerInput(samples, 100, fs, 0, np.ones(100, bool), np.zeros(100, bool),
        ("a", "b"), ("CA3", "ca3"), 199, "CA3", 1)
    with pytest.raises(ValueError, match="exactly one CA3"):
        c.step(duplicate)


@pytest.mark.parametrize("kwargs", [
    {"pulse_duration_ms": float("nan")}, {"pulse_frequency_hz": float("inf")},
    {"refractory_s": -1},
])
def test_actuator_rejects_invalid_waveform_configuration(kwargs):
    with pytest.raises(ValueError):
        PulseActuator(**kwargs)


def test_actuator_analog_requires_both_arm_gates_and_explicit_commit_tail():
    fs = 1000
    valid = np.ones(20, bool)
    actuator = PulseActuator(pulse_duration_ms=10, refractory_s=.005)
    analog = ControllerAction(kind="analog", analog_v=np.ones(20))
    assert not np.any(actuator.process(analog, first_sample_index=0, sample_count=20,
        sample_rate_hz=fs, armed=True, validity_mask=valid, generation=1).waveform_v)
    actuator.arm()
    assert not np.any(actuator.process(analog, first_sample_index=0, sample_count=20,
        sample_rate_hz=fs, armed=False, validity_mask=valid, generation=1).waveform_v)
    np.testing.assert_array_equal(actuator.process(analog, first_sample_index=0, sample_count=20,
        sample_rate_hz=fs, armed=True, validity_mask=valid, generation=1).waveform_v, 1)

    event = ControllerAction(kind="pulse_events", pulse_sample_indices=(15,))
    first = actuator.process(event, first_sample_index=0, sample_count=20, sample_rate_hz=fs,
        armed=True, validity_mask=valid, amplitude_v=.5, generation=1, scheduler_lead_samples=100)
    assert first.pulse_events[0].target_onset_sample_index == 115
    pid = first.pulse_events[0].pulse_id
    actuator.commit(pid, 3)
    assert actuator.cancel_uncommitted() == ()
    tail = actuator.process(ControllerAction(), first_sample_index=20, sample_count=20,
        sample_rate_hz=fs, armed=False, validity_mask=valid, generation=2)
    full_wave = .5 * np.sin(2*np.pi*100*np.arange(10)/fs)
    timeline = np.r_[first.waveform_v, tail.waveform_v]
    np.testing.assert_allclose(timeline[15:25], full_wave, atol=1e-14)
    assert tail.pulse_events[0].pulse_id == pid
    assert actuator.committed_pulse_ids() == (pid,)
    # A wholly uncommitted request is canceled and cannot appear after a switch.
    staged = actuator.process(ControllerAction(kind="pulse_events", pulse_sample_indices=(40,)),
        first_sample_index=40, sample_count=20, sample_rate_hz=fs, armed=True,
        validity_mask=valid, amplitude_v=.5, generation=2)
    staged_id = next(event.pulse_id for event in staged.pulse_events if event.status == "rendered_for_ao")
    assert staged_id in actuator.cancel_uncommitted()
    late = actuator.process(ControllerAction(), first_sample_index=60, sample_count=20,
        sample_rate_hz=fs, armed=False, validity_mask=valid, generation=3)
    assert not np.any(late.waveform_v)  # a late cancellation never moves an expired tail to a new index


def test_actuator_chunk_invariant_rate_crossings_and_rejects_old_or_future_events():
    fs = 1000
    action = ControllerAction(kind="pulse_rate", requested_rate_hz=2)
    def run(chunks):
        a = PulseActuator(refractory_s=.1); a.arm(); events=[]; start=0
        for count_samples in chunks:
            result = a.process(action, first_sample_index=start, sample_count=count_samples,
                sample_rate_hz=fs, armed=True, validity_mask=np.ones(count_samples, bool),
                generation=1)
            events.extend(e.source_onset_sample_index for e in result.pulse_events if e.status == "rendered_for_ao")
            start += count_samples
        return events
    assert run([1000]) == run([100] * 10)
    a = PulseActuator(); a.arm()
    old = a.process(ControllerAction(kind="pulse_events", pulse_sample_indices=(9,)),
        first_sample_index=10, sample_count=10, sample_rate_hz=fs, armed=True,
        validity_mask=np.ones(10, bool), generation=1)
    future = a.process(ControllerAction(kind="pulse_events", pulse_sample_indices=(30,)),
        first_sample_index=10, sample_count=10, sample_rate_hz=fs, armed=True,
        validity_mask=np.ones(10, bool), generation=1)
    assert old.pulse_events[0].status == "rejected_expired"
    assert future.pulse_events[0].status == "rejected_future"


def test_pulse_plan_stages_complete_waveform_with_one_source_to_target_lead():
    actuator = PulseActuator(pulse_duration_ms=10, refractory_s=0)
    actuator.arm()
    result = actuator.process(ControllerAction(kind="pulse_events", pulse_sample_indices=(80,)),
        first_sample_index=0, sample_count=100, sample_rate_hz=1000, armed=True,
        validity_mask=np.ones(100, bool), amplitude_v=.25, generation=7,
        scheduler_lead_samples=2000, latest_sample_index=99)

    plan, = result.pulse_plans
    event, = result.pulse_events
    assert plan.pulse_id == event.pulse_id
    assert (plan.generation, plan.source_onset_sample_index, plan.target_onset_sample_index) == (7, 80, 2080)
    assert plan.latest_sample_index == 99
    assert plan.waveform_v.shape == (10,)
    assert not plan.waveform_v.flags.writeable
    np.testing.assert_array_equal(plan.waveform_v, result.waveform_v[80:90])
    # The scheduler anchors this full waveform at target 2080, so its exclusive
    # end is 2090; the source-to-target lead is applied exactly once.
    assert plan.target_onset_sample_index + plan.waveform_v.size == 2090
    # A generation switch can cancel the actuator's uncommitted local render,
    # while a scheduler that already admitted the plan retains its exact target.
    assert actuator.cancel_uncommitted() == (plan.pulse_id,)
    assert plan.target_onset_sample_index == 2080


class _FakePulseConfig:
    min_interval_sec = .5
    pulse_duration_ms = 5.0
    pulse_freq_hz = 100.0
    waveform = "sine"


class _FakeEsnStreamer:
    chunk_size = 100
    fs_in = 1000
    fs_train = 1000
    decim_q = 1
    stim_clip_v = (-10.0, 10.0)
    pulse_cfg = _FakePulseConfig()

    def configure(self, **kwargs):
        pass

    def reset(self):
        pass

    def process_chunk(self, values, *, ctx_index):
        return values[ctx_index:ctx_index + 1].copy()


def test_esn_threshold_routes_through_shared_pulse_renderer_and_parity_is_preserved():
    bridge = EchoChamberEsnBridge(_FakeEsnStreamer(), runtime_chunk_size=100,
        sample_rate=1000, passthrough_dc_block_hz=0, pulse_threshold_std=1,
        pulse_window_sec=1.0)
    bridge.configure(stim_mode="threshold_pulse", stim_gain=1)
    controller = EsnController(bridge)
    def esn_input(x, start):
        n = len(x)
        return ControllerInput(np.vstack((np.zeros(n), np.asarray(x, float))), start, 1000, 1,
            np.ones(n, bool), np.zeros(n, bool), ("CA3", "EC"), ("CA3", "EC"),
            start + n - 1, "EC", 1)
    for start in range(0, 1000, 100):
        controller.step(esn_input(np.zeros(100), start))
    spike = np.zeros(100); spike[50] = 10
    result = controller.step(esn_input(spike, 1000))
    assert result.action.kind == "pulse_events"
    assert result.action.pulse_sample_indices == (1050,)
    actuator = PulseActuator(); actuator.arm()
    actual = actuator.process(result.action, first_sample_index=1000, sample_count=100,
        sample_rate_hz=1000, armed=True, validity_mask=np.ones(100, bool), amplitude_v=.4,
        generation=1).waveform_v
    reference = PulseActuator(); reference.arm()
    expected = reference.process(ControllerAction(kind="pulse_events", pulse_sample_indices=(1050,)),
        first_sample_index=1000, sample_count=100, sample_rate_hz=1000, armed=True,
        validity_mask=np.ones(100, bool), amplitude_v=.4, generation=1).waveform_v
    np.testing.assert_array_equal(actual, expected)
    detailed = bridge.detailed_diagnostics()
    np.testing.assert_array_equal(detailed.model_output[0], spike)

    # Passthrough explicitly returns the legacy bridge conversion unchanged; off returns no action.
    controller.configure(stim_mode="passthrough", stim_gain=1)
    passed = controller.step(esn_input(spike, 1100))
    assert passed.action.kind == "analog"
    np.testing.assert_array_equal(passed.action.analog_v, bridge.detailed_diagnostics().passthrough_output[0])
    controller.configure(stim_mode="off")
    assert controller.step(esn_input(spike, 1200)).action.kind == "none"
