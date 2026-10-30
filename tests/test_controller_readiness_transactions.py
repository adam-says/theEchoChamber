"""Readiness must belong to the installed controller transaction generation."""
from __future__ import annotations

import threading

import numpy as np
import pytest

import echoChamber as app
from controllers import CausalEnvelope, RuleController


class _NoUiCore(app.ProcessingCore):
    def _publish_ui(self, *args, **kwargs):
        pass


def _settings(band_hz=(100.0, 250.0)):
    return {"upper_z": 3.0, "lower_z": 1.5, "band_hz": band_hz}


def _state(settings=None):
    state = app.RuntimeState(
        esn_ready=False, start_paused=False, electrode_labels=("CA3", "EC"), ctx_index=1,
        pulse_threshold_std=3.0, pulse_window_sec=10.0, controller="rule",
        controller_ready={"esn": False, "rule": True, "pid": True},
        controller_settings={"rule": dict(settings or _settings())},
    )
    state.set_stim("threshold_pulse", 0.5)
    state.set_mode("closed-loop")
    state.acknowledge_drain(state.snapshot().generation)
    return state


def _core(state, settings=None):
    config = app.AppConfig(processing_block_ms=5.0, ao_target_lead_ms=100.0,
                           controller_settings={"rule": dict(settings or _settings())})
    config.validate()
    recorder = object()  # recording is disabled in these state-boundary tests
    core = _NoUiCore(config, state, app.EsnRuntime(None, config), recorder,
                     object(), None, app.Telemetry())
    return config, core


def _input_block(config, first_sample_index):
    # Deterministic broadband input gives the short test calibration a finite,
    # non-flat envelope without replacing RuleController's real feature path.
    rng = np.random.default_rng(20261029 + first_sample_index)
    return rng.normal(0.0, 0.2, size=(2, config.chunk_size))


def _process(core, config, sample_index):
    return core.process(
        _input_block(config, sample_index), sample_index=sample_index,
        ao_target_sample_index=sample_index + config.ao_lead_samples,
        input_is_electrode_mV=True, input_roles=("CA3", "Cortex"),
        input_labels=("CA3", "EC"), input_ctx_index=1,
    )


def test_old_calibrated_rule_result_cannot_restore_readiness_after_band_edit():
    old_settings = _settings((100.0, 250.0))
    state = _state(old_settings)
    config, core = _core(state, old_settings)
    # Short calibration is only a test fixture; the production Rule feature,
    # ProcessingCore, and RuntimeState transaction boundaries remain real.
    old_rule = RuleController(
        config.sample_rate, upper_z=3.0, lower_z=1.5,
        band_hz=old_settings["band_hz"],
        envelope=CausalEnvelope(config.sample_rate, old_settings["band_hz"],
                                settle_s=0.0, calibration_s=config.chunk_size / config.sample_rate),
    )
    core.controllers["rule"] = old_rule

    _, _, _, reason = _process(core, config, 0)
    assert reason is None
    assert old_rule.envelope.baseline is not None
    assert state.snapshot().controller_calibrated

    computed = threading.Event()
    release = threading.Event()
    returned = []
    errors = []
    original_step = old_rule.step

    def compute_then_pause(inp):
        result = original_step(inp)
        assert result.diagnostics.calibrated
        returned.append(result)
        computed.set()
        if not release.wait(5.0):
            raise TimeoutError("test did not release the old Rule result")
        return result

    old_rule.step = compute_then_pause

    def process_old_generation():
        try:
            _process(core, config, config.chunk_size)
        except Exception as exc:  # transport to the owning test thread
            errors.append(exc)

    worker = threading.Thread(target=process_old_generation, name="blocked-old-rule-step")
    worker.start()
    try:
        assert computed.wait(3.0), "old Rule step did not reach the post-computation barrier"
        assert returned and returned[0].diagnostics.calibrated

        new_settings = _settings((110.0, 250.0))
        prepared = app.build_controller("rule", config.sample_rate, new_settings)
        state.set_controller_settings("rule", new_settings, prepared)
        changed_generation = state.snapshot().generation
        assert state.snapshot().transition_pending
        state.acknowledge_drain(changed_generation)
        assert not state.snapshot().transition_pending
        assert not state.snapshot().controller_calibrated
    finally:
        release.set()
        worker.join(timeout=5.0)

    assert not worker.is_alive(), "blocked core processing did not finish"
    assert not errors, errors
    ready_after_old_result = state.snapshot().controller_calibrated
    arm_rejected = False
    try:
        state.set_armed(True)
    except ValueError:
        arm_rejected = True

    assert not ready_after_old_result and arm_rejected, (
        "a calibrated result from the prior band/generation restored readiness and allowed arming "
        "after drain acknowledgement"
    )


def test_reusable_baseline_drain_ack_does_not_arm_before_prepared_controller_is_applied():
    settings = _settings((100.0, 250.0))
    state = _state(settings)
    state.update_controller_calibration("rule", True)
    config, core = _core(state, settings)
    old_rule = RuleController(config.sample_rate, **settings)
    old_rule.envelope.baseline = (0.25, 0.05)
    core.controllers["rule"] = old_rule

    # Same-band settings preserve the reusable baseline, but the newly prepared
    # controller remains unapplied until the next owner-thread process boundary.
    prepared = app.build_controller("rule", config.sample_rate, settings)
    state.set_controller_settings("rule", settings, prepared)
    generation = state.snapshot().generation
    assert state.snapshot().controller_calibrated
    state.acknowledge_drain(generation)
    assert not state.snapshot().transition_pending

    arm_rejected = False
    try:
        state.set_armed(True)
    except ValueError:
        arm_rejected = True
    assert arm_rejected, (
        "drain acknowledgement alone allowed arming before the prepared controller was installed"
    )


def test_settings_revision_survives_generation_changes_switches_and_latest_edit():
    """Install the latest selected-controller revision before calibration and rearm."""
    initial = _settings((100.0, 250.0))
    state = _state(initial)
    state.update_controller_calibration("rule", True)
    config, core = _core(state, initial)
    old_rule = RuleController(config.sample_rate, **initial)
    old_rule.envelope.baseline = (0.25, 0.05)
    core.controllers["rule"] = old_rule

    first_revision = _settings((110.0, 250.0))
    first_prepared = app.build_controller("rule", config.sample_rate, first_revision)
    state.set_controller_settings("rule", first_revision, first_prepared)
    assert state.set_controller("pid")
    assert state.set_controller("rule")

    # A newer edit replaces the still-pending Rule object. The currently
    # selected controller's latest revision is the one Core must install.
    latest_revision = _settings((120.0, 250.0))
    latest_prepared = app.build_controller("rule", config.sample_rate, latest_revision)
    latest_prepared.envelope.settle_samples = 0
    latest_prepared.envelope.calibration_samples = config.chunk_size
    state.set_controller_settings("rule", latest_revision, latest_prepared)
    state.set_stim("threshold_pulse", 0.6)  # advances generation after that edit
    generation = state.snapshot().generation
    state.acknowledge_drain(generation)
    assert not state.snapshot().transition_pending

    blocked_before_install = False
    try:
        state.set_armed(True)
    except ValueError:
        blocked_before_install = True
    assert blocked_before_install, "settings edit and drain acknowledgement armed before Core applied it"

    _, _, _, reason = _process(core, config, 0)
    assert reason is None
    assert core.controllers["rule"] is latest_prepared
    assert core.controllers["rule"].envelope.band_hz == latest_revision["band_hz"]
    assert state.snapshot().controller_calibrated
    assert not state.snapshot().transition_pending

    # Once the owner thread installed the latest revision and that controller
    # produced a calibrated result, an explicit arm must succeed despite the
    # intervening controller switches and later stimulus-generation change.
    state.set_armed(True)
    assert state.snapshot().armed
