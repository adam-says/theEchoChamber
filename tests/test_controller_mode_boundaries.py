from __future__ import annotations

import pytest

import echoChamber as app


def make_state(controller: str = "esn") -> app.RuntimeState:
    return app.RuntimeState(
        esn_ready=True,
        start_paused=False,
        electrode_labels=("CA3", "EC"),
        ctx_index=1,
        pulse_threshold_std=3.0,
        pulse_window_sec=10.0,
        controller=controller,
        controller_ready={"esn": True, "rule": True, "pid": True},
    )


def visible_stim_state(state: app.RuntimeState) -> tuple[object, ...]:
    snapshot = state.snapshot()
    return (
        snapshot.controller,
        snapshot.stim_mode,
        snapshot.stim_gain,
        snapshot.generation,
        snapshot.transition_pending,
        snapshot.armed,
    )


@pytest.mark.parametrize("controller", ["rule", "pid"])
def test_rule_and_pid_reject_passthrough_without_changing_runtime_state(controller: str) -> None:
    state = make_state(controller)
    before = visible_stim_state(state)

    with pytest.raises(ValueError, match="fixed pulse output"):
        state.set_stim("passthrough", 1.0)

    # The UI-facing mode, gain, selected controller and transaction state stay
    # at their prior values after the rejected request.
    assert visible_stim_state(state) == before


@pytest.mark.parametrize("controller", ["rule", "pid"])
def test_selected_rule_or_pid_still_rejects_passthrough_after_controller_switch(
    controller: str,
) -> None:
    state = make_state("esn")
    assert state.set_controller(controller)
    before = visible_stim_state(state)

    with pytest.raises(ValueError, match="fixed pulse output"):
        state.set_stim("passthrough", 1.0)

    assert visible_stim_state(state) == before


def test_esn_retains_passthrough_across_controller_selection_transactions() -> None:
    state = make_state("rule")

    # After selecting ESN, its legacy analog mode remains an accepted setting.
    assert state.set_controller("esn")
    before = state.snapshot()
    state.set_stim("passthrough", 1.0)
    passthrough = state.snapshot()
    assert passthrough.controller == "esn"
    assert passthrough.stim_mode == "passthrough"
    assert passthrough.stim_gain == 1.0
    assert passthrough.generation == before.generation + 1
    assert passthrough.transition_pending
    assert not passthrough.armed

    # A later switch makes the Rule boundary authoritative; rejecting the same
    # mode leaves the selected controller and displayed setting untouched.
    state.set_stim("off", 0.0)
    assert state.set_controller("rule")
    rule_before = visible_stim_state(state)
    with pytest.raises(ValueError, match="fixed pulse output"):
        state.set_stim("passthrough", 1.0)
    assert visible_stim_state(state) == rule_before

    # Switching back to ESN restores the accepted passthrough boundary.
    state.set_stim("off", 0.0)
    assert state.set_controller("esn")
    state.set_stim("passthrough", 1.0)
    final = state.snapshot()
    assert final.controller == "esn"
    assert final.stim_mode == "passthrough"
    assert final.stim_gain == 1.0
