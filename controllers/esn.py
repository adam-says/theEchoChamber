"""Thin typed adapter around the existing ESN bridge and unchanged artifact."""
from __future__ import annotations

import numpy as np

from esn_bridge import EchoChamberEsnBridge
from .types import ControllerAction, ControllerDiagnostics, ControllerInput, StepResult


class EsnController:
    name = "esn"

    def __init__(self, bridge: EchoChamberEsnBridge):
        self.bridge = bridge

    def configure(self, **kwargs) -> None:
        self.bridge.configure(**kwargs)

    @property
    def ready(self) -> bool:
        return True

    def reset(self, *, keep_calibration: bool = False) -> None:
        self.bridge.reset()

    def drain(self) -> None:
        return None

    def step(self, data: ControllerInput) -> StepResult:
        if data.samples_mv.ndim != 2 or data.samples_mv.shape[1] != self.bridge.runtime_chunk_size:
            raise ValueError("input block shape does not match ESN bridge runtime chunk size")
        if not np.all(data.validity_mask):
            self.bridge.reset()
            action = ControllerAction()
        else:
            self.bridge.process(data.samples_mv, ctx_index=data.ctx_index, render_legacy=False)
            details = self.bridge.detailed_diagnostics()
            if self.bridge.stim_mode == "off":
                action = ControllerAction()
            elif self.bridge.stim_mode == "threshold_pulse":
                onset = data.first_sample_index + int(details.trigger_sample) if details.pulse_fired and details.trigger_sample is not None else None
                action = ControllerAction(kind="pulse_events", pulse_sample_indices=() if onset is None else (onset,))
            else:
                # Keep the existing bridge's conditioning and unit-to-AO conversion unchanged.
                values = details.passthrough_output
                if values is None:
                    raise RuntimeError("ESN bridge omitted its conditioned passthrough output")
                action = ControllerAction(kind="analog", analog_v=values.reshape(-1).copy())
        details = self.bridge.detailed_diagnostics()
        diag = ControllerDiagnostics(self.name, True, True, float("nan"), float("nan"), float("nan"),
            float("nan"), details.pulse_threshold, details.pulse_peak, 0.0, "active", data.generation,
            "threshold event" if details.pulse_fired else "")
        return StepResult(action, diag)
