"""Shared typed contract for interchangeable controller implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ControllerInput:
    samples_mv: np.ndarray
    first_sample_index: int
    sample_rate_hz: float
    ctx_index: int
    validity_mask: np.ndarray
    stimulated_mask: np.ndarray | None
    channel_labels: tuple[str, ...]
    roles: tuple[str, ...]
    latest_sample_index: int
    region: str
    generation: int
    feedback: "ActuatorFeedback | None" = None


@dataclass(frozen=True)
class ActuatorFeedback:
    accepted: bool
    reason: str = ""
    applied_rate_hz: float | None = None
    armed: bool = True


@dataclass(frozen=True)
class ControllerAction:
    kind: Literal["none", "pulse_events", "pulse_rate", "analog"] = "none"
    requested_rate_hz: float = 0.0
    analog_v: np.ndarray | None = None
    pulse_sample_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class ControllerDiagnostics:
    controller: str
    ready: bool
    calibrated: bool
    envelope_mv: float
    baseline_mean_mv: float
    baseline_std_mv: float
    z_score: float
    target_z: float
    error: float
    output_rate_hz: float
    state: str
    generation: int
    reason: str = ""
    feature_valid: bool = True
    feature_recovery_samples: int = 0
    feature_recovery_remaining_samples: int = 0
    stimulated_sample_count: int = 0
    feature_invalid_sample_count: int = 0


@dataclass(frozen=True)
class StepResult:
    action: ControllerAction
    diagnostics: ControllerDiagnostics


def ca3_channel_index(roles: tuple[str, ...]) -> int:
    """Validate regional roles and find CA3 without assuming physical channel order."""
    matches = [i for i, role in enumerate(roles) if str(role).casefold() == "ca3"]
    if len(matches) != 1:
        raise ValueError(f"controller input requires exactly one CA3 role; found {len(matches)}")
    if len(roles) == 2:
        cortical = [i for i, role in enumerate(roles) if str(role).casefold() in {"cortex", "ec"}]
        if len(cortical) != 1 or cortical[0] == matches[0]:
            raise ValueError("two-channel controller input requires one CA3 and one Cortex/EC role")
    return matches[0]
