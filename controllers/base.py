from __future__ import annotations

from typing import Protocol

from .types import ControllerInput, StepResult


class Controller(Protocol):
    name: str

    def step(self, data: ControllerInput) -> StepResult: ...
    def reset(self, *, keep_calibration: bool = False) -> None: ...
    def drain(self) -> None: ...
