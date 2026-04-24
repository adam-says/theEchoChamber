from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


DEFAULT_HYPERPARAMS: Dict[str, Any] = {
    # From ESN_test_working.ipynb / hyperopt config
    "units": 10,
    "sr": 5.684959305880698,
    "lr": 0.3665234728579373,
    "iss": 0.01750341268204161,
    "ridge": 4.4065269894699847e-10,
    "seed": 1234,
}


def build_esn(
    units: int,
    sr: float,
    lr: float,
    iss: float,
    ridge: float,
    seed: int = 1234,
    *,
    rc_connectivity: float = 0.1,
    input_connectivity: float = 0.1,
    bias_scaling: float = 1.0,
) -> Any:
    """
    Build the reservoirpy ESN graph: Input >> Reservoir >> Ridge.
    Lazy-imports reservoirpy so the module can be imported without deps installed.
    """
    from reservoirpy.nodes import Reservoir, Ridge, Input

    source = Input()
    reservoir = Reservoir(
        units=units,
        lr=lr,
        sr=sr,
        input_scaling=iss,
        bias_scaling=bias_scaling,
        rc_connectivity=rc_connectivity,
        input_connectivity=input_connectivity,
        seed=seed,
    )
    readout = Ridge(ridge=ridge)
    model = source >> reservoir >> readout
    return model


@dataclass
class ESNModel:
    """
    Thin wrapper around a reservoirpy model to provide a stable interface.
    """

    model: Any
    fs_train: int = 2000

    def reset(self) -> None:
        # reservoirpy exposes reset via run/fit kwargs; for safety, do nothing here.
        # Streaming resets are handled by using reset=True on fit and by clearing preprocess state.
        try:
            self.model.reset()
        except Exception:
            pass

    def fit(self, x: np.ndarray, y: np.ndarray, *, reset: bool = True) -> "ESNModel":
        self.model = self.model.fit(x, y, reset=reset)
        return self

    def run(self, x: np.ndarray, *, reset: bool = False) -> np.ndarray:
        return self.model.run(x, reset=reset)

    def step(self, x: np.ndarray) -> np.ndarray:
        """
        Step the model with one sample (or a short sequence). reservoirpy keeps state across calls.
        x: shape (n, in_dim) or (in_dim,)
        returns y: shape (n, out_dim)
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        # Graph models in reservoirpy are callable.
        y = self.model(x)
        return np.asarray(y)

