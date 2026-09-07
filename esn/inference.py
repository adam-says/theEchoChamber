"""Small deterministic ESN recurrence for inference only."""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from .artifact import InferenceArtifact


Backend = Literal["auto", "numpy", "numba"]


def _run_numpy(
    values: np.ndarray,
    W: np.ndarray,
    Win: np.ndarray,
    bias: np.ndarray,
    Wout: np.ndarray,
    readout_bias: np.ndarray,
    leak_rate: float,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.empty((values.shape[0], 1), dtype=np.float64)
    current = state.copy()
    keep = 1.0 - leak_rate
    win = Win[:, 0]
    wout = Wout[:, 0]
    for index in range(values.shape[0]):
        current = keep * current + leak_rate * np.tanh(
            W @ current + win * values[index, 0] + bias
        )
        output[index, 0] = np.dot(wout, current) + readout_bias[0]
    return output, current


def _build_numba_runner() -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    try:
        from numba import njit
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("Numba backend requested but numba is not installed") from exc

    return njit(cache=True, fastmath=False)(_run_numpy)


class StatefulESN:
    """Stateful recurrence matching ReservoirPy 0.3.11's internal ESN update."""

    def __init__(self, artifact: InferenceArtifact, backend: Backend = "auto") -> None:
        if backend not in ("auto", "numpy", "numba"):
            raise ValueError(f"unsupported ESN backend: {backend}")
        self.artifact = artifact
        self.state = artifact.initial_state.copy()
        self.backend_requested = backend
        self.backend = "numpy"
        self._runner: Callable[..., tuple[np.ndarray, np.ndarray]] = _run_numpy
        if backend in ("auto", "numba"):
            try:
                self._runner = _build_numba_runner()
                self.backend = "numba"
                # Compile before acquisition.  The result is discarded.
                self._runner(
                    np.zeros((1, 1), dtype=np.float64),
                    artifact.W,
                    artifact.Win,
                    artifact.reservoir_bias,
                    artifact.Wout,
                    artifact.readout_bias,
                    artifact.leak_rate,
                    self.state,
                )
            except Exception:
                if backend == "numba":
                    raise
                self._runner = _run_numpy
                self.backend = "numpy"

    def reset(self) -> None:
        self.state = self.artifact.initial_state.copy()

    def run(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.ndim != 2 or data.shape[1] != 1:
            raise ValueError(f"ESN input must have shape (samples, 1); got {data.shape}")
        if not np.all(np.isfinite(data)):
            raise ValueError("ESN input contains non-finite values")
        output, state = self._runner(
            data,
            self.artifact.W,
            self.artifact.Win,
            self.artifact.reservoir_bias,
            self.artifact.Wout,
            self.artifact.readout_bias,
            self.artifact.leak_rate,
            self.state,
        )
        self.state = np.asarray(state, dtype=np.float64)
        return np.asarray(output, dtype=np.float64)
