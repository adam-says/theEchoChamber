from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np


def build_lowpass_fir(fs: float, cutoff_hz: float = 25.0, numtaps: int = 8000) -> np.ndarray:
    """
    Build FIR coefficients matching the notebook preprocessing:
    scipy.signal.firwin(numtaps, cutoff, window='hamming', pass_zero='lowpass', scale=True, fs=fs)
    """
    # Lazy import to keep module importable even without scipy installed.
    from scipy.signal import firwin

    return firwin(
        numtaps,
        cutoff_hz,
        window="hamming",
        pass_zero="lowpass",
        scale=True,
        fs=fs,
    ).astype(np.float64)


@dataclass
class StreamingFIR:
    """Stateful FIR filter using scipy.signal.lfilter with persistent zi."""

    b: np.ndarray
    a: np.ndarray = field(default_factory=lambda: np.array([1.0], dtype=np.float64))
    zi: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.b = np.asarray(self.b, dtype=np.float64)
        self.a = np.asarray(self.a, dtype=np.float64)

    def reset(self) -> None:
        self.zi = None

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        x: shape (n,) or (n, channels)
        returns same shape as x
        """
        from scipy.signal import lfilter

        x = np.asarray(x, dtype=np.float64)
        orig_shape = x.shape
        if x.ndim == 1:
            x2 = x[:, None]
        elif x.ndim == 2:
            x2 = x
        else:
            raise ValueError(f"StreamingFIR expects 1D or 2D input, got shape={x.shape}")

        n_ch = x2.shape[1]
        if self.zi is None:
            # A zero state is explicit, reproducible, and matches offline
            # causal ``lfilter`` preprocessing.  ``lfilter_zi`` represents the
            # steady state for a unit step; using it unscaled injected an
            # arbitrary startup transient into every recording.
            state_size = max(self.a.size, self.b.size) - 1
            self.zi = np.zeros((state_size, n_ch), dtype=np.float64)

        y = np.empty_like(x2, dtype=np.float64)
        for ch in range(n_ch):
            y[:, ch], self.zi[:, ch] = lfilter(self.b, self.a, x2[:, ch], zi=self.zi[:, ch])

        if orig_shape == y.shape:
            return y
        return y[:, 0]


@dataclass
class StreamingDecimator:
    """
    Streaming decimator for integer factor q.
    Expects chunks aligned to q; if not, it will buffer remainder samples.
    """

    q: int = 10
    aa_fir: Optional[StreamingFIR] = None
    _buffer: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._buffer = None
        if self.aa_fir is not None:
            self.aa_fir.reset()

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        x: shape (n,) or (n, channels)
        returns: shape (ceil(n/q) or floor with buffering, channels)
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x2 = x[:, None]
        elif x.ndim == 2:
            x2 = x
        else:
            raise ValueError(f"StreamingDecimator expects 1D or 2D input, got shape={x.shape}")

        if self._buffer is not None:
            x2 = np.vstack([self._buffer, x2])
            self._buffer = None

        if self.aa_fir is not None:
            x2 = self.aa_fir.process(x2)

        n = x2.shape[0]
        n_out = n // self.q
        n_used = n_out * self.q
        if n_used < n:
            self._buffer = x2[n_used:, :]
            x2 = x2[:n_used, :]

        if x2.size == 0:
            return np.zeros((0, x2.shape[1]), dtype=np.float64)

        y = x2[:: self.q, :]
        return y


@dataclass
class StreamingUpsampler:
    """
    Streaming upsampler for integer factor q.
    Maintains last sample to provide continuity for linear interpolation.
    """

    q: int = 10
    method: Literal["linear", "zoh"] = "linear"
    _last: Optional[np.ndarray] = None  # shape (channels,)

    def reset(self) -> None:
        self._last = None

    def process(self, x: np.ndarray, out_len: Optional[int] = None) -> np.ndarray:
        """
        x: shape (n, channels) or (n,) (treated as single channel)
        out_len: optional explicit output length (default n*q)
        returns shape (out_len, channels)
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x2 = x[:, None]
        elif x.ndim == 2:
            x2 = x
        else:
            raise ValueError(f"StreamingUpsampler expects 1D or 2D input, got shape={x.shape}")

        n, ch = x2.shape
        if n == 0:
            return np.zeros((0, ch), dtype=np.float64)

        if out_len is None:
            out_len = n * self.q

        if self.method == "zoh":
            y = np.repeat(x2, self.q, axis=0)
            y = y[:out_len, :]
            self._last = x2[-1, :].copy()
            return y

        # linear interpolation
        # Build anchor points with continuity: prepend last value if available
        if self._last is None:
            anchors = x2
        else:
            anchors = np.vstack([self._last[None, :], x2])

        # Interpolate between successive anchors at q steps per interval
        intervals = anchors.shape[0] - 1
        if intervals <= 0:
            y = np.repeat(anchors, out_len, axis=0)[:out_len, :]
            self._last = anchors[-1, :].copy()
            return y

        t = np.linspace(0.0, 1.0, self.q, endpoint=False, dtype=np.float64)[:, None]
        pieces = []
        for i in range(intervals):
            start = anchors[i, :][None, :]
            end = anchors[i + 1, :][None, :]
            seg = start + (end - start) * t
            pieces.append(seg)

        y = np.vstack(pieces)
        if self._last is None:
            # If no previous anchor, we generated intervals == n-1 segments => (n-1)*q.
            # To align with expected n*q, append the final sample q times.
            if y.shape[0] < n * self.q:
                tail = np.repeat(x2[-1:, :], n * self.q - y.shape[0], axis=0)
                y = np.vstack([y, tail])

        y = y[:out_len, :]
        self._last = x2[-1, :].copy()
        return y

