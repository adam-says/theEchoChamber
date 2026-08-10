from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict, Literal, Optional, Tuple

import numpy as np
from collections import deque

from .model import ESNModel
from .preprocessing import StreamingDecimator, StreamingFIR, StreamingUpsampler, build_lowpass_fir


StimMode = Literal["off", "passthrough", "threshold_pulse"]


def _clip_to_range(x: np.ndarray, min_v: float = -10.0, max_v: float = 10.0) -> np.ndarray:
    return np.clip(x, min_v, max_v)


@dataclass
class ThresholdPulseConfig:
    """
    Heuristic pulse generator driven by ESN output. Designed to be safe and tunable.
    """

    window_sec: float = 1.0
    thresh_std: float = 3.0
    min_interval_sec: float = 0.5
    pulse_duration_ms: float = 5.0
    pulse_freq_hz: float = 100.0
    waveform: Literal["sine", "square"] = "sine"


@dataclass
class ESNStreamer:
    """
    Real-time streamer:
    - picks CTX channel
    - causal FIR lowpass (stateful)
    - decimates 20 kHz -> 2 kHz
    - scales
    - stateful ESN stepping
    - inverse scale
    - upsample 2 kHz -> 20 kHz
    - maps to stimulation output mode
    """

    esn: ESNModel
    scaler: any
    fs_in: int = 20000
    fs_train: int = 2000
    chunk_size: int = 100
    cutoff_hz: float = 25.0
    fir_numtaps: int = 8000
    aa_cutoff_hz: float = 500.0
    aa_numtaps: int = 201
    decim_q: int = 10
    up_method: Literal["linear", "zoh"] = "linear"

    stim_mode: StimMode = "passthrough"
    stim_gain: float = 1.0
    stim_clip_v: Tuple[float, float] = (-10.0, 10.0)
    pulse_cfg: ThresholdPulseConfig = field(default_factory=ThresholdPulseConfig)

    _fir: StreamingFIR = field(init=False)
    _dec: StreamingDecimator = field(init=False)
    _up: StreamingUpsampler = field(init=False)
    _recent: Deque[float] = field(init=False)
    _samples_since_pulse: int = field(default=10**9, init=False)

    def __post_init__(self) -> None:
        if self.fs_in % self.fs_train != 0:
            raise ValueError(f"fs_in ({self.fs_in}) must be integer-multiple of fs_train ({self.fs_train}).")
        self.decim_q = int(self.fs_in // self.fs_train)
        if self.decim_q <= 0:
            raise ValueError("Invalid decimation factor.")
        if self.chunk_size % self.decim_q != 0:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be multiple of decim_q ({self.decim_q}) for fixed-latency chunks."
            )
        
        aa_b = build_lowpass_fir(fs=float(self.fs_in),cutoff_hz=self.aa_cutoff_hz,numtaps=self.aa_numtaps)
        aa_fir = StreamingFIR(b=aa_b)

        self._dec = StreamingDecimator(q=self.decim_q, aa_fir=aa_fir)

        b = build_lowpass_fir(fs=float(self.fs_in), cutoff_hz=self.aa_cutoff_hz, numtaps=self.aa_numtaps)
        self._fir = StreamingFIR(b=b)
        self._up = StreamingUpsampler(q=self.decim_q, method=self.up_method)

        maxlen = int(self.pulse_cfg.window_sec * self.fs_train)
        self._recent = deque(maxlen=max(1, maxlen))

    def configure(
        self,
        *,
        stim_mode: Optional[StimMode] = None,
        stim_gain: Optional[float] = None,
        pulse_cfg: Optional[ThresholdPulseConfig] = None,
    ) -> None:
        if stim_mode is not None:
            self.stim_mode = stim_mode
        if stim_gain is not None:
            self.stim_gain = float(stim_gain)
        if pulse_cfg is not None:
            self.pulse_cfg = pulse_cfg
            maxlen = int(self.pulse_cfg.window_sec * self.fs_train)
            self._recent = deque(self._recent, maxlen=max(1, maxlen))

    def reset(self) -> None:
        self._fir.reset()
        self._dec.reset()
        self._up.reset()
        self._recent.clear()
        self._samples_since_pulse = 10**9
        try:
            self.esn.reset()
        except Exception:
            pass

    def _stim_map_passthrough(self, y_20k: np.ndarray) -> np.ndarray:
        stim = y_20k * self.stim_gain
        stim = _clip_to_range(stim, *self.stim_clip_v)
        return stim

    def _stim_map_threshold_pulse(self, y_2k: np.ndarray, y_20k: np.ndarray) -> np.ndarray:
        # Update recent buffer at 2 kHz
        for v in y_2k[:, 0].tolist():
            self._recent.append(float(v))

        if len(self._recent) < max(5, int(0.1 * self.fs_train)):
            return np.zeros_like(y_20k)

        mu = float(np.mean(self._recent))
        sigma = float(np.std(self._recent)) + 1e-12
        thresh = mu + self.pulse_cfg.thresh_std * sigma

        min_interval_samples_2k = int(self.pulse_cfg.min_interval_sec * self.fs_train)
        self._samples_since_pulse += y_2k.shape[0]

        should_fire = (float(np.max(y_2k[:, 0])) > thresh) and (self._samples_since_pulse >= min_interval_samples_2k)
        if not should_fire:
            return np.zeros_like(y_20k)

        self._samples_since_pulse = 0

        # Build pulse at 20 kHz
        dur_samples = max(1, int((self.pulse_cfg.pulse_duration_ms / 1000.0) * self.fs_in))
        dur_samples = min(dur_samples, y_20k.shape[0])
        t = np.arange(dur_samples, dtype=np.float64) / float(self.fs_in)

        if self.pulse_cfg.waveform == "square":
            wave = np.sign(np.sin(2.0 * np.pi * self.pulse_cfg.pulse_freq_hz * t))
        else:
            wave = np.sin(2.0 * np.pi * self.pulse_cfg.pulse_freq_hz * t)

        stim = np.zeros_like(y_20k)
        stim[:dur_samples, 0] = wave * self.stim_gain
        stim = _clip_to_range(stim, *self.stim_clip_v)
        return stim

    def process_chunk(self, ai_chunk: np.ndarray, *, ctx_index: int = 1) -> np.ndarray:
        """
        ai_chunk: shape (n_channels, chunk_size) at fs_in
        returns: stimulation array shape (1, chunk_size) at fs_in
        """
        ai_chunk = np.asarray(ai_chunk, dtype=np.float64)
        if ai_chunk.ndim != 2:
            raise ValueError(f"ai_chunk must be 2D (channels, samples); got {ai_chunk.shape}")
        if ai_chunk.shape[1] != self.chunk_size:
            raise ValueError(f"Expected chunk_size={self.chunk_size}, got {ai_chunk.shape[1]}")
        if ctx_index < 0 or ctx_index >= ai_chunk.shape[0]:
            raise ValueError(f"ctx_index={ctx_index} out of range for channels={ai_chunk.shape[0]}")

        x_20k = ai_chunk[ctx_index, :]

        # Anti-alias filter at 20 kHz, then decimate 20 kHz -> 2 kHz.
        # StreamingDecimator applies its aa_fir before taking every q-th sample.
        x_2k = self._dec.process(x_20k)  # (10,1)

        # Apply the existing ESN-band low-pass at 2 kHz
        x_2k_f = self._fir.process(x_2k)  # (10,1)

        # Scale
        x_scaled = self.scaler.transform(x_2k_f)  # (10,1)

        # ESN step (sequence of 10 samples)
        y_scaled = self.esn.step(x_scaled)  # (10,1)

        # Inverse scale
        y_2k = self.scaler.inverse_transform(y_scaled)

        # Upsample 2k -> 20k to match DAQ chunk size
        y_20k = self._up.process(y_2k, out_len=self.chunk_size)  # (100,1)

        if self.stim_mode == "off":
            stim = np.zeros_like(y_20k)
        elif self.stim_mode == "threshold_pulse":
            stim = self._stim_map_threshold_pulse(y_2k=y_2k, y_20k=y_20k)
        else:
            stim = self._stim_map_passthrough(y_20k=y_20k)

        return stim.T  # (1, chunk_size)

