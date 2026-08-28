"""Causal streaming wrapper around the inference-only ESN recurrence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

import numpy as np

from .artifact import InferenceArtifact
from .inference import Backend, StatefulESN
from .preprocessing import (
    StreamingDecimator,
    StreamingFIR,
    StreamingUpsampler,
    build_lowpass_fir,
)


@dataclass
class ThresholdPulseConfig:
    """Pulse-shape metadata consumed by the application-owned bridge."""

    min_interval_sec: float = 0.5
    pulse_duration_ms: float = 5.0
    pulse_freq_hz: float = 100.0
    waveform: Literal["sine", "square"] = "sine"


@dataclass
class InferenceStreamer:
    """Return raw CA3 predictions in target units at the DAQ sample rate.

    Stimulation mapping is intentionally owned by ``esn_bridge.py``.
    """

    artifact: InferenceArtifact
    backend: Backend = "auto"
    up_method: Literal["linear", "zoh"] = "linear"
    stim_clip_v: Tuple[float, float] = (-10.0, 10.0)
    pulse_cfg: ThresholdPulseConfig = field(default_factory=ThresholdPulseConfig)

    fs_in: int = field(init=False)
    fs_train: int = field(init=False)
    chunk_size: int = field(init=False)
    decim_q: int = field(init=False)
    _model: StatefulESN = field(init=False)
    _dec: StreamingDecimator = field(init=False)
    _fir: StreamingFIR = field(init=False)
    _up: StreamingUpsampler = field(init=False)

    def __post_init__(self) -> None:
        self.artifact.validate()
        self.fs_in = self.artifact.fs_input_hz
        self.fs_train = self.artifact.fs_model_hz
        self.chunk_size = self.artifact.preferred_chunk_size
        self.decim_q = self.fs_in // self.fs_train
        anti_alias = build_lowpass_fir(
            float(self.fs_in), self.artifact.aa_cutoff_hz, self.artifact.aa_numtaps
        )
        model_band = build_lowpass_fir(
            float(self.fs_train),
            self.artifact.model_cutoff_hz,
            self.artifact.model_numtaps,
        )
        self._dec = StreamingDecimator(self.decim_q, StreamingFIR(anti_alias))
        self._fir = StreamingFIR(model_band)
        self._up = StreamingUpsampler(self.decim_q, self.up_method)
        self._model = StatefulESN(self.artifact, backend=self.backend)

    @property
    def backend_name(self) -> str:
        return self._model.backend

    def configure(self, **_: object) -> None:
        """Compatibility no-op; the application bridge owns stimulation."""

    def reset(self) -> None:
        self._dec.reset()
        self._fir.reset()
        self._up.reset()
        self._model.reset()

    def process_chunk(self, ai_chunk: np.ndarray, *, ctx_index: int = 1) -> np.ndarray:
        data = np.asarray(ai_chunk, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.chunk_size:
            raise ValueError(
                f"expected AI shape (channels, {self.chunk_size}); got {data.shape}"
            )
        if ctx_index < 0 or ctx_index >= data.shape[0]:
            raise ValueError(f"ctx_index={ctx_index} out of range for {data.shape[0]} channels")
        x_model = self._dec.process(data[ctx_index])
        x_model = self._fir.process(x_model)
        x_scaled = x_model * self.artifact.input_scale + self.artifact.input_offset
        y_scaled = self._model.run(x_scaled)
        y_model = (y_scaled - self.artifact.target_offset) / self.artifact.target_scale
        y_daq = self._up.process(y_model, out_len=self.chunk_size)
        if y_daq.shape != (self.chunk_size, 1):
            raise RuntimeError(f"inference streamer produced unexpected shape {y_daq.shape}")
        return y_daq.T
