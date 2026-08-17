"""Application-owned compatibility layer between Echo Chamber and the ESN.

The `esn` package is treated as a frozen prediction engine.  
This module adapts DAQ-sized blocks to the artifact's preferred chunk
size and owns all Echo Chamber stimulation mapping and diagnostics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np


StimMode = Literal["off", "passthrough", "threshold_pulse"]


@dataclass(frozen=True)
class BridgeDiagnostics:
    model_output: np.ndarray
    pulse_threshold: float
    pulse_peak: float
    pulse_fired: bool


class EchoChamberEsnBridge:
    """Adapt an unmodified ESN streamer to the application's runtime contract."""

    def __init__(
        self,
        streamer: Any,
        *,
        runtime_chunk_size: int,
        sample_rate: int,
        passthrough_dc_block_hz: float = 0.5,
        pulse_threshold_std: float = 3.0,
        pulse_window_sec: float = 10.0,
        ao_command_gain_v_per_esn_unit: float = 1.0,
    ) -> None:
        self.streamer = streamer
        self.runtime_chunk_size = int(runtime_chunk_size)
        self.sample_rate = int(sample_rate)
        self.artifact_sample_rate = int(getattr(streamer, "fs_in", self.sample_rate))
        self.preferred_chunk_size = int(getattr(streamer, "chunk_size", 0))
        self.model_rate = int(getattr(streamer, "fs_train", 2_000))
        self.decim_q = int(getattr(streamer, "decim_q", self.sample_rate // self.model_rate))
        self.ao_command_gain_v_per_esn_unit = float(ao_command_gain_v_per_esn_unit)
        if self.runtime_chunk_size <= 0 or self.preferred_chunk_size <= 0:
            raise ValueError("runtime and artifact chunk sizes must be positive")
        if self.sample_rate != self.artifact_sample_rate:
            raise ValueError(
                f"DAQ sample_rate={self.sample_rate} does not match artifact "
                f"fs_in={self.artifact_sample_rate}; explicit resampling is required"
            )
        if self.runtime_chunk_size % self.preferred_chunk_size:
            raise ValueError(
                f"runtime chunk_size={self.runtime_chunk_size} must be a multiple of "
                f"artifact chunk_size={self.preferred_chunk_size}"
            )
        if self.sample_rate <= 0 or self.model_rate <= 0 or self.sample_rate % self.model_rate:
            raise ValueError("sample rate must be an integer multiple of the ESN model rate")
        if self.decim_q != self.sample_rate // self.model_rate:
            raise ValueError("artifact decimation factor is inconsistent with its sample rates")
        if not np.isfinite(self.ao_command_gain_v_per_esn_unit) or self.ao_command_gain_v_per_esn_unit <= 0:
            raise ValueError("AO command gain must be finite and positive")

        self.stim_mode: StimMode = "passthrough"
        self.stim_gain = 1.0
        self.passthrough_dc_block_hz = 0.0
        self.pulse_threshold_std = 1.0
        self.pulse_window_sec = 10.0
        pulse_cfg = getattr(streamer, "pulse_cfg", None)
        self.pulse_min_interval_sec = float(getattr(pulse_cfg, "min_interval_sec", 0.5))
        self.pulse_duration_ms = float(getattr(pulse_cfg, "pulse_duration_ms", 5.0))
        self.pulse_freq_hz = float(getattr(pulse_cfg, "pulse_freq_hz", 100.0))
        self.pulse_waveform = str(getattr(pulse_cfg, "waveform", "sine"))
        self.stim_clip_v = tuple(getattr(streamer, "stim_clip_v", (-10.0, 10.0)))

        self._recent: deque[float] = deque()
        self._samples_since_pulse = 10**9
        self._dc_previous_input: Optional[float] = None
        self._dc_previous_output = 0.0
        self._diagnostics = BridgeDiagnostics(
            np.full((1, self.runtime_chunk_size), np.nan), float("nan"), float("nan"), False
        )

        # The original streamer already contains stimulation modes. Keep it in
        # its identity-like passthrough configuration and perform all choices
        # after its returned prediction crosses this bridge.
        self.streamer.configure(stim_mode="passthrough", stim_gain=1.0)
        # The original passthrough mapper clips before returning. This artifact
        # predicts around 21.8, so its default +/-10 clip would collapse the
        # prediction to a constant 10 before the application can DC-block it.
        # Preserve that configured range above for the final bridge command,
        # but make the prediction boundary itself non-clipping.
        if hasattr(self.streamer, "stim_clip_v"):
            self.streamer.stim_clip_v = (-float("inf"), float("inf"))
        self.configure(
            stim_mode="passthrough",
            stim_gain=1.0,
            passthrough_dc_block_hz=passthrough_dc_block_hz,
            pulse_threshold_std=pulse_threshold_std,
            pulse_window_sec=pulse_window_sec,
        )

    @classmethod
    def load(
        cls,
        artifact: Path | str,
        **kwargs: Any,
    ) -> "EchoChamberEsnBridge":
        # Progress bars and terminal output do not belong in the real-time path.
        try:
            from reservoirpy.utils import verbosity

            verbosity(0)
        except Exception:
            pass
        from esn import load_artifact

        return cls(load_artifact(str(artifact)), **kwargs)

    def configure(
        self,
        *,
        stim_mode: Optional[StimMode] = None,
        stim_gain: Optional[float] = None,
        passthrough_dc_block_hz: Optional[float] = None,
        pulse_threshold_std: Optional[float] = None,
        pulse_window_sec: Optional[float] = None,
    ) -> None:
        if stim_mode is not None:
            if stim_mode not in {"off", "passthrough", "threshold_pulse"}:
                raise ValueError(f"invalid stimulation mode: {stim_mode}")
            self.stim_mode = stim_mode
        if stim_gain is not None:
            if not np.isfinite(stim_gain) or stim_gain < 0:
                raise ValueError("stimulation gain must be finite and non-negative")
            self.stim_gain = float(stim_gain)
        if passthrough_dc_block_hz is not None:
            if not np.isfinite(passthrough_dc_block_hz) or passthrough_dc_block_hz < 0:
                raise ValueError("DC-block frequency must be finite and non-negative")
            self.passthrough_dc_block_hz = float(passthrough_dc_block_hz)
        if pulse_threshold_std is not None:
            if not np.isfinite(pulse_threshold_std) or pulse_threshold_std <= 0:
                raise ValueError("pulse threshold must be finite and positive")
            self.pulse_threshold_std = float(pulse_threshold_std)
        if pulse_window_sec is not None:
            if not np.isfinite(pulse_window_sec) or pulse_window_sec < 1:
                raise ValueError("pulse window must be finite and at least one second")
            self.pulse_window_sec = float(pulse_window_sec)
            maxlen = max(1, int(self.pulse_window_sec * self.model_rate))
            self._recent = deque(self._recent, maxlen=maxlen)

    def reset(self) -> None:
        self.streamer.reset()
        self._recent.clear()
        self._samples_since_pulse = 10**9
        self._dc_previous_input = None
        self._dc_previous_output = 0.0
        self._diagnostics = BridgeDiagnostics(
            np.full((1, self.runtime_chunk_size), np.nan), float("nan"), float("nan"), False
        )

    def process(self, ai_chunk: np.ndarray, *, ctx_index: int) -> np.ndarray:
        data = np.asarray(ai_chunk, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.runtime_chunk_size:
            raise ValueError(
                f"expected AI shape (channels, {self.runtime_chunk_size}); got {data.shape}"
            )
        predictions: list[np.ndarray] = []
        for start in range(0, self.runtime_chunk_size, self.preferred_chunk_size):
            piece = data[:, start:start + self.preferred_chunk_size]
            prediction = np.asarray(
                self.streamer.process_chunk(piece, ctx_index=ctx_index), dtype=np.float64
            )
            expected = (1, self.preferred_chunk_size)
            if prediction.shape != expected or not np.all(np.isfinite(prediction)):
                raise ValueError(f"ESN returned invalid output {prediction.shape}; expected {expected}")
            predictions.append(prediction)
        model_output = np.concatenate(predictions, axis=1)

        threshold, peak, fired = self._update_pulse_state(model_output)
        if self.stim_mode == "off":
            command = np.zeros_like(model_output)
        elif self.stim_mode == "threshold_pulse":
            command = self._pulse_command(model_output.shape[1]) if fired else np.zeros_like(model_output)
        else:
            command = self._passthrough_command(model_output)

        self._diagnostics = BridgeDiagnostics(model_output.copy(), threshold, peak, fired)
        return command

    def diagnostics(self, samples: int) -> tuple[np.ndarray, float, float, bool]:
        diagnostics = self._diagnostics
        model = diagnostics.model_output
        if model.shape != (1, samples):
            model = np.full((1, samples), np.nan)
        return model.copy(), diagnostics.pulse_threshold, diagnostics.pulse_peak, diagnostics.pulse_fired

    def _update_pulse_state(self, model_output: np.ndarray) -> tuple[float, float, bool]:
        # The frozen ESN exposes only its 20 kHz prediction. Sampling it at the
        # model rate establishes an explicit, public application boundary.
        values = np.asarray(model_output[0, ::self.decim_q], dtype=np.float64)
        peak = float(np.max(values))
        warmup = max(5, int(min(1.0, self.pulse_window_sec) * self.model_rate))
        threshold = float("nan")
        fired = False
        if len(self._recent) >= warmup:
            history = np.fromiter(self._recent, dtype=np.float64)
            threshold = float(np.mean(history) + self.pulse_threshold_std * (np.std(history) + 1e-12))
            interval = int(self.pulse_min_interval_sec * self.model_rate)
            self._samples_since_pulse += values.size
            fired = peak > threshold and self._samples_since_pulse >= interval
            if fired:
                self._samples_since_pulse = 0
        self._recent.extend(float(value) for value in values)
        return threshold, peak, fired

    def _passthrough_command(self, model_output: np.ndarray) -> np.ndarray:
        source = model_output.reshape(-1)
        if self.passthrough_dc_block_hz > 0:
            dt_s = 1.0 / self.sample_rate
            rc_s = 1.0 / (2.0 * np.pi * self.passthrough_dc_block_hz)
            alpha = rc_s / (rc_s + dt_s)
            blocked = np.empty_like(source)
            previous_input = self._dc_previous_input
            previous_output = self._dc_previous_output
            for index, value in enumerate(source):
                if previous_input is None:
                    previous_input = float(value)
                    blocked[index] = 0.0
                    continue
                previous_output = alpha * (previous_output + float(value) - previous_input)
                blocked[index] = previous_output
                previous_input = float(value)
            self._dc_previous_input = previous_input
            self._dc_previous_output = previous_output
            source = blocked
        command = (source.reshape(1, -1) * self.stim_gain
                   * self.ao_command_gain_v_per_esn_unit)
        return np.clip(command, float(self.stim_clip_v[0]), float(self.stim_clip_v[1]))

    def _pulse_command(self, samples: int) -> np.ndarray:
        duration = min(samples, max(1, int(self.pulse_duration_ms * self.sample_rate / 1_000.0)))
        time_s = np.arange(duration, dtype=np.float64) / self.sample_rate
        phase = 2.0 * np.pi * self.pulse_freq_hz * time_s
        wave = np.sign(np.sin(phase)) if self.pulse_waveform == "square" else np.sin(phase)
        command = np.zeros((1, samples), dtype=np.float64)
        command[0, :duration] = (
            wave * self.stim_gain * self.ao_command_gain_v_per_esn_unit
        )
        return np.clip(command, float(self.stim_clip_v[0]), float(self.stim_clip_v[1]))
