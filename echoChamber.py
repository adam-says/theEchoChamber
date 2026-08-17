"""The Echo Chamber

Main script.
This module intentionally treats ``esn`` as a read-only black box.  It adds the
hardware, safety, recording, monitoring, and test boundaries around it.

Recording format
----------------
Each recording is one portable HDF5 file containing compressed ``float32`` AI
and AO signals, calibration metadata, block-rate diagnostics, and sparse event
tables. Writes are batched on a dedicated thread so file I/O never runs in the
acquisition path.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import datetime as dt
import json
import logging
import math
import os
import queue
import re
import signal
import sys
import threading
import time
import traceback
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Optional

# The runtime ESN consists of very small matrix operations.  Allowing a BLAS
# library to create a worker pool for them is slower than one thread and can
# starve the NI AO scheduler.  These must be set before NumPy/SciPy load.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import h5py
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

try:
    import nidaqmx
    from nidaqmx.constants import (
        AcquisitionType,
        RegenerationMode,
        TerminalConfiguration,
    )
    from nidaqmx.stream_readers import AnalogMultiChannelReader
    from nidaqmx.stream_writers import AnalogSingleChannelWriter

    HAS_NIDAQMX = True
except (ImportError, OSError):
    nidaqmx = None
    AcquisitionType = RegenerationMode = TerminalConfiguration = None
    AnalogMultiChannelReader = AnalogSingleChannelWriter = None
    HAS_NIDAQMX = False


LOG = logging.getLogger("echoChamber")
BASE_DIR: Final = Path(__file__).resolve().parent
ESN_ARTIFACT: Final = BASE_DIR / "esn_artifact.pkl"
DEFAULT_MOCK_REPLAY: Final = BASE_DIR / "sample" / "test_AI0_CA3_AI1_CTX.h5"
H5_FORMAT_TAG: Final = "echoChamber_H5_v3"
VALID_MODES: Final = {"control", "closed-loop"}
VALID_STIM_MODES: Final = {"off", "passthrough", "threshold_pulse"}
RECORDED_MODE_VALUES: Final = {
    ("control", "off"): 0.0,
    ("control", "passthrough"): 0.0,
    ("control", "threshold_pulse"): 0.0,
    ("closed-loop", "off"): 1.0,
    ("closed-loop", "passthrough"): 2.0,
    ("closed-loop", "threshold_pulse"): 3.0,
}

# Scope/dummy-load tests only. Set this back to False before connecting AO to
# an isolator, electrode, or any biological preparation.
DRY_TEST_ALLOW_SUSTAINED_AO = False
# Removes the ESN prediction's DC baseline before passthrough gain/AO. Set to
# 0.0 only to compare the raw model output on a scope.
DEFAULT_PASSTHROUGH_DC_BLOCK_HZ = 0.5
# Pulse fires when the ESN output peak exceeds its recent mean by this many
# standard deviations. Scope-testing default; validate before stimulation.
DEFAULT_PULSE_THRESHOLD_STD = 3.0
# Length of the rolling ESN-output baseline used by threshold-pulse mode.
# It is collected continuously in all closed-loop stimulation modes.
DEFAULT_PULSE_WINDOW_SEC = 10.0


@dataclass(frozen=True)
class SafetyConfig:
    """Independent limits applied after the unmodified ESN runtime."""
# These parameters should work with the ISO-Flex stimulus isolator
# This should be for pulse-only mode, for now.

    max_command_v: float = 5.0
    max_slew_v_per_s: float = 2_000.0 # Not needed for pulse only
    max_abs_area_v_s: float = 0.010 # Not needed for pulse only
    area_window_s: float = 1.0 
    max_active_fraction: float = 0.10
    active_threshold_v: float = 3.5 # Value specific for the ISO-Flex
    max_consecutive_active_s: float = 0.050
    isolator_command_v_per_output_unit: Optional[float] = None
    allow_sustained_output_for_dry_test: bool = False

    def validate(self) -> None:
        positive = {
            "max_command_v": self.max_command_v,
            "max_slew_v_per_s": self.max_slew_v_per_s,
            "max_abs_area_v_s": self.max_abs_area_v_s,
            "area_window_s": self.area_window_s,
            "max_consecutive_active_s": self.max_consecutive_active_s,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.max_active_fraction <= 1:
            raise ValueError("max_active_fraction must be in (0, 1]")
        if self.isolator_command_v_per_output_unit is not None:
            if not math.isfinite(self.isolator_command_v_per_output_unit) or self.isolator_command_v_per_output_unit <= 0:
                raise ValueError("isolator conversion must be finite and positive")


@dataclass(frozen=True)
class AppConfig:
    device: str = "Dev1"
    ai_channels: tuple[str, str] = ("ai0", "ai1")
    ao_channel: str = "ao0"
    electrode_labels: tuple[str, str] = ("CA3", "Cortex")
    # MultiClamp Primary Output voltage gain (output mV / electrode mV).
    # Electrode-referred input is therefore DAQ_V * 1000 / gain, in mV.
    amplifier_gain: tuple[float, float] = (10.0, 10.0)
    # AO remains a voltage command. This maps one numeric ESN output unit
    # (currently assumed to be mV) to NI AO volts before the UI stim gain.
    ao_command_gain_v_per_esn_unit: float = 1.0
    ctx_index: int = 1
    sample_rate: int = 20_000
    processing_block_ms: float = 10.0
    ao_target_lead_ms: float = 100.0
    passthrough_dc_block_hz: float = DEFAULT_PASSTHROUGH_DC_BLOCK_HZ
    pulse_threshold_std: float = DEFAULT_PULSE_THRESHOLD_STD
    pulse_window_sec: float = DEFAULT_PULSE_WINDOW_SEC
    ai_min_v: float = -10.0
    ai_max_v: float = 10.0
    terminal_config: str = "RSE"
    visual_downsample: int = 2
    ui_interval_s: float = 0.1
    ws_host: str = "127.0.0.1"
    ws_port: int = 8765
    record_dir: Path = BASE_DIR / "recordings"
    logger_queue_blocks: int = 2_000
    recorder_batch_s: float = 0.25
    recorder_flush_s: float = 1.0
    ui_queue_packets: int = 1
    watchdog_timeout_s: float = 0.250
    mock_replay: Optional[Path] = None
    actual_stim_monitor_channel: Optional[str] = None
    start_paused: bool = False
    open_browser: bool = True
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    @property
    def physical_ai_channels(self) -> tuple[str, ...]:
        channels = tuple(self._physical(c) for c in self.ai_channels)
        if self.actual_stim_monitor_channel:
            channels += (self._physical(self.actual_stim_monitor_channel),)
        return channels

    @property
    def physical_ao_channel(self) -> str:
        return self._physical(self.ao_channel)

    @property
    def chunk_size(self) -> int:
        return int(round(self.sample_rate * self.processing_block_ms / 1_000.0))

    @property
    def ao_lead_samples(self) -> int:
        return int(round(self.sample_rate * self.ao_target_lead_ms / 1_000.0))

    @property
    def ao_lead_chunks(self) -> int:
        return self.ao_lead_samples // self.chunk_size

    @property
    def ao_deadline_samples(self) -> int:
        """Derived safe-command deadline; not an experiment setting."""
        return max(2 * self.chunk_size, self.ao_lead_samples // 2)

    def _physical(self, channel: str) -> str:
        return channel if "/" in channel else f"{self.device}/{channel}"

    def validate(self) -> None:
        if len(self.ai_channels) != 2 or len(set(self.ai_channels)) != 2:
            raise ValueError("exactly two distinct LFP AI channels are required")
        if len(self.amplifier_gain) != 2 or not all(
            math.isfinite(value) and value > 0 for value in self.amplifier_gain
        ):
            raise ValueError("two finite positive amplifier gains are required")
        if not math.isfinite(self.ao_command_gain_v_per_esn_unit) or self.ao_command_gain_v_per_esn_unit <= 0:
            raise ValueError("AO command gain must be finite and positive")
        if self.ctx_index not in (0, 1):
            raise ValueError("ctx_index must be 0 or 1")
        if self.sample_rate <= 0:
            raise ValueError("sample rate must be positive")
        if not math.isfinite(self.processing_block_ms) or self.processing_block_ms <= 0:
            raise ValueError("processing_block_ms must be finite and positive")
        exact_chunk_samples = self.sample_rate * self.processing_block_ms / 1_000.0
        if not math.isclose(exact_chunk_samples, round(exact_chunk_samples)):
            raise ValueError("processing_block_ms must resolve to a whole number of samples")
        if not math.isfinite(self.ao_target_lead_ms) or self.ao_target_lead_ms <= 0:
            raise ValueError("ao_target_lead_ms must be finite and positive")
        exact_lead_samples = self.sample_rate * self.ao_target_lead_ms / 1_000.0
        if not math.isclose(exact_lead_samples, round(exact_lead_samples)):
            raise ValueError("ao_target_lead_ms must resolve to a whole number of samples")
        if self.ao_lead_samples % self.chunk_size:
            raise ValueError("AO target lead must resolve to a whole number of processing chunks")
        if self.ao_lead_chunks < 4:
            raise ValueError("AO target lead must contain at least four processing chunks")
        if self.passthrough_dc_block_hz < 0:
            raise ValueError("passthrough_dc_block_hz must be non-negative")
        if not math.isfinite(self.pulse_threshold_std) or self.pulse_threshold_std <= 0:
            raise ValueError("pulse_threshold_std must be finite and positive")
        if not math.isfinite(self.pulse_window_sec) or self.pulse_window_sec < 1:
            raise ValueError("pulse_window_sec must be finite and at least one second")
        if self.sample_rate % 2_000 or self.chunk_size % (self.sample_rate // 2_000):
            raise ValueError("configuration is incompatible with the fixed ESN runtime")
        if self.ai_min_v >= self.ai_max_v:
            raise ValueError("invalid AI range")
        if self.visual_downsample <= 0 or self.logger_queue_blocks <= 0:
            raise ValueError("queue and downsample values must be positive")
        if not math.isfinite(self.recorder_batch_s) or self.recorder_batch_s <= 0:
            raise ValueError("recorder_batch_s must be finite and positive")
        if not math.isfinite(self.recorder_flush_s) or self.recorder_flush_s <= 0:
            raise ValueError("recorder_flush_s must be finite and positive")
        self.safety.validate()


@dataclass(frozen=True)
class StateSnapshot:
    running: bool
    acquiring: bool
    recording: bool
    mode: str
    stim_mode: str
    stim_gain: float
    pulse_threshold_std: float
    pulse_window_sec: float
    fault: Optional[str]
    esn_ready: bool
    electrode_labels: tuple[str, str]
    ctx_index: int


class RuntimeState:
    def __init__(self, *, esn_ready: bool, start_paused: bool,
                 electrode_labels: tuple[str, str], ctx_index: int,
                 pulse_threshold_std: float, pulse_window_sec: float) -> None:
        self._lock = threading.RLock()
        self._running = True
        self._acquiring = not start_paused
        self._recording = False
        self._mode = "control"
        self._stim_mode = "passthrough"
        self._stim_gain = 1.0
        self._pulse_threshold_std = pulse_threshold_std
        self._pulse_window_sec = pulse_window_sec
        self._fault: Optional[str] = None
        self._esn_ready = esn_ready
        self._electrode_labels = electrode_labels
        self._ctx_index = ctx_index

    def snapshot(self) -> StateSnapshot:
        with self._lock:
            return StateSnapshot(
                self._running,
                self._acquiring,
                self._recording,
                self._mode,
                self._stim_mode,
                self._stim_gain,
                self._pulse_threshold_std,
                self._pulse_window_sec,
                self._fault,
                self._esn_ready,
                self._electrode_labels,
                self._ctx_index,
            )

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._acquiring = False
            self._mode = "control"

    def set_acquiring(self, value: bool) -> None:
        with self._lock:
            if self._fault and value:
                raise ValueError("cannot start acquisition while faulted")
            self._acquiring = bool(value)

    def set_recording(self, value: bool) -> None:
        with self._lock:
            self._recording = bool(value)

    def set_mode(self, mode: str) -> bool:
        if mode not in VALID_MODES:
            raise ValueError(f"invalid mode: {mode}")
        with self._lock:
            if mode == "closed-loop" and not self._esn_ready:
                raise ValueError("closed-loop unavailable: ESN did not pass startup")
            if mode == "closed-loop" and self._fault:
                raise ValueError("closed-loop unavailable while faulted")
            changed = self._mode != mode
            self._mode = mode
            return changed

    def set_stim(self, stim_mode: str, gain: float, max_gain: float = 10.0) -> None:
        if stim_mode not in VALID_STIM_MODES:
            raise ValueError(f"invalid stimulation mode: {stim_mode}")
        if not math.isfinite(gain) or not 0 <= gain <= max_gain:
            raise ValueError(f"stim_gain must be finite and in [0, {max_gain}]")
        with self._lock:
            self._stim_mode = stim_mode
            self._stim_gain = float(gain)

    def set_pulse_threshold_std(self, value: float) -> None:
        if not math.isfinite(value) or not 0 < value <= 20:
            raise ValueError("pulse threshold must be finite and in (0, 20] standard deviations")
        with self._lock:
            self._pulse_threshold_std = float(value)

    def set_pulse_window_sec(self, value: float) -> None:
        if not math.isfinite(value) or not 1 <= value <= 120:
            raise ValueError("pulse baseline window must be finite and between 1 and 120 seconds")
        with self._lock:
            self._pulse_window_sec = float(value)

    def set_cortex_channel(self, channel: str) -> tuple[str, str]:
        """Assign Cortex/CA3 to physical AI0/AI1 before recording begins."""
        if channel not in {"ai0", "ai1"}:
            raise ValueError("Cortex channel must be ai0 or ai1")
        with self._lock:
            if self._recording:
                raise ValueError("stop recording before changing the Cortex/CA3 channel assignment")
            self._ctx_index = 0 if channel == "ai0" else 1
            self._electrode_labels = ("Cortex", "CA3") if self._ctx_index == 0 else ("CA3", "Cortex")
            return self._electrode_labels

    def fault(self, message: str) -> None:
        with self._lock:
            self._fault = message
            self._mode = "control"
            self._acquiring = False

    def clear_fault(self) -> None:
        with self._lock:
            self._fault = None


@dataclass
class Telemetry:
    lock: threading.Lock = field(default_factory=threading.Lock)
    blocks: int = 0
    sample_index: int = 0
    ai_read_ms: float = 0.0
    esn_ms: float = 0.0
    ao_write_ms: float = 0.0
    block_ms: float = 0.0
    max_block_ms: float = 0.0
    late_blocks: int = 0
    safety_trips: int = 0
    logger_backlog: int = 0
    logger_high_water: int = 0
    last_heartbeat_ns: int = field(default_factory=time.perf_counter_ns)

    def update_block(self, *, sample_index: int, ai_ms: float, esn_ms: float, ao_ms: float, block_ms: float,
                     deadline_ms: float, logger_backlog: int) -> None:
        with self.lock:
            self.blocks += 1
            self.sample_index = sample_index
            self.ai_read_ms = ai_ms
            self.esn_ms = esn_ms
            self.ao_write_ms = ao_ms
            self.block_ms = block_ms
            self.max_block_ms = max(self.max_block_ms, block_ms)
            self.late_blocks += int(block_ms > deadline_ms)
            self.logger_backlog = logger_backlog
            self.logger_high_water = max(self.logger_high_water, logger_backlog)
            self.last_heartbeat_ns = time.perf_counter_ns()

    def trip(self) -> None:
        with self.lock:
            self.safety_trips += 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "blocks": self.blocks,
                "sample_index": self.sample_index,
                "ai_read_ms": self.ai_read_ms,
                "esn_ms": self.esn_ms,
                "ao_write_ms": self.ao_write_ms,
                "block_ms": self.block_ms,
                "max_block_ms": self.max_block_ms,
                "late_blocks": self.late_blocks,
                "safety_trips": self.safety_trips,
                "logger_backlog": self.logger_backlog,
                "logger_high_water": self.logger_high_water,
                "last_heartbeat_ns": self.last_heartbeat_ns,
            }


class EsnRuntime:
    """The only boundary that calls ESN code."""

    def __init__(self, artifact: Path, config: AppConfig) -> None:
        self.streamer: Any = None
        self.error: Optional[str] = None
        try:
            from esn_bridge import EchoChamberEsnBridge

            self.streamer = EchoChamberEsnBridge.load(
                artifact,
                runtime_chunk_size=config.chunk_size,
                sample_rate=config.sample_rate,
                passthrough_dc_block_hz=config.passthrough_dc_block_hz,
                pulse_threshold_std=config.pulse_threshold_std,
                pulse_window_sec=config.pulse_window_sec,
                ao_command_gain_v_per_esn_unit=config.ao_command_gain_v_per_esn_unit,
            )
            expected = self.streamer.preferred_chunk_size
            if expected != config.chunk_size:
                LOG.info(
                    "ESN bridge adapting artifact chunk_size=%d to runtime chunk_size=%d",
                    expected,
                    config.chunk_size,
                )
            # Validate and warm the live instance before acquisition/watchdog
            # startup.  ReservoirPy's first call can perform several seconds of
            # lazy initialization; warming only a disposable probe leaves that
            # delay in the real-time path.
            output = np.asarray(
                self.streamer.process(np.zeros((2, config.chunk_size)), ctx_index=config.ctx_index)
            )
            if output.shape != (1, config.chunk_size) or not np.all(np.isfinite(output)):
                raise ValueError(f"ESN self-test returned invalid output {output.shape}")
            self.streamer.reset()
            # Measure the live artifact, rather than guessing from its model
            # size.  This is run only at startup and state is reset afterward.
            probe = np.zeros((2, config.chunk_size), dtype=np.float64)
            timings_ms: list[float] = []
            for _ in range(20):
                started = time.perf_counter_ns()
                self.streamer.process(probe, ctx_index=config.ctx_index)
                timings_ms.append((time.perf_counter_ns() - started) / 1e6)
            self.streamer.reset()
            deadline_ms = 1_000 * config.chunk_size / config.sample_rate
            LOG.info(
                "ESN timing benchmark: median=%.3f ms, p95=%.3f ms, block deadline=%.3f ms",
                float(np.median(timings_ms)), float(np.percentile(timings_ms, 95)), deadline_ms,
            )
            LOG.info("Pulse threshold configured at %.3f SD above a %.1f s rolling baseline", config.pulse_threshold_std, config.pulse_window_sec)
            if float(np.percentile(timings_ms, 95)) >= deadline_ms:
                LOG.warning(
                    "ESN timing benchmark exceeds the %.3f ms block deadline; "
                    "closed-loop output is not sustainable at this chunk size", deadline_ms,
                )
            self.streamer.configure(stim_mode="passthrough", stim_gain=1.0)
            LOG.info("ESN artifact loaded and passed startup self-test")
        except Exception as exc:
            self.streamer = None
            self.error = f"{type(exc).__name__}: {exc}"
            LOG.error("ESN unavailable: %s", self.error)

    @property
    def ready(self) -> bool:
        return self.streamer is not None

    def configure(self, stim_mode: str, gain: float, pulse_threshold_std: float | None = None,
                  pulse_window_sec: float | None = None) -> None:
        if not self.streamer:
            raise RuntimeError(self.error or "ESN unavailable")
        self.streamer.configure(
            stim_mode=stim_mode,
            stim_gain=gain,
            pulse_threshold_std=pulse_threshold_std,
            pulse_window_sec=pulse_window_sec,
        )

    def reset(self) -> None:
        if self.streamer:
            self.streamer.reset()

    def process(self, data: np.ndarray, ctx_index: int) -> np.ndarray:
        if not self.streamer:
            raise RuntimeError(self.error or "ESN unavailable")
        output = np.asarray(self.streamer.process(data, ctx_index=ctx_index), dtype=np.float64)
        if output.shape != (1, data.shape[1]):
            raise ValueError(f"ESN returned {output.shape}, expected {(1, data.shape[1])}")
        return output

    def diagnostics(self, samples: int) -> tuple[np.ndarray, float, float, bool]:
        """Return pre-stimulation model output and pulse-decision metrics."""
        if not self.streamer:
            return np.zeros((1, samples)), float("nan"), float("nan"), False
        return self.streamer.diagnostics(samples)


class StimulusSafetyAdapter:
    def __init__(self, config: SafetyConfig, sample_rate: int) -> None:
        self.config = config
        self.sample_rate = sample_rate
        self.previous_v = 0.0
        self.area_history: deque[float] = deque(maxlen=max(1, int(config.area_window_s * sample_rate)))
        self.active_history: deque[int] = deque(maxlen=max(1, int(config.area_window_s * sample_rate)))
        self.consecutive_active = 0
        self.last_reason: Optional[str] = None

    def reset(self) -> None:
        self.previous_v = 0.0
        self.area_history.clear()
        self.active_history.clear()
        self.consecutive_active = 0
        self.last_reason = None

    def process(self, raw: np.ndarray) -> tuple[np.ndarray, Optional[str]]:
        signal_v = np.asarray(raw, dtype=np.float64).reshape(-1)
        if self.config.isolator_command_v_per_output_unit:
            signal_v = signal_v * self.config.isolator_command_v_per_output_unit
        if not np.all(np.isfinite(signal_v)):
            return self._trip(signal_v.size, "non-finite ESN output")

        signal_v = np.clip(signal_v, -self.config.max_command_v, self.config.max_command_v)
        max_step = self.config.max_slew_v_per_s / self.sample_rate
        safe = np.empty_like(signal_v)
        previous = self.previous_v
        for index, value in enumerate(signal_v):
            previous += float(np.clip(value - previous, -max_step, max_step))
            safe[index] = previous
        self.previous_v = float(safe[-1]) if safe.size else self.previous_v

        dt_s = 1.0 / self.sample_rate
        for value in safe:
            self.area_history.append(float(value) * dt_s)
            active = int(abs(float(value)) >= self.config.active_threshold_v)
            self.active_history.append(active)
            self.consecutive_active = self.consecutive_active + 1 if active else 0

        if not self.config.allow_sustained_output_for_dry_test:
            if abs(sum(self.area_history)) > self.config.max_abs_area_v_s:
                return self._trip(safe.size, "net command area limit exceeded")
            if len(self.active_history) == self.active_history.maxlen and (
                sum(self.active_history) / len(self.active_history) > self.config.max_active_fraction
            ):
                return self._trip(safe.size, "stimulation duty-cycle limit exceeded")
            if self.consecutive_active > int(self.config.max_consecutive_active_s * self.sample_rate):
                return self._trip(safe.size, "continuous stimulation limit exceeded")

        self.last_reason = None
        return safe.reshape(1, -1), None

    def _trip(self, count: int, reason: str) -> tuple[np.ndarray, str]:
        self.last_reason = reason
        self.reset()
        self.last_reason = reason
        return np.zeros((1, count), dtype=np.float64), reason


@dataclass(frozen=True)
class RecordBlock:
    sample_index: int
    ai: np.ndarray
    calibrated_lfp: np.ndarray
    raw_esn: np.ndarray
    model_esn: np.ndarray
    pulse_threshold: np.ndarray
    pulse_peak: np.ndarray
    pulse_fired: np.ndarray
    safe_ao: np.ndarray
    actual_stim: Optional[np.ndarray]
    mode_value: float


class H5Recorder:
    """Asynchronous, batched HDF5 recorder owned by one writer thread."""

    def __init__(self, config: AppConfig, metadata: dict[str, Any]) -> None:
        self.config = config
        self.metadata = metadata
        self.items: queue.Queue[Optional[RecordBlock]] = queue.Queue(maxsize=config.logger_queue_blocks)
        self._lock = threading.RLock()
        self._file: Optional[h5py.File] = None
        self._ai_dataset: Optional[h5py.Dataset] = None
        self._ao_dataset: Optional[h5py.Dataset] = None
        self._actual_dataset: Optional[h5py.Dataset] = None
        self._diagnostics_dataset: Optional[h5py.Dataset] = None
        self._mode_dataset: Optional[h5py.Dataset] = None
        self._pulse_dataset: Optional[h5py.Dataset] = None
        self._gap_dataset: Optional[h5py.Dataset] = None
        self._data_path: Optional[Path] = None
        self._accepting = False
        self._error: Optional[str] = None
        self._committed_samples = 0
        self._last_source_end: Optional[int] = None
        self._last_mode: Optional[int] = None
        self._last_flush = time.monotonic()
        self._batch_samples = max(
            config.chunk_size,
            int(math.ceil(config.recorder_batch_s * config.sample_rate / config.chunk_size))
            * config.chunk_size,
        )
        self._thread = threading.Thread(target=self._writer_loop, name="hdf5-recorder", daemon=True)
        self._thread.start()

    @property
    def backlog(self) -> int:
        return self.items.qsize()

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    @staticmethod
    def safe_custom_name(value: Optional[str]) -> str:
        if value is None or not value.strip():
            return "echo"
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_-")
        cleaned = re.sub(r"_+", "_", cleaned)[:80].rstrip("_-")
        if not cleaned:
            raise ValueError("recording name must contain at least one letter or number")
        return cleaned

    def start(self, custom_name: Optional[str] = None) -> Path:
        with self._lock:
            if self._accepting or self._file:
                raise RuntimeError("recording is already active")
            self.config.record_dir.mkdir(parents=True, exist_ok=True)
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = self.safe_custom_name(custom_name)
            self._data_path = self.config.record_dir / f"{stamp}_{suffix}.h5"
            if self._data_path.exists():
                try:
                    with h5py.File(self._data_path, "r") as existing:
                        is_echo_chamber = existing.attrs.get("schema") == "echo-chamber-recording"
                        committed = int(existing.attrs.get("committed_samples", self.config.sample_rate))
                except (OSError, ValueError, TypeError) as exc:
                    raise FileExistsError(
                        f"refusing to replace unrecognized recording: {self._data_path.name}"
                    ) from exc
                if not is_echo_chamber or committed >= self.config.sample_rate:
                    raise FileExistsError(
                        f"refusing to replace recording with at least one second of data: "
                        f"{self._data_path.name}"
                    )
                LOG.warning(
                    "Replacing premature same-second recording %s (%d committed samples)",
                    self._data_path.name, committed,
                )
                self._data_path.unlink()
            self._file = h5py.File(self._data_path, "x", libver="latest")
            meta = dict(self.metadata)
            meta.update({
                "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "data_file": self._data_path.name,
                "format": "Echo Chamber HDF5 v3; compact typed signals and sparse events",
                "format_tag": H5_FORMAT_TAG,
                "ao_pipeline_delay_samples": self.config.ao_lead_samples,
                "ao_pipeline_delay_seconds": self.config.ao_lead_samples / self.config.sample_rate,
                "lfp_scaling": {
                    "formula": "electrode_mV = ai_raw_V * 1000 / amplifier_gain",
                    "amplifier_gain": list(self.config.amplifier_gain),
                    "electrode_labels": list(self.config.electrode_labels),
                },
            })
            string_type = h5py.string_dtype(encoding="utf-8")
            signals = self._file.create_group("signals")
            events = self._file.create_group("events")
            diagnostics = self._file.create_group("diagnostics")
            compression = {"compression": "gzip", "compression_opts": 1, "shuffle": True}
            self._ai_dataset = signals.create_dataset(
                "ai_raw_V", shape=(2, 0), maxshape=(2, None),
                chunks=(2, self._batch_samples), dtype=np.float32, **compression,
            )
            self._ai_dataset.attrs["channel_labels"] = np.asarray(
                self.config.electrode_labels, dtype=string_type
            )
            self._ai_dataset.attrs["unit"] = "V"
            self._ao_dataset = signals.create_dataset(
                "ao_command_V", shape=(1, 0), maxshape=(1, None),
                chunks=(1, self._batch_samples), dtype=np.float32, **compression,
            )
            self._ao_dataset.attrs["unit"] = "V"
            if self.config.actual_stim_monitor_channel:
                self._actual_dataset = signals.create_dataset(
                    "actual_stim_monitor_V", shape=(1, 0), maxshape=(1, None),
                    chunks=(1, self._batch_samples), dtype=np.float32, **compression,
                )
                self._actual_dataset.attrs["unit"] = "V"
            diagnostic_dtype = np.dtype([
                ("sample_offset", "<u8"), ("ai_sample_index", "<u8"), ("sample_count", "<u4"),
                ("raw_esn_mean_V", "<f4"), ("raw_esn_min_V", "<f4"),
                ("raw_esn_max_V", "<f4"), ("model_esn_mean_mV", "<f4"),
                ("pulse_threshold_mV", "<f4"), ("pulse_peak_mV", "<f4"),
            ])
            self._diagnostics_dataset = diagnostics.create_dataset(
                "blocks", shape=(0,), maxshape=(None,), chunks=(max(1, self._batch_samples // self.config.chunk_size),),
                dtype=diagnostic_dtype, **compression,
            )
            self._mode_dataset = events.create_dataset(
                "mode_changes", shape=(0,), maxshape=(None,), chunks=(128,),
                dtype=np.dtype([("sample_offset", "<u8"), ("mode", "u1")]), **compression,
            )
            self._pulse_dataset = events.create_dataset(
                "pulses", shape=(0,), maxshape=(None,), chunks=(128,),
                dtype=np.dtype([
                    ("sample_offset", "<u8"), ("ai_sample_index", "<u8"),
                    ("threshold_mV", "<f4"), ("peak_mV", "<f4"),
                ]), **compression,
            )
            self._gap_dataset = events.create_dataset(
                "sample_discontinuities", shape=(0,), maxshape=(None,), chunks=(64,),
                dtype=np.dtype([
                    ("sample_offset", "<u8"), ("expected_ai_sample_index", "<u8"),
                    ("actual_ai_sample_index", "<u8"),
                ]), **compression,
            )
            self._file.attrs["schema"] = "echo-chamber-recording"
            self._file.attrs["schema_version"] = 3
            self._file.attrs["format_tag"] = H5_FORMAT_TAG
            self._file.attrs["metadata_json"] = json.dumps(meta, default=str)
            self._file.attrs["sample_rate_hz"] = self.config.sample_rate
            self._file.attrs["first_ai_sample_index"] = np.uint64(0)
            self._file.attrs["committed_samples"] = 0
            # uint8 remains readable across older and newer HDF5/h5py versions.
            self._file.attrs["complete"] = np.uint8(0)
            self._file.flush()
            self._error = None
            self._committed_samples = 0
            self._last_source_end = None
            self._last_mode = None
            self._last_flush = time.monotonic()
            self._accepting = True
            return self._data_path

    def set_electrode_labels(self, labels: tuple[str, str]) -> None:
        """Apply a UI assignment to the next recording's metadata."""
        with self._lock:
            if self._accepting or self._file:
                raise RuntimeError("cannot change recording labels while recording")
            self.metadata["configuration"] = dict(self.metadata["configuration"], electrode_labels=list(labels))
            self.config = dataclasses.replace(self.config, electrode_labels=labels)

    def submit(self, block: RecordBlock) -> None:
        with self._lock:
            accepting = self._accepting
        if not accepting:
            return
        try:
            self.items.put_nowait(block)
        except queue.Full as exc:
            raise RuntimeError("recording queue overflow; raw data integrity cannot be guaranteed") from exc

    def stop_recording(self, timeout: float = 10.0) -> None:
        with self._lock:
            if not self._file:
                self._accepting = False
                return
            self._accepting = False
        self.items.join()
        with self._lock:
            if self._file:
                self._file.attrs["committed_samples"] = self._committed_samples
                self._file.attrs["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
                self._file.attrs.modify("complete", np.uint8(self._error is None))
                self._file.flush()
                with contextlib.suppress(Exception):
                    os.fsync(self._file.id.get_vfd_handle())
                self._file.close()
                self._file = None
                self._ai_dataset = self._ao_dataset = self._actual_dataset = None
                self._diagnostics_dataset = self._mode_dataset = None
                self._pulse_dataset = self._gap_dataset = None
        if self._error:
            raise RuntimeError(self._error)

    def close(self) -> None:
        self.stop_recording()
        self.items.put(None)
        self._thread.join(timeout=5.0)

    def _writer_loop(self) -> None:
        while True:
            first = self.items.get()
            if first is None:
                self.items.task_done()
                return
            batch = [first]
            sample_count = first.ai.shape[1]
            deadline = time.monotonic() + self.config.recorder_batch_s
            closing = False
            while sample_count < self._batch_samples:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self.items.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    closing = True
                    break
                batch.append(item)
                sample_count += item.ai.shape[1]
            try:
                with self._lock:
                    if self._file is None or self._ai_dataset is None or self._ao_dataset is None:
                        raise RuntimeError("recording file closed before queued blocks drained")
                    start = self._committed_samples
                    ai = np.hstack([item.ai for item in batch]).astype(np.float32, copy=False)
                    ao = np.hstack([item.safe_ao for item in batch]).astype(np.float32, copy=False)
                    stop = start + ai.shape[1]
                    self._ai_dataset.resize(stop, axis=1)
                    self._ao_dataset.resize(stop, axis=1)
                    self._ai_dataset[:, start:stop] = ai
                    self._ao_dataset[:, start:stop] = ao
                    if self._actual_dataset is not None:
                        actual = np.hstack([
                            item.actual_stim if item.actual_stim is not None
                            else np.full((1, item.ai.shape[1]), np.nan)
                            for item in batch
                        ]).astype(np.float32, copy=False)
                        self._actual_dataset.resize(stop, axis=1)
                        self._actual_dataset[:, start:stop] = actual
                    offset = start
                    for item in batch:
                        self._append_block_records(item, offset)
                        offset += item.ai.shape[1]
                    self._committed_samples = stop
                    self._file.attrs.modify("committed_samples", stop)
                    if time.monotonic() - self._last_flush >= self.config.recorder_flush_s:
                        self._file.flush()
                        self._last_flush = time.monotonic()
            except Exception as exc:
                with self._lock:
                    self._error = f"recording writer failed: {type(exc).__name__}: {exc}"
                    self._accepting = False
                LOG.exception("Recording writer failed")
            finally:
                for _ in batch:
                    self.items.task_done()
                if closing:
                    self.items.task_done()
            if closing:
                return

    @staticmethod
    def _finite_mean(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else math.nan

    @staticmethod
    def _finite_min(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.min(finite)) if finite.size else math.nan

    @staticmethod
    def _finite_max(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.max(finite)) if finite.size else math.nan

    @staticmethod
    def _append_records(dataset: h5py.Dataset, records: np.ndarray) -> None:
        if not records.size:
            return
        start = dataset.shape[0]
        dataset.resize(start + records.shape[0], axis=0)
        dataset[start:] = records

    def _append_block_records(self, item: RecordBlock, offset: int) -> None:
        assert self._file is not None
        count = item.ai.shape[1]
        if self._last_source_end is None:
            self._file.attrs.modify("first_ai_sample_index", np.uint64(item.sample_index))
        elif item.sample_index != self._last_source_end:
            record = np.asarray(
                [(offset, self._last_source_end, item.sample_index)], dtype=self._gap_dataset.dtype
            )
            self._append_records(self._gap_dataset, record)
        self._last_source_end = item.sample_index + count

        mode = int(item.mode_value)
        if mode != self._last_mode:
            self._append_records(
                self._mode_dataset, np.asarray([(offset, mode)], dtype=self._mode_dataset.dtype)
            )
            self._last_mode = mode

        diagnostic = np.asarray([(
            offset, item.sample_index, count,
            self._finite_mean(item.raw_esn), self._finite_min(item.raw_esn),
            self._finite_max(item.raw_esn), self._finite_mean(item.model_esn),
            self._finite_mean(item.pulse_threshold), self._finite_mean(item.pulse_peak),
        )], dtype=self._diagnostics_dataset.dtype)
        self._append_records(self._diagnostics_dataset, diagnostic)

        if np.any(item.pulse_fired > 0.5):
            pulse = np.asarray([(
                offset, item.sample_index,
                self._finite_mean(item.pulse_threshold), self._finite_mean(item.pulse_peak),
            )], dtype=self._pulse_dataset.dtype)
            self._append_records(self._pulse_dataset, pulse)


class UiHub:
    def __init__(self, queue_size: int) -> None:
        self.queue_size = queue_size
        self._clients: set[asyncio.Queue[str]] = set()

    def subscribe(self) -> asyncio.Queue[str]:
        client: asyncio.Queue[str] = asyncio.Queue(maxsize=self.queue_size)
        self._clients.add(client)
        return client

    def unsubscribe(self, client: asyncio.Queue[str]) -> None:
        self._clients.discard(client)

    def publish(self, packet: str) -> None:
        for client in tuple(self._clients):
            if client.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    client.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                client.put_nowait(packet)


def ui_series(values: np.ndarray, downsample: int) -> list[list[float | None]]:
    """JSON-safe display data: non-finite values become browser nulls."""
    display = np.asarray(values, dtype=np.float64)[:, ::downsample]
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in display
    ]


class ProcessingCore:
    def __init__(self, config: AppConfig, state: RuntimeState, esn: EsnRuntime, recorder: H5Recorder,
                 hub: UiHub, event_loop: asyncio.AbstractEventLoop, telemetry: Telemetry) -> None:
        self.config = config
        self.state = state
        self.esn = esn
        self.recorder = recorder
        self.hub = hub
        self.event_loop = event_loop
        self.telemetry = telemetry
        self.safety = StimulusSafetyAdapter(config.safety, config.sample_rate)
        self.sample_index = 0
        self.ui_ai: list[np.ndarray] = []
        self.ui_ao: list[np.ndarray] = []
        self.ui_preview: list[np.ndarray] = []
        self.ui_threshold: list[np.ndarray] = []
        self.last_ui_ns = time.perf_counter_ns()
        self._last_esn_config: Optional[tuple[str, float, float]] = None
        # ESN/filter state belongs exclusively to the DAQ thread.  UI commands
        # only change RuntimeState; transitions are applied here at a block
        # boundary so reset() can never race process_chunk().
        self._active_mode = "control"

    def process(self, all_ai: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, Optional[str]]:
        snapshot = self.state.snapshot()
        lfp = np.asarray(all_ai[:2], dtype=np.float64)
        electrode_lfp_mv = lfp * (
            1_000.0 / np.asarray(self.config.amplifier_gain, dtype=np.float64)
        ).reshape(2, 1)
        actual_stim = np.asarray(all_ai[2:3], dtype=np.float64) if all_ai.shape[0] > 2 else None
        raw = np.zeros((1, self.config.chunk_size), dtype=np.float64)
        model_esn = np.full((1, self.config.chunk_size), np.nan)
        pulse_threshold = np.full((1, self.config.chunk_size), np.nan)
        pulse_peak = np.full((1, self.config.chunk_size), np.nan)
        pulse_fired = np.zeros((1, self.config.chunk_size), dtype=np.float64)
        esn_ms = 0.0
        safety_reason: Optional[str] = None

        if snapshot.mode != self._active_mode:
            self.esn.reset()
            self.safety.reset()
            self._last_esn_config = None
            self._active_mode = snapshot.mode

        if snapshot.mode == "closed-loop":
            desired_config = (snapshot.stim_mode, snapshot.stim_gain, snapshot.pulse_threshold_std, snapshot.pulse_window_sec)
            if desired_config != self._last_esn_config:
                self.esn.configure(*desired_config)
                self._last_esn_config = desired_config
            started = time.perf_counter_ns()
            raw = self.esn.process(electrode_lfp_mv, snapshot.ctx_index)
            esn_ms = (time.perf_counter_ns() - started) / 1e6
            model_esn, threshold, peak, fired = self.esn.diagnostics(self.config.chunk_size)
            pulse_threshold.fill(threshold)
            pulse_peak.fill(peak)
            pulse_fired.fill(float(fired))
            safe, safety_reason = self.safety.process(raw)
            if safety_reason:
                self.telemetry.trip()
                LOG.error("Stimulation safety trip: %s", safety_reason)
                self.state.fault(f"stimulation safety trip: {safety_reason}")
        else:
            safe = np.zeros_like(raw)
            self.safety.reset()

        if snapshot.recording:
            self.recorder.submit(RecordBlock(
                self.sample_index, lfp.copy(), electrode_lfp_mv.copy(), raw.copy(), model_esn.copy(),
                pulse_threshold.copy(), pulse_peak.copy(), pulse_fired.copy(), safe.copy(),
                actual_stim.copy() if actual_stim is not None else None,
                RECORDED_MODE_VALUES[(snapshot.mode, snapshot.stim_mode)],
            ))

        # Display-only pulse context: centre the model output for legibility,
        # then show its adaptive threshold relative to that same baseline.
        # This does not affect thresholding, safety, or AO output.
        if snapshot.stim_mode == "threshold_pulse" and np.all(np.isfinite(model_esn)):
            baseline = float(np.mean(model_esn))
            preview = ((model_esn - baseline) * snapshot.stim_gain
                       * self.config.ao_command_gain_v_per_esn_unit)
            threshold_preview = np.full_like(
                preview,
                (threshold - baseline) * snapshot.stim_gain
                * self.config.ao_command_gain_v_per_esn_unit,
            )
        else:
            preview = np.full_like(safe, np.nan)
            threshold_preview = np.full_like(safe, np.nan)
        self._publish_ui(electrode_lfp_mv, safe, preview, threshold_preview, snapshot, safety_reason)
        self.sample_index += self.config.chunk_size
        return raw, safe, esn_ms, safety_reason

    def _publish_ui(self, ai: np.ndarray, ao: np.ndarray, preview: np.ndarray, threshold: np.ndarray,
                    snapshot: StateSnapshot,
                    safety_reason: Optional[str]) -> None:
        raise NotImplementedError("Echo Chamber requires the off-thread UI processing core")


@dataclass(frozen=True)
class UiJob:
    """A best-effort display update, independent of real-time acquisition."""

    ai: np.ndarray
    ao: np.ndarray
    preview: np.ndarray
    threshold: np.ndarray
    snapshot: StateSnapshot
    safety_reason: Optional[str]


class AsyncUiProcessingCore(ProcessingCore):
    """Moves UI downsampling and JSON serialization off the DAQ thread."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ui_jobs: queue.Queue[Optional[UiJob]] = queue.Queue(maxsize=self.config.ui_queue_packets)
        self._ui_stopping = threading.Event()
        self._ui_thread = threading.Thread(target=self._ui_worker, name="ui-serializer", daemon=True)
        self._ui_thread.start()

    def close_ui(self) -> None:
        self._ui_stopping.set()
        with contextlib.suppress(queue.Empty):
            if self._ui_jobs.full():
                self._ui_jobs.get_nowait()
        with contextlib.suppress(queue.Full):
            self._ui_jobs.put_nowait(None)
        self._ui_thread.join(timeout=3.0)

    def _publish_ui(self, ai: np.ndarray, ao: np.ndarray, preview: np.ndarray, threshold: np.ndarray,
                    snapshot: StateSnapshot,
                    safety_reason: Optional[str]) -> None:
        # The DAQ thread copies data and queues a bounded latest-only job. It
        # never converts arrays to lists or JSON, and stale display data is
        # deliberately discarded rather than delaying hardware timing.
        self.ui_ai.append(ai.copy())
        self.ui_ao.append(ao.copy())
        self.ui_preview.append(preview.copy())
        self.ui_threshold.append(threshold.copy())
        now = time.perf_counter_ns()
        if (now - self.last_ui_ns) / 1e9 < self.config.ui_interval_s:
            return
        job = UiJob(
            np.hstack(self.ui_ai), np.hstack(self.ui_ao), np.hstack(self.ui_preview),
            np.hstack(self.ui_threshold), snapshot, safety_reason,
        )
        self.ui_ai.clear()
        self.ui_ao.clear()
        self.ui_preview.clear()
        self.ui_threshold.clear()
        self.last_ui_ns = now
        if self._ui_jobs.full():
            with contextlib.suppress(queue.Empty):
                self._ui_jobs.get_nowait()
        with contextlib.suppress(queue.Full):
            self._ui_jobs.put_nowait(job)

    def _ui_worker(self) -> None:
        while not self._ui_stopping.is_set():
            try:
                job = self._ui_jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                return
            try:
                snapshot = job.snapshot
                packet = json.dumps({
                    "ai": job.ai[:, ::self.config.visual_downsample].tolist(),
                    "ao": job.ao[:, ::self.config.visual_downsample].tolist(),
                    "passthrough_preview": ui_series(job.preview, self.config.visual_downsample),
                    "pulse_threshold_preview": ui_series(job.threshold, self.config.visual_downsample),
                    "mode": snapshot.mode,
                    "is_recording": snapshot.recording,
                    "is_acquiring": snapshot.acquiring,
                    "stim_mode": snapshot.stim_mode,
                    "stim_gain": snapshot.stim_gain,
                    "pulse_threshold_std": snapshot.pulse_threshold_std,
                    "pulse_window_sec": snapshot.pulse_window_sec,
                    "fs": self.config.sample_rate / self.config.visual_downsample,
                    "fault": snapshot.fault,
                    "esn_ready": snapshot.esn_ready,
                    "safety_trip": job.safety_reason,
                    "telemetry": self.telemetry.snapshot(),
                    "channels": list(snapshot.electrode_labels),
                    "cortex_ai": f"ai{snapshot.ctx_index}",
                    "ao_is_command": True,
                    "ai_unit": "mV",
                    "ao_unit": "V",
                    "amplifier_gain": list(self.config.amplifier_gain),
                    "ao_command_gain_v_per_esn_unit": self.config.ao_command_gain_v_per_esn_unit,
                })
                self.event_loop.call_soon_threadsafe(self.hub.publish, packet)
            except Exception:
                LOG.exception("UI serialization failed")


class BaseDaq:
    def __init__(self, config: AppConfig, state: RuntimeState, core: ProcessingCore, telemetry: Telemetry) -> None:
        self.config = config
        self.state = state
        self.core = core
        self.telemetry = telemetry
        self.last_command = np.zeros((1, config.chunk_size), dtype=np.float64)

    def run(self) -> None:
        raise NotImplementedError

    def request_safe_zero(self) -> None:
        self.last_command.fill(0.0)


@dataclass
class AoScheduleStats:
    submitted_blocks: int = 0
    zero_substitutions: int = 0
    late_commands: int = 0
    min_queued_samples: Optional[int] = None
    max_write_ms: float = 0.0
    max_loop_pause_ms: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def submitted(self, late: bool) -> None:
        with self.lock:
            self.submitted_blocks += 1
            self.late_commands += int(late)

    def refill(self, queued: int, zeros: int, write_ms: float, pause_ms: float) -> None:
        with self.lock:
            self.zero_substitutions += zeros
            self.min_queued_samples = queued if self.min_queued_samples is None else min(self.min_queued_samples, queued)
            self.max_write_ms = max(self.max_write_ms, write_ms)
            self.max_loop_pause_ms = max(self.max_loop_pause_ms, pause_ms)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "submitted_blocks": self.submitted_blocks,
                "zero_substitutions": self.zero_substitutions,
                "late_commands": self.late_commands,
                "min_queued_samples": self.min_queued_samples,
                "max_write_ms": self.max_write_ms,
                "max_loop_pause_ms": self.max_loop_pause_ms,
            }


class ScheduledAo:
    """Sole owner of a non-regenerating AO task with proactive safe refills."""

    def __init__(self, config: AppConfig, state: RuntimeState, trigger_terminal: str) -> None:
        self.config = config
        self.state = state
        self.trigger_terminal = trigger_terminal
        self.commands: dict[int, np.ndarray] = {}
        self.command_lock = threading.Lock()
        self.ready = threading.Event()
        self.clock_started = threading.Event()
        self.stop_requested = threading.Event()
        self.force_safe = threading.Event()
        self.error: Optional[Exception] = None
        self.stats = AoScheduleStats()
        self._next_block = 0
        self.thread = threading.Thread(target=self._run, name="ao-writer", daemon=True)

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(10.0):
            raise RuntimeError("AO scheduler did not become ready")
        if self.error:
            raise RuntimeError(f"AO scheduler setup failed: {self.error}") from self.error

    def notify_clock_started(self) -> None:
        self.clock_started.set()

    def submit(self, target_block: int, command: np.ndarray) -> None:
        value = np.asarray(command, dtype=np.float64).reshape(-1)
        if value.size != self.config.chunk_size:
            raise ValueError(f"AO command has {value.size} samples; expected {self.config.chunk_size}")
        with self.command_lock:
            late = target_block < self._next_block
            self.stats.submitted(late)
            if not late and not self.force_safe.is_set():
                self.commands[target_block] = value.copy()

    def request_safe_zero(self) -> None:
        self.force_safe.set()
        with self.command_lock:
            self.commands.clear()

    def stop(self) -> None:
        self.request_safe_zero()
        self.stop_requested.set()
        self.clock_started.set()
        self.thread.join(5.0)
        self._force_zero()

    def _take(self, target: int) -> Optional[np.ndarray]:
        if self.force_safe.is_set():
            return None
        with self.command_lock:
            return self.commands.pop(target, None)

    def _take_contiguous(self, first_target: int, maximum: int) -> list[np.ndarray]:
        """Remove every immediately available command, up to ``maximum``."""
        if self.force_safe.is_set():
            return []
        with self.command_lock:
            blocks: list[np.ndarray] = []
            for target in range(first_target, first_target + maximum):
                block = self.commands.get(target)
                if block is None:
                    break
                blocks.append(self.commands.pop(target))
            return blocks

    def _discard_committed_commands(self) -> None:
        """Remove commands whose output positions were already committed."""
        with self.command_lock:
            stale = [target for target in self.commands if target < self._next_block]
            for target in stale:
                self.commands.pop(target, None)

    def _run(self) -> None:
        chunk = self.config.chunk_size
        lead_samples = self.config.ao_lead_samples
        deadline_samples = self.config.ao_deadline_samples
        last_loop_ns = time.perf_counter_ns()
        try:
            with nidaqmx.Task("echo-chamber-ao") as task:
                task.ao_channels.add_ao_voltage_chan(
                    self.config.physical_ao_channel,
                    min_val=-self.config.safety.max_command_v,
                    max_val=self.config.safety.max_command_v,
                )
                task.timing.cfg_samp_clk_timing(
                    self.config.sample_rate,
                    source=f"/{self.config.device}/ai/SampleClock",
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=lead_samples * 2,
                )
                task.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION
                # ``OutStream`` exposes the output-buffer setting as a
                # property in nidaqmx-python; it has no cfg_output_buffer()
                # method.
                task.out_stream.output_buf_size = lead_samples * 2
                task.triggers.start_trigger.cfg_dig_edge_start_trig(self.trigger_terminal)
                writer = AnalogSingleChannelWriter(task.out_stream, auto_start=False)
                writer.write_many_sample(np.zeros(lead_samples, dtype=np.float64), timeout=5.0)
                self._next_block = self.config.ao_lead_chunks
                task.start()
                self.ready.set()
                self.clock_started.wait()

                while not self.stop_requested.is_set():
                    now_ns = time.perf_counter_ns()
                    pause_ms = (now_ns - last_loop_ns) / 1e6
                    last_loop_ns = now_ns
                    generated = int(task.out_stream.total_samp_per_chan_generated)
                    queued = self._next_block * chunk - generated
                    deficit_blocks = max(0, math.ceil((lead_samples - queued) / chunk))
                    if deficit_blocks and (queued <= deadline_samples or self.force_safe.is_set()):
                        first_target = self._next_block
                        refill = self._take_contiguous(first_target, deficit_blocks)
                        zero_count = deficit_blocks - len(refill)
                        if zero_count:
                            refill.extend(np.zeros(chunk, dtype=np.float64) for _ in range(zero_count))
                        snapshot = self.state.snapshot()
                        active_miss = (
                            zero_count > 0
                            and not self.force_safe.is_set()
                            and snapshot.mode == "closed-loop"
                            and snapshot.stim_mode != "off"
                        )

                        if refill:
                            if active_miss:
                                # Once an active command misses its deadline,
                                # the complete refill and every future command
                                # are forced safe before the fault is published.
                                self.request_safe_zero()
                                refill = [np.zeros_like(block) for block in refill]
                                zero_count = len(refill)
                            write_started = time.perf_counter_ns()
                            writer.write_many_sample(np.concatenate(refill), timeout=0.5)
                            write_ms = (time.perf_counter_ns() - write_started) / 1e6
                            self._next_block += len(refill)
                            self._discard_committed_commands()
                            self.stats.refill(queued, zero_count, write_ms, pause_ms)
                            if zero_count:
                                LOG.warning(
                                    "AO deadline substituted %d safe block(s), first target=%d, queued=%d samples",
                                    zero_count, first_target, queued,
                                )
                            if active_miss:
                                self.state.fault(
                                    "AO command missed its derived half-lead deadline; safe zeros were written"
                                )
                    time.sleep(min(0.001, chunk / self.config.sample_rate / 8))
        except Exception as exc:
            self.error = exc
            LOG.error("AO scheduler failed: %s\n%s", exc, traceback.format_exc())
            self.state.fault(f"AO scheduler failure: {type(exc).__name__}: {exc}")
        finally:
            self.ready.set()

    def _force_zero(self) -> None:
        try:
            with nidaqmx.Task("echo-chamber-ao-zero") as task:
                task.ao_channels.add_ao_voltage_chan(
                    self.config.physical_ao_channel,
                    min_val=-self.config.safety.max_command_v,
                    max_val=self.config.safety.max_command_v,
                )
                task.write(0.0, auto_start=True)
        except Exception:
            LOG.exception("Could not explicitly return AO to zero")


@dataclass(frozen=True)
class InputBatch:
    sample_index: int
    values: np.ndarray
    read_ms: float


class RealDaq(BaseDaq):
    """Independent AI reader, processor, and proactive AO scheduling pipeline."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.scheduler: Optional[ScheduledAo] = None

    def preflight(self) -> dict[str, Any]:
        if not HAS_NIDAQMX:
            raise RuntimeError("nidaqmx and the native NI-DAQmx driver are required")
        system = nidaqmx.system.System.local()
        names = [device.name for device in system.devices]
        if self.config.device not in names:
            raise RuntimeError(f"NI device {self.config.device!r} not found; available={names}")
        device = system.devices[self.config.device]
        available_ai = {channel.name for channel in device.ai_physical_chans}
        available_ao = {channel.name for channel in device.ao_physical_chans}
        missing_ai = set(self.config.physical_ai_channels) - available_ai
        if missing_ai:
            raise RuntimeError(f"missing AI channels: {sorted(missing_ai)}")
        if self.config.physical_ao_channel not in available_ao:
            raise RuntimeError(f"missing AO channel: {self.config.physical_ao_channel}")
        return {
            "device": device.name,
            "product_type": device.product_type,
            "serial_number": getattr(device, "serial_num", None),
            "ai": self.config.physical_ai_channels,
            "ao": self.config.physical_ao_channel,
        }

    def run(self) -> None:
        info = self.preflight()
        LOG.info("Hardware preflight passed: %s", info)
        terminal_name = {"DIFFERENTIAL": "DIFF", "PSEUDODIFFERENTIAL": "PSEUDO_DIFF"}.get(
            self.config.terminal_config.upper(), self.config.terminal_config.upper()
        )
        terminal = getattr(TerminalConfiguration, terminal_name, None)
        if terminal is None:
            raise ValueError(f"unknown terminal configuration: {self.config.terminal_config}")

        with nidaqmx.Task("echo-chamber-ai") as read_task:
            for channel in self.config.physical_ai_channels:
                read_task.ai_channels.add_ai_voltage_chan(
                    channel, terminal_config=terminal, min_val=self.config.ai_min_v, max_val=self.config.ai_max_v,
                )
            read_task.timing.cfg_samp_clk_timing(
                self.config.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.config.ao_lead_samples * 4,
            )
            reader = AnalogMultiChannelReader(read_task.in_stream)
            input_queue: queue.Queue[InputBatch] = queue.Queue(maxsize=self.config.ao_lead_chunks * 2)
            reader_stop = threading.Event()
            reader_errors: list[Exception] = []
            scheduler = ScheduledAo(self.config, self.state, read_task.triggers.start_trigger.term)
            self.scheduler = scheduler
            scheduler.start()
            read_task.start()
            scheduler.notify_clock_started()

            def ai_reader() -> None:
                sample_index = 0
                buffer = np.empty((len(self.config.physical_ai_channels), self.config.chunk_size), dtype=np.float64)
                try:
                    while not reader_stop.is_set() and self.state.snapshot().running:
                        started = time.perf_counter_ns()
                        reader.read_many_sample(
                            buffer,
                            number_of_samples_per_channel=self.config.chunk_size,
                            timeout=max(1.0, 4 * self.config.chunk_size / self.config.sample_rate),
                        )
                        batch = InputBatch(sample_index, buffer.copy(), (time.perf_counter_ns() - started) / 1e6)
                        sample_index += self.config.chunk_size
                        input_queue.put(batch, timeout=0.1)
                except Exception as exc:
                    if not reader_stop.is_set():
                        reader_errors.append(exc)
                        self.state.fault(f"AI reader failure: {type(exc).__name__}: {exc}")

            reader_thread = threading.Thread(target=ai_reader, name="ai-reader", daemon=True)
            reader_thread.start()
            LOG.info(
                "AO pipeline started: chunk=%d samples (%.1f ms), target lead=%d samples (%.1f ms), derived deadline=%.1f ms",
                self.config.chunk_size,
                1_000 * self.config.chunk_size / self.config.sample_rate,
                self.config.ao_lead_samples,
                self.config.ao_target_lead_ms,
                1_000 * self.config.ao_deadline_samples / self.config.sample_rate,
            )

            try:
                while self.state.snapshot().running:
                    block_started = time.perf_counter_ns()
                    try:
                        batch = input_queue.get(timeout=0.1)
                    except queue.Empty:
                        if reader_errors:
                            raise RuntimeError(f"AI reader failure: {reader_errors[0]}") from reader_errors[0]
                        continue
                    if self.state.snapshot().acquiring:
                        _, safe, esn_ms, _ = self.core.process(batch.values)
                    else:
                        safe = np.zeros((1, self.config.chunk_size), dtype=np.float64)
                        esn_ms = 0.0
                    if self.state.snapshot().mode != "closed-loop":
                        safe = np.zeros_like(safe)
                    target = batch.sample_index // self.config.chunk_size + self.config.ao_lead_chunks
                    scheduler.submit(target, safe)
                    self.last_command = safe
                    block_ms = (time.perf_counter_ns() - block_started) / 1e6
                    self.telemetry.update_block(
                        sample_index=self.core.sample_index, ai_ms=batch.read_ms, esn_ms=esn_ms, ao_ms=0.0,
                        block_ms=block_ms, deadline_ms=1_000 * self.config.chunk_size / self.config.sample_rate,
                        logger_backlog=self.core.recorder.backlog,
                    )
                    if scheduler.error:
                        raise RuntimeError(f"AO scheduler failure: {scheduler.error}") from scheduler.error
                    if self.core.recorder.error:
                        raise RuntimeError(self.core.recorder.error)
            finally:
                reader_stop.set()
                with contextlib.suppress(Exception):
                    read_task.stop()
                reader_thread.join(3.0)
                scheduler.stop()
                self.scheduler = None
                self.request_safe_zero()
                LOG.info("AO scheduler statistics: %s", scheduler.stats.snapshot())

    def request_safe_zero(self) -> None:
        super().request_safe_zero()
        if self.scheduler is not None:
            self.scheduler.request_safe_zero()


def load_mock_replay(path: Path, electrode_labels: tuple[str, str], sample_rate: int) -> np.ndarray:
    """Load an HDF5 recording and return named raw LFP rows."""
    if path.suffix not in {".h5", ".hdf5"}:
        raise ValueError("mock replay must be an Echo Chamber .h5 or .hdf5 recording")
    if path.suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as source:
            if source.attrs.get("format_tag") == H5_FORMAT_TAG:
                labels = tuple(
                    value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    for value in source["signals/ai_raw_V"].attrs["channel_labels"]
                )
                requested_indices = [labels.index(label) for label in electrode_labels]
                committed = min(
                    int(source.attrs.get("committed_samples", source["signals/ai_raw_V"].shape[1])),
                    source["signals/ai_raw_V"].shape[1],
                )
                replay = np.vstack([
                    np.asarray(source["signals/ai_raw_V"][index, :committed])
                    for index in requested_indices
                ])
                if replay.shape[1] == 0:
                    raise ValueError(f"mock replay contains no committed data: {path}")
                return replay
            rows = tuple(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in source["row_names"][:]
            )
            requested = tuple(f"{label}_raw_V" for label in electrode_labels)
            if not all(name in rows for name in requested):
                available = [name for name in rows if name.endswith("_raw_V")]
                raise ValueError(f"mock replay requires rows {requested}; available raw rows={available}")
            committed = min(
                int(source.attrs.get("committed_samples", source["data"].shape[1])),
                source["data"].shape[1],
            )
            replay = np.vstack([
                np.asarray(source["data"][rows.index(name), :committed])
                for name in requested
            ])
        if replay.shape[1] == 0:
            raise ValueError(f"mock replay contains no committed data: {path}")
        LOG.info(
            "Loaded repeating HDF5 mock replay %s: rows=%s, samples=%d, duration=%.3f s",
            path, requested, replay.shape[1], replay.shape[1] / sample_rate,
        )
        return replay


class MockDaq(BaseDaq):
    def __init__(self, *args: Any, overload_ms: float = 0.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.overload_ms = overload_ms
        self.replay_index = 0
        if self.config.mock_replay is None:
            raise ValueError("mock mode requires a replay recording")
        self.replay = load_mock_replay(
            self.config.mock_replay, self.config.electrode_labels, self.config.sample_rate
        )

    def run(self) -> None:
        LOG.info("Mock replay acquisition started: %s", self.config.mock_replay)
        period_s = self.config.chunk_size / self.config.sample_rate
        next_deadline = time.perf_counter()
        while self.state.snapshot().running:
            snapshot = self.state.snapshot()
            if not snapshot.acquiring:
                time.sleep(min(0.05, period_s))
                next_deadline = time.perf_counter()
                continue
            block_started = time.perf_counter_ns()
            next_deadline += period_s
            ai = self._next_input()
            if self.overload_ms:
                time.sleep(self.overload_ms / 1_000)
            _, safe, esn_ms, _ = self.core.process(ai)
            if self.state.snapshot().mode != "closed-loop":
                safe = np.zeros_like(safe)
            self.last_command = safe
            block_ms = (time.perf_counter_ns() - block_started) / 1e6
            self.telemetry.update_block(
                sample_index=self.core.sample_index, ai_ms=0.0, esn_ms=esn_ms, ao_ms=0.0, block_ms=block_ms,
                deadline_ms=period_s * 1_000, logger_backlog=self.core.recorder.backlog,
            )
            remaining = next_deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_deadline = time.perf_counter()
        self.request_safe_zero()

    def _next_input(self) -> np.ndarray:
        indices = (np.arange(self.config.chunk_size) + self.replay_index) % self.replay.shape[1]
        self.replay_index = int((self.replay_index + self.config.chunk_size) % self.replay.shape[1])
        return self.replay[:, indices].copy()


class Watchdog:
    def __init__(self, config: AppConfig, state: RuntimeState, telemetry: Telemetry, daq: BaseDaq) -> None:
        self.config = config
        self.state = state
        self.telemetry = telemetry
        self.daq = daq
        self.thread = threading.Thread(target=self._run, name="watchdog", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while self.state.snapshot().running:
            time.sleep(min(0.05, self.config.watchdog_timeout_s / 4))
            snapshot = self.state.snapshot()
            if not snapshot.acquiring or snapshot.mode != "closed-loop":
                continue
            age_s = (time.perf_counter_ns() - self.telemetry.snapshot()["last_heartbeat_ns"]) / 1e9
            if age_s > self.config.watchdog_timeout_s:
                self.daq.request_safe_zero()
                self.state.fault(f"DAQ watchdog expired after {age_s:.3f}s")
                LOG.critical("DAQ watchdog expired after %.3fs", age_s)


def serializable_config(config: AppConfig) -> dict[str, Any]:
    result = dataclasses.asdict(config)
    result["chunk_size"] = config.chunk_size
    result["ao_lead_samples"] = config.ao_lead_samples
    result["ao_deadline_samples"] = config.ao_deadline_samples
    result["record_dir"] = str(config.record_dir)
    result["mock_replay"] = str(config.mock_replay) if config.mock_replay else None
    return result


async def websocket_handler(websocket: Any, state: RuntimeState, recorder: H5Recorder,
                            hub: UiHub, shutdown: asyncio.Event) -> None:
    client_queue = hub.subscribe()
    LOG.info("UI client connected")

    async def receive() -> None:
        async for message in websocket:
            response: dict[str, Any]
            try:
                command = json.loads(message)
                if not isinstance(command, dict) or not isinstance(command.get("command"), str):
                    raise ValueError("command must be a JSON object with a command string")
                name = command["command"]
                response_extra: dict[str, Any] = {}
                if name == "start_recording":
                    recording_path = recorder.start(command.get("filename"))
                    state.set_recording(True)
                    response_extra["filename"] = recording_path.name
                elif name == "stop_recording":
                    state.set_recording(False)
                    await asyncio.to_thread(recorder.stop_recording)
                elif name == "start_acquisition":
                    state.set_acquiring(True)
                elif name == "stop_acquisition":
                    state.set_mode("control")
                    state.set_acquiring(False)
                elif name == "set_mode":
                    state.set_mode(str(command.get("mode", "")))
                elif name == "set_stim":
                    stim_mode = str(command.get("stim_mode", state.snapshot().stim_mode))
                    gain = float(command.get("stim_gain", state.snapshot().stim_gain))
                    state.set_stim(stim_mode, gain)
                elif name == "set_pulse_threshold":
                    state.set_pulse_threshold_std(float(command.get("pulse_threshold_std")))
                elif name == "set_pulse_window":
                    state.set_pulse_window_sec(float(command.get("pulse_window_sec")))
                elif name == "set_cortex_channel":
                    labels = state.set_cortex_channel(str(command.get("channel", "")))
                    recorder.set_electrode_labels(labels)
                elif name == "clear_fault":
                    state.clear_fault()
                elif name == "shutdown":
                    # Enter the safe state before asking main() to tear down the
                    # DAQ tasks, recorder, and WebSocket server.
                    state.set_mode("control")
                    state.set_acquiring(False)
                    state.set_recording(False)
                    await asyncio.to_thread(recorder.stop_recording)
                    shutdown.set()
                else:
                    raise ValueError(f"unknown command: {name}")
                response = {"type": "command_result", "command": name, "ok": True, **response_extra}
            except Exception as exc:
                response = {"type": "command_result", "command": locals().get("name", "unknown"), "ok": False, "error": str(exc)}
                LOG.warning("Rejected UI command: %s", exc)
            await websocket.send(json.dumps(response))

    async def transmit() -> None:
        while state.snapshot().running:
            packet = await client_queue.get()
            await websocket.send(packet)

    try:
        receiver = asyncio.create_task(receive())
        transmitter = asyncio.create_task(transmit())
        done, pending = await asyncio.wait({receiver, transmitter}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            with contextlib.suppress(ConnectionClosed):
                task.result()
    finally:
        hub.unsubscribe(client_queue)
        LOG.info("UI client disconnected")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Echo Chamber compact HDF5 closed-loop LFP system")
    parser.add_argument(
        "--mock", action="store_true",
        help=f"replay the bundled sample recording ({DEFAULT_MOCK_REPLAY.relative_to(BASE_DIR)})",
    )
    parser.add_argument(
        "--mock-replay", type=Path,
        help="replay another Echo Chamber .h5 recording; implies --mock",
    )
    parser.add_argument("--mock-overload-ms", type=float, default=0.0, help="inject processing delay for overload tests")
    parser.add_argument("--device", default="Dev1")
    parser.add_argument("--ai", nargs=2, default=("ai0", "ai1"), metavar=("ELECTRODE_1", "ELECTRODE_2"))
    parser.add_argument("--ao", default="ao0")
    parser.add_argument("--amplifier-gain", nargs=2, type=float, default=(10.0, 10.0),
                        metavar=("ELECTRODE_1", "ELECTRODE_2"),
                        help="MultiClamp Primary Output voltage gain for AI0 and AI1")
    parser.add_argument("--ao-command-gain", type=float, default=1.0,
                        help="NI AO volts per numeric ESN output unit (default: 1)")
    parser.add_argument("--stim-monitor-ai", help="optional AI channel measuring actual stimulus")
    parser.add_argument("--terminal-config", default="RSE", choices=("DIFFERENTIAL", "RSE", "NRSE"))
    parser.add_argument("--sample-rate", type=int, default=20_000)
    parser.add_argument("--processing-block-ms", type=float, default=10.0,
                        help="DAQ/ESN processing interval in milliseconds")
    parser.add_argument("--ao-target-lead-ms", type=float, default=100.0,
                        help="desired queued AO duration; refill and batching are automatic")
    parser.add_argument("--passthrough-dc-block-hz", type=float, default=DEFAULT_PASSTHROUGH_DC_BLOCK_HZ)
    parser.add_argument("--pulse-threshold-std", type=float, default=DEFAULT_PULSE_THRESHOLD_STD)
    parser.add_argument("--pulse-window-sec", type=float, default=DEFAULT_PULSE_WINDOW_SEC)
    parser.add_argument("--max-command-v", type=float, default=1.0)
    parser.add_argument("--max-slew-v-per-s", type=float, default=2_000.0)
    parser.add_argument("--dry-test-allow-sustained-ao", action="store_true",
                        default=DRY_TEST_ALLOW_SUSTAINED_AO,
                        help="scope/dummy-load only: disable charge, duty-cycle, and continuous-output trips")
    parser.add_argument("--record-dir", type=Path, default=BASE_DIR / "recordings")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--run-seconds", type=float, help="stop automatically; useful for soak tests")
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    mock_replay = args.mock_replay.resolve() if args.mock_replay else (DEFAULT_MOCK_REPLAY if args.mock else None)
    return AppConfig(
        device=args.device,
        ai_channels=tuple(args.ai),
        ao_channel=args.ao,
        amplifier_gain=tuple(args.amplifier_gain),
        ao_command_gain_v_per_esn_unit=args.ao_command_gain,
        sample_rate=args.sample_rate,
        processing_block_ms=args.processing_block_ms,
        ao_target_lead_ms=args.ao_target_lead_ms,
        passthrough_dc_block_hz=args.passthrough_dc_block_hz,
        pulse_threshold_std=args.pulse_threshold_std,
        pulse_window_sec=args.pulse_window_sec,
        terminal_config=args.terminal_config,
        record_dir=args.record_dir.resolve(),
        ws_host=args.host,
        ws_port=args.port,
        mock_replay=mock_replay,
        actual_stim_monitor_channel=args.stim_monitor_ai,
        start_paused=args.start_paused,
        open_browser=not args.no_browser,
        safety=SafetyConfig(
            max_command_v=args.max_command_v,
            max_slew_v_per_s=args.max_slew_v_per_s,
            allow_sustained_output_for_dry_test=args.dry_test_allow_sustained_ao,
        ),
    )


async def main() -> int:
    args = build_parser().parse_args()
    config = config_from_args(args)
    config.validate()
    LOG.info("Configuration: %s", json.dumps(serializable_config(config), default=str))
    if config.safety.allow_sustained_output_for_dry_test:
        LOG.warning(
            "DRY TEST MODE: sustained AO is allowed; use only with a scope or dummy load. "
            "Amplitude and slew limits remain active."
        )

    esn = EsnRuntime(ESN_ARTIFACT, config)
    state = RuntimeState(
        esn_ready=esn.ready,
        start_paused=config.start_paused,
        electrode_labels=config.electrode_labels,
        ctx_index=config.ctx_index,
        pulse_threshold_std=config.pulse_threshold_std,
        pulse_window_sec=config.pulse_window_sec,
    )
    telemetry = Telemetry()
    hub = UiHub(config.ui_queue_packets)
    metadata = {
        "application": "echoChamber.py",
        "configuration": serializable_config(config),
        "esn_artifact": str(ESN_ARTIFACT),
        "esn_ready": esn.ready,
        "esn_error": esn.error,
        "esn_bridge": "application-owned chunk adaptation and stimulation mapping",
        "ao_scheduler": "dynamic non-regenerating writer; no fixed refill count or mandatory batch size",
    }
    recorder = H5Recorder(config, metadata)
    loop = asyncio.get_running_loop()
    core = AsyncUiProcessingCore(config, state, esn, recorder, hub, loop, telemetry)
    replay_mode = args.mock or args.mock_replay is not None
    daq: BaseDaq = MockDaq(config, state, core, telemetry, overload_ms=args.mock_overload_ms) if replay_mode else RealDaq(config, state, core, telemetry)
    watchdog = Watchdog(config, state, telemetry, daq)

    def daq_worker() -> None:
        try:
            daq.run()
        except Exception as exc:
            message = f"DAQ failure: {type(exc).__name__}: {exc}"
            LOG.error("%s\n%s", message, traceback.format_exc())
            daq.request_safe_zero()
            state.fault(message)

    daq_thread = threading.Thread(target=daq_worker, name="daq", daemon=True)
    daq_thread.start()
    watchdog.start()

    shutdown = asyncio.Event()
    server = await websockets.serve(
        lambda websocket: websocket_handler(websocket, state, recorder, hub, shutdown),
        config.ws_host,
        config.ws_port,
    )
    LOG.info("UI server listening at ws://%s:%d", config.ws_host, config.ws_port)
    if config.open_browser:
        webbrowser.open((BASE_DIR / "index.html").as_uri())

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, shutdown.set)
    timer: Optional[asyncio.Task[None]] = None
    if args.run_seconds is not None:
        async def stop_later() -> None:
            await asyncio.sleep(max(0.0, args.run_seconds))
            shutdown.set()
        timer = asyncio.create_task(stop_later())

    try:
        while not shutdown.is_set():
            await asyncio.sleep(0.2)
            snapshot = state.snapshot()
            if snapshot.fault and not daq_thread.is_alive():
                LOG.error("DAQ stopped in fault state: %s", snapshot.fault)
                shutdown.set()
    finally:
        state.set_recording(False)
        state.stop()
        daq.request_safe_zero()
        await asyncio.to_thread(recorder.close)
        daq_thread.join(timeout=3.0)
        core.close_ui()
        server.close()
        await server.wait_closed()
        if timer:
            timer.cancel()
        LOG.info("Shutdown complete; telemetry=%s", telemetry.snapshot())
    return 1 if state.snapshot().fault else 0


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(threadName)s %(message)s")
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
