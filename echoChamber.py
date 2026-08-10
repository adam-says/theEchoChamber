"""Fail-safe closed-loop LFP acquisition for the Echo Chamber.

This module intentionally treats ``esn`` as a read-only black box.  It adds the
hardware, safety, recording, monitoring, and test boundaries that are missing
from the original prototype in ``closed_loop.py``.

Recording format
----------------
Each recording has a JSON metadata sidecar and a ``.npyseq`` data file.  The
data file is a sequence of ordinary, non-pickled NumPy arrays.  Read it with::

    with open(path, "rb") as stream:
        while True:
            try:
                block = np.load(stream, allow_pickle=False)
            except (EOFError, ValueError):
                break

Rows are documented in the metadata file.  This keeps the runtime dependency
free while retaining chunked binary writes and exact sample indices.
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
import signal
import threading
import time
import traceback
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Literal, Optional

import numpy as np
import websockets

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
VALID_MODES: Final = {"control", "closed-loop"}
VALID_STIM_MODES: Final = {"off", "passthrough", "threshold_pulse"}


@dataclass(frozen=True)
class SafetyConfig:
    """Independent limits applied after the unmodified ESN runtime."""

    max_command_v: float = 1.0
    max_slew_v_per_s: float = 2_000.0
    max_abs_area_v_s: float = 0.010
    area_window_s: float = 1.0
    max_active_fraction: float = 0.25
    active_threshold_v: float = 1e-3
    max_consecutive_active_s: float = 0.100
    isolator_command_v_per_output_unit: Optional[float] = None

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
    lfp_units_per_volt: tuple[float, float] = (1.0, 1.0)
    lfp_unit_label: str = "unscaled_V"
    ctx_index: int = 1
    sample_rate: int = 20_000
    chunk_size: int = 100
    ao_lead_chunks: int = 4
    ai_min_v: float = -10.0
    ai_max_v: float = 10.0
    terminal_config: str = "DIFFERENTIAL"
    visual_downsample: int = 100
    ui_interval_s: float = 0.1
    ws_host: str = "127.0.0.1"
    ws_port: int = 8765
    record_dir: Path = BASE_DIR / "recordings"
    logger_queue_blocks: int = 2_000
    ui_queue_packets: int = 1
    watchdog_timeout_s: float = 0.250
    mock_seed: int = 7
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

    def _physical(self, channel: str) -> str:
        return channel if "/" in channel else f"{self.device}/{channel}"

    def validate(self) -> None:
        if len(self.ai_channels) != 2 or len(set(self.ai_channels)) != 2:
            raise ValueError("exactly two distinct LFP AI channels are required")
        if len(self.lfp_units_per_volt) != 2 or not all(
            math.isfinite(value) and value > 0 for value in self.lfp_units_per_volt
        ):
            raise ValueError("two finite positive LFP scaling factors are required")
        if self.ctx_index not in (0, 1):
            raise ValueError("ctx_index must be 0 or 1")
        if self.sample_rate <= 0 or self.chunk_size <= 0:
            raise ValueError("sample rate and chunk size must be positive")
        if self.ao_lead_chunks < 2:
            raise ValueError("ao_lead_chunks must be at least 2")
        if self.sample_rate % 2_000 or self.chunk_size % (self.sample_rate // 2_000):
            raise ValueError("configuration is incompatible with the fixed ESN runtime")
        if self.ai_min_v >= self.ai_max_v:
            raise ValueError("invalid AI range")
        if self.visual_downsample <= 0 or self.logger_queue_blocks <= 0:
            raise ValueError("queue and downsample values must be positive")
        self.safety.validate()


@dataclass(frozen=True)
class StateSnapshot:
    running: bool
    acquiring: bool
    recording: bool
    mode: str
    stim_mode: str
    stim_gain: float
    fault: Optional[str]
    esn_ready: bool


class RuntimeState:
    def __init__(self, *, esn_ready: bool, start_paused: bool) -> None:
        self._lock = threading.RLock()
        self._running = True
        self._acquiring = not start_paused
        self._recording = False
        self._mode = "control"
        self._stim_mode = "passthrough"
        self._stim_gain = 1.0
        self._fault: Optional[str] = None
        self._esn_ready = esn_ready

    def snapshot(self) -> StateSnapshot:
        with self._lock:
            return StateSnapshot(
                self._running,
                self._acquiring,
                self._recording,
                self._mode,
                self._stim_mode,
                self._stim_gain,
                self._fault,
                self._esn_ready,
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
    """The only boundary that calls collaborator-owned ESN code."""

    def __init__(self, artifact: Path, config: AppConfig) -> None:
        self.streamer: Any = None
        self.error: Optional[str] = None
        try:
            from esn import load_artifact

            self.streamer = load_artifact(str(artifact))
            expected = getattr(self.streamer, "chunk_size", config.chunk_size)
            if expected != config.chunk_size:
                raise ValueError(f"artifact chunk_size={expected}, app chunk_size={config.chunk_size}")
            # Validate the public contract without advancing the live instance.
            probe = load_artifact(str(artifact))
            output = np.asarray(probe.process_chunk(np.zeros((2, config.chunk_size)), ctx_index=config.ctx_index))
            if output.shape != (1, config.chunk_size) or not np.all(np.isfinite(output)):
                raise ValueError(f"ESN self-test returned invalid output {output.shape}")
            self.streamer.configure(stim_mode="passthrough", stim_gain=1.0)
            LOG.info("ESN artifact loaded and passed startup self-test")
        except Exception as exc:
            self.streamer = None
            self.error = f"{type(exc).__name__}: {exc}"
            LOG.error("ESN unavailable: %s", self.error)

    @property
    def ready(self) -> bool:
        return self.streamer is not None

    def configure(self, stim_mode: str, gain: float) -> None:
        if not self.streamer:
            raise RuntimeError(self.error or "ESN unavailable")
        self.streamer.configure(stim_mode=stim_mode, stim_gain=gain)

    def reset(self) -> None:
        if self.streamer:
            self.streamer.reset()

    def process(self, data: np.ndarray, ctx_index: int) -> np.ndarray:
        if not self.streamer:
            raise RuntimeError(self.error or "ESN unavailable")
        output = np.asarray(self.streamer.process_chunk(data, ctx_index=ctx_index), dtype=np.float64)
        if output.shape != (1, data.shape[1]):
            raise ValueError(f"ESN returned {output.shape}, expected {(1, data.shape[1])}")
        return output


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
    safe_ao: np.ndarray
    actual_stim: Optional[np.ndarray]
    mode_value: float


class BinaryRecorder:
    def __init__(self, config: AppConfig, metadata: dict[str, Any]) -> None:
        self.config = config
        self.metadata = metadata
        self.items: queue.Queue[Optional[RecordBlock]] = queue.Queue(maxsize=config.logger_queue_blocks)
        self._lock = threading.RLock()
        self._file: Optional[Any] = None
        self._data_path: Optional[Path] = None
        self._meta_path: Optional[Path] = None
        self._accepting = False
        self._error: Optional[str] = None
        self._thread = threading.Thread(target=self._writer_loop, name="binary-recorder", daemon=True)
        self._thread.start()

    @property
    def backlog(self) -> int:
        return self.items.qsize()

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    def start(self) -> tuple[Path, Path]:
        with self._lock:
            if self._accepting or self._file:
                raise RuntimeError("recording is already active")
            self.config.record_dir.mkdir(parents=True, exist_ok=True)
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._data_path = self.config.record_dir / f"{stamp}_echo.npyseq"
            self._meta_path = self.config.record_dir / f"{stamp}_echo.json"
            self._file = self._data_path.open("xb")
            meta = dict(self.metadata)
            meta.update({
                "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "data_file": self._data_path.name,
                "format": "sequential NumPy arrays; float64; no pickle",
                "rows": [
                    "ai_sample_index",
                    "ao_target_sample_index",
                    *(f"{label}_raw_V" for label in self.config.electrode_labels),
                    *(f"{label}_{self.config.lfp_unit_label}" for label in self.config.electrode_labels),
                    "raw_esn_mapped_output",
                    "safe_ao_command_V",
                    "actual_stim_monitor_V" if self.config.actual_stim_monitor_channel else "actual_stim_unavailable",
                    "mode",
                ],
                "ao_pipeline_delay_samples": self.config.ao_lead_chunks * self.config.chunk_size,
                "ao_pipeline_delay_seconds": self.config.ao_lead_chunks * self.config.chunk_size / self.config.sample_rate,
            })
            self._meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
            self._error = None
            self._accepting = True
            return self._data_path, self._meta_path

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
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()
                self._file = None
        if self._error:
            raise RuntimeError(self._error)

    def close(self) -> None:
        self.stop_recording()
        self.items.put(None)
        self._thread.join(timeout=5.0)

    def _writer_loop(self) -> None:
        while True:
            item = self.items.get()
            try:
                if item is None:
                    return
                sample_ids = np.arange(item.sample_index, item.sample_index + item.ai.shape[1], dtype=np.float64)
                target_ids = sample_ids + self.config.ao_lead_chunks * self.config.chunk_size
                actual = item.actual_stim if item.actual_stim is not None else np.full((1, item.ai.shape[1]), np.nan)
                matrix = np.vstack((
                    sample_ids.reshape(1, -1), target_ids.reshape(1, -1), item.ai, item.calibrated_lfp,
                    item.raw_esn, item.safe_ao, actual,
                    np.full((1, item.ai.shape[1]), item.mode_value),
                ))
                with self._lock:
                    if not self._file:
                        raise RuntimeError("recording file closed before queued blocks drained")
                    np.save(self._file, matrix, allow_pickle=False)
            except Exception as exc:
                with self._lock:
                    self._error = f"recording writer failed: {type(exc).__name__}: {exc}"
                    self._accepting = False
                LOG.exception("Recording writer failed")
            finally:
                self.items.task_done()


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


class ProcessingCore:
    def __init__(self, config: AppConfig, state: RuntimeState, esn: EsnRuntime, recorder: BinaryRecorder,
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
        self.last_ui_ns = time.perf_counter_ns()
        self._last_esn_config: Optional[tuple[str, float]] = None

    def process(self, all_ai: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, Optional[str]]:
        block_start = time.perf_counter_ns()
        snapshot = self.state.snapshot()
        lfp = np.asarray(all_ai[:2], dtype=np.float64)
        calibrated_lfp = lfp * np.asarray(self.config.lfp_units_per_volt, dtype=np.float64).reshape(2, 1)
        actual_stim = np.asarray(all_ai[2:3], dtype=np.float64) if all_ai.shape[0] > 2 else None
        raw = np.zeros((1, self.config.chunk_size), dtype=np.float64)
        esn_ms = 0.0
        safety_reason: Optional[str] = None

        if snapshot.mode == "closed-loop":
            desired_config = (snapshot.stim_mode, snapshot.stim_gain)
            if desired_config != self._last_esn_config:
                self.esn.configure(*desired_config)
                self._last_esn_config = desired_config
            started = time.perf_counter_ns()
            raw = self.esn.process(lfp, self.config.ctx_index)
            esn_ms = (time.perf_counter_ns() - started) / 1e6
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
                self.sample_index, lfp.copy(), calibrated_lfp.copy(), raw.copy(), safe.copy(),
                actual_stim.copy() if actual_stim is not None else None,
                1.0 if snapshot.mode == "closed-loop" else 0.0,
            ))

        self._publish_ui(lfp, safe, snapshot, safety_reason)
        block_ms = (time.perf_counter_ns() - block_start) / 1e6
        start_index = self.sample_index
        self.sample_index += self.config.chunk_size
        return raw, safe, esn_ms, safety_reason

    def _publish_ui(self, ai: np.ndarray, ao: np.ndarray, snapshot: StateSnapshot,
                    safety_reason: Optional[str]) -> None:
        self.ui_ai.append(ai.copy())
        self.ui_ao.append(ao.copy())
        now = time.perf_counter_ns()
        if (now - self.last_ui_ns) / 1e9 < self.config.ui_interval_s:
            return
        combined_ai = np.hstack(self.ui_ai)
        combined_ao = np.hstack(self.ui_ao)
        telemetry = self.telemetry.snapshot()
        packet = json.dumps({
            "ai": combined_ai[:, ::self.config.visual_downsample].tolist(),
            "ao": combined_ao[:, ::self.config.visual_downsample].tolist(),
            "mode": snapshot.mode,
            "is_recording": snapshot.recording,
            "is_acquiring": snapshot.acquiring,
            "stim_mode": snapshot.stim_mode,
            "stim_gain": snapshot.stim_gain,
            "fs": self.config.sample_rate / self.config.visual_downsample,
            "fault": snapshot.fault,
            "esn_ready": snapshot.esn_ready,
            "safety_trip": safety_reason,
            "telemetry": telemetry,
            "channels": list(self.config.electrode_labels),
            "ao_is_command": True,
        })
        self.event_loop.call_soon_threadsafe(self.hub.publish, packet)
        self.ui_ai.clear()
        self.ui_ao.clear()
        self.last_ui_ns = now


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


class RealDaq(BaseDaq):
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
            "serial_number": getattr(device, "dev_serial_num", None),
            "ai": self.config.physical_ai_channels,
            "ao": self.config.physical_ao_channel,
        }

    def run(self) -> None:
        info = self.preflight()
        LOG.info("Hardware preflight passed: %s", info)
        terminal = getattr(TerminalConfiguration, self.config.terminal_config.upper(), None)
        if terminal is None:
            raise ValueError(f"unknown terminal configuration: {self.config.terminal_config}")

        with nidaqmx.Task("echo-ai") as read_task, nidaqmx.Task("echo-ao") as write_task:
            for channel in self.config.physical_ai_channels:
                read_task.ai_channels.add_ai_voltage_chan(
                    channel, terminal_config=terminal, min_val=self.config.ai_min_v, max_val=self.config.ai_max_v,
                )
            write_task.ao_channels.add_ao_voltage_chan(
                self.config.physical_ao_channel,
                min_val=-self.config.safety.max_command_v,
                max_val=self.config.safety.max_command_v,
            )
            read_task.timing.cfg_samp_clk_timing(
                self.config.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.config.chunk_size * self.config.ao_lead_chunks * 4,
            )
            write_task.timing.cfg_samp_clk_timing(
                self.config.sample_rate,
                source=f"/{self.config.device}/ai/SampleClock",
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.config.chunk_size * self.config.ao_lead_chunks * 4,
            )
            write_task.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION
            write_task.out_stream.cfg_output_buffer(self.config.chunk_size * self.config.ao_lead_chunks * 4)
            write_task.triggers.start_trigger.cfg_dig_edge_start_trig(read_task.triggers.start_trigger.term)

            reader = AnalogMultiChannelReader(read_task.in_stream)
            writer = AnalogSingleChannelWriter(write_task.out_stream, auto_start=False)
            ai = np.empty((len(self.config.physical_ai_channels), self.config.chunk_size), dtype=np.float64)
            zeros = np.zeros((self.config.chunk_size * self.config.ao_lead_chunks,), dtype=np.float64)
            writer.write_many_sample(zeros, timeout=5.0)
            write_task.start()
            read_task.start()
            LOG.info("Synchronized hardware acquisition started with %d queued zero chunks", self.config.ao_lead_chunks)

            try:
                while self.state.snapshot().running:
                    block_started = time.perf_counter_ns()
                    read_started = time.perf_counter_ns()
                    reader.read_many_sample(
                        ai, number_of_samples_per_channel=self.config.chunk_size,
                        timeout=max(1.0, 4 * self.config.chunk_size / self.config.sample_rate),
                    )
                    ai_ms = (time.perf_counter_ns() - read_started) / 1e6
                    if self.state.snapshot().acquiring:
                        _, safe, esn_ms, _ = self.core.process(ai)
                    else:
                        # Continue draining the hardware AI buffer while paused so
                        # that a later restart cannot begin with an overflow or
                        # stale samples.
                        safe = np.zeros((1, self.config.chunk_size), dtype=np.float64)
                        esn_ms = 0.0
                    write_started = time.perf_counter_ns()
                    writer.write_many_sample(safe.reshape(-1), timeout=1.0)
                    ao_ms = (time.perf_counter_ns() - write_started) / 1e6
                    self.last_command = safe
                    block_ms = (time.perf_counter_ns() - block_started) / 1e6
                    self.telemetry.update_block(
                        sample_index=self.core.sample_index, ai_ms=ai_ms, esn_ms=esn_ms, ao_ms=ao_ms,
                        block_ms=block_ms, deadline_ms=1_000 * self.config.chunk_size / self.config.sample_rate,
                        logger_backlog=self.core.recorder.backlog,
                    )
                    if self.core.recorder.error:
                        raise RuntimeError(self.core.recorder.error)
            finally:
                with contextlib.suppress(Exception):
                    self._write_zero(writer)
                    write_task.stop()
                with contextlib.suppress(Exception):
                    read_task.stop()
                self.request_safe_zero()

    def _write_zero(self, writer: Any) -> None:
        writer.write_many_sample(np.zeros(self.config.chunk_size, dtype=np.float64), timeout=1.0)


class MockDaq(BaseDaq):
    def __init__(self, *args: Any, overload_ms: float = 0.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.overload_ms = overload_ms
        self.rng = np.random.default_rng(self.config.mock_seed)
        self.replay: Optional[np.ndarray] = None
        self.replay_index = 0
        if self.config.mock_replay:
            replay = np.load(self.config.mock_replay, allow_pickle=False)
            if replay.ndim != 2 or replay.shape[0] < 2:
                raise ValueError("mock replay must have shape (at least 2, samples)")
            self.replay = np.asarray(replay[:2], dtype=np.float64)

    def run(self) -> None:
        LOG.info("Deterministic mock acquisition started (seed=%d)", self.config.mock_seed)
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
        if self.replay is not None:
            indices = (np.arange(self.config.chunk_size) + self.replay_index) % self.replay.shape[1]
            self.replay_index = int((self.replay_index + self.config.chunk_size) % self.replay.shape[1])
            return self.replay[:, indices].copy()
        t = (np.arange(self.config.chunk_size) + self.core.sample_index) / self.config.sample_rate
        common = 0.05 * np.sin(2 * np.pi * 8 * t)
        seizure = np.zeros_like(t)
        phase = t % 10.0
        active = (phase >= 6.0) & (phase < 7.0)
        seizure[active] = 0.25 * np.sin(2 * np.pi * 18 * t[active])
        noise = self.rng.normal(0.0, 0.01, (2, self.config.chunk_size))
        return np.vstack((common + seizure, 0.8 * common + 0.9 * seizure)) + noise


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
    result["record_dir"] = str(config.record_dir)
    result["mock_replay"] = str(config.mock_replay) if config.mock_replay else None
    return result


async def websocket_handler(websocket: Any, state: RuntimeState, recorder: BinaryRecorder,
                            esn: EsnRuntime, hub: UiHub) -> None:
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
                if name == "start_recording":
                    recorder.start()
                    state.set_recording(True)
                elif name == "stop_recording":
                    state.set_recording(False)
                    await asyncio.to_thread(recorder.stop_recording)
                elif name == "start_acquisition":
                    state.set_acquiring(True)
                elif name == "stop_acquisition":
                    state.set_mode("control")
                    state.set_acquiring(False)
                elif name == "set_mode":
                    changed = state.set_mode(str(command.get("mode", "")))
                    if changed:
                        esn.reset()
                elif name == "set_stim":
                    stim_mode = str(command.get("stim_mode", state.snapshot().stim_mode))
                    gain = float(command.get("stim_gain", state.snapshot().stim_gain))
                    state.set_stim(stim_mode, gain)
                elif name == "clear_fault":
                    state.clear_fault()
                else:
                    raise ValueError(f"unknown command: {name}")
                response = {"type": "command_result", "command": name, "ok": True}
            except Exception as exc:
                response = {"type": "command_result", "command": "unknown", "ok": False, "error": str(exc)}
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
            with contextlib.suppress(websockets.exceptions.ConnectionClosed):
                task.result()
    finally:
        hub.unsubscribe(client_queue)
        LOG.info("UI client disconnected")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-safe closed-loop LFP system")
    parser.add_argument("--mock", action="store_true", help="run deterministic simulated acquisition")
    parser.add_argument("--mock-replay", type=Path, help="replay a NumPy array shaped (2, samples)")
    parser.add_argument("--mock-overload-ms", type=float, default=0.0, help="inject processing delay for overload tests")
    parser.add_argument("--device", default="Dev1")
    parser.add_argument("--ai", nargs=2, default=("ai0", "ai1"), metavar=("ELECTRODE_1", "ELECTRODE_2"))
    parser.add_argument("--ao", default="ao0")
    parser.add_argument("--lfp-units-per-volt", nargs=2, type=float, default=(1.0, 1.0),
                        metavar=("ELECTRODE_1", "ELECTRODE_2"))
    parser.add_argument("--lfp-unit-label", default="unscaled_V")
    parser.add_argument("--stim-monitor-ai", help="optional AI channel measuring actual stimulus")
    parser.add_argument("--terminal-config", default="DIFFERENTIAL", choices=("DIFFERENTIAL", "RSE", "NRSE"))
    parser.add_argument("--sample-rate", type=int, default=20_000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--ao-lead-chunks", type=int, default=4)
    parser.add_argument("--max-command-v", type=float, default=1.0)
    parser.add_argument("--max-slew-v-per-s", type=float, default=2_000.0)
    parser.add_argument("--record-dir", type=Path, default=BASE_DIR / "recordings")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--run-seconds", type=float, help="stop automatically; useful for soak tests")
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        device=args.device,
        ai_channels=tuple(args.ai),
        ao_channel=args.ao,
        lfp_units_per_volt=tuple(args.lfp_units_per_volt),
        lfp_unit_label=args.lfp_unit_label,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        ao_lead_chunks=args.ao_lead_chunks,
        terminal_config=args.terminal_config,
        record_dir=args.record_dir.resolve(),
        ws_host=args.host,
        ws_port=args.port,
        mock_replay=args.mock_replay.resolve() if args.mock_replay else None,
        actual_stim_monitor_channel=args.stim_monitor_ai,
        start_paused=args.start_paused,
        open_browser=not args.no_browser,
        safety=SafetyConfig(max_command_v=args.max_command_v, max_slew_v_per_s=args.max_slew_v_per_s),
    )


async def main() -> int:
    args = build_parser().parse_args()
    config = config_from_args(args)
    config.validate()
    LOG.info("Configuration: %s", json.dumps(serializable_config(config), default=str))

    esn = EsnRuntime(ESN_ARTIFACT, config)
    state = RuntimeState(esn_ready=esn.ready, start_paused=config.start_paused)
    telemetry = Telemetry()
    hub = UiHub(config.ui_queue_packets)
    metadata = {
        "application": "echoChamber.py",
        "configuration": serializable_config(config),
        "esn_artifact": str(ESN_ARTIFACT),
        "esn_ready": esn.ready,
        "esn_error": esn.error,
    }
    recorder = BinaryRecorder(config, metadata)
    loop = asyncio.get_running_loop()
    core = ProcessingCore(config, state, esn, recorder, hub, loop, telemetry)
    daq: BaseDaq = MockDaq(config, state, core, telemetry, overload_ms=args.mock_overload_ms) if args.mock else RealDaq(config, state, core, telemetry)
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

    server = await websockets.serve(
        lambda websocket: websocket_handler(websocket, state, recorder, esn, hub),
        config.ws_host,
        config.ws_port,
    )
    LOG.info("UI server listening at ws://%s:%d", config.ws_host, config.ws_port)
    if config.open_browser:
        webbrowser.open((BASE_DIR / "index.html").as_uri())

    shutdown = asyncio.Event()
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
        server.close()
        await server.wait_closed()
        if timer:
            timer.cancel()
        LOG.info("Shutdown complete; telemetry=%s", telemetry.snapshot())
    return 1 if state.snapshot().fault else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(threadName)s %(message)s")
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
