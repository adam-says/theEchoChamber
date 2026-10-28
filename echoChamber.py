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
import hashlib
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
from controllers import (
    ActuatorFeedback, ControllerAction, ControllerInput, EsnController, PulseEvent, PulsePlan,
    PIDController, PulseActuator, RuleController,
)
from replay import load_replay

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
ESN_ARTIFACT: Final = BASE_DIR / "artifacts" / "esn_corrected_v1.npz"
DEFAULT_MOCK_REPLAY: Final = BASE_DIR / "sample" / "test_AI0_CA3_AI1_CTX.h5"
DEFAULT_HARDWARE_PROFILE: Final = BASE_DIR / "hardware_profiles" / "lab_pending_test.json"
H5_FORMAT_TAG: Final = "echoChamber_H5_v5"
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
# CONFIRMED PENDING TEST: these are the current ISO-Flex/pulse-mode working
# values. They must be checked on the NI rig with a scope or dummy load; the
# application warns but does not block hardware testing with them.

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
        if not math.isfinite(self.active_threshold_v) or not 0 < self.active_threshold_v <= self.max_command_v:
            raise ValueError("active_threshold_v must be in (0, max_command_v]")
        if self.isolator_command_v_per_output_unit is not None:
            raise ValueError(
                "isolator_command_v_per_output_unit is obsolete; express the single conversion "
                "with ao_command_gain_v_per_esn_unit"
            )


def pulse_amplitude_limit_v(safety: SafetyConfig, sample_rate_hz: int,
                            *, pulse_duration_ms: float = 5.0,
                            pulse_frequency_hz: float = 100.0,
                            refractory_s: float = 0.5,
                            max_rate_hz: float = 2.0) -> float:
    """Conservatively cap fixed-pulse AO volts against the complete safety profile."""
    pulse_samples = max(1, int(round(pulse_duration_ms * sample_rate_hz / 1000.0)))
    waveform = np.sin(2 * np.pi * pulse_frequency_hz
                      * np.arange(pulse_samples, dtype=np.float64) / sample_rate_hz)
    unit_area_v_s = float(np.sum(np.abs(waveform)) / sample_rate_hz)
    unit_slew_v_per_s = float(np.max(np.abs(np.diff(np.r_[0.0, waveform, 0.0]))) * sample_rate_hz)
    limit = min(safety.max_command_v,
                safety.max_slew_v_per_s / max(unit_slew_v_per_s, np.finfo(float).eps))
    if not safety.allow_sustained_output_for_dry_test:
        pulse_count = max(1, int(math.ceil(safety.area_window_s / refractory_s)))
        if unit_area_v_s > 0:
            limit = min(limit, safety.max_abs_area_v_s / (pulse_count * unit_area_v_s))
        if pulse_duration_ms / 1000.0 > safety.max_consecutive_active_s:
            return 0.0
        if max_rate_hz * pulse_duration_ms / 1000.0 > safety.max_active_fraction:
            return 0.0
    return max(0.0, float(limit))


@dataclass(frozen=True)
class AppConfig:
    device: str = "Dev1"
    ai_channels: tuple[str, str] = ("ai0", "ai1")
    ao_channel: str = "ao0"
    # CONFIRMED PENDING TEST: AI0=CA3 and AI1=Cortex, based on the filename and
    # collaborator handoff despite conflicting metadata in the sample HDF5.
    electrode_labels: tuple[str, str] = ("CA3", "Cortex")
    source_channel_regions: Optional[tuple[str, str]] = None
    # MultiClamp Primary Output voltage gain (output mV / electrode mV).
    # Electrode-referred input is therefore DAQ_V * 1000 / gain, in mV.
    # CONFIRMED PENDING TEST: use gain 10 until the reported "100 mV/mV"
    # setting (which would imply gain 100) is checked on the amplifier.
    amplifier_gain: tuple[float, float] = (10.0, 10.0)
    # AO remains a voltage command. This maps one numeric ESN output unit
    # (currently assumed to be mV) to NI AO volts before the UI stim gain.
    # CONFIRMED PENDING TEST: one NI AO volt per physical ESN output unit.
    ao_command_gain_v_per_esn_unit: float = 1.0
    ctx_index: int = 1
    sample_rate: int = 20_000
    processing_block_ms: float = 5.0
    ao_target_lead_ms: float = 100.0
    stimulation_recovery_ms: float = 0.0
    passthrough_dc_block_hz: float = DEFAULT_PASSTHROUGH_DC_BLOCK_HZ
    pulse_threshold_std: float = DEFAULT_PULSE_THRESHOLD_STD
    pulse_window_sec: float = DEFAULT_PULSE_WINDOW_SEC
    # CONFIRMED PENDING TEST: trigger either positive or negative excursions.
    pulse_polarity: str = "absolute"
    esn_backend: str = "auto"
    model_artifact: Path = ESN_ARTIFACT
    controller: str = "esn"
    controller_config: Optional[Path] = None
    controller_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    hardware_profile: Optional[Path] = None
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
    mock_replay_data: Any = field(default=None, repr=False, compare=False)
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
        if self.source_channel_regions is not None and len(self.source_channel_regions) != 2:
            raise ValueError("source_channel_regions must contain exactly two labels")
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
        if not math.isfinite(self.stimulation_recovery_ms) or self.stimulation_recovery_ms < 0:
            raise ValueError("stimulation_recovery_ms must be finite and non-negative")
        if self.passthrough_dc_block_hz < 0:
            raise ValueError("passthrough_dc_block_hz must be non-negative")
        if not math.isfinite(self.pulse_threshold_std) or self.pulse_threshold_std <= 0:
            raise ValueError("pulse_threshold_std must be finite and positive")
        if not math.isfinite(self.pulse_window_sec) or self.pulse_window_sec < 1:
            raise ValueError("pulse_window_sec must be finite and at least one second")
        if self.pulse_polarity not in {"absolute", "positive", "negative"}:
            raise ValueError("pulse_polarity must be absolute, positive, or negative")
        if self.esn_backend not in {"auto", "numpy", "numba"}:
            raise ValueError("esn_backend must be auto, numpy, or numba")
        if self.controller not in {"esn", "rule", "pid"}:
            raise ValueError("controller must be esn, rule, or pid")
        if self.controller_config is not None and not self.controller_config.is_file():
            raise FileNotFoundError(self.controller_config)
        if self.mock_replay_data is not None and self.mock_replay_data.sample_rate_hz != self.sample_rate:
            raise ValueError("mock replay sample rate must match configured sample rate")
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
    pulse_polarity: str
    armed: bool
    fault: Optional[str]
    esn_ready: bool
    electrode_labels: tuple[str, str]
    ctx_index: int
    controller: str = "esn"
    generation: int = 0
    controller_ready: bool = False
    transition_pending: bool = False
    controller_calibrated: bool = False
    calibration_epoch: int = 0
    controller_settings: tuple[tuple[str, Any], ...] = ()
    input_mapping_locked: bool = False


class RuntimeState:
    def __init__(self, *, esn_ready: bool, start_paused: bool,
                 electrode_labels: tuple[str, str], ctx_index: int,
                 pulse_threshold_std: float, pulse_window_sec: float,
                 pulse_polarity: str = "absolute", controller: str = "esn",
                 controller_ready: Optional[dict[str, bool]] = None,
                 controller_settings: Optional[dict[str, dict[str, Any]]] = None,
                 pulse_max_amplitude_v: float = 10.0,
                 input_mapping_locked: bool = False) -> None:
        self._lock = threading.RLock()
        self._running = True
        self._acquiring = not start_paused
        self._recording = False
        self._mode = "control"
        self._stim_mode = "off"
        self._stim_gain = 0.0
        self._pulse_threshold_std = pulse_threshold_std
        self._pulse_window_sec = pulse_window_sec
        self._pulse_polarity = pulse_polarity
        self._armed = False
        self._fault: Optional[str] = None
        self._esn_ready = esn_ready
        self._controller = controller
        self._controller_ready = dict(controller_ready or {"esn": esn_ready, "rule": True, "pid": True})
        self._controller_calibrated = {"esn": esn_ready, "rule": False, "pid": False}
        self._generation = 0
        self._transition_pending = False
        self._calibration_epoch = 0
        self._pulse_max_amplitude_v = float(pulse_max_amplitude_v)
        self._controller_settings = {name: dict(value) for name, value in (controller_settings or {}).items()}
        self._input_mapping_locked = bool(input_mapping_locked)
        self._prepared_controllers: dict[str, Any] = {}
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
                self._pulse_polarity,
                self._armed,
                self._fault,
                self._esn_ready,
                self._electrode_labels,
                self._ctx_index,
                self._controller,
                self._generation,
                self._controller_ready.get(self._controller, False),
                self._transition_pending,
                self._controller_calibrated.get(self._controller, False),
                self._calibration_epoch,
                tuple(sorted(self._controller_settings.get(self._controller, {}).items())),
                self._input_mapping_locked,
            )

    def lock_input_mapping(self) -> None:
        with self._lock:
            self._input_mapping_locked = True

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._acquiring = False
            self._mode = "control"
            self._armed = False

    def set_acquiring(self, value: bool) -> None:
        with self._lock:
            if self._fault and value:
                raise ValueError("cannot start acquisition while faulted")
            value = bool(value)
            if value and not self._acquiring:
                self._generation += 1
                self._transition_pending = True
                self._calibration_epoch += 1
                self._controller_calibrated.update({"rule": False, "pid": False})
            elif not value and self._acquiring:
                self._generation += 1
                self._transition_pending = True
                self._armed = False
            self._acquiring = value
            if not value:
                self._armed = False

    def set_recording(self, value: bool) -> None:
        with self._lock:
            self._recording = bool(value)

    def set_mode(self, mode: str) -> bool:
        if mode not in VALID_MODES:
            raise ValueError(f"invalid mode: {mode}")
        with self._lock:
            if mode == "closed-loop" and not self._controller_ready.get(self._controller, False):
                raise ValueError(f"closed-loop unavailable: {self._controller} controller is not ready")
            if mode == "closed-loop" and self._fault:
                raise ValueError("closed-loop unavailable while faulted")
            changed = self._mode != mode
            self._mode = mode
            if changed:
                self._generation += 1
                self._transition_pending = True
                self._armed = False
            if mode != "closed-loop":
                self._armed = False
            return changed

    def set_stim(self, stim_mode: str, gain: float, max_gain: float = 10.0) -> None:
        if stim_mode not in VALID_STIM_MODES:
            raise ValueError(f"invalid stimulation mode: {stim_mode}")
        if not math.isfinite(gain) or not 0 <= gain <= max_gain:
            raise ValueError(f"stim_gain must be finite and in [0, {max_gain}]")
        with self._lock:
            if self._controller in {"rule", "pid"} and stim_mode not in {"off", "threshold_pulse"}:
                raise ValueError("Rule and PID controllers use fixed pulse output")
            if (stim_mode, float(gain)) != (self._stim_mode, self._stim_gain):
                self._generation += 1
                self._transition_pending = True
                self._armed = False
            self._stim_mode = stim_mode
            self._stim_gain = float(gain)
            if stim_mode == "off" or gain == 0:
                self._armed = False

    def set_armed(self, value: bool) -> None:
        with self._lock:
            if value:
                if self._fault:
                    raise ValueError("cannot arm stimulation while faulted")
                if not self._acquiring or self._mode != "closed-loop":
                    raise ValueError("start acquisition and closed-loop computation before arming")
                if self._stim_mode == "off" or self._stim_gain <= 0:
                    raise ValueError("select a non-zero stimulation mode and gain before arming")
                if not self._controller_ready.get(self._controller, False):
                    raise ValueError(f"cannot arm stimulation while {self._controller} is unavailable")
                if self._transition_pending:
                    raise ValueError("wait for output drain to complete before arming")
                if self._controller in {"rule", "pid"} and not self._controller_calibrated.get(self._controller, False):
                    raise ValueError(f"cannot arm until {self._controller} baseline calibration is complete")
                if (self._stim_mode == "threshold_pulse" or self._controller in {"rule", "pid"}) \
                        and self._stim_gain > self._pulse_max_amplitude_v:
                    raise ValueError(f"pulse amplitude exceeds configured safety limit ({self._pulse_max_amplitude_v:.3f} V)")
            elif self._armed:
                self._generation += 1
                self._transition_pending = True
            self._armed = bool(value)

    def set_controller(self, controller: str) -> bool:
        if controller not in {"esn", "rule", "pid"}:
            raise ValueError("controller must be esn, rule, or pid")
        with self._lock:
            if self._fault and controller != self._controller:
                raise ValueError("cannot switch controller while faulted")
            if not self._controller_ready.get(controller, False):
                raise ValueError(f"{controller} controller is not ready")
            if controller == self._controller:
                return False
            self._controller = controller
            self._armed = False
            self._generation += 1
            self._transition_pending = True
            return True

    def set_controller_settings(self, controller: str, settings: dict[str, Any], prepared: Any) -> None:
        with self._lock:
            if controller != self._controller:
                raise ValueError("settings can only be applied to the selected controller")
            old = self._controller_settings.get(controller, {})
            self._controller_settings[controller] = dict(settings)
            self._prepared_controllers[controller] = prepared
            if old.get("band_hz") != settings.get("band_hz"):
                self._calibration_epoch += 1
                self._controller_calibrated[controller] = False
            self._generation += 1
            self._transition_pending = True
            self._armed = False

    def take_prepared_controller(self, controller: str) -> Any:
        with self._lock:
            return self._prepared_controllers.pop(controller, None)

    def acknowledge_drain(self, generation: int) -> None:
        with self._lock:
            if generation == self._generation:
                self._transition_pending = False

    def set_pulse_threshold_std(self, value: float) -> None:
        if not math.isfinite(value) or not 0 < value <= 20:
            raise ValueError("pulse threshold must be finite and in (0, 20] standard deviations")
        with self._lock:
            if value != self._pulse_threshold_std:
                self._generation += 1
                self._transition_pending = True
                self._armed = False
            self._pulse_threshold_std = float(value)

    def set_pulse_window_sec(self, value: float) -> None:
        if not math.isfinite(value) or not 1 <= value <= 120:
            raise ValueError("pulse baseline window must be finite and between 1 and 120 seconds")
        with self._lock:
            if value != self._pulse_window_sec:
                self._generation += 1
                self._transition_pending = True
                self._armed = False
            self._pulse_window_sec = float(value)

    def set_pulse_polarity(self, value: str) -> None:
        if value not in {"absolute", "positive", "negative"}:
            raise ValueError("pulse polarity must be absolute, positive, or negative")
        with self._lock:
            if value != self._pulse_polarity:
                self._generation += 1
                self._transition_pending = True
                self._armed = False
            self._pulse_polarity = value

    def set_cortex_channel(self, channel: str) -> tuple[str, str]:
        """Assign Cortex/CA3 to physical AI0/AI1 before recording begins."""
        if channel not in {"ai0", "ai1"}:
            raise ValueError("Cortex channel must be ai0 or ai1")
        with self._lock:
            if self._input_mapping_locked:
                raise ValueError("Cortex/CA3 mapping is fixed by the calibrated mV input source")
            if self._recording:
                raise ValueError("stop recording before changing the Cortex/CA3 channel assignment")
            self._ctx_index = 0 if channel == "ai0" else 1
            cortex_label = next((label for label in self._electrode_labels
                                 if any(key in label.casefold() for key in ("cortex", "ctx", "ec"))), "Cortex")
            ca3_label = next((label for label in self._electrode_labels if "ca3" in label.casefold()), "CA3")
            self._electrode_labels = ((cortex_label, ca3_label) if self._ctx_index == 0
                                      else (ca3_label, cortex_label))
            self._generation += 1
            self._transition_pending = True
            self._calibration_epoch += 1
            self._controller_calibrated.update({"rule": False, "pid": False})
            return self._electrode_labels

    def update_controller_calibration(self, controller: str, calibrated: bool) -> None:
        with self._lock:
            if controller in {"rule", "pid"}:
                self._controller_calibrated[controller] = bool(calibrated)

    def invalidate_controller_calibration(self) -> None:
        """Clear cached controller readiness after a true source discontinuity."""
        with self._lock:
            self._controller_calibrated.update({"rule": False, "pid": False})

    def fault(self, message: str) -> None:
        with self._lock:
            self._fault = message
            self._mode = "control"
            self._acquiring = False
            self._armed = False

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
        self.production_eligible = False
        try:
            from esn_bridge import EchoChamberEsnBridge

            self.streamer = EchoChamberEsnBridge.load(
                artifact,
                runtime_chunk_size=config.chunk_size,
                sample_rate=config.sample_rate,
                passthrough_dc_block_hz=config.passthrough_dc_block_hz,
                pulse_threshold_std=config.pulse_threshold_std,
                pulse_window_sec=config.pulse_window_sec,
                pulse_polarity=config.pulse_polarity,
                esn_backend=config.esn_backend,
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
            artifact_object = getattr(self.streamer.streamer, "artifact", None)
            manifest = getattr(artifact_object, "manifest", {})
            self.production_eligible = bool(manifest.get("production_eligible", False))
            LOG.info("ESN artifact loaded and passed startup self-test")
        except Exception as exc:
            self.streamer = None
            self.error = f"{type(exc).__name__}: {exc}"
            LOG.error("ESN unavailable: %s", self.error)

    @property
    def ready(self) -> bool:
        return self.streamer is not None

    def configure(self, stim_mode: str, gain: float, pulse_threshold_std: float | None = None,
                  pulse_window_sec: float | None = None, pulse_polarity: str | None = None) -> None:
        if not self.streamer:
            raise RuntimeError(self.error or "ESN unavailable")
        self.streamer.configure(
            stim_mode=stim_mode,
            stim_gain=gain,
            pulse_threshold_std=pulse_threshold_std,
            pulse_window_sec=pulse_window_sec,
            pulse_polarity=pulse_polarity,
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
        self.area_sum_v_s = 0.0
        self.area_sum_compensation = 0.0
        self.active_count = 0
        self.consecutive_active = 0
        self.last_reason: Optional[str] = None

    def reset(self) -> None:
        self.previous_v = 0.0
        self.area_history.clear()
        self.active_history.clear()
        self.area_sum_v_s = 0.0
        self.area_sum_compensation = 0.0
        self.active_count = 0
        self.consecutive_active = 0
        self.last_reason = None

    def admit_pulse_tail(self, waveform_v: np.ndarray) -> tuple[np.ndarray, Optional[str]]:
        """Validate a fully synthesized pulse tail with the same rolling safety rules.

        This evaluates a copy of current safety history so it can reserve safe
        future output without advancing the live history ahead of the AO clock.
        """
        candidate = StimulusSafetyAdapter(self.config, self.sample_rate)
        candidate.previous_v = self.previous_v
        candidate.area_history = deque(self.area_history, maxlen=self.area_history.maxlen)
        candidate.active_history = deque(self.active_history, maxlen=self.active_history.maxlen)
        candidate.area_sum_v_s = self.area_sum_v_s
        candidate.area_sum_compensation = self.area_sum_compensation
        candidate.active_count = self.active_count
        candidate.consecutive_active = self.consecutive_active
        return candidate.process(np.asarray(waveform_v, dtype=np.float64))

    def process(self, raw: np.ndarray) -> tuple[np.ndarray, Optional[str]]:
        signal_v = np.asarray(raw, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(signal_v)):
            return self._trip(signal_v.size, "non-finite ESN output")

        command_limit = self.config.max_command_v
        if signal_v.size and (np.any(signal_v > command_limit) or np.any(signal_v < -command_limit)):
            signal_v = np.clip(signal_v, -command_limit, command_limit)
        max_step = self.config.max_slew_v_per_s / self.sample_rate
        safe = np.empty_like(signal_v)
        previous = self.previous_v
        for index, value in enumerate(signal_v):
            delta = float(value) - previous
            if delta > max_step:
                previous += max_step
            elif delta < -max_step:
                previous -= max_step
            else:
                previous = float(value)
            safe[index] = previous
        self.previous_v = float(safe[-1]) if safe.size else self.previous_v

        dt_s = 1.0 / self.sample_rate
        for value in safe:
            area = float(value) * dt_s
            if len(self.area_history) == self.area_history.maxlen:
                self._accumulate_area(-self.area_history[0])
            self.area_history.append(area)
            self._accumulate_area(area)
            active = int(abs(float(value)) >= self.config.active_threshold_v)
            if len(self.active_history) == self.active_history.maxlen:
                self.active_count -= self.active_history[0]
            self.active_history.append(active)
            self.active_count += active
            self.consecutive_active = self.consecutive_active + 1 if active else 0

        if not self.config.allow_sustained_output_for_dry_test:
            # The tolerance only makes the incremental rolling sum more
            # conservative at a floating-point boundary.
            area_tolerance = max(1e-15, abs(self.area_sum_v_s) * 8 * np.finfo(float).eps)
            if abs(self.area_sum_v_s) > self.config.max_abs_area_v_s - area_tolerance:
                return self._trip(safe.size, "net command area limit exceeded")
            if len(self.active_history) == self.active_history.maxlen and (
                self.active_count / len(self.active_history) > self.config.max_active_fraction
            ):
                return self._trip(safe.size, "stimulation duty-cycle limit exceeded")
            if self.consecutive_active > int(self.config.max_consecutive_active_s * self.sample_rate):
                return self._trip(safe.size, "continuous stimulation limit exceeded")

        self.last_reason = None
        return safe.reshape(1, -1), None

    def _accumulate_area(self, value: float) -> None:
        """Kahan-add one signed sample area to the rolling-window total."""
        adjusted = value - self.area_sum_compensation
        total = self.area_sum_v_s + adjusted
        self.area_sum_compensation = (total - self.area_sum_v_s) - adjusted
        self.area_sum_v_s = total

    def _trip(self, count: int, reason: str) -> tuple[np.ndarray, str]:
        self.last_reason = reason
        self.reset()
        self.last_reason = reason
        return np.zeros((1, count), dtype=np.float64), reason


@dataclass(frozen=True)
class RecordBlock:
    sample_index: int
    ao_target_sample_index: int
    ai: np.ndarray
    raw_esn: np.ndarray
    model_esn: np.ndarray
    pulse_threshold: np.ndarray
    pulse_peak: np.ndarray
    pulse_fired: np.ndarray
    safe_ao: np.ndarray
    actual_stim: Optional[np.ndarray]
    armed: bool
    mode_value: float
    ai_electrode_mV: Optional[np.ndarray] = None
    controller_id: str = "esn"
    config_generation: int = 0
    controller_diagnostics: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class SafetyAdmittedPulsePlan:
    """A complete pulse waveform approved by the shared output safety boundary."""

    pulse_id: str
    generation: int
    source_onset_sample_index: int
    target_onset_sample_index: int
    latest_sample_index: int
    waveform_v: np.ndarray


class H5Recorder:
    """Asynchronous, batched HDF5 recorder owned by one writer thread."""

    def __init__(self, config: AppConfig, metadata: dict[str, Any]) -> None:
        self.config = config
        self.metadata = metadata
        self.items: queue.Queue[Optional[RecordBlock]] = queue.Queue(maxsize=config.logger_queue_blocks)
        self._lock = threading.RLock()
        self._admission_lock = threading.Lock()
        self._recording_open = False
        self._submit_stats_lock = threading.Lock()
        self._submit_count = 0
        self._submit_wait_total_ns = 0
        self._submit_wait_max_ns = 0
        self._file: Optional[h5py.File] = None
        self._ai_dataset: Optional[h5py.Dataset] = None
        self._electrode_dataset: Optional[h5py.Dataset] = None
        self._ao_dataset: Optional[h5py.Dataset] = None
        self._actual_dataset: Optional[h5py.Dataset] = None
        self._diagnostics_dataset: Optional[h5py.Dataset] = None
        self._controller_dataset: Optional[h5py.Dataset] = None
        self._mode_dataset: Optional[h5py.Dataset] = None
        self._pulse_dataset: Optional[h5py.Dataset] = None
        self._gap_dataset: Optional[h5py.Dataset] = None
        self._data_path: Optional[Path] = None
        self._accepting = False
        self._error: Optional[str] = None
        self._committed_samples = 0
        self._last_source_end: Optional[int] = None
        self._last_mode: Optional[int] = None
        self._action_stage_lock = threading.Lock()
        self._pending_action_stages: list[tuple[int, int, str, str, float, str, str, int, int, int]] = []
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
                "format": "Echo Chamber HDF5 v5; hardware-indexed signals and sparse events",
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
            self._electrode_dataset = signals.create_dataset(
                "electrode_mV", shape=(2, 0), maxshape=(2, None),
                chunks=(2, self._batch_samples), dtype=np.float32, **compression,
            )
            self._electrode_dataset.attrs["channel_labels"] = np.asarray(self.config.electrode_labels, dtype=string_type)
            self._electrode_dataset.attrs["channel_roles"] = np.asarray(
                ["Cortex" if i == self.config.ctx_index else "CA3" for i in range(2)], dtype=string_type)
            regions = (self.config.source_channel_regions or tuple(
                "CA3" if "ca3" in label.casefold() else
                "EC" if label.strip().casefold() == "ec" or "entorhinal" in label.casefold() else
                "Cortex" for label in self.config.electrode_labels))
            self._electrode_dataset.attrs["channel_regions"] = np.asarray(
                regions, dtype=string_type)
            self._electrode_dataset.attrs["unit"] = "mV"
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
                ("sample_offset", "<u8"), ("ai_sample_index", "<u8"),
                ("ao_target_sample_index", "<u8"), ("sample_count", "<u4"),
                ("raw_esn_mean_V", "<f4"), ("raw_esn_min_V", "<f4"),
                ("raw_esn_max_V", "<f4"), ("model_esn_mean_mV", "<f4"),
                ("pulse_threshold_mV", "<f4"), ("pulse_peak_mV", "<f4"),
                ("armed", "u1"),
            ])
            self._diagnostics_dataset = diagnostics.create_dataset(
                "blocks", shape=(0,), maxshape=(None,), chunks=(max(1, self._batch_samples // self.config.chunk_size),),
                dtype=diagnostic_dtype, **compression,
            )
            controller_dtype = np.dtype([
                ("sample_offset", "<u8"), ("config_generation", "<u8"),
                ("controller_id", "S8"), ("ready", "u1"), ("calibrated", "u1"),
                ("envelope_mV", "<f4"), ("baseline_mean_mV", "<f4"),
                ("baseline_std_mV", "<f4"), ("z_score", "<f4"),
                ("target_z", "<f4"), ("error", "<f4"), ("output_rate_Hz", "<f4"),
                ("state_code", "u1"),
                ("feature_valid", "u1"), ("feature_eligible_sample_count", "<u4"),
                ("feature_recovery_samples", "<u4"),
                ("feature_recovery_remaining_samples", "<u4"),
                ("stimulated_sample_count", "<u4"),
                ("feature_invalid_sample_count", "<u4"),
                ("feature_recovery_reason", "S40"),
            ])
            self._controller_dataset = diagnostics.create_dataset(
                "controller_blocks", shape=(0,), maxshape=(None,), chunks=(128,),
                dtype=controller_dtype, **compression)
            events.create_dataset("action_stages", shape=(0,), maxshape=(None,), chunks=(128,),
                                  dtype=np.dtype([("request_id", "S32"), ("generation", "<u8"),
                                                  ("source_sample_index", "<u8"), ("latest_sample_index", "<u8"),
                                                  ("target_sample_index", "<u8"), ("sample_index", "<u8"),
                                                  ("stage", "S24"), ("controller", "S16"),
                                                  ("requested_rate_Hz", "<f4"), ("reason", "S96")]), **compression)
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
            self._file.attrs["schema_version"] = 5
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
            with self._action_stage_lock:
                self._pending_action_stages.clear()
            with self._admission_lock:
                self._recording_open = True
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
        started = time.perf_counter_ns()
        with self._admission_lock:
            waited = time.perf_counter_ns() - started
            if not self._accepting:
                return
            try:
                self.items.put_nowait(block)
            except queue.Full as exc:
                raise RuntimeError("recording queue overflow; raw data integrity cannot be guaranteed") from exc
        with self._submit_stats_lock:
            self._submit_count += 1
            self._submit_wait_total_ns += waited
            self._submit_wait_max_ns = max(self._submit_wait_max_ns, waited)

    def submit_wait_snapshot(self) -> dict[str, float | int]:
        with self._submit_stats_lock:
            count = self._submit_count
            total = self._submit_wait_total_ns
            maximum = self._submit_wait_max_ns
        return {"submissions": count, "total_wait_us": total / 1_000.0,
                "max_wait_us": maximum / 1_000.0,
                "mean_wait_us": total / max(1, count) / 1_000.0}

    def record_action_stage(self, *, sample_index: int, generation: int, stage: str,
                            controller: str, requested_rate_hz: float = 0.0, reason: str = "",
                            request_id: str = "", source_sample_index: Optional[int] = None,
                            latest_sample_index: Optional[int] = None,
                            target_sample_index: Optional[int] = None) -> None:
        # Per-block signals and compact diagnostics are the normal timeline.
        # The event table is reserved for pulse lifecycle IDs, exceptional AO
        # outcomes, drain acknowledgements, and explicitly nonzero analog
        # outcomes. In particular, virtual clock ticks and routine zero writes
        # must not create one row per block.
        if stage in {"simulated_generated", "proposed"}:
            return
        if stage in {"ao_committed", "tail_preserved", "cancelled", "late"} and not request_id:
            return
        if stage == "zero_substituted" and not request_id and reason == "no current-generation command":
            return
        with self._admission_lock:
            if not self._accepting:
                return
            self._append_action_stage(sample_index=sample_index, generation=generation,
                stage=stage, controller=controller, requested_rate_hz=requested_rate_hz,
                reason=reason, request_id=request_id, source_sample_index=source_sample_index,
                latest_sample_index=latest_sample_index, target_sample_index=target_sample_index)

    def _append_action_stage(self, *, sample_index: int, generation: int, stage: str,
                             controller: str, requested_rate_hz: float = 0.0, reason: str = "",
                             request_id: str = "", source_sample_index: Optional[int] = None,
                             latest_sample_index: Optional[int] = None,
                             target_sample_index: Optional[int] = None) -> None:
        """Append an event derived from an already-admitted recorder block."""
        with self._action_stage_lock:
            self._pending_action_stages.append((sample_index, generation, stage, controller,
                requested_rate_hz, reason, request_id,
                sample_index if source_sample_index is None else source_sample_index,
                sample_index if latest_sample_index is None else latest_sample_index,
                sample_index if target_sample_index is None else target_sample_index))

    def _flush_action_stages(self) -> None:
        if self._file is None:
            return
        with self._action_stage_lock:
            rows, self._pending_action_stages = self._pending_action_stages, []
        if rows:
            dataset = self._file["events/action_stages"]
            start = dataset.shape[0]
            dataset.resize(start + len(rows), axis=0)
            dataset[start:] = np.asarray([
                (request_id.encode()[:32], generation, source, latest, target, sample,
                 stage.encode()[:24], controller.encode()[:16], rate, reason.encode()[:96])
                for sample, generation, stage, controller, rate, reason, request_id, source, latest, target in rows
            ], dtype=dataset.dtype)

    def stop_recording(self, timeout: float = 10.0) -> None:
        with self._admission_lock:
            has_recording = self._recording_open
            self._accepting = False
        if not has_recording:
            return
        deadline = time.monotonic() + timeout
        while self.items.unfinished_tasks:
            if not self._thread.is_alive():
                raise RuntimeError(self._error or "recording writer stopped before its queue drained")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"recording queue did not drain within {timeout:.1f}s "
                    f"({self.items.unfinished_tasks} item(s) remain)"
                )
            time.sleep(0.01)
        with self._lock:
            if self._file:
                self._file.attrs["committed_samples"] = self._committed_samples
                self._flush_action_stages()
                self._file.attrs["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
                self._file.attrs.modify("complete", np.uint8(self._error is None))
                self._file.flush()
                with contextlib.suppress(Exception):
                    os.fsync(self._file.id.get_vfd_handle())
                self._file.close()
                self._file = None
                self._ai_dataset = self._electrode_dataset = self._ao_dataset = self._actual_dataset = None
                self._diagnostics_dataset = self._mode_dataset = None
                self._controller_dataset = None
                self._pulse_dataset = self._gap_dataset = None
        with self._admission_lock:
            self._recording_open = False
        if self._error:
            raise RuntimeError(self._error)

    def close(self) -> None:
        self.stop_recording()
        self.items.put(None, timeout=1.0)
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise TimeoutError("recording writer did not stop within 5 seconds")


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
                    electrode = np.hstack([
                        item.ai_electrode_mV if item.ai_electrode_mV is not None else
                        item.ai * (1000.0 / np.asarray(self.config.amplifier_gain)[:, None])
                        for item in batch
                    ]).astype(np.float32, copy=False)
                    ao = np.hstack([item.safe_ao for item in batch]).astype(np.float32, copy=False)
                    stop = start + ai.shape[1]
                    self._ai_dataset.resize(stop, axis=1)
                    self._electrode_dataset.resize(stop, axis=1)
                    self._ao_dataset.resize(stop, axis=1)
                    self._ai_dataset[:, start:stop] = ai
                    self._electrode_dataset[:, start:stop] = electrode
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
                    self._flush_action_stages()
                    if time.monotonic() - self._last_flush >= self.config.recorder_flush_s:
                        self._file.flush()
                        self._last_flush = time.monotonic()
            except Exception as exc:
                with self._lock:
                    self._flush_action_stages()
                    self._error = f"recording writer failed: {type(exc).__name__}: {exc}"
                with self._admission_lock:
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
            offset, item.sample_index, item.ao_target_sample_index, count,
            self._finite_mean(item.raw_esn), self._finite_min(item.raw_esn),
            self._finite_max(item.raw_esn), self._finite_mean(item.model_esn),
            self._finite_mean(item.pulse_threshold), self._finite_mean(item.pulse_peak),
            int(item.armed),
        )], dtype=self._diagnostics_dataset.dtype)
        self._append_records(self._diagnostics_dataset, diagnostic)

        diag = item.controller_diagnostics or {}
        states = {"inactive": 0, "calibrating": 1, "active": 2, "armed": 3,
                  "latched": 4, "fault": 5}
        controller_record = np.asarray([(
            offset, item.config_generation, item.controller_id.encode()[:8],
            int(bool(diag.get("ready", item.controller_id == "esn"))),
            int(bool(diag.get("calibrated", item.controller_id == "esn"))),
            float(diag.get("envelope_mv", np.nan)), float(diag.get("baseline_mean_mv", np.nan)),
            float(diag.get("baseline_std_mv", np.nan)), float(diag.get("z_score", np.nan)),
            float(diag.get("target_z", np.nan)), float(diag.get("error", np.nan)),
            float(diag.get("output_rate_hz", diag.get("requested_rate_hz", np.nan))),
            states.get(str(diag.get("state", "inactive")), 0),
            int(bool(diag.get("feature_valid", True))),
            max(0, int(item.ai.shape[1]) - int(diag.get("feature_invalid_sample_count", 0))),
            int(diag.get("feature_recovery_samples", 0)),
            int(diag.get("feature_recovery_remaining_samples", 0)),
            int(diag.get("stimulated_sample_count", 0)),
            int(diag.get("feature_invalid_sample_count", 0)),
            (str(diag.get("reason", "")) if diag.get("reason") == "stimulus_feature_recovery" else "").encode()[:40],
        )], dtype=self._controller_dataset.dtype)
        self._append_records(self._controller_dataset, controller_record)
        for event in diag.get("pulse_events", ()):
            self._append_action_stage(
                sample_index=int(event.get("target_onset_sample_index", item.ao_target_sample_index)),
                generation=int(event.get("generation", item.config_generation)),
                stage=str(event.get("status", "proposed")), controller=item.controller_id,
                request_id=str(event.get("pulse_id", "")),
                source_sample_index=int(event.get("source_onset_sample_index", item.sample_index)),
                latest_sample_index=int(diag.get("latest_sample_index", item.sample_index)),
                target_sample_index=int(event.get("target_onset_sample_index", item.ao_target_sample_index)),
                reason="controller and actuator proposal; AO scheduler outcome follows",
            )

        if np.any(item.pulse_fired > 0.5):
            pulse = np.asarray([(
                offset, item.sample_index,
                self._finite_mean(item.pulse_threshold), self._finite_mean(item.pulse_peak),
            )], dtype=self._pulse_dataset.dtype)
            self._append_records(self._pulse_dataset, pulse)


def record_sparse_ao_outcome(recorder: Optional[H5Recorder], *, sample_index: int,
                             generation: int, stage: str, controller: str,
                             waveform: Optional[np.ndarray] = None,
                             events: tuple[Any, ...] = (), reason: str = "") -> None:
    """Persist pulse-ID outcomes and exceptional/nonzero analog outcomes only."""
    if recorder is None:
        return
    pulse_events = tuple(events)
    if pulse_events:
        for event in pulse_events:
            recorder.record_action_stage(
                sample_index=int(event.target_onset_sample_index), generation=int(event.generation),
                stage=stage, controller=controller, request_id=str(event.pulse_id),
                source_sample_index=int(event.source_onset_sample_index),
                latest_sample_index=int(event.latest_sample_index),
                target_sample_index=int(event.target_onset_sample_index), reason=reason,
            )
        return
    nonzero = waveform is not None and bool(np.any(np.asarray(waveform) != 0.0))
    if nonzero:
        accepted = stage in {"ao_committed", "tail_preserved", "analog_accepted"}
        recorder.record_action_stage(
            sample_index=sample_index, generation=generation,
            stage="analog_accepted" if accepted else "analog_rejected",
            controller=controller, reason=reason,
        )
    elif stage == "zero_substituted":
        recorder.record_action_stage(
            sample_index=sample_index, generation=generation, stage=stage,
            controller=controller, reason=reason,
        )


def record_drain_ack(recorder: Optional[H5Recorder], *, sample_index: int,
                     generation: int, controller: str, end_sample: int) -> None:
    if recorder is not None:
        recorder.record_action_stage(
            sample_index=sample_index, generation=generation,
            stage="drain_acknowledged", controller=controller,
            target_sample_index=end_sample,
            reason="generated AO counter reached preserved output end and committed zero barrier",
        )


class UiHub:
    def __init__(self, queue_size: int) -> None:
        self.queue_size = queue_size
        self._clients: set[asyncio.Queue[str]] = set()
        self._latest: Optional[str] = None

    def subscribe(self) -> asyncio.Queue[str]:
        client: asyncio.Queue[str] = asyncio.Queue(maxsize=self.queue_size)
        self._clients.add(client)
        if self._latest is not None:
            client.put_nowait(self._latest)
        return client

    def unsubscribe(self, client: asyncio.Queue[str]) -> None:
        self._clients.discard(client)

    def publish(self, packet: str) -> None:
        self._latest = packet
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


def json_safe(value: Any) -> Any:
    """Convert runtime diagnostics to strict JSON without leaking NaN/Infinity."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.ndarray,)):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


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
        self.ui_sample_indices: list[np.ndarray] = []
        self.last_ui_ns = time.perf_counter_ns()
        self._last_esn_config: Optional[tuple[str, float, float, float, str]] = None
        # ESN/filter state belongs exclusively to the DAQ thread.  UI commands
        # only change RuntimeState; transitions are applied here at a block
        # boundary so reset() can never race process_chunk().
        self._active_mode = "control"
        initial_state = state.snapshot()
        self.controllers: dict[str, Any] = {"esn": EsnController(esn.streamer)} if esn.ready else {}
        settings = config.controller_settings
        self.controllers["rule"] = build_controller("rule", config.sample_rate,
                                                     settings.get("rule", {}))
        self.controllers["pid"] = build_controller("pid", config.sample_rate,
                                                    settings.get("pid", {}))
        warm_input = ControllerInput(
            samples_mv=np.zeros((2, config.chunk_size), dtype=np.float64),
            first_sample_index=0, sample_rate_hz=config.sample_rate, ctx_index=initial_state.ctx_index,
            validity_mask=np.ones(config.chunk_size, dtype=bool),
            stimulated_mask=np.zeros(config.chunk_size, dtype=bool),
            channel_labels=initial_state.electrode_labels,
            roles=tuple("CA3" if i != initial_state.ctx_index else "Cortex" for i in range(2)),
            latest_sample_index=config.chunk_size - 1, region="CA3", generation=-1,
            feedback=ActuatorFeedback(False, "warm-up", armed=False),
        )
        for name in ("rule", "pid"):
            controller = self.controllers[name]
            controller.step(warm_input)
            controller.reset(keep_calibration=False)
        self.controller_config = settings.get(config.controller, {})
        self.actuator = PulseActuator()
        self._last_source_end: Optional[int] = None
        self._feedback = ActuatorFeedback(True, "initial", armed=False)
        self._last_controller_generation = -1
        self._last_diagnostics: dict[str, Any] = {}
        self._stimulated_intervals: list[tuple[int, int]] = []
        self._active_controller = initial_state.controller
        self._feature_signature: Optional[tuple[Any, ...]] = None
        self._actuator_lock = threading.RLock()
        self._last_pulse_events: tuple[Any, ...] = ()
        self._last_admitted_pulse_plans: tuple[SafetyAdmittedPulsePlan, ...] = ()
        self._actuator_commits: queue.SimpleQueue[tuple[str, int]] = queue.SimpleQueue()
        self._ao_feedback: queue.SimpleQueue[tuple[int, bool, str]] = queue.SimpleQueue()
        self.last_processed_generation = state.snapshot().generation

    @property
    def last_pulse_events(self) -> tuple[Any, ...]:
        return self._last_pulse_events

    @property
    def last_admitted_pulse_plans(self) -> tuple[SafetyAdmittedPulsePlan, ...]:
        return self._last_admitted_pulse_plans

    @staticmethod
    def _envelope_feature_signature(controller: Any) -> Optional[tuple[Any, ...]]:
        envelope = getattr(controller, "envelope", None)
        if envelope is None:
            return None
        return (float(envelope.fs), tuple(float(value) for value in envelope.band_hz),
                int(envelope.window))

    def _invalidate_cached_controller_calibration(self) -> None:
        for name in ("rule", "pid"):
            controller = self.controllers.get(name)
            if controller is not None:
                controller.reset(keep_calibration=False)
        self.state.invalidate_controller_calibration()

    def commit_actuator_event(self, pulse_id: str, samples: int) -> None:
        self._actuator_commits.put((pulse_id, samples))

    def record_ao_outcome(self, generation: int, accepted: bool, reason: str) -> None:
        self._ao_feedback.put((generation, accepted, reason))

    def process(
        self,
        all_ai: np.ndarray,
        *,
        sample_index: Optional[int] = None,
        ao_target_sample_index: Optional[int] = None,
        input_is_electrode_mV: bool = False,
        raw_ai_V: Optional[np.ndarray] = None,
        input_roles: Optional[tuple[str, str]] = None,
        input_labels: Optional[tuple[str, str]] = None,
        input_ctx_index: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, float, Optional[str]]:
        if input_is_electrode_mV and input_ctx_index is not None:
            self.state.lock_input_mapping()
        snapshot = self.state.snapshot()
        self.last_processed_generation = snapshot.generation
        # Each submitted AO command owns only events proposed by this exact
        # processing block. A transition/paused block must never carry a prior
        # block's event metadata under the new generation.
        self._last_pulse_events = ()
        self._last_admitted_pulse_plans = ()
        actuator_plans: tuple[PulsePlan, ...] = ()
        while True:
            try:
                pulse_id, committed_samples = self._actuator_commits.get_nowait()
            except queue.Empty:
                break
            with self._actuator_lock:
                with contextlib.suppress(KeyError):
                    self.actuator.commit(pulse_id, committed_samples)
        while True:
            try:
                feedback_generation, accepted, reason = self._ao_feedback.get_nowait()
            except queue.Empty:
                break
            if feedback_generation == snapshot.generation:
                self._feedback = ActuatorFeedback(accepted, reason, applied_rate_hz=None,
                                                  armed=snapshot.armed)
        source_sample_index = self.sample_index if sample_index is None else int(sample_index)
        self.sample_index = source_sample_index
        command_target_index = (
            source_sample_index if ao_target_sample_index is None else int(ao_target_sample_index)
        )
        input_ctx = snapshot.ctx_index if input_ctx_index is None else int(input_ctx_index)
        labels = tuple(input_labels or snapshot.electrode_labels)
        roles = tuple(input_roles or tuple(
            "CA3" if "ca3" in label.casefold() else "Cortex" if i == input_ctx else "unknown"
            for i, label in enumerate(labels)
        ))
        feature_signature = (
            float(self.config.sample_rate), input_ctx, labels, roles,
            "electrode_mV" if input_is_electrode_mV else "ni_ai_raw_V",
            () if input_is_electrode_mV else tuple(float(gain) for gain in self.config.amplifier_gain),
            int(snapshot.calibration_epoch),
        )
        lfp = np.asarray(all_ai[:2], dtype=np.float64)
        electrode_lfp_mv = lfp.copy() if input_is_electrode_mV else lfp * (
            1_000.0 / np.asarray(self.config.amplifier_gain, dtype=np.float64)
        ).reshape(2, 1)
        block_end = source_sample_index + electrode_lfp_mv.shape[1]
        self._stimulated_intervals = [(a, b) for a, b in self._stimulated_intervals if b > source_sample_index]
        stimulated = np.zeros(electrode_lfp_mv.shape[1], dtype=bool)
        for start, stop in self._stimulated_intervals:
            lo, hi = max(start, source_sample_index), min(stop, block_end)
            if lo < hi:
                stimulated[lo-source_sample_index:hi-source_sample_index] = True
        source_context_changed = self._feature_signature is not None and feature_signature != self._feature_signature
        source_gap = (snapshot.acquiring and self._last_source_end is not None
                      and source_sample_index != self._last_source_end)
        invalid_source_sample = (snapshot.acquiring and np.any(
            ~np.isfinite(electrode_lfp_mv).all(axis=0)
        ))
        if source_context_changed or source_gap or invalid_source_sample:
            self._invalidate_cached_controller_calibration()
        recording_ai = np.asarray(raw_ai_V, dtype=np.float64) if raw_ai_V is not None else (
            lfp if not input_is_electrode_mV else np.full_like(lfp, np.nan)
        )
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
            self._last_esn_config = None
            self._active_mode = snapshot.mode

        if snapshot.generation != self._last_controller_generation and not snapshot.transition_pending:
            feature_changed = feature_signature != self._feature_signature
            outgoing = self.controllers.get(self._active_controller)
            for name, controller in self.controllers.items():
                if name == snapshot.controller:
                    continue
                controller.drain()
            prepared = self.state.take_prepared_controller(snapshot.controller)
            previous = self.controllers.get(snapshot.controller)
            if prepared is not None:
                self.controllers[snapshot.controller] = prepared
                self.controller_config = dict(snapshot.controller_settings)
            active = self.controllers.get(snapshot.controller)
            if active is not None:
                active_envelope = getattr(active, "envelope", None)
                reusable_baseline = None
                if not feature_changed and active_envelope is not None:
                    target_feature = self._envelope_feature_signature(active)
                    for candidate in (outgoing, previous):
                        candidate_envelope = getattr(candidate, "envelope", None)
                        if (candidate_envelope is not None
                                and self._envelope_feature_signature(candidate) == target_feature
                                and candidate_envelope.baseline is not None):
                            reusable_baseline = tuple(candidate_envelope.baseline)
                            break
                    if reusable_baseline is not None:
                        active_envelope.baseline = reusable_baseline
                keep_calibration = bool(
                    not feature_changed and active_envelope is not None
                    and active_envelope.baseline is not None
                )
                active.reset(keep_calibration=keep_calibration)
                if snapshot.controller in {"rule", "pid"}:
                    self.state.update_controller_calibration(snapshot.controller, keep_calibration)
            self._active_controller = snapshot.controller
            self.actuator.disarm()
            self.actuator.cancel_uncommitted()
            self.actuator.reset(preserve_refractory=True)
            self._last_controller_generation = snapshot.generation

        diagnostics: dict[str, Any] = {}
        command = np.zeros((1, self.config.chunk_size), dtype=np.float64)
        if not snapshot.transition_pending and snapshot.acquiring and snapshot.mode == "closed-loop":
            controller = self.controllers.get(snapshot.controller)
            if controller is None:
                self.state.fault(f"selected controller {snapshot.controller} is unavailable")
                return raw, command, esn_ms, "selected controller unavailable"
            if snapshot.controller == "esn":
                desired_config = (snapshot.stim_mode, snapshot.stim_gain, snapshot.pulse_threshold_std,
                                  snapshot.pulse_window_sec, snapshot.pulse_polarity)
                if desired_config != self._last_esn_config:
                    controller.configure(stim_mode=snapshot.stim_mode, stim_gain=snapshot.stim_gain,
                                         pulse_threshold_std=snapshot.pulse_threshold_std,
                                         pulse_window_sec=snapshot.pulse_window_sec,
                                         pulse_polarity=snapshot.pulse_polarity)
                    self._last_esn_config = desired_config
            valid = np.isfinite(electrode_lfp_mv).all(axis=0)
            if self._last_source_end is not None and source_sample_index != self._last_source_end:
                valid[:] = False
                controller.reset(keep_calibration=False)
            inp = ControllerInput(
                samples_mv=electrode_lfp_mv, first_sample_index=source_sample_index,
                sample_rate_hz=self.config.sample_rate,
                ctx_index=snapshot.ctx_index if input_ctx_index is None else input_ctx_index,
                validity_mask=valid, stimulated_mask=stimulated,
                channel_labels=input_labels or snapshot.electrode_labels,
                roles=input_roles or tuple("CA3" if "ca3" in label.casefold() else "Cortex" if i == snapshot.ctx_index else "unknown"
                                           for i, label in enumerate(snapshot.electrode_labels)),
                latest_sample_index=source_sample_index + self.config.chunk_size - 1,
                region="CA3", generation=snapshot.generation, feedback=dataclasses.replace(
                    self._feedback, armed=bool(snapshot.armed)),
            )
            started = time.perf_counter_ns()
            result = controller.step(inp)
            self.state.update_controller_calibration(snapshot.controller, result.diagnostics.calibrated)
            esn_ms = (time.perf_counter_ns() - started) / 1e6
            diagnostics = dataclasses.asdict(result.diagnostics)
            diagnostics["source_sample_index"] = source_sample_index
            diagnostics["latest_sample_index"] = inp.latest_sample_index
            diagnostics["action_kind"] = result.action.kind
            if snapshot.controller in {"rule", "pid"} or snapshot.stim_mode == "threshold_pulse":
                diagnostics["pulse_amplitude_V"] = snapshot.stim_gain
            if snapshot.controller == "esn":
                model_esn, threshold, peak, fired = self.esn.diagnostics(self.config.chunk_size)
                pulse_threshold.fill(threshold)
                pulse_peak.fill(peak)
                pulse_fired.fill(float(fired))
                diagnostics.update({"threshold_mV": threshold, "peak_mV": peak, "pulse_fired": bool(fired)})
            with self._actuator_lock:
                if snapshot.armed:
                    self.actuator.arm()
                else:
                    self.actuator.disarm()
                act = self.actuator.process(
                    result.action, first_sample_index=source_sample_index,
                    sample_count=self.config.chunk_size, sample_rate_hz=self.config.sample_rate,
                    armed=snapshot.armed, validity_mask=valid,
                    amplitude_v=snapshot.stim_gain, generation=snapshot.generation,
                    scheduler_lead_samples=self.config.ao_lead_samples,
                    latest_sample_index=source_sample_index + self.config.chunk_size - 1,
                )
                self._last_pulse_events = act.pulse_events
                actuator_plans = act.pulse_plans
            raw = act.waveform_v.reshape(1, -1)
            command = raw.copy()
            pulse_fired = (raw != 0).astype(np.float64)
            for event in act.pulse_events:
                if event.status in {"rendered_for_ao", "committed_tail_rendered"}:
                    start = event.target_onset_sample_index
                    stop = start + round(self.actuator.pulse_duration_ms * self.config.sample_rate / 1000.0)
                    stop += round(self.config.stimulation_recovery_ms * self.config.sample_rate / 1000.0)
                    self._stimulated_intervals.append((start, stop))
                    if source_sample_index <= start < source_sample_index + self.config.chunk_size:
                        pulse_threshold[0, start-source_sample_index] = result.diagnostics.target_z
            diagnostics.update({"requested_rate_hz": act.requested_rate_hz,
                                "rendered_pulse_count": act.rendered_count,
                                "actuator_reason": act.reason,
                                "pulse_events": [dataclasses.asdict(event) for event in act.pulse_events]})
            rejected_events = [event for event in act.pulse_events if event.status.startswith("rejected")]
            if rejected_events or not snapshot.armed:
                self._feedback = ActuatorFeedback(False,
                    rejected_events[0].status if rejected_events else "disarmed",
                    applied_rate_hz=None, armed=snapshot.armed)
        else:
            command = np.zeros_like(raw)
            self._feedback = ActuatorFeedback(False, "controller inactive", 0.0, armed=False)
        if safety_reason:
            if self._last_pulse_events:
                self._last_pulse_events = tuple(
                    dataclasses.replace(event, status="rejected_safety", rendered_sample_count=0)
                    if event.status in {"rendered_for_ao", "committed_tail_rendered"} else event
                    for event in self._last_pulse_events
                )
                diagnostics["pulse_events"] = [dataclasses.asdict(event)
                                               for event in self._last_pulse_events]
                diagnostics["rendered_pulse_count"] = 0
            self._last_admitted_pulse_plans = ()
            self.telemetry.trip()
            LOG.error("Stimulation safety trip: %s", safety_reason)
            self.state.fault(f"stimulation safety trip: {safety_reason}")
        output_permitted = (snapshot.mode == "closed-loop" and snapshot.armed
                            and not snapshot.transition_pending and snapshot.fault is None)
        if output_permitted:
            safe, safety_reason = self.safety.process(command)
        else:
            safe, safety_reason = self.safety.process(np.zeros_like(command))
            safe.fill(0.0)
        if output_permitted and not safety_reason and actuator_plans:
            events_by_id = {event.pulse_id: event for event in self._last_pulse_events}
            admitted: list[SafetyAdmittedPulsePlan] = []
            for plan in actuator_plans:
                event = events_by_id.get(plan.pulse_id)
                if event is None or event.rendered_sample_count <= 0:
                    safety_reason = f"pulse plan {plan.pulse_id} lacks a rendered proposal segment"
                    safe.fill(0.0)
                    admitted.clear()
                    break
                rendered = min(event.rendered_sample_count, plan.waveform_v.size)
                safe_tail, plan_reason = self.safety.admit_pulse_tail(plan.waveform_v[rendered:])
                if plan_reason:
                    safety_reason = f"full pulse plan failed safety admission: {plan_reason}"
                    safe.fill(0.0)
                    admitted.clear()
                    break
                safe_waveform = np.asarray(plan.waveform_v, dtype=np.float64).copy()
                offset = plan.source_onset_sample_index - source_sample_index
                safe_end = min(self.config.chunk_size, offset + rendered)
                if offset < 0 or safe_end - offset != rendered:
                    safety_reason = f"pulse plan {plan.pulse_id} segment is outside its source block"
                    safe.fill(0.0)
                    admitted.clear()
                    break
                safe_waveform[:rendered] = safe[0, offset:safe_end]
                safe_waveform[rendered:] = safe_tail.reshape(-1)
                admitted.append(SafetyAdmittedPulsePlan(
                    plan.pulse_id, plan.generation, plan.source_onset_sample_index,
                    plan.target_onset_sample_index, plan.latest_sample_index, safe_waveform,
                ))
            if not safety_reason:
                self._last_admitted_pulse_plans = tuple(admitted)
        if safety_reason:
            if self._last_pulse_events:
                self._last_pulse_events = tuple(
                    dataclasses.replace(event, status="rejected_safety", rendered_sample_count=0)
                    if event.status in {"rendered_for_ao", "committed_tail_rendered"} else event
                    for event in self._last_pulse_events
                )
                diagnostics["pulse_events"] = [dataclasses.asdict(event)
                                               for event in self._last_pulse_events]
                diagnostics["rendered_pulse_count"] = 0
            self._last_admitted_pulse_plans = ()
            self.telemetry.trip()
            LOG.error("Stimulation safety trip: %s", safety_reason)
            self.state.fault(f"stimulation safety trip: {safety_reason}")
        if safety_reason or not output_permitted:
            self._feedback = ActuatorFeedback(False, safety_reason or "disarmed or draining",
                                              applied_rate_hz=None, armed=False)
        elif not any(event.status.startswith("rejected") for event in self._last_pulse_events):
            self._feedback = ActuatorFeedback(True, "request accepted pending AO scheduling",
                                              applied_rate_hz=None, armed=True)
        if snapshot.acquiring:
            self._last_source_end = source_sample_index + self.config.chunk_size

        if snapshot.recording and snapshot.acquiring:
            self.recorder.submit(RecordBlock(
                source_sample_index, command_target_index, recording_ai.copy(), raw.copy(), model_esn.copy(),
                pulse_threshold.copy(), pulse_peak.copy(), pulse_fired.copy(), safe.copy(),
                actual_stim.copy() if actual_stim is not None else None,
                snapshot.armed,
                RECORDED_MODE_VALUES[(snapshot.mode, snapshot.stim_mode)],
                electrode_lfp_mv.copy(), snapshot.controller, snapshot.generation, diagnostics,
            ))
        self._last_diagnostics = diagnostics
        self._feature_signature = feature_signature

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
        self._input_source_kind = ("electrode_mV" if input_is_electrode_mV else "ni_ai_raw_V")
        self._publish_ui(electrode_lfp_mv, safe, preview, threshold_preview, snapshot, safety_reason)
        self.sample_index = source_sample_index + self.config.chunk_size
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
    sample_indices: np.ndarray
    snapshot: StateSnapshot
    safety_reason: Optional[str]
    controller_diagnostics: dict[str, Any]
    input_source_kind: str


RULE_CONTROLLER_DEFAULTS: Final = {"upper_z": 3.0, "lower_z": 1.5, "band_hz": (100.0, 250.0)}
PID_CONTROLLER_DEFAULTS: Final = {"kp": 0.0, "ki": 0.0, "kd": 0.0, "target_z": 0.0,
    "direction": 1, "max_rate_hz": 2.0, "update_ms": 5.0, "derivative_tau_s": 0.020,
    "band_hz": (100.0, 250.0)}


def controller_settings_for(controller: str, supplied: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if controller not in {"rule", "pid"}:
        raise ValueError("settings are available only for Rule and PID controllers")
    defaults = RULE_CONTROLLER_DEFAULTS if controller == "rule" else PID_CONTROLLER_DEFAULTS
    value = dict(defaults)
    if supplied:
        unknown = set(supplied) - set(defaults)
        if unknown:
            raise ValueError(f"unknown {controller} settings: {sorted(unknown)}")
        value.update(supplied)
    if "band_hz" in value:
        band = value["band_hz"]
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            raise ValueError("band_hz must contain low and high frequencies")
        value["band_hz"] = tuple(float(v) for v in band)
    return value


def complete_controller_settings(controller: str, supplied: dict[str, Any]) -> dict[str, Any]:
    if controller not in {"rule", "pid"}:
        raise ValueError("settings are available only for Rule and PID controllers")
    defaults = RULE_CONTROLLER_DEFAULTS if controller == "rule" else PID_CONTROLLER_DEFAULTS
    missing = set(defaults) - set(supplied)
    if missing:
        raise ValueError(f"controller settings transaction is incomplete; missing {sorted(missing)}")
    return controller_settings_for(controller, supplied)


def build_controller(controller: str, sample_rate: float, settings: Optional[dict[str, Any]] = None) -> Any:
    values = controller_settings_for(controller, settings)
    return RuleController(sample_rate, **values) if controller == "rule" else PIDController(sample_rate, **values)


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
        self.ui_sample_indices.append(
            np.arange(self.sample_index, self.sample_index + ai.shape[1], dtype=np.uint64)
        )
        now = time.perf_counter_ns()
        if (now - self.last_ui_ns) / 1e9 < self.config.ui_interval_s:
            return
        job = UiJob(
            np.hstack(self.ui_ai), np.hstack(self.ui_ao), np.hstack(self.ui_preview),
            np.hstack(self.ui_threshold), np.hstack(self.ui_sample_indices), snapshot, safety_reason,
            dict(self._last_diagnostics), self._input_source_kind,
        )
        self.ui_ai.clear()
        self.ui_ao.clear()
        self.ui_preview.clear()
        self.ui_threshold.clear()
        self.ui_sample_indices.clear()
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
                telemetry = self.telemetry.snapshot()
                telemetry["recorder_submit_wait"] = self.recorder.submit_wait_snapshot()
                packet = json.dumps(json_safe({
                    "ai": job.ai[:, ::self.config.visual_downsample].tolist(),
                    "ao": job.ao[:, ::self.config.visual_downsample].tolist(),
                    "passthrough_preview": ui_series(job.preview, self.config.visual_downsample),
                    "pulse_threshold_preview": ui_series(job.threshold, self.config.visual_downsample),
                    "sample_indices": job.sample_indices[::self.config.visual_downsample].tolist(),
                    "mode": snapshot.mode,
                    "is_recording": snapshot.recording,
                    "is_acquiring": snapshot.acquiring,
                    "stim_mode": snapshot.stim_mode,
                    "stim_gain": snapshot.stim_gain,
                    "pulse_amplitude_V": (snapshot.stim_gain if snapshot.controller in {"rule", "pid"}
                                           or snapshot.stim_mode == "threshold_pulse" else None),
                    "max_pulse_amplitude_V": pulse_amplitude_limit_v(self.config.safety,
                                                                      self.config.sample_rate,
                                                                      pulse_duration_ms=self.actuator.pulse_duration_ms,
                                                                      pulse_frequency_hz=self.actuator.pulse_frequency_hz,
                                                                      refractory_s=self.actuator.refractory_s),
                    "pulse_threshold_std": snapshot.pulse_threshold_std,
                    "pulse_window_sec": snapshot.pulse_window_sec,
                    "pulse_polarity": snapshot.pulse_polarity,
                    "armed": snapshot.armed,
                    "fs": self.config.sample_rate / self.config.visual_downsample,
                    "sample_rate_hz": self.config.sample_rate,
                    "fault": snapshot.fault,
                    "esn_ready": snapshot.esn_ready,
                    "controller": snapshot.controller,
                    "controller_ready": snapshot.controller_ready,
                    "controller_calibrated": snapshot.controller_calibrated,
                    "transition_pending": snapshot.transition_pending,
                    "controller_settings": dict(snapshot.controller_settings),
                    "config_generation": snapshot.generation,
                    "input_mapping_locked": snapshot.input_mapping_locked,
                    "controller_diagnostics": job.controller_diagnostics,
                    "safety_trip": job.safety_reason,
                    "telemetry": telemetry,
                    "channels": list(snapshot.electrode_labels),
                    "cortex_ai": f"ai{snapshot.ctx_index}",
                    "ao_is_command": True,
                    "ai_unit": "mV",
                    "ao_unit": "V",
                    "amplifier_gain": list(self.config.amplifier_gain),
                    "ao_command_gain_v_per_esn_unit": self.config.ao_command_gain_v_per_esn_unit,
                    "input_source_kind": job.input_source_kind,
                }), allow_nan=False)
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
    active_demand_misses: int = 0
    pulse_tail_misses: int = 0
    fault_rejections: int = 0
    cancelled_blocks: int = 0
    min_queued_samples: Optional[int] = None
    max_write_ms: float = 0.0
    max_loop_pause_ms: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def submitted(self, late: bool) -> None:
        with self.lock:
            self.submitted_blocks += 1
            self.late_commands += int(late)

    def fault_rejected(self) -> None:
        with self.lock:
            self.fault_rejections += 1

    def cancelled(self) -> None:
        with self.lock:
            self.cancelled_blocks += 1

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
                "active_demand_misses": self.active_demand_misses,
                "pulse_tail_misses": self.pulse_tail_misses,
                "fault_rejections": self.fault_rejections,
                "cancelled_blocks": self.cancelled_blocks,
                "min_queued_samples": self.min_queued_samples,
                "max_write_ms": self.max_write_ms,
                "max_loop_pause_ms": self.max_loop_pause_ms,
            }


class AoDrainBarrier:
    """Tracks successfully written command samples and counter-based drain completion."""

    def __init__(self) -> None:
        self.last_nonzero_end = 0
        self.last_zero_commit_end = 0
        self.pending_generation: Optional[int] = None
        self.end_sample = 0

    def committed(self, first_sample: int, blocks: list[np.ndarray], chunk_size: int) -> None:
        end = first_sample + len(blocks) * chunk_size
        all_zero = True
        for offset, block in enumerate(blocks):
            nonzero = np.flatnonzero(np.asarray(block) != 0.0)
            if nonzero.size:
                all_zero = False
                self.last_nonzero_end = max(self.last_nonzero_end,
                                            first_sample + offset * chunk_size + int(nonzero[-1]) + 1)
        if all_zero:
            self.last_zero_commit_end = end

    def begin(self, generation: int, generated_sample_count: int) -> None:
        self.pending_generation = generation
        self.end_sample = max(generated_sample_count, self.last_nonzero_end)

    def preserve_until(self, exclusive_end: int) -> None:
        self.end_sample = max(self.end_sample, int(exclusive_end))

    def drained(self, generation: int, generated_sample_count: int) -> bool:
        # ``generated_sample_count`` is the next sample index to be generated.
        # Reaching exclusive_end means only samples through exclusive_end - 1
        # have left the device buffer; one additional generated sample proves
        # the first post-output sample (zero) has actually advanced.
        barrier_end = self.end_sample + 1
        return (self.pending_generation == generation
                and generated_sample_count >= barrier_end
                and self.last_zero_commit_end >= barrier_end)

    def acknowledge(self) -> None:
        self.pending_generation = None


class AoCommandLifecycle:
    """Shared reservation, generation cancellation and pulse-tail policy for NI and mock AO."""

    def __init__(self) -> None:
        self.commands: dict[int, tuple[np.ndarray, int, tuple[Any, ...]]] = {}
        self.committed_pulse_samples: dict[str, int] = {}
        self.pulse_exclusive_ends: dict[str, int] = {}

    def submit(self, target: int, waveform: np.ndarray, generation: int,
               events: tuple[Any, ...] = (), *, chunk_size: int | None = None,
               pulse_plans: tuple[SafetyAdmittedPulsePlan, ...] = ()) -> None:
        value = np.asarray(waveform, dtype=np.float64).copy()
        events = tuple(event for event in events
                       if event.status in {"rendered_for_ao", "committed_tail_rendered"}
                       and event.rendered_sample_count > 0)
        existing = self.commands.get(target)
        if existing is not None and existing[1] != generation and any(
                self.committed_pulse_samples.get(event.pulse_id, 0) > 0 for event in existing[2]):
            # Never replace a staged old-generation tail after its pulse has
            # started. Its complete safe plan owns these original AO samples.
            return
        self.commands[target] = (value, generation, tuple(events))
        if pulse_plans and (chunk_size is None or chunk_size <= 0):
            raise ValueError("chunk_size is required to stage complete pulse plans")
        for plan in pulse_plans:
            cursor = 0
            while cursor < plan.waveform_v.size:
                absolute = plan.target_onset_sample_index + cursor
                block, offset = divmod(absolute, int(chunk_size))
                count_samples = min(int(chunk_size) - offset, plan.waveform_v.size - cursor)
                old = self.commands.get(block)
                if old is not None and old[1] != plan.generation and any(
                        self.committed_pulse_samples.get(event.pulse_id, 0) > 0 for event in old[2]):
                    cursor += count_samples
                    continue
                if old is None or old[1] != plan.generation:
                    block_value = np.zeros(int(chunk_size), dtype=np.float64)
                    block_events: tuple[Any, ...] = ()
                else:
                    block_value = old[0].copy()
                    block_events = old[2]
                block_value[offset:offset + count_samples] = plan.waveform_v[cursor:cursor + count_samples]
                block_events = tuple(event for event in block_events if event.pulse_id != plan.pulse_id)
                block_events += (PulseEvent(
                    plan.pulse_id, plan.source_onset_sample_index,
                    plan.source_onset_sample_index, plan.target_onset_sample_index,
                    "rendered_for_ao", plan.generation, count_samples, plan.latest_sample_index,
                ),)
                self.commands[block] = (block_value, plan.generation, block_events)
                cursor += count_samples

    def switch(self, old_generation: int) -> tuple[list[tuple[int, tuple[np.ndarray, int, tuple[Any, ...]]]],
                                                   list[tuple[int, tuple[np.ndarray, int, tuple[Any, ...]]]]]:
        """Keep tails for pulses with AO committed samples at their original target indices."""
        cancelled = []
        preserved = []
        for target, entry in tuple(self.commands.items()):
            if entry[1] > old_generation:
                continue
            waveform, generation, events = entry
            keep_events = tuple(event for event in events
                                if self.committed_pulse_samples.get(event.pulse_id, 0) > 0)
            if keep_events:
                self.commands[target] = (waveform, generation, keep_events)
                preserved.append((target, self.commands[target]))
            else:
                self.commands.pop(target, None)
                cancelled.append((target, entry))
        return cancelled, preserved

    def eligible(self, entry: tuple[np.ndarray, int, tuple[Any, ...]], generation: int,
                 transition_pending: bool) -> bool:
        if entry[1] == generation:
            return True
        return bool(transition_pending and entry[2] and all(
            self.committed_pulse_samples.get(event.pulse_id, 0) > 0 for event in entry[2]
        ))

    def has_committed_pulse(self, events: tuple[Any, ...]) -> bool:
        return any(self.committed_pulse_samples.get(event.pulse_id, 0) > 0 for event in events)

    @staticmethod
    def missing_refill_is_urgent(first_target: int, available_blocks: int,
                                 generated_samples: int, chunk_size: int,
                                 deadline_samples: int) -> bool:
        """Whether the first unavailable target would cross the safe queue deadline.

        A block's input is available only after its last sample has arrived and
        processing has submitted its AO command. Do not write an irreversible
        zero for a farther-future block just because the scheduler refills a
        batch at the half-lead watermark.
        """
        queued_after_available = ((first_target + available_blocks) * chunk_size
                                  - generated_samples)
        return queued_after_available <= deadline_samples

    def committed(self, events: tuple[Any, ...], sample_rate: int) -> None:
        for event in events:
            if event.rendered_sample_count:
                self.committed_pulse_samples[event.pulse_id] = (
                    self.committed_pulse_samples.get(event.pulse_id, 0) + event.rendered_sample_count)
                self.pulse_exclusive_ends[event.pulse_id] = (
                    event.target_onset_sample_index + round(5.0 * sample_rate / 1000.0))


@dataclass(frozen=True)
class GeneratedAoBlock:
    sample_index: int
    waveform_v: np.ndarray
    generation: int
    stage: str


class SimulatedAoScheduler:
    """Deterministic AO clock and generation barrier shared by mock replay and benchmarks."""

    def __init__(self, config: AppConfig, state: RuntimeState,
                 recorder: Optional[H5Recorder] = None, commit_event: Any = None,
                 ao_outcome: Any = None) -> None:
        self.config, self.state, self.recorder, self.commit_event = config, state, recorder, commit_event
        self.ao_outcome = ao_outcome
        self.lifecycle = AoCommandLifecycle()
        self.commands = self.lifecycle.commands
        self.lock = threading.RLock()
        self.generated_sample_count = 0
        self._elapsed_samples = 0.0
        self._next_write_block = config.ao_lead_chunks
        self._committed_output: dict[int, tuple[np.ndarray, int, tuple[Any, ...], str]] = {}
        self._generation = state.snapshot().generation
        self.drain_barrier = AoDrainBarrier()
        self.drain_barrier.last_zero_commit_end = config.ao_lead_samples
        self._committed_pulse_samples = self.lifecycle.committed_pulse_samples
        self._pulse_exclusive_ends = self.lifecycle.pulse_exclusive_ends
        self.stats = {"submitted_blocks": 0, "committed_blocks": 0, "zero_substitutions": 0,
                      "late_commands": 0, "rejected_stale": 0, "cancelled_blocks": 0,
                      "rejected_fault": 0,
                      "active_demand_misses": 0,
                      "pulse_tail_misses": 0,
                      "max_queue_depth": 0, "pulse_requests": 0, "pulse_committed_samples": 0}

    def submit(self, target_block: int, command: np.ndarray, generation: Optional[int] = None,
               pulse_events: tuple[Any, ...] = (),
               pulse_plans: tuple[SafetyAdmittedPulsePlan, ...] = ()) -> bool:
        value = np.asarray(command, dtype=np.float64).reshape(-1)
        if value.size != self.config.chunk_size:
            raise ValueError(f"AO command has {value.size} samples; expected {self.config.chunk_size}")
        current = self.state.snapshot()
        generation = current.generation if generation is None else int(generation)
        with self.lock:
            self.stats["submitted_blocks"] += 1
            if current.fault is not None:
                self.stats["rejected_fault"] += 1
                record_sparse_ao_outcome(self.recorder, sample_index=target_block * self.config.chunk_size,
                    generation=generation, stage="rejected", controller=current.controller,
                    waveform=value, events=pulse_events, reason="latched fault rejected AO command")
                if self.ao_outcome is not None:
                    self.ao_outcome(generation, False, "latched fault rejected AO command")
                return False
            if generation != current.generation:
                self.stats["rejected_stale"] += 1
                record_sparse_ao_outcome(self.recorder, sample_index=target_block * self.config.chunk_size,
                    generation=generation, stage="rejected", controller=current.controller,
                    waveform=value, events=pulse_events, reason="stale generation rejected AO command")
                return False
            if target_block < self._next_write_block:
                self.stats["late_commands"] += 1
                record_sparse_ao_outcome(self.recorder, sample_index=target_block * self.config.chunk_size,
                    generation=generation, stage="late", controller=current.controller,
                    waveform=value, events=pulse_events, reason="target block already passed")
                return False
            self.lifecycle.submit(target_block, value, generation, pulse_events,
                                  chunk_size=self.config.chunk_size, pulse_plans=pulse_plans)
            self.stats["pulse_requests"] += sum(event.status == "rendered_for_ao" for event in pulse_events)
            self.stats["max_queue_depth"] = max(self.stats["max_queue_depth"], len(self.commands))
            return True

    def advance(self, *, service_refill: bool = True) -> GeneratedAoBlock:
        """Commit and generate exactly one AO block, returning hardware-indexed output."""
        with self.lock:
            chunk = self.config.chunk_size
            sample = self.generated_sample_count
            target = sample // chunk
            snapshot = self.state.snapshot()
            committed = self._committed_output.pop(target, None)
            if committed is None:
                output = np.zeros(chunk, dtype=np.float64)
                stage = "initial_zero" if sample < self.config.ao_lead_samples else "zero_substituted"
                generated_generation, events = snapshot.generation, ()
                if stage == "zero_substituted":
                    self.stats["zero_substitutions"] += 1
                    if self.stats["active_demand_misses"] == 0:
                        self.stats["active_demand_misses"] += 1
                        self.state.fault("simulated AO host refill stalled past committed output")
            else:
                output, generation, events, stage = committed
                generated_generation = generation
            self.generated_sample_count += chunk
            if service_refill:
                self._service_refill_locked()
            if service_refill and self.drain_barrier.drained(snapshot.generation, self.generated_sample_count):
                record_drain_ack(self.recorder, sample_index=self.generated_sample_count,
                    generation=snapshot.generation, controller=snapshot.controller,
                    end_sample=self.drain_barrier.end_sample)
                self.state.acknowledge_drain(snapshot.generation)
                self.drain_barrier.acknowledge()
            return GeneratedAoBlock(sample, output.copy(), generated_generation, stage)

    def _service_refill_locked(self) -> None:
        """Perform one host-side refill after AO time has advanced."""
        chunk = self.config.chunk_size
        snapshot = self.state.snapshot()
        generated = self.generated_sample_count
        if snapshot.generation != self._generation:
            old = self._generation
            self._generation = snapshot.generation
            self.drain_barrier.begin(snapshot.generation, generated)
            for pulse_id, committed in self._committed_pulse_samples.items():
                if committed:
                    self.drain_barrier.preserve_until(self._pulse_exclusive_ends[pulse_id])
            cancelled, preserved = self.lifecycle.switch(old)
            self.stats["cancelled_blocks"] += len(cancelled)
            for _, (_, _, events) in preserved:
                for event in events:
                    self.drain_barrier.preserve_until(self._pulse_exclusive_ends[event.pulse_id])
            if self.recorder is not None:
                for stage, groups in (("cancelled", cancelled), ("tail_preserved", preserved)):
                    for block, (waveform, _, events) in groups:
                        record_sparse_ao_outcome(self.recorder, sample_index=block * chunk,
                            generation=old, stage=stage, controller=snapshot.controller,
                            waveform=waveform, events=events,
                            reason="generation changed before AO write")
        if snapshot.fault is not None and self.commands:
            pending_fault_commands = tuple(self.commands.items())
            self.commands.clear()
            self.stats["cancelled_blocks"] += len(pending_fault_commands)
            for block, entry in pending_fault_commands:
                if block < math.ceil(generated / chunk):
                    self.stats["late_commands"] += 1
                if self.lifecycle.has_committed_pulse(entry[2]):
                    self.stats["pulse_tail_misses"] += 1
                    for event in entry[2]:
                        if self.recorder is not None:
                            self.recorder.record_action_stage(
                                sample_index=event.target_onset_sample_index,
                                generation=event.generation, stage="tail_missed",
                                controller=snapshot.controller, request_id=event.pulse_id,
                                source_sample_index=event.source_onset_sample_index,
                                latest_sample_index=event.latest_sample_index,
                                target_sample_index=event.target_onset_sample_index,
                                reason="latched fault cancelled a partly committed pulse tail")
                if self.ao_outcome is not None:
                    self.ao_outcome(entry[1], False, "latched fault cancelled pending AO command")
                record_sparse_ao_outcome(self.recorder, sample_index=block * chunk,
                    generation=entry[1], stage="cancelled", controller=snapshot.controller,
                    waveform=entry[0], events=entry[2],
                    reason="latched fault cancelled pending AO command")
        frontier = math.ceil(generated / chunk)
        if self._next_write_block < frontier:
            # A starved host cannot retroactively fill target blocks already
            # generated as zeros. Retire those commands as late and continue
            # from the current hardware sample frontier.
            for block in tuple(self.commands):
                if block < frontier:
                    entry = self.commands.pop(block)
                    self.stats["late_commands"] += 1
                    missed_committed_tail = self.lifecycle.has_committed_pulse(entry[2])
                    if missed_committed_tail:
                        self.stats["pulse_tail_misses"] += 1
                        self.state.fault("partly committed pulse tail missed its original AO samples")
                        if self.recorder is not None:
                            for event in entry[2]:
                                if self.lifecycle.committed_pulse_samples.get(event.pulse_id, 0) > 0:
                                    self.recorder.record_action_stage(
                                        sample_index=event.target_onset_sample_index,
                                        generation=event.generation, stage="tail_missed",
                                        controller=snapshot.controller, request_id=event.pulse_id,
                                        source_sample_index=event.source_onset_sample_index,
                                        latest_sample_index=event.latest_sample_index,
                                        target_sample_index=event.target_onset_sample_index,
                                        reason="host stall elapsed the original target of a committed pulse tail")
                    if self.ao_outcome is not None:
                        self.ao_outcome(entry[1], False,
                                        "target samples elapsed before the host scheduler serviced AO")
                    record_sparse_ao_outcome(self.recorder, sample_index=block * chunk,
                        generation=entry[1], stage="late", controller=snapshot.controller,
                        waveform=entry[0], events=entry[2],
                        reason="target samples elapsed before the host scheduler serviced AO")
            self._next_write_block = frontier
        queued = self._next_write_block * chunk - generated
        deficit = max(0, math.ceil((self.config.ao_lead_samples - queued) / chunk))
        if not deficit or queued > self.config.ao_deadline_samples:
            if self.drain_barrier.drained(snapshot.generation, generated):
                record_drain_ack(self.recorder, sample_index=generated,
                    generation=snapshot.generation, controller=snapshot.controller,
                    end_sample=self.drain_barrier.end_sample)
                self.state.acknowledge_drain(snapshot.generation)
                self.drain_barrier.acknowledge()
            return

        reservations = []
        for block in range(self._next_write_block, self._next_write_block + deficit):
            pending = self.commands.get(block)
            if pending is None or not self.lifecycle.eligible(
                    pending, snapshot.generation, snapshot.transition_pending):
                break
            self.commands.pop(block, None)
            reservations.append((block, pending))
        missing = deficit - len(reservations)
        producer_running = (snapshot.mode == "closed-loop" and snapshot.acquiring
                            and not snapshot.transition_pending and snapshot.fault is None)
        urgent_missing = (missing > 0 and self.lifecycle.missing_refill_is_urgent(
            self._next_write_block, len(reservations), generated, chunk,
            self.config.ao_deadline_samples))
        active_output = producer_running and snapshot.armed and snapshot.stim_mode != "off"
        active_miss = urgent_missing and active_output
        if active_miss:
            self.stats["active_demand_misses"] += 1
            self.state.fault("simulated AO active deadline missed; refill replaced with zeros")
        defer_missing = missing > 0 and producer_running and not urgent_missing
        if defer_missing:
            # Commit only the contiguous submitted prefix. The next processing
            # block can fill the horizon before it is urgent.
            refill = reservations
        else:
            refill = reservations + [(block, None) for block in
                                     range(self._next_write_block + len(reservations),
                                           self._next_write_block + deficit)]

        for block, pending in refill:
            original_pending = pending
            if self.state.snapshot().fault is not None:
                if pending is not None:
                    self.stats["cancelled_blocks"] += 1
                    if self.ao_outcome is not None:
                        self.ao_outcome(pending[1], False, "latched fault cancelled pending AO command")
                command = np.zeros(chunk, dtype=np.float64)
                generation, events, action_stage = snapshot.generation, (), "zero_substituted"
                self.stats["zero_substitutions"] += 1
                pending = None
            elif pending is not None and not active_miss:
                command, generation, events = pending
                action_stage = "tail_preserved" if generation != snapshot.generation else "ao_committed"
            elif pending is not None:
                _, generation, events = pending
                command = np.zeros(chunk, dtype=np.float64)
                action_stage = "zero_substituted"
            else:
                command = np.zeros(chunk, dtype=np.float64)
                generation, events, action_stage = snapshot.generation, (), "zero_substituted"
                if not active_miss:
                    self.stats["zero_substitutions"] += 1
            self._committed_output[block] = (command.copy(), generation, tuple(events), action_stage)
            self.drain_barrier.committed(block * chunk, [command], chunk)
            self._next_write_block += 1
            self.stats["committed_blocks"] += 1
            if action_stage != "zero_substituted":
                self.lifecycle.committed(events, self.config.sample_rate)
            for event in events if action_stage != "zero_substituted" else ():
                if event.rendered_sample_count:
                    self.stats["pulse_committed_samples"] += event.rendered_sample_count
                    self.drain_barrier.preserve_until(self._pulse_exclusive_ends[event.pulse_id])
                    if self.commit_event is not None:
                        self.commit_event(event.pulse_id, event.rendered_sample_count)
            if self.ao_outcome is not None:
                self.ao_outcome(generation, action_stage in {"ao_committed", "tail_preserved"}, action_stage)
            if action_stage != "zero_substituted":
                record_sparse_ao_outcome(self.recorder, sample_index=block * chunk,
                    generation=generation, stage=action_stage, controller=snapshot.controller,
                    waveform=command, events=events, reason="simulated AO write committed")
            elif active_miss or self.state.snapshot().fault is not None:
                original_events = original_pending[2] if original_pending is not None else ()
                original_waveform = original_pending[0] if original_pending is not None else None
                terminal_stage = ("zero_substituted" if active_miss else
                                  "cancelled" if original_pending is not None else "zero_substituted")
                record_sparse_ao_outcome(self.recorder, sample_index=block * chunk,
                    generation=generation, stage=terminal_stage, controller=snapshot.controller,
                    waveform=original_waveform, events=original_events,
                    reason="faulted AO refill committed as safe zero")
        if self.drain_barrier.drained(snapshot.generation, generated):
            record_drain_ack(self.recorder, sample_index=generated,
                generation=snapshot.generation, controller=snapshot.controller,
                end_sample=self.drain_barrier.end_sample)
            self.state.acknowledge_drain(snapshot.generation)
            self.drain_barrier.acknowledge()
    def advance_elapsed(self, elapsed_seconds: float) -> tuple[GeneratedAoBlock, ...]:
        """Consume elapsed AO time first, then service the host writer once.

        This models an OS-stalled AO host: it cannot perform historical refills
        during a period in which it was not scheduled. The hardware sample clock
        still consumes only blocks that were committed before the stall.
        """
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        with self.lock:
            self._elapsed_samples += elapsed_seconds * self.config.sample_rate
            block_count = int(self._elapsed_samples // self.config.chunk_size)
            self._elapsed_samples -= block_count * self.config.chunk_size
        generated = tuple(self.advance(service_refill=False) for _ in range(block_count))
        with self.lock:
            self._service_refill_locked()
        return generated

    def stats_snapshot(self) -> dict[str, Any]:
        return {**self.stats, "queue_depth": len(self.commands),
                "generated_sample_count": self.generated_sample_count,
                "barrier_generation": self.drain_barrier.pending_generation,
                "barrier_end_sample": self.drain_barrier.end_sample}


class ScheduledAo:
    """Sole owner of a non-regenerating AO task with proactive safe refills."""

    def __init__(self, config: AppConfig, state: RuntimeState, trigger_terminal: str,
                 recorder: Optional[H5Recorder] = None, commit_event: Any = None,
                 ao_outcome: Any = None) -> None:
        self.config = config
        self.state = state
        self.trigger_terminal = trigger_terminal
        self.recorder = recorder
        self.commit_event = commit_event
        self.ao_outcome = ao_outcome
        self.lifecycle = AoCommandLifecycle()
        self.commands = self.lifecycle.commands
        self.command_lock = threading.Lock()
        self.ready = threading.Event()
        self.clock_started = threading.Event()
        self.stop_requested = threading.Event()
        self.force_safe = threading.Event()
        self.error: Optional[Exception] = None
        self.stats = AoScheduleStats()
        self._next_block = 0
        self._observed_generation = 0
        self.drain_barrier = AoDrainBarrier()
        self._committed_pulse_samples = self.lifecycle.committed_pulse_samples
        self._pulse_exclusive_ends = self.lifecycle.pulse_exclusive_ends
        self.thread = threading.Thread(target=self._run, name="ao-writer", daemon=True)

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(10.0):
            raise RuntimeError("AO scheduler did not become ready")
        if self.error:
            raise RuntimeError(f"AO scheduler setup failed: {self.error}") from self.error

    def notify_clock_started(self) -> None:
        self.clock_started.set()

    def submit(self, target_block: int, command: np.ndarray, generation: Optional[int] = None,
               pulse_events: tuple[Any, ...] = (),
               pulse_plans: tuple[SafetyAdmittedPulsePlan, ...] = ()) -> bool:
        value = np.asarray(command, dtype=np.float64).reshape(-1)
        if value.size != self.config.chunk_size:
            raise ValueError(f"AO command has {value.size} samples; expected {self.config.chunk_size}")
        with self.command_lock:
            generation = self.state.snapshot().generation if generation is None else int(generation)
            late = target_block < self._next_block
            self.stats.submitted(late)
            snapshot = self.state.snapshot()
            if snapshot.fault is not None:
                self.stats.fault_rejected()
                record_sparse_ao_outcome(self.recorder, sample_index=target_block * self.config.chunk_size,
                    generation=generation, stage="rejected", controller=snapshot.controller,
                    waveform=value, events=pulse_events, reason="latched fault rejected AO command")
                if self.ao_outcome is not None:
                    self.ao_outcome(generation, False, "latched fault rejected AO command")
                return False
            if not late and not self.force_safe.is_set() and generation == snapshot.generation:
                self.lifecycle.submit(target_block, value, generation, pulse_events,
                                      chunk_size=self.config.chunk_size, pulse_plans=pulse_plans)
                return True
            record_sparse_ao_outcome(self.recorder, sample_index=target_block * self.config.chunk_size,
                generation=generation, stage="late" if late else "rejected",
                controller=snapshot.controller, waveform=value, events=pulse_events,
                reason="AO target elapsed, generation changed, or scheduler is forced safe")
            return False

    def request_safe_zero(self) -> None:
        self.force_safe.set()
        with self.command_lock:
            cancelled = tuple(self.commands.items())
            self.commands.clear()
        for target, entry in cancelled:
            self.stats.cancelled()
            record_sparse_ao_outcome(self.recorder, sample_index=target * self.config.chunk_size,
                generation=entry[1], stage="cancelled", controller=self.state.snapshot().controller,
                waveform=entry[0], events=entry[2],
                reason="latched fault or stop cancelled pending AO command")
            if self.ao_outcome is not None:
                self.ao_outcome(entry[1], False, "latched fault or stop cancelled pending AO command")

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

    def _take_contiguous(self, first_target: int, maximum: int, generation: int) -> list[tuple[int, np.ndarray, int, tuple[Any, ...]]]:
        """Reserve current-generation commands without removing them before write success."""
        if self.force_safe.is_set():
            return []
        with self.command_lock:
            blocks: list[tuple[int, np.ndarray, int, tuple[Any, ...]]] = []
            for target in range(first_target, first_target + maximum):
                entry = self.commands.get(target)
                if entry is None:
                    break
                if not self.lifecycle.eligible(entry, generation, self.state.snapshot().transition_pending):
                    break
                blocks.append((target, entry[0], entry[1], entry[2]))
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
                    snapshot = self.state.snapshot()
                    if snapshot.fault is not None and not self.force_safe.is_set():
                        # A latched fault is stronger than a recoverable mode
                        # transition: discard all not-yet-written commands,
                        # including whole-pulse plans and preserved tails.
                        self.request_safe_zero()
                    if snapshot.generation != self._observed_generation:
                        old_generation = self._observed_generation
                        self._observed_generation = snapshot.generation
                        self.drain_barrier.begin(snapshot.generation, generated)
                        for pulse_id, committed in self._committed_pulse_samples.items():
                            if committed:
                                self.drain_barrier.preserve_until(self._pulse_exclusive_ends[pulse_id])
                        with self.command_lock:
                            cancelled, preserved = self.lifecycle.switch(old_generation)
                        for _, (_, _, events) in preserved:
                            for event in events:
                                self.drain_barrier.preserve_until(self._pulse_exclusive_ends[event.pulse_id])
                        if self.recorder is not None:
                            for stage, groups in (("cancelled", cancelled), ("tail_preserved", preserved)):
                                for target_block, (waveform, _, events) in groups:
                                    record_sparse_ao_outcome(self.recorder,
                                        sample_index=target_block * chunk, generation=old_generation,
                                        stage=stage, controller=snapshot.controller, waveform=waveform,
                                        events=events, reason="generation transition before AO output")
                    generated_frontier = math.ceil(generated / chunk)
                    if self._next_block < generated_frontier:
                        with self.command_lock:
                            elapsed_targets = []
                            for target in tuple(self.commands):
                                if target < generated_frontier:
                                    elapsed_targets.append((target, self.commands.pop(target)))
                            self._next_block = generated_frontier
                        missed_partial_tail = any(
                            self.lifecycle.has_committed_pulse(entry[2]) for _, entry in elapsed_targets
                        )
                        active_host_miss = (
                            snapshot.mode == "closed-loop" and snapshot.acquiring
                            and not snapshot.transition_pending and snapshot.fault is None
                        )
                        with self.stats.lock:
                            self.stats.late_commands += len(elapsed_targets)
                            self.stats.pulse_tail_misses += int(missed_partial_tail)
                            self.stats.active_demand_misses += int(active_host_miss)
                        reason = ("partly committed pulse tail missed its original AO samples"
                                  if missed_partial_tail else "AO host refill stalled past committed output")
                        for target_block, entry in elapsed_targets:
                            _, generation, events = entry
                            if self.ao_outcome is not None:
                                self.ao_outcome(generation, False, reason)
                            record_sparse_ao_outcome(self.recorder,
                                sample_index=target_block * chunk, generation=generation,
                                stage="tail_missed" if missed_partial_tail else "late",
                                controller=snapshot.controller, waveform=entry[0], events=events,
                                reason=reason)
                        if missed_partial_tail or active_host_miss:
                            self.request_safe_zero()
                            self.state.fault(reason)
                    queued = self._next_block * chunk - generated
                    deficit_blocks = max(0, math.ceil((lead_samples - queued) / chunk))
                    if deficit_blocks and (queued <= deadline_samples or self.force_safe.is_set()):
                        first_target = self._next_block
                        reserved = self._take_contiguous(first_target, deficit_blocks, snapshot.generation)
                        refill = [value for _, value, _, _ in reserved]
                        missing_count = deficit_blocks - len(refill)
                        zero_count = missing_count
                        latest_state = self.state.snapshot()
                        producer_running = (
                            not self.force_safe.is_set()
                            and latest_state.generation == snapshot.generation
                            and not latest_state.transition_pending
                            and latest_state.acquiring
                            and latest_state.mode == "closed-loop"
                        )
                        active_output = (producer_running and latest_state.armed
                                         and latest_state.stim_mode != "off")
                        urgent_missing = (
                            missing_count > 0
                            and self.lifecycle.missing_refill_is_urgent(
                                first_target, len(reserved), generated, chunk, deadline_samples)
                        )
                        active_miss = (
                            urgent_missing and active_output
                        )
                        defer_missing = missing_count > 0 and producer_running and not urgent_missing
                        if zero_count and not defer_missing:
                            refill.extend(np.zeros(chunk, dtype=np.float64) for _ in range(zero_count))
                        elif defer_missing:
                            # The writable frontier is still beyond the safe
                            # deadline. Submit the ready prefix and wait for
                            # processing to supply the future suffix.
                            zero_count = 0

                        if refill:
                            zeroed_reserved: list[tuple[int, np.ndarray, int, tuple[Any, ...]]] = []
                            fault_before_write = self.state.snapshot().fault is not None
                            if active_miss or fault_before_write:
                                # Once an active command misses its deadline,
                                # the complete refill and every future command
                                # are forced safe before the fault is published.
                                self.request_safe_zero()
                                refill = [np.zeros_like(block) for block in refill]
                                zero_count = len(refill)
                                zeroed_reserved = reserved
                                reserved = []
                            write_started = time.perf_counter_ns()
                            writer.write_many_sample(np.concatenate(refill), timeout=0.5)
                            write_ms = (time.perf_counter_ns() - write_started) / 1e6
                            self.drain_barrier.committed(self._next_block * chunk, refill, chunk)
                            for _, _, _, events in reserved:
                                self.lifecycle.committed(events, self.config.sample_rate)
                                for event in events:
                                    if event.rendered_sample_count:
                                        self.drain_barrier.preserve_until(
                                            self._pulse_exclusive_ends[event.pulse_id])
                                        if self.commit_event is not None:
                                            self.commit_event(event.pulse_id, event.rendered_sample_count)
                            with self.command_lock:
                                for target, block, _, _ in reserved:
                                    current = self.commands.get(target)
                                    if current is not None and current[0] is block:
                                        self.commands.pop(target, None)
                            if self.recorder is not None:
                                for offset, block in enumerate(refill):
                                    target = first_target + offset
                                    if not active_miss and not fault_before_write and offset < len(reserved):
                                        _, command, stage_generation, events = reserved[offset]
                                        record_sparse_ao_outcome(self.recorder, sample_index=target * chunk,
                                            generation=stage_generation, stage="ao_committed",
                                            controller=snapshot.controller, waveform=command, events=events,
                                            reason="NI writer accepted AO samples")
                                    elif active_miss or fault_before_write:
                                        rejected = (zeroed_reserved[offset]
                                                    if offset < len(zeroed_reserved) else None)
                                        # Reserved pulse IDs were already
                                        # finalized as cancelled by
                                        # request_safe_zero. Record the block
                                        # substitution without duplicating a
                                        # second pulse terminal state.
                                        events = ()
                                        command = None
                                        record_sparse_ao_outcome(self.recorder, sample_index=target * chunk,
                                            generation=rejected[2] if rejected is not None else snapshot.generation,
                                            stage="zero_substituted", controller=snapshot.controller,
                                            waveform=command, events=events,
                                            reason="faulted refill was written as safe zero")
                            if self.ao_outcome is not None:
                                for offset in range(len(refill)):
                                    submitted = not active_miss and not fault_before_write and offset < len(reserved)
                                    if (active_miss or fault_before_write) and offset < len(zeroed_reserved):
                                        # request_safe_zero already delivered the
                                        # rejection for this reserved command.
                                        continue
                                    generation = reserved[offset][2] if submitted else snapshot.generation
                                    self.ao_outcome(generation, submitted,
                                                    "AO writer accepted command" if submitted else
                                                    ("latched fault cancelled reserved AO command"
                                                     if fault_before_write else "zero substituted"))
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
                            if self.drain_barrier.drained(snapshot.generation, generated):
                                record_drain_ack(self.recorder, sample_index=generated,
                                    generation=snapshot.generation, controller=snapshot.controller,
                                    end_sample=self.drain_barrier.end_sample)
                                self.state.acknowledge_drain(snapshot.generation)
                                self.drain_barrier.acknowledge()
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
            scheduler = ScheduledAo(self.config, self.state, read_task.triggers.start_trigger.term,
                                    self.core.recorder, self.core.commit_actuator_event,
                                    self.core.record_ao_outcome)
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
                    _, safe, esn_ms, _ = self.core.process(
                        batch.values,
                        sample_index=batch.sample_index,
                        ao_target_sample_index=batch.sample_index + self.config.ao_lead_samples,
                    )
                    if self.state.snapshot().mode != "closed-loop":
                        safe = np.zeros_like(safe)
                    target = batch.sample_index // self.config.chunk_size + self.config.ao_lead_chunks
                    command_generation = self.core.last_processed_generation
                    if not scheduler.submit(target, safe, command_generation, self.core.last_pulse_events,
                                            self.core.last_admitted_pulse_plans):
                        self.core.record_ao_outcome(command_generation, False, "late or stale command rejected")
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


def load_mock_replay(path: Path, electrode_labels: tuple[str, str], sample_rate: int):
    """Compatibility wrapper for the typed finite replay loader."""
    replay = load_replay(path)
    if replay.sample_rate_hz != sample_rate:
        raise ValueError(f"mock replay rate {replay.sample_rate_hz:g} Hz does not match configured {sample_rate} Hz")
    def role(label: str) -> str:
        lowered = label.casefold()
        return "CA3" if "ca3" in lowered else "Cortex" if any(key in lowered for key in ("cortex", "ctx", "ec")) else "unknown"
    indexes = {role(label): i for i, label in enumerate(replay.channel_labels)}
    requested_roles = [role(label) for label in electrode_labels]
    if len(indexes) < 2 or set(requested_roles) != set(indexes):
        raise ValueError(f"mock replay channels {replay.channel_labels} do not match configured {electrode_labels}")
    order = [indexes[name] for name in requested_roles]
    return dataclasses.replace(replay, electrode_mV=replay.electrode_mV[order],
                               raw_V=replay.raw_V[order] if replay.raw_V is not None else None,
                               channel_labels=electrode_labels,
                               channel_roles=tuple(replay.channel_roles[i] for i in order),
                               ctx_index=order.index(replay.ctx_index))


class MockDaq(BaseDaq):
    def __init__(self, *args: Any, overload_ms: float = 0.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.overload_ms = overload_ms
        self.replay_index = 0
        self.mock_sample_index = 0
        self._source_epoch: Optional[float] = None
        self._observed_acquiring = self.state.snapshot().acquiring
        self._ao_stop = threading.Event()
        self._ao_thread: Optional[threading.Thread] = None
        self.scheduler = SimulatedAoScheduler(
            self.config, self.state, self.core.recorder, self.core.commit_actuator_event,
            self.core.record_ao_outcome,
        )
        if self.config.mock_replay is None:
            raise ValueError("mock mode requires a replay recording")
        self.replay = (self.config.mock_replay_data if self.config.mock_replay_data is not None else
            load_mock_replay(self.config.mock_replay, self.config.electrode_labels, self.config.sample_rate))

    def run(self) -> None:
        LOG.info("Mock replay acquisition started: %s", self.config.mock_replay)
        period_s = self.config.chunk_size / self.config.sample_rate
        self._source_epoch = time.perf_counter()
        self._ao_stop.clear()
        self._ao_thread = threading.Thread(target=self._run_ao_clock, args=(period_s,),
                                           name="mock-ao-clock", daemon=True)
        self._ao_thread.start()
        try:
            while self.state.snapshot().running:
                snapshot = self.state.snapshot()
                if snapshot.acquiring and not self._observed_acquiring:
                    # A deliberate pause drops acquisition samples while the
                    # shared hardware timeline continues. Resume at the current
                    # AO sample frontier and record the resulting source gap.
                    self.replay_index = min(
                        max(self.replay_index, self.scheduler.generated_sample_count),
                        self.replay.sample_count,
                    )
                    self.mock_sample_index = self.replay_index
                    self._source_epoch = time.perf_counter() - (
                        self.replay_index / self.config.sample_rate)
                self._observed_acquiring = snapshot.acquiring
                if not snapshot.acquiring:
                    source_index = self.replay_index
                    self.core.process(
                        np.full((2, self.config.chunk_size), np.nan), sample_index=source_index,
                        ao_target_sample_index=source_index + self.config.ao_lead_samples,
                        input_is_electrode_mV=True,
                    )
                    target = math.ceil(self.scheduler.generated_sample_count / self.config.chunk_size)
                    target += self.config.ao_lead_chunks
                    zeros = np.zeros((1, self.config.chunk_size), dtype=np.float64)
                    if not self.scheduler.submit(target, zeros, snapshot.generation):
                        self.core.record_ao_outcome(snapshot.generation, False, "late or stale paused mock block")
                    time.sleep(0.05)
                    continue

                if self.replay_index >= self.replay.sample_count:
                    self.state.set_armed(False)
                    self.state.set_acquiring(False)
                    continue
                source_index = self.replay_index
                assert self._source_epoch is not None
                source_available_at = self._source_epoch + (
                    source_index + self.config.chunk_size) / self.config.sample_rate
                remaining = source_available_at - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                block_started = time.perf_counter_ns()
                electrode_mv, raw_v, valid_count = self._next_input()
                if self.overload_ms:
                    time.sleep(self.overload_ms / 1_000)
                _, safe, esn_ms, _ = self.core.process(
                    electrode_mv, sample_index=source_index,
                    ao_target_sample_index=source_index + self.config.ao_lead_samples,
                    input_is_electrode_mV=True, raw_ai_V=raw_v,
                    input_roles=self.replay.channel_roles, input_labels=self.replay.channel_labels,
                    input_ctx_index=self.replay.ctx_index,
                )
                if self.state.snapshot().mode != "closed-loop":
                    safe = np.zeros_like(safe)
                target_block = source_index // self.config.chunk_size + self.config.ao_lead_chunks
                command_generation = self.core.last_processed_generation
                if not self.scheduler.submit(target_block, safe, command_generation,
                                             self.core.last_pulse_events,
                                             self.core.last_admitted_pulse_plans):
                    self.core.record_ao_outcome(command_generation, False,
                                                "late or stale simulated command rejected")
                block_ms = (time.perf_counter_ns() - block_started) / 1e6
                self.mock_sample_index = source_index + valid_count
                self.telemetry.update_block(
                    sample_index=self.mock_sample_index, ai_ms=0.0, esn_ms=esn_ms, ao_ms=0.0,
                    block_ms=block_ms, deadline_ms=period_s * 1_000,
                    logger_backlog=self.core.recorder.backlog,
                )
        finally:
            self._ao_stop.set()
            if self._ao_thread is not None:
                self._ao_thread.join(3.0)
            self._ao_thread = None
            self.request_safe_zero()

    def _run_ao_clock(self, period_s: float) -> None:
        last = time.perf_counter()
        while not self._ao_stop.is_set() and self.state.snapshot().running:
            now = time.perf_counter()
            try:
                generated = self.scheduler.advance_elapsed(now - last)
            except Exception as exc:
                self.state.fault(f"simulated AO clock failure: {type(exc).__name__}: {exc}")
                return
            last = now
            if generated:
                self.last_command = generated[-1].waveform_v.reshape(1, -1)
                self.mock_sample_index = max(self.mock_sample_index, self.scheduler.generated_sample_count)
            self._ao_stop.wait(min(0.001, period_s / 8))

    def _next_input(self) -> np.ndarray:
        if self.replay_index >= self.replay.sample_count:
            self.state.set_armed(False)
            self.state.set_acquiring(False)
            return (np.full((2, self.config.chunk_size), np.nan),
                    np.full((2, self.config.chunk_size), np.nan), 0)
        stop = min(self.replay.sample_count, self.replay_index + self.config.chunk_size)
        count = stop - self.replay_index
        mv = np.full((2, self.config.chunk_size), np.nan, dtype=np.float64)
        mv[:, :count] = self.replay.electrode_mV[:, self.replay_index:stop]
        raw = np.full_like(mv, np.nan)
        if self.replay.raw_V is not None:
            raw[:, :count] = self.replay.raw_V[:, self.replay_index:stop]
        self.replay_index = stop
        return mv, raw, count


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
    result = dataclasses.asdict(dataclasses.replace(config, mock_replay_data=None))
    result["chunk_size"] = config.chunk_size
    result["ao_lead_samples"] = config.ao_lead_samples
    result["ao_deadline_samples"] = config.ao_deadline_samples
    result["record_dir"] = str(config.record_dir)
    result["mock_replay"] = str(config.mock_replay) if config.mock_replay else None
    result.pop("mock_replay_data", None)
    result["model_artifact"] = str(config.model_artifact)
    result["hardware_profile"] = str(config.hardware_profile) if config.hardware_profile else None
    return result


async def websocket_handler(websocket: Any, state: RuntimeState, recorder: H5Recorder,
                            hub: UiHub, shutdown: asyncio.Event, config: AppConfig) -> None:
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
                elif name == "set_controller":
                    controller = str(command.get("controller", ""))
                    state.set_controller(controller)
                    if controller in {"rule", "pid"} and state.snapshot().stim_mode != "threshold_pulse":
                        pulse_limit = pulse_amplitude_limit_v(config.safety, config.sample_rate)
                        state.set_stim("threshold_pulse", min(state.snapshot().stim_gain, pulse_limit),
                                       max_gain=pulse_limit)
                elif name == "set_controller_settings":
                    controller = state.snapshot().controller
                    supplied = command.get("settings")
                    if not isinstance(supplied, dict):
                        raise ValueError("settings must be a complete JSON object")
                    settings = complete_controller_settings(controller, supplied)
                    prepared = build_controller(controller, config.sample_rate, settings)
                    state.set_controller_settings(controller, settings, prepared)
                elif name == "set_stim":
                    stim_mode = str(command.get("stim_mode", state.snapshot().stim_mode))
                    gain = float(command.get("stim_gain", state.snapshot().stim_gain))
                    max_gain = 10.0
                    if (state.snapshot().controller in {"rule", "pid"}
                            or stim_mode == "threshold_pulse"):
                        max_gain = pulse_amplitude_limit_v(config.safety, config.sample_rate)
                    state.set_stim(stim_mode, gain, max_gain=max_gain)
                elif name == "set_armed":
                    if (bool(command.get("armed", False))
                            and (state.snapshot().controller in {"rule", "pid"}
                                 or state.snapshot().stim_mode == "threshold_pulse")):
                        pulse_limit = pulse_amplitude_limit_v(config.safety, config.sample_rate)
                        if state.snapshot().stim_gain > pulse_limit:
                            raise ValueError(f"pulse amplitude exceeds configured safety limit ({pulse_limit:.3f} V)")
                    state.set_armed(bool(command.get("armed", False)))
                elif name == "set_pulse_threshold":
                    state.set_pulse_threshold_std(float(command.get("pulse_threshold_std")))
                elif name == "set_pulse_window":
                    state.set_pulse_window_sec(float(command.get("pulse_window_sec")))
                elif name == "set_pulse_polarity":
                    state.set_pulse_polarity(str(command.get("pulse_polarity", "")))
                elif name == "set_cortex_channel":
                    labels = state.set_cortex_channel(str(command.get("channel", "")))
                    recorder.set_electrode_labels(labels)
                elif name == "clear_fault":
                    raise ValueError("faults are latched; safely restart the application after correcting the cause")
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
            await websocket.send(json.dumps(json_safe(response), allow_nan=False))

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
    parser.add_argument(
        "--hardware-profile", type=Path,
        help="calibration/safety profile (real-hardware default: included pending-test profile)",
    )
    parser.add_argument("--amplifier-gain", nargs=2, type=float,
                        metavar=("ELECTRODE_1", "ELECTRODE_2"),
                        help="override MultiClamp output gains; logged as pending test on real hardware")
    parser.add_argument("--ao-command-gain", type=float,
                        help="override NI AO volts per ESN output unit; logged on real hardware")
    parser.add_argument("--stim-monitor-ai", help="optional AI channel measuring actual stimulus")
    parser.add_argument("--terminal-config", default="RSE", choices=("DIFFERENTIAL", "RSE", "NRSE"))
    parser.add_argument("--sample-rate", type=int, default=20_000)
    parser.add_argument("--processing-block-ms", type=float, default=5.0,
                        help="DAQ/ESN processing interval in milliseconds")
    parser.add_argument("--ao-target-lead-ms", type=float, default=100.0,
                        help="desired queued AO duration; refill and batching are automatic")
    parser.add_argument("--passthrough-dc-block-hz", type=float, default=DEFAULT_PASSTHROUGH_DC_BLOCK_HZ)
    parser.add_argument("--pulse-threshold-std", type=float, default=DEFAULT_PULSE_THRESHOLD_STD)
    parser.add_argument("--pulse-window-sec", type=float, default=DEFAULT_PULSE_WINDOW_SEC)
    parser.add_argument("--pulse-polarity", choices=("absolute", "positive", "negative"), default="absolute")
    parser.add_argument("--model-artifact", type=Path, default=ESN_ARTIFACT)
    parser.add_argument("--controller", choices=("esn", "rule", "pid"), default="esn")
    parser.add_argument("--controller-config", type=Path, help="JSON settings for the selected rule or PID controller")
    parser.add_argument("--esn-backend", choices=("auto", "numpy", "numba"), default="auto")
    parser.add_argument("--max-command-v", type=float)
    parser.add_argument("--max-slew-v-per-s", type=float)
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


def load_hardware_profile(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    profile = json.loads(resolved.read_text(encoding="utf-8"))
    if profile.get("format_tag") != "echo-chamber-hardware-profile-v1":
        raise ValueError("unsupported hardware profile format")
    validation_status = profile.get("validation_status")
    if profile.get("validated_for_real_hardware") is not True:
        if validation_status != "confirmed_pending_test":
            raise ValueError(
                "hardware profile must be validated or explicitly marked confirmed_pending_test"
            )
        LOG.warning(
            "HARDWARE PROFILE CONFIRMED PENDING TEST: calibration and stimulation limits "
            "have not yet been verified on the NI rig (%s)", resolved,
        )
    required = {
        "amplifier_gain",
        "ao_command_gain_v_per_esn_unit",
        "electrode_labels",
        "ctx_index",
        "safety",
    }
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError(f"hardware profile is missing fields: {missing}")
    safety_required = {
        "max_command_v",
        "max_slew_v_per_s",
        "max_abs_area_v_s",
        "area_window_s",
        "max_active_fraction",
        "active_threshold_v",
        "max_consecutive_active_s",
    }
    safety = profile.get("safety")
    if not isinstance(safety, dict):
        raise ValueError("hardware profile safety must be an object")
    missing_safety = sorted(safety_required - set(safety))
    if missing_safety:
        raise ValueError(f"hardware profile safety is missing fields: {missing_safety}")
    gains = profile.get("amplifier_gain")
    labels = profile.get("electrode_labels")
    if not isinstance(gains, list) or len(gains) != 2:
        raise ValueError("hardware profile amplifier_gain must contain exactly two values")
    if not isinstance(labels, list) or len(labels) != 2 or not all(
        isinstance(label, str) and label.strip() for label in labels
    ):
        raise ValueError("hardware profile electrode_labels must contain two non-empty labels")
    if profile.get("ctx_index") not in (0, 1):
        raise ValueError("hardware profile ctx_index must be 0 or 1")
    numeric_values = {
        "amplifier_gain[0]": gains[0],
        "amplifier_gain[1]": gains[1],
        "ao_command_gain_v_per_esn_unit": profile.get("ao_command_gain_v_per_esn_unit"),
        **{f"safety.{name}": safety[name] for name in safety_required},
    }
    for name, value in numeric_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"hardware profile {name} must be a finite number")
    return profile


def config_from_args(args: argparse.Namespace) -> AppConfig:
    mock_replay = args.mock_replay.resolve() if args.mock_replay else (DEFAULT_MOCK_REPLAY if args.mock else None)
    profile_path = (
        args.hardware_profile.resolve()
        if args.hardware_profile
        else (DEFAULT_HARDWARE_PROFILE if mock_replay is None else None)
    )
    profile = load_hardware_profile(profile_path) if profile_path else {}
    if mock_replay is None and not profile:
        raise ValueError("real NI hardware requires a calibration/safety profile")
    if mock_replay is None and args.dry_test_allow_sustained_ao:
        LOG.warning(
            "REAL-HARDWARE DRY TEST OVERRIDE ENABLED: charge, duty-cycle, and continuous-output "
            "limits are bypassed; connect only a scope or dummy load"
        )
    if mock_replay is None and any(
        value is not None
        for value in (
            args.amplifier_gain,
            args.ao_command_gain,
            args.max_command_v,
            args.max_slew_v_per_s,
        )
    ):
        LOG.warning(
            "REAL-HARDWARE CALIBRATION/SAFETY OVERRIDE CONFIRMED PENDING TEST: "
            "command-line values replace the profile for this run"
        )
    profile_safety = dict(profile.get("safety", {}))
    amplifier_gain = tuple(
        args.amplifier_gain or profile.get("amplifier_gain", (10.0, 10.0))
    )
    ao_command_gain = float(
        args.ao_command_gain
        if args.ao_command_gain is not None
        else profile.get("ao_command_gain_v_per_esn_unit", 1.0)
    )
    electrode_labels = tuple(profile.get("electrode_labels", ("CA3", "Cortex")))
    ctx_index = int(profile.get("ctx_index", 1))
    replay_data = load_replay(mock_replay) if mock_replay is not None else None
    source_channel_regions = None
    if replay_data is not None:
        electrode_labels = replay_data.channel_labels
        ctx_index = replay_data.ctx_index
        configured_regions = replay_data.metadata.get("channel_regions")
        if isinstance(configured_regions, (list, tuple)) and len(configured_regions) == 2:
            source_channel_regions = tuple(str(region) for region in configured_regions)
    controller_settings = {
        "rule": controller_settings_for("rule"),
        "pid": controller_settings_for("pid"),
    }
    if args.controller_config:
        loaded_settings = json.loads(args.controller_config.resolve().read_text(encoding="utf-8"))
        if not isinstance(loaded_settings, dict):
            raise ValueError("controller config file must contain a JSON object")
        if args.controller not in {"rule", "pid"}:
            raise ValueError("--controller-config applies only to rule or pid")
        controller_settings[args.controller] = controller_settings_for(args.controller, loaded_settings)
        build_controller(args.controller, args.sample_rate, controller_settings[args.controller])
    return AppConfig(
        device=args.device,
        ai_channels=tuple(args.ai),
        ao_channel=args.ao,
        electrode_labels=electrode_labels,
        source_channel_regions=source_channel_regions,
        ctx_index=ctx_index,
        amplifier_gain=amplifier_gain,
        ao_command_gain_v_per_esn_unit=ao_command_gain,
        sample_rate=args.sample_rate,
        processing_block_ms=args.processing_block_ms,
        ao_target_lead_ms=args.ao_target_lead_ms,
        passthrough_dc_block_hz=args.passthrough_dc_block_hz,
        pulse_threshold_std=args.pulse_threshold_std,
        pulse_window_sec=args.pulse_window_sec,
        pulse_polarity=args.pulse_polarity,
        esn_backend=args.esn_backend,
        model_artifact=args.model_artifact.resolve(),
        controller=args.controller,
        controller_config=args.controller_config.resolve() if args.controller_config else None,
        controller_settings=controller_settings,
        hardware_profile=profile_path,
        terminal_config=args.terminal_config,
        record_dir=args.record_dir.resolve(),
        ws_host=args.host,
        ws_port=args.port,
        mock_replay=mock_replay,
        mock_replay_data=replay_data,
        actual_stim_monitor_channel=args.stim_monitor_ai,
        start_paused=args.start_paused,
        open_browser=not args.no_browser,
        safety=SafetyConfig(
            max_command_v=float(
                args.max_command_v if args.max_command_v is not None
                else profile_safety.get("max_command_v", 5.0)
            ),
            max_slew_v_per_s=float(
                args.max_slew_v_per_s if args.max_slew_v_per_s is not None
                else profile_safety.get("max_slew_v_per_s", 2_000.0)
            ),
            max_abs_area_v_s=float(profile_safety.get("max_abs_area_v_s", 0.010)),
            area_window_s=float(profile_safety.get("area_window_s", 1.0)),
            max_active_fraction=float(profile_safety.get("max_active_fraction", 0.10)),
            active_threshold_v=float(profile_safety.get("active_threshold_v", 3.5)),
            max_consecutive_active_s=float(profile_safety.get("max_consecutive_active_s", 0.050)),
            isolator_command_v_per_output_unit=profile_safety.get("isolator_command_v_per_output_unit"),
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

    esn = EsnRuntime(config.model_artifact, config)
    replay_mode = config.mock_replay is not None
    if not replay_mode and esn.ready and not esn.production_eligible:
        LOG.warning(
            "MODEL VALIDATION PENDING: the selected ESN artifact is not marked "
            "production-eligible; hardware testing is continuing by explicit research choice"
        )
    state = RuntimeState(
        esn_ready=esn.ready,
        start_paused=config.start_paused,
        electrode_labels=config.electrode_labels,
        ctx_index=config.ctx_index,
        pulse_threshold_std=config.pulse_threshold_std,
        pulse_window_sec=config.pulse_window_sec,
        pulse_polarity=config.pulse_polarity,
        controller=config.controller,
        controller_ready={"esn": esn.ready, "rule": True, "pid": True},
        controller_settings=config.controller_settings,
        pulse_max_amplitude_v=pulse_amplitude_limit_v(config.safety, config.sample_rate),
        input_mapping_locked=replay_mode,
    )
    telemetry = Telemetry()
    hub = UiHub(config.ui_queue_packets)
    initial = state.snapshot()
    hub.publish(json.dumps(json_safe({
        "ai": [[], []], "ao": [[]], "passthrough_preview": [[]],
        "pulse_threshold_preview": [[]], "sample_indices": [],
        "mode": initial.mode, "is_recording": initial.recording,
        "controller": initial.controller, "controller_ready": initial.controller_ready,
                    "controller_calibrated": initial.controller_calibrated,
        "transition_pending": initial.transition_pending,
                    "controller_settings": dict(initial.controller_settings),
        "config_generation": initial.generation, "controller_diagnostics": {},
        "input_mapping_locked": initial.input_mapping_locked,
        "is_acquiring": initial.acquiring, "stim_mode": initial.stim_mode,
        "stim_gain": initial.stim_gain,
        "pulse_amplitude_V": (initial.stim_gain if initial.controller in {"rule", "pid"}
                               or initial.stim_mode == "threshold_pulse" else None),
        "max_pulse_amplitude_V": pulse_amplitude_limit_v(config.safety, config.sample_rate),
        "pulse_threshold_std": initial.pulse_threshold_std,
        "pulse_window_sec": initial.pulse_window_sec,
        "pulse_polarity": initial.pulse_polarity, "armed": initial.armed,
        "fault": initial.fault, "esn_ready": initial.esn_ready,
        "channels": list(initial.electrode_labels), "cortex_ai": f"ai{initial.ctx_index}",
        "fs": config.sample_rate / config.visual_downsample,
        "sample_rate_hz": config.sample_rate, "ai_unit": "mV", "ao_unit": "V",
        "amplifier_gain": list(config.amplifier_gain),
        "ao_command_gain_v_per_esn_unit": config.ao_command_gain_v_per_esn_unit,
        "input_source_kind": ("replay_electrode_mV" if config.mock_replay is not None
                              else "ni_ai_raw_V"),
    }), allow_nan=False))
    metadata = {
        "application": "echoChamber.py",
        "configuration": serializable_config(config),
        "esn_artifact": str(config.model_artifact),
        "esn_ready": esn.ready,
        "controller": config.controller,
        "controller_config": str(config.controller_config) if config.controller_config else None,
        "esn_error": esn.error,
        "esn_bridge": "application-owned chunk adaptation and stimulation mapping",
        "ao_scheduler": "dynamic non-regenerating writer; no fixed refill count or mandatory batch size",
        "pulse_amplitude_unit": "AO V; independent of ESN passthrough scaling",
    }
    recorder = H5Recorder(config, metadata)
    loop = asyncio.get_running_loop()
    shutdown = asyncio.Event()
    # Bind the UI endpoint before starting any DAQ or AO thread. A port error
    # therefore cannot leave hardware acquisition running without controls.
    server = await websockets.serve(
        lambda websocket: websocket_handler(websocket, state, recorder, hub, shutdown, config),
        config.ws_host,
        config.ws_port,
    )
    LOG.info("UI server listening at ws://%s:%d", config.ws_host, config.ws_port)

    core = AsyncUiProcessingCore(config, state, esn, recorder, hub, loop, telemetry)
    daq: BaseDaq = (
        MockDaq(config, state, core, telemetry, overload_ms=args.mock_overload_ms)
        if replay_mode else RealDaq(config, state, core, telemetry)
    )
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(threadName)s %(message)s")
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
