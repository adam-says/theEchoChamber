"""Reproducible benchmarks that exercise the production processing core."""
from __future__ import annotations

import asyncio
import csv
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlsplit, urlunsplit
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import websockets
import h5py

import echoChamber as app
from replay import ReplayData, load_replay


REPLAY_DIRECTORY = Path(__file__).resolve().parent / "recordings" / "ca3_slow_waves"
DEFAULT_TRACES = tuple(
    REPLAY_DIRECTORY / f"{name}.h5"
    for name in ("2026_05_29_0001", "2026_08_05_0000", "2026_08_06_0000")
)
CONTROLLERS = ("rule", "pid", "esn")
LEAD_SWEEP_MS = (100.0, 75.0, 50.0, 40.0, 30.0, 20.0)
PID_SYNTHETIC_CONFIG = Path(__file__).resolve().parent / "controller_configs" / "pid_synthetic_demo.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_sources(paths: Iterable[Path]) -> list[ReplayData]:
    result = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"replay file not found: {path}. Import the canonical ABF files into "
                "recordings/ca3_slow_waves; see docs/replay_import.md."
            )
        replay = load_replay(path)
        if not np.isclose(replay.sample_rate_hz, 20_000):
            raise ValueError(f"{path} has {replay.sample_rate_hz:g} Hz; expected 20000 Hz")
        if replay.channel_roles.count("CA3") != 1 or replay.channel_roles.count("Cortex") != 1:
            raise ValueError(f"{path} must map CA3 and Cortex roles")
        result.append(replay)
    return result


@dataclass
class RunMetrics:
    case_id: str
    source_name: str
    controller: str
    sample_rate_hz: float
    block_samples: int
    block_ms: float
    ao_lead_ms: float
    deadline_ms: float
    source_samples: int = 0
    skipped_source_samples: int = 0
    source_index_gap_samples: int = 0
    source_availability_lag_median_ms: float = 0.0
    source_availability_lag_p95_ms: float = 0.0
    source_availability_lag_max_ms: float = 0.0
    source_backlog_blocks_max: int = 0
    processed_blocks: int = 0
    duration_source_s: float = 0.0
    process_median_ms: float = 0.0
    process_p95_ms: float = 0.0
    process_p99_ms: float = 0.0
    process_max_ms: float = 0.0
    process_thread_cpu_median_ms: float = 0.0
    process_thread_cpu_p95_ms: float = 0.0
    process_thread_cpu_max_ms: float = 0.0
    process_cpu_median_ms: float = 0.0
    process_cpu_p95_ms: float = 0.0
    process_cpu_max_ms: float = 0.0
    process_slow_call_count: int = 0
    gc_pause_count: int = 0
    gc_pause_max_ms: float = 0.0
    max_requested_rate_hz: float = 0.0
    bounded_rate_request_count: int = 0
    deadline_exceeded_blocks: int = 0
    requested_rate_blocks: int = 0
    requested_pulse_events: int = 0
    rendered_pulse_events: int = 0
    rejected_pulse_events: int = 0
    pulse_commit_samples: int = 0
    ao_nonzero_generated_samples: int = 0
    ao_committed_blocks: int = 0
    ao_zero_substitutions: int = 0
    ao_late_commands: int = 0
    ao_rejected_stale: int = 0
    ao_cancelled_blocks: int = 0
    ao_max_queue_depth: int = 0
    ao_active_demand_misses: int = 0
    ao_generated_sample_count: int = 0
    ao_drain_complete: bool = False
    ao_final_queue_depth: int = 0
    ao_noncommitted_nonzero_blocks: int = 0
    ao_clock_mode: str = "virtual_5ms"
    ao_clock_max_service_gap_ms: float = 0.0
    ao_clock_gaps_over_block_period: int = 0
    browser_ui_required: bool = False
    recording_completed: bool = False
    recording_committed_samples: int = 0
    component_timings_ms: dict[str, dict[str, float | int]] = field(default_factory=dict)
    recorder_max_backlog: int = 0
    ui_packets_submitted: int = 0
    ui_packets_consumed: int = 0
    ui_transport_kind: str = "websocket_client_without_browser_rendering"
    browser_connected: bool = False
    injected_stall_count: int = 0
    injected_stall_ms_total: float = 0.0
    runtime_fault: str | None = None
    generated_ao_blocks: int = 0
    pulse_request_ids: list[str] = field(default_factory=list)
    pulse_commit_ids: list[str] = field(default_factory=list)
    last_controller_ready: bool = False
    last_controller_calibrated: bool = False
    last_controller_z_score: float | None = None
    last_controller_reason: str = ""
    recording_active: bool = True
    physical_pulse_claims: bool = False
    simulation: bool = False
    notes: list[str] = field(default_factory=list)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot encode {type(value).__name__}")


def json_safe(value: Any) -> Any:
    """Convert paths, NumPy scalars, and non-finite values to strict JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_progress(path: Path | None, value: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_safe(value), allow_nan=False, separators=(",", ":")) + "\n")
        stream.flush()


def acceptance_failures(metrics: RunMetrics) -> list[str]:
    """Evaluate hard software acceptance limits for a completed benchmark case."""
    failures: list[str] = []
    if metrics.processed_blocks <= 0:
        failures.append("no processing blocks completed")
    if metrics.skipped_source_samples or metrics.source_index_gap_samples:
        failures.append("paced source data was skipped or has an index gap")
    if not np.isfinite(metrics.process_p99_ms) or metrics.process_p99_ms >= 2.5:
        failures.append("process p99 is not below 2.5 ms")
    if not np.isfinite(metrics.process_max_ms) or metrics.process_max_ms >= metrics.deadline_ms:
        failures.append("maximum process time reached or exceeded the block deadline")
    if metrics.deadline_exceeded_blocks:
        failures.append("one or more blocks exceeded the processing deadline")
    if metrics.ao_active_demand_misses:
        failures.append("AO active-demand deadline miss occurred")
    if metrics.ao_late_commands:
        failures.append("late AO command was reported")
    if metrics.ao_rejected_stale:
        failures.append("stale AO command was rejected")
    if metrics.ao_noncommitted_nonzero_blocks:
        failures.append("non-zero AO block was generated without an accepted commit stage")
    if metrics.ao_final_queue_depth:
        failures.append("AO command queue was not empty after drain")
    if not metrics.ao_drain_complete:
        failures.append("AO drain barrier did not complete")
    if not metrics.recording_completed:
        failures.append("HDF5 recorder did not finalize as complete")
    expected_recording_samples = metrics.processed_blocks * metrics.block_samples
    if metrics.recording_committed_samples != expected_recording_samples:
        failures.append("recorded sample count does not match submitted processing blocks")
    if not metrics.ui_packets_submitted or not metrics.ui_packets_consumed:
        failures.append("UI packets were not submitted and consumed")
    if metrics.browser_ui_required and not metrics.browser_connected:
        failures.append("required browser UI did not remain connected")
    if metrics.requested_pulse_events != metrics.rendered_pulse_events:
        failures.append("requested pulse events were not all rendered or rejected")
    return failures


def parse_ui_packet(packet: str) -> dict[str, Any]:
    """Parse the browser wire format strictly; Python's NaN extension is invalid JSON."""
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")
    decoded = json.loads(packet, parse_constant=reject_constant)
    if not isinstance(decoded, dict):
        raise ValueError("UI packet must be a JSON object")
    return decoded


def _run_id(source_name: str, controller: str, lead_ms: float) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_source = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_name)
    return f"{stamp}_{safe_source}_{controller}_{lead_ms:g}ms"


def _browser_connection_url(base_url: str, ws_port: int) -> str:
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.netloc or parts.query or parts.fragment:
        raise ValueError("browser base URL must be an HTTP(S) origin or index.html URL without query/fragment")
    path = parts.path.rstrip("/")
    if not path.endswith("index.html"):
        path = f"{path}/index.html"
    return urlunsplit((parts.scheme, parts.netloc, path, f"wsPort={int(ws_port)}", ""))


def _make_config(controller: str, lead_ms: float, output_dir: Path,
                 pid_settings_override: dict[str, float] | None = None) -> app.AppConfig:
    if controller not in CONTROLLERS:
        raise ValueError(f"unknown controller: {controller}")
    pid_document = json.loads(PID_SYNTHETIC_CONFIG.read_text(encoding="utf-8"))
    pid_settings = {key: value for key, value in pid_document.items()
                    if key in app.PID_CONTROLLER_DEFAULTS}
    if pid_settings_override:
        pid_settings.update(pid_settings_override)
    settings = {
        "rule": app.controller_settings_for("rule"),
        "pid": app.controller_settings_for("pid", pid_settings),
    }
    config = app.AppConfig(
        sample_rate=20_000,
        processing_block_ms=5.0,
        ao_target_lead_ms=float(lead_ms),
        controller=controller,
        controller_config=PID_SYNTHETIC_CONFIG if controller == "pid" else None,
        controller_settings=settings,
        record_dir=output_dir,
        open_browser=False,
        ui_interval_s=0.1,
        safety=app.SafetyConfig(
            max_command_v=5.0, max_slew_v_per_s=2_000.0,
            max_abs_area_v_s=0.010, area_window_s=1.0,
            max_active_fraction=0.10, active_threshold_v=3.5,
            max_consecutive_active_s=0.050,
        ),
    )
    config.validate()
    return config


def _replay_blocks(replay: ReplayData, limit_samples: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    block = 100
    used = 0
    for first, values in replay.iter_chunks(block):
        if limit_samples is not None and used >= limit_samples:
            break
        if limit_samples is not None:
            values = values[:, :max(0, min(values.shape[1], limit_samples - used))]
        if values.shape[1] == 0:
            break
        for offset in range(0, values.shape[1], block):
            part = values[:, offset:offset + block]
            padded = np.full((2, block), np.nan, dtype=np.float64)
            padded[:, :part.shape[1]] = part
            yield first + offset, padded
            used += part.shape[1]


class _Harness:
    def __init__(self, config: app.AppConfig, source_name: str, metadata: dict[str, Any],
                 *, progress_path: Path | None, progress_interval_s: float,
                 paced: bool, simulation: bool, on_generated: Callable[[app.GeneratedAoBlock], None] | None,
                 browser_ui: bool = False, external_browser: bool = False,
                 browser_base_url: str = "http://127.0.0.1:8000",
                 inject_stall_ms: float = 0.0,
                 stall_at_source_index: int | None = None, stall_on_first_pulse: bool = False):
        self.config, self.progress_path = config, progress_path
        self.progress_interval_s, self.paced = progress_interval_s, paced
        self.on_generated = on_generated
        self.inject_stall_ms = float(inject_stall_ms)
        self.stall_at_source_index = stall_at_source_index
        self.stall_on_first_pulse = stall_on_first_pulse
        self._stall_used = False
        self.esn = app.EsnRuntime(config.model_artifact, config)
        self.state = app.RuntimeState(
            esn_ready=self.esn.ready, start_paused=False,
            electrode_labels=("CA3", "Cortex"), ctx_index=1,
            pulse_threshold_std=config.pulse_threshold_std,
            pulse_window_sec=config.pulse_window_sec,
            controller=config.controller,
            controller_ready={"esn": self.esn.ready, "rule": True, "pid": True},
            controller_settings=config.controller_settings,
        )
        self.telemetry = app.Telemetry()
        self.hub = _CountingUiHub(config.ui_queue_packets)
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.loop.run_forever, name="benchmark-ui-loop", daemon=True)
        self.loop_thread.start()
        self._ao_clock_stop = threading.Event()
        self._ao_clock_thread: threading.Thread | None = None
        self._ao_clock_error: BaseException | None = None
        self._observed_ao_fault: str | None = None
        self._observed_late_count = 0
        self._observed_active_miss_count = 0
        self._ao_clock_max_gap_s = 0.0
        self._ao_clock_gap_count = 0
        self.source_availability_lags_s: list[float] = []
        self._last_source_availability_lag_ms = 0.0
        self._generated_lock = threading.Lock()
        self.recorder = app.H5Recorder(config, metadata)
        self.recording_path = self.recorder.start(_run_id(source_name, config.controller, config.ao_target_lead_ms))
        self.state.set_recording(True)
        try:
            self.core = app.AsyncUiProcessingCore(config, self.state, self.esn, self.recorder,
                                                  self.hub, self.loop, self.telemetry)
        except Exception:
            self.state.set_recording(False)
            self.recorder.stop_recording()
            self.recorder.close()
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop_thread.join(timeout=3)
            self.loop.close()
            raise
        self.shutdown = None
        self.ui_server, self.ui_client, self.ui_consumer_task, self.ui_port = self._run_on_loop(
            self._start_ui_transport())
        self.browser_connected = False
        self.browser_ui_required = bool(browser_ui)
        if browser_ui:
            url = _browser_connection_url(browser_base_url, self.ui_port)
            if external_browser:
                write_progress(self.progress_path, {
                    "type": "browser_required", "case_id": self.recording_path.stem,
                    "ws_port": self.ui_port, "url": url,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                })
            else:
                opened = app.webbrowser.open(url, new=1)
                if not opened:
                    self.abort()
                    raise RuntimeError("could not open the controller UI browser")
            deadline = time.monotonic() + 120.0
            while self.hub.client_count < 2 and time.monotonic() < deadline:
                time.sleep(0.05)
            self.browser_connected = self.hub.client_count >= 2
            if not self.browser_connected:
                self.abort()
                raise TimeoutError("browser UI did not connect to the benchmark WebSocket")
        self.pulse_commit_samples = 0
        self._pulse_commit_ids: set[str] = set()
        self.scheduler = app.SimulatedAoScheduler(config, self.state, self.recorder,
                                                  self._commit_actuator_event,
                                                  self.core.record_ao_outcome)
        self.metrics = RunMetrics(
            case_id=self.recording_path.stem, source_name=source_name, controller=config.controller,
            sample_rate_hz=config.sample_rate, block_samples=config.chunk_size,
            block_ms=config.processing_block_ms, ao_lead_ms=config.ao_target_lead_ms,
            deadline_ms=config.chunk_size * 1000 / config.sample_rate,
            simulation=simulation,
            browser_ui_required=self.browser_ui_required,
            ao_clock_mode="independent_monotonic_wall" if paced else "virtual_5ms",
        )
        self.component_latencies_ns: dict[str, list[int]] = {}
        self._instrument_core_components()
        self.latencies_ms: list[float] = []
        self.thread_cpu_ns: list[int] = []
        self.process_cpu_ns: list[int] = []
        self._gc_starts: dict[tuple[int, int], int] = {}
        self._gc_pauses_ns: list[int] = []
        self._gc_events: list[tuple[int, int, int]] = []
        self._slow_process_calls = 0
        self._gc_callback = self._collect_gc_timing
        gc.callbacks.append(self._gc_callback)
        self._gc_callback_registered = True
        self._seen_pulse_events: set[str] = set()
        self.metrics_start_ns = time.perf_counter_ns()
        self.last_progress_time = time.monotonic()
        self.next_block = 0
        self.base_sample_index: int | None = None
        self.last_ui_count = 0
        self.state.set_mode("closed-loop")
        self.state.set_stim("threshold_pulse", 0.5)
        self.pacing_epoch: float | None = None
        self.last_source_index: int | None = None

    def process(self, source_index: int, values_mv: np.ndarray, roles: tuple[str, str],
                labels: tuple[str, str], ctx_index: int, *, arm_after_drain: bool = True,
                stim_enabled: bool = True) -> app.GeneratedAoBlock:
        if self.base_sample_index is None:
            self.base_sample_index = source_index
            if self.stall_at_source_index is not None:
                # CLI supplies an offset in seconds, independent of the trace's
                # absolute sample-index origin.
                self.stall_at_source_index += source_index
            self.pacing_epoch = time.perf_counter()
            if self.paced:
                self._start_ao_clock()
        if self.paced:
            lag_s = self._await_paced_source(source_index)
            self.source_availability_lags_s.append(lag_s)
            self._last_source_availability_lag_ms = lag_s * 1000.0
        if self.last_source_index is not None and source_index <= self.last_source_index:
            raise ValueError("replay source sample indices must increase monotonically")
        if self.last_source_index is not None:
            gap = source_index - self.last_source_index - self.config.chunk_size
            if gap > 0:
                self.metrics.source_index_gap_samples += gap
        self.last_source_index = source_index
        if values_mv.shape != (2, self.config.chunk_size):
            raise ValueError(f"processing input shape must be (2, {self.config.chunk_size})")
        start = time.perf_counter_ns()
        thread_cpu_start = time.thread_time_ns()
        process_cpu_start = time.process_time_ns()
        try:
            _, safe, esn_ms, _ = self.core.process(
                values_mv, sample_index=source_index,
                ao_target_sample_index=source_index + self.config.ao_lead_samples,
                input_is_electrode_mV=True, input_roles=roles, input_labels=labels,
                input_ctx_index=ctx_index,
            )
        finally:
            end = time.perf_counter_ns()
            thread_cpu_elapsed = time.thread_time_ns() - thread_cpu_start
            process_cpu_elapsed = time.process_time_ns() - process_cpu_start
            self.thread_cpu_ns.append(thread_cpu_elapsed)
            self.process_cpu_ns.append(process_cpu_elapsed)
            elapsed_ms = (end - start) / 1e6
            if elapsed_ms >= self.metrics.deadline_ms:
                self._slow_process_calls += 1
                if self._slow_process_calls <= 20:
                    related_gc = [
                        {"generation": generation, "pause_ms": pause / 1e6}
                        for gc_end, pause, generation in self._gc_events
                        if start <= gc_end <= end
                    ]
                    write_progress(self.progress_path, {
                        "type": "slow_core_process", "source_sample_index": source_index,
                        "wall_ms": elapsed_ms, "thread_cpu_ms": thread_cpu_elapsed / 1e6,
                        "process_cpu_ms": process_cpu_elapsed / 1e6,
                        "process_cpu_minus_thread_ms": max(0.0, process_cpu_elapsed - thread_cpu_elapsed) / 1e6,
                        "gc_pauses": related_gc,
                    })
        self.latencies_ms.append(elapsed_ms)
        self.metrics.processed_blocks += 1
        self.metrics.source_samples += int(np.count_nonzero(np.isfinite(values_mv[0])))
        self.metrics.deadline_exceeded_blocks += int(elapsed_ms > self.metrics.deadline_ms)
        diagnostics = self.core._last_diagnostics
        self.metrics.last_controller_ready = bool(diagnostics.get("ready", False))
        self.metrics.last_controller_calibrated = bool(diagnostics.get("calibrated", False))
        z_score = diagnostics.get("z_score")
        self.metrics.last_controller_z_score = float(z_score) if z_score is not None and np.isfinite(z_score) else None
        self.metrics.last_controller_reason = str(diagnostics.get("reason", ""))
        requested_rate = float(diagnostics.get("requested_rate_hz", 0) or 0)
        self.metrics.requested_rate_blocks += int(requested_rate > 0)
        self.metrics.max_requested_rate_hz = max(self.metrics.max_requested_rate_hz, requested_rate)
        if requested_rate > 0:
            self.metrics.bounded_rate_request_count += int(0 <= requested_rate <= 2.0)
        for event in self.core.last_pulse_events:
            status = str(event.status)
            if event.pulse_id in self._seen_pulse_events:
                continue
            self._seen_pulse_events.add(event.pulse_id)
            if status.startswith("rejected"):
                self.metrics.rejected_pulse_events += 1
            else:
                self.metrics.requested_pulse_events += 1
                if status in {"rendered_for_ao", "committed_tail_rendered"}:
                    self.metrics.rendered_pulse_events += 1

        snapshot = self.state.snapshot()
        base = self.base_sample_index
        target_block = ((source_index - base) // self.config.chunk_size) + self.config.ao_lead_chunks
        if snapshot.mode != "closed-loop" or snapshot.fault:
            safe = np.zeros_like(safe)

        pulse_ids = [event.pulse_id for event in self.core.last_pulse_events
                     if not str(event.status).startswith("rejected")]
        for pulse_id in pulse_ids:
            if pulse_id not in self.metrics.pulse_request_ids:
                self.metrics.pulse_request_ids.append(pulse_id)
        inject = (not self._stall_used and self.inject_stall_ms > 0
                  and self.stall_at_source_index is not None
                  and source_index == self.stall_at_source_index)
        inject |= (not self._stall_used and self.stall_on_first_pulse and bool(pulse_ids))
        if inject:
            self._stall_used = True
            self.metrics.injected_stall_count += 1
            self.metrics.injected_stall_ms_total += self.inject_stall_ms
            if self.paced:
                time.sleep(self.inject_stall_ms / 1000.0)
            else:
                # Advance the shared AO clock before submitting this block.
                # A command that missed its target cannot be accepted retroactively.
                self._advance_virtual(self.inject_stall_ms / 1000.0)
        command_generation = self.core.last_processed_generation
        scheduled = self.scheduler.submit(target_block, safe, command_generation,
                                          self.core.last_pulse_events,
                                          self.core.last_admitted_pulse_plans)
        if not scheduled:
            self.core.record_ao_outcome(command_generation, False, "late or stale simulated command rejected")
        scheduler_stats = self.scheduler.stats_snapshot()
        late_count = int(scheduler_stats.get("late_commands", 0))
        if late_count > self._observed_late_count:
            first_late = self._observed_late_count == 0
            self._observed_late_count = late_count
            if first_late:
                write_progress(self.progress_path, {
                    "type": "first_late_command", "source_sample_index": source_index,
                    "target_block": target_block,
                    "generated_sample_count": scheduler_stats.get("generated_sample_count"),
                    "next_write_block": getattr(self.scheduler, "_next_write_block", None),
                    "pacing_epoch_monotonic": self.pacing_epoch,
                    "observed_monotonic": time.perf_counter(), "ao": scheduler_stats,
                })
        active_misses = int(scheduler_stats.get("active_demand_misses", 0))
        if active_misses > self._observed_active_miss_count:
            first_miss = self._observed_active_miss_count == 0
            self._observed_active_miss_count = active_misses
            if first_miss:
                write_progress(self.progress_path, {
                    "type": "first_active_demand_miss",
                    "source_sample_index": source_index, "target_block": target_block,
                    "pulse_events": [asdict(event) for event in self.core.last_pulse_events],
                    "generated_sample_count": scheduler_stats.get("generated_sample_count"),
                    "next_write_block": getattr(self.scheduler, "_next_write_block", None),
                    "pacing_epoch_monotonic": self.pacing_epoch,
                    "observed_monotonic": time.perf_counter(), "ao": scheduler_stats,
                })
        generated = app.GeneratedAoBlock(self.scheduler.generated_sample_count, np.zeros(self.config.chunk_size),
                                         command_generation, "no_tick")
        if not self.paced:
            # The virtual source clock advances exactly one 5 ms block per input
            # block. Processing time does not stretch a finite replay's sample axis.
            blocks = self._advance_virtual(self.config.chunk_size / self.config.sample_rate)
            if blocks:
                generated = blocks[-1]
        self.telemetry.update_block(
            sample_index=self.core.sample_index, ai_ms=0.0, esn_ms=esn_ms, ao_ms=0.0,
            block_ms=elapsed_ms, deadline_ms=self.metrics.deadline_ms,
            logger_backlog=self.recorder.backlog,
        )
        state_after_ao = self.state.snapshot()
        if (not state_after_ao.transition_pending and not state_after_ao.armed
                and not state_after_ao.fault and arm_after_drain and stim_enabled):
            try:
                self.state.set_armed(True)
            except ValueError:
                pass
        if not stim_enabled and self.state.snapshot().armed:
            self.state.set_armed(False)

        now = time.monotonic()
        if now - self.last_progress_time >= self.progress_interval_s:
            write_progress(self.progress_path, {
                "type": "run_progress", "case_id": self.metrics.case_id,
                "controller": self.config.controller, "source_samples": self.metrics.source_samples,
                "skipped_source_samples": self.metrics.skipped_source_samples,
                "source_index_gap_samples": self.metrics.source_index_gap_samples,
                "source_availability_lag_ms": self._last_source_availability_lag_ms,
                "processed_blocks": self.metrics.processed_blocks,
                "elapsed_source_seconds": self.metrics.source_samples / self.config.sample_rate,
                "latency_p99_ms_so_far": float(np.percentile(self.latencies_ms, 99)),
                "ao": self.scheduler.stats_snapshot(),
                "recorder_backlog": self.recorder.backlog,
                "recording_path": str(self.recording_path),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })
            self.last_progress_time = now
        if self.paced:
            assert self.pacing_epoch is not None
            deadline = self.pacing_epoch + (self.metrics.processed_blocks + 1) * (
                self.config.chunk_size / self.config.sample_rate)
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
        if self._ao_clock_error is not None:
            raise RuntimeError(f"simulated AO clock failed: {self._ao_clock_error}") from self._ao_clock_error
        return generated

    def _collect_gc_timing(self, phase: str, info: dict[str, Any]) -> None:
        """Capture GC pause timing with an in-memory callback only."""
        generation = int(info.get("generation", -1))
        key = (threading.get_ident(), generation)
        now = time.perf_counter_ns()
        if phase == "start":
            self._gc_starts[key] = now
        elif phase == "stop":
            started = self._gc_starts.pop(key, None)
            if started is not None:
                pause = max(0, now - started)
                self._gc_pauses_ns.append(pause)
                self._gc_events.append((now, pause, generation))

    def _await_paced_source(self, source_index: int) -> float:
        """Wait until this finite-source block has acquired its original samples.

        If processing is late, the caller continues with this block and catches
        up through the buffered source in order. It never seeks to the latest
        wall-clock index or discards a valid replay block.
        """
        assert self.pacing_epoch is not None and self.base_sample_index is not None
        block_s = self.config.chunk_size / self.config.sample_rate
        available_at = self.pacing_epoch + (
            (source_index - self.base_sample_index) // self.config.chunk_size + 1
        ) * block_s
        remaining = available_at - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        return max(0.0, time.perf_counter() - available_at)

    def _consume_generated(self, block: app.GeneratedAoBlock) -> None:
        with self._generated_lock:
            self.metrics.generated_ao_blocks += 1
            self.metrics.ao_nonzero_generated_samples += int(np.count_nonzero(block.waveform_v))
            self.metrics.ao_committed_blocks += int(block.stage in {"ao_committed", "tail_preserved"})
            if np.any(block.waveform_v) and block.stage not in {"ao_committed", "tail_preserved"}:
                self.metrics.ao_noncommitted_nonzero_blocks += 1
            if self.on_generated:
                self.on_generated(block)

    def _instrument_component(self, owner: Any, method_name: str, metric_name: str) -> None:
        method = getattr(owner, method_name, None)
        if method is None:
            return

        def timed(*args, **kwargs):
            started = time.perf_counter_ns()
            try:
                return method(*args, **kwargs)
            finally:
                self.component_latencies_ns.setdefault(metric_name, []).append(
                    time.perf_counter_ns() - started)

        setattr(owner, method_name, timed)

    def _instrument_core_components(self) -> None:
        # Nested component timings are diagnostic and remain inside each full
        # process duration; they are never deducted from deadline evidence.
        self._instrument_component(self.core.safety, "process", "safety_process")
        self._instrument_component(self.core.actuator, "process", "actuator_process")
        self._instrument_component(self.recorder, "submit", "recorder_submit")
        self._instrument_component(self.core, "_publish_ui", "ui_submission")
        self._instrument_component(self.esn, "diagnostics", "esn_diagnostics")
        for name, controller in self.core.controllers.items():
            self._instrument_component(controller, "step", f"controller_{name}_step")
            envelope = getattr(controller, "envelope", None)
            if envelope is not None:
                self._instrument_component(envelope, "process", f"feature_{name}_envelope")

    def _advance_virtual(self, elapsed_seconds: float) -> tuple[app.GeneratedAoBlock, ...]:
        blocks = self.scheduler.advance_elapsed(elapsed_seconds)
        for block in blocks:
            self._consume_generated(block)
        return blocks

    def _start_ao_clock(self) -> None:
        if self._ao_clock_thread is not None:
            return
        self._ao_clock_thread = threading.Thread(target=self._ao_clock_loop, name="benchmark-simulated-ao-clock", daemon=True)
        self._ao_clock_thread.start()

    def _ao_clock_loop(self) -> None:
        last = time.perf_counter()
        while not self._ao_clock_stop.is_set():
            now = time.perf_counter()
            elapsed = now - last
            last = now
            self._ao_clock_max_gap_s = max(self._ao_clock_max_gap_s, elapsed)
            if elapsed > self.config.chunk_size / self.config.sample_rate:
                self._ao_clock_gap_count += 1
            try:
                self._advance_virtual(elapsed)
                state_fault = self.state.snapshot().fault
                if state_fault and state_fault != self._observed_ao_fault:
                    self._observed_ao_fault = state_fault
                    stats = self.scheduler.stats_snapshot()
                    write_progress(self.progress_path, {
                        "type": "ao_clock_fault", "fault": state_fault,
                        "source_sample_index": self.last_source_index,
                        "pulse_events": [asdict(event) for event in self.core.last_pulse_events],
                        "generated_sample_count": stats.get("generated_sample_count"),
                        "next_write_block": getattr(self.scheduler, "_next_write_block", None),
                        "pacing_epoch_monotonic": self.pacing_epoch,
                        "observed_monotonic": time.perf_counter(), "ao": stats,
                    })
            except BaseException as exc:
                self._ao_clock_error = exc
                self.state.fault(f"simulated AO clock failure: {type(exc).__name__}: {exc}")
                self._ao_clock_stop.set()
                return
            self._ao_clock_stop.wait(0.001)

    def finish(self) -> RunMetrics:
        self.state.set_armed(False)
        # Let the independent AO consumer cross the generation barrier and
        # drain the configured lead. Unpaced finite replays use exact virtual
        # sample time; paced runs keep consuming against the wall clock.
        drain_blocks = self.config.ao_lead_chunks + 4
        if self.paced:
            wait_until = time.monotonic() + (drain_blocks * self.config.chunk_size
                                             / self.config.sample_rate) + 2.0
            while time.monotonic() < wait_until:
                stats = self.scheduler.stats_snapshot()
                if (not self.state.snapshot().transition_pending
                        and stats["generated_sample_count"] >= self.config.ao_lead_samples
                        + self.metrics.source_samples + drain_blocks * self.config.chunk_size):
                    break
                if self._ao_clock_error is not None:
                    break
                time.sleep(0.002)
        else:
            for _ in range(drain_blocks):
                self._advance_virtual(self.config.chunk_size / self.config.sample_rate)
        self._stop_ao_clock()
        final_state = self.state.snapshot()
        self.metrics.ao_drain_complete = not final_state.transition_pending
        self.metrics.runtime_fault = final_state.fault
        if not self.metrics.ao_drain_complete:
            self.metrics.notes.append("AO drain barrier did not complete before run finalization")
        self.metrics.pulse_commit_ids = sorted(self._pulse_commit_ids)
        self.metrics.pulse_commit_samples = self.pulse_commit_samples
        self.metrics.ui_packets_submitted = self.hub.publish_count
        self.metrics.ui_packets_consumed = self.ui_consumed_count
        self.metrics.browser_connected = self.browser_connected
        if self.browser_connected:
            self.metrics.ui_transport_kind = "browser_and_websocket_client"
        self.metrics.recorder_max_backlog = self.telemetry.snapshot()["logger_high_water"]
        self.state.set_recording(False)
        self.recorder.stop_recording()
        self.recorder.close()
        try:
            with h5py.File(self.recording_path, "r") as recording:
                self.metrics.recording_completed = int(recording.attrs.get("complete", 0)) == 1
                self.metrics.recording_committed_samples = int(recording.attrs.get("committed_samples", 0))
        except (OSError, KeyError, TypeError, ValueError):
            self.metrics.recording_completed = False
        scheduler_stats = self.scheduler.stats_snapshot()
        self.metrics.ao_committed_blocks = int(scheduler_stats["committed_blocks"])
        self.metrics.ao_zero_substitutions = int(scheduler_stats["zero_substitutions"])
        self.metrics.ao_late_commands = int(scheduler_stats["late_commands"])
        self.metrics.ao_rejected_stale = int(scheduler_stats["rejected_stale"])
        self.metrics.ao_cancelled_blocks = int(scheduler_stats["cancelled_blocks"])
        self.metrics.ao_max_queue_depth = int(scheduler_stats["max_queue_depth"])
        self.metrics.ao_active_demand_misses = int(scheduler_stats.get("active_demand_misses", 0))
        self.metrics.ao_generated_sample_count = int(scheduler_stats["generated_sample_count"])
        self.metrics.ao_final_queue_depth = int(scheduler_stats.get("queue_depth", 0))
        self.metrics.ao_clock_max_service_gap_ms = self._ao_clock_max_gap_s * 1000
        self.metrics.ao_clock_gaps_over_block_period = self._ao_clock_gap_count
        self.metrics.duration_source_s = self.metrics.source_samples / self.config.sample_rate
        if self.latencies_ms:
            self.metrics.process_median_ms = float(np.median(self.latencies_ms))
            self.metrics.process_p95_ms = float(np.percentile(self.latencies_ms, 95))
            self.metrics.process_p99_ms = float(np.percentile(self.latencies_ms, 99))
            self.metrics.process_max_ms = float(np.max(self.latencies_ms))
        self.metrics.process_slow_call_count = self._slow_process_calls
        if self.thread_cpu_ns:
            cpu_ms = np.asarray(self.thread_cpu_ns, dtype=np.float64) / 1e6
            self.metrics.process_thread_cpu_median_ms = float(np.median(cpu_ms))
            self.metrics.process_thread_cpu_p95_ms = float(np.percentile(cpu_ms, 95))
            self.metrics.process_thread_cpu_max_ms = float(np.max(cpu_ms))
        if self.process_cpu_ns:
            cpu_ms = np.asarray(self.process_cpu_ns, dtype=np.float64) / 1e6
            self.metrics.process_cpu_median_ms = float(np.median(cpu_ms))
            self.metrics.process_cpu_p95_ms = float(np.percentile(cpu_ms, 95))
            self.metrics.process_cpu_max_ms = float(np.max(cpu_ms))
        self.metrics.gc_pause_count = len(self._gc_pauses_ns)
        if self._gc_pauses_ns:
            self.metrics.gc_pause_max_ms = max(self._gc_pauses_ns) / 1e6
        if self.source_availability_lags_s:
            lags_ms = np.asarray(self.source_availability_lags_s, dtype=np.float64) * 1000.0
            self.metrics.source_availability_lag_median_ms = float(np.median(lags_ms))
            self.metrics.source_availability_lag_p95_ms = float(np.percentile(lags_ms, 95))
            self.metrics.source_availability_lag_max_ms = float(np.max(lags_ms))
            self.metrics.source_backlog_blocks_max = int(np.floor(
                self.metrics.source_availability_lag_max_ms / self.metrics.block_ms))
        self.metrics.component_timings_ms = {
            name: {
                "count": len(samples),
                "median": float(np.median(samples) / 1e6),
                "p95": float(np.percentile(samples, 95) / 1e6),
                "p99": float(np.percentile(samples, 99) / 1e6),
                "max": float(np.max(samples) / 1e6),
            }
            for name, samples in self.component_latencies_ns.items() if samples
        }
        write_progress(self.progress_path, {
            "type": "run_complete", "case_id": self.metrics.case_id,
            "processed_blocks": self.metrics.processed_blocks, "ao": scheduler_stats,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        self.core.close_ui()
        self._run_on_loop(self._stop_ui_transport())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=3)
        self.loop.close()
        self._remove_gc_callback()
        return self.metrics

    def _remove_gc_callback(self) -> None:
        if self._gc_callback_registered:
            try:
                gc.callbacks.remove(self._gc_callback)
            except ValueError:
                pass
            self._gc_callback_registered = False

    def abort(self) -> None:
        """Stop worker threads and close an incomplete evidence file on failure."""
        try:
            self._stop_ao_clock()
            self.state.set_recording(False)
            self.recorder.stop_recording()
        finally:
            self.recorder.close()
            self.core.close_ui()
            self._run_on_loop(self._stop_ui_transport())
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop_thread.join(timeout=3)
            if not self.loop.is_closed():
                self.loop.close()
        self._remove_gc_callback()

    def _stop_ao_clock(self) -> None:
        self._ao_clock_stop.set()
        if self._ao_clock_thread is not None:
            self._ao_clock_thread.join(timeout=3.0)
            if self._ao_clock_thread.is_alive():
                raise TimeoutError("simulated AO clock did not stop")
            self._ao_clock_thread = None

    def _commit_actuator_event(self, pulse_id: str, samples: int) -> None:
        self.pulse_commit_samples += int(samples)
        self._pulse_commit_ids.add(str(pulse_id))
        self.core.commit_actuator_event(pulse_id, samples)

    async def _start_ui_transport(self):
        shutdown = asyncio.Event()
        server = await websockets.serve(
            lambda websocket: app.websocket_handler(
                websocket, self.state, self.recorder, self.hub, shutdown, self.config,
            ), self.config.ws_host, 0,
        )
        port = server.sockets[0].getsockname()[1]
        client = await websockets.connect(f"ws://{self.config.ws_host}:{port}")
        self.shutdown = shutdown
        self.ui_consumed_count = 0

        async def consume_packets() -> None:
            async for packet in client:
                parse_ui_packet(packet)
                self.ui_consumed_count += 1

        consumer = asyncio.create_task(consume_packets(), name="benchmark-websocket-ui-consumer")
        return server, client, consumer, port

    def _run_on_loop(self, awaitable):
        future = asyncio.run_coroutine_threadsafe(awaitable, self.loop)
        return future.result(timeout=10.0)

    async def _stop_ui_transport(self) -> None:
        if self.shutdown is not None:
            self.shutdown.set()
        await self.ui_client.close()
        self.ui_server.close()
        await self.ui_server.wait_closed()
        if not self.ui_consumer_task.done():
            self.ui_consumer_task.cancel()
            await asyncio.gather(self.ui_consumer_task, return_exceptions=True)


class _CountingUiHub(app.UiHub):
    def __init__(self, queue_size: int):
        super().__init__(queue_size)
        self.publish_count = 0

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def publish(self, packet: str) -> None:
        self.publish_count += 1
        super().publish(packet)


def run_replay_case(replay_path: Path, controller: str, *, lead_ms: float = 100.0,
                    limit_seconds: float | None = None, paced: bool = False,
                    progress_path: Path | None = None, progress_interval_s: float = 10.0,
                    recording_dir: Path | None = None, on_metrics: Callable[[RunMetrics], None] | None = None,
                    browser_ui: bool = False,
                    external_browser: bool = False,
                    browser_base_url: str = "http://127.0.0.1:8000",
                    expected_source_samples: int | None = None,
                    inject_stall_ms: float = 0.0,
                    stall_at_seconds: float | None = None) -> tuple[RunMetrics, dict[str, Any]]:
    replay_path = replay_path.resolve()
    if not replay_path.is_file():
        raise FileNotFoundError(
            f"replay file not found: {replay_path}. Import the canonical ABF files into "
            "recordings/ca3_slow_waves; see docs/replay_import.md, or pass an existing file with --trace."
        )
    replay = load_replay(replay_path)
    if replay.sample_rate_hz != 20_000:
        raise ValueError("benchmark trace must be 20 kHz")
    if expected_source_samples is not None and replay.sample_count != expected_source_samples:
        raise ValueError(
            f"{replay_path} has {replay.sample_count} samples; expected {expected_source_samples}"
        )
    if paced and limit_seconds is not None and limit_seconds > replay.duration_seconds:
        raise ValueError(
            f"paced run requested {limit_seconds:g}s but finite source ends at {replay.duration_seconds:g}s; "
            "repeat only with an explicit boundary reset and separate logged segments"
        )
    if recording_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="echo-chamber-benchmark-")
        record_dir = Path(tmp.name)
    else:
        tmp = None
        record_dir = recording_dir
        record_dir.mkdir(parents=True, exist_ok=True)
    config = _make_config(controller, lead_ms, record_dir)
    trace_meta = {
        "path": str(replay_path), "sha256": file_sha256(replay_path),
        "sample_count": replay.sample_count, "sample_rate_hz": replay.sample_rate_hz,
        "duration_seconds": replay.duration_seconds,
        "labels": replay.channel_labels, "roles": replay.channel_roles,
        "ctx_index": replay.ctx_index, "segments": [asdict(s) for s in replay.segments],
        "metadata": replay.metadata,
    }
    case_id = _run_id(replay_path.stem, controller, lead_ms)
    metadata = {
        "application": "Echo Chamber controller benchmark",
        "benchmark_case_id": case_id,
        "source": trace_meta,
        "configuration": app.serializable_config(config),
        "git_revision": git_revision(),
        "physical_pulse_claims": False,
    }
    harness = _Harness(config, replay_path.stem, metadata, progress_path=progress_path,
                       progress_interval_s=progress_interval_s, paced=paced,
                       simulation=False, on_generated=None, browser_ui=browser_ui,
                       external_browser=external_browser, browser_base_url=browser_base_url,
                       inject_stall_ms=inject_stall_ms,
                       stall_at_source_index=(int(round(stall_at_seconds * replay.sample_rate_hz))
                                              if stall_at_seconds is not None else None))
    limit_samples = int(round(limit_seconds * replay.sample_rate_hz)) if limit_seconds is not None else None
    try:
        for source_index, values in _replay_blocks(replay, limit_samples):
            harness.process(source_index, values, replay.channel_roles, replay.channel_labels, replay.ctx_index)
        metrics = harness.finish()
        if on_metrics:
            on_metrics(metrics)
    except BaseException:
        try:
            harness.abort()
        except Exception:
            pass
        raise
    finally:
        if tmp is not None:
            tmp.cleanup()
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "git_revision": git_revision(),
        "source_path": str(replay_path), "source_sha256": trace_meta["sha256"],
        "config": app.serializable_config(config),
        "recording_file": str(harness.recording_path),
        "recording_file_retained": recording_dir is not None,
        "ao_scheduler": harness.scheduler.stats_snapshot(),
        "core_telemetry": harness.telemetry.snapshot(),
        "ui_transport": {"kind": metrics.ui_transport_kind,
                         "browser_connected": metrics.browser_connected,
                         "packets_submitted": metrics.ui_packets_submitted,
                         "packets_consumed": metrics.ui_packets_consumed,
                         "browser_rendering_verified": False},
        "physical_pulse_claims": False,
    }
    return metrics, environment


def write_results(metrics: list[RunMetrics], environments: list[dict[str, Any]], output_dir: Path,
                  *, run_kind: str, acceptance: list[dict[str, Any]] | None = None) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"controller_{run_kind}_{stamp}.json"
    csv_path = output_dir / f"controller_{run_kind}_{stamp}.csv"
    document = {
        "schema": "echo-chamber-controller-benchmark-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(), "python": platform.python_version(),
        "machine": platform.platform(), "processor": platform.processor(),
        "reference": {"sample_rate_hz": 20_000, "block_samples": 100,
                      "block_ms": 5.0, "ao_lead_ms": 100.0,
                      "deadline_ms": 5.0, "lead_sweep_ms": list(LEAD_SWEEP_MS)},
        "physical_pulse_claims": False,
        "acceptance": acceptance or [],
        "cases": [asdict(x) for x in metrics], "environments": environments,
    }
    json_path.write_text(json.dumps(json_safe(document), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if metrics:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            fields = list(asdict(metrics[0]).keys()) + ["acceptance_passed", "acceptance_failures"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            acceptance_by_id = {item["case_id"]: item for item in (acceptance or [])}
            for item in metrics:
                row = asdict(item)
                evaluated = acceptance_by_id.get(item.case_id)
                row["acceptance_passed"] = evaluated["passed"] if evaluated else ""
                row["acceptance_failures"] = json.dumps(evaluated["failures"]) if evaluated else ""
                writer.writerow(row)
    else:
        csv_path.write_text("case_id\n", encoding="utf-8")
    return json_path, csv_path
