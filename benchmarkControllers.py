"""Reproducible benchmarks that exercise the production processing core."""
from __future__ import annotations

import asyncio
import copy
import csv
from collections import deque
import gc
import hashlib
import json
import math
import os
import platform
import queue
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit, urlunsplit
from dataclasses import asdict, dataclass, field, fields, replace
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
REFERENCE_COVERAGE = {
    "duration_seconds": 1800.0,
    "processed_blocks": 360_000,
    "source_samples": 36_000_000,
    "sample_rate_hz": 20_000.0,
    "block_samples": 100,
    "block_ms": 5.0,
    "deadline_ms": 5.0,
}
EXPECTED_CONTROLLERS = ("rule", "pid", "esn")
AO_FAULT_CORE_HISTORY_CAPACITY = 32


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
    recorder_backpressure_policy: str = "disabled"
    recorder_queue_capacity_blocks: int = 0
    recorder_backpressure_threshold_blocks: int = 0
    recorder_backpressure_wait_count: int = 0
    recorder_backpressure_wait_ms: float = 0.0
    recorder_backpressure_max_backlog: int = 0
    max_requested_rate_hz: float = 0.0
    actuator_rate_phase_high_water: float = 0.0
    actuator_rate_phase_reset_count: int = 0
    actuator_rate_phase_final: float = 0.0
    bounded_rate_request_count: int = 0
    deadline_exceeded_blocks: int = 0
    requested_rate_blocks: int = 0
    requested_pulse_events: int = 0
    rendered_pulse_events: int = 0
    rejected_pulse_events: int = 0
    unclassified_pulse_events: int = 0
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
    browser_disconnect_count: int = 0
    browser_disconnected_duration_ms: float = 0.0
    browser_connected_at_finish: bool = False
    browser_disconnect_open_at_finish: bool = False
    injected_stall_count: int = 0
    injected_stall_ms_total: float = 0.0
    runtime_fault: str | None = None
    progress_diagnostic_error: str | None = None
    progress_events_accepted: int = 0
    progress_events_written: int = 0
    progress_events_dropped: int = 0
    progress_events_rejected: int = 0
    progress_events_pending: int = 0
    progress_queue_high_water: int = 0
    progress_writer_close_timed_out: bool = False
    progress_writer_owner_alive: bool = False
    generated_ao_blocks: int = 0
    pulse_request_ids: list[str] = field(default_factory=list)
    pulse_rejected_ids: list[str] = field(default_factory=list)
    pulse_commit_ids: list[str] = field(default_factory=list)
    last_controller_ready: bool = False
    last_controller_calibrated: bool = False
    last_controller_z_score: float | None = None
    last_controller_reason: str = ""
    recording_active: bool = True
    physical_pulse_claims: bool = False
    simulation: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _CoreTimingContext:
    """Immutable, bounded context captured outside the measured Core call."""

    source_sample_index: int
    start_perf_counter_ns: int
    end_perf_counter_ns: int
    wall_ms: float
    thread_cpu_ms: float
    process_cpu_ms: float
    gc_overlap: bool
    source_availability_lag_ms: float | None
    source_availability_lag_sample_index: int | None
    mode: str
    acquiring: bool
    armed: bool
    transition_pending: bool
    generation: int
    core_processed_generation: int | None
    state_observation: str
    controller: str
    controller_ready: bool
    controller_calibrated: bool
    fault: str | None
    controller_reason: str
    pulse_events: tuple[Any, ...]


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
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _capture_actuator_rate_phase(metrics: RunMetrics, actuator: Any) -> dict[str, float | int]:
    """Record float64 actuator phase diagnostics separately from rounded H5 traces."""
    metrics.actuator_rate_phase_high_water = float(actuator.rate_phase_high_water)
    metrics.actuator_rate_phase_reset_count = int(actuator.rate_phase_reset_count)
    metrics.actuator_rate_phase_final = float(actuator.rate_phase)
    return {
        "high_water": metrics.actuator_rate_phase_high_water,
        "reset_count": metrics.actuator_rate_phase_reset_count,
        "final": metrics.actuator_rate_phase_final,
    }


def write_progress(path: Path | None, value: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_safe(value), allow_nan=False, separators=(",", ":")) + "\n")
        stream.flush()


@dataclass(frozen=True)
class _ProgressItem:
    sequence: int
    observed_perf_counter_ns: int
    payload: dict[str, Any]
    recent_core_window: tuple[tuple[float, int, int], ...] | None = None


class HarnessOwnerShutdownError(RuntimeError):
    """A per-run producer/diagnostic owner did not terminate before its deadline."""

    def __init__(self, owner_kind: str, status: dict[str, Any],
                 *, progress_path: Path | str | None = None) -> None:
        if owner_kind not in {"progress_writer", "simulated_ao"}:
            raise ValueError(f"unsupported harness owner kind: {owner_kind}")
        self.owner_kind = owner_kind
        self.progress_path = str(progress_path) if progress_path is not None else None
        self.retained_temporary_path: str | None = None
        self.refresh(status)
        super().__init__(self._message())

    def retain_temporary_path(self, path: Path) -> None:
        self.retained_temporary_path = str(path)
        self.add_note(
            "Temporary recording directory retained because a simulated AO owner may still use it.")

    def refresh(self, status: dict[str, Any]) -> None:
        # Keep status values JSON-safe and detached from mutable owner state.
        self.owner_status = {
            str(key): value for key, value in status.items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
        self.owner_unresolved = bool(status.get("owner_alive", False))
        self.original_error = status.get("error")
        self.accepted = int(status.get("accepted", 0)) if self.owner_kind == "progress_writer" else 0
        self.written = int(status.get("written", 0)) if self.owner_kind == "progress_writer" else 0
        self.dropped = int(status.get("dropped", 0)) if self.owner_kind == "progress_writer" else 0
        self.rejected = int(status.get("rejected", 0)) if self.owner_kind == "progress_writer" else 0
        self.pending = int(status.get("pending", 0)) if self.owner_kind == "progress_writer" else 0
        self.close_timed_out = bool(status.get("close_timed_out", False))
        self.args = (self._message(),)

    def _message(self) -> str:
        location = f" ({self.progress_path})" if self.progress_path is not None else ""
        counts = (
            f", accepted={self.accepted}, written={self.written}, dropped={self.dropped}, "
            f"rejected={self.rejected}, pending={self.pending}"
            if self.owner_kind == "progress_writer" else ""
        )
        return (f"{self.owner_kind} shutdown incomplete{location}: "
                f"owner_unresolved={self.owner_unresolved}{counts}; "
                f"error={self.original_error or 'none'}")

    def failure_fields(self) -> dict[str, Any]:
        fields = {
            "shutdown_owner_kind": self.owner_kind,
            "owner_unresolved": self.owner_unresolved,
            "shutdown_owner_error": self.original_error,
            "shutdown_owner_status": self.owner_status,
        }
        if self.progress_path is not None:
            fields["progress_path"] = self.progress_path
        if self.retained_temporary_path is not None:
            fields["retained_temporary_path"] = self.retained_temporary_path
        if self.owner_kind == "progress_writer":
            fields.update({
                "progress_events_accepted": self.accepted,
                "progress_events_written": self.written,
                "progress_events_dropped": self.dropped,
                "progress_events_rejected": self.rejected,
                "progress_events_pending": self.pending,
                "progress_writer_close_timed_out": self.close_timed_out,
            })
        return fields


class _HarnessProgressWriter:
    """Bounded single-owner progress serialization and file writer."""

    _STOP = object()

    def __init__(self, path: Path, *, capacity: int = 256,
                 close_timeout_s: float = 5.0) -> None:
        if capacity <= 0:
            raise ValueError("progress writer capacity must be positive")
        if close_timeout_s <= 0:
            raise ValueError("progress writer close timeout must be positive")
        self.path = Path(path)
        self.close_timeout_s = float(close_timeout_s)
        self._queue: queue.Queue[_ProgressItem | object] = queue.Queue(maxsize=capacity)
        self._admission_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._closed = False
        self._stop_enqueued = False
        self._write_failed = False
        self._first_error: str | None = None
        self._sequence = 0
        self._accepted = 0
        self._written = 0
        self._dropped = 0
        self._rejected = 0
        self._high_water = 0
        self._close_timed_out = False
        self._thread = threading.Thread(
            target=self._run, name="harness-progress-writer", daemon=True)
        self._thread.start()

    def _latch_error_locked(self, message: str) -> None:
        if self._first_error is None:
            self._first_error = message

    def submit(self, value: dict[str, Any], *,
               recent_core_window: tuple[tuple[float, int, int], ...] | None = None) -> bool:
        """Detach and enqueue without waiting for the file writer."""
        try:
            payload = copy.deepcopy(dict(value))
            window = tuple(recent_core_window) if recent_core_window is not None else None
        except BaseException as exc:
            with self._state_lock:
                self._latch_error_locked(
                    f"could not detach progress observation: {type(exc).__name__}: {exc}")
                self._rejected += 1
            return False

        with self._admission_lock:
            with self._state_lock:
                if self._closed:
                    self._latch_error_locked("progress observation submitted after writer close")
                    self._rejected += 1
                    return False
                self._sequence += 1
                sequence = self._sequence
            item = _ProgressItem(
                sequence=sequence,
                observed_perf_counter_ns=time.perf_counter_ns(),
                payload=payload,
                recent_core_window=window,
            )
            try:
                with self._state_lock:
                    self._queue.put_nowait(item)
                    self._accepted += 1
                    self._high_water = max(self._high_water, self._queue.qsize())
            except queue.Full:
                with self._state_lock:
                    self._latch_error_locked(
                        f"progress writer queue overflow at sequence {sequence} "
                        f"(capacity {self._queue.maxsize})")
                    self._rejected += 1
                return False
            return True

    def _add_recent_window_fields(self, payload: dict[str, Any],
                                  window: tuple[tuple[float, int, int], ...] | None) -> None:
        if payload.get("type") != "run_progress" or window is None:
            return
        payload["recent_core_window_count"] = len(window)
        payload["recent_core_window_start_sample_index"] = window[0][1] if window else None
        payload["recent_core_window_end_sample_index"] = window[-1][2] if window else None
        payload["recent_core_window_range_convention"] = "half-open [start, end)"
        payload["recent_core_window_scope"] = (
            "last up to 2048 completed Core calls; includes inactive and post-fault calls"
        )
        payload["recent_core_p99_ms"] = (
            float(np.percentile([entry[0] for entry in window], 99)) if window else None
        )

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                assert isinstance(item, _ProgressItem)
                with self._state_lock:
                    skip_write = self._write_failed
                if skip_write:
                    with self._state_lock:
                        self._dropped += 1
                    continue
                payload = item.payload
                payload["progress_sequence"] = item.sequence
                payload["progress_observed_perf_counter_ns"] = item.observed_perf_counter_ns
                self._add_recent_window_fields(payload, item.recent_core_window)
                try:
                    # json_safe, strict JSON encoding, directory creation, and
                    # all file writes stay on this one owner thread.
                    write_progress(self.path, payload)
                except BaseException as exc:
                    with self._state_lock:
                        self._write_failed = True
                        self._latch_error_locked(
                            f"progress writer failed: {type(exc).__name__}: {exc}")
                        self._dropped += 1
                else:
                    with self._state_lock:
                        self._written += 1
            except BaseException as exc:
                with self._state_lock:
                    self._write_failed = True
                    self._latch_error_locked(
                        f"progress worker failed: {type(exc).__name__}: {exc}")
                    self._dropped += 1
            finally:
                self._queue.task_done()

    def close(self) -> dict[str, Any]:
        """Stop admission and make a bounded attempt to drain/join the writer."""
        with self._close_lock:
            first_timeout = False
            # Synchronize with a submit already inside admission, then publish
            # closed and release this lock before waiting for queue capacity.
            # Later submitters can now reject promptly while the owner drains.
            with self._admission_lock:
                with self._state_lock:
                    self._closed = True
                    should_signal = not self._stop_enqueued
            if should_signal:
                try:
                    self._queue.put(self._STOP, timeout=self.close_timeout_s)
                except queue.Full:
                    with self._state_lock:
                        first_timeout = not self._close_timed_out
                        self._close_timed_out = True
                        self._latch_error_locked(
                            "timed out waiting to enqueue progress-writer stop marker; "
                            "accepted observations remain pending")
                else:
                    with self._state_lock:
                        self._stop_enqueued = True
            if first_timeout:
                sys.stderr.write(
                    "Progress writer shutdown timed out before stop marker admission; owner remains unresolved.\n")
            self._thread.join(timeout=self.close_timeout_s)
            if self._thread.is_alive():
                first_timeout = False
                with self._state_lock:
                    first_timeout = not self._close_timed_out
                    self._close_timed_out = True
                    self._latch_error_locked(
                        "timed out joining progress writer; owner remains unresolved")
                if first_timeout:
                    sys.stderr.write(
                        "Progress writer shutdown timed out while owner remains alive; case will stop.\n")
            with self._state_lock:
                pending = max(0, self._accepted - self._written - self._dropped)
                return {
                    "error": self._first_error,
                    "accepted": self._accepted,
                    "written": self._written,
                    "dropped": self._dropped,
                    "rejected": self._rejected,
                    "pending": pending,
                    "queue_high_water": self._high_water,
                    "closed": self._closed,
                    "owner_alive": self._thread.is_alive(),
                    "stop_enqueued": self._stop_enqueued,
                    "close_timed_out": self._close_timed_out,
                }

    @property
    def error(self) -> str | None:
        with self._state_lock:
            return self._first_error


def acceptance_failures(metrics: RunMetrics) -> list[str]:
    """Evaluate runtime checks that apply to both diagnostics and full cases."""
    failures: list[str] = []
    if metrics.runtime_fault:
        failures.append(f"runtime fault was latched: {metrics.runtime_fault}")
    if metrics.progress_diagnostic_error:
        failures.append(f"progress diagnostics failed: {metrics.progress_diagnostic_error}")
    if metrics.processed_blocks <= 0:
        failures.append("no processing blocks completed")
    if metrics.skipped_source_samples or metrics.source_index_gap_samples:
        failures.append("paced source data was skipped or has an index gap")
    if not np.isfinite(metrics.process_p99_ms) or metrics.process_p99_ms >= 2.5:
        failures.append("process p99 is not below 2.5 ms")
    if not np.isfinite(metrics.process_max_ms) or metrics.process_max_ms >= REFERENCE_COVERAGE["deadline_ms"]:
        failures.append("maximum process time reached or exceeded the block deadline")
    if metrics.deadline_ms != REFERENCE_COVERAGE["deadline_ms"]:
        failures.append("configured process deadline does not match the 5 ms reference")
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
        failures.append("required browser UI never connected")
    if metrics.browser_ui_required and metrics.browser_disconnect_count:
        failures.append("required browser UI disconnected during the run")
    if metrics.browser_ui_required and not metrics.browser_connected_at_finish:
        failures.append("required browser UI was disconnected at run completion")
    if (metrics.unclassified_pulse_events
            or metrics.requested_pulse_events != metrics.rendered_pulse_events + metrics.rejected_pulse_events):
        failures.append("requested pulse events were not all categorized as rendered or rejected")
    return failures


def _observed_coverage(metrics: RunMetrics) -> dict[str, float | int]:
    return {
        "duration_seconds": float(metrics.duration_source_s),
        "processed_blocks": int(metrics.processed_blocks),
        "source_samples": int(metrics.source_samples),
        "sample_rate_hz": float(metrics.sample_rate_hz),
        "block_samples": int(metrics.block_samples),
        "block_ms": float(metrics.block_ms),
        "deadline_ms": float(metrics.deadline_ms),
    }


def _coverage_failures(metrics: RunMetrics) -> list[str]:
    observed = _observed_coverage(metrics)
    failures = []
    for key in ("duration_seconds", "processed_blocks", "source_samples", "sample_rate_hz",
                "block_samples", "block_ms", "deadline_ms"):
        expected = REFERENCE_COVERAGE[key]
        actual = observed[key]
        if key == "duration_seconds":
            expected_s = float(expected)
            sample_period_s = 1 / float(REFERENCE_COVERAGE["sample_rate_hz"])
            numeric_epsilon_s = 2 * math.ulp(expected_s)
            matches = (math.isfinite(float(actual))
                       and abs(float(actual) - expected_s) <= sample_period_s + numeric_epsilon_s)
        else:
            matches = actual == expected
        if not matches:
            failures.append(f"full replay coverage {key} is {actual}, expected {expected}")
    return failures


def acceptance_scope(run_kind: str, metrics: RunMetrics,
                     requested_scope: str | None = None) -> str:
    """Classify intentional short runs as diagnostics, malformed full runs as failures."""
    if run_kind not in {"matrix", "paced"} or requested_scope == "diagnostic_prefix":
        return "diagnostic_prefix"
    if requested_scope in {"full_replay_case", "paced_30min_case"}:
        return "paced_30min_case" if run_kind == "paced" else "full_replay_case"
    # Direct callers without CLI intent can infer a short prefix from duration.
    expected_duration = float(REFERENCE_COVERAGE["duration_seconds"])
    observed_duration = float(metrics.duration_source_s)
    sample_period_s = 1 / float(REFERENCE_COVERAGE["sample_rate_hz"])
    numeric_epsilon_s = 2 * math.ulp(expected_duration)
    if (not math.isfinite(observed_duration)
            or observed_duration < expected_duration - sample_period_s - numeric_epsilon_s):
        return "diagnostic_prefix"
    return "paced_30min_case" if run_kind == "paced" else "full_replay_case"


def evaluate_case_acceptance(metrics: RunMetrics, run_kind: str,
                             requested_scope: str | None = None) -> dict[str, Any]:
    """Separate mode-independent runtime checks from duration-scoped acceptance."""
    scope = acceptance_scope(run_kind, metrics, requested_scope)
    runtime_failures = acceptance_failures(metrics)
    coverage_failures = _coverage_failures(metrics)
    observed = _observed_coverage(metrics)
    coverage_complete = not coverage_failures
    if scope == "diagnostic_prefix":
        full_failures: list[str] = []
        full_passed: bool | None = None
        full_status = "NOT EVALUATED"
    else:
        full_failures = [*runtime_failures, *coverage_failures]
        full_passed = not full_failures
        full_status = "PASS" if full_passed else "FAIL"
    return {
        "case_id": metrics.case_id,
        "scope": scope,
        "runtime_checks_passed": not runtime_failures,
        "runtime_failures": runtime_failures,
        "coverage": {
            "required": dict(REFERENCE_COVERAGE),
            "observed": observed,
            "complete": coverage_complete,
            "failures": coverage_failures,
        },
        "full_acceptance_status": full_status,
        "full_acceptance_passed": full_passed,
        # `passed` remains for readers of the original v1 schema; it now means
        # full acceptance only and is null for diagnostic prefixes.
        "passed": full_passed,
        "failures": full_failures if full_passed is not None else runtime_failures,
    }


def failed_case_acceptance(case_id: str, failure: dict[str, Any],
                           requested_scope: str) -> dict[str, Any]:
    reason = f"{failure.get('exception_type', 'Exception')}: {failure.get('message', 'case failed')}"
    full_passed = False if requested_scope != "diagnostic_prefix" else None
    return {
        "case_id": case_id,
        "scope": requested_scope,
        "runtime_checks_passed": False,
        "runtime_failures": [reason],
        "coverage": {"required": dict(REFERENCE_COVERAGE), "observed": None,
                     "complete": False, "failures": ["case produced no complete coverage metrics"]},
        "full_acceptance_status": "FAIL" if full_passed is False else "NOT EVALUATED",
        "full_acceptance_passed": full_passed,
        "passed": full_passed,
        "failures": [reason],
    }


def evaluate_collection_acceptance(run_kind: str, metrics: list[RunMetrics],
                                   case_failures: list[dict[str, Any]], *,
                                   requested_scope: str, finalized: bool) -> dict[str, Any]:
    """Keep the nine-case/three-controller collection gate separate from case scope."""
    if run_kind not in {"matrix", "paced"} or requested_scope == "diagnostic_prefix":
        return {
            "collection_kind": run_kind,
            "status": "NOT EVALUATED",
            "full_acceptance_passed": None,
            "required": None,
            "observed": None,
            "failures": [],
        }
    required = {
        "case_count": 9 if run_kind == "matrix" else 3,
        "controller_count": 3,
        "controllers": list(EXPECTED_CONTROLLERS),
        "trace_count": 3 if run_kind == "matrix" else None,
        "unique_source_controller_pairs": 9 if run_kind == "matrix" else 3,
    }
    source_names = {item.source_name for item in metrics}
    controllers = {item.controller for item in metrics}
    pairs = {(item.source_name, item.controller) for item in metrics}
    case_scopes = [acceptance_scope(run_kind, item, requested_scope) for item in metrics]
    case_acceptance = [evaluate_case_acceptance(item, run_kind, requested_scope) for item in metrics]
    full_acceptance_failed_ids = [
        item["case_id"] for item in case_acceptance
        if item["full_acceptance_passed"] is False
    ]
    observed = {
        "completed_case_count": len(metrics),
        "failed_case_count": len(case_failures),
        "full_scope_case_count": sum(scope != "diagnostic_prefix" for scope in case_scopes),
        "source_count": len(source_names),
        "source_names": sorted(source_names),
        "controller_count": len(controllers),
        "controllers": sorted(controllers),
        "unique_source_controller_pairs": len(pairs),
        "full_acceptance_passed_case_count": sum(
            item["full_acceptance_passed"] is True for item in case_acceptance),
        "failed_full_acceptance_case_ids": full_acceptance_failed_ids,
    }
    failures = []
    if case_failures:
        failures.append(f"{len(case_failures)} case(s) failed before producing metrics")
    runtime_failed = [item.case_id for item in metrics if acceptance_failures(item)]
    if runtime_failed:
        failures.append(f"runtime checks failed for {len(runtime_failed)} completed case(s)")
    if full_acceptance_failed_ids:
        failures.append(
            f"full acceptance failed for {len(full_acceptance_failed_ids)} completed case(s)")
    if run_kind == "matrix":
        complete_collection = (
            len(metrics) == 9 and not case_failures and len(source_names) == 3
            and controllers == set(EXPECTED_CONTROLLERS) and len(pairs) == 9
            and all(scope == "full_replay_case" for scope in case_scopes)
            and all(item["full_acceptance_passed"] is True for item in case_acceptance)
        )
        if not complete_collection:
            failures.append("full matrix requires all 9 source/controller cases across 3 traces")
    else:
        complete_collection = (
            len(metrics) == 3 and not case_failures
            and controllers == set(EXPECTED_CONTROLLERS) and len(pairs) == 3
            and all(scope == "paced_30min_case" for scope in case_scopes)
            and all(item["full_acceptance_passed"] is True for item in case_acceptance)
        )
        if not complete_collection:
            failures.append("paced collection requires one full 30-minute case for each of 3 controllers")
    if not finalized:
        return {
            "collection_kind": run_kind,
            "status": "IN PROGRESS",
            "full_acceptance_passed": None,
            "required": required,
            "observed": observed,
            "failures": failures,
        }
    if failures:
        status = "FAIL" if runtime_failed or case_failures or full_acceptance_failed_ids else "INCOMPLETE"
        return {
            "collection_kind": run_kind,
            "status": status,
            "full_acceptance_passed": False,
            "required": required,
            "observed": observed,
            "failures": failures,
        }
    status = "PASS"
    return {
        "collection_kind": run_kind,
        "status": status,
        "full_acceptance_passed": True,
        "required": required,
        "observed": observed,
        "failures": [],
    }


def _record_pulse_event_metrics(metrics: RunMetrics, seen: set[str], events: Iterable[Any]) -> None:
    """Count each proposed pulse ID once and classify every status explicitly."""
    rendered_statuses = {"rendered_for_ao", "committed_tail_rendered"}
    for event in events:
        pulse_id = str(event.pulse_id)
        if pulse_id in seen:
            continue
        seen.add(pulse_id)
        metrics.requested_pulse_events += 1
        metrics.pulse_request_ids.append(pulse_id)
        status = str(event.status)
        if status.startswith("rejected"):
            metrics.rejected_pulse_events += 1
            metrics.pulse_rejected_ids.append(pulse_id)
        elif status in rendered_statuses:
            metrics.rendered_pulse_events += 1
        else:
            metrics.unclassified_pulse_events += 1
            metrics.notes.append(f"unclassified pulse event {pulse_id}: {status}")


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


class _CommittedAoRangeCounter:
    """Count confirmed samples once using the scheduler's monotonic AO ranges.

    AO outcomes are published by one ordered writer. Keeping the greatest
    committed exclusive sample index makes duplicate or delayed old ranges
    idempotent with constant memory.
    """

    def __init__(self) -> None:
        self._committed_through = 0

    def add(self, pulse_id: str, start_sample_index: int, end_sample_index: int) -> int:
        start, end = int(start_sample_index), int(end_sample_index)
        if end <= start:
            return 0
        newly_committed = max(0, end - max(start, self._committed_through))
        self._committed_through = max(self._committed_through, end)
        return newly_committed


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


def _release_temporary_recording_directory(path: Path | None,
                                           active_error: BaseException | None) -> None:
    if path is None or not path.exists():
        return
    if (isinstance(active_error, HarnessOwnerShutdownError)
            and active_error.owner_kind == "simulated_ao"
            and active_error.owner_unresolved):
        active_error.retain_temporary_path(path)
        return
    try:
        shutil.rmtree(path)
    except OSError as exc:
        if active_error is None:
            raise
        active_error.add_note(
            f"Could not remove temporary recording directory {path}: "
            f"{type(exc).__name__}: {exc}")


class _Harness:
    UNPACED_RECORDER_NO_PROGRESS_TIMEOUT_S = 30.0

    def __init__(self, config: app.AppConfig, source_name: str, metadata: dict[str, Any],
                 *, progress_path: Path | None, progress_interval_s: float,
                 paced: bool, simulation: bool, on_generated: Callable[[app.GeneratedAoBlock], None] | None,
                 browser_ui: bool = False, external_browser: bool = False,
                 browser_base_url: str = "http://127.0.0.1:8000",
                 inject_stall_ms: float = 0.0,
                 stall_at_source_index: int | None = None, stall_on_first_pulse: bool = False):
        self.config, self.progress_path = config, progress_path
        self._progress_writer: _HarnessProgressWriter | None = None
        self._progress_writer_init_lock = threading.Lock()
        self._progress_diagnostic_lock = threading.Lock()
        self._progress_diagnostic_error: str | None = None
        self._recent_core_latencies: deque[tuple[float, int, int]] = deque(maxlen=2048)
        self.progress_interval_s, self.paced = progress_interval_s, paced
        self.on_generated = on_generated
        self.inject_stall_ms = float(inject_stall_ms)
        self.stall_at_source_index = stall_at_source_index
        self.stall_on_first_pulse = stall_on_first_pulse
        self._stall_used = False
        self.browser_ui_required = bool(browser_ui)
        self._browser_presence = _BrowserPresenceTracker()
        self.esn = app.EsnRuntime(config.model_artifact, config)
        self.state = app.RuntimeState(
            esn_ready=self.esn.ready, start_paused=False,
            electrode_labels=config.electrode_labels, ctx_index=config.ctx_index,
            pulse_threshold_std=config.pulse_threshold_std,
            pulse_window_sec=config.pulse_window_sec,
            controller=config.controller,
            controller_ready={"esn": self.esn.ready, "rule": True, "pid": True},
            controller_settings=config.controller_settings,
        )
        self.telemetry = app.Telemetry()
        self.hub = _CountingUiHub(config.ui_queue_packets, self._browser_presence)
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.loop.run_forever, name="benchmark-ui-loop", daemon=True)
        self.loop_thread.start()
        self._ao_clock_stop = threading.Event()
        self._ao_clock_thread: threading.Thread | None = None
        self._ao_clock_error: BaseException | None = None
        self._ao_clock_stop_timed_out = False
        self._pending_ao_shutdown_error: HarnessOwnerShutdownError | None = None
        self._ao_stop_timeout_s = 3.0
        self._observed_ao_fault: str | None = None
        self._observed_late_count = 0
        self._observed_active_miss_count = 0
        self._ao_clock_max_gap_s = 0.0
        self._ao_clock_gap_count = 0
        self.source_availability_lags_s: list[float] = []
        self._last_source_availability_lag_ms = 0.0
        self._active_core_source_index: int | None = None
        self._last_core_call_started_source_index: int | None = None
        self._latest_source_sample_index_seen: int | None = None
        self._active_core_source_lag_ms: float | None = None
        self._active_core_source_lag_index: int | None = None
        self._recent_core_rows: deque[_CoreTimingContext] = deque(
            maxlen=AO_FAULT_CORE_HISTORY_CAPACITY)
        self._recent_core_lock = threading.Lock()
        self._gc_event_cursor = 0
        self._generated_lock = threading.Lock()
        self._progress_writer_result: dict[str, Any] | None = None
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
        if browser_ui:
            url = _browser_connection_url(browser_base_url, self.ui_port)
            if external_browser:
                self._write_progress({
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
                writer_error = self._progress_writer_error()
                if writer_error is not None:
                    self.abort()
                    raise RuntimeError(f"progress diagnostics failed during browser startup: {writer_error}")
                time.sleep(0.05)
            if not self._browser_presence.snapshot(time.perf_counter_ns())["ever_connected"]:
                self.abort()
                raise TimeoutError("browser UI did not connect to the benchmark WebSocket")
        try:
            self._initialize_harness_runtime(config, source_name, simulation, paced)
        except BaseException as primary_error:
            try:
                cleanup_errors = self.abort()
            except HarnessOwnerShutdownError as cleanup_error:
                cleanup_error.add_note(
                    f"Harness initialization also failed: "
                    f"{type(primary_error).__name__}: {primary_error}")
                raise
            except BaseException as cleanup_error:
                primary_error.add_note(
                    f"Progress-writer initialization cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}")
            else:
                if cleanup_errors:
                    primary_error.add_note(
                        "Additional initialization cleanup failures: " + " | ".join(cleanup_errors))
            raise

    def _initialize_harness_runtime(self, config: app.AppConfig, source_name: str,
                                    simulation: bool, paced: bool) -> None:
        self.pulse_commit_samples = 0
        self._pulse_commit_ids: set[str] = set()
        self._pulse_commit_ranges = _CommittedAoRangeCounter()
        self._pulse_commit_lock = threading.Lock()
        self.scheduler = self._create_scheduler(config)
        self.metrics = RunMetrics(
            case_id=self.recording_path.stem, source_name=source_name, controller=config.controller,
            sample_rate_hz=config.sample_rate, block_samples=config.chunk_size,
            block_ms=config.processing_block_ms, ao_lead_ms=config.ao_target_lead_ms,
            deadline_ms=config.chunk_size * 1000 / config.sample_rate,
            simulation=simulation,
            browser_ui_required=self.browser_ui_required,
            ao_clock_mode="independent_monotonic_wall" if paced else "virtual_5ms",
            recorder_backpressure_policy=("disabled_for_paced" if paced else "bounded_queue_wait_between_blocks"),
            recorder_queue_capacity_blocks=config.logger_queue_blocks,
            recorder_backpressure_threshold_blocks=max(1, math.floor(config.logger_queue_blocks * 0.8)),
        )
        self.component_latencies_ns: dict[str, list[int]] = {}
        self._instrument_core_components()
        self.latencies_ms: list[float] = []
        self.thread_cpu_ns: list[int] = []
        self.process_cpu_ns: list[int] = []
        self._gc_starts: dict[tuple[int, int], int] = {}
        self._gc_pauses_ns: list[int] = []
        self._gc_events: list[tuple[int, int, int]] = []
        # GC callbacks can run reentrantly while a same-thread snapshot is
        # allocating; RLock avoids deadlocking the diagnostic path.
        self._gc_lock = threading.RLock()
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

    def _write_progress(self, value: dict[str, Any], *,
                        recent_core_window: tuple[tuple[float, int, int], ...] | None = None
                        ) -> bool:
        """Queue a detached progress observation without waiting for disk I/O."""
        if self.progress_path is None:
            return True
        try:
            writer = getattr(self, "_progress_writer", None)
            if writer is None:
                init_lock = getattr(self, "_progress_writer_init_lock", None)
                if init_lock is None:
                    init_lock = threading.Lock()
                    self._progress_writer_init_lock = init_lock
                with init_lock:
                    writer = getattr(self, "_progress_writer", None)
                    if writer is None:
                        writer = _HarnessProgressWriter(self.progress_path)
                        self._progress_writer = writer
            return writer.submit(value, recent_core_window=recent_core_window)
        except BaseException as exc:
            self._latch_progress_diagnostic_error(
                f"could not enqueue progress observation: {type(exc).__name__}: {exc}")
            return False

    def _latch_progress_diagnostic_error(self, message: str) -> None:
        lock = getattr(self, "_progress_diagnostic_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._progress_diagnostic_lock = lock
        with lock:
            if getattr(self, "_progress_diagnostic_error", None) is None:
                self._progress_diagnostic_error = message

    def _close_progress_writer(self) -> dict[str, Any] | None:
        writer = getattr(self, "_progress_writer", None)
        result = writer.close() if writer is not None else None
        if result is None and getattr(self, "_progress_diagnostic_error", None) is None:
            return None
        error = self._progress_diagnostic_error
        if result is not None:
            writer_error = result["error"]
            if error is None:
                error = writer_error
            if error is None and result.get("pending", 0):
                error = "progress writer exited with accepted observations still pending"
            if error is None and result.get("owner_alive"):
                error = "progress writer owner remains alive after shutdown timeout"
            result["error"] = error
            self._progress_writer_result = result
        else:
            result = {
                "error": error, "accepted": 0, "written": 0, "dropped": 0,
                "pending": 0, "queue_high_water": 0, "closed": True,
                "owner_alive": False, "close_timed_out": False,
            }
            self._progress_writer_result = result
        metrics = getattr(self, "metrics", None)
        if metrics is not None:
            metrics.progress_diagnostic_error = error
            metrics.progress_events_accepted = int(result.get("accepted", 0))
            metrics.progress_events_written = int(result["written"])
            metrics.progress_events_dropped = int(result["dropped"])
            metrics.progress_events_rejected = int(result.get("rejected", 0))
            metrics.progress_events_pending = int(result.get("pending", 0))
            metrics.progress_queue_high_water = int(result["queue_high_water"])
            metrics.progress_writer_close_timed_out = bool(result.get("close_timed_out", False))
            metrics.progress_writer_owner_alive = bool(result.get("owner_alive", False))
        if result.get("owner_alive"):
            raise HarnessOwnerShutdownError(
                "progress_writer", result, progress_path=self.progress_path)
        return result

    def _progress_writer_error(self) -> str | None:
        if getattr(self, "_progress_diagnostic_error", None) is not None:
            return self._progress_diagnostic_error
        writer = getattr(self, "_progress_writer", None)
        return writer.error if writer is not None else None

    def _record_core_latency(self, source_index: int, elapsed_ms: float) -> None:
        self.latencies_ms.append(float(elapsed_ms))
        self._recent_core_latencies.append((
            float(elapsed_ms), int(source_index), int(source_index + self.config.chunk_size)))

    def _mark_core_call_started(self, source_index: int,
                                source_lag_ms: float | None) -> None:
        with self._recent_core_lock:
            self._active_core_source_index = int(source_index)
            self._last_core_call_started_source_index = int(source_index)
            self._active_core_source_lag_ms = source_lag_ms
            self._active_core_source_lag_index = int(source_index) if source_lag_ms is not None else None

    def _remember_completed_core_call(self, source_index: int, start_ns: int, end_ns: int,
                                     thread_cpu_ns: int, process_cpu_ns: int,
                                     source_lag_ms: float | None) -> _CoreTimingContext:
        """Capture immutable context after the timed Core interval has ended.

        The state and controller observations are sampled after Core and are
        not asserted to be the exact state consumed by that call.
        """
        with self._gc_lock:
            new_gc_events = tuple(self._gc_events[self._gc_event_cursor:])
            self._gc_event_cursor = len(self._gc_events)
            active_gc_starts = tuple(self._gc_starts.values())
        gc_overlap = any(
            gc_end - pause < end_ns and gc_end > start_ns
            for gc_end, pause, _generation in new_gc_events
        ) or any(gc_start < end_ns for gc_start in active_gc_starts)
        state_after_core = self.state.snapshot()
        diagnostics_after_core = dict(getattr(self.core, "_last_diagnostics", {}))
        pulse_events = tuple(getattr(self.core, "last_pulse_events", ()))
        row = _CoreTimingContext(
            source_sample_index=int(source_index),
            start_perf_counter_ns=int(start_ns), end_perf_counter_ns=int(end_ns),
            wall_ms=float((end_ns - start_ns) / 1e6),
            thread_cpu_ms=float(thread_cpu_ns / 1e6),
            process_cpu_ms=float(process_cpu_ns / 1e6), gc_overlap=bool(gc_overlap),
            source_availability_lag_ms=source_lag_ms,
            source_availability_lag_sample_index=(
                int(source_index) if source_lag_ms is not None else None),
            mode=str(state_after_core.mode), acquiring=bool(state_after_core.acquiring),
            armed=bool(state_after_core.armed),
            transition_pending=bool(state_after_core.transition_pending),
            generation=int(state_after_core.generation),
            core_processed_generation=(
                int(self.core.last_processed_generation)
                if getattr(self.core, "last_processed_generation", None) is not None else None
            ),
            state_observation="after_core_call",
            controller=str(state_after_core.controller),
            controller_ready=bool(state_after_core.controller_ready),
            controller_calibrated=bool(state_after_core.controller_calibrated),
            fault=state_after_core.fault,
            controller_reason=str(diagnostics_after_core.get("reason", "")),
            pulse_events=pulse_events,
        )
        # The history append and active marker clear are after end_ns. They
        # never change the established Core timing boundary.
        with self._recent_core_lock:
            self._recent_core_rows.append(row)
            self._active_core_source_index = None
            self._active_core_source_lag_ms = None
            self._active_core_source_lag_index = None
        return row

    def process(self, source_index: int, values_mv: np.ndarray, roles: tuple[str, str],
                labels: tuple[str, str], ctx_index: int, *, arm_after_drain: bool = True,
                stim_enabled: bool = True) -> app.GeneratedAoBlock:
        source_availability_lag_ms: float | None = None
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
            source_availability_lag_ms = lag_s * 1000.0
        if self.last_source_index is not None and source_index <= self.last_source_index:
            raise ValueError("replay source sample indices must increase monotonically")
        if self.last_source_index is not None:
            gap = source_index - self.last_source_index - self.config.chunk_size
            if gap > 0:
                self.metrics.source_index_gap_samples += gap
        with self._recent_core_lock:
            self.last_source_index = source_index
            self._latest_source_sample_index_seen = source_index
        if values_mv.shape != (2, self.config.chunk_size):
            raise ValueError(f"processing input shape must be (2, {self.config.chunk_size})")
        self._wait_for_unpaced_recorder_capacity(source_index)
        # This start marker is published before the measured Core interval.
        # If it remains set in a fault snapshot, completion history has not yet
        # been published; it does not prove Core is still executing.
        self._mark_core_call_started(source_index, source_availability_lag_ms)
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
            self._remember_completed_core_call(
                source_index, start, end, thread_cpu_elapsed, process_cpu_elapsed,
                source_availability_lag_ms,
            )
            if elapsed_ms >= self.metrics.deadline_ms:
                self._slow_process_calls += 1
                if self._slow_process_calls <= 20:
                    related_gc = [
                        {"generation": generation, "pause_ms": pause / 1e6}
                        for gc_end, pause, generation in self._gc_events
                        if start <= gc_end <= end
                    ]
                    self._write_progress({
                        "type": "slow_core_process", "source_sample_index": source_index,
                        "wall_ms": elapsed_ms, "thread_cpu_ms": thread_cpu_elapsed / 1e6,
                        "process_cpu_ms": process_cpu_elapsed / 1e6,
                        "process_cpu_minus_thread_ms": max(0.0, process_cpu_elapsed - thread_cpu_elapsed) / 1e6,
                        "gc_pauses": related_gc,
                    })
        self._record_core_latency(source_index, elapsed_ms)
        self.metrics.processed_blocks += 1
        self.metrics.source_samples += int(np.count_nonzero(np.isfinite(values_mv[0])))
        self.metrics.recorder_backpressure_max_backlog = max(
            self.metrics.recorder_backpressure_max_backlog, self.recorder.backlog)
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
        _record_pulse_event_metrics(self.metrics, self._seen_pulse_events, self.core.last_pulse_events)

        snapshot = self.state.snapshot()
        base = self.base_sample_index
        target_block = ((source_index - base) // self.config.chunk_size) + self.config.ao_lead_chunks
        if snapshot.mode != "closed-loop" or snapshot.fault:
            safe = np.zeros_like(safe)

        pulse_ids = [event.pulse_id for event in self.core.last_pulse_events
                     if not str(event.status).startswith("rejected")]
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
                self._write_progress({
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
                self._write_progress({
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
            recent_window = tuple(self._recent_core_latencies)
            self._write_progress({
                "type": "run_progress", "case_id": self.metrics.case_id,
                "controller": self.config.controller, "source_samples": self.metrics.source_samples,
                "skipped_source_samples": self.metrics.skipped_source_samples,
                "source_index_gap_samples": self.metrics.source_index_gap_samples,
                "source_availability_lag_ms": self._last_source_availability_lag_ms,
                "processed_blocks": self.metrics.processed_blocks,
                "elapsed_source_seconds": self.metrics.source_samples / self.config.sample_rate,
                "ao": self.scheduler.stats_snapshot(),
                "recorder_backlog": self.recorder.backlog,
                "recorder_backpressure_wait_count": self.metrics.recorder_backpressure_wait_count,
                "recorder_backpressure_wait_ms": self.metrics.recorder_backpressure_wait_ms,
                "recorder_backpressure_policy": self.metrics.recorder_backpressure_policy,
                "recording_path": str(self.recording_path),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }, recent_core_window=recent_window)
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

    def _wait_for_unpaced_recorder_capacity(self, source_index: int) -> None:
        """Pause offline input between blocks while the bounded HDF5 queue drains.

        Paced runs never wait here. This wait is outside `core.process` and does
        not advance the virtual AO clock or change the source sample index.
        """
        if self.paced:
            return
        threshold = self.metrics.recorder_backpressure_threshold_blocks
        backlog = self.recorder.backlog
        self.metrics.recorder_backpressure_max_backlog = max(
            self.metrics.recorder_backpressure_max_backlog, backlog)
        if backlog < threshold:
            return
        wait_started_ns = time.perf_counter_ns()
        no_progress_timeout_ns = int(self.UNPACED_RECORDER_NO_PROGRESS_TIMEOUT_S * 1e9)
        last_progress_ns = time.monotonic_ns()
        last_backlog = backlog
        self.metrics.recorder_backpressure_wait_count += 1
        if self.metrics.recorder_backpressure_wait_count == 1:
            self._write_progress({
                "type": "recorder_backpressure_started",
                "source_sample_index": source_index,
                "queue_depth": backlog,
                "queue_capacity_blocks": self.config.logger_queue_blocks,
                "threshold_blocks": threshold,
                "policy": self.metrics.recorder_backpressure_policy,
            })
        while backlog >= threshold:
            if self.recorder.error is not None:
                raise RuntimeError(f"HDF5 writer failed while unpaced producer waited: {self.recorder.error}")
            time.sleep(0.001)
            backlog = self.recorder.backlog
            self.metrics.recorder_backpressure_max_backlog = max(
                self.metrics.recorder_backpressure_max_backlog, backlog)
            now_ns = time.monotonic_ns()
            if backlog < last_backlog:
                last_backlog = backlog
                last_progress_ns = now_ns
            elif now_ns - last_progress_ns >= no_progress_timeout_ns:
                elapsed_ms = (time.perf_counter_ns() - wait_started_ns) / 1e6
                self.metrics.recorder_backpressure_wait_ms += elapsed_ms
                details = {
                    "type": "recorder_backpressure_timeout",
                    "source_sample_index": source_index,
                    "queue_depth": backlog,
                    "queue_capacity_blocks": self.config.logger_queue_blocks,
                    "threshold_blocks": threshold,
                    "no_progress_timeout_s": self.UNPACED_RECORDER_NO_PROGRESS_TIMEOUT_S,
                    "wait_duration_ms": elapsed_ms,
                    "recorder_error": self.recorder.error,
                }
                self._write_progress(details)
                raise TimeoutError(
                    "HDF5 recorder queue made no progress while unpaced producer waited: "
                    f"source_sample_index={source_index}, queue_depth={backlog}, "
                    f"queue_capacity_blocks={self.config.logger_queue_blocks}, threshold_blocks={threshold}, "
                    f"no_progress_timeout_s={self.UNPACED_RECORDER_NO_PROGRESS_TIMEOUT_S:g}, "
                    f"wait_duration_ms={elapsed_ms:.3f}, recorder_error={self.recorder.error!r}"
                )
        self.metrics.recorder_backpressure_wait_ms += (
            time.perf_counter_ns() - wait_started_ns) / 1e6

    def _collect_gc_timing(self, phase: str, info: dict[str, Any]) -> None:
        """Capture GC pause timing with an in-memory callback only."""
        generation = int(info.get("generation", -1))
        key = (threading.get_ident(), generation)
        now = time.perf_counter_ns()
        with self._gc_lock:
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

    def _snapshot_recent_core_context(self) -> dict[str, Any]:
        """Copy bounded immutable history and in-flight source markers."""
        with self._recent_core_lock:
            rows = tuple(self._recent_core_rows)
            active_source = self._active_core_source_index
            last_started_source = self._last_core_call_started_source_index
            latest_source_seen = self._latest_source_sample_index_seen
            active_lag_ms = self._active_core_source_lag_ms
            active_lag_index = self._active_core_source_lag_index
        latest = rows[-1] if rows else None
        lag_ms = active_lag_ms if active_lag_index is not None else (
            latest.source_availability_lag_ms if latest is not None else None)
        lag_index = active_lag_index if active_lag_index is not None else (
            latest.source_availability_lag_sample_index if latest is not None else None)
        return {
            "latest_source_sample_index_seen": latest_source_seen,
            "latest_core_call_started_source_sample_index": last_started_source,
            "core_call_started_not_yet_published_source_sample_index": active_source,
            "latest_completed_core_source_sample_index": (
                latest.source_sample_index if latest is not None else None),
            "source_availability_lag_ms": lag_ms,
            "source_availability_lag_sample_index": lag_index,
            "recent_core_history_capacity": AO_FAULT_CORE_HISTORY_CAPACITY,
            "recent_core_history": [asdict(row) for row in rows],
            "latest_completed_pulse_events": (
                [asdict(event) for event in latest.pulse_events]
                if latest is not None else []),
        }

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
                state_snapshot = self.state.snapshot()
                state_fault = state_snapshot.fault
                if state_fault and state_fault != self._observed_ao_fault:
                    self._observed_ao_fault = state_fault
                    stats = self.scheduler.stats_snapshot()
                    observed_ns = time.perf_counter_ns()
                    recent_context = self._snapshot_recent_core_context()
                    self._write_progress({
                        "type": "ao_clock_fault", "fault": state_fault,
                        "source_sample_index": recent_context["latest_source_sample_index_seen"],
                        "latest_completed_pulse_events": recent_context.pop(
                            "latest_completed_pulse_events"),
                        "generated_sample_count": stats.get("generated_sample_count"),
                        "next_write_block": getattr(self.scheduler, "_next_write_block", None),
                        "pacing_epoch_monotonic": self.pacing_epoch,
                        "observed_monotonic": observed_ns / 1e9,
                        "observation_time_monotonic_ns": observed_ns,
                        "observation_is_fault_onset": False,
                        "history_overhead_note": (
                            "Recent-row capture runs after each measured Core call; it is outside the recorded Core interval "
                            "but adds harness work that may affect later scheduling."
                        ),
                        "snapshot_semantics_note": (
                            "Service gap is the AO host's observed interval, not a proven fault onset or cause. "
                            "AO, runtime state, and recent Core rows are sampled sequentially, not atomically. "
                            "A started-but-unpublished Core index is not proof that Core is still executing."
                        ),
                        "ao_service_gap_ms": elapsed * 1000.0,
                        "ao_service_gap_over_block_period": (
                            elapsed > self.config.chunk_size / self.config.sample_rate),
                        "ao_max_service_gap_ms_at_observation": self._ao_clock_max_gap_s * 1000.0,
                        "ao_service_gaps_over_block_period_at_observation": self._ao_clock_gap_count,
                        "state_observed_after_ao_advance": {
                            "mode": state_snapshot.mode,
                            "acquiring": state_snapshot.acquiring,
                            "armed": state_snapshot.armed,
                            "transition_pending": state_snapshot.transition_pending,
                            "generation": state_snapshot.generation,
                            "controller": state_snapshot.controller,
                            "controller_ready": state_snapshot.controller_ready,
                            "controller_calibrated": state_snapshot.controller_calibrated,
                            "fault": state_snapshot.fault,
                        },
                        "ao": stats,
                        **recent_context,
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
        self._flush_final_core_notifications()
        actuator_phase = _capture_actuator_rate_phase(self.metrics, self.core.actuator)
        final_state = self.state.snapshot()
        self.metrics.ao_drain_complete = not final_state.transition_pending
        self.metrics.runtime_fault = final_state.fault
        if not self.metrics.ao_drain_complete:
            self.metrics.notes.append("AO drain barrier did not complete before run finalization")
        self.metrics.pulse_commit_ids = sorted(self._pulse_commit_ids)
        self.metrics.pulse_commit_samples = self.pulse_commit_samples
        self.metrics.ui_packets_submitted = self.hub.publish_count
        self.metrics.ui_packets_consumed = self.ui_consumed_count
        self.metrics.recorder_max_backlog = self.telemetry.snapshot()["logger_high_water"]
        self.state.set_recording(False)
        self.recorder.stop_recording()
        self.recorder.close()
        browser_presence = self._browser_presence.snapshot(time.perf_counter_ns())
        self.metrics.browser_connected = bool(browser_presence["ever_connected"])
        self.metrics.browser_disconnect_count = int(browser_presence["disconnect_count"])
        self.metrics.browser_disconnected_duration_ms = float(browser_presence["disconnected_duration_ms"])
        self.metrics.browser_connected_at_finish = bool(browser_presence["connected_at_finish"])
        self.metrics.browser_disconnect_open_at_finish = bool(browser_presence["disconnect_open_at_finish"])
        if self.metrics.browser_connected:
            self.metrics.ui_transport_kind = "browser_and_websocket_client"
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
        self._write_progress({
            "type": "run_complete", "case_id": self.metrics.case_id,
            "processed_blocks": self.metrics.processed_blocks, "ao": scheduler_stats,
            "actuator_rate_phase": actuator_phase,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        self.core.close_ui()
        self._run_on_loop(self._stop_ui_transport())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=3)
        self.loop.close()
        self._remove_gc_callback()
        self._close_progress_writer()
        return self.metrics

    def _remove_gc_callback(self) -> None:
        if self._gc_callback_registered:
            try:
                gc.callbacks.remove(self._gc_callback)
            except ValueError:
                pass
            self._gc_callback_registered = False

    def abort(self) -> list[str]:
        """Attempt every bounded cleanup step and return any cleanup failures."""
        errors: list[str] = []

        def attempt(label: str, operation: Callable[[], Any]) -> Any:
            try:
                return operation()
            except BaseException as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
                return None

        attempt("disarm runtime before abort", lambda: self.state.set_armed(False))
        try:
            self._stop_ao_clock()
        except TimeoutError as exc:
            status = self._ao_owner_status(str(exc))
            pending_error = self._pending_ao_shutdown_error
            if pending_error is None:
                pending_error = HarnessOwnerShutdownError(
                    "simulated_ao", status, progress_path=self.progress_path)
                self._pending_ao_shutdown_error = pending_error
            else:
                pending_error.refresh(status)
            if status["owner_alive"]:
                if errors:
                    pending_error.add_note("Disarm errors: " + " | ".join(errors))
                raise pending_error
            self._ao_clock_thread = None
        except BaseException as exc:
            errors.append(f"stop simulated AO clock: {type(exc).__name__}: {exc}")

        if getattr(self, "_ao_clock_thread", None) is not None:
            if not self._ao_clock_thread.is_alive():
                self._ao_clock_thread = None
            else:
                status = self._ao_owner_status("simulated AO clock remains alive")
                pending_error = self._pending_ao_shutdown_error
                if pending_error is None:
                    pending_error = HarnessOwnerShutdownError(
                        "simulated_ao", status, progress_path=self.progress_path)
                    self._pending_ao_shutdown_error = pending_error
                else:
                    pending_error.refresh(status)
                raise pending_error
        core = getattr(self, "core", None)
        if core is not None:
            attempt("drain final Core AO notifications", self._flush_final_core_notifications)
        attempt("disable recording", lambda: self.state.set_recording(False))
        attempt("stop recorder admission", self.recorder.stop_recording)
        attempt("close recorder", self.recorder.close)
        attempt("close core UI", self.core.close_ui)
        attempt("stop UI transport", lambda: self._run_on_loop(self._stop_ui_transport()))
        attempt("request UI loop stop", lambda: self.loop.call_soon_threadsafe(self.loop.stop))
        attempt("join UI loop thread", lambda: self.loop_thread.join(timeout=3))
        if self.loop_thread.is_alive():
            errors.append("join UI loop thread: thread remained alive after 3 seconds")
        if not self.loop.is_closed() and not self.loop.is_running():
            attempt("close UI event loop", self.loop.close)
        elif self.loop.is_running():
            errors.append("close UI event loop: loop is still running")
        attempt("remove GC callback", self._remove_gc_callback)

        if errors:
            attempt("write cleanup diagnostics", lambda: self._write_progress({
                "type": "cleanup_errors",
                "case_id": getattr(getattr(self, "metrics", None), "case_id", None),
                "errors": errors, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }))
        try:
            progress_result = self._close_progress_writer()
        except HarnessOwnerShutdownError as exc:
            if errors:
                exc.add_note("Additional cleanup failures: " + " | ".join(errors))
            pending_ao_error = self._pending_ao_shutdown_error
            if pending_ao_error is not None and exc is not pending_ao_error:
                exc.add_note(f"Earlier simulated AO shutdown failure: {pending_ao_error}")
            raise
        if progress_result and progress_result.get("error"):
            errors.append(f"progress diagnostics: {progress_result['error']}")
        pending_ao_error = self._pending_ao_shutdown_error
        if pending_ao_error is not None:
            pending_ao_error.refresh(self._ao_owner_status(
                pending_ao_error.original_error or "simulated AO stop previously timed out"))
            raise pending_ao_error
        return errors

    def _stop_ao_clock(self) -> None:
        self._ao_clock_stop.set()
        if self._ao_clock_thread is not None:
            self._ao_clock_thread.join(timeout=self._ao_stop_timeout_s)
            if self._ao_clock_thread.is_alive():
                self._ao_clock_stop_timed_out = True
                raise TimeoutError("simulated AO clock did not stop")
            self._ao_clock_thread = None

    def _ao_owner_status(self, error: str | None = None) -> dict[str, Any]:
        thread = getattr(self, "_ao_clock_thread", None)
        alive = bool(thread is not None and thread.is_alive())
        return {
            "owner_alive": alive,
            "owner_thread_name": thread.name if thread is not None else None,
            "owner_thread_ident": thread.ident if thread is not None else None,
            "stop_requested": bool(getattr(self, "_ao_clock_stop", threading.Event()).is_set()),
            "stop_timed_out": bool(getattr(self, "_ao_clock_stop_timed_out", False)),
            "error": error,
        }

    def _flush_final_core_notifications(self) -> None:
        """Persist final AO owner outcomes without advancing finite source time."""
        self.core.drain_ao_notifications()

    def _create_scheduler(self, config: app.AppConfig) -> app.SimulatedAoScheduler:
        scheduler = app.SimulatedAoScheduler(
            config, self.state, self.recorder,
            commit_event=self._commit_actuator_event,
            ao_outcome=self.core.record_ao_outcome,
            pulse_outcome=self._receive_pulse_outcome,
        )
        self.core.set_local_account_sink(scheduler.enqueue_local_account)
        return scheduler

    def _receive_pulse_outcome(self, outcome: app.PulseAoOutcome) -> None:
        """Queue authoritative ranges to Core and count only new confirmed AO samples."""
        self.core.enqueue_ao_outcome(outcome)
        if outcome.status != "committed":
            return
        with self._pulse_commit_lock:
            newly_committed = self._pulse_commit_ranges.add(
                outcome.pulse_id, outcome.start_sample_index, outcome.end_sample_index,
            )
            if newly_committed:
                self.pulse_commit_samples += newly_committed
                self._pulse_commit_ids.add(outcome.pulse_id)

    def _commit_actuator_event(self, pulse_id: str, samples: int) -> None:
        """Legacy observer hook; typed absolute AO outcomes own commitment accounting."""
        return None

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


class _BrowserPresenceTracker:
    """Track real external UI WebSocket presence without claiming rendering."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ever_connected = False
        self._connected = False
        self._disconnect_count = 0
        self._disconnected_duration_ns = 0
        self._disconnected_since_ns: int | None = None

    def observe(self, connected: bool, timestamp_ns: int | None = None) -> None:
        now_ns = time.perf_counter_ns() if timestamp_ns is None else int(timestamp_ns)
        with self._lock:
            if connected:
                if not self._connected:
                    if self._disconnected_since_ns is not None:
                        self._disconnected_duration_ns += max(0, now_ns - self._disconnected_since_ns)
                        self._disconnected_since_ns = None
                    self._connected = True
                    self._ever_connected = True
            elif self._connected:
                self._connected = False
                self._disconnect_count += 1
                self._disconnected_since_ns = now_ns

    def observe_external_count(self, count: int, timestamp_ns: int | None = None) -> None:
        self.observe(int(count) > 0, timestamp_ns)

    def snapshot(self, timestamp_ns: int | None = None) -> dict[str, Any]:
        now_ns = time.perf_counter_ns() if timestamp_ns is None else int(timestamp_ns)
        with self._lock:
            open_disconnect = self._ever_connected and not self._connected
            duration_ns = self._disconnected_duration_ns
            if open_disconnect and self._disconnected_since_ns is not None:
                duration_ns += max(0, now_ns - self._disconnected_since_ns)
            return {
                "ever_connected": self._ever_connected,
                "disconnect_count": self._disconnect_count,
                "disconnected_duration_ms": duration_ns / 1e6,
                "connected_at_finish": self._connected,
                "disconnect_open_at_finish": open_disconnect,
            }


class _CountingUiHub(app.UiHub):
    def __init__(self, queue_size: int, browser_presence: _BrowserPresenceTracker):
        super().__init__(queue_size)
        self.publish_count = 0
        self.last_packet: str | None = None
        self.browser_presence = browser_presence
        self._external_browser_present = False

    def _update_external_presence(self) -> None:
        present = self.client_count >= 2
        if present != self._external_browser_present:
            self._external_browser_present = present
            self.browser_presence.observe(present, time.perf_counter_ns())

    def subscribe(self) -> asyncio.Queue[str]:
        client = super().subscribe()
        self._update_external_presence()
        return client

    def unsubscribe(self, client: asyncio.Queue[str]) -> None:
        super().unsubscribe(client)
        self._update_external_presence()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def publish(self, packet: str) -> None:
        self.publish_count += 1
        self.last_packet = packet
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
        temporary_recording_dir = Path(tempfile.mkdtemp(prefix="echo-chamber-benchmark-"))
        record_dir = temporary_recording_dir
    else:
        temporary_recording_dir = None
        record_dir = recording_dir
        record_dir.mkdir(parents=True, exist_ok=True)
    try:
        config = replace(
            _make_config(controller, lead_ms, record_dir),
            electrode_labels=replay.channel_labels,
            ctx_index=replay.ctx_index,
        )
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
    except BaseException as exc:
        _release_temporary_recording_directory(temporary_recording_dir, exc)
        raise
    limit_samples = int(round(limit_seconds * replay.sample_rate_hz)) if limit_seconds is not None else None
    try:
        for source_index, values in _replay_blocks(replay, limit_samples):
            harness.process(source_index, values, replay.channel_roles, replay.channel_labels, replay.ctx_index)
        metrics = harness.finish()
        if on_metrics:
            on_metrics(metrics)
    except BaseException as primary_error:
        try:
            cleanup_errors = harness.abort()
        except HarnessOwnerShutdownError as cleanup_error:
            if cleanup_error is primary_error:
                raise
            cleanup_error.add_note(
                f"Run also failed: {type(primary_error).__name__}: {primary_error}")
            raise cleanup_error from primary_error
        if isinstance(primary_error, HarnessOwnerShutdownError):
            if primary_error.owner_kind == "progress_writer":
                status = getattr(harness, "_progress_writer_result", None)
            else:
                status = harness._ao_owner_status(primary_error.original_error)
            if status is not None:
                primary_error.refresh(status)
        if cleanup_errors:
            primary_error.add_note("Additional cleanup failures: " + " | ".join(cleanup_errors))
        raise
    finally:
        _release_temporary_recording_directory(temporary_recording_dir, sys.exc_info()[1])
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
                         "browser_disconnect_count": metrics.browser_disconnect_count,
                         "browser_disconnected_duration_ms": metrics.browser_disconnected_duration_ms,
                         "browser_connected_at_finish": metrics.browser_connected_at_finish,
                         "packets_submitted": metrics.ui_packets_submitted,
                         "packets_consumed": metrics.ui_packets_consumed,
                         "browser_rendering_verified": False},
        "ui_channel_mapping": _ui_channel_mapping(harness.hub.last_packet),
        "physical_pulse_claims": False,
    }
    return metrics, environment


def _ui_channel_mapping(packet: str | None) -> dict[str, Any] | None:
    """Extract channel fields from the latest actual UI WebSocket frame."""
    if packet is None:
        return None
    message = parse_ui_packet(packet)
    return {key: message.get(key) for key in ("channels", "cortex_ai")}


def write_results(metrics: list[RunMetrics], environments: list[dict[str, Any]], output_dir: Path,
                  *, run_kind: str, acceptance: list[dict[str, Any]] | None = None,
                  case_failures: list[dict[str, Any]] | None = None,
                  collection_acceptance: dict[str, Any] | None = None) -> tuple[Path, Path]:
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
        "collection_acceptance": collection_acceptance,
        "cases": [asdict(x) for x in metrics], "environments": environments,
        "case_failures": case_failures or [],
    }
    json_path.write_text(json.dumps(json_safe(document), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        metric_fields = [item.name for item in fields(RunMetrics)]
        csv_fields = metric_fields + ["case_status", "failure_type", "failure_message",
                                      "acceptance_scope", "runtime_checks_passed", "runtime_failures",
                                      "full_acceptance_status", "full_acceptance_passed",
                                      "required_duration_seconds", "required_blocks", "required_samples",
                                      "required_sample_rate_hz", "required_block_samples",
                                      "required_block_ms", "required_deadline_ms",
                                      "observed_duration_seconds", "observed_blocks", "observed_samples",
                                      "observed_sample_rate_hz", "observed_block_samples",
                                      "observed_block_ms", "observed_deadline_ms", "coverage_complete",
                                      "collection_status", "collection_full_acceptance_passed",
                                      "collection_required_cases", "collection_observed_cases",
                                      "collection_required_controllers", "collection_observed_controllers",
                                      "collection_required_traces", "collection_observed_traces",
                                      "collection_required_pairs", "collection_observed_pairs",
                                      "collection_failures",
                                      "acceptance_passed", "acceptance_failures"]
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        acceptance_by_id = {item["case_id"]: item for item in (acceptance or [])}
        for item in metrics:
            row = asdict(item)
            row["case_status"] = "completed"
            evaluated = acceptance_by_id.get(item.case_id)
            _write_acceptance_csv_fields(row, evaluated)
            _write_collection_csv_fields(row, collection_acceptance)
            writer.writerow(row)
        for failure in case_failures or []:
            row = {name: "" for name in metric_fields}
            for name in ("case_id", "source_name", "controller", "ao_lead_ms"):
                if name in failure:
                    row[name] = failure[name]
            row.update({
                "case_status": "failed",
                "failure_type": failure.get("exception_type", "Exception"),
                "failure_message": failure.get("message", ""),
            })
            _write_acceptance_csv_fields(row, acceptance_by_id.get(str(failure.get("case_id", ""))))
            _write_collection_csv_fields(row, collection_acceptance)
            writer.writerow(row)
    return json_path, csv_path


def write_case_checkpoint(output_dir: Path, *, run_kind: str, case_index: int,
                          metrics: RunMetrics | None = None,
                          environment: dict[str, Any] | None = None,
                          failure: dict[str, Any] | None = None,
                          requested_scope: str | None = None,
                          collection_progress: dict[str, Any] | None = None) -> tuple[Path, Path]:
    """Persist one case immediately; the checkpoint never marks a matrix complete."""
    if (metrics is None) == (failure is None):
        raise ValueError("provide exactly one of metrics or failure")
    output_dir.mkdir(parents=True, exist_ok=True)
    case_status = "completed" if metrics is not None else "failed"
    case_id = metrics.case_id if metrics is not None else str(failure.get("case_id", "unknown"))
    safe_case_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in case_id)
    stem = f"controller_case_{case_index:03d}_{safe_case_id}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    acceptance = (evaluate_case_acceptance(metrics, run_kind, requested_scope) if metrics is not None else
                  failed_case_acceptance(case_id, failure or {},
                                         requested_scope or "diagnostic_prefix"))
    document = {
        "schema": "echo-chamber-controller-case-checkpoint-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(), "python": platform.python_version(),
        "machine": platform.platform(), "processor": platform.processor(),
        "run_kind": run_kind, "case_index": case_index,
        "checkpoint_scope": "single_case", "case_status": case_status,
        "matrix_status": "in_progress", "matrix_complete": False,
        "case": asdict(metrics) if metrics is not None else None,
        "environment": environment,
        "failure": failure,
        "acceptance": acceptance,
        "collection_progress": collection_progress,
        "physical_pulse_claims": False,
    }
    json_path.write_text(json.dumps(json_safe(document), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    metric_fields = [item.name for item in fields(RunMetrics)]
    row = asdict(metrics) if metrics is not None else {name: "" for name in metric_fields}
    if failure is not None:
        for name in ("case_id", "source_name", "controller", "ao_lead_ms"):
            if name in failure:
                row[name] = failure[name]
    row.update({
        "case_status": case_status,
        "failure_type": failure.get("exception_type", "Exception") if failure else "",
        "failure_message": failure.get("message", "") if failure else "",
        "matrix_complete": False,
    })
    _write_acceptance_csv_fields(row, acceptance)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields + [
            "case_status", "failure_type", "failure_message", "acceptance_passed",
            "acceptance_scope", "runtime_checks_passed", "runtime_failures",
            "full_acceptance_status", "full_acceptance_passed",
            "required_duration_seconds", "required_blocks", "required_samples",
            "required_sample_rate_hz", "required_block_samples",
            "required_block_ms", "required_deadline_ms",
            "observed_duration_seconds", "observed_blocks", "observed_samples",
            "observed_sample_rate_hz", "observed_block_samples",
            "observed_block_ms", "observed_deadline_ms", "coverage_complete",
            "collection_status", "collection_full_acceptance_passed",
            "collection_required_cases", "collection_observed_cases",
            "collection_required_controllers", "collection_observed_controllers",
            "collection_required_traces", "collection_observed_traces",
            "collection_required_pairs", "collection_observed_pairs",
            "collection_failures",
            "acceptance_failures", "matrix_complete",
        ])
        writer.writeheader()
        writer.writerow(row)
    return json_path, csv_path


def _write_acceptance_csv_fields(row: dict[str, Any], acceptance: dict[str, Any] | None) -> None:
    if acceptance is None:
        return
    coverage = acceptance.get("coverage") or {}
    required = coverage.get("required") or {}
    observed = coverage.get("observed") or {}
    row.update({
        "acceptance_scope": acceptance.get("scope", ""),
        "runtime_checks_passed": acceptance.get("runtime_checks_passed"),
        "runtime_failures": json.dumps(acceptance.get("runtime_failures", [])),
        "full_acceptance_status": acceptance.get("full_acceptance_status", ""),
        "full_acceptance_passed": acceptance.get("full_acceptance_passed"),
        "required_duration_seconds": required.get("duration_seconds", ""),
        "required_blocks": required.get("processed_blocks", ""),
        "required_samples": required.get("source_samples", ""),
        "required_sample_rate_hz": required.get("sample_rate_hz", ""),
        "required_block_samples": required.get("block_samples", ""),
        "required_block_ms": required.get("block_ms", ""),
        "required_deadline_ms": required.get("deadline_ms", ""),
        "observed_duration_seconds": observed.get("duration_seconds", ""),
        "observed_blocks": observed.get("processed_blocks", ""),
        "observed_samples": observed.get("source_samples", ""),
        "observed_sample_rate_hz": observed.get("sample_rate_hz", ""),
        "observed_block_samples": observed.get("block_samples", ""),
        "observed_block_ms": observed.get("block_ms", ""),
        "observed_deadline_ms": observed.get("deadline_ms", ""),
        "coverage_complete": coverage.get("complete"),
        "acceptance_passed": acceptance.get("full_acceptance_passed"),
        "acceptance_failures": json.dumps(acceptance.get("failures", [])),
    })


def _write_collection_csv_fields(row: dict[str, Any], collection: dict[str, Any] | None) -> None:
    if collection is None:
        return
    required = collection.get("required") or {}
    observed = collection.get("observed") or {}
    row.update({
        "collection_status": collection.get("status", ""),
        "collection_full_acceptance_passed": collection.get("full_acceptance_passed"),
        "collection_required_cases": required.get("case_count", ""),
        "collection_observed_cases": observed.get("completed_case_count", ""),
        "collection_required_controllers": required.get("controller_count", ""),
        "collection_observed_controllers": observed.get("controller_count", ""),
        "collection_required_traces": required.get("trace_count", ""),
        "collection_observed_traces": observed.get("source_count", ""),
        "collection_required_pairs": required.get("unique_source_controller_pairs", ""),
        "collection_observed_pairs": observed.get("unique_source_controller_pairs", ""),
        "collection_failures": json.dumps(collection.get("failures", [])),
    })
