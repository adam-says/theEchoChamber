"""Diagnostic-only first-fault ESN runtime probe.

This adds per-call timers around existing phases. Its timings are diagnostic,
not acceptance evidence. The Core timing gate, configuration, sample path,
recorder, UI transport, actuator, and AO scheduler remain unchanged. No
external browser rendering is claimed.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmarkControllers as bench  # noqa: E402
import echoChamber as app  # noqa: E402

PHASE_CAPACITY = 50_000


class PhaseProbe:
    """Bounded in-memory phase samples; no hot-path serialization or I/O."""

    def __init__(self) -> None:
        self.rows: deque[tuple[int | None, str, int, int, int]] = deque(maxlen=PHASE_CAPACITY)
        self.lock = threading.Lock()
        self.local = threading.local()
        self.active: dict[int, list[tuple[str, int | None, int, int]]] = {}
        self.core_thread_id: int | None = None
        self.ao_thread_id: int | None = None
        self.previous_core_end_ns: int | None = None
        self.previous_core_thread_cpu_end_ns: int | None = None
        self.active_miss: dict[str, Any] | None = None
        self.schedule_to_publish_ns: deque[int] = deque(maxlen=1000)
        self.publish_wall_ns: deque[int] = deque(maxlen=1000)
        self.publish_thread_ids: deque[int] = deque(maxlen=1000)
        self.ui_queue_lock_rows: deque[tuple[int | None, str, int, int]] = deque(maxlen=5000)

    def current_source(self) -> int | None:
        return getattr(self.local, "source_index", None)

    def append(self, source_index: int | None, name: str, start: int, end: int,
               cpu_start: int, cpu_end: int, thread_id: int) -> None:
        row = (source_index, name, max(0, end - start), max(0, cpu_end - cpu_start), thread_id)
        with self.lock:
            self.rows.append(row)

    def timed(self, owner: Any, method_name: str, label: str,
              *, source_arg: int | None = None) -> None:
        original = getattr(owner, method_name)

        def wrapped(*args, **kwargs):
            ident = threading.get_ident()
            source = self.current_source()
            if source is None and source_arg is not None:
                try:
                    source = int(args[source_arg])
                except (IndexError, TypeError, ValueError):
                    source = None
            start = time.perf_counter_ns()
            cpu_start = time.thread_time_ns()
            with self.lock:
                self.active.setdefault(ident, []).append((label, source, start, ident))
            try:
                return original(*args, **kwargs)
            finally:
                end = time.perf_counter_ns()
                cpu_end = time.thread_time_ns()
                self.append(source, label, start, end, cpu_start, cpu_end, ident)
                with self.lock:
                    stack = self.active.get(ident)
                    if stack:
                        stack.pop()
                        if not stack:
                            self.active.pop(ident, None)

        setattr(owner, method_name, wrapped)

    def snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = tuple(self.rows)
        return [
            {"source_sample_index": source, "phase": name, "wall_ns": wall,
             "thread_cpu_ns": cpu, "thread_id": ident}
            for source, name, wall, cpu, ident in rows
        ]

    def active_snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            values = tuple(item for stack in self.active.values() for item in stack)
        now = time.perf_counter_ns()
        return [{"phase": name, "source_sample_index": source,
                 "started_perf_counter_ns": start, "age_at_fault_observation_ns": max(0, now-start),
                 "thread_id": ident}
                for name, source, start, ident in values]


class TimedPrimitiveLock:
    """Delegate a Queue's existing lock and separate contention from hold time."""

    def __init__(self, lock, probe: PhaseProbe) -> None:
        self.lock = lock
        self.probe = probe
        self.local = threading.local()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        start = time.perf_counter_ns()
        if timeout == -1:
            acquired = self.lock.acquire(blocking)
        else:
            acquired = self.lock.acquire(blocking, timeout)
        end = time.perf_counter_ns()
        if acquired:
            self.local.held = (start, end, self.probe.current_source(), threading.get_ident())
        return acquired

    def release(self) -> None:
        end = time.perf_counter_ns()
        held = getattr(self.local, "held", None)
        self.lock.release()
        if held is not None:
            start, acquired, source, ident = held
            with self.probe.lock:
                self.probe.ui_queue_lock_rows.append((source, "wait", acquired - start, ident))
                self.probe.ui_queue_lock_rows.append((source, "hold", end - acquired, ident))
            self.local.held = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_args):
        self.release()

    def is_owned(self) -> bool:
        if self.lock.acquire(False):
            self.lock.release()
            return False
        return True

    def locked(self) -> bool:
        return self.lock.locked()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_hashes(trace: Path, model_artifact: Path) -> dict[str, str]:
    selected = [ROOT / name for name in (
        "benchmarkControllers.py", "echoChamber.py", "esn_bridge.py",
        "controllers/esn.py", "esn/streaming.py", "esn/inference.py",
        "esn/preprocessing.py", "esn/model.py", "esn/artifact.py",
    )]
    selected.extend((trace, model_artifact, model_artifact.with_suffix(".json"),
                     Path(__file__).resolve()))
    return {str(path.resolve()): sha256(path) for path in selected if path.is_file()}


def _service_capture(probe: PhaseProbe, harness: bench._Harness) -> None:
    scheduler = harness.scheduler
    original = scheduler._service_refill_locked
    chunk = scheduler.config.chunk_size

    def wrapped() -> None:
        generated = scheduler.generated_sample_count
        next_write = scheduler._next_write_block
        queued = next_write * chunk - generated
        deficit = max(0, math.ceil((scheduler.config.ao_lead_samples - queued) / chunk))
        snapshot = scheduler.state.snapshot()
        ready = 0
        for target in range(next_write, next_write + deficit):
            pending = scheduler.commands.get(target)
            if pending is None or not scheduler.lifecycle.eligible(
                    pending, snapshot.generation, snapshot.transition_pending):
                break
            ready += 1
        missing = deficit - ready
        first_missing = next_write + ready if missing else None
        headroom = first_missing * chunk - generated if first_missing is not None else None
        urgent = bool(missing and scheduler.lifecycle.missing_refill_is_urgent(
            next_write, ready, generated, chunk, scheduler.config.ao_deadline_samples))
        producer_running = (snapshot.mode == "closed-loop" and snapshot.acquiring
                            and not snapshot.transition_pending and snapshot.fault is None)
        active_output = producer_running and snapshot.armed and snapshot.stim_mode != "off"
        before_miss = scheduler.stats.get("active_demand_misses", 0)
        near = bool(missing and queued <= scheduler.config.ao_deadline_samples + chunk)
        pre = None
        if near:
            context = harness._snapshot_recent_core_context()
            source_base = harness.base_sample_index
            expected_source = None
            if first_missing is not None and source_base is not None:
                expected_source = (int(source_base)
                                   + (first_missing - scheduler.config.ao_lead_chunks) * chunk)
            pre = {
                "observed_perf_counter_ns": time.perf_counter_ns(),
                "ao_thread_id": threading.get_ident(),
                "generated_sample_count_before_refill": generated,
                "generated_frontier_block": math.ceil(generated / chunk),
                "next_write_block_before_refill": next_write,
                "queued_samples_before_refill": queued,
                "queued_ms_before_refill": queued * 1000.0 / scheduler.config.sample_rate,
                "lead_samples": scheduler.config.ao_lead_samples,
                "deadline_samples": scheduler.config.ao_deadline_samples,
                "deadline_ms": scheduler.config.ao_deadline_samples * 1000.0 / scheduler.config.sample_rate,
                "deficit_blocks": deficit, "ready_contiguous_prefix_blocks": ready,
                "missing_blocks": missing, "first_missing_target_block": first_missing,
                "first_missing_target_sample_index": (first_missing * chunk
                                                       if first_missing is not None else None),
                "first_missing_target_headroom_samples": headroom,
                "first_missing_target_headroom_ms": (headroom * 1000.0 / scheduler.config.sample_rate
                                                     if headroom is not None else None),
                "expected_source_sample_index_for_missing_target": expected_source,
                "urgent_missing_precondition": urgent,
                "producer_running_precondition": producer_running,
                "active_output_precondition": active_output,
                "state_before_refill": {
                    "mode": snapshot.mode, "acquiring": snapshot.acquiring,
                    "armed": snapshot.armed, "transition_pending": snapshot.transition_pending,
                    "generation": snapshot.generation, "stim_mode": snapshot.stim_mode,
                    "fault": snapshot.fault,
                },
                "core_context_observed_before_refill": context,
            }
        start_ns = time.perf_counter_ns()
        cpu_start_ns = time.thread_time_ns()
        probe.ao_thread_id = threading.get_ident()
        try:
            original()
        finally:
            end_ns = time.perf_counter_ns()
            cpu_end_ns = time.thread_time_ns()
        if pre is not None:
            probe.append(None, "ao_service_refill_near_deadline", start_ns, end_ns,
                         cpu_start_ns, cpu_end_ns, threading.get_ident())
        after_miss = scheduler.stats.get("active_demand_misses", 0)
        if pre is not None and after_miss > before_miss and probe.active_miss is None:
            pre.update({
                "active_demand_misses_before": before_miss,
                "active_demand_misses_after": after_miss,
                "generated_sample_count_after_refill": scheduler.generated_sample_count,
                "next_write_block_after_refill": scheduler._next_write_block,
                "queue_depth_after_refill": len(scheduler.commands),
                "state_observed_after_refill": dataclasses.asdict(scheduler.state.snapshot()),
                "active_phases_at_fault_observation": probe.active_snapshot(),
                "phase_ring_at_fault_observation": probe.snapshot(),
                "timing_interpretation": (
                    "AO scheduler decision observation, not proven fault onset; "
                    "nested wall timings may overlap and include thread scheduling waits."
                ),
            })
            probe.active_miss = pre

    scheduler._service_refill_locked = wrapped


def instrument(harness: bench._Harness, probe: PhaseProbe) -> None:
    """Add source-indexed observers without changing the measured Core boundary."""
    core = harness.core
    original_core_process = core.process

    def core_process(*args, **kwargs):
        source = kwargs.get("sample_index")
        probe.local.source_index = int(source) if source is not None else None
        ident = threading.get_ident()
        probe.core_thread_id = ident
        start = time.perf_counter_ns()
        cpu_start = time.thread_time_ns()
        with probe.lock:
            probe.active.setdefault(ident, []).append(
                ("processing_core_call", int(source) if source is not None else None, start, ident)
            )
        previous = probe.previous_core_end_ns
        if previous is not None:
            previous_cpu = probe.previous_core_thread_cpu_end_ns
            probe.append(int(source), "producer_between_core_calls", previous, start,
                         previous_cpu if previous_cpu is not None else cpu_start,
                         cpu_start, ident)
        try:
            return original_core_process(*args, **kwargs)
        finally:
            end = time.perf_counter_ns()
            cpu_end = time.thread_time_ns()
            probe.append(int(source) if source is not None else None,
                         "processing_core_total_wrapper", start, end, cpu_start, cpu_end, ident)
            probe.previous_core_end_ns = end
            probe.previous_core_thread_cpu_end_ns = cpu_end
            with probe.lock:
                stack = probe.active.get(ident)
                if stack:
                    stack.pop()
                    if not stack:
                        probe.active.pop(ident, None)
            probe.local.source_index = None

    core.process = core_process
    # These nested spans overlap the full Core interval. The harness retains
    # the original exact measured Core boundary and acceptance calculation.
    probe.timed(core.controllers["esn"], "step", "esn_controller_step")
    bridge = harness.esn.streamer
    probe.timed(bridge, "process", "esn_bridge_process")
    probe.timed(bridge, "_condition_model_output", "esn_conditioning")
    probe.timed(bridge, "_update_pulse_state", "esn_event_detector")
    streamer = bridge.streamer
    probe.timed(streamer, "process_chunk", "esn_streamer_process_chunk")
    probe.timed(streamer._dec, "process", "esn_decimator_and_anti_alias")
    probe.timed(streamer._fir, "process", "esn_model_band_filter")
    probe.timed(streamer._model, "run", "esn_model_recurrence")
    probe.timed(streamer._up, "process", "esn_upsampler")
    probe.timed(core.safety, "process", "safety_process")
    probe.timed(core.actuator, "process", "actuator_process")
    probe.timed(harness.recorder, "submit", "recorder_admission")
    probe.timed(core, "_publish_ui", "ui_submission_ring_and_job")
    probe.timed(core._display_ring, "append", "ui_ring_append")
    probe.timed(core._display_ring, "take_new_points", "ui_ring_take_new_points")
    probe.timed(core._ui_jobs, "put_nowait", "ui_job_queue_admission")
    ui_queue = core._ui_jobs
    observed_mutex = TimedPrimitiveLock(ui_queue.mutex, probe)
    ui_queue.mutex = observed_mutex
    for condition in (ui_queue.not_empty, ui_queue.not_full, ui_queue.all_tasks_done):
        condition._lock = observed_mutex
        condition.acquire = observed_mutex.acquire
        condition.release = observed_mutex.release
        condition._is_owned = observed_mutex.is_owned

    # Async event-loop publication is separate from the synchronous Core UI
    # ring/job phase and has no direct source-index timestamp.
    original_hub_publish = harness.hub.publish

    def hub_publish(packet: str) -> None:
        start = time.perf_counter_ns()
        ident = threading.get_ident()
        try:
            return original_hub_publish(packet)
        finally:
            probe.publish_wall_ns.append(time.perf_counter_ns() - start)
            probe.publish_thread_ids.append(ident)

    harness.hub.publish = hub_publish
    original_call_soon = harness.loop.call_soon_threadsafe

    def call_soon_threadsafe(callback, *args, **kwargs):
        if callback is hub_publish:
            scheduled_ns = time.perf_counter_ns()

            def dispatch(*callback_args):
                probe.schedule_to_publish_ns.append(max(0, time.perf_counter_ns() - scheduled_ns))
                return callback(*callback_args)

            return original_call_soon(dispatch, *args, **kwargs)
        return original_call_soon(callback, *args, **kwargs)

    harness.loop.call_soon_threadsafe = call_soon_threadsafe
    _service_capture(probe, harness)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bench.json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace", type=Path,
        default=ROOT.parent / "tmp" / "ca3_import_validation" / "2026_05_29_0001.h5",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=360.0)
    parser.add_argument("--lead-ms", type=float, default=100.0)
    parser.add_argument("--progress-interval-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0:
        parser.error("duration must be finite and positive")
    trace = args.trace.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TEMP", str(ROOT.parent / "tmp"))
    os.environ.setdefault("TMP", str(ROOT.parent / "tmp"))
    hashes_before = selected_hashes(trace, app.ESN_ARTIFACT)
    revision_before = bench.git_revision()
    replay = bench.load_replay(trace)
    config = replace(
        bench._make_config("esn", args.lead_ms, output / "recordings"),
        electrode_labels=replay.channel_labels, ctx_index=replay.ctx_index,
    )
    metadata = {
        "application": "Echo Chamber ESN first-fault phase diagnostic",
        "source": {"path": str(trace), "sha256": hashes_before[str(trace)],
                   "sample_count": replay.sample_count, "sample_rate_hz": replay.sample_rate_hz,
                   "labels": replay.channel_labels, "roles": replay.channel_roles,
                   "ctx_index": replay.ctx_index},
        "configuration": app.serializable_config(config),
        "git_revision": revision_before,
        "diagnostic_only": True,
        "physical_pulse_claims": False,
    }
    progress_path = output / "progress.jsonl"
    harness = bench._Harness(
        config, trace.stem, metadata, progress_path=progress_path,
        progress_interval_s=args.progress_interval_seconds,
        paced=True, simulation=False, on_generated=None,
        browser_ui=False, external_browser=False,
    )
    probe = PhaseProbe()
    stop_reason = "duration_limit"
    primary_error: BaseException | None = None
    metrics = None
    try:
        instrument(harness, probe)
        limit_samples = min(replay.sample_count, int(round(args.duration_seconds * replay.sample_rate_hz)))
        for source_index, values in bench._replay_blocks(replay, limit_samples):
            harness.process(source_index, values, replay.channel_roles, replay.channel_labels, replay.ctx_index)
            if harness.state.snapshot().fault is not None:
                stop_reason = "first_runtime_fault_observed_after_core_and_scheduler_submission"
                break
        metrics = harness.finish()
    except BaseException as exc:
        primary_error = exc
        try:
            harness.abort()
        except BaseException as cleanup_error:
            exc.add_note(f"Diagnostic cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")
    hashes_after = selected_hashes(trace, app.ESN_ARTIFACT)
    report = {
        "schema": "echo-chamber-esn-ao-phase-diagnostic-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "stop_reason": stop_reason,
        "trace": str(trace),
        "controller": "esn", "lead_ms": args.lead_ms,
        "duration_limit_seconds": args.duration_seconds,
        "timing_boundary": "Harness Core.process wall-time gate unchanged; phase spans are nested diagnostics",
        "observer_overhead": (
            "Per-call perf_counter_ns/thread_time_ns wrappers and bounded in-memory phase ring; "
            "these measurements may perturb scheduling and are not acceptance evidence."
        ),
        "interpretation_limits": [
            "AO event is an observation of a scheduler decision, not a proven physical fault onset.",
            "Nested phase wall spans overlap and include possible thread scheduling waits; do not sum as CPU attribution.",
            "A zero or quantized thread CPU delta does not prove exclusive blocking or identify the cause.",
            "WebSocket client transport is active, but no external browser was connected or rendering verified.",
            "AO values are software commands; this probe makes no physical delivery or latency claim.",
        ],
        "thread_ids": {"core": probe.core_thread_id, "ao_scheduler": probe.ao_thread_id,
                       "hub_publish_threads": sorted(set(probe.publish_thread_ids))},
        "phase_ring_capacity": PHASE_CAPACITY,
        "phase_samples": probe.snapshot(),
        "ao_decision_at_first_active_miss": probe.active_miss,
        "background_hub_publish": {
            "count": len(probe.publish_wall_ns),
            "wall_ns": list(probe.publish_wall_ns),
            "schedule_to_publish_ns": list(probe.schedule_to_publish_ns),
        },
        "ui_job_queue_mutex_samples": [
            {"source_sample_index": source, "kind": kind, "duration_ns": duration,
             "thread_id": ident}
            for source, kind, duration, ident in probe.ui_queue_lock_rows
        ],
        "metrics": dataclasses.asdict(metrics) if metrics is not None else None,
        "recording_file": str(harness.recording_path),
        "recording_completed": bool(metrics.recording_completed) if metrics is not None else False,
        "recording_committed_samples": int(metrics.recording_committed_samples) if metrics is not None else None,
        "progress_file": str(progress_path),
        "error": f"{type(primary_error).__name__}: {primary_error}" if primary_error else None,
        "revision_before": revision_before,
        "revision_after": bench.git_revision(),
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
        "hashes_unchanged": hashes_before == hashes_after,
    }
    write_json(output / "phase-diagnostic.json", report)
    print(json.dumps({key: report[key] for key in (
        "stop_reason", "recording_completed", "recording_committed_samples",
        "revision_before", "revision_after", "hashes_unchanged", "error",
    )}, indent=2, allow_nan=False))
    return int(primary_error is not None or not report["hashes_unchanged"])


if __name__ == "__main__":
    raise SystemExit(main())
