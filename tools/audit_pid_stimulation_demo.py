#!/usr/bin/env python3
"""Audit the saved, software-only PID tracking comparison and make its figure.

The auditor reads compact JSON, HDF5 diagnostic/event datasets, and the saved
electrode/AO-command arrays. It never treats an AO commit as physical delivery.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_RATE = 20_000
EXPECTED_BLOCK = 100
EXPECTED_LEAD_MS = 100.0
EXPECTED_DURATION_S = 64.0
EXPECTED_SAMPLES = int(EXPECTED_RATE * EXPECTED_DURATION_S)
PULSE_SAMPLES = 100  # Core PulseActuator default: 5 ms at 20 kHz.
PULSE_FREQUENCY_HZ = 100.0
PULSE_AMPLITUDE_V = 0.5  # Existing harness set_stim("threshold_pulse", 0.5).
EXPECTED_PID = {
    "kp": 0.2, "ki": 0.06, "kd": 0.0, "target_z": 1.0,
    "direction": 1, "max_rate_hz": 2.0, "update_ms": 5.0,
}
PHASES = ("stimulated", "no_stimulation_control", "capacity_reference")


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\0")
    return str(value)


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, np.generic):
        return _safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class PhaseAudit:
    name: str
    case_id: str
    recording_path: Path
    failures: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    figure_data: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return not self.failures


def _check_runtime_metrics(name: str, phase: dict[str, Any], *,
                           expected_samples: int, requires_pulses: bool) -> list[str]:
    failures: list[str] = []
    metrics = phase.get("metrics", {})
    if not isinstance(metrics, dict):
        return [f"{name}: metrics are missing or malformed"]
    if int(metrics.get("source_samples", -1)) != expected_samples:
        failures.append(f"{name}: source_samples is not the complete finite source")
    if int(metrics.get("recording_committed_samples", -1)) != expected_samples:
        failures.append(f"{name}: recording_committed_samples is incomplete")
    if not metrics.get("recording_completed"):
        failures.append(f"{name}: recorder did not close complete")
    if not metrics.get("ao_drain_complete"):
        failures.append(f"{name}: AO drain was not acknowledged")
    if int(metrics.get("ao_final_queue_depth", -1)) != 0:
        failures.append(f"{name}: AO queue is not empty after drain")
    if metrics.get("runtime_fault"):
        failures.append(f"{name}: runtime fault: {metrics['runtime_fault']}")
    for key in ("ao_active_demand_misses", "ao_late_commands", "ao_rejected_stale",
                "ao_noncommitted_nonzero_blocks", "source_index_gap_samples", "skipped_source_samples",
                "progress_events_dropped", "progress_events_rejected", "progress_events_pending"):
        if int(metrics.get(key, 0) or 0) != 0:
            failures.append(f"{name}: {key} is nonzero ({metrics.get(key)})")
    if metrics.get("physical_pulse_claims") is not False:
        failures.append(f"{name}: physical_pulse_claims must be false")

    requests = set(map(str, metrics.get("pulse_request_ids") or []))
    commits = set(map(str, metrics.get("pulse_commit_ids") or []))
    expected_requests = int(metrics.get("requested_pulse_events", -1))
    expected_commits = int(metrics.get("pulse_commit_samples", -1))
    generated_nonzero = int(metrics.get("ao_nonzero_generated_samples", -1))
    rendered = int(metrics.get("rendered_pulse_events", -1))
    rejected = int(metrics.get("rejected_pulse_events", -1))
    unclassified = int(metrics.get("unclassified_pulse_events", -1))
    if expected_requests != len(requests):
        failures.append(f"{name}: request count does not match unique pulse request IDs")
    if requires_pulses:
        if not requests:
            failures.append(f"{name}: no PID pulse proposals were recorded")
        if requests != commits:
            failures.append(f"{name}: proposal IDs and committed IDs differ")
        if rendered != len(requests) or rejected != 0 or unclassified != 0:
            failures.append(f"{name}: pulse proposal classification is incomplete or rejected")
        if expected_commits != len(requests) * PULSE_SAMPLES:
            failures.append(f"{name}: committed pulse samples do not equal full pulse widths")
        if generated_nonzero <= 0:
            failures.append(f"{name}: no nonzero generated AO samples were reported")
    else:
        if requests or commits or expected_requests != 0 or expected_commits != 0 or generated_nonzero != 0:
            failures.append(f"{name}: disabled control unexpectedly proposed or generated stimulation")
    return failures


def _metadata_pid_settings(metadata: dict[str, Any]) -> dict[str, Any]:
    return (metadata.get("configuration", {}).get("controller_settings", {}).get("pid", {})
            if isinstance(metadata.get("configuration"), dict) else {})


def _collect_range_coverage(rows: np.ndarray, request_id: str, target: int) -> tuple[list[tuple[int, int]], list[str]]:
    spans: list[tuple[int, int]] = []
    failures: list[str] = []
    for row in rows:
        if _text(row["request_id"]) != request_id or _text(row["stage"]) != "ao_commit":
            continue
        start = int(row["sample_index"])
        end = int(row["range_end_sample_index"])
        if end <= start:
            failures.append(f"{request_id}: committed AO range has no known positive extent")
        else:
            spans.append((start, end))
    spans.sort()
    if not spans:
        failures.append(f"{request_id}: no committed AO ranges")
        return spans, failures
    cursor = target
    for start, end in spans:
        if start != cursor:
            failures.append(f"{request_id}: AO committed ranges have a gap or overlap at {cursor}")
        cursor = end
    if spans[0][0] != target or cursor != target + PULSE_SAMPLES:
        failures.append(f"{request_id}: committed AO ranges do not cover the original pulse exactly")
    return spans, failures


def _read_h5(phase_name: str, phase: dict[str, Any], recording_path: Path,
             *, expected_samples: int) -> tuple[list[str], dict[str, Any] | None]:
    failures: list[str] = []
    if not recording_path.is_file():
        return [f"{phase_name}: recording file is missing: {recording_path}"], None
    arrays: dict[str, Any] = {}
    metrics = phase.get("metrics", {})
    try:
        with h5py.File(recording_path, "r") as h5:
            attrs = h5.attrs
            metadata = json.loads(_text(attrs["metadata_json"]))
            config = metadata.get("configuration", {})
            fs = int(config.get("sample_rate", -1))
            block = int(config.get("chunk_size", -1))
            lead_ms = float(config.get("ao_target_lead_ms", float("nan")))
            if int(attrs.get("complete", 0)) != 1:
                failures.append(f"{phase_name}: H5 complete flag is not set")
            if int(attrs.get("committed_samples", -1)) != expected_samples:
                failures.append(f"{phase_name}: H5 committed source extent is incomplete")
            if _text(attrs.get("format_tag", "")) != "echoChamber_H5_v5":
                failures.append(f"{phase_name}: H5 format is not v5")
            if fs != EXPECTED_RATE or int(attrs.get("sample_rate_hz", -1)) != EXPECTED_RATE:
                failures.append(f"{phase_name}: recorded sample rate is not 20 kHz")
            if block != EXPECTED_BLOCK:
                failures.append(f"{phase_name}: recorded block size is not 100 samples")
            if not math.isclose(lead_ms, EXPECTED_LEAD_MS, abs_tol=1e-9):
                failures.append(f"{phase_name}: recorded output lead is not 100 ms")
            if metadata.get("physical_pulse_claims") is not False:
                failures.append(f"{phase_name}: H5 metadata makes a physical pulse claim")
            if metadata.get("plant_claim") != "generic synthetic plant, not tissue model":
                failures.append(f"{phase_name}: plant identity is missing or not the generic synthetic plant")
            if metadata.get("controller") != "pid":
                failures.append(f"{phase_name}: H5 controller metadata is not PID")
            if int(attrs.get("committed_samples", -1)) != int(metrics.get("recording_committed_samples", -2)):
                failures.append(f"{phase_name}: H5 extent disagrees with report committed samples")
            if phase_name != "capacity_reference":
                pid_settings = _metadata_pid_settings(metadata)
                for key, expected in EXPECTED_PID.items():
                    if key not in pid_settings or not math.isclose(float(pid_settings[key]), float(expected), abs_tol=1e-12):
                        failures.append(f"{phase_name}: H5 PID setting {key} differs from accepted tracking preset")
            else:
                forced_settings = phase.get("pid_reference_settings", {})
                pid_settings = _metadata_pid_settings(metadata)
                for key in ("kp", "ki", "kd", "target_z", "max_rate_hz"):
                    if (key not in forced_settings or key not in pid_settings
                            or not math.isclose(float(forced_settings[key]), float(pid_settings[key]),
                                                rel_tol=0.0, abs_tol=1e-12)):
                        failures.append(
                            f"capacity_reference: H5 PID setting {key} differs from the explicit forced-demand reference")

            signal_shapes = {key: tuple(h5[key].shape) for key in (
                "signals/ai_raw_V", "signals/electrode_mV", "signals/ao_command_V")}
            if signal_shapes["signals/ai_raw_V"] != (2, expected_samples):
                failures.append(f"{phase_name}: raw AI signal shape does not cover the source")
            if signal_shapes["signals/electrode_mV"] != (2, expected_samples):
                failures.append(f"{phase_name}: electrode signal shape does not cover the source")
            if signal_shapes["signals/ao_command_V"] != (1, expected_samples):
                failures.append(f"{phase_name}: source-indexed AO command shape does not cover the source")
            if len(h5["events/sample_discontinuities"]) != 0:
                failures.append(f"{phase_name}: H5 contains source discontinuities")

            block_rows = h5["diagnostics/blocks"]
            ctrl_rows = h5["diagnostics/controller_blocks"]
            expected_blocks = expected_samples // EXPECTED_BLOCK
            if len(block_rows) != expected_blocks or len(ctrl_rows) != expected_blocks:
                failures.append(f"{phase_name}: diagnostic block row count is incomplete")
            offsets = np.arange(expected_blocks, dtype=np.uint64) * EXPECTED_BLOCK
            for start in range(0, min(expected_blocks, len(block_rows)), 16_384):
                stop = min(expected_blocks, start + 16_384)
                rows = block_rows[start:stop]
                if not np.array_equal(rows["sample_offset"], offsets[start:stop]):
                    failures.append(f"{phase_name}: H5 source block offsets contain a gap or reorder")
                    break
                if not np.all(rows["sample_count"] == EXPECTED_BLOCK):
                    failures.append(f"{phase_name}: H5 source block contains a partial block")
                    break
                if not np.array_equal(rows["ao_target_sample_index"],
                                      rows["ai_sample_index"] + int(round(fs * lead_ms / 1000.0))):
                    failures.append(f"{phase_name}: source-to-AO target indices disagree with configured lead")
                    break
            for start in range(0, min(expected_blocks, len(ctrl_rows)), 16_384):
                stop = min(expected_blocks, start + 16_384)
                rows = ctrl_rows[start:stop]
                if not np.array_equal(rows["sample_offset"], offsets[start:stop]):
                    failures.append(f"{phase_name}: controller diagnostic offsets contain a gap or reorder")
                    break
                if any(_text(item) != "pid" for item in rows["controller_id"]):
                    failures.append(f"{phase_name}: controller diagnostics contain a non-PID row")
                    break

            signal = h5["signals/ao_command_V"]
            ao_nonzero = 0
            ao_peak = 0.0
            available_ao = min(expected_samples, int(signal.shape[1]))
            for start in range(0, available_ao, 1_000_000):
                data = np.asarray(signal[0, start:min(available_ao, start + 1_000_000)])
                if not np.all(np.isfinite(data)):
                    failures.append(f"{phase_name}: AO command has non-finite samples")
                    break
                ao_nonzero += int(np.count_nonzero(data))
                ao_peak = max(ao_peak, float(np.max(np.abs(data))) if data.size else 0.0)
            if ao_nonzero != int(metrics.get("ao_nonzero_generated_samples", -1)):
                failures.append(f"{phase_name}: H5 AO command nonzero count differs from generated-AO metric")
            if ao_peak > float(config.get("safety", {}).get("max_command_v", 0.0)) + 1e-6:
                failures.append(f"{phase_name}: H5 AO command exceeds the configured voltage limit")
            if not phase_name == "no_stimulation_control" and ao_peak > PULSE_AMPLITUDE_V + 1e-5:
                failures.append(f"{phase_name}: generated pulse command exceeds the existing 0.5 V demo amplitude")
            if phase_name == "no_stimulation_control" and ao_nonzero:
                failures.append(f"{phase_name}: disabled control H5 contains a nonzero AO command")

            action_rows = h5["events/action_stages"][:]
            rendered: dict[str, dict[str, int]] = {}
            stage_ids: dict[str, set[str]] = {}
            for row in action_rows:
                stage = _text(row["stage"])
                request_id = _text(row["request_id"])
                if request_id:
                    stage_ids.setdefault(stage, set()).add(request_id)
                if stage == "rendered_for_ao" and request_id:
                    rendered[request_id] = {
                        "source_sample_index": int(row["source_sample_index"]),
                        "target_sample_index": int(row["target_sample_index"]),
                    }
            report_requests = set(map(str, metrics.get("pulse_request_ids") or []))
            report_commits = set(map(str, metrics.get("pulse_commit_ids") or []))
            h5_committed = stage_ids.get("ao_committed", set())
            h5_reservations = stage_ids.get("ao_reserved", set())
            range_ids = stage_ids.get("ao_commit", set())
            if set(rendered) != report_requests:
                failures.append(f"{phase_name}: H5 rendered pulse IDs differ from report requests")
            if h5_committed != report_commits or range_ids != report_commits or h5_reservations != report_commits:
                failures.append(f"{phase_name}: H5 AO outcome IDs do not match report committed IDs")
            if stage_ids.get("ao_cancelled") or stage_ids.get("ao_write_unknown"):
                failures.append(f"{phase_name}: H5 includes canceled or unknown pulse-output ranges")

            pulse_rows: list[dict[str, Any]] = []
            commits = action_rows[action_rows["stage"] == b"ao_commit"]
            ordered_ids = sorted(report_commits, key=lambda item: rendered.get(item, {}).get(
                "source_sample_index", 2**63 - 1))
            for pulse_id in ordered_ids:
                event = rendered.get(pulse_id)
                if event is None:
                    failures.append(f"{phase_name}: committed pulse {pulse_id} has no rendered source origin")
                    continue
                source_index = event["source_sample_index"]
                target_index = event["target_sample_index"]
                if target_index - source_index != int(round(fs * lead_ms / 1000.0)):
                    failures.append(f"{phase_name}: pulse {pulse_id} source and AO target indices disagree")
                spans, span_failures = _collect_range_coverage(commits, pulse_id, target_index)
                failures.extend(f"{phase_name}: {item}" for item in span_failures)
                wave = np.asarray(signal[0, source_index:source_index + PULSE_SAMPLES], dtype=np.float64)
                if wave.size != PULSE_SAMPLES:
                    failures.append(f"{phase_name}: source waveform for {pulse_id} is truncated")
                    continue
                expected_wave = PULSE_AMPLITUDE_V * np.sin(
                    2.0 * np.pi * PULSE_FREQUENCY_HZ * np.arange(PULSE_SAMPLES) / fs)
                if not np.allclose(wave, expected_wave, rtol=1e-5, atol=2e-6):
                    failures.append(f"{phase_name}: source AO command for {pulse_id} is not the configured safe sine pulse")
                slew = float(np.max(np.abs(np.diff(np.r_[0.0, wave, 0.0]))) * fs)
                area = float(np.sum(np.abs(wave)) / fs)
                safety = config.get("safety", {})
                if slew > float(safety.get("max_slew_v_per_s", 0.0)) + 1e-3:
                    failures.append(f"{phase_name}: pulse {pulse_id} exceeds the configured slew limit")
                if area > float(safety.get("max_abs_area_v_s", 0.0)) + 1e-9:
                    failures.append(f"{phase_name}: pulse {pulse_id} exceeds the configured absolute-area limit")
                if pulse_rows and source_index - pulse_rows[-1]["source_sample_index"] < int(round(0.5 * fs)):
                    failures.append(f"{phase_name}: pulse IDs violate the shared 0.5 s refractory rule")
                pulse_rows.append({
                    "pulse_id": pulse_id, "source_sample_index": source_index,
                    "target_sample_index": target_index,
                    "effect_start_sample_index": target_index + int(round(
                        float(metadata["plant"]["effect_delay_ms"]) * fs / 1000.0)),
                    "committed_ranges": spans,
                    "committed_samples": sum(end - start for start, end in spans),
                    "command_nonzero_samples": int(np.count_nonzero(wave)),
                    "command_peak_v": float(np.max(np.abs(wave))),
                })
            if int(metrics.get("pulse_commit_samples", 0)) != sum(
                    int(item["committed_samples"]) for item in pulse_rows):
                failures.append(f"{phase_name}: report pulse-commit sample count differs from H5 AO ranges")

            # Read the signal array only in block-sized slices for reproducible activity summaries.
            electrode = h5["signals/electrode_mV"]
            available_samples = min(expected_samples, int(electrode.shape[1]))
            available_blocks = min(len(block_rows), available_samples // EXPECTED_BLOCK)
            rms_rows = np.full(available_blocks, np.nan, dtype=np.float64)
            for block_start in range(0, available_blocks, 10_000):
                block_stop = min(available_blocks, block_start + 10_000)
                sample_start = block_start * EXPECTED_BLOCK
                sample_stop = block_stop * EXPECTED_BLOCK
                samples = np.asarray(electrode[0, sample_start:sample_stop], dtype=np.float64)
                if not np.all(np.isfinite(samples)):
                    failures.append(f"{phase_name}: CA3 electrode recording contains non-finite samples")
                    break
                rms_rows[block_start:block_stop] = np.sqrt(
                    np.mean(samples.reshape(-1, EXPECTED_BLOCK) ** 2, axis=1))
            if len(ctrl_rows):
                plot_blocks = min(available_blocks, len(ctrl_rows))
                diagnostic_rows = ctrl_rows[:plot_blocks]
                rms_rows = rms_rows[:plot_blocks]
                trace_time = (diagnostic_rows["sample_offset"].astype(np.float64)
                              + EXPECTED_BLOCK / 2.0) / fs
                arrays = {
                    "time_s": trace_time,
                    "z_score": diagnostic_rows["z_score"].astype(np.float64),
                    "envelope_mv": diagnostic_rows["envelope_mV"].astype(np.float64),
                    "target_z": diagnostic_rows["target_z"].astype(np.float64),
                    "feature_valid": diagnostic_rows["feature_valid"].astype(bool),
                    "ca3_rms_mv": rms_rows,
                    "pulse_rows": pulse_rows,
                    "ao_nonzero_samples": ao_nonzero,
                    "ao_peak_v": ao_peak,
                    "action_stage_counts": {stage: len(ids) for stage, ids in stage_ids.items()},
                    "metadata": metadata,
                    "signal_shapes": {k: list(v) for k, v in signal_shapes.items()},
                }
            return failures, arrays
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return failures + [f"{phase_name}: could not audit H5 evidence: {type(exc).__name__}: {exc}"], None


def _phase_path(report_path: Path, phase_name: str, phase: dict[str, Any]) -> Path:
    case_id = str(phase.get("metrics", {}).get("case_id", ""))
    if not case_id or Path(case_id).name != case_id:
        return report_path.parent / "recordings" / "__invalid_case_id__.h5"
    root = report_path.parent / "recordings"
    if phase_name == "capacity_reference":
        root = root / "capacity_reference"
    return root / f"{case_id}.h5"


def _h5_activity_summary(data: dict[str, Any], plant: dict[str, Any]) -> dict[str, Any]:
    time_s = np.asarray(data["time_s"], dtype=np.float64)
    rms = np.asarray(data["ca3_rms_mv"], dtype=np.float64)
    finite = np.isfinite(rms)
    baseline = (time_s >= 2.0) & (time_s < 10.5) & finite
    disturbance_start = float(plant["disturbance_start_s"])
    disturbance_stop = float(plant["disturbance_stop_s"])
    disturbance = (time_s >= disturbance_start + 1.0) & (time_s < disturbance_stop) & finite
    recovery = (time_s >= disturbance_stop + 1.0) & finite
    result: dict[str, Any] = {}
    for name, mask in (("baseline", baseline), ("disturbance", disturbance), ("recovery", recovery)):
        result[f"{name}_mean_ca3_block_rms_mv"] = float(np.mean(rms[mask])) if np.any(mask) else None
        result[f"{name}_observed_blocks"] = int(np.count_nonzero(mask))
    result["disturbance_window_complete"] = bool(
        result["disturbance_observed_blocks"] == int(round(
            (disturbance_stop - disturbance_start - 1.0) * EXPECTED_RATE / EXPECTED_BLOCK)))
    return result


def _capacity_reference_failures(capacity: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if capacity.get("capacity_reference_reaches_target") is not True:
        failures.append("capacity_reference: forced-demand capacity did not reach target")
    if int(capacity.get("generated_effect_pulses_in_steady_window", 0)) <= 0:
        failures.append("capacity_reference: no generated effect pulses in the steady window")
    forced = capacity.get("pid_reference_settings", {})
    expected_forced = {"kp": 1.0, "ki": 0.0, "kd": 0.0,
                       "target_z": -1000.0, "max_rate_hz": 2.0}
    for key, expected in expected_forced.items():
        if (key not in forced or not math.isclose(float(forced[key]), expected,
                                                  rel_tol=0.0, abs_tol=1e-12)):
            failures.append(
                f"capacity_reference: forced-demand reference {key} differs from its declared fixture")
    return failures


def audit_report(report_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if report.get("schema") != "echo-chamber-responsive-simulation-matrix-v1":
        failures.append("report schema is not echo-chamber-responsive-simulation-matrix-v1")
    if report.get("scenario") != "capacity_checked_pid_tracking":
        failures.append("report scenario is not the accepted capacity-checked tracking fixture")
    if report.get("physical_pulse_claims") is not False:
        failures.append("report physical_pulse_claims must be false")
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != 1 or cases[0].get("controller") != "pid":
        failures.append("report must contain exactly one PID comparison")
        cases = cases if isinstance(cases, list) else []
    if not cases:
        return {"passed": False, "failures": failures}, {"phases": {}}
    case = cases[0]
    if case.get("plant_is_tissue_model") is not False:
        failures.append("report must identify the plant as not a tissue model")
    if case.get("simulation_acceptance_passed") is not True:
        failures.append("existing tracking, recovery, and capacity acceptance did not pass")
    if case.get("tracking_gate_passed") is not True:
        failures.append("tracking gate did not pass")
    if case.get("plant_recovery_passed") is not True:
        failures.append("plant recovery gate did not pass")
    if case.get("tracking_iae_improved") is not True:
        failures.append("tracking IAE did not improve over the no-stimulation control")
    if float(case.get("observed_disturbance_rms_reduction_mv") or 0.0) <= 0:
        failures.append("saved activity comparison does not show positive disturbance RMS reduction")
    plant = case.get("stimulated", {}).get("plant", {})
    for key, expected in (("disturbance_amplitude_mv", 0.003), ("disturbance_start_s", 12.0),
                          ("disturbance_stop_s", 42.0), ("effect_recovery_s", 5.0),
                          ("amplitude_reduction_per_pulse_mv", 0.0005), ("seed", 1414)):
        if key not in plant or not math.isclose(float(plant[key]), expected, rel_tol=0.0, abs_tol=1e-12):
            failures.append(f"synthetic plant parameter {key} differs from the predeclared fixture")
    config = case.get("stimulated", {}).get("environment", {}).get("config", {})
    for key, expected in EXPECTED_PID.items():
        actual = config.get("controller_settings", {}).get("pid", {}).get(key)
        if actual is None or not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
            failures.append(f"stimulated PID setting {key} differs from the predeclared target-1 fixture")
    if not math.isclose(float(config.get("ao_target_lead_ms", float("nan"))), EXPECTED_LEAD_MS, abs_tol=1e-9):
        failures.append("stimulated configuration output lead is not 100 ms")

    phase_payloads = {
        "stimulated": case.get("stimulated", {}),
        "no_stimulation_control": case.get("no_stimulation_control", {}),
        "capacity_reference": case.get("capacity_reference", {}),
    }
    phase_results: dict[str, Any] = {}
    phase_arrays: dict[str, Any] = {}
    for phase_name in PHASES:
        phase = phase_payloads[phase_name]
        expected_samples = int(round(float(phase.get("duration_seconds", EXPECTED_DURATION_S)) * EXPECTED_RATE))
        if expected_samples != EXPECTED_SAMPLES:
            phase_failures = [f"{phase_name}: duration is not the required 64 seconds"]
        else:
            phase_failures = []
        phase_failures += _check_runtime_metrics(
            phase_name, phase, expected_samples=EXPECTED_SAMPLES,
            requires_pulses=(phase_name != "no_stimulation_control"),
        )
        recording_path = _phase_path(report_path, phase_name, phase)
        h5_failures, arrays = _read_h5(
            phase_name, phase, recording_path, expected_samples=EXPECTED_SAMPLES)
        phase_failures.extend(h5_failures)
        if phase_name == "stimulated":
            tracking = case.get("stimulated", {}).get("controller_tracking", {})
            if tracking.get("tracking_response_gate_passed") is not True:
                phase_failures.append("stimulated: target-tracking response gate did not pass")
            if tracking.get("raw_ripple_within_bounds") is not True:
                phase_failures.append("stimulated: raw ripple bound did not pass")
        if phase_name == "capacity_reference":
            capacity = case.get("capacity_reference", {})
            phase_failures.extend(_capacity_reference_failures(capacity))
        if arrays is not None:
            phase_arrays[phase_name] = arrays
        phase_results[phase_name] = {
            "passed": not phase_failures, "failures": phase_failures,
            "case_id": phase.get("metrics", {}).get("case_id"),
            "recording_path": str(recording_path.resolve()),
            "runtime_summary": {
                key: phase.get("metrics", {}).get(key) for key in (
                    "source_samples", "recording_committed_samples", "recording_completed",
                    "ao_drain_complete", "runtime_fault", "ao_active_demand_misses",
                    "ao_late_commands", "deadline_exceeded_blocks", "process_p99_ms", "process_max_ms",
                    "requested_pulse_events", "pulse_commit_samples", "ao_nonzero_generated_samples",
                    "pulse_request_ids", "pulse_commit_ids",
                )
            },
            "h5_summary": ({
                "ao_nonzero_samples": arrays["ao_nonzero_samples"],
                "ao_peak_v": arrays["ao_peak_v"],
                "pulse_count": len(arrays["pulse_rows"]),
                "pulse_rows": arrays["pulse_rows"],
                "action_stage_counts": arrays["action_stage_counts"],
                "signal_shapes": arrays["signal_shapes"],
            } if arrays is not None else None),
        }
        failures.extend(phase_failures)
    activity_crosscheck: dict[str, Any] | None = None
    if "stimulated" in phase_arrays and "no_stimulation_control" in phase_arrays:
        active_activity = _h5_activity_summary(phase_arrays["stimulated"], plant)
        control_activity = _h5_activity_summary(phase_arrays["no_stimulation_control"], plant)
        effect = None
        if (active_activity["disturbance_mean_ca3_block_rms_mv"] is not None
                and control_activity["disturbance_mean_ca3_block_rms_mv"] is not None):
            effect = (control_activity["disturbance_mean_ca3_block_rms_mv"]
                      - active_activity["disturbance_mean_ca3_block_rms_mv"])
        activity_crosscheck = {
            "stimulated": active_activity,
            "no_stimulation_control": control_activity,
            "h5_disturbance_rms_reduction_mv": effect,
            "report_disturbance_rms_reduction_mv": case.get("observed_disturbance_rms_reduction_mv"),
            "positive_measured_h5_effect": bool(effect is not None and effect > 0.0),
            "report_h5_effect_match": bool(effect is not None and case.get(
                "observed_disturbance_rms_reduction_mv") is not None
                and math.isclose(effect, float(case["observed_disturbance_rms_reduction_mv"]),
                                 rel_tol=0.0, abs_tol=1e-5)),
        }
        if not active_activity["disturbance_window_complete"]:
            failures.append("stimulated H5 does not contain the full declared disturbance interval")
        if not control_activity["disturbance_window_complete"]:
            failures.append("control H5 does not contain the full declared disturbance interval")
        if not activity_crosscheck["positive_measured_h5_effect"]:
            failures.append("recorded H5 CA3 activity does not show a positive disturbance reduction")
        if not activity_crosscheck["report_h5_effect_match"]:
            failures.append("report disturbance-RMS effect does not match the saved H5 signals")
    output = {
        "schema": "pid-stimulation-demo-audit-v1",
        "report_path": str(report_path.resolve()),
        "report_git_revision": case.get("stimulated", {}).get("environment", {}).get("git_revision"),
        "report_schema": report.get("schema"),
        "scenario": report.get("scenario"),
        "duration_seconds": EXPECTED_DURATION_S,
        "sample_rate_hz": EXPECTED_RATE,
        "block_samples": EXPECTED_BLOCK,
        "ao_lead_ms": EXPECTED_LEAD_MS,
        "pid_settings": EXPECTED_PID,
        "plant_scope": "generic deterministic synthetic plant, not a tissue model",
        "physical_pulse_claims": False,
        "process_timing": {
            name: phase_results.get(name, {}).get("runtime_summary", {})
            for name in PHASES
        },
        "phase_results": phase_results,
        "h5_activity_crosscheck": activity_crosscheck,
        "demo_passed": not failures,
        "failures": failures,
        "scope_note": (
            "Software-only evidence through ProcessingCore, safety, simulated AO scheduler and H5Recorder. "
            "The capacity phase is an explicit forced-demand reference, not regulated-controller output. "
            "Process deadline counts are retained as diagnostics; they do not imply physical delivery."
        ),
    }
    return _safe_json(output), {"case": case, "phases": phase_arrays}


def _write_curated_csv(output_dir: Path, phase_data: dict[str, Any]) -> tuple[Path, Path]:
    series_path = output_dir / "tracking-series.csv"
    pulse_path = output_dir / "pulse-events.csv"
    phases = phase_data["phases"]
    active = phases.get("stimulated")
    control = phases.get("no_stimulation_control")
    capacity = phases.get("capacity_reference")
    if not active:
        raise ValueError("stimulated H5 diagnostics are unavailable; cannot write a truthful time-series export")
    n = len(active["time_s"])
    with series_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "stimulated_z", "stimulated_envelope_mV", "stimulated_target_z",
                         "stimulated_feature_valid", "stimulated_ca3_block_rms_mV",
                         "no_stimulation_z", "no_stimulation_envelope_mV", "no_stimulation_feature_valid",
                         "no_stimulation_ca3_block_rms_mV"))
        for i in range(n):
            writer.writerow((
                float(active["time_s"][i]), float(active["z_score"][i]), float(active["envelope_mv"][i]),
                float(active["target_z"][i]), int(active["feature_valid"][i]), float(active["ca3_rms_mv"][i]),
                float(control["z_score"][i]) if control and i < len(control["z_score"]) else "",
                float(control["envelope_mv"][i]) if control and i < len(control["envelope_mv"]) else "",
                int(control["feature_valid"][i]) if control and i < len(control["feature_valid"]) else "",
                float(control["ca3_rms_mv"][i]) if control and i < len(control["ca3_rms_mv"]) else "",
            ))
    with pulse_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("phase", "pulse_id", "source_sample_index",
            "target_sample_index", "effect_start_sample_index", "committed_samples",
            "command_nonzero_samples", "command_peak_v"))
        writer.writeheader()
        for phase_name, data in (("regulated_stimulated", active),
                                 ("no_stimulation_control", control),
                                 ("forced_capacity_reference", capacity)):
            if data:
                for row in data["pulse_rows"]:
                    writer.writerow({"phase": phase_name, **{k: row[k] for k in (
                        "pulse_id", "source_sample_index", "target_sample_index",
                        "effect_start_sample_index", "committed_samples",
                        "command_nonzero_samples", "command_peak_v")}})
    return series_path, pulse_path


def _make_figure(case: dict[str, Any], phase_data: dict[str, Any], output_path: Path) -> None:
    phases = phase_data["phases"]
    plant = case["stimulated"]["plant"]
    disturbance_start = float(plant["disturbance_start_s"])
    disturbance_stop = float(plant["disturbance_stop_s"])
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1.25, 0.7]})
    ax_rms, ax_z, ax_pulse = axes
    colors = {"stimulated": "#1769aa", "no_stimulation_control": "#8a8a8a",
              "capacity_reference": "#cc6b2c"}
    for name in ("stimulated", "no_stimulation_control"):
        data = phases.get(name)
        if data is None:
            continue
        ax_rms.plot(data["time_s"], data["ca3_rms_mv"], color=colors[name], linewidth=0.8,
                    alpha=0.9, label=name.replace("_", " "))
        valid_z = np.where(data["feature_valid"], data["z_score"], np.nan)
        ax_z.plot(data["time_s"], valid_z, color=colors[name], linewidth=0.75,
                  alpha=0.85, label=name.replace("_", " "))
    for ax in axes[:2]:
        ax.axvspan(disturbance_start, disturbance_stop, color="#f2c14e", alpha=0.18,
                   label="predeclared plant disturbance" if ax is ax_rms else None)
        ax.grid(True, alpha=0.25)
    target = float(case["stimulated"]["controller_tracking"]["target_z"])
    ax_z.axhline(target, color="#222222", linestyle="--", linewidth=1.1, label=f"target z={target:g}")
    ax_rms.set_ylabel("Recorded CA3 block RMS (mV)")
    ax_rms.set_title("Generic synthetic plant output and actual software AO pulse events")
    ax_z.set_ylabel("Causal CA3 envelope z-score")
    ax_z.legend(loc="upper right", ncol=2, fontsize=8)
    ax_rms.legend(loc="upper right", ncol=2, fontsize=8)
    yrows = {"stimulated": 2, "no_stimulation_control": 1, "capacity_reference": 0}
    for name, y in yrows.items():
        data = phases.get(name)
        if data is None:
            continue
        rows = data["pulse_rows"]
        times = [float(row["target_sample_index"]) / EXPECTED_RATE for row in rows]
        ax_pulse.scatter(times, [y] * len(times), s=12, marker="|", linewidths=1.1,
                         color=colors[name])
    counts = {name: len(phases.get(name, {}).get("pulse_rows", [])) for name in yrows}
    ax_pulse.set_yticks(
        [0, 1, 2],
        [f"forced capacity ({counts['capacity_reference']})",
         f"no-stimulation control ({counts['no_stimulation_control']})",
         f"regulated PID ({counts['stimulated']})"],
    )
    ax_pulse.set_ylabel("Committed AO ranges")
    ax_pulse.set_xlabel("Source time (s); pulse marks use AO target sample index")
    ax_pulse.set_xlim(0, EXPECTED_DURATION_S)
    ax_pulse.grid(True, axis="x", alpha=0.25)
    fig.suptitle("Software-only PID tracking demonstration — not tissue or hardware evidence", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def audit(report_path: Path, output_dir: Path) -> tuple[Path, int]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing audit output: {output_dir}")
    output_dir.mkdir(parents=True)
    audit_result, phase_data = audit_report(report_path)
    audit_path = output_dir / "audit.json"
    audit_path.write_text(json.dumps(_safe_json(audit_result), indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
    series = pulses = figure = None
    if phase_data["phases"].get("stimulated"):
        series, pulses = _write_curated_csv(output_dir, phase_data)
        figure = output_dir / "pid-stimulation-demo.png"
        _make_figure(phase_data["case"], phase_data, figure)
    print(json.dumps({"audit": str(audit_path), "passed": audit_result["demo_passed"],
                      "failures": len(audit_result["failures"]),
                      "figure": str(figure) if figure else None,
                      "series_csv": str(series) if series else None,
                      "pulse_csv": str(pulses) if pulses else None}))
    return audit_path, 0 if audit_result["demo_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="completed responsive_simulation JSON from --tracking-simulation")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for the audit, curated CSV files, and figure")
    args = parser.parse_args(argv)
    try:
        _, code = audit(args.report.resolve(), args.output_dir.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
