"""Summarize controller validation JSON into reviewer-facing evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_FS = 20_000
EXPECTED_BLOCK_SAMPLES = 100
EXPECTED_BLOCKS = 360_000
EXPECTED_SAMPLES = 36_000_000
EXPECTED_SECONDS = 1_800.0
EXPECTED_CONTROLLERS = {"rule", "pid", "esn"}
MAX_P99_MS = 2.5
MAX_PROCESS_MS = 5.0


def _load(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"validation result must contain a JSON object: {path}")
    return doc


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _trace_summary(trace: Any) -> dict[str, Any]:
    rows = trace if isinstance(trace, list) else []
    z_values = [_number(row.get("z")) for row in rows if isinstance(row, dict)]
    z_values = [value for value in z_values if value is not None]
    rates = [_number(row.get("requested_rate_hz")) for row in rows if isinstance(row, dict)]
    rates = [value for value in rates if value is not None]
    times = [_number(row.get("time_s")) for row in rows if isinstance(row, dict)]
    times = [value for value in times if value is not None]
    return {
        "sample_count": len(rows),
        "valid_z_count": len(z_values),
        "observed_z_min": min(z_values) if z_values else None,
        "observed_z_max": max(z_values) if z_values else None,
        "requested_rate_min_hz": min(rates) if rates else None,
        "requested_rate_max_hz": max(rates) if rates else None,
        "first_time_s": min(times) if times else None,
        "last_time_s": max(times) if times else None,
    }


def _full_case(case: dict[str, Any]) -> bool:
    sample_rate = _number(case.get("sample_rate_hz"))
    block_samples = int(case.get("block_samples", 0) or 0)
    duration = _number(case.get("duration_source_s"))
    return (int(case.get("processed_blocks", 0)) == EXPECTED_BLOCKS
            and int(case.get("source_samples", 0)) == EXPECTED_SAMPLES
            and sample_rate == EXPECTED_FS and block_samples == EXPECTED_BLOCK_SAMPLES
            and duration is not None and abs(duration - EXPECTED_SECONDS) <= 1 / EXPECTED_FS)


def _case_record(case: dict[str, Any], source_file: Path, run_kind: str,
                 listed_failures: list[str] | None = None,
                 provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    provenance = provenance or {}
    process_p99 = _number(case.get("process_p99_ms"))
    process_max = _number(case.get("process_max_ms"))
    source_samples = int(case.get("source_samples", 0) or 0)
    blocks = int(case.get("processed_blocks", 0) or 0)
    metrics = case.get("metrics", {}) if isinstance(case.get("metrics"), dict) else {}
    failures = list(listed_failures or [])

    def require(condition: bool, message: str) -> None:
        if not condition and message not in failures:
            failures.append(message)

    if process_p99 is None:
        failures.append("process p99 timing is missing or non-finite")
    else:
        require(process_p99 < MAX_P99_MS, f"process p99 {process_p99:.3f} ms is not below {MAX_P99_MS:.1f} ms")
    if process_max is None:
        failures.append("maximum process timing is missing or non-finite")
    else:
        require(process_max < MAX_PROCESS_MS, f"maximum process time {process_max:.3f} ms is not below {MAX_PROCESS_MS:.1f} ms")

    gaps = int(case.get("source_index_gap_samples", 0) or 0)
    skipped = int(case.get("skipped_source_samples", 0) or 0)
    require(gaps == 0 and skipped == 0,
            f"source discontinuity or skipped samples: gaps={gaps}, skipped={skipped}")
    ao_misses = int(case.get("ao_active_demand_misses", 0) or 0)
    ao_late = int(case.get("ao_late_commands", 0) or 0)
    ao_stale = int(case.get("ao_rejected_stale", 0) or 0)
    ao_zero = int(case.get("ao_zero_substitutions", 0) or 0)
    ao_noncommitted = int(case.get("ao_noncommitted_nonzero_blocks", 0) or 0)
    require(ao_misses == 0, f"AO active-demand misses: {ao_misses}")
    require(ao_late == 0, f"late AO commands: {ao_late}")
    require(ao_stale == 0, f"stale AO rejections: {ao_stale}")
    require(ao_noncommitted == 0, f"noncommitted nonzero AO blocks: {ao_noncommitted}")
    ui_submitted = int(case.get("ui_packets_submitted", 0) or 0)
    ui_consumed = int(case.get("ui_packets_consumed", 0) or 0)
    require(ui_submitted > 0 and ui_consumed > 0,
            f"UI packets missing: submitted={ui_submitted}, consumed={ui_consumed}")
    requested_pulses = int(case.get("requested_pulse_events", 0) or 0)
    rendered_pulses = int(case.get("rendered_pulse_events", 0) or 0)
    rejected_pulses = int(case.get("rejected_pulse_events", 0) or 0)
    unclassified_pulses = int(case.get("unclassified_pulse_events", 0) or 0)
    require(requested_pulses == rendered_pulses + rejected_pulses + unclassified_pulses,
            "pulse proposals are not fully partitioned: "
            f"requested={requested_pulses}, rendered={rendered_pulses}, "
            f"rejected={rejected_pulses}, unclassified={unclassified_pulses}")
    require(unclassified_pulses == 0, f"unclassified pulse events: {unclassified_pulses}")

    recording_complete = bool(case.get("recording_completed", False))
    recording_samples = int(case.get("recording_committed_samples", 0) or 0)
    expected_recording_samples = blocks * int(case.get("block_samples", 0) or 0)
    require(recording_complete, "recorder did not finalize as complete")
    require(recording_samples == source_samples == expected_recording_samples,
            "recorded/source/processed sample counts disagree: "
            f"recorded={recording_samples}, source={source_samples}, processed={expected_recording_samples}")

    browser_required = bool(case.get("browser_ui_required", False))
    browser_connected = bool(case.get("browser_connected", False))
    browser_at_finish = bool(case.get("browser_connected_at_finish", False))
    browser_disconnects = int(case.get("browser_disconnect_count", 0) or 0)
    browser_disconnect_seconds = _number(case.get("browser_disconnected_duration_s",
                                                   case.get("browser_disconnect_duration_s",
                                                            case.get("browser_disconnected_duration_seconds",
                                                                     case.get("browser_disconnect_duration_seconds")))))
    browser_disconnect_ms = _number(case.get("browser_disconnected_duration_ms"))
    if browser_disconnect_seconds is None and browser_disconnect_ms is not None:
        browser_disconnect_seconds = browser_disconnect_ms / 1000.0
    browser_open_at_finish = bool(case.get("browser_disconnect_open_at_finish", False))
    if run_kind == "paced":
        require(browser_required, "paced acceptance run did not require a browser UI")
    if browser_required:
        require(browser_connected and browser_at_finish and browser_disconnects == 0
                and browser_disconnect_seconds == 0.0 and not browser_open_at_finish,
                "required browser was absent, disconnected at any point, or missing at finish")

    complete_source = _full_case(case)
    if not complete_source:
        failures.append("short or incomplete source run; not a full-duration acceptance result")
    if int(case.get("deadline_exceeded_blocks", 0) or 0):
        failures.append(f"deadline-exceeded blocks: {int(case['deadline_exceeded_blocks'])}")
    if int(case.get("ao_final_queue_depth", 0) or 0):
        failures.append(f"AO queue not drained: depth={int(case['ao_final_queue_depth'])}")
    if not bool(case.get("ao_drain_complete", False)):
        failures.append("AO drain did not complete")
    if case.get("runtime_fault"):
        failures.append(f"runtime fault: {case['runtime_fault']}")

    status = "FAIL" if any(not item.startswith("short or incomplete") for item in failures) else "SHORT"
    if complete_source and not failures:
        status = "PASS"
    physical_claims = bool(case.get("physical_pulse_claims", False))
    return {
        "case_id": str(case.get("case_id", "unknown")),
        "source_name": str(case.get("source_name", "unknown")),
        "controller": str(case.get("controller", "unknown")),
        "run_kind": run_kind,
        "result_file": str(source_file),
        "source_path": str(provenance.get("source_path", case.get("source_path", "unknown"))),
        "source_sha256": provenance.get("source_sha256", case.get("source_sha256")),
        "status": status,
        "full_1800s_coverage": complete_source,
        "processed_blocks": blocks,
        "expected_blocks": EXPECTED_BLOCKS,
        "source_samples": source_samples,
        "expected_samples": EXPECTED_SAMPLES,
        "duration_source_s": _number(case.get("duration_source_s")),
        "block_coverage_fraction": blocks / EXPECTED_BLOCKS,
        "sample_coverage_fraction": source_samples / EXPECTED_SAMPLES,
        "duration_coverage_fraction": (_number(case.get("duration_source_s")) or 0.0) / EXPECTED_SECONDS,
        "sample_rate_hz": _number(case.get("sample_rate_hz")),
        "block_samples": int(case.get("block_samples", 0) or 0),
        "ao_lead_ms": _number(case.get("ao_lead_ms")),
        "process_p99_ms": process_p99,
        "process_max_ms": process_max,
        "deadline_exceeded_blocks": int(case.get("deadline_exceeded_blocks", 0) or 0),
        "source_index_gap_samples": gaps,
        "skipped_source_samples": skipped,
        "ao_active_demand_misses": ao_misses,
        "ao_late_commands": ao_late,
        "ao_rejected_stale": ao_stale,
        "ao_zero_substitutions": ao_zero,
        "ao_noncommitted_nonzero_blocks": ao_noncommitted,
        "ao_cancelled_blocks": int(case.get("ao_cancelled_blocks", 0) or 0),
        "ao_final_queue_depth": int(case.get("ao_final_queue_depth", 0) or 0),
        "ao_drain_complete": bool(case.get("ao_drain_complete", False)),
        "ao_committed_blocks": int(case.get("ao_committed_blocks", 0) or 0),
        "ao_generated_nonzero_samples": int(case.get("ao_nonzero_generated_samples", 0) or 0),
        "requested_pulse_events": requested_pulses,
        "rendered_pulse_events": rendered_pulses,
        "rejected_pulse_events": rejected_pulses,
        "unclassified_pulse_events": unclassified_pulses,
        "pulse_commit_samples": int(case.get("pulse_commit_samples", 0) or 0),
        "pulse_request_ids": list(case.get("pulse_request_ids", [])),
        "pulse_rejected_ids": list(case.get("pulse_rejected_ids", [])),
        "pulse_commit_ids": list(case.get("pulse_commit_ids", [])),
        "recording_completed": recording_complete,
        "recording_committed_samples": recording_samples,
        "recording_expected_samples_from_processed_blocks": expected_recording_samples,
        "browser_required": browser_required,
        "browser_connected": browser_connected,
        "browser_connected_at_finish": browser_at_finish,
        "browser_disconnect_count": browser_disconnects,
        "browser_disconnected_duration_s": browser_disconnect_seconds,
        "browser_disconnect_open_at_finish": browser_open_at_finish,
        "ui_transport_kind": str(case.get("ui_transport_kind", "unknown")),
        "ui_packets_submitted": ui_submitted,
        "ui_packets_consumed": ui_consumed,
        "recorder_backpressure_policy": str(case.get("recorder_backpressure_policy", "unknown")),
        "recorder_backpressure_wait_count": int(case.get("recorder_backpressure_wait_count", 0) or 0),
        "recorder_backpressure_wait_ms": _number(case.get("recorder_backpressure_wait_ms")),
        "unpaced_backpressure_excluded_from_process_timing": run_kind != "paced",
        "physical_pulse_claims": physical_claims,
        "actuator_evidence_kind": "physical monitor recorded" if case.get("physical_monitor_samples") is not None else "software simulated scheduler",
        "software_generated_nonzero_command_samples": int(case.get("ao_nonzero_generated_samples", 0) or 0),
        "physical_monitor_samples": (int(case["physical_monitor_samples"])
                                     if case.get("physical_monitor_samples") is not None else None),
        "failures": failures,
    }


def _legacy_short_records(doc: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    records = []
    for name, metric in (("unpaced", doc.get("throughput")), ("paced", doc.get("paced", {}).get("timing"))):
        if not isinstance(metric, dict):
            continue
        p99 = _number(metric.get("p99_ms"))
        maximum = _number(metric.get("maximum_ms"))
        count = int(metric.get("count", 0) or 0)
        failures = []
        if p99 is not None and p99 >= MAX_P99_MS:
            failures.append(f"process p99 {p99:.3f} ms is not below {MAX_P99_MS:.1f} ms")
        if maximum is not None and maximum >= MAX_PROCESS_MS:
            failures.append(f"maximum process time {maximum:.3f} ms is not below {MAX_PROCESS_MS:.1f} ms")
        records.append({
            "case_id": f"{path.stem}:{name}", "source_name": str(doc.get("input", "unknown")),
            "controller": str(doc.get("configuration", {}).get("controller", "unspecified")),
            "run_kind": "legacy_short_benchmark", "result_file": str(path), "status": "SHORT",
            "full_1800s_coverage": False, "processed_blocks": count, "expected_blocks": EXPECTED_BLOCKS,
            "source_samples": None, "expected_samples": EXPECTED_SAMPLES,
            "block_coverage_fraction": count / EXPECTED_BLOCKS,
            "sample_coverage_fraction": None,
            "duration_coverage_fraction": (_number(metric.get("duration_s")) or 0.0) / EXPECTED_SECONDS,
            "duration_source_s": _number(metric.get("duration_s")),
            "process_p99_ms": p99, "process_max_ms": maximum,
            "deadline_exceeded_blocks": int(metric.get("over_deadline", 0) or 0),
            "source_index_gap_samples": None, "skipped_source_samples": None,
            "ao_active_demand_misses": None, "ao_late_commands": None,
            "ao_rejected_stale": None, "ao_zero_substitutions": None,
            "ao_noncommitted_nonzero_blocks": None, "ao_committed_blocks": None,
            "ao_generated_nonzero_samples": None, "recording_completed": None,
            "recording_active": None,
            "requested_pulse_events": None, "rendered_pulse_events": None,
            "rejected_pulse_events": None, "pulse_commit_samples": None,
            "pulse_request_ids": [], "pulse_commit_ids": [],
            "recording_committed_samples": None, "browser_required": None,
            "browser_connected": None, "browser_connected_at_finish": None,
            "browser_disconnect_count": None, "recorder_backpressure_policy": "not reported",
            "recorder_backpressure_wait_count": None, "recorder_backpressure_wait_ms": None,
            "unpaced_backpressure_excluded_from_process_timing": name == "unpaced",
            "physical_pulse_claims": False, "software_generated_nonzero_command_samples": None,
            "physical_monitor_samples": None,
            "failures": failures + ["legacy short benchmark lacks full-run recorder, AO, and browser evidence"],
        })
    return records


def _simulation_records(doc: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    cases = doc.get("cases")
    if cases is None and (doc.get("schema") == "echo-chamber-responsive-simulation-v1"
                          or ("metrics" in doc and "stimulation_enabled" in doc and "plant" in doc)):
        cases = [doc]
    records = []
    for case in cases or []:
        controller = str(case.get("controller", "unknown"))
        stimulated = case.get("stimulated", {})
        metrics = stimulated.get("metrics", {}) if isinstance(stimulated, dict) else {}
        if not metrics and isinstance(case.get("metrics"), dict):
            metrics = case["metrics"]
        tracking = case.get("controller_tracking", {})
        if not tracking and isinstance(stimulated, dict):
            tracking = stimulated.get("controller_tracking", {})
        recovery = case.get("plant_recovery", {})
        if not recovery and isinstance(stimulated, dict):
            recovery = stimulated.get("plant_recovery", {})
        regulated_usage = case.get("regulated_controller_output_usage", {})
        if not regulated_usage and isinstance(stimulated, dict):
            regulated_usage = stimulated.get("regulated_controller_output_usage", {})
        plant = case.get("plant", {})
        if not plant and isinstance(stimulated, dict):
            plant = stimulated.get("plant", {})
        capacity_info = case.get("plant_rate_capacity", {})
        if not capacity_info and isinstance(stimulated, dict):
            capacity_info = stimulated.get("plant_rate_capacity", {})
        environment = case.get("environment", {})
        if not environment and isinstance(stimulated, dict):
            environment = stimulated.get("environment", {})
        config = environment.get("config", {}) if isinstance(environment, dict) else {}
        controller_settings = config.get("controller_settings", {}) if isinstance(config, dict) else {}
        settings = controller_settings.get(controller, {}) if isinstance(controller_settings, dict) else {}
        reference_settings = capacity_info.get("pid_reference_settings", {}) if isinstance(capacity_info, dict) else {}
        max_rate = _number(settings.get("max_rate_hz")) if isinstance(settings, dict) else None
        disturbance = _number(plant.get("disturbance_amplitude_mv")) if isinstance(plant, dict) else None
        per_pulse = _number(plant.get("amplitude_reduction_per_pulse_mv")) if isinstance(plant, dict) else None
        recovery_s = _number(plant.get("effect_recovery_s")) if isinstance(plant, dict) else None
        max_reduction = _number(plant.get("max_amplitude_reduction_mv")) if isinstance(plant, dict) else None
        capacity = _number(capacity_info.get(
            "requested_rate_mean_capacity_mv",
            capacity_info.get("maximum_sustainable_mean_reduction_mv")))
        capacity_margin = _number(capacity_info.get(
            "steady_reduction_margin_mv",
            capacity_info.get("requested_rate_capacity_margin_mv",
                              capacity_info.get("mean_capacity_margin_mv"))))
        post_pulse_capacity = _number(capacity_info.get(
            "requested_rate_post_pulse_capacity_mv",
            capacity_info.get("maximum_steady_post_pulse_reduction_mv")))
        capacity_reaches = capacity_info.get(
            "capacity_reference_reaches_target",
            capacity_info.get("requested_rate_reaches_full_disturbance",
                              capacity_info.get("reachable_at_rate_limit")))
        effective_capacity = _number(capacity_info.get("effective_rate_mean_capacity_mv"))
        generated_capacity = _number(capacity_info.get(
            "steady_mean_reduction_mv",
            capacity_info.get("measured_generated_mean_capacity_mv")))
        generated_reaches_target = capacity_info.get("capacity_reference_reaches_target")
        reference_correction = _number(capacity_info.get(
            "target_correction_required_mv",
            capacity_info.get("target_correction_required_mv_for_capacity_reference_z")))
        capacity_basis = "reported plant_rate_capacity" if capacity_info else "not reported"
        if capacity is None and None not in (per_pulse, max_rate, recovery_s, max_reduction):
            capacity = min(max_reduction, per_pulse * max_rate * recovery_s)
            capacity_basis = "derived model estimate"
        if capacity_reaches is None and capacity is not None and disturbance is not None:
            capacity_reaches = bool(capacity >= disturbance)
        requested = _number(metrics.get("requested_pulse_events"))
        rendered = _number(metrics.get("rendered_pulse_events"))
        rejected = _number(metrics.get("rejected_pulse_events"))
        unclassified = _number(metrics.get("unclassified_pulse_events"))
        if (unclassified is None and rejected == 0 and requested is not None
                and requested == rendered):
            # Older short diagnostic reports predate the explicit unknown-status counter.
            unclassified = 0.0
        sim_failures = [f"runtime fault: {metrics['runtime_fault']}"] if metrics.get("runtime_fault") else []
        if any(value is not None for value in (requested, rendered, rejected, unclassified)):
            classified_total = (rendered or 0) + (rejected or 0) + (unclassified or 0)
            if requested is None or rendered is None or rejected is None or unclassified is None or requested != classified_total:
                sim_failures.append(
                    f"pulse proposals are not fully partitioned: requested={requested}, rendered={rendered}, "
                    f"rejected={rejected}, unclassified={unclassified}")
            if unclassified:
                sim_failures.append(f"unclassified pulse events: {unclassified:g}")
        activity_change = _number(case.get("observed_disturbance_rms_reduction_mv"))
        records.append({
            "case_id": str(case.get("case_id", f"{controller}:synthetic")),
            "source_name": "synthetic plant",
            "controller": controller, "run_kind": "synthetic_simulation",
            "result_file": str(path), "status": "SIMULATION",
            "ao_lead_ms": _number(case.get("lead_ms")),
            "duration_seconds": _number(case.get("duration_seconds",
                                                  stimulated.get("duration_seconds") if stimulated else None)),
            "processed_blocks": int(metrics.get("processed_blocks", 0) or 0),
            "expected_blocks": None,
            "source_samples": int(metrics.get("source_samples", 0) or 0),
            "expected_samples": None,
            "coverage_requirement": "scenario-specific; not the 1800-second replay gate",
            "process_p99_ms": _number(metrics.get("process_p99_ms")),
            "process_max_ms": _number(metrics.get("process_max_ms")),
            "runtime_fault": metrics.get("runtime_fault"),
            "physical_pulse_claims": False,
            "accepted_generated_command_samples": int(case.get("accepted_generated_command_samples",
                                                               case.get("generated_nonzero_ao_command_samples", 0)) or 0),
            "pulse_commit_samples": int(case.get("pulse_commit_samples",
                                                 metrics.get("pulse_commit_samples", 0)) or 0),
            "requested_pulse_events": int(requested or 0),
            "rendered_pulse_events": int(rendered or 0),
            "rejected_pulse_events": int(rejected or 0),
            "unclassified_pulse_events": int(unclassified or 0),
            "pulse_request_ids": list(metrics.get("pulse_request_ids", [])),
            "pulse_rejected_ids": list(metrics.get("pulse_rejected_ids", [])),
            "simulated_effect_pulse_count": int(case.get(
                "plant_effect_pulse_count",
                stimulated.get("plant_effect_pulse_count", 0) if isinstance(stimulated, dict) else 0) or 0),
            "rate_capacity_reaches_disturbance": capacity_reaches,
            "effective_rate_mean_capacity_mv": effective_capacity,
            "measured_generated_mean_capacity_mv": generated_capacity,
            "measured_generated_rate_reaches_target": generated_reaches_target,
            "capacity_reference_kind": capacity_info.get("reference_kind"),
            "capacity_reference_settings": {
                key: reference_settings.get(key) for key in ("kp", "ki", "kd", "target_z", "max_rate_hz")
            } if isinstance(reference_settings, dict) else {},
            "regulated_controller_settings": {
                key: settings.get(key) for key in ("kp", "ki", "kd", "target_z", "max_rate_hz")
            } if isinstance(settings, dict) else {},
            "capacity_reference_target_z": _number(capacity_info.get("target_z_for_capacity")),
            "capacity_reference_window_start_s": _number(capacity_info.get("steady_window_start_s")),
            "capacity_reference_window_stop_s": _number(capacity_info.get("steady_window_stop_s")),
            "capacity_reference_window_complete": capacity_info.get("steady_window_complete"),
            "capacity_reference_window_expected_samples": capacity_info.get("steady_window_expected_samples"),
            "capacity_reference_window_observed_samples": capacity_info.get("steady_window_observed_samples"),
            "capacity_reference_reaches_target": generated_reaches_target,
            "capacity_reference_target_correction_mv": reference_correction,
            "capacity_reference_generated_pulses": capacity_info.get("generated_effect_pulses_in_steady_window"),
            "capacity_reference_generated_rate_hz": _number(capacity_info.get("generated_effect_rate_hz_in_steady_window")),
            "regulated_controller_usage": {
                key: regulated_usage.get(key) for key in (
                    "controller_target_z", "generated_effect_pulse_count_in_disturbance",
                    "generated_effect_rate_hz_in_disturbance", "formula_mean_reduction_at_observed_rate_mv",
                    "target_correction_required_mv_for_capacity_reference_z",
                )
            } if isinstance(regulated_usage, dict) else {},
            "capacity_reference_source_coverage_fraction": _number(capacity_info.get("source_coverage_fraction")),
            "capacity_reference_source_gap_samples": capacity_info.get("source_gap_samples"),
            "capacity_reference_skipped_source_samples": capacity_info.get("skipped_source_samples"),
            "requested_rate_limit_hz": _number(capacity_info.get("requested_rate_limit_hz")),
            "effective_rate_bound_hz": _number(capacity_info.get("effective_rate_bound_hz")),
            "measured_generated_rate_hz": _number(capacity_info.get("measured_generated_rate_hz")),
            "target_correction_required_mv": _number(capacity_info.get("target_correction_required_mv")),
            "measured_generated_target_margin_mv": _number(capacity_info.get("measured_generated_target_margin_mv")),
            "ao_zero_substitutions": int(metrics.get("ao_zero_substitutions", 0) or 0),
            "ao_active_demand_misses": int(metrics.get("ao_active_demand_misses", 0) or 0),
            "ao_late_commands": int(metrics.get("ao_late_commands", 0) or 0),
            "ao_rejected_stale": int(metrics.get("ao_rejected_stale", 0) or 0),
            "ao_drain_complete": bool(metrics.get("ao_drain_complete", False)),
            "ao_final_queue_depth": int(metrics.get("ao_final_queue_depth", 0) or 0),
            "disturbance_amplitude_mv": disturbance,
            "max_amplitude_reduction_mv": max_reduction,
            "amplitude_reduction_per_pulse_mv": per_pulse,
            "max_rate_hz": max_rate,
            "effect_recovery_s": recovery_s,
            "estimated_sustainable_mean_reduction_mv": capacity,
            "estimated_capacity_reaches_disturbance": capacity_reaches,
            "capacity_basis": capacity_basis,
            "plant_capacity_margin_mv": capacity_margin,
            "maximum_steady_post_pulse_reduction_mv": post_pulse_capacity,
            "tracking_metrics": {
                key: tracking.get(key) for key in (
                    "target_z", "sample_count", "expected_sample_count", "coverage_fraction",
                    "window_complete", "iae_z_seconds", "window_mean_abs_error_z",
                    "initial_mean_error_z", "final_mean_error_z", "mean_rate_hz", "rate_std_hz",
                    "saturation_fraction", "settling_time_seconds", "rate_direction_reversals",
                    "rate_within_bounds", "settling_band_z", "settling_window_seconds",
                    "final_window_mean_error_z", "final_window_raw_mae_z",
                    "final_window_moving_mean_error_z", "final_window_moving_mean_abs_error_z",
                    "moving_mean_error_max_abs_z", "raw_error_std_z", "raw_error_std_limit_z",
                    "raw_ripple_within_bounds", "settled_within_band", "tracking_response_gate_passed",
                    "final_window_moving_mean_abs_error_z",
                    "post_settling_raw_error_std_z",
                    "source_coverage_fraction", "source_window_coverage_fraction", "full_source_coverage",
                    "source_gap_samples", "skipped_source_samples", "expected_eligible_blocks",
                    "eligible_observation_blocks", "eligible_observation_coverage_fraction",
                    "known_feature_invalid_blocks", "known_feature_invalid_samples",
                    "stimulated_mask_samples", "feature_guard_samples", "masked_coverage_fraction",
                    "unexplained_feature_invalid_samples", "baseline_std_mv",
                    "generated_effect_pulse_count_in_disturbance", "measured_generated_effect_rate_hz",
                )
            },
            "tracking_gate_passed": case.get("tracking_gate_passed",
                                               stimulated.get("tracking_gate_passed") if isinstance(stimulated, dict) else None),
            "plant_recovery_metrics": recovery if isinstance(recovery, dict) else {},
            "plant_recovery_passed": case.get(
                "plant_recovery_passed",
                recovery.get("bounded_recovery") if isinstance(recovery, dict) else None),
            "tracking_iae_improved": case.get("tracking_iae_improved"),
            "simulation_acceptance_passed": case.get(
                "simulation_acceptance_passed",
                stimulated.get("simulation_acceptance_passed") if isinstance(stimulated, dict) else None),
            "controller_tracking_trace_summary": _trace_summary(case.get(
                "controller_tracking_trace",
                stimulated.get("controller_tracking_trace", []) if isinstance(stimulated, dict) else [])),
            "observed_synthetic_disturbance_change_mV": activity_change,
            "failures": sim_failures,
            "note": "Simulated or software-scheduled commands only. This does not report physical monitor measurements or biological efficacy.",
        })
    return records


def _parse_results(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    benchmark_cases: list[dict[str, Any]] = []
    simulation_cases: list[dict[str, Any]] = []
    report_errors: list[dict[str, Any]] = []
    for path in paths:
        doc = _load(path)
        schema = str(doc.get("schema", ""))
        if schema == "echo-chamber-controller-benchmark-v1":
            acceptance = {str(item.get("case_id")): item for item in doc.get("acceptance", [])}
            failed = {str(item.get("case_id")): item for item in doc.get("case_failures", [])}
            environments = doc.get("environments", [])
            for case_index, item in enumerate(doc.get("cases", [])):
                case_id = str(item.get("case_id", "unknown"))
                acceptance_entry = acceptance.get(case_id, {})
                listed = list(acceptance_entry.get("failures", []))
                if acceptance_entry and not acceptance_entry.get("passed", False) and not listed:
                    listed.append("validation report marked this case as acceptance-failed")
                exception = failed.get(case_id)
                if exception:
                    listed = list(listed) + [f"{exception.get('exception_type', 'Exception')}: {exception.get('message', 'case failed')}"]
                if item.get("ao_clock_mode") == "independent_monotonic_wall":
                    kind = "paced"
                elif "lead_sweep" in path.stem:
                    kind = "lead_sweep"
                elif "smoke" in path.stem:
                    kind = "smoke"
                else:
                    kind = "matrix"
                provenance = environments[case_index] if case_index < len(environments) and isinstance(environments[case_index], dict) else {}
                benchmark_cases.append(_case_record(item, path, kind, list(listed), provenance))
            for item in doc.get("case_failures", []):
                if not any(case.get("case_id") == item.get("case_id") for case in doc.get("cases", [])):
                    report_errors.append({"result_file": str(path), "case_id": item.get("case_id"),
                                          "run_kind": _infer_run_kind(path),
                                          "status": "FAIL", "failure": item})
        elif (schema in {"echo-chamber-responsive-simulation-matrix-v1", "echo-chamber-responsive-simulation-v1"}
              or ("metrics" in doc and "stimulation_enabled" in doc and "plant" in doc)):
            simulation_cases.extend(_simulation_records(doc, path))
        elif "throughput" in doc and "paced" in doc:
            benchmark_cases.extend(_legacy_short_records(doc, path))
        else:
            report_errors.append({"result_file": str(path), "status": "UNRECOGNIZED",
                                  "failure": f"unsupported report schema: {schema or 'missing'}"})
    return benchmark_cases, simulation_cases, report_errors


def _infer_run_kind(path: Path) -> str:
    name = path.stem.casefold()
    if "paced" in name:
        return "paced"
    if "lead_sweep" in name or "leadsweep" in name:
        return "lead_sweep"
    if "smoke" in name:
        return "smoke"
    return "matrix"


def _required_group_status(group: str, cases: list[dict[str, Any]],
                           errors: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [case for case in cases if case["run_kind"] == group and case.get("ao_lead_ms") == 100.0]
    selected_errors = [item for item in errors if item.get("run_kind") == group]
    if not selected and not selected_errors:
        return {"status": "NOT REPORTED", "case_count": 0, "failures": []}
    failures = []
    for item in selected_errors:
        failures.append(f"{item.get('case_id', item.get('result_file'))}: {item.get('failure')}")
    for case in selected:
        if case.get("status") == "FAIL":
            failures.extend(f"{case['case_id']}: {message}" for message in case.get("failures", []))
    if failures:
        return {"status": "FAIL", "case_count": len(selected), "failures": failures}
    if any(not case.get("full_1800s_coverage") or case.get("status") != "PASS" for case in selected):
        return {"status": "INCOMPLETE", "case_count": len(selected),
                "failures": ["required full-duration case coverage or acceptance criteria are incomplete"]}
    if group == "matrix":
        keys = {(case["source_name"], case["controller"]) for case in selected}
        complete = (len(selected) == 9 and len({case["source_name"] for case in selected}) == 3
                    and keys == {(source, controller) for source in {case["source_name"] for case in selected}
                                 for controller in EXPECTED_CONTROLLERS})
    else:
        complete = len(selected) == 3 and {case["controller"] for case in selected} == EXPECTED_CONTROLLERS
    if not complete:
        return {"status": "INCOMPLETE", "case_count": len(selected),
                "failures": ["required matrix is missing one or more source/controller cases"]
                if group == "matrix" else ["paced set is missing one or more controllers"]}
    return {"status": "PASS", "case_count": len(selected), "failures": []}


def _required_run_groups(benchmark_cases: list[dict[str, Any]],
                         report_errors: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {group: _required_group_status(group, benchmark_cases, report_errors)
            for group in ("matrix", "paced")}


def _diagnostic_status(benchmark_cases: list[dict[str, Any]], simulation_cases: list[dict[str, Any]],
                       report_errors: list[dict[str, Any]]) -> str:
    diagnostic_cases = [case for case in benchmark_cases if case["run_kind"] in {"smoke", "lead_sweep"}]
    errors = [item for item in report_errors if item.get("run_kind") in {"smoke", "lead_sweep"}]
    if any(case["status"] == "FAIL" for case in diagnostic_cases) or errors:
        return "FAILURES REPORTED"
    if any(case.get("failures") for case in simulation_cases):
        return "FAILURES REPORTED"
    if diagnostic_cases or simulation_cases:
        return "NO DIAGNOSTIC FAILURES"
    return "NOT REPORTED"


def _overall_status(groups: dict[str, dict[str, Any]], report_errors: list[dict[str, Any]]) -> str:
    if any(item.get("status") == "UNRECOGNIZED" or "INPUT" in str(item.get("status", ""))
           or "IMPORT VERIFICATION FAILED" in str(item.get("status", "")) for item in report_errors):
        return "EVIDENCE INPUT FAILURE"
    values = {name: item["status"] for name, item in groups.items()}
    if "FAIL" in values.values():
        return "REQUIRED FULL RUNS FAILED"
    matrix = values["matrix"]
    paced = values["paced"]
    if matrix == "PASS" and paced == "PASS":
        return "FULL MATRIX AND PACED RUNS PASS"
    if matrix == "PASS":
        return "FULL 9-CASE MATRIX PASS; PACED RUN NOT COMPLETE"
    if paced == "PASS":
        return "THREE-CONTROLLER PACED RUN PASS; MATRIX NOT COMPLETE"
    if "INCOMPLETE" in values.values():
        return "REQUIRED FULL RUNS INCOMPLETE"
    return "SHORT OR PARTIAL — NOT ACCEPTANCE"


def _plot(records: list[dict[str, Any]], path: Path) -> None:
    plotted = [x for x in records if _number(x.get("process_p99_ms")) is not None
               or _number(x.get("process_max_ms")) is not None]
    fig, ax = plt.subplots(figsize=(max(8, min(18, 0.62 * len(plotted) + 4)), 5.5), constrained_layout=True)
    if plotted:
        x = list(range(len(plotted)))
        p99 = [_number(row.get("process_p99_ms")) for row in plotted]
        maximum = [_number(row.get("process_max_ms")) for row in plotted]
        ax.scatter(x, p99, label="Process p99", marker="o", color="#2368a0", s=45)
        ax.scatter(x, maximum, label="Process maximum", marker="x", color="#bd4b36", s=48)
        labels = []
        for row in plotted:
            source = str(row.get("source_name", "?"))
            lead = row.get("ao_lead_ms")
            lead_label = f"{lead:g} ms" if isinstance(lead, (int, float)) else ""
            labels.append(f"{row.get('controller','?')} · {row.get('status','?')}\n{source[:15]} {lead_label}".strip())
        ax.set_xticks(x, labels, rotation=40, ha="right", fontsize=8)
        ax.axhline(MAX_P99_MS, color="#2368a0", linestyle="--", linewidth=1, label="p99 limit 2.5 ms")
        ax.axhline(MAX_PROCESS_MS, color="#bd4b36", linestyle=":", linewidth=1, label="block limit 5 ms")
        ax.set_ylabel("Processing time (ms)")
        ax.legend(loc="upper left", ncols=2, fontsize=8)
        ax.set_title("Controller timing evidence · duration gates are reported separately")
    else:
        ax.text(0.5, 0.5, "No process timing values in supplied reports", ha="center", va="center")
        ax.set_axis_off()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _format_timing_ms(value: Any) -> str:
    number = _number(value)
    return f"{number:.3f}" if number is not None else "n/a"


def _markdown(summary: dict[str, Any]) -> str:
    lines = ["# Controller validation evidence", "",
             f"**Overall status:** {summary['overall_status']}", "",
             f"**Short diagnostic status:** {summary['diagnostic_status']}", "",
             "Physical hardware and monitor behavior remain **unverified**. Software AO scheduling and synthetic-plant commands are reported separately and do not establish physical delivery or biological effect.", "",
             f"Coverage reference: {EXPECTED_SECONDS:.0f} seconds, {EXPECTED_BLOCKS:,} blocks, {EXPECTED_SAMPLES:,} samples per full recording.", "",
             "Per-block process timing covers the controller, renderer, safety adapter, recorder submission, and UI submission. Asynchronous health checks remain separate. Unpaced recorder backpressure waits occur between blocks and are excluded from that timing. Browser connection and packet consumption do not prove that a person saw rendered frames.", "",
             "## Required full-duration runs", "",
             f"- Nine-case unpaced matrix: **{summary['required_run_groups']['matrix']['status']}** ({summary['required_run_groups']['matrix']['case_count']}/9 cases).",
             f"- Three-controller paced runs: **{summary['required_run_groups']['paced']['status']}** ({summary['required_run_groups']['paced']['case_count']}/3 controllers).", ""]
    required_cases = [case for case in summary["benchmark_cases"] if case["run_kind"] in {"matrix", "paced"}]
    if required_cases:
        lines.extend(["| Status | Run | Source | Controller | Blocks | Samples | p99 ms | Max ms | Gaps | Skipped | AO misses, late, stale, zero | Recorder | Browser lifecycle |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|"])
        for case in required_cases:
            p99 = case.get("process_p99_ms")
            maximum = case.get("process_max_ms")
            ao = ", ".join(str(case.get(key) if case.get(key) is not None else "n/a") for key in
                            ("ao_active_demand_misses", "ao_late_commands", "ao_rejected_stale", "ao_zero_substitutions"))
            if case.get("browser_required"):
                browser = (f"connected={case.get('browser_connected')}; finish={case.get('browser_connected_at_finish')}; "
                           f"disconnects={case.get('browser_disconnect_count')}; duration_s={case.get('browser_disconnected_duration_s')}; "
                           f"open={case.get('browser_disconnect_open_at_finish')}")
            else:
                browser = "not required"
            recorder = (f"complete={case.get('recording_completed')}; "
                        f"committed={case.get('recording_committed_samples'):,}/"
                        f"source={case.get('source_samples'):,}/processed={case.get('recording_expected_samples_from_processed_blocks'):,}"
                        if case.get("recording_committed_samples") is not None
                        else "not reported")
            lines.append(f"| {case['status']} | {case['run_kind']} @ {case.get('ao_lead_ms','n/a')} ms | {case['source_name']} | {case['controller']} | {case['processed_blocks']:,} / {case['expected_blocks']:,} | {case['source_samples'] if case['source_samples'] is not None else 'n/a'} / {case['expected_samples']:,} | {_format_timing_ms(p99)} | {_format_timing_ms(maximum)} | {case.get('source_index_gap_samples') if case.get('source_index_gap_samples') is not None else 'n/a'} | {case.get('skipped_source_samples') if case.get('skipped_source_samples') is not None else 'n/a'} | {ao} | {recorder} | {browser} |")
    else:
        lines.append("No required 1800-second matrix or paced cases were supplied.")
    diagnostic_cases = [case for case in summary["benchmark_cases"] if case["run_kind"] in {"smoke", "lead_sweep"}]
    lines.extend(["", "## Short timing diagnostics", ""])
    if diagnostic_cases:
        lines.append("These smoke and lead-sweep runs show timing behavior only. Their duration does not count toward the 1800-second matrix or paced gates.")
        lines.extend(["", "| Status | Run | Source | Controller | Lead ms | Blocks | p99 ms | Max ms |", "|---|---|---|---|---:|---:|---:|---:|"])
        for case in diagnostic_cases:
            lines.append(f"| {case['status']} | {case['run_kind']} | {case['source_name']} | {case['controller']} | {case.get('ao_lead_ms','n/a')} | {case['processed_blocks']:,} | {_format_timing_ms(case.get('process_p99_ms'))} | {_format_timing_ms(case.get('process_max_ms'))} |")
    else:
        lines.append("No smoke or lead-sweep diagnostics were supplied.")
    lines.extend(["", "## Failures and limits", ""])
    any_failures = False
    for case in summary["benchmark_cases"]:
        if case.get("failures"):
            any_failures = True
            lines.append(f"- **{case['case_id']} ({case['status']}):** " + "; ".join(case["failures"]))
    for case in summary["simulation_cases"]:
        if case.get("failures"):
            any_failures = True
            lines.append(f"- **{case['case_id']} (SIMULATION):** " + "; ".join(case["failures"]))
    for item in summary["report_errors"]:
        any_failures = True
        lines.append(f"- **{item.get('case_id', item.get('result_file'))} ({item['status']}):** {item['failure']}")
    if not any_failures:
        lines.append("No measured case failures were reported. Short runs remain non-acceptance evidence.")
    lines.extend(["", "## Synthetic simulations", ""])
    if summary["simulation_cases"]:
        lines.append("Synthetic simulation results describe software-accepted commands and modeled plant activity only. Their run duration is separate from the required 30-minute replay coverage.")
        lines.append("")
        lines.append("| Controller | Duration s | Blocks | Samples | Simulated accepted command samples | Modeled pulses | Proposals req/render/reject/unknown | AO zero/miss/late/stale/drained | Activity change mV | Tracking | Recovery | IAE improved | Simulation acceptance |")
        lines.append("|---|---:|---:|---:|---:|---:|---|---|---:|---|---|---|---|")
        for case in summary["simulation_cases"]:
            change = case.get("observed_synthetic_disturbance_change_mV")
            change_text = f"{change:.4f}" if isinstance(change, (int, float)) else "not compared"
            reachability = case.get("capacity_reference_reaches_target", case.get("rate_capacity_reaches_disturbance"))
            reachability_text = "reachable" if reachability is True else "unreachable" if reachability is False else "not reported"
            pulse_partition = f"{case.get('requested_pulse_events',0)}/{case.get('rendered_pulse_events',0)}/{case.get('rejected_pulse_events',0)}/{case.get('unclassified_pulse_events',0)}"
            ao_summary = (f"{case.get('ao_zero_substitutions')}/{case.get('ao_active_demand_misses')}/"
                          f"{case.get('ao_late_commands')}/{case.get('ao_rejected_stale')}/"
                          f"{case.get('ao_drain_complete') and case.get('ao_final_queue_depth') == 0}")
            lines.append(f"| {case['controller']} | {case.get('duration_seconds','n/a')} | {case.get('processed_blocks',0):,} | {case.get('source_samples',0):,} | {case.get('accepted_generated_command_samples',0):,} | {case.get('simulated_effect_pulse_count',0)} | {pulse_partition} | {ao_summary} | {change_text} | {case.get('tracking_gate_passed') if case.get('tracking_gate_passed') is not None else 'not reported'} | {case.get('plant_recovery_passed') if case.get('plant_recovery_passed') is not None else 'not reported'} | {case.get('tracking_iae_improved') if case.get('tracking_iae_improved') is not None else 'not reported'} | {case.get('simulation_acceptance_passed') if case.get('simulation_acceptance_passed') is not None else 'not reported'} |")
        lines.extend(["", "### Max-request capacity reference and regulated PID usage", "",
                      "| Controller | Reference kind and PID settings | Reference window / observed samples | Measured reduction / margin / correction mV | Reference pulses / rate Hz | Reference reaches target | Regulated PID settings | Regulated rate / formula mean reduction mV |",
                      "|---|---|---|---:|---:|---|---|---|"])
        for case in summary["simulation_cases"]:
            reference = case.get("capacity_reference_settings", {})
            regulated = case.get("regulated_controller_settings", {})
            usage = case.get("regulated_controller_usage", {})
            reference_pid = (f"{case.get('capacity_reference_kind') or 'not reported'}; "
                             f"Kp={reference.get('kp')}, Ki={reference.get('ki')}, target={reference.get('target_z')}")
            reference_window = (f"{case.get('capacity_reference_window_start_s')}–{case.get('capacity_reference_window_stop_s')} s / "
                                f"{case.get('capacity_reference_window_observed_samples')}/"
                                f"{case.get('capacity_reference_window_expected_samples')}")
            reduction_values = (case.get("measured_generated_mean_capacity_mv"),
                                case.get("plant_capacity_margin_mv"),
                                case.get("capacity_reference_target_correction_mv"))
            reduction_text = " / ".join(f"{value:.6f}" if isinstance(value, (int, float)) else "n/a"
                                        for value in reduction_values)
            reference_rate = case.get("capacity_reference_generated_rate_hz")
            reference_usage = (f"{case.get('capacity_reference_generated_pulses')} / "
                               f"{reference_rate:g}" if isinstance(reference_rate, (int, float)) else
                               f"{case.get('capacity_reference_generated_pulses')} / n/a")
            regulated_pid = f"Kp={regulated.get('kp')}, Ki={regulated.get('ki')}, target={regulated.get('target_z')}"
            regulated_rate = usage.get("generated_effect_rate_hz_in_disturbance")
            regulated_mean = usage.get("formula_mean_reduction_at_observed_rate_mv")
            regulated_usage = (f"{regulated_rate:g} / {regulated_mean:.6f}"
                               if isinstance(regulated_rate, (int, float)) and isinstance(regulated_mean, (int, float))
                               else f"{regulated_rate if regulated_rate is not None else 'n/a'} / "
                                    f"{regulated_mean if regulated_mean is not None else 'n/a'}")
            reachability = case.get("capacity_reference_reaches_target")
            reachability_text = "yes" if reachability is True else "no" if reachability is False else "not reported"
            lines.append(f"| {case['controller']} | {reference_pid} | {reference_window} | {reduction_text} | {reference_usage} | {reachability_text} | {regulated_pid} | {regulated_usage} |")
        tracking_cases = [case for case in summary["simulation_cases"]
                          if case.get("tracking_metrics", {}).get("target_z") is not None]
        if tracking_cases:
            lines.extend(["", "### PID tracking windows", "",
                          "| Controller | Target z | Observed z range | Requested rate range Hz | Window | Coverage | IAE z·s | Final moving-mean error z | Final moving-mean absolute error z | Raw MAE z | Raw ripple SD z | Post-settling raw SD z | Settling s | Response gate | Recovery gate | Pair acceptance |",
                          "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"])
            for case in tracking_cases:
                tracking = case["tracking_metrics"]
                trace = case.get("controller_tracking_trace_summary", {})
                z_range = (f"{trace['observed_z_min']:.3g}–{trace['observed_z_max']:.3g}"
                           if trace.get("observed_z_min") is not None and trace.get("observed_z_max") is not None else "n/a")
                rate_range = (f"{trace['requested_rate_min_hz']:.3g}–{trace['requested_rate_max_hz']:.3g}"
                              if trace.get("requested_rate_min_hz") is not None and trace.get("requested_rate_max_hz") is not None else "n/a")
                window = "complete" if tracking.get("window_complete") else "incomplete"
                lines.append(f"| {case['controller']} | {tracking.get('target_z') if tracking.get('target_z') is not None else 'n/a'} | {z_range} | {rate_range} | {window} | {tracking.get('coverage_fraction') if tracking.get('coverage_fraction') is not None else 'n/a'} | {tracking.get('iae_z_seconds') if tracking.get('iae_z_seconds') is not None else 'n/a'} | {tracking.get('final_window_moving_mean_error_z') if tracking.get('final_window_moving_mean_error_z') is not None else 'n/a'} | {tracking.get('final_window_moving_mean_abs_error_z') if tracking.get('final_window_moving_mean_abs_error_z') is not None else 'n/a'} | {tracking.get('final_window_raw_mae_z') if tracking.get('final_window_raw_mae_z') is not None else 'n/a'} | {tracking.get('raw_error_std_z') if tracking.get('raw_error_std_z') is not None else 'n/a'} | {tracking.get('post_settling_raw_error_std_z') if tracking.get('post_settling_raw_error_std_z') is not None else 'n/a'} | {tracking.get('settling_time_seconds') if tracking.get('settling_time_seconds') is not None else 'not reached'} | {case.get('tracking_gate_passed') if case.get('tracking_gate_passed') is not None else 'not reported'} | {case.get('plant_recovery_passed') if case.get('plant_recovery_passed') is not None else 'not reported'} | {case.get('simulation_acceptance_passed') if case.get('simulation_acceptance_passed') is not None else 'not reported'} |")
            lines.extend(["", "### Source and eligible-window coverage", "",
                          "| Controller | Full source coverage | Source window coverage | Eligible blocks observed/expected | Eligible coverage | Known invalid blocks/samples | Stimulated mask samples | Feature guard samples | Masked coverage | Unexplained invalid samples | Source gaps/skips |",
                          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
            for case in tracking_cases:
                tracking = case["tracking_metrics"]
                source_coverage = tracking.get("source_coverage_fraction")
                if source_coverage is None:
                    source_coverage = tracking.get("full_source_coverage")
                observed = tracking.get("eligible_observation_blocks")
                expected = tracking.get("expected_eligible_blocks")
                eligible_pair = f"{observed if observed is not None else 'n/a'}/{expected if expected is not None else 'n/a'}"
                invalid_pair = f"{tracking.get('known_feature_invalid_blocks') if tracking.get('known_feature_invalid_blocks') is not None else 'n/a'}/{tracking.get('known_feature_invalid_samples') if tracking.get('known_feature_invalid_samples') is not None else 'n/a'}"
                gaps_skips = f"{tracking.get('source_gap_samples') if tracking.get('source_gap_samples') is not None else 'n/a'}/{tracking.get('skipped_source_samples') if tracking.get('skipped_source_samples') is not None else 'n/a'}"
                lines.append(f"| {case['controller']} | {source_coverage if source_coverage is not None else 'not reported'} | {tracking.get('source_window_coverage_fraction') if tracking.get('source_window_coverage_fraction') is not None else 'n/a'} | {eligible_pair} | {tracking.get('eligible_observation_coverage_fraction') if tracking.get('eligible_observation_coverage_fraction') is not None else 'n/a'} | {invalid_pair} | {tracking.get('stimulated_mask_samples') if tracking.get('stimulated_mask_samples') is not None else 'n/a'} | {tracking.get('feature_guard_samples') if tracking.get('feature_guard_samples') is not None else 'n/a'} | {tracking.get('masked_coverage_fraction') if tracking.get('masked_coverage_fraction') is not None else 'n/a'} | {tracking.get('unexplained_feature_invalid_samples') if tracking.get('unexplained_feature_invalid_samples') is not None else 'n/a'} | {gaps_skips} |")
        lines.append("")
        lines.append("The max-request reference uses the actual processing core with its reference PID settings; it is separate from regulated PID output usage. Reference capacity and a passing capacity target do not imply that the regulated controller reached its target. Tracking, recovery, IAE improvement, and pair acceptance are reported separately. AO zero substitutions are shown as measured; active misses, late commands, stale rejections, and the drain state are separate checks. Simulation block/sample totals are against that scenario's requested duration, not the 30-minute replay requirement.")
    else:
        lines.append("No simulation reports were supplied.")
    lines.extend(["", "## Import verification", ""])
    source = summary.get("import_verification")
    if source:
        lines.append(f"ABF import verification status: **{source.get('status', 'unknown')}** across {len(source.get('results', []))} source files.")
        lines.append("See the JSON summary for source and canonical replay hashes.")
    else:
        lines.append("No import verification report was supplied.")
    lines.extend(["", "## Timing plot", "", "![Controller timing evidence](controller_timing.png)", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, action="append", required=True,
                        help="validation JSON result; repeat to combine matrix, paced, lead, and simulation reports")
    parser.add_argument("--import-verification", type=Path,
                        help="optional output from tools/verify_replay_import.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in args.results:
        if not path.is_file():
            parser.error(f"result JSON does not exist: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_cases, simulation_cases, report_errors = _parse_results(args.results)
    import_verification = _load(args.import_verification) if args.import_verification else None
    if import_verification is not None:
        import_results = import_verification.get("results", [])
        invalid_imports = [
            result.get("source", "unknown") for result in import_results
            if not (result.get("values_exactly_equal_to_pyabf") and result.get("finite_eof")
                    and result.get("gap_count") == 0 and result.get("channel_regions") == ["CA3", "EC"]
                    and result.get("unit") == "mV")
        ]
        if import_verification.get("status") != "verified" or len(import_results) != 3 or invalid_imports:
            report_errors.append({"status": "IMPORT VERIFICATION FAILED", "result_file": str(args.import_verification),
                                  "failure": {"status": import_verification.get("status"),
                                              "result_count": len(import_results),
                                              "invalid_sources": invalid_imports}})
    summary = {
        "schema": "echo-chamber-controller-evidence-summary-v1",
        "required_run_groups": _required_run_groups(benchmark_cases, report_errors),
        "overall_status": None,
        "diagnostic_status": _diagnostic_status(benchmark_cases, simulation_cases, report_errors),
        "coverage_reference": {"duration_seconds": EXPECTED_SECONDS, "sample_rate_hz": EXPECTED_FS,
                               "block_samples": EXPECTED_BLOCK_SAMPLES, "blocks": EXPECTED_BLOCKS,
                               "samples_per_channel": EXPECTED_SAMPLES},
        "timing_limits_ms": {"process_p99_exclusive": MAX_P99_MS, "maximum_process_exclusive": MAX_PROCESS_MS},
        "hardware_status": "unverified",
        "physical_monitor_claims": False,
        "browser_rendering_verified": False,
        "unpaced_recorder_backpressure_wait_is_excluded_from_per_block_process_timing": True,
        "input_results": [str(p) for p in args.results],
        "benchmark_cases": benchmark_cases,
        "simulation_cases": simulation_cases,
        "report_errors": report_errors,
        "import_verification": import_verification,
        "limitations": [
            "Short or failed runs are not full-duration acceptance passes.",
            "Software-scheduled or synthetic-plant commands are not physical monitor measurements.",
            "Hardware loopback, four-hour soak, and biological efficacy remain unverified.",
            "Unpaced recorder backpressure waits occur between processing blocks and are not added to per-block process timing.",
        ],
    }
    summary["overall_status"] = _overall_status(summary["required_run_groups"], report_errors)
    png = args.output_dir / "controller_timing.png"
    _plot(benchmark_cases + simulation_cases, png)
    json_path = args.output_dir / "controller_validation_summary.json"
    markdown_path = args.output_dir / "controller_validation_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({"overall_status": summary["overall_status"], "cases": len(benchmark_cases),
                      "simulations": len(simulation_cases), "markdown": str(markdown_path),
                      "json": str(json_path), "plot": str(png)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
