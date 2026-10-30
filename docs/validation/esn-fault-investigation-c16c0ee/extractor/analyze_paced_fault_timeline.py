#!/usr/bin/env python3
"""Extract a bounded, provenance-preserving timeline around paced AO faults.

Reads compact reports/snapshots/progress logs and H5 metadata, diagnostics,
discontinuity rows, and sparse AO action stages. It does not read AI/electrode
sample payloads or hash full recordings.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py

BLOCK = 100


def text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\0")
    return str(value)


def source_row(dataset, sample_index):
    if sample_index is None or sample_index % BLOCK:
        return None
    row_index = sample_index // BLOCK
    if row_index >= len(dataset):
        return None
    row = dataset[row_index]
    return {name: _scalar(row[name]) for name in dataset.dtype.names or ()}


def _scalar(value):
    if hasattr(value, "item"):
        value = value.item()
    return text(value) if isinstance(value, bytes) else value


def inspect_h5(path: Path, source_index: int, target_index: int, window: int):
    with h5py.File(path, "r") as h5:
        metadata = json.loads(text(h5.attrs["metadata_json"]))
        committed = int(h5.attrs.get("committed_samples", -1))
        block_rows = h5["diagnostics/blocks"]
        controller_rows = h5["diagnostics/controller_blocks"]
        action_rows = h5["events/action_stages"]
        block = source_row(block_rows, source_index)
        controller = source_row(controller_rows, source_index)
        previous_block = source_row(block_rows, source_index - BLOCK)
        zero_ranges = []
        stage_counts = Counter()
        start_bound = max(0, target_index - window)
        end_bound = target_index + window
        for offset in range(0, len(action_rows), 4096):
            batch = action_rows[offset:offset + 4096]
            for row in batch:
                stage = text(row["stage"])
                stage_counts[stage] += 1
                if stage != "zero_substituted":
                    continue
                start = int(row["sample_index"])
                end = int(row["range_end_sample_index"])
                if start < end_bound and end > start_bound:
                    zero_ranges.append({
                        "start_sample_index": start,
                        "end_sample_index_exclusive": end,
                        "overlap_start": max(start, start_bound),
                        "overlap_end_exclusive": min(end, end_bound),
                        "generation": int(row["generation"]),
                        "controller": text(row["controller"]),
                        "reason": text(row["reason"]),
                    })
        gaps = h5["events/sample_discontinuities"]
        return {
            "path": str(path.resolve()),
            "complete": int(h5.attrs.get("complete", 0)),
            "committed_samples": committed,
            "block_rows": len(block_rows),
            "controller_rows": len(controller_rows),
            "signal_shapes": {
                name: list(h5[name].shape)
                for name in ("signals/ai_raw_V", "signals/electrode_mV", "signals/ao_command_V")
            },
            "source_discontinuity_rows": len(gaps),
            "metadata_git_revision": metadata.get("git_revision"),
            "metadata_configuration": {
                k: metadata.get("configuration", {}).get(k)
                for k in ("controller", "sample_rate", "chunk_size", "ao_target_lead_ms")
            },
            "metadata_source": {
                k: metadata.get("source", {}).get(k)
                for k in ("path", "sha256", "sample_count", "sample_rate_hz")
            },
            "source_block_at_fault_marker": block,
            "previous_source_block": previous_block,
            "controller_diagnostics_at_fault_marker": controller,
            "action_stage_counts": dict(stage_counts),
            "zero_substituted_ranges_near_target": zero_ranges,
            "action_row_count": len(action_rows),
        }


def inspect_case(report_path: Path, snapshot_path: Path, window: int):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    case = report["cases"][0]
    env = report["environments"][0]
    fault_source = int(snapshot["source_sample_index"])
    lead_samples = int(case["ao_lead_ms"] * case["sample_rate_hz"] / 1000)
    target = fault_source + lead_samples
    progress_path = report_path.parent / "progress.jsonl"
    progress = []
    slow_core_events = []
    if progress_path.is_file():
        with progress_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                event = json.loads(line)
                event_type = event.get("type")
                if event_type == "slow_core_process":
                    slow_core_events.append(event)
                source = event.get("source_sample_index")
                if event_type == "ao_clock_fault" or (
                    source is not None and abs(int(source) - fault_source) <= window
                    and event_type in {"slow_core_process", "first_late_command", "gc_pause"}
                ):
                    progress.append(event)
    component = case.get("component_timings_ms", {})
    ui_max_ms = component.get("ui_submission", {}).get("max")
    core_calls_over_ui_max = [
        {"source_sample_index": row.get("source_sample_index"), "wall_ms": row.get("wall_ms"),
         "progress_sequence": row.get("progress_sequence"),
         "progress_observed_perf_counter_ns": row.get("progress_observed_perf_counter_ns")}
        for row in slow_core_events
        if ui_max_ms is not None and float(row.get("wall_ms", 0.0)) >= float(ui_max_ms)
    ]
    fault_observation_ns = snapshot.get("observation_time_monotonic_ns")
    for row in core_calls_over_ui_max:
        row["slow_event_written_ms_after_fault_observation"] = (
            (row["progress_observed_perf_counter_ns"] - fault_observation_ns) / 1e6
            if fault_observation_ns is not None and row["progress_observed_perf_counter_ns"] is not None
            else None
        )
    return {
        "report": str(report_path.resolve()),
        "snapshot": str(snapshot_path.resolve()),
        "report_revision": report.get("git_revision"),
        "report_schema": report.get("schema"),
        "physical_pulse_claims": report.get("physical_pulse_claims"),
        "controller": case.get("controller"),
        "case_id": case.get("case_id"),
        "source_h5": env.get("source_path"),
        "source_h5_sha256": env.get("source_sha256"),
        "recording_file": env.get("recording_file"),
        "timing_summary": {
            k: case.get(k) for k in (
                "source_samples", "recording_committed_samples", "duration_source_s",
                "process_p99_ms", "process_max_ms", "deadline_exceeded_blocks",
                "ao_active_demand_misses", "ao_late_commands", "ao_generated_sample_count",
                "ao_zero_substitutions", "ao_clock_max_service_gap_ms",
                "ao_clock_gaps_over_block_period", "source_availability_lag_max_ms",
                "source_backlog_blocks_max", "recording_completed", "ao_drain_complete",
                "browser_disconnect_count", "browser_connected_at_finish",
                "progress_events_accepted", "progress_events_written",
                "progress_events_dropped", "progress_events_pending", "runtime_fault",
            )
        },
        "component_timing_aggregate_ms": component,
        "ui_timing_crosscheck": {
            "core_deadline_exceeded_blocks": case.get("deadline_exceeded_blocks"),
            "slow_core_progress_events_written": len(slow_core_events),
            "slow_core_progress_event_cap": 20,
            "slow_core_log_below_cap": len(slow_core_events) < 20,
            "ui_submission_max_ms": ui_max_ms,
            "core_calls_at_or_above_ui_submission_max": core_calls_over_ui_max,
            "component_wrapper_is_nested_in_core_process": True,
            "core_calls_ui_publish_once_per_block_by_source_inspection": True,
            "interpretation": (
                "When the slow-call log is below its cap and includes every deadline miss, "
                "a unique Core event at or above the UI maximum identifies the source block "
                "whose synchronous UI-submission wrapper reached that maximum. This correlates "
                "timing only; it does not identify the line or cause of the delay."
            ),
        },
        "fault_observation": {
            "type": snapshot.get("type"),
            "fault": snapshot.get("fault"),
            "is_fault_onset": snapshot.get("observation_is_fault_onset"),
            "observation_monotonic_ns": snapshot.get("observation_time_monotonic_ns"),
            "fault_marker_source_sample_index": fault_source,
            "latest_completed_core_source_sample_index": snapshot.get("latest_completed_core_source_sample_index"),
            "core_call_started_not_yet_published_source_sample_index": snapshot.get("core_call_started_not_yet_published_source_sample_index"),
            "source_availability_lag_ms_at_observation": snapshot.get("source_availability_lag_ms"),
            "generated_sample_count_at_observation": snapshot.get("generated_sample_count"),
            "source_to_generated_lead_samples": int(snapshot.get("generated_sample_count", 0)) - fault_source,
            "source_to_generated_lead_ms": 1000.0 * (int(snapshot.get("generated_sample_count", 0)) - fault_source) / int(case["sample_rate_hz"]),
            "expected_target_sample_index": target,
            "target_minus_generated_samples": target - int(snapshot.get("generated_sample_count", 0)),
            "target_headroom_ms_at_observation": 1000.0 * (target - int(snapshot.get("generated_sample_count", 0))) / int(case["sample_rate_hz"]),
            "next_write_block_after_scheduler_service": snapshot.get("next_write_block"),
            "post_service_write_frontier_sample_index": int(snapshot.get("next_write_block", 0)) * BLOCK,
            "post_service_frontier_minus_generated_samples": int(snapshot.get("next_write_block", 0)) * BLOCK - int(snapshot.get("generated_sample_count", 0)),
            "ao_service_gap_ms_at_observation": snapshot.get("ao_service_gap_ms"),
            "ao_max_service_gap_ms_seen_by_observation": snapshot.get("ao_max_service_gap_ms_at_observation"),
            "ao_service_gaps_over_5ms_by_observation": snapshot.get("ao_service_gaps_over_block_period_at_observation"),
            "ao_stats": snapshot.get("ao"),
            "snapshot_semantics_note": snapshot.get("snapshot_semantics_note"),
        },
        "progress_context": progress,
        "h5_context": inspect_h5(Path(env["recording_file"]), fault_source, target, window),
        "inference_boundary": (
            "The AO-fault progress record is a post-advance observation, explicitly not fault onset. "
            "Component timings are per-component aggregates with no source-index join. Same-index "
            "slow-core evidence and global maxima can suggest a candidate, but do not identify a "
            "causal component. H5 source/AO ranges establish saved timeline facts, not physical output."
        ),
    }


def summarize_simulation(path: Path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for case in doc.get("cases", []):
        phases = {}
        for phase_name in ("stimulated", "no_stimulation_control"):
            phase = case.get(phase_name, {})
            metrics = phase.get("metrics", {})
            phases[phase_name] = {
                "case_id": phase.get("case_id"),
                "git_revision": phase.get("environment", {}).get("git_revision"),
                "source_checksum": phase.get("environment", {}).get("source_checksum"),
                "duration_seconds": phase.get("duration_seconds"),
                "metrics": {k: metrics.get(k) for k in (
                    "source_samples", "recording_committed_samples", "runtime_fault",
                    "ao_active_demand_misses", "ao_zero_substitutions", "ao_generated_sample_count",
                    "process_p99_ms", "process_max_ms", "deadline_exceeded_blocks",
                    "requested_pulse_events", "pulse_commit_samples", "ao_nonzero_generated_samples",
                )},
            }
        cases.append({"controller": case.get("controller"), **phases})
    return {
        "path": str(path.resolve()),
        "schema": doc.get("schema"),
        "scenario": doc.get("scenario"),
        "physical_pulse_claims": doc.get("physical_pulse_claims"),
        "cases": cases,
        "note": "Simulation phase summaries are not paired to a fault-onset snapshot unless one is separately supplied.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=2, action="append", metavar=("REPORT", "SNAPSHOT"), required=True)
    parser.add_argument("--simulation-report", type=Path, action="append", default=[])
    parser.add_argument("--window-samples", type=int, default=20_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error(f"refusing to overwrite existing output: {out}")
    result = {
        "schema": "paced-ao-fault-timeline-analysis-v1",
        "method": "bounded progress/H5 metadata-diagnostic-action-stage extraction; no raw AI/electrode samples or full-file hashes",
        "window_samples": args.window_samples,
        "paced_fault_cases": [inspect_case(Path(r), Path(s), args.window_samples) for r, s in args.case],
        "simulation_comparisons": [summarize_simulation(p) for p in args.simulation_report],
        "global_conclusion": (
            "Observed faults are safety outcomes from a missing active AO refill. The saved snapshots "
            "are post-advance observations, not atomic fault-onset records. Attribute CPU, UI, recorder, "
            "AO-host, OS, or clock-jump causality only when timestamped per-source evidence supports it."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "paced_cases": len(result["paced_fault_cases"]), "simulation_reports": len(result["simulation_comparisons"])}))


if __name__ == "__main__":
    main()
