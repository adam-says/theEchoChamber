"""Bound PID pulse-rate phase from recorded HDF5 v5 block diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np


MAX_RATE_HZ = 2.0


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.split(b"\0", 1)[0].decode("utf-8", errors="replace")
    return str(value)


def _rate_bounds(stored_value: Any) -> tuple[float, float, float]:
    """Return stored float32 rate and neighboring-float bounds, clipped to 0..2 Hz."""
    stored = np.float32(stored_value)
    if not np.isfinite(stored):
        raise ValueError("controller_blocks contains a non-finite output_rate_Hz")
    lower = np.nextafter(stored, np.float32(-np.inf))
    upper = np.nextafter(stored, np.float32(np.inf))
    bounded_rate = max(0.0, min(MAX_RATE_HZ, float(stored)))
    return (bounded_rate, max(0.0, min(MAX_RATE_HZ, float(lower))),
            max(0.0, min(MAX_RATE_HZ, float(upper))))


def _pulse_request_ids(events: Any) -> set[str]:
    ids: set[str] = set()
    if events.dtype.names is None or "request_id" not in events.dtype.names:
        return ids
    for row in events:
        if "controller" in events.dtype.names and _decode(row["controller"]) != "pid":
            continue
        request_id = _decode(row["request_id"])
        if request_id:
            ids.add(request_id)
    return ids


def analyze_recording(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as recording:
        if _decode(recording.attrs.get("format_tag", "")) != "echoChamber_H5_v5":
            raise ValueError(f"{path}: expected echoChamber_H5_v5 recording")
        sample_rate = float(recording.attrs["sample_rate_hz"])
        committed_samples = int(recording.attrs.get("committed_samples", 0))
        complete = bool(recording.attrs.get("complete", 0))
        diagnostics = recording.get("diagnostics")
        if diagnostics is None or "blocks" not in diagnostics or "controller_blocks" not in diagnostics:
            raise ValueError(f"{path}: missing diagnostics/blocks or diagnostics/controller_blocks")
        blocks = diagnostics["blocks"][:]
        controller_blocks = diagnostics["controller_blocks"][:]
        event_dataset = recording.get("events/action_stages")
        event_rows = event_dataset[:] if event_dataset is not None else np.empty(0)

    if not blocks.size or not controller_blocks.size:
        raise ValueError(f"{path}: recording has no complete diagnostic blocks")
    if len(blocks) != len(controller_blocks):
        raise ValueError(f"{path}: diagnostics block counts do not match")
    if sample_rate <= 0:
        raise ValueError(f"{path}: invalid sample_rate_hz {sample_rate}")

    block_by_offset = {int(row["sample_offset"]): row for row in blocks}
    if len(block_by_offset) != len(blocks):
        raise ValueError(f"{path}: diagnostics/blocks has duplicate sample offsets")
    records = []
    seen_offsets: set[int] = set()
    for row in controller_blocks:
        offset = int(row["sample_offset"])
        if offset in seen_offsets:
            raise ValueError(f"{path}: diagnostics/controller_blocks has duplicate sample offsets")
        seen_offsets.add(offset)
        block = block_by_offset.get(offset)
        if block is None:
            raise ValueError(f"{path}: no diagnostics/blocks row for offset {offset}")
        if _decode(row["controller_id"]) != "pid":
            raise ValueError(f"{path}: expected PID diagnostics, found {_decode(row['controller_id'])!r}")
        rate, rate_lower, rate_upper = _rate_bounds(row["output_rate_Hz"])
        count = int(block["sample_count"])
        if count <= 0:
            raise ValueError(f"{path}: non-positive sample_count at offset {offset}")
        records.append({
            "offset": offset,
            "first_sample": int(block["ai_sample_index"]),
            "sample_count": count,
            "generation": int(row["config_generation"]),
            "armed": bool(block["armed"]),
            "ready": bool(row["ready"]),
            "rate_hz": rate,
            "rate_lower_hz": rate_lower,
            "rate_upper_hz": rate_upper,
        })
    records.sort(key=lambda row: row["offset"])
    diagnostic_samples = sum(row["sample_count"] for row in records)
    offset_contiguous = all(
        row["offset"] == (0 if index == 0 else
                          records[index - 1]["offset"] + records[index - 1]["sample_count"])
        for index, row in enumerate(records)
    )
    source_contiguous = all(
        row["first_sample"] == (records[index - 1]["first_sample"] + records[index - 1]["sample_count"])
        for index, row in enumerate(records) if index > 0
    )
    coverage_complete = (complete and diagnostic_samples == committed_samples
                         and offset_contiguous and source_contiguous)

    request_ids = _pulse_request_ids(event_rows) if event_rows.size else set()
    positive_rate_blocks = sum(row["rate_hz"] > 0.0 for row in records)
    positive_armed_blocks = sum(row["rate_hz"] > 0.0 and row["armed"] for row in records)
    segments: list[dict[str, Any]] = []
    phase_episodes: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    previous: dict[str, Any] | None = None
    phase_lower = 0.0
    phase_upper = 1.0  # Unknown residual phase at file start until an observed disarm.
    phase_known_from_reset = False
    initial_phase_unknown_demand = False
    coverage_gap_samples = 0
    phase_start_sample: int | None = None

    def close_segment() -> None:
        nonlocal active
        if active is not None:
            active["duration_seconds"] = active["sample_count"] / sample_rate
            segments.append(active)
            active = None

    def close_phase_episode(end_sample: int) -> None:
        nonlocal phase_lower, phase_upper, phase_known_from_reset, phase_start_sample
        phase_episodes.append({
            "start_sample_index": phase_start_sample,
            "end_sample_index_exclusive": end_sample,
            "duration_seconds": ((end_sample - phase_start_sample) / sample_rate
                                 if phase_start_sample is not None else 0.0),
            "phase_lower_cycles": phase_lower,
            "phase_upper_cycles": phase_upper,
            "crossing_count_lower": math.floor(phase_lower),
            "crossing_count_upper": math.floor(phase_upper),
            "initial_phase_known": phase_known_from_reset,
        })
        phase_lower = 0.0
        phase_upper = 0.0
        phase_start_sample = None

    for row in records:
        source_gap = (previous is not None and row["first_sample"] >
                      previous["first_sample"] + previous["sample_count"])
        offset_gap = (previous is not None and row["offset"] >
                      previous["offset"] + previous["sample_count"])
        gap_samples = max(
            0,
            row["first_sample"] - (previous["first_sample"] + previous["sample_count"])
            if source_gap else 0,
            row["offset"] - (previous["offset"] + previous["sample_count"])
            if offset_gap else 0,
        )
        contiguous = previous is None or (not source_gap and not offset_gap)
        phase_active = row["armed"] and row["ready"] and row["rate_hz"] > 0.0
        keep = active is not None and contiguous and phase_active
        if not keep:
            close_segment()

        if gap_samples:
            # Missing samples are uncertain, not a reset. Add the maximum possible rate.
            phase_upper += MAX_RATE_HZ * gap_samples / sample_rate
            coverage_gap_samples += gap_samples

        if not row["armed"]:
            if previous is not None and previous["armed"]:
                close_phase_episode(row["first_sample"])
            # An observed unarmed block is the recorded authoritative reset.
            phase_known_from_reset = True
            phase_lower = phase_upper = 0.0
            previous = row
            continue

        if phase_start_sample is None:
            phase_start_sample = row["first_sample"]
        if not phase_known_from_reset:
            initial_phase_unknown_demand = True
        phase_lower += row["rate_lower_hz"] * row["sample_count"] / sample_rate
        phase_upper += row["rate_upper_hz"] * row["sample_count"] / sample_rate

        if phase_active:
            if active is None:
                active = {
                    "start_sample_index": row["first_sample"],
                    "end_sample_index_exclusive": row["first_sample"],
                    "generation": row["generation"],
                    "block_count": 0,
                    "sample_count": 0,
                    "positive_rate_min_hz": row["rate_hz"],
                    "positive_rate_max_hz": row["rate_hz"],
                }
            active["block_count"] += 1
            active["sample_count"] += row["sample_count"]
            active["end_sample_index_exclusive"] = row["first_sample"] + row["sample_count"]
            active["positive_rate_min_hz"] = min(active["positive_rate_min_hz"], row["rate_hz"])
            active["positive_rate_max_hz"] = max(active["positive_rate_max_hz"], row["rate_hz"])
        previous = row
    close_segment()
    if records and records[-1]["armed"]:
        close_phase_episode(records[-1]["first_sample"] + records[-1]["sample_count"])

    longest = max(segments, key=lambda row: row["sample_count"], default=None)
    largest_phase = max(phase_episodes, key=lambda row: row["phase_upper_cycles"], default=None)
    max_upper = max((row["phase_upper_cycles"] for row in phase_episodes), default=0.0)
    observed_requests = len(request_ids)
    if observed_requests:
        conclusion = "pulse requests observed; phase bounds are descriptive only"
    elif not coverage_complete:
        conclusion = "inconclusive: recording is incomplete or diagnostic/source coverage is not contiguous"
    elif initial_phase_unknown_demand:
        conclusion = "inconclusive: the recording began with armed demand before an observed disarm reset"
    elif coverage_gap_samples:
        conclusion = "inconclusive: source/output gaps prevent an explanatory phase conclusion"
    elif max_upper < 1.0:
        conclusion = ("phase upper bound stays below one crossing in every recorded armed interval; "
                     "this supports why no pulse request was generated")
    else:
        conclusion = ("inconclusive: a recorded interval could reach a crossing; "
                     "unrecorded actuator resets or other constraints may reduce phase")

    return {
        "recording": str(path),
        "format_tag": "echoChamber_H5_v5",
        "recording_complete": complete,
        "sample_rate_hz": sample_rate,
        "committed_samples": committed_samples,
        "diagnostic_samples": diagnostic_samples,
        "diagnostic_sample_count_matches_committed": diagnostic_samples == committed_samples,
        "diagnostic_offsets_contiguous": offset_contiguous,
        "source_samples_contiguous": source_contiguous,
        "coverage_complete_for_explanation": coverage_complete,
        "uncertain_gap_samples": coverage_gap_samples,
        "controller_block_count": len(records),
        "positive_rate_block_count": positive_rate_blocks,
        "positive_armed_rate_block_count": positive_armed_blocks,
        "recorded_armed_positive_intervals": len(segments),
        "phase_episodes_since_recorded_disarm": len(phase_episodes),
        "initial_phase_unknown_demand": initial_phase_unknown_demand,
        "longest_positive_demand_interval": longest,
        "largest_integrated_phase_interval": largest_phase,
        "largest_phase_upper_cycles": max_upper,
        "crossings_with_conservative_uncertainty": {
            "lower": sum(row["crossing_count_lower"] for row in phase_episodes),
            "upper": sum(row["crossing_count_upper"] for row in phase_episodes),
        },
        "observed_unique_scheduler_pulse_request_count": observed_requests,
        "phase_conclusion": conclusion,
        "rate_bound_method": "nextafter neighbors of recorded float32 rate, clipped to [0, 2] Hz; stored zero is not treated as an exact reset",
        "limitations": [
            "The initial residual phase is unknown until a recorded unarmed block establishes a reset.",
            "Source/output gaps carry a maximum-rate upper bound; a gap is never assumed to reset phase.",
            "Stored float32 zero may represent a tiny positive rate, so it is not treated as an exact reset.",
            "Additional live resets, including resets within blocks, remain unrecorded; this is not bitwise recovery of live phase.",
            "This analysis is not a controller pass, bug verdict, or physical pulse measurement.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="+", type=Path, help="completed PID HDF5 v5 recordings")
    args = parser.parse_args()
    results = [analyze_recording(path) for path in args.recordings]
    print(json.dumps({"schema": "echo-chamber-pid-rate-phase-analysis-v1", "results": results},
                     indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
