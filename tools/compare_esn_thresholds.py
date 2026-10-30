"""Replay recorded ESN inputs and compare rolling-threshold decisions.

The recording stores only a float32 mean of each block's model prediction,
not the full prediction stream. This probe replays the recorded electrode_mV
input through the unchanged numeric ESN recurrence once, then feeds the very
same conditioned output to both the new incremental threshold and the
pre-optimization NumPy oracle. It checks event parity on that replay and uses
recorded block means and thresholds as separate reconstruction gates. The
HDF5 input is float32, so this cannot claim bitwise identity with an original
float64 live DAQ stream unless the captured values match the canonical input.

Run from the repository root after the HDF5 recordings are closed:
    python tools/compare_esn_thresholds.py --artifact path/to/model.npz run1.h5 run2.h5 run3.h5

The default replays every committed chunk in each recording. Use
``--limit-seconds 30`` for a 30-second prefix; the JSON report labels this as
``limited_prefix`` and marks whether the whole selected prefix completed.
The old NumPy oracle and incremental detector see the same bridge-conditioned
prediction block, so the result checks threshold statistics and exact trigger
indices without treating float32 block-mean diagnostics as full predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from esn_bridge import EchoChamberEsnBridge  # noqa: E402
from esn import load_artifact  # noqa: E402


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _old_step(
    history: deque[float],
    samples_since_pulse: int,
    previous: float | None,
    bridge: EchoChamberEsnBridge,
    conditioned: np.ndarray,
) -> tuple[float, float, bool, int | None, int, float | None]:
    """Pre-optimization decision path; preserve history and event semantics."""
    values = np.asarray(conditioned[0, ::bridge.decim_q], dtype=np.float64)
    if bridge.pulse_polarity == "positive":
        metric = values
    elif bridge.pulse_polarity == "negative":
        metric = -values
    else:
        metric = np.abs(values)
    peak = float(np.max(metric))
    warmup = max(5, int(min(1.0, bridge.pulse_window_sec) * bridge.model_rate))
    threshold = float("nan")
    fired = False
    trigger_sample: int | None = None
    samples_since_pulse += values.size
    if len(history) >= warmup:
        recent = np.fromiter(history, dtype=np.float64)
        threshold = float(np.mean(recent) + bridge.pulse_threshold_std * (np.std(recent) + 1e-12))
        crossings = np.flatnonzero(
            (metric > threshold)
            & np.r_[previous is None or previous <= threshold, metric[:-1] <= threshold]
        )
        interval = int(bridge.pulse_min_interval_sec * bridge.model_rate)
        if crossings.size and samples_since_pulse >= interval:
            fired = True
            trigger_sample = int(crossings[0]) * bridge.decim_q
            samples_since_pulse = 0
    previous = float(metric[-1]) if metric.size else previous
    if not fired:
        history.extend(float(value) for value in metric)
    return threshold, peak, fired, trigger_sample, samples_since_pulse, previous


def _load_recording(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as source:
        if source.attrs.get("schema") != "echo-chamber-recording":
            raise ValueError(f"{path}: unsupported HDF5 schema")
        tag = _text(source.attrs.get("format_tag", ""))
        if tag != "echoChamber_H5_v5":
            raise ValueError(f"{path}: expected echoChamber_H5_v5, found {tag!r}")
        if not bool(source.attrs.get("complete", 0)):
            raise ValueError(f"{path}: recording is not closed (complete=0)")
        signal_dataset = source["signals/electrode_mV"]
        if signal_dataset.ndim != 2 or signal_dataset.shape[0] != 2:
            raise ValueError(f"{path}: expected a two-channel electrode_mV dataset")
        committed = int(source.attrs.get("committed_samples", 0))
        if committed != int(signal_dataset.shape[1]):
            raise ValueError(
                f"{path}: committed_samples={committed} differs from electrode_mV length={signal_dataset.shape[1]}"
            )
        if committed <= 0:
            raise ValueError(f"{path}: recording contains no committed input samples")
        metadata = json.loads(_text(source.attrs.get("metadata_json", "{}")))
        config = metadata.get("configuration", {})
        required = ("sample_rate", "chunk_size", "ctx_index", "pulse_threshold_std",
                    "pulse_window_sec", "pulse_polarity")
        missing = [name for name in required if name not in config]
        if missing:
            raise ValueError(f"{path}: recording metadata is missing {missing}")
        chunk = int(config["chunk_size"])
        if chunk <= 0 or committed % chunk:
            raise ValueError(f"{path}: committed input does not contain complete {chunk}-sample blocks")
        expected_offsets = list(range(0, committed, chunk))
        # Runtime discontinuities are explicit and cause a controller reset and
        # a zero-action block. Keep them in the block replay below.
        gaps = source.get("events/sample_discontinuities")
        gap_records = {}
        if gaps is not None:
            gap_values = list(gaps[:])
            gap_offsets = [int(row["sample_offset"]) for row in gap_values]
            if len(gap_offsets) != len(set(gap_offsets)):
                raise ValueError(f"{path}: duplicate sample-discontinuity offsets")
            gap_records = {
                int(row["sample_offset"]): (
                    int(row["expected_ai_sample_index"]),
                    int(row["actual_ai_sample_index"]),
                )
                for row in gap_values
            }
        if any(offset not in expected_offsets for offset in gap_records):
            raise ValueError(f"{path}: a sample discontinuity is not aligned to a complete input block")
        records = source["diagnostics/controller_blocks"][:]
        controller_rows = {}
        for row in records:
            offset = int(row["sample_offset"])
            if offset in controller_rows:
                raise ValueError(f"{path}: duplicate controller diagnostics at sample offset {offset}")
            controller_rows[offset] = row
        block_diagnostics = source["diagnostics/blocks"][:]
        if len(block_diagnostics) == 0:
            raise ValueError(f"{path}: recording has no block diagnostics")
        block_offsets = [int(row["sample_offset"]) for row in block_diagnostics]
        if block_offsets != expected_offsets:
            raise ValueError(f"{path}: block diagnostics are incomplete, duplicated, or out of order")
        controller_offsets = [int(row["sample_offset"]) for row in records]
        if controller_offsets != expected_offsets:
            raise ValueError(f"{path}: controller diagnostics are incomplete, duplicated, or out of order")
        if any(int(row["sample_count"]) != chunk for row in block_diagnostics):
            raise ValueError(f"{path}: block diagnostics contain a partial or mis-sized input block")
        generations = {int(row["config_generation"]) for row in records}
        if len(generations) != 1:
            raise ValueError(
                f"{path}: configuration generation changed during recording; per-generation ESN settings are unavailable"
            )
        block_sample_index = {
            int(row["sample_offset"]): int(row["ai_sample_index"])
            for row in block_diagnostics
        }
        if block_sample_index[0] != 0:
            raise ValueError(f"{path}: recording starts after sample zero; prior ESN state cannot be reconstructed")
        if 0 in gap_records:
            raise ValueError(f"{path}: discontinuity at the first recorded block has no reconstructable predecessor")
        for previous_offset, current_offset in zip(expected_offsets, expected_offsets[1:]):
            expected_sample = block_sample_index[previous_offset] + chunk
            current_sample = block_sample_index[current_offset]
            gap = gap_records.get(current_offset)
            if current_sample != expected_sample:
                if gap is None or gap != (expected_sample, current_sample):
                    raise ValueError(
                        f"{path}: sample-index discontinuity at offset {current_offset} lacks matching gap metadata"
                    )
            elif gap is not None:
                raise ValueError(f"{path}: redundant gap metadata at offset {current_offset}")
        block_model_mean = {
            int(row["sample_offset"]): float(row["model_esn_mean_mV"])
            for row in block_diagnostics
        }
        block_pulse_threshold = {
            int(row["sample_offset"]): float(row["pulse_threshold_mV"])
            for row in block_diagnostics
        }
        block_state_code = {
            int(row["sample_offset"]): int(row["state_code"])
            for row in records
        }
        return {
            "path": path,
            "metadata": metadata,
            "config": config,
            "committed": committed,
            "gap_offsets": set(gap_records),
            "controller_rows": controller_rows,
            "block_sample_index": block_sample_index,
            "block_model_mean": block_model_mean,
            "block_pulse_threshold": block_pulse_threshold,
            "block_state_code": block_state_code,
            "first_sample_index": int(block_diagnostics[0]["ai_sample_index"]),
        }


def _new_bridge(record: dict[str, Any], artifact: Path) -> EchoChamberEsnBridge:
    config = record["config"]
    streamer = load_artifact(str(artifact), backend=str(config.get("esn_backend", "auto")))
    bridge = EchoChamberEsnBridge(
        streamer,
        runtime_chunk_size=int(config["chunk_size"]),
        sample_rate=int(config["sample_rate"]),
        passthrough_dc_block_hz=float(config.get("passthrough_dc_block_hz", 0.5)),
        pulse_threshold_std=float(config["pulse_threshold_std"]),
        pulse_window_sec=float(config["pulse_window_sec"]),
        pulse_polarity=str(config["pulse_polarity"]),
        ao_command_gain_v_per_esn_unit=float(config.get("ao_command_gain_v_per_esn_unit", 1.0)),
    )
    # Threshold statistics are computed in every mode by the live bridge.
    # Selecting threshold_pulse here exposes detector results without changing
    # the ESN recurrence; render_legacy=False prevents any AO waveform creation.
    bridge.configure(stim_mode="threshold_pulse", stim_gain=1.0)
    return bridge


def _quantiles(values_ns: list[int]) -> dict[str, float | None]:
    if not values_ns:
        return {"median_ms": None, "p99_ms": None}
    values = np.asarray(values_ns, dtype=np.float64) / 1e6
    return {"median_ms": float(np.median(values)), "p99_ms": float(np.percentile(values, 99))}


def replay(record: dict[str, Any], artifact: Path, max_seconds: float | None) -> dict[str, Any]:
    config = record["config"]
    chunk = int(config["chunk_size"])
    rate = int(config["sample_rate"])
    if chunk <= 0 or rate <= 0 or record["committed"] % chunk:
        raise ValueError(f"{record['path']}: input length is not a whole number of configured chunks")
    if int(config["ctx_index"]) not in (0, 1):
        raise ValueError(f"{record['path']}: ctx_index must identify one of the two recorded channels")
    if record["first_sample_index"] != 0:
        raise ValueError(f"{record['path']}: recording begins after sample zero; ESN state may predate capture")

    bridge = _new_bridge(record, artifact)
    history: deque[float] = deque(maxlen=max(1, int(float(config["pulse_window_sec"]) * bridge.model_rate)))
    old_since_pulse = 10**9
    old_previous: float | None = None
    last_generation: int | None = None
    was_active = False
    mismatches: list[dict[str, Any]] = []
    threshold_errors: list[float] = []
    threshold_error_failures = 0
    threshold_error_examples: list[dict[str, Any]] = []
    recorded_model_mean_errors: list[float] = []
    recorded_threshold_errors: list[float] = []
    reconstruction_failures = 0
    execution_evidence_mismatches = 0
    model_mean_mismatches = 0
    threshold_record_mismatches = 0
    reconstruction_failure_examples: list[dict[str, Any]] = []
    model_hash = hashlib.sha256()
    old_trigger_hash = hashlib.sha256()
    new_trigger_hash = hashlib.sha256()
    old_trigger_count = new_trigger_count = 0
    new_stats_ns: list[int] = []
    old_stats_ns: list[int] = []
    bridge_total_ns: list[int] = []
    skipped_blocks = reset_count = compared_blocks = 0
    processed_blocks = 0
    max_blocks = record["committed"] // chunk
    if max_seconds is not None:
        max_blocks = min(max_blocks, max(1, int(max_seconds * rate // chunk)))

    original_update = bridge._update_pulse_state

    def timed_update(conditioned: np.ndarray):
        started = time.perf_counter_ns()
        try:
            return original_update(conditioned)
        finally:
            new_stats_ns.append(time.perf_counter_ns() - started)

    bridge._update_pulse_state = timed_update  # type: ignore[method-assign]
    input_file = h5py.File(record["path"], "r")
    for block_number in range(max_blocks):
        processed_blocks = block_number + 1
        offset = block_number * chunk
        end = offset + chunk
        row = record["controller_rows"].get(offset)
        controller_name = _text(row["controller_id"]).rstrip("\x00") if row is not None else ""
        generation = int(row["config_generation"]) if row is not None else None
        executed = (controller_name == "esn"
                    and record["block_state_code"].get(offset) == 2)
        recorded_mean = record["block_model_mean"].get(offset, float("nan"))
        recorded_block_threshold = record["block_pulse_threshold"].get(offset, float("nan"))
        has_gap_event = offset in record["gap_offsets"]
        samples = np.asarray(input_file["signals/electrode_mV"][:, offset:end], dtype=np.float64)
        try:
            block_first_sample = record["block_sample_index"][offset]
        except KeyError as exc:
            raise ValueError(f"{record['path']}: diagnostics lack block at offset {offset}") from exc
        invalid = not np.all(np.isfinite(samples))

        if executed and (not was_active or generation != last_generation or has_gap_event or invalid):
            bridge.reset()
            history.clear()
            old_since_pulse = 10**9
            old_previous = None
            reset_count += 1
        if not executed:
            if np.isfinite(recorded_mean):
                reconstruction_failures += 1
                execution_evidence_mismatches += 1
                if len(reconstruction_failure_examples) < 20:
                    reconstruction_failure_examples.append({
                        "sample_offset": offset,
                        "reason": "finite model diagnostic on a block without recorded ESN execution",
                    })
            if np.isfinite(recorded_block_threshold):
                reconstruction_failures += 1
                execution_evidence_mismatches += 1
                threshold_record_mismatches += 1
                if len(reconstruction_failure_examples) < 20:
                    reconstruction_failure_examples.append({
                        "sample_offset": offset,
                        "reason": "finite threshold diagnostic on a block without recorded ESN execution",
                    })
            if was_active:
                bridge.reset()
                history.clear()
                old_since_pulse = 10**9
                old_previous = None
                reset_count += 1
            skipped_blocks += 1
            was_active = False
            last_generation = generation
            continue
        if invalid or has_gap_event:
            if np.isfinite(recorded_mean):
                reconstruction_failures += 1
                execution_evidence_mismatches += 1
                if len(reconstruction_failure_examples) < 20:
                    reconstruction_failure_examples.append({
                        "sample_offset": offset,
                        "reason": "finite model diagnostic on invalid or discontinuous input block",
                    })
            if np.isfinite(recorded_block_threshold):
                reconstruction_failures += 1
                execution_evidence_mismatches += 1
                threshold_record_mismatches += 1
                if len(reconstruction_failure_examples) < 20:
                    reconstruction_failure_examples.append({
                        "sample_offset": offset,
                        "reason": "finite threshold diagnostic on invalid or discontinuous input block",
                    })
            skipped_blocks += 1
            was_active = False
            last_generation = generation
            continue

        started = time.perf_counter_ns()
        bridge.process(samples, ctx_index=int(config["ctx_index"]), render_legacy=False)
        bridge_total_ns.append(time.perf_counter_ns() - started)
        details = bridge.detailed_diagnostics()
        assert details.conditioned_output is not None
        model_hash.update(np.asarray(details.model_output, dtype="<f8").tobytes())
        replayed_mean_f32 = np.float32(np.mean(details.model_output))
        if np.isfinite(recorded_mean):
            mean_error = abs(float(replayed_mean_f32) - recorded_mean)
            recorded_model_mean_errors.append(mean_error)
            if replayed_mean_f32 != np.float32(recorded_mean):
                reconstruction_failures += 1
                model_mean_mismatches += 1
                if len(reconstruction_failure_examples) < 20:
                    reconstruction_failure_examples.append({
                        "sample_offset": offset,
                        "recorded_model_mean_f32": recorded_mean,
                        "replayed_model_mean_f32": float(replayed_mean_f32),
                        "absolute_error": mean_error,
                    })
        else:
            reconstruction_failures += 1
            model_mean_mismatches += 1
            if len(reconstruction_failure_examples) < 20:
                reconstruction_failure_examples.append({
                    "sample_offset": offset,
                    "reason": "missing finite model diagnostic on an executed valid ESN block",
                })
        old_started = time.perf_counter_ns()
        expected = _old_step(history, old_since_pulse, old_previous, bridge, details.conditioned_output)
        old_stats_ns.append(time.perf_counter_ns() - old_started)
        old_threshold, old_peak, old_fired, old_trigger, old_since_pulse, old_previous = expected
        actual = bridge._diagnostics
        new_trigger = actual.trigger_sample
        new_threshold = actual.pulse_threshold
        recorded_threshold = recorded_block_threshold
        if np.isfinite(old_threshold) and np.isfinite(recorded_threshold):
            threshold_record_error = abs(
                float(np.float32(old_threshold)) - float(np.float32(recorded_threshold))
            )
            recorded_threshold_errors.append(threshold_record_error)
            if np.float32(old_threshold) != np.float32(recorded_threshold):
                reconstruction_failures += 1
                threshold_record_mismatches += 1
                if len(reconstruction_failure_examples) < 20:
                    reconstruction_failure_examples.append({
                        "sample_offset": offset,
                        "reason": "replayed threshold differs from recorded float32 threshold",
                        "recorded_threshold_f32": recorded_threshold,
                        "replayed_threshold_f32": float(np.float32(old_threshold)),
                        "absolute_error": threshold_record_error,
                    })
        elif np.isfinite(old_threshold) != np.isfinite(recorded_threshold):
            reconstruction_failures += 1
            threshold_record_mismatches += 1
            if len(reconstruction_failure_examples) < 20:
                reconstruction_failure_examples.append({
                    "sample_offset": offset,
                    "reason": "replayed and recorded threshold finite-status mismatch",
                    "recorded_threshold": _finite_or_none(recorded_threshold),
                    "replayed_threshold": _finite_or_none(old_threshold),
                })
        compared_blocks += 1
        if np.isfinite(old_threshold) and np.isfinite(new_threshold):
            threshold_error = abs(old_threshold - new_threshold)
            threshold_errors.append(threshold_error)
            threshold_tolerance = max(
                2e-12, 4.0 * abs(float(np.spacing(old_threshold)))
            )
            if threshold_error > threshold_tolerance:
                threshold_error_failures += 1
                if len(threshold_error_examples) < 20:
                    threshold_error_examples.append({
                        "sample_index": block_first_sample,
                        "old_threshold": old_threshold,
                        "new_threshold": new_threshold,
                        "absolute_error": threshold_error,
                        "tolerance": threshold_tolerance,
                    })
        elif np.isfinite(old_threshold) != np.isfinite(new_threshold):
            threshold_error_failures += 1
            if len(threshold_error_examples) < 20:
                threshold_error_examples.append({
                    "sample_index": block_first_sample,
                    "old_threshold": _finite_or_none(old_threshold),
                    "new_threshold": _finite_or_none(new_threshold),
                    "reason": "finite-status mismatch",
                })
        old_absolute_trigger = (
            block_first_sample + old_trigger if old_trigger is not None else None
        )
        new_absolute_trigger = (
            block_first_sample + new_trigger if new_trigger is not None else None
        )
        if old_fired and old_absolute_trigger is not None:
            old_trigger_hash.update(int(old_absolute_trigger).to_bytes(8, "little", signed=True))
            old_trigger_count += 1
        if actual.pulse_fired and new_absolute_trigger is not None:
            new_trigger_hash.update(int(new_absolute_trigger).to_bytes(8, "little", signed=True))
            new_trigger_count += 1
        event_mismatch = (
            bool(actual.pulse_fired) != old_fired
            or new_trigger != old_trigger
            or (old_fired and old_trigger is None)
            or (bool(actual.pulse_fired) and new_trigger is None)
        )
        if event_mismatch:
            if len(mismatches) < 20:
                mismatches.append({
                    "sample_index": block_first_sample, "old_fired": old_fired,
                    "new_fired": bool(actual.pulse_fired),
                    "old_trigger_sample_index": old_absolute_trigger,
                    "new_trigger_sample_index": new_absolute_trigger,
                    "old_threshold": _finite_or_none(old_threshold),
                    "new_threshold": _finite_or_none(new_threshold),
                })
            # Stop on the first divergence: later histories legitimately differ
            # because each algorithm excludes its own detected-event blocks.
            break
        if bridge._samples_since_pulse != old_since_pulse:
            raise AssertionError("old/new refractory counters diverged despite matching event state")
        was_active = True
        last_generation = generation
    input_file.close()

    max_threshold_error = max(threshold_errors, default=None)
    full_blocks = record["committed"] // chunk
    selected_coverage_complete = processed_blocks == max_blocks and not mismatches
    detector_gate_passed = (
        compared_blocks > 0 and not mismatches and threshold_error_failures == 0
        and bool(threshold_errors) and old_trigger_count == new_trigger_count
        and old_trigger_hash.digest() == new_trigger_hash.digest()
    )
    reconstruction_gate_passed = (
        reconstruction_failures == 0 and bool(recorded_model_mean_errors)
        and bool(recorded_threshold_errors)
    )
    return {
        "recording": str(record["path"]),
        "artifact": str(artifact),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "recording_float32_input_replayed_through_esn": True,
        "actual_live_float64_daq_bitwise_reproduction": False,
        "configuration": {key: config[key] for key in (
            "sample_rate", "chunk_size", "ctx_index", "pulse_threshold_std",
            "pulse_window_sec", "pulse_polarity", "passthrough_dc_block_hz") if key in config},
        "coverage": {
            "scope": "limited_prefix" if max_seconds is not None else "full_recording",
            "requested_limit_seconds": max_seconds,
            "planned_blocks": max_blocks,
            "processed_blocks": processed_blocks,
            "selected_input_complete": selected_coverage_complete,
            "all_committed_input_covered": max_blocks == full_blocks and processed_blocks == full_blocks,
            "committed_input_samples": record["committed"],
            "selected_input_samples": max_blocks * chunk,
            "stopped_at_first_event_mismatch": bool(mismatches),
        },
        "blocks": {"compared_esn": compared_blocks, "skipped": skipped_blocks,
                   "resets": reset_count},
        "event_parity": {"mismatches": len(mismatches), "examples": mismatches},
        "detector_parity_gate": {
            "passed": detector_gate_passed,
            "requires_exact_fired_and_trigger_indices": True,
            "requires_bounded_threshold_error": True,
        },
        "threshold_abs_error": {"max": max_threshold_error,
                                "p99": float(np.percentile(threshold_errors, 99)) if threshold_errors else None,
                                "mean": float(np.mean(threshold_errors)) if threshold_errors else None,
                                "compared_blocks": len(threshold_errors),
                                "tolerance_rule": "max(2e-12, 4 * abs(spacing(old_numpy_threshold)))",
                                "gate_failures": threshold_error_failures,
                                "gate_passed": threshold_error_failures == 0,
                                "failure_examples": threshold_error_examples},
        "trigger_indices": {
            "old_numpy": {"count": old_trigger_count, "sha256": old_trigger_hash.hexdigest()},
            "incremental": {"count": new_trigger_count, "sha256": new_trigger_hash.hexdigest()},
            "identical": (old_trigger_count == new_trigger_count
                          and old_trigger_hash.digest() == new_trigger_hash.digest()),
        },
        "recording_reconstruction": {
            "scope": "exact float32 block means and threshold diagnostics; HDF5 does not store full model predictions",
            "model_block_mean_check": {
                "compared_blocks": len(recorded_model_mean_errors),
                "mismatches": model_mean_mismatches,
                "max_abs_float32_error": max(recorded_model_mean_errors, default=None),
            },
            "recorded_threshold_check": {
                "compared_blocks": len(recorded_threshold_errors),
                "mismatches": threshold_record_mismatches,
                "max_abs_float32_error": max(recorded_threshold_errors, default=None),
            },
            "execution_evidence_mismatches": execution_evidence_mismatches,
            "total_gate_failures": reconstruction_failures,
            "mismatch_examples": reconstruction_failure_examples,
            "gate_passed": reconstruction_gate_passed,
        },
        "rolling_stat_rebuild_count": bridge.rolling_stat_rebuild_count,
        "exact_threshold_fallback_count": bridge.exact_threshold_fallback_count,
        "prediction_stream_sha256": model_hash.hexdigest(),
        "timing": {"bridge_total": _quantiles(bridge_total_ns),
                   "new_threshold_stats": _quantiles(new_stats_ns),
                   "old_numpy_threshold_stats": _quantiles(old_stats_ns)},
        "limitations": [
            "The reconstruction gate compares only per-block float32 means; it does not recover full model predictions from those means.",
            "Exact float32 mean matching is authoritative only when the captured electrode_mV inputs match the canonical input samples; live DAQ recordings may lose float64 detail before HDF5 storage and can correctly fail this gate.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path, help="numeric ESN .npz artifact")
    parser.add_argument("--json-output", type=Path, help="write full results to this JSON path")
    parser.add_argument("--limit-seconds", type=float, help="limit each replay after this duration")
    parser.add_argument("recordings", nargs="+", type=Path)
    args = parser.parse_args()
    if args.limit_seconds is not None and args.limit_seconds <= 0:
        parser.error("--limit-seconds must be positive")
    artifact = args.artifact.resolve()
    if not artifact.is_file():
        parser.error(f"ESN artifact does not exist: {artifact}")
    results = [replay(_load_recording(path.resolve()), artifact, args.limit_seconds)
               for path in args.recordings]
    report = {"probe": "ESN rolling threshold replay parity",
              "result": "pass" if all(
                  item["coverage"]["selected_input_complete"]
                  and item["blocks"]["compared_esn"] > 0
                  and item["event_parity"]["mismatches"] == 0
                  and item["threshold_abs_error"]["gate_passed"]
                  and item["threshold_abs_error"]["compared_blocks"] > 0
                  and item["trigger_indices"]["identical"]
                  and item["recording_reconstruction"]["gate_passed"]
                  and item["detector_parity_gate"]["passed"]
                  for item in results
              ) else "fail",
              "recordings": results}
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.json_output:
        args.json_output.resolve().write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
