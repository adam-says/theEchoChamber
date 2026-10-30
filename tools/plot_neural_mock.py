"""Plot paired neural mock output and export source-clock event timestamps.

This is a report/trace reader only. It does not run the mock or infer physical
latency. The x-axis is the 20 kHz virtual sample clock stored in the trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_TRACE_ARRAYS = (
    "sample_index", "true_lfp_mV", "artifact_mV", "observed_ca3_mV",
    "generated_ao_V", "artifact_oracle_mask", "core_feature_valid",
    "controller_source_block_start", "controller_z_score",
)


def _trace_path(report_path: Path, condition: dict[str, Any]) -> Path:
    value = condition.get("trace_path")
    if not isinstance(value, str) or not value:
        raise ValueError("condition is missing trace_path")
    path = Path(value)
    if path.is_absolute() and path.is_file():
        return path
    sibling = report_path.parent / path.name
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(f"trace_path from report is unavailable: {path}")


def _load_trace(path: Path, expected_samples: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_TRACE_ARRAYS) - set(archive.files))
        if missing:
            raise ValueError(f"{path.name} is missing required arrays: {missing}")
        data = {name: np.asarray(archive[name]) for name in REQUIRED_TRACE_ARRAYS}
    indices = data["sample_index"]
    if indices.shape != (expected_samples,) or not np.array_equal(
            indices, np.arange(expected_samples, dtype=indices.dtype)):
        raise ValueError(f"{path.name} sample_index is not a complete contiguous source timeline")
    for name in REQUIRED_TRACE_ARRAYS:
        array = data[name]
        expected = expected_samples if name not in {
            "controller_source_block_start", "controller_z_score"} else expected_samples // 100
        if array.shape != (expected,):
            raise ValueError(f"{path.name} array {name} has unexpected shape {array.shape}")
        if name != "controller_z_score" and array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
            raise ValueError(f"{path.name} array {name} contains non-finite values")
    return data


def _first_divergence(left: np.ndarray, right: np.ndarray, *, tolerance: float = 0.0) -> int | None:
    if left.shape != right.shape:
        raise ValueError("paired trace arrays have different shapes")
    differing = np.flatnonzero(np.abs(left.astype(np.float64) - right.astype(np.float64)) > tolerance)
    return int(differing[0]) if differing.size else None


def _validate_report(report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    if report.get("schema") != "echo-chamber-mock-neural-demo-v1":
        raise ValueError("report schema is not echo-chamber-mock-neural-demo-v1")
    if report.get("closure_demonstrated") is not True:
        raise ValueError("report does not declare closed-loop demonstration complete")
    gates = report.get("closure_gates", {})
    if not isinstance(gates, dict) or not gates or any(value is not True for value in gates.values()):
        raise ValueError("one or more neural mock closure gates failed")
    if report.get("artifact", {}).get("sample_rate_hz") != 20_000:
        raise ValueError("report artifact clock is not 20 kHz")
    if report.get("paired_comparison", {}).get("virtual_clock_only") is not True:
        raise ValueError("report does not identify its sample clock as virtual")
    if report.get("paired_comparison", {}).get("physical_latency_claim") is not False:
        raise ValueError("report contains an unexpected physical-latency claim")
    off, on = report.get("off"), report.get("rule_feedback")
    if not isinstance(off, dict) or not isinstance(on, dict):
        raise ValueError("report is missing paired off and feedback conditions")
    sample_rate = int(on.get("sample_rate_hz", 0))
    if sample_rate != 20_000 or int(off.get("sample_rate_hz", 0)) != sample_rate:
        raise ValueError("paired conditions do not use the expected 20 kHz sample clock")
    expected = int(report.get("paired_comparison", {}).get("paired_source_samples", 0))
    if expected < 1 or expected % 100:
        raise ValueError("paired source sample count is missing or not aligned to 100-sample blocks")
    for name, condition in (("off", off), ("rule_feedback", on)):
        if (condition.get("source_samples_processed") != expected
                or condition.get("recorded_samples") != expected
                or condition.get("h5_complete") is not True
                or condition.get("complete_finite_h5_input_coverage") is not True
                or condition.get("h5_input_sample_mismatches") != 0
                or condition.get("h5_diagnostic_source_offsets_complete") is not True):
            raise ValueError(f"{name} condition does not have complete source and recording coverage")
        if condition.get("fault") or condition.get("scheduler_faults"):
            raise ValueError(f"{name} condition contains a runtime or scheduler fault")
        if condition.get("transition_pending_after_drain") is True:
            raise ValueError(f"{name} condition did not finish its AO drain")
        if int(condition.get("scheduler_submit_rejections", -1)) != 0:
            raise ValueError(f"{name} condition has AO scheduler submit rejections")
    first_generated = on.get("first_generated_nonzero_ao_sample_index")
    effect_sample = on.get("first_neural_effect_sample_index")
    target_sample = on.get("first_rule_target_onset_sample_index")
    source_sample = on.get("first_rule_decision_source_sample_index")
    latest_sample = on.get("first_rule_latest_sample_index")
    valid_latest = report.get("paired_comparison", {}).get(
        "first_finite_rule_measurement_latest_used_sample_index")
    required = (first_generated, effect_sample, target_sample, source_sample, latest_sample, valid_latest)
    if any(value is None for value in required):
        raise ValueError("feedback event sample indices are incomplete")
    first_generated, effect_sample = int(first_generated), int(effect_sample)
    target_sample, source_sample, latest_sample = map(int, (target_sample, source_sample, latest_sample))
    valid_latest = int(valid_latest)
    configured_delay = int(on.get("configured_neural_effect_delay_samples", -1))
    if effect_sample - first_generated != configured_delay:
        raise ValueError("reported neural effect onset disagrees with configured sample delay")
    if int(report.get("paired_comparison", {}).get("first_latent_lfp_divergence_sample_index", -1)) != effect_sample:
        raise ValueError("first paired latent divergence does not match reported neural effect onset")
    # The event source onset and scheduled AO target use the configured 100 ms
    # source-to-target lead. The later latest-used sample is a different clock point.
    if target_sample - source_sample != int(sample_rate * 0.1):
        raise ValueError("configured target onset does not follow the 100 ms source lead")
    if latest_sample < source_sample or valid_latest < target_sample:
        raise ValueError("latest-used source indices precede their associated event")
    return off, on, expected, first_generated


def _event_rows(report: dict[str, Any], off: dict[str, Any], on: dict[str, Any],
                first_generated: int, sample_rate: int) -> list[dict[str, Any]]:
    comparison = report["paired_comparison"]
    decision = int(on["first_rule_decision_source_sample_index"])
    latest = int(on["first_rule_latest_sample_index"])
    target = int(on["first_rule_target_onset_sample_index"])
    generated = int(on["first_generated_nonzero_ao_sample_index"])
    effect = int(on["first_neural_effect_sample_index"])
    valid_start = int(comparison["first_finite_rule_measurement_divergence_block_start_sample_index"])
    valid_latest = int(comparison["first_finite_rule_measurement_latest_used_sample_index"])
    events = [
        ("first_decision_source_onset", decision,
         "Event source onset sample; not the decision block start."),
        ("decision_latest_sample_used", latest,
         "Last source sample used by the decision block; the following AO availability boundary is latest+1."),
        ("configured_ao_target_onset", target,
         "Scheduled target sample; target minus event source onset is the configured 100 ms lead."),
        ("first_generated_nonzero_ao", generated,
         "First nonzero sample in generated_ao_V on the virtual AO sample clock."),
        ("first_latent_neural_effect", effect,
         "First feedback/off latent LFP divergence; this is model time, not a physical latency."),
        ("first_later_valid_controller_block_start", valid_start,
         "Start index of the block with the first later finite controller measurement."),
        ("first_later_valid_controller_latest_used", valid_latest,
         "Latest source sample used by that valid controller measurement."),
    ]
    result = []
    for event, sample_index, note in events:
        result.append({
            "condition": "rule_feedback", "event": event,
            "sample_index": sample_index,
            "time_s_virtual_sample_clock": f"{sample_index / sample_rate:.6f}",
            "relative_to_first_generated_nonzero_ms": f"{(sample_index-first_generated)*1000/sample_rate:.3f}",
            "note": note,
        })
    result.append({
        "condition": "off_control", "event": "paired_reference_first_generated_sample",
        "sample_index": "", "time_s_virtual_sample_clock": "",
        "relative_to_first_generated_nonzero_ms": "",
        "note": "Off control contains no generated nonzero AO; timeline aligns to the feedback condition's first nonzero sample.",
    })
    return result


def _plot(report: dict[str, Any], off: dict[str, Any], on: dict[str, Any],
          off_data: dict[str, np.ndarray], on_data: dict[str, np.ndarray],
          expected: int, first_generated: int, output_path: Path) -> None:
    fs = int(on["sample_rate_hz"])
    effect = int(on["first_neural_effect_sample_index"])
    valid_latest = int(report["paired_comparison"]["first_finite_rule_measurement_latest_used_sample_index"])
    source_onset = int(on["first_rule_decision_source_sample_index"])
    latest_used = int(on["first_rule_latest_sample_index"])
    target = int(on["first_rule_target_onset_sample_index"])
    lo = max(0, first_generated - int(0.110 * fs))
    hi = min(expected, first_generated + int(0.250 * fs))
    sl = slice(lo, hi)
    x_ms = (on_data["sample_index"][sl].astype(np.float64) - first_generated) * 1000.0 / fs
    off_slice = slice(lo, hi)

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [0.9, 1.25, 1.25, 0.8]})
    ax_ao, ax_latent, ax_observed, ax_valid = axes
    colors = {"off": "#808080", "on": "#1769aa", "artifact": "#cf6b2c",
              "mask": "#d9a52e", "valid": "#2d8a57"}

    ax_ao.plot(x_ms, off_data["generated_ao_V"][off_slice], color=colors["off"],
               linewidth=0.8, label="off control generated AO")
    ax_ao.plot(x_ms, on_data["generated_ao_V"][sl], color=colors["on"],
               linewidth=0.9, label="feedback generated AO")
    ax_ao.set_ylabel("Generated AO (V)")
    ax_ao.set_title("Generated AO pulse and source-to-effect timeline (virtual sample clock)")
    ax_ao.grid(True, alpha=0.25)

    ax_latent.plot(x_ms, off_data["true_lfp_mV"][off_slice], color=colors["off"],
                   linewidth=0.8, label="off control latent LFP")
    ax_latent.plot(x_ms, on_data["true_lfp_mV"][sl], color=colors["on"],
                   linewidth=0.8, label="feedback latent LFP")
    ax_latent.set_ylabel("Latent model LFP (mV)")
    ax_latent.set_title("Latent neural state (separate from electrical artifact; not directly observed)")
    ax_latent.grid(True, alpha=0.25)

    ax_observed.plot(x_ms, off_data["observed_ca3_mV"][off_slice], color=colors["off"],
                     linewidth=0.75, label="off control recorded input")
    ax_observed.plot(x_ms, on_data["observed_ca3_mV"][sl], color=colors["on"],
                     linewidth=0.75, label="feedback recorded input")
    artifact_axis = ax_observed.twinx()
    artifact_axis.plot(x_ms, on_data["artifact_mV"][sl], color=colors["artifact"],
                       linewidth=0.75, alpha=0.85, linestyle="--", label="feedback artifact signal")
    oracle = on_data["artifact_oracle_mask"][sl].astype(bool)
    ax_observed.fill_between(x_ms, 0, 1, where=oracle, transform=ax_observed.get_xaxis_transform(),
                             color=colors["mask"], alpha=0.16, step="post",
                             label="artifact oracle mask")
    ax_observed.set_ylabel("Recorded CA3 input (mV)")
    artifact_axis.set_ylabel("Artifact (mV)")
    ax_observed.set_title("Recorded sensor input and electrical artifact (separate y axes)")
    ax_observed.grid(True, alpha=0.2)

    ax_valid.step(x_ms, off_data["core_feature_valid"][off_slice].astype(np.uint8),
                  where="post", color=colors["off"], linewidth=0.9, label="off Core feature-valid")
    ax_valid.step(x_ms, on_data["core_feature_valid"][sl].astype(np.uint8),
                  where="post", color=colors["valid"], linewidth=1.0,
                  label="feedback Core feature-valid")
    ax_valid.step(x_ms, on_data["artifact_oracle_mask"][sl].astype(np.uint8),
                  where="post", color=colors["mask"], linewidth=0.8, linestyle=":",
                  label="feedback artifact oracle mask")
    ax_valid.set_ylim(-0.08, 1.08)
    ax_valid.set_yticks([0, 1], ["0", "1"])
    ax_valid.set_ylabel("Feature state")
    ax_valid.set_xlabel("Virtual sample time relative to first generated nonzero AO (ms); not host or NI latency")
    ax_valid.grid(True, alpha=0.2)

    markers = [
        (source_onset, "source onset", "#6c6c6c", "--"),
        (latest_used, "latest source used", "#343434", ":"),
        (target, "configured AO target", "#cc6b2c", "--"),
        (first_generated, "first nonzero AO", colors["on"], "-"),
        (effect, "latent response starts", "#7a5195", "-"),
        (valid_latest, "first later valid measurement", colors["valid"], "-.") ,
    ]
    handles = []
    labels = []
    for sample, label, color, style in markers:
        xpos = (sample - first_generated) * 1000.0 / fs
        line = ax_ao.axvline(xpos, color=color, linestyle=style, linewidth=0.9, alpha=0.9)
        for ax in axes[1:]:
            ax.axvline(xpos, color=color, linestyle=style, linewidth=0.7, alpha=0.45)
        handles.append(line)
        labels.append(f"{label}: {sample}")

    phase_handles, phase_labels = ax_ao.get_legend_handles_labels()
    fig.legend(phase_handles, phase_labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.975), frameon=False, fontsize=8)
    lat_h, lat_l = ax_latent.get_legend_handles_labels()
    obs_h, obs_l = ax_observed.get_legend_handles_labels()
    art_h, art_l = artifact_axis.get_legend_handles_labels()
    observed_handles: list[Any] = []
    observed_labels: list[str] = []
    for handle, label in zip(obs_h + art_h, obs_l + art_l):
        if label not in observed_labels:
            observed_handles.append(handle)
            observed_labels.append(label)
    ax_observed.legend(observed_handles, observed_labels, loc="upper right", ncol=2,
                       fontsize=7, framealpha=0.9)
    ax_latent.legend(lat_h, lat_l, loc="upper right", ncol=2, fontsize=7, framealpha=0.9)
    valid_h, valid_l = ax_valid.get_legend_handles_labels()
    ax_valid.legend(valid_h, [
        "Off Core feature valid", "Feedback Core feature valid", "Artifact oracle mask",
    ], loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3,
       frameon=False, fontsize=7)
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.012), frameon=False, fontsize=7)

    fig.suptitle(
        f"Paired neural-mock closure example — {expected:,} raw samples retained per condition; "
        "engineering LFP-like proxy, not tissue or physical timing evidence",
        y=0.998, fontsize=11)
    fig.text(0.5, 0.045,
             "Core feature masking suppresses contaminated feedback features; it does not remove raw samples from H5.",
             ha="center", fontsize=8)
    axes[0].set_xlim(-110, 250)
    axes[0].set_xticks(np.arange(-100, 251, 50))
    fig.tight_layout(rect=(0.02, 0.075, 0.98, 0.945))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def generate(report_path: Path, output_dir: Path) -> tuple[Path, Path]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    off_report, on_report, expected, first_generated = _validate_report(report)
    fs = int(on_report["sample_rate_hz"])
    off_data = _load_trace(_trace_path(report_path, off_report), expected)
    on_data = _load_trace(_trace_path(report_path, on_report), expected)
    paired_first_effect = _first_divergence(off_data["true_lfp_mV"], on_data["true_lfp_mV"])
    if paired_first_effect != int(on_report["first_neural_effect_sample_index"]):
        raise ValueError("NPZ latent LFP divergence does not match report effect sample")
    generated = on_data["generated_ao_V"]
    nonzero = np.flatnonzero(generated != 0.0)
    if not nonzero.size or int(nonzero[0]) != first_generated:
        raise ValueError("NPZ generated AO first nonzero sample does not match report")
    if np.count_nonzero(generated) != int(on_report["scheduler_generated_nonzero_samples"]):
        raise ValueError("NPZ generated AO sample count does not match scheduler report")
    if np.any(off_data["generated_ao_V"] != 0.0):
        raise ValueError("off control trace contains generated nonzero AO")
    if _first_divergence(off_data["observed_ca3_mV"], on_data["observed_ca3_mV"]) != int(
            report["paired_comparison"]["first_observed_ca3_divergence_sample_index"]):
        raise ValueError("NPZ observed input divergence does not match report")
    if int(np.count_nonzero(on_data["artifact_oracle_mask"])) != int(on_report["artifact_mask_sample_count"]):
        raise ValueError("NPZ artifact oracle mask count does not match report")
    invalid_count = int(np.count_nonzero(~on_data["core_feature_valid"]))
    if invalid_count != int(on_report["core_feature_invalid_sample_count"]):
        raise ValueError("NPZ Core feature-valid mask count does not match report")
    if np.any(on_data["core_feature_valid"] & on_data["artifact_oracle_mask"]):
        raise ValueError("Core marked a known artifact sample feature-valid")
    block_latest = on_data["controller_source_block_start"] + 99
    z_on = on_data["controller_z_score"].astype(np.float64)
    z_off = off_data["controller_z_score"].astype(np.float64)
    joint = np.isfinite(z_on) & np.isfinite(z_off)
    differing_blocks = np.flatnonzero(joint & (np.abs(z_on - z_off) > 1e-7))
    first_valid_change_latest = (int(block_latest[differing_blocks[0]])
                                 if differing_blocks.size else None)
    expected_latest = int(report["paired_comparison"]["first_finite_rule_measurement_latest_used_sample_index"])
    if first_valid_change_latest != expected_latest:
        raise ValueError("NPZ first later valid controller measurement differs from report")

    output_dir.mkdir(parents=True)
    figure = output_dir / "neural-mock.png"
    timing_csv = output_dir / "timing-events.csv"
    _plot(report, off_report, on_report, off_data, on_data, expected, first_generated, figure)
    event_rows = _event_rows(report, off_report, on_report, first_generated, fs)
    with timing_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "condition", "event", "sample_index", "time_s_virtual_sample_clock",
            "relative_to_first_generated_nonzero_ms", "note"))
        writer.writeheader()
        writer.writerows(event_rows)
    print(json.dumps({"figure": str(figure), "timing_events": str(timing_csv),
                      "samples_per_condition": expected,
                      "first_generated_nonzero_sample": first_generated,
                      "first_neural_effect_sample": int(on_report["first_neural_effect_sample_index"]),
                      "delay_samples": int(on_report["first_neural_effect_sample_index"]) - first_generated,
                      "virtual_clock_only": True, "physical_latency_claim": False}))
    return figure, timing_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="closed paired neural-mock report JSON")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for neural-mock.png and timing-events.csv")
    args = parser.parse_args(argv)
    try:
        generate(args.report.resolve(), args.output_dir.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
