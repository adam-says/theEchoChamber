#!/usr/bin/env python3
"""Run paired off/Rule feedback traces with the shared AO command path.

This is a deterministic engineering fixture, not a calibrated neural/tissue
model. It uses a virtual sample clock and reports sample coordinates; it makes
no physical latency or biological efficacy claim.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
from scipy.signal import butter, sosfilt, welch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import echoChamber as app
from controllers import CausalEnvelope
from mock_neural import (
    ElectrodeArtifactParameters,
    NeuralMassParameters,
    SyntheticElectrodeObservation,
    TwoPopulationNeuralMass,
)


DEFAULT_RULE_SETTINGS = {"upper_z": 1.0, "lower_z": 0.5, "band_hz": (100.0, 250.0)}
DEFAULT_DURATION_S = 24.0
DISTURBANCE_START_S = 12.0
DISTURBANCE_END_S = 18.0
BASELINE_DRIVE = 0.05
DISTURBED_DRIVE = 0.30
PULSE_AMPLITUDE_V = 0.5


class _UnavailableEsn:
    ready = False
    streamer = None

    @staticmethod
    def reset() -> None:
        return None


class _HeadlessCore(app.ProcessingCore):
    """Keep actual Core logic while omitting only best-effort UI display work."""

    def _publish_ui(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _disturbance(first_sample: int, sample_rate_hz: int, size: int) -> np.ndarray:
    times = (first_sample + np.arange(size, dtype=np.int64)) / sample_rate_hz
    return np.where(
        (times >= DISTURBANCE_START_S) & (times < DISTURBANCE_END_S),
        DISTURBED_DRIVE,
        BASELINE_DRIVE,
    ).astype(np.float64)


def _controller_metrics(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    finite_z = np.asarray([
        float(row["z_score"]) for row in diagnostics
        if row.get("z_score") is not None and math.isfinite(float(row["z_score"]))
    ], dtype=np.float64)
    calibrated = next((row for row in diagnostics if row.get("calibrated")), None)
    return {
        "calibration_first_sample_index": (
            calibrated.get("source_sample_index") if calibrated is not None else None),
        "calibrated_block_count": sum(bool(row.get("calibrated")) for row in diagnostics),
        "finite_z_score_samples": int(finite_z.size),
        "max_z_score": float(np.max(finite_z)) if finite_z.size else None,
        "last_diagnostics": diagnostics[-1] if diagnostics else None,
    }


def _band_report(signal: np.ndarray, fs: int, start_s: float, stop_s: float) -> dict[str, Any]:
    start = int(round(start_s * fs))
    stop = int(round(stop_s * fs))
    segment = np.asarray(signal[start:stop], dtype=np.float64)
    if segment.size == 0 or not np.all(np.isfinite(segment)):
        return {"start_s": start_s, "stop_s": stop_s, "sample_count": int(segment.size),
                "rms_mV": None}
    sos = butter(4, (100.0, 250.0), btype="bandpass", fs=fs, output="sos")
    filtered = sosfilt(sos, np.asarray(signal, dtype=np.float64))
    window = filtered[start:stop]
    return {"start_s": start_s, "stop_s": stop_s, "sample_count": int(segment.size),
            "rms_mV": float(np.sqrt(np.mean(window * window)))}


def _first_nonzero_index(values: np.ndarray) -> int | None:
    nonzero = np.flatnonzero(np.asarray(values) != 0.0)
    return int(nonzero[0]) if nonzero.size else None


def _core_feature_block(core: app.ProcessingCore, before: Any,
                        sample_count: int) -> tuple[np.ndarray, bool, str]:
    """Read only a feature mask produced by this block; identify intentional skips."""
    controller = core.controllers.get("rule")
    if controller is None:
        raise RuntimeError("actual Core no longer owns the selected Rule controller")
    mask = np.asarray(controller.envelope.feature_valid_mask, dtype=bool).reshape(-1)
    if mask.size == sample_count:
        return mask.copy(), True, ""
    if mask.size != 0:
        raise RuntimeError(
            f"actual Rule feature mask has {mask.size} samples; expected 0 or {sample_count}")

    reasons = []
    if before.transition_pending:
        reasons.append("transition_pending")
    if not before.acquiring:
        reasons.append("not_acquiring")
    if before.mode != "closed-loop":
        reasons.append("not_closed_loop")
    if before.fault is not None:
        reasons.append("faulted")
    if not reasons:
        raise RuntimeError(
            "Core was eligible to step Rule but produced no feature mask; "
            f"diagnostics={core._last_diagnostics!r}")
    if core._last_diagnostics:
        raise RuntimeError("Core skipped feature computation but retained nonempty diagnostics")
    return np.zeros(sample_count, dtype=bool), False, "+".join(reasons)


def _h5_text_array(values: Any) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in np.asarray(values).reshape(-1)]


@dataclass
class ConditionResult:
    name: str
    recording_path: Path
    arrays_path: Path
    metrics: dict[str, Any]
    arrays: dict[str, np.ndarray]


def _run_core_condition(
    *,
    name: str,
    feedback_enabled: bool,
    output_dir: Path,
    duration_s: float,
    seed: int,
    rule_settings: dict[str, Any],
) -> ConditionResult:
    parameters = NeuralMassParameters()
    artifact_parameters = ElectrodeArtifactParameters(sample_rate_hz=parameters.sample_rate_hz)
    fs = parameters.sample_rate_hz
    chunk = 100
    count = int(round(duration_s * fs))
    if count % chunk:
        raise ValueError("duration must resolve to complete 100-sample processing chunks")
    if DISTURBANCE_START_S * fs < 11 * fs:
        raise ValueError("the activity disturbance must begin after the full 11-second calibration window")
    if count / fs <= DISTURBANCE_END_S:
        raise ValueError("duration must include the full activity disturbance and a recovery window")

    settings = {"rule": app.controller_settings_for("rule", rule_settings),
                "pid": app.controller_settings_for("pid")}
    record_dir = output_dir / "recordings"
    config = app.AppConfig(
        sample_rate=fs,
        processing_block_ms=5.0,
        ao_target_lead_ms=100.0,
        stimulation_recovery_ms=artifact_parameters.blanking_ms,
        electrode_labels=("CA3", "Cortex"),
        source_channel_regions=("CA3", "Cortex"),
        ctx_index=1,
        controller="rule",
        controller_settings=settings,
        record_dir=record_dir,
        open_browser=False,
    )
    config.validate()
    amp_limit = app.pulse_amplitude_limit_v(
        config.safety, fs,
        pulse_duration_ms=5.0, pulse_frequency_hz=100.0,
        refractory_s=0.5, max_rate_hz=2.0,
    )
    if feedback_enabled and PULSE_AMPLITUDE_V > amp_limit + 1e-12:
        raise ValueError(f"mock amplitude {PULSE_AMPLITUDE_V} V exceeds runtime pulse limit {amp_limit} V")

    state = app.RuntimeState(
        esn_ready=False, start_paused=False,
        electrode_labels=("CA3", "Cortex"), ctx_index=1,
        pulse_threshold_std=config.pulse_threshold_std,
        pulse_window_sec=config.pulse_window_sec,
        pulse_polarity=config.pulse_polarity,
        controller="rule",
        controller_ready={"esn": False, "rule": True, "pid": True},
        controller_settings=settings,
        pulse_max_amplitude_v=amp_limit,
        input_mapping_locked=True,
    )
    metadata = {
        "configuration": app.serializable_config(config),
        "source": {"kind": "synthetic_ei_neural_mass", "sample_rate_hz": fs,
                   "seed": int(seed), "source_samples": count},
        "mock_neural": {
            "model": "two-population reduced E/I neural-mass engineering fixture",
            "biological_calibration": False,
            "activity_units": "normalized deviations in [-1, 1]",
            "lfp_projection": "5 mV * (E_activity_deviation - I_activity_deviation)",
            "ao_state_effect_delay_ms": parameters.ao_effect_delay_ms,
            "electrode_artifact": asdict(artifact_parameters),
            "artifact_oracle_used_for_controller_input": False,
            "rule_settings": settings["rule"],
            "disturbance": {"start_s": DISTURBANCE_START_S,
                            "stop_s": DISTURBANCE_END_S, "drive": DISTURBED_DRIVE,
                            "baseline_drive": BASELINE_DRIVE},
        },
    }
    recorder = app.H5Recorder(config, metadata)
    recording_path = recorder.start(f"neural-mock-{name}")
    state.set_recording(True)
    loop = asyncio.new_event_loop()
    core = _HeadlessCore(config, state, _UnavailableEsn(), recorder,
                         app.UiHub(config.ui_queue_packets), loop, app.Telemetry())
    scheduler = app.SimulatedAoScheduler(
        config, state, recorder,
        ao_outcome=core.record_ao_outcome,
        pulse_outcome=core.enqueue_ao_outcome,
    )
    core.set_local_account_sink(scheduler.enqueue_local_account)
    # Leave the initial state visible to the scheduler so it owns the zero
    # drain transaction at the current absolute sample frontier.
    state.set_mode("closed-loop")
    if feedback_enabled:
        state.set_stim("threshold_pulse", PULSE_AMPLITUDE_V, max_gain=amp_limit)
    model = TwoPopulationNeuralMass(parameters, seed=seed)
    observation = SyntheticElectrodeObservation(artifact_parameters)
    clean_shadow = CausalEnvelope(fs, band_hz=(100.0, 250.0))

    arrays = {
        "sample_index": np.arange(count, dtype=np.uint64),
        "true_lfp_mV": np.empty(count, dtype=np.float32),
        "artifact_mV": np.empty(count, dtype=np.float32),
        "observed_ca3_mV": np.empty(count, dtype=np.float32),
        "cortex_reference_mV": np.zeros(count, dtype=np.float32),
        "generated_ao_V": np.empty(count, dtype=np.float32),
        "artifact_oracle_mask": np.empty(count, dtype=bool),
        "core_feature_valid": np.empty(count, dtype=bool),
        "clean_shadow_envelope_mV": np.full(count, np.nan, dtype=np.float32),
        "e_activity_deviation": np.empty(count, dtype=np.float32),
        "i_activity_deviation": np.empty(count, dtype=np.float32),
        "inhibitory_effect_state": np.empty(count, dtype=np.float32),
        "controller_source_block_start": np.arange(0, count, chunk, dtype=np.uint64),
        "controller_z_score": np.full(count // chunk, np.nan, dtype=np.float32),
        "controller_envelope_mV": np.full(count // chunk, np.nan, dtype=np.float32),
        "controller_baseline_mean_mV": np.full(count // chunk, np.nan, dtype=np.float32),
        "controller_baseline_std_mV": np.full(count // chunk, np.nan, dtype=np.float32),
        "controller_calibrated": np.zeros(count // chunk, dtype=bool),
        "controller_last_sample_feature_valid": np.zeros(count // chunk, dtype=bool),
        "controller_feature_computed": np.zeros(count // chunk, dtype=bool),
        "controller_feature_recovery_remaining_samples": np.zeros(count // chunk, dtype=np.uint32),
    }
    event_log: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    feature_skip_rows: list[dict[str, Any]] = []
    scheduler_generated_nonzero = 0
    drain_generated_nonzero = 0
    drain_committed_tail_blocks: list[dict[str, Any]] = []
    drain_unexplained_nonzero_blocks: list[dict[str, Any]] = []
    scheduler_faults: list[str] = []
    source_samples_processed = 0
    explicit_arm_sample: int | None = None
    first_ao_nonzero: int | None = None
    first_effect_nonzero: int | None = None
    submit_rejections = 0
    try:
        for first in range(0, count, chunk):
            generated = scheduler.advance()
            if generated.sample_index != first:
                raise RuntimeError(
                    f"virtual AO/source phase mismatch: AO={generated.sample_index}, source={first}")
            generated_ao = np.asarray(generated.waveform_v, dtype=np.float64).reshape(-1)
            nonzero = _first_nonzero_index(generated_ao)
            if nonzero is not None and first_ao_nonzero is None:
                first_ao_nonzero = first + nonzero
            scheduler_generated_nonzero += int(np.count_nonzero(generated_ao))

            drive = _disturbance(first, fs, chunk)
            latent, states_out = model.step_block(first, drive, generated_ao)
            observed = observation.observe(first, latent, generated_ao)
            arrays["true_lfp_mV"][first:first + chunk] = latent
            arrays["artifact_mV"][first:first + chunk] = observed["artifact_mV"]
            arrays["observed_ca3_mV"][first:first + chunk] = observed["electrode_mV"]
            arrays["generated_ao_V"][first:first + chunk] = generated_ao
            arrays["artifact_oracle_mask"][first:first + chunk] = observed["artifact_oracle_mask"]
            for key in ("e_activity_deviation", "i_activity_deviation", "inhibitory_effect_state"):
                arrays[key][first:first + chunk] = states_out[key]
            clean_env = clean_shadow.process(
                latent, first, np.ones(chunk, dtype=bool), np.zeros(chunk, dtype=bool))
            arrays["clean_shadow_envelope_mV"][first:first + chunk] = clean_env

            cortex = np.zeros(chunk, dtype=np.float64)
            input_mv = np.vstack((observed["electrode_mV"], cortex))
            pre_core_snapshot = state.snapshot()
            _, safe, _esn_ms, safety_reason = core.process(
                input_mv,
                sample_index=first,
                ao_target_sample_index=first + config.ao_lead_samples,
                input_is_electrode_mV=True,
                input_roles=("CA3", "Cortex"),
                input_labels=("CA3", "Cortex"),
                input_ctx_index=1,
            )
            source_samples_processed += chunk
            controller = core.controllers["rule"]
            feature_valid, feature_computed, skip_reason = _core_feature_block(
                core, pre_core_snapshot, chunk)
            arrays["core_feature_valid"][first:first + chunk] = feature_valid
            diag = dict(core._last_diagnostics)
            diag["source_sample_index"] = first
            diag["feature_computed"] = feature_computed
            diag["feature_skip_reason"] = skip_reason
            diagnostic_rows.append(diag)
            block_index = first // chunk
            arrays["controller_feature_computed"][block_index] = feature_computed
            if not feature_computed:
                feature_skip_rows.append({"source_sample_index": first,
                                          "reason": skip_reason})
            for array_name, diagnostic_name in (
                ("controller_z_score", "z_score"),
                ("controller_envelope_mV", "envelope_mv"),
                ("controller_baseline_mean_mV", "baseline_mean_mv"),
                ("controller_baseline_std_mV", "baseline_std_mv"),
            ):
                value = diag.get(diagnostic_name)
                if value is not None and math.isfinite(float(value)):
                    arrays[array_name][block_index] = float(value)
            arrays["controller_calibrated"][block_index] = bool(diag.get("calibrated", False))
            arrays["controller_last_sample_feature_valid"][block_index] = bool(feature_valid[-1])
            arrays["controller_feature_recovery_remaining_samples"][block_index] = int(
                diag.get("feature_recovery_remaining_samples", 0) or 0)
            if safety_reason:
                scheduler_faults.append(f"safety: {safety_reason}")
            for event in core.last_pulse_events:
                row = asdict(event)
                event_log.append(row)

            snapshot = state.snapshot()
            target_block = first // chunk + config.ao_lead_chunks
            if not scheduler.submit(
                target_block, safe[0], core.last_processed_generation,
                core.last_pulse_events, core.last_admitted_pulse_plans,
            ):
                submit_rejections += 1

            if (feedback_enabled and not explicit_arm_sample and snapshot.controller_calibrated
                    and not snapshot.transition_pending and snapshot.fault is None):
                state.set_armed(True)
                explicit_arm_sample = first + chunk
            snapshot = state.snapshot()
            if snapshot.fault:
                scheduler_faults.append(f"runtime: {snapshot.fault}")
                break

        before_drain = state.snapshot()
        # Finite-source EOF is itself a transaction boundary. Start a real
        # recoverable drain even when the controller was already disarmed;
        # otherwise buffered zero commands can remain queued without a barrier.
        state.begin_recoverable_drain()
        with core._actuator_lock:
            core.actuator.disarm()
        drain_steps = 0
        for _ in range(config.ao_lead_chunks + 8):
            snapshot = state.snapshot()
            if (not snapshot.transition_pending
                    and scheduler.drain_barrier.pending_generation is None
                    and scheduler.stats_snapshot()["queue_depth"] == 0):
                break
            drained = scheduler.advance()
            drain_steps += 1
            if np.any(drained.waveform_v):
                block = {
                    "sample_index": int(drained.sample_index),
                    "generation": int(drained.generation),
                    "stage": str(drained.stage),
                    "nonzero_samples": int(np.count_nonzero(drained.waveform_v)),
                }
                drain_generated_nonzero += block["nonzero_samples"]
                if drained.stage in {"ao_committed", "tail_preserved"}:
                    drain_committed_tail_blocks.append(block)
                else:
                    drain_unexplained_nonzero_blocks.append(block)
                    scheduler_faults.append(
                        "nonzero AO during drain lacked committed-command provenance")
        core.drain_ao_notifications()
        after_drain = state.snapshot()
        scheduler_stats = scheduler.stats_snapshot()
        state.set_recording(False)
        recorder.stop_recording()
        recorder.close()
        if loop.is_running():
            raise RuntimeError("headless event loop unexpectedly started")
        loop.close()
    except BaseException:
        state.set_recording(False)
        try:
            core.drain_ao_notifications()
        except Exception:
            pass
        try:
            recorder.close()
        except Exception:
            pass
        if not loop.is_closed():
            loop.close()
        raise

    for key, values in arrays.items():
        if key.startswith("controller_"):
            arrays[key] = values[:source_samples_processed // chunk]
            continue
        if values.shape[0] != source_samples_processed:
            values = values[:source_samples_processed]
            arrays[key] = values
    arrays_path = output_dir / f"{name}-model-trace.npz"
    np.savez_compressed(arrays_path, **arrays)

    h5_input_sample_mismatches = 0
    h5_diagnostic_offset_mismatches = 0
    h5_diagnostic_count_mismatches = 0
    h5_diagnostic_rows = 0
    h5_diagnostic_source_offsets_complete = False
    try:
        with h5py.File(recording_path, "r") as recorded:
            h5_samples = int(recorded.attrs["committed_samples"])
            h5_complete = bool(recorded.attrs.get("complete", 0))
            electrode_dataset = recorded["signals/electrode_mV"]
            h5_electrode_samples = int(electrode_dataset.shape[1])
            h5_ao_samples = int(recorded["signals/ao_command_V"].shape[1])
            h5_channel_labels = _h5_text_array(electrode_dataset.attrs["channel_labels"])
            h5_channel_roles = _h5_text_array(electrode_dataset.attrs["channel_roles"])
            h5_channel_regions = _h5_text_array(electrode_dataset.attrs["channel_regions"])
            h5_electrode_unit = str(electrode_dataset.attrs["unit"])
            cortex_expected = np.zeros(100_000, dtype=np.float32)
            for start in range(0, min(h5_electrode_samples, source_samples_processed), 100_000):
                stop = min(start + 100_000, h5_electrode_samples, source_samples_processed)
                expected_ca3 = arrays["observed_ca3_mV"][start:stop]
                recorded_ca3 = electrode_dataset[0, start:stop]
                recorded_cortex = electrode_dataset[1, start:stop]
                if not np.array_equal(recorded_ca3, expected_ca3):
                    h5_input_sample_mismatches += int(np.count_nonzero(recorded_ca3 != expected_ca3))
                expected_cortex = cortex_expected[:stop - start]
                if not np.array_equal(recorded_cortex, expected_cortex):
                    h5_input_sample_mismatches += int(np.count_nonzero(recorded_cortex != expected_cortex))

            block_rows = recorded["diagnostics/blocks"]
            h5_diagnostic_rows = int(block_rows.shape[0])
            expected_block_count = source_samples_processed // chunk
            previous_end = 0
            for start in range(0, h5_diagnostic_rows, 4096):
                rows = block_rows[start:start + 4096]
                for row in rows:
                    offset = int(row["sample_offset"])
                    row_count = int(row["sample_count"])
                    if offset != previous_end:
                        h5_diagnostic_offset_mismatches += 1
                    if row_count != chunk:
                        h5_diagnostic_count_mismatches += 1
                    previous_end = offset + row_count
            h5_diagnostic_source_offsets_complete = bool(
                h5_diagnostic_rows == expected_block_count
                and previous_end == source_samples_processed
                and h5_diagnostic_offset_mismatches == 0
                and h5_diagnostic_count_mismatches == 0)
    except Exception as exc:
        h5_samples, h5_complete, h5_electrode_samples, h5_ao_samples = 0, False, 0, 0
        h5_channel_labels, h5_channel_roles, h5_channel_regions, h5_electrode_unit = [], [], [], None
        h5_input_sample_mismatches = 1
        h5_diagnostic_offset_mismatches = 1
        h5_diagnostic_count_mismatches = 1
        scheduler_faults.append(f"H5 verification: {type(exc).__name__}: {exc}")

    generated_trace = arrays["generated_ao_V"].astype(np.float64)
    actual_effect = arrays["inhibitory_effect_state"].astype(np.float64)
    first_effect = _first_nonzero_index(actual_effect)
    first_effect_nonzero = first_effect
    actual_stimulated = np.flatnonzero(arrays["artifact_oracle_mask"])
    actual_invalid = np.flatnonzero(~arrays["core_feature_valid"])
    unmasked_artifact_samples = int(np.count_nonzero(
        arrays["artifact_oracle_mask"] & arrays["core_feature_valid"]))
    recovery_checks: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    pulse_samples = round(5.0 * fs / 1_000.0)
    blank_samples = round(artifact_parameters.blanking_ms * fs / 1_000.0)
    for event in event_log:
        if event.get("status") != "rendered_for_ao" or event.get("pulse_id") in seen_event_ids:
            continue
        seen_event_ids.add(str(event.get("pulse_id")))
        target = int(event["target_onset_sample_index"])
        expected_mask_end = target + pulse_samples + blank_samples
        valid_after_mask = np.flatnonzero(arrays["core_feature_valid"][expected_mask_end:])
        first_valid = (expected_mask_end + int(valid_after_mask[0])
                       if valid_after_mask.size else None)
        comparison: dict[str, Any] = {
            "pulse_id": event.get("pulse_id"),
            "target_onset_sample_index": target,
            "artifact_oracle_end_exclusive": expected_mask_end,
            "expected_first_valid_sample_index": expected_mask_end + int(
                core.controllers["rule"].envelope.feature_recovery_samples),
            "observed_first_core_feature_valid_sample_index": first_valid,
            "core_mask_covers_oracle_interval": bool(
                not np.any(arrays["core_feature_valid"][target:expected_mask_end])),
        }
        if first_valid is not None:
            block_index = (first_valid // chunk)
            block_end_index = min((block_index + 1) * chunk - 1,
                                  source_samples_processed - 1)
            actual_env = float(arrays["controller_envelope_mV"][block_index])
            clean_env = float(arrays["clean_shadow_envelope_mV"][block_end_index])
            comparison["first_valid_block_end_sample_index"] = block_end_index
            comparison["clean_shadow_envelope_mV_at_block_end"] = clean_env
            comparison["core_envelope_mV_at_block_end"] = (
                actual_env if math.isfinite(actual_env) else None)
            comparison["absolute_envelope_difference_mV"] = (
                abs(actual_env - clean_env)
                if math.isfinite(actual_env) and math.isfinite(clean_env) else None)
        recovery_checks.append(comparison)
    metrics = {
        "condition": name,
        "feedback_enabled": feedback_enabled,
        "runtime_controller": "rule",
        "rule_settings": settings["rule"],
        "sample_rate_hz": fs,
        "chunk_size": chunk,
        "duration_s_requested": duration_s,
        "source_samples_processed": source_samples_processed,
        "recorded_samples": h5_samples,
        "h5_complete": h5_complete,
        "h5_electrode_samples": h5_electrode_samples,
        "h5_ao_command_samples": h5_ao_samples,
        "h5_electrode_channel_labels": h5_channel_labels,
        "h5_electrode_channel_roles": h5_channel_roles,
        "h5_electrode_channel_regions": h5_channel_regions,
        "h5_electrode_unit": h5_electrode_unit,
        "h5_input_sample_mismatches": h5_input_sample_mismatches,
        "h5_diagnostic_rows": h5_diagnostic_rows,
        "h5_diagnostic_offset_mismatches": h5_diagnostic_offset_mismatches,
        "h5_diagnostic_count_mismatches": h5_diagnostic_count_mismatches,
        "h5_diagnostic_source_offsets_complete": h5_diagnostic_source_offsets_complete,
        "complete_finite_h5_input_coverage": bool(
            h5_complete and h5_samples == source_samples_processed
            and h5_electrode_samples == source_samples_processed
            and h5_input_sample_mismatches == 0
            and h5_diagnostic_source_offsets_complete
            and h5_channel_labels == ["CA3", "Cortex"]
            and h5_channel_roles == ["CA3", "Cortex"]
            and h5_channel_regions == ["CA3", "Cortex"]
            and h5_electrode_unit == "mV"),
        "explicit_arm_source_sample_index": explicit_arm_sample,
        "rule_calibration": _controller_metrics(diagnostic_rows),
        "controller_feature_skip_blocks": feature_skip_rows,
        "rule_decision_events": event_log,
        "first_rule_decision_source_sample_index": (
            event_log[0].get("source_onset_sample_index") if event_log else None),
        "first_rule_latest_sample_index": event_log[0].get("latest_sample_index") if event_log else None,
        "first_rule_target_onset_sample_index": (
            event_log[0].get("target_onset_sample_index") if event_log else None),
        "first_generated_nonzero_ao_sample_index": first_ao_nonzero,
        "first_neural_effect_sample_index": first_effect_nonzero,
        "configured_neural_effect_delay_samples": round(
            parameters.ao_effect_delay_ms * fs / 1_000.0),
        "neural_effect_delay_from_first_generated_nonzero_samples": (
            first_effect_nonzero - first_ao_nonzero
            if first_effect_nonzero is not None and first_ao_nonzero is not None else None),
        "scheduler": scheduler_stats,
        "scheduler_generated_nonzero_samples": scheduler_generated_nonzero,
        "drain_generated_nonzero_samples": drain_generated_nonzero,
        "drain_committed_tail_blocks": drain_committed_tail_blocks,
        "drain_unexplained_nonzero_blocks": drain_unexplained_nonzero_blocks,
        "scheduler_submit_rejections": submit_rejections,
        "transition_pending_after_drain": after_drain.transition_pending,
        "drain_steps": drain_steps,
        "drain_barrier_pending_after_drain": scheduler_stats.get("barrier_generation") is not None,
        "fault": after_drain.fault,
        "scheduler_faults": scheduler_faults,
        "artifact_mask_sample_count": int(actual_stimulated.size),
        "core_feature_invalid_sample_count": int(actual_invalid.size),
        "core_valid_inside_artifact_mask_samples": unmasked_artifact_samples,
        "core_mask_covers_artifact_oracle": unmasked_artifact_samples == 0,
        "feature_recovery_samples": int(core.controllers["rule"].envelope.feature_recovery_samples),
        "feature_recovery_comparison_to_clean_shadow": recovery_checks,
        "artifact_oracle_supplied_to_core": False,
        "recording_path": str(recording_path),
        "trace_path": str(arrays_path),
        "band_rms": {
            "baseline_2_to_10s": _band_report(arrays["true_lfp_mV"], fs, 2.0, 10.0),
            "disturbance_12_to_18s": _band_report(arrays["true_lfp_mV"], fs, 12.0, 18.0),
            "recovery_18_to_end": _band_report(arrays["true_lfp_mV"], fs, 18.0, duration_s),
        },
        "model_state_bounds": {
            "max_abs_e": float(np.max(np.abs(arrays["e_activity_deviation"]))),
            "max_abs_i": float(np.max(np.abs(arrays["i_activity_deviation"]))),
            "max_inhibitory_effect": float(np.max(arrays["inhibitory_effect_state"])),
        },
    }
    return ConditionResult(name, recording_path, arrays_path, metrics, arrays)


def _compare_pair(off: ConditionResult, rule: ConditionResult) -> dict[str, Any]:
    length = min(off.arrays["true_lfp_mV"].size, rule.arrays["true_lfp_mV"].size)
    if length == 0:
        divergence = None
    else:
        different = np.flatnonzero(
            off.arrays["true_lfp_mV"][:length] != rule.arrays["true_lfp_mV"][:length])
        divergence = int(different[0]) if different.size else None
    generated = np.flatnonzero(rule.arrays["generated_ao_V"] != 0.0)
    observed_changed = np.flatnonzero(
        off.arrays["observed_ca3_mV"][:length] != rule.arrays["observed_ca3_mV"][:length])
    block_count = min(off.arrays["controller_z_score"].size,
                      rule.arrays["controller_z_score"].size)
    off_z = off.arrays["controller_z_score"][:block_count]
    rule_z = rule.arrays["controller_z_score"][:block_count]
    comparable_z = np.isfinite(off_z) & np.isfinite(rule_z)
    z_difference = np.flatnonzero(comparable_z & (np.abs(off_z - rule_z) > 1e-6))
    first_z_divergence_block_start = (
        int(rule.arrays["controller_source_block_start"][z_difference[0]])
        if z_difference.size else None)
    first_z_divergence_latest_used = (
        first_z_divergence_block_start + int(rule.metrics["chunk_size"]) - 1
        if first_z_divergence_block_start is not None else None)
    first_effect = rule.metrics.get("first_neural_effect_sample_index")
    # The effect index is already the first delayed model response; do not add
    # the configured actuator-to-model delay a second time.
    response_floor = int(first_effect) if first_effect is not None else None
    block_starts = rule.arrays["controller_source_block_start"][:block_count]
    response_valid = (comparable_z
                      & rule.arrays["controller_last_sample_feature_valid"][:block_count]
                      & off.arrays["controller_last_sample_feature_valid"][:block_count])
    if response_floor is not None:
        response_valid &= block_starts >= response_floor
    response_difference = np.flatnonzero(
        response_valid & (np.abs(off_z - rule_z) > 1e-6))
    first_later_valid_measurement_change = (
        int(block_starts[response_difference[0]] + rule.metrics["chunk_size"] - 1)
        if response_difference.size else None)
    no_latent_divergence_before_effect = bool(
        first_effect is None or divergence is None or divergence >= int(first_effect))
    return {
        "same_seed_and_disturbance": True,
        "paired_source_samples": int(length),
        "first_latent_lfp_divergence_sample_index": divergence,
        "first_generated_ao_nonzero_sample_index": int(generated[0]) if generated.size else None,
        "first_observed_ca3_divergence_sample_index": (
            int(observed_changed[0]) if observed_changed.size else None),
        "first_finite_rule_measurement_divergence_block_start_sample_index": (
            first_z_divergence_block_start),
        "first_finite_rule_measurement_latest_used_sample_index": first_z_divergence_latest_used,
        "first_later_valid_rule_measurement_change_sample_index": first_later_valid_measurement_change,
        "no_latent_divergence_before_neural_effect": no_latent_divergence_before_effect,
        "latent_divergence_after_neural_effect": bool(
            divergence is not None and first_effect is not None
            and divergence >= int(first_effect)),
        "latent_divergence_after_generated_ao": bool(
            divergence is not None and generated.size and divergence >= int(generated[0])),
        "controller_measurement_divergence_after_generated_ao": bool(
            first_z_divergence_latest_used is not None and generated.size
            and first_z_divergence_latest_used >= int(generated[0])),
        "rule_generated_nonzero_samples": int(generated.size),
        "off_rule_h5_input_sample_counts_match": (
            off.metrics["h5_electrode_samples"] == rule.metrics["h5_electrode_samples"]),
        "virtual_clock_only": True,
        "physical_latency_claim": False,
    }


def run_pair(output_dir: Path, *, duration_s: float = DEFAULT_DURATION_S,
             seed: int = 7, upper_z: float = 1.0, lower_z: float = 0.5) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    rule_settings = {"upper_z": upper_z, "lower_z": lower_z,
                     "band_hz": (100.0, 250.0)}
    off = _run_core_condition(name="off", feedback_enabled=False,
                              output_dir=output_dir, duration_s=duration_s,
                              seed=seed, rule_settings=rule_settings)
    closed = _run_core_condition(name="rule-feedback", feedback_enabled=True,
                                 output_dir=output_dir, duration_s=duration_s,
                                 seed=seed, rule_settings=rule_settings)
    report = {
        "schema": "echo-chamber-mock-neural-demo-v1",
        "status": "diagnostic only; no biological calibration or physical-output claim",
        "model": asdict(NeuralMassParameters()),
        "artifact": asdict(ElectrodeArtifactParameters()),
        "timeline_s": {"baseline_and_calibration": [0.0, 12.0],
                       "activity_disturbance": [12.0, 18.0],
                       "recovery": [18.0, duration_s]},
        "rule_engineering_preset": rule_settings,
        "off": off.metrics,
        "rule_feedback": closed.metrics,
        "paired_comparison": _compare_pair(off, closed),
    }
    first_rule_output = report["paired_comparison"]["first_generated_ao_nonzero_sample_index"]
    first_effect = closed.metrics["first_neural_effect_sample_index"]
    delay_samples = closed.metrics["configured_neural_effect_delay_samples"]
    proposal_count = sum(
        event.get("status") == "rendered_for_ao"
        for event in closed.metrics["rule_decision_events"])
    gates = {
        "paired_conditions_have_equal_complete_input": bool(
            off.metrics["complete_finite_h5_input_coverage"]
            and closed.metrics["complete_finite_h5_input_coverage"]
            and off.metrics["source_samples_processed"] == closed.metrics["source_samples_processed"]
            and off.metrics["source_samples_processed"] == round(duration_s * 20_000)),
        "off_condition_has_no_generated_nonzero_ao": (
            off.metrics["scheduler_generated_nonzero_samples"] == 0
            and off.metrics["drain_generated_nonzero_samples"] == 0),
        "rule_proposed_a_pulse": proposal_count > 0,
        "rule_pulse_was_committed": closed.metrics["scheduler"].get(
            "pulse_committed_samples", 0) >= 100,
        "rule_pulse_was_generated": bool(
            first_rule_output is not None
            and closed.metrics["scheduler_generated_nonzero_samples"] > 0),
        "neural_effect_followed_configured_delay": bool(
            first_rule_output is not None and first_effect is not None
            and int(first_effect) - int(first_rule_output) >= max(400, int(delay_samples))),
        "paired_latent_response_starts_after_neural_effect": bool(
            report["paired_comparison"]["no_latent_divergence_before_neural_effect"]
            and report["paired_comparison"]["latent_divergence_after_neural_effect"]),
        "later_valid_controller_measurement_changed": (
            report["paired_comparison"][
                "first_later_valid_rule_measurement_change_sample_index"] is not None),
        "core_mask_covers_artifact_oracle": (
            closed.metrics["artifact_mask_sample_count"] > 0
            and closed.metrics["core_mask_covers_artifact_oracle"]),
        "drain_completed_without_new_or_unexplained_output": bool(
            off.metrics["fault"] is None and closed.metrics["fault"] is None
            and not off.metrics["transition_pending_after_drain"]
            and not closed.metrics["transition_pending_after_drain"]
            and not off.metrics["drain_unexplained_nonzero_blocks"]
            and not closed.metrics["drain_unexplained_nonzero_blocks"]
            and off.metrics["scheduler"].get("queue_depth", 0) == 0
            and closed.metrics["scheduler"].get("queue_depth", 0) == 0
            and not off.metrics["scheduler_faults"]
            and not closed.metrics["scheduler_faults"]),
    }
    report["closure_gates"] = gates
    report["closure_demonstrated"] = all(gates.values())
    report["status"] = (
        "diagnostic only; circuit-closure gates passed, no biological calibration or physical-output claim"
        if report["closure_demonstrated"] else
        "diagnostic only; one or more circuit-closure gates failed; no biological calibration or physical-output claim")
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(_native(report), indent=2, allow_nan=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new output directory for H5 recordings, NPZ traces, and report")
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--upper-z", type=float, default=1.0,
                        help="mock Rule threshold, an engineering fixture parameter")
    parser.add_argument("--lower-z", type=float, default=0.5,
                        help="mock Rule rearm threshold, an engineering fixture parameter")
    args = parser.parse_args(argv)
    try:
        result = run_pair(args.output_dir.resolve(), duration_s=args.duration_s,
                          seed=args.seed, upper_z=args.upper_z, lower_z=args.lower_z)
    except Exception as exc:
        print(f"mock run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_native({
        "report_path": str((args.output_dir.resolve() / "report.json")),
        "closure_demonstrated": result["closure_demonstrated"],
        "closure_gates": result["closure_gates"],
        "off_recording": result["off"]["recording_path"],
        "rule_recording": result["rule_feedback"]["recording_path"],
        "paired_comparison": result["paired_comparison"],
        "faults": {"off": result["off"]["fault"],
                   "rule_feedback": result["rule_feedback"]["fault"]},
    }), indent=2, allow_nan=False))
    return 0 if result["closure_demonstrated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
