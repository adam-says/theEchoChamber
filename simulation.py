"""Deterministic generic responsive plant for software-path validation only."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import threading

import echoChamber as app
from benchmarkControllers import (
    _Harness, _make_config, file_sha256, git_revision, write_progress,
)


@dataclass
class PlantSettings:
    sample_rate_hz: int = 20_000
    block_samples: int = 100
    carrier_hz: float = 150.0
    baseline_amplitude_mv: float = 0.12
    disturbance_amplitude_mv: float = 0.18
    disturbance_start_s: float = 12.0
    disturbance_stop_s: float = 20.0
    noise_sd_mv: float = 0.004
    effect_delay_ms: float = 20.0
    effect_recovery_s: float = 0.50
    amplitude_reduction_per_pulse_mv: float = 0.035
    max_amplitude_reduction_mv: float = 0.10
    seed: int = 1414


TRACKING_PLANT_SETTINGS = PlantSettings(
    # This is a rate-limited example, not a guarantee of tracking. With the
    # 181.9 ms post-mask feature guard, 100 ms lead, 5 ms pulse and 0.5 s
    # actuator refractory, the ideal cadence is about 1.27 Hz, not 2 Hz.
    # At tau=5 s that cadence has about 0.00318 mV mean capacity before
    # target margin; the run reports the actual generated cadence separately.
    disturbance_amplitude_mv=0.003,
    disturbance_start_s=12.0,
    disturbance_stop_s=42.0,
    effect_delay_ms=20.0,
    effect_recovery_s=5.0,
    amplitude_reduction_per_pulse_mv=0.0005,
    max_amplitude_reduction_mv=0.020,
    seed=1414,
)
TRACKING_PID_SETTINGS = {"kp": 0.2, "ki": 0.06, "kd": 0.0, "target_z": 1.0,
                         "direction": 1, "max_rate_hz": 2.0, "update_ms": 5.0,
                         "derivative_tau_s": 0.020, "band_hz": (100.0, 250.0)}
TRACKING_ERROR_BAND_Z = 0.5
TRACKING_SETTLING_WINDOW_S = 1.0
TRACKING_MIN_MOVING_WINDOW_ELIGIBLE_COVERAGE = 0.95
# Z-score calibration has unit baseline standard deviation; the 2 SD ripple
# ceiling allows measurement noise while rejecting large sustained oscillation.
TRACKING_RAW_ERROR_STD_LIMIT_Z = 2.0


def _steady_pulse_capacity(settings: PlantSettings, rate_hz: float) -> tuple[float, float]:
    if rate_hz <= 0 or settings.effect_recovery_s <= 0:
        return 0.0, 0.0
    period_s = 1.0 / rate_hz
    recovery_fraction = float(np.exp(-period_s / settings.effect_recovery_s))
    uncapped_mean = settings.amplitude_reduction_per_pulse_mv * rate_hz * settings.effect_recovery_s
    uncapped_peak = settings.amplitude_reduction_per_pulse_mv / max(1e-12, 1.0 - recovery_fraction)
    peak = min(settings.max_amplitude_reduction_mv, uncapped_peak)
    mean = min(uncapped_mean, peak * settings.effect_recovery_s / period_s * (1.0 - recovery_fraction))
    return float(mean), float(peak)


def plant_rate_capacity(settings: PlantSettings, max_rate_hz: float, *,
                        measured_generated_rate_hz: float | None = None,
                        target_correction_mv: float | None = None,
                        effective_max_rate_hz: float | None = None) -> dict[str, float | bool | None]:
    """Report requested-rate capacity separately from generated-command capacity."""
    mean_capacity, post_pulse_capacity = _steady_pulse_capacity(settings, max_rate_hz)
    disturbance = float(settings.disturbance_amplitude_mv)
    measured_mean = measured_post = measured_margin = measured_reaches = None
    if measured_generated_rate_hz is not None:
        measured_mean, measured_post = _steady_pulse_capacity(settings, measured_generated_rate_hz)
        measured_margin = (measured_mean - target_correction_mv
                           if target_correction_mv is not None else None)
        measured_reaches = (measured_margin >= 0.0) if measured_margin is not None else None
    return {
        "requested_rate_limit_hz": float(max_rate_hz),
        "requested_rate_mean_capacity_mv": mean_capacity,
        "requested_rate_post_pulse_capacity_mv": post_pulse_capacity,
        "effective_rate_bound_hz": effective_max_rate_hz,
        "effective_rate_mean_capacity_mv": (
            _steady_pulse_capacity(settings, effective_max_rate_hz)[0]
            if effective_max_rate_hz is not None else None),
        "effective_rate_post_pulse_capacity_mv": (
            _steady_pulse_capacity(settings, effective_max_rate_hz)[1]
            if effective_max_rate_hz is not None else None),
        "disturbance_amplitude_mv": disturbance,
        "requested_rate_capacity_margin_mv": float(mean_capacity - disturbance),
        "requested_rate_reaches_full_disturbance": bool(mean_capacity >= disturbance),
        "measured_generated_rate_hz": measured_generated_rate_hz,
        "measured_generated_mean_capacity_mv": measured_mean,
        "measured_generated_post_pulse_capacity_mv": measured_post,
        "target_correction_required_mv": target_correction_mv,
        "measured_generated_target_margin_mv": measured_margin,
        "measured_generated_rate_reaches_target": measured_reaches,
    }


def _effective_rate_bound_hz(lead_ms: float, feature_diagnostics: list[dict[str, Any]],
                             settings: PlantSettings) -> tuple[float, dict[str, float]]:
    guard_samples = max((int(item.get("feature_recovery_samples", 0) or 0)
                         for item in feature_diagnostics), default=0)
    guard_s = guard_samples / settings.sample_rate_hz
    pulse_s = 0.005
    refractory_s = 0.5
    parts = {
        "ao_lead_ms": float(lead_ms), "pulse_width_ms": pulse_s * 1000.0,
        "feature_recovery_samples": float(guard_samples),
        "feature_recovery_s": guard_s, "actuator_refractory_s": refractory_s,
    }
    return 1.0 / max(1e-12, lead_ms / 1000.0 + pulse_s + guard_s + refractory_s), parts


def _summarize_capacity_reference(plant: ResponsiveSyntheticPlant, metrics: Any, *,
                                  duration_s: float, lead_ms: float,
                                  target_z: float, target_correction_mv: float | None,
                                  max_rate_hz: float, feature_diagnostics: list[dict[str, Any]],
                                  scheduler_stats: dict[str, Any]) -> dict[str, Any]:
    settings = plant.settings
    block_s = settings.block_samples / settings.sample_rate_hz
    steady_start = settings.disturbance_start_s + 2.0 * settings.effect_recovery_s
    steady_stop = settings.disturbance_stop_s
    sample_times = (np.arange(len(plant.reduction_history_mv), dtype=np.float64) + 0.5) * block_s
    reductions = np.asarray(plant.reduction_history_mv, dtype=np.float64)
    steady_mask = (sample_times >= steady_start) & (sample_times < steady_stop)
    steady_values = reductions[steady_mask]
    steady_mean = float(np.mean(steady_values)) if steady_values.size else float("nan")
    generated_effects = [index for index in plant.command_effect_sample_indices
                         if steady_start * settings.sample_rate_hz <= index < steady_stop * settings.sample_rate_hz]
    window_seconds = max(0.0, steady_stop - steady_start)
    measured_rate = len(generated_effects) / window_seconds if window_seconds else 0.0
    expected_samples = int(np.ceil(duration_s * settings.sample_rate_hz / settings.block_samples)) * settings.block_samples
    metric = metrics.get if isinstance(metrics, dict) else lambda name, default=0: getattr(metrics, name, default)
    source_samples = int(metric("source_samples"))
    source_coverage = source_samples / expected_samples if expected_samples else 0.0
    source_gaps = int(metric("source_index_gap_samples"))
    skipped_samples = int(metric("skipped_source_samples"))
    full_source = bool(source_coverage >= 1.0 - 1e-12 and source_gaps == 0 and skipped_samples == 0)
    expected_window_samples = int(round(window_seconds * settings.sample_rate_hz))
    observed_window_samples = int(steady_values.size * settings.block_samples)
    window_complete = bool(window_seconds > 0 and steady_values.size == round(window_seconds / block_s))
    bound, bound_basis = _effective_rate_bound_hz(lead_ms, feature_diagnostics, settings)
    max_requested_observed = float(metric("max_requested_rate_hz"))
    max_request_limit_reached = max_requested_observed >= max_rate_hz - 1e-9
    theoretical = plant_rate_capacity(
        settings, max_rate_hz, effective_max_rate_hz=bound,
        measured_generated_rate_hz=measured_rate, target_correction_mv=target_correction_mv,
    )
    margin = steady_mean - target_correction_mv if np.isfinite(steady_mean) else None
    return {
        "reference_kind": "actual_processing_core_max_request",
        "controller": "pid",
        "pid_reference_settings": {"kp": 1.0, "ki": 0.0, "kd": 0.0,
                                    "target_z": -1000.0, "max_rate_hz": max_rate_hz},
        "lead_ms": float(lead_ms), "effective_rate_bound_hz": bound,
        "effective_rate_bound_basis": bound_basis,
        "requested_rate_limit_hz": float(max_rate_hz),
        "target_z_for_capacity": float(target_z),
        "target_correction_required_mv": target_correction_mv,
        "steady_window_start_s": float(steady_start),
        "steady_window_stop_s": float(steady_stop),
        "steady_window_complete": window_complete,
        "steady_window_expected_samples": expected_window_samples,
        "steady_window_observed_samples": observed_window_samples,
        "steady_mean_reduction_mv": steady_mean,
        "steady_reduction_margin_mv": margin,
        "generated_effect_pulses_in_steady_window": len(generated_effects),
        "generated_effect_rate_hz_in_steady_window": float(measured_rate),
        "requested_rate_mean_capacity_mv": theoretical["requested_rate_mean_capacity_mv"],
        "effective_rate_mean_capacity_mv": theoretical["effective_rate_mean_capacity_mv"],
        "generated_cadence_formula_mean_capacity_mv": theoretical["measured_generated_mean_capacity_mv"],
        "source_coverage_fraction": float(source_coverage),
        "source_gap_samples": source_gaps,
        "skipped_source_samples": skipped_samples,
        "full_source_coverage": full_source,
        "max_requested_rate_observed_hz": max_requested_observed,
        "max_request_limit_reached": bool(max_request_limit_reached),
        "scheduler": scheduler_stats,
        "capacity_reference_reaches_target": bool(
            full_source and window_complete and max_request_limit_reached
            and margin is not None and margin >= 0.0),
        "physical_pulse_claims": False,
    }


class ResponsiveSyntheticPlant:
    """A delayed, generic inhibitory response driven by generated AO commands."""

    def __init__(self, settings: PlantSettings | None = None):
        self.settings = settings or PlantSettings()
        self.fs = self.settings.sample_rate_hz
        self.rng = np.random.default_rng(self.settings.seed)
        self.effect_delay_samples = round(self.settings.effect_delay_ms * self.fs / 1000)
        self.pending_effects: dict[int, int] = {}
        self.reduction_mv = 0.0
        self.last_sample_index: int | None = None
        self.last_generated_active = False
        self.generated_nonzero_samples = 0
        self.rejected_or_zero_blocks = 0
        self.command_effect_sample_count = 0
        self.command_effect_pulse_count = 0
        self.command_effect_sample_indices: list[int] = []
        self.observed_rms_mv: list[float] = []
        self.reduction_history_mv: list[float] = []
        self.last_reduction_samples_mv = np.zeros(0, dtype=np.float64)
        self._lock = threading.RLock()

    def observe_generated_ao(self, generated: app.GeneratedAoBlock) -> None:
        """Queue only output the AO scheduler actually generated."""
        with self._lock:
            self._observe_generated_ao(generated)

    def _observe_generated_ao(self, generated: app.GeneratedAoBlock) -> None:
        waveform = np.asarray(generated.waveform_v, dtype=np.float64).reshape(-1)
        self.generated_nonzero_samples += int(np.count_nonzero(waveform))
        accepted = generated.stage in {"ao_committed", "tail_preserved"}
        if not accepted or not np.any(waveform):
            self.rejected_or_zero_blocks += 1
        active = np.abs(waveform) >= 0.04 if accepted else np.zeros(waveform.size, dtype=bool)
        edges = np.flatnonzero(active & ~np.r_[self.last_generated_active, active[:-1]])
        for edge in edges:
            effect_index = generated.sample_index + int(edge) + self.effect_delay_samples
            self.pending_effects[effect_index] = self.pending_effects.get(effect_index, 0) + 1
        self.last_generated_active = bool(active[-1]) if active.size else self.last_generated_active

    def next_block(self, sample_index: int, count: int | None = None) -> np.ndarray:
        with self._lock:
            return self._next_block(sample_index, count)

    def _next_block(self, sample_index: int, count: int | None = None) -> np.ndarray:
        count = self.settings.block_samples if count is None else int(count)
        if count <= 0:
            raise ValueError("block size must be positive")
        if self.last_sample_index is not None and sample_index != self.last_sample_index:
            # A discontinuity restarts the plant state instead of inventing the
            # missing command or activity history.
            self.pending_effects.clear()
            self.reduction_mv = 0.0
            self.last_generated_active = False
        t = (sample_index + np.arange(count, dtype=np.int64)) / self.fs
        reduction = np.zeros(count, dtype=np.float64)
        for i in range(count):
            absolute_index = sample_index + i
            event_count = self.pending_effects.pop(absolute_index, 0)
            if event_count:
                self.command_effect_sample_count += event_count
                self.command_effect_pulse_count += event_count
                self.command_effect_sample_indices.extend([absolute_index] * event_count)
                self.reduction_mv = min(
                    self.settings.max_amplitude_reduction_mv,
                    self.reduction_mv + event_count * self.settings.amplitude_reduction_per_pulse_mv,
                )
            elif self.settings.effect_recovery_s > 0:
                self.reduction_mv *= np.exp(-1.0 / (self.fs * self.settings.effect_recovery_s))
            reduction[i] = self.reduction_mv
        disturbed = (t >= self.settings.disturbance_start_s) & (t < self.settings.disturbance_stop_s)
        self.last_reduction_samples_mv = reduction.copy()
        amplitude = np.maximum(
            0.001,
            self.settings.baseline_amplitude_mv
            + disturbed.astype(np.float64) * self.settings.disturbance_amplitude_mv
            - reduction,
        )
        ca3 = amplitude * np.sin(2 * np.pi * self.settings.carrier_hz * t)
        ca3 += self.rng.normal(0.0, self.settings.noise_sd_mv, count)
        ec = 0.035 * np.sin(2 * np.pi * 8.0 * t) + self.rng.normal(0.0, self.settings.noise_sd_mv, count)
        self.observed_rms_mv.append(float(np.sqrt(np.mean(ca3 * ca3))))
        self.reduction_history_mv.append(float(np.mean(reduction)))
        self.last_sample_index = sample_index + count
        return np.vstack((ca3, ec)).astype(np.float64)


def _summarize_activity(plant: ResponsiveSyntheticPlant) -> dict[str, float]:
    values = np.asarray(plant.observed_rms_mv, dtype=np.float64)
    block_s = plant.settings.block_samples / plant.fs
    times = (np.arange(values.size) + 0.5) * block_s
    baseline = values[(times >= 2.0) & (times < 10.5)]
    disturbance = values[(times >= plant.settings.disturbance_start_s + 1.0)
                         & (times < plant.settings.disturbance_stop_s)]
    recovery = values[times >= plant.settings.disturbance_stop_s + 1.0]
    return {
        "baseline_ca3_block_rms_mv": float(np.mean(baseline)) if baseline.size else float("nan"),
        "disturbance_ca3_block_rms_mv": float(np.mean(disturbance)) if disturbance.size else float("nan"),
        "post_disturbance_ca3_block_rms_mv": float(np.mean(recovery)) if recovery.size else float("nan"),
        "mean_reduction_mv": float(np.mean(plant.reduction_history_mv)) if plant.reduction_history_mv else 0.0,
        "max_reduction_mv": float(np.max(plant.reduction_history_mv)) if plant.reduction_history_mv else 0.0,
    }


def _summarize_target_tracking(trace: list[tuple[float, float, float]], settings: PlantSettings,
                              target_z: float, max_rate_hz: float, *,
                              feature_diagnostics: list[dict[str, Any]] | None = None,
                              source_duration_s: float | None = None,
                              source_samples_seen: int | None = None,
                              source_gap_samples: int = 0,
                              skipped_source_samples: int = 0) -> dict[str, float | int | bool | None]:
    if not trace:
        return {"target_z": target_z, "sample_count": 0, "iae_z_seconds": float("nan"),
                "expected_sample_count": 0, "coverage_fraction": 0.0, "window_complete": False,
                "source_coverage_fraction": 0.0, "full_source_coverage": False,
                "source_gap_samples": int(source_gap_samples),
                "skipped_source_samples": int(skipped_source_samples),
                "expected_eligible_blocks": 0, "eligible_observation_blocks": 0,
                "eligible_observation_coverage_fraction": 0.0,
                "known_feature_invalid_blocks": 0, "known_feature_invalid_samples": 0,
                "stimulated_mask_samples": 0, "feature_guard_samples": 0,
                "masked_coverage_fraction": 0.0, "unexplained_feature_invalid_samples": 0,
                "window_mean_abs_error_z": float("nan"), "initial_mean_error_z": float("nan"),
                "final_mean_error_z": float("nan"), "mean_rate_hz": 0.0, "rate_std_hz": 0.0,
                "saturation_fraction": 0.0, "settling_time_seconds": None,
                "settling_band_z": TRACKING_ERROR_BAND_Z, "settling_window_seconds": TRACKING_SETTLING_WINDOW_S,
                "final_window_mean_error_z": float("nan"), "final_window_raw_mae_z": float("nan"),
                "final_window_moving_mean_error_z": float("nan"),
                "final_window_moving_mean_abs_error_z": float("nan"),
                "moving_mean_error_max_abs_z": float("nan"), "raw_error_std_z": float("nan"),
                "post_settling_raw_error_std_z": float("nan"),
                "raw_error_std_limit_z": TRACKING_RAW_ERROR_STD_LIMIT_Z,
                "raw_ripple_within_bounds": False,
                "settled_within_band": False, "tracking_response_gate_passed": False,
                "rate_direction_reversals": 0,
                "rate_within_bounds": True}
    data = np.asarray(trace, dtype=np.float64)
    time_s, z, rate = data.T
    block_s = settings.block_samples / settings.sample_rate_hz
    source_duration = source_duration_s if source_duration_s is not None else (
        float(time_s[-1] + block_s) if time_s.size else 0.0)
    expected_source_blocks = int(np.ceil(source_duration / block_s)) if source_duration > 0 else 0
    expected_source_samples = expected_source_blocks * settings.block_samples
    source_samples = (int(source_samples_seen) if source_samples_seen is not None
                      else min(len(trace), expected_source_blocks) * settings.block_samples)
    source_coverage = source_samples / expected_source_samples if expected_source_samples else 0.0
    full_source_coverage = bool(
        source_coverage >= 1.0 - 1e-12 and source_gap_samples == 0 and skipped_source_samples == 0)
    if feature_diagnostics is None:
        feature_diagnostics = [{"feature_valid": True, "feature_invalid_sample_count": 0,
                                "stimulated_sample_count": 0, "feature_recovery_samples": 0}
                               for _ in trace]
    if len(feature_diagnostics) != len(trace):
        raise ValueError("one controller feature diagnostic is required per source block")
    feature_valid_flags: list[bool] = []
    invalid_counts: list[int] = []
    stimulated_counts: list[int] = []
    guard_counts: list[int] = []
    known_invalid_flags: list[bool] = []
    unexplained_invalid_by_block: list[int] = []
    for diagnostic in feature_diagnostics:
        invalid_count = int(diagnostic.get("feature_invalid_sample_count", 0) or 0)
        stimulated_count = int(diagnostic.get("stimulated_sample_count", 0) or 0)
        recovery_samples = int(diagnostic.get("feature_recovery_samples", 0) or 0)
        feature_valid = bool(diagnostic.get("feature_valid", invalid_count == 0))
        unexplained_this_block = 0
        invalid_count = max(0, invalid_count)
        stimulated_count = max(0, stimulated_count)
        if invalid_count > settings.block_samples or stimulated_count > invalid_count:
            unexplained_this_block += max(settings.block_samples,
                                          invalid_count - settings.block_samples,
                                          stimulated_count - invalid_count)
            known_invalid = False
            guard_count = 0
        else:
            guard_count = invalid_count - stimulated_count
            invalid_is_explained = (
                invalid_count == 0
                or (feature_valid is False and (stimulated_count > 0 or
                                                (guard_count > 0 and recovery_samples > 0)))
            )
            if not invalid_is_explained:
                unexplained_this_block += invalid_count or settings.block_samples
            known_invalid = bool(invalid_count > 0 and invalid_is_explained)
            if not feature_valid and invalid_count == 0:
                unexplained_this_block += settings.block_samples
        if feature_valid and invalid_count > 0:
            unexplained_this_block = max(unexplained_this_block, invalid_count)
            known_invalid = False
        feature_valid_flags.append(feature_valid)
        invalid_counts.append(invalid_count)
        stimulated_counts.append(stimulated_count)
        guard_counts.append(guard_count)
        known_invalid_flags.append(known_invalid)
        unexplained_invalid_by_block.append(unexplained_this_block)
    feature_eligible = np.asarray([
        feature_valid_flags[i] and invalid_counts[i] == 0 for i in range(len(trace))
    ], dtype=bool)
    valid = np.isfinite(z) & np.isfinite(rate) & feature_eligible
    window_start = settings.disturbance_start_s + 0.5
    window_end = settings.disturbance_stop_s
    expected_count = max(0, int(round((window_end - window_start) / block_s)))
    expected_window = (time_s >= window_start) & (time_s < window_end)
    observed_source_blocks = int(np.count_nonzero(expected_window))
    window_indices = np.flatnonzero(expected_window)
    known_invalid_blocks = int(sum(known_invalid_flags[i] for i in window_indices))
    known_invalid_samples = int(sum(invalid_counts[i] for i in window_indices if known_invalid_flags[i]))
    unexplained_invalid_samples = int(sum(unexplained_invalid_by_block[i] for i in window_indices))
    stimulated_samples = int(sum(stimulated_counts[i] for i in window_indices if known_invalid_flags[i]))
    guard_samples = int(sum(guard_counts[i] for i in window_indices if known_invalid_flags[i]))
    expected_eligible = max(0, expected_count - known_invalid_blocks)
    window = valid & expected_window
    eligible_observations = int(np.count_nonzero(window))
    eligible_coverage = (eligible_observations / expected_eligible) if expected_eligible else 0.0
    source_window_coverage = observed_source_blocks / expected_count if expected_count else 0.0
    coverage = eligible_coverage
    masked_coverage = (known_invalid_samples / (expected_count * settings.block_samples)
                       if expected_count else 0.0)
    if not np.any(window) or coverage < 0.95:
        return {"target_z": target_z, "sample_count": 0, "iae_z_seconds": float("nan"),
                "expected_sample_count": expected_count, "coverage_fraction": coverage,
                "window_complete": False, "source_coverage_fraction": source_coverage,
                "source_window_coverage_fraction": source_window_coverage,
                "full_source_coverage": full_source_coverage,
                "source_gap_samples": int(source_gap_samples),
                "skipped_source_samples": int(skipped_source_samples),
                "expected_eligible_blocks": expected_eligible,
                "eligible_observation_blocks": eligible_observations,
                "eligible_observation_coverage_fraction": eligible_coverage,
                "known_feature_invalid_blocks": known_invalid_blocks,
                "known_feature_invalid_samples": known_invalid_samples,
                "stimulated_mask_samples": stimulated_samples,
                "feature_guard_samples": guard_samples,
                "masked_coverage_fraction": masked_coverage,
                "unexplained_feature_invalid_samples": unexplained_invalid_samples,
                "window_mean_abs_error_z": float("nan"), "initial_mean_error_z": float("nan"),
                "final_mean_error_z": float("nan"), "mean_rate_hz": 0.0, "rate_std_hz": 0.0,
                "saturation_fraction": 0.0, "settling_time_seconds": None,
                "settling_band_z": TRACKING_ERROR_BAND_Z, "settling_window_seconds": TRACKING_SETTLING_WINDOW_S,
                "final_window_mean_error_z": float("nan"), "final_window_raw_mae_z": float("nan"),
                "final_window_moving_mean_error_z": float("nan"),
                "final_window_moving_mean_abs_error_z": float("nan"),
                "moving_mean_error_max_abs_z": float("nan"), "raw_error_std_z": float("nan"),
                "post_settling_raw_error_std_z": float("nan"),
                "raw_error_std_limit_z": TRACKING_RAW_ERROR_STD_LIMIT_Z,
                "raw_ripple_within_bounds": False,
                "settled_within_band": False, "tracking_response_gate_passed": False,
                "rate_direction_reversals": 0,
                "rate_within_bounds": True}
    wt, wz, wr = time_s[window], z[window], rate[window]
    err = wz - target_z
    initial = float(np.mean(err[wt < settings.disturbance_start_s + 1.5]))
    final = float(np.mean(err[wt >= max(settings.disturbance_start_s + 1.5,
                                        settings.disturbance_stop_s - 2.0)]))
    band = TRACKING_ERROR_BAND_Z
    settling_time = None
    # Keep each sample on the source 5 ms grid. Explicit feature masks reduce
    # the eligible denominator; they never compress elapsed time.
    width = max(1, round(TRACKING_SETTLING_WINDOW_S / block_s))
    grid_first = int(round(window_start / block_s))
    grid_size = expected_count
    grid_error = np.zeros(grid_size, dtype=np.float64)
    grid_valid = np.zeros(grid_size, dtype=bool)
    grid_known_invalid = np.zeros(grid_size, dtype=bool)
    seen_grid = np.zeros(grid_size, dtype=bool)
    for i, sample_time in enumerate(time_s):
        block_index = int(round(sample_time / block_s)) - grid_first
        if 0 <= block_index < grid_size:
            if seen_grid[block_index]:
                unexplained_invalid_samples += settings.block_samples
                grid_valid[block_index] = False
                grid_known_invalid[block_index] = False
            else:
                seen_grid[block_index] = True
                grid_valid[block_index] = bool(valid[i])
                grid_known_invalid[block_index] = known_invalid_flags[i]
                if valid[i]:
                    grid_error[block_index] = float(z[i] - target_z)
    moving_count = max(0, grid_size - width + 1)
    moving_errors = np.full(moving_count, np.nan, dtype=np.float64)
    moving_times = np.full(moving_count, np.nan, dtype=np.float64)
    rolling_sum = np.r_[0.0, np.cumsum(grid_error * grid_valid)]
    rolling_valid = np.r_[0, np.cumsum(grid_valid.astype(np.int64))]
    rolling_masked = np.r_[0, np.cumsum(grid_known_invalid.astype(np.int64))]
    moving_coverages: list[float] = []
    for i in range(moving_count):
        stop = i + width
        eligible_count = width - int(rolling_masked[stop] - rolling_masked[i])
        valid_count = int(rolling_valid[stop] - rolling_valid[i])
        coverage_i = valid_count / eligible_count if eligible_count > 0 else 0.0
        moving_coverages.append(coverage_i)
        if (eligible_count > 0 and coverage_i >= TRACKING_MIN_MOVING_WINDOW_ELIGIBLE_COVERAGE
                and valid_count > 0):
            moving_errors[i] = float((rolling_sum[stop] - rolling_sum[i]) / valid_count)
            moving_times[i] = window_start + stop * block_s
    finite_moving = np.isfinite(moving_errors)
    for i in range(moving_errors.size):
        remaining = moving_errors[i:]
        remaining = remaining[np.isfinite(remaining)]
        if (finite_moving[i] and remaining.size
                and np.all(np.abs(remaining) <= band)):
            settling_time = float(moving_times[i] - settings.disturbance_start_s)
            break
    rate_delta = np.diff(wr)
    rate_direction = np.sign(np.where(np.abs(rate_delta) >= 0.02, rate_delta, 0.0))
    rate_direction = rate_direction[rate_direction != 0]
    rate_reversals = int(np.count_nonzero(rate_direction[1:] != rate_direction[:-1])) if rate_direction.size > 1 else 0
    final_window = wt >= max(window_start, window_end - TRACKING_SETTLING_WINDOW_S)
    final_errors = err[final_window]
    final_moving = moving_errors[moving_times >= window_end - TRACKING_SETTLING_WINDOW_S]
    final_moving = final_moving[np.isfinite(final_moving)]
    final_moving_mae = float(np.mean(np.abs(final_moving))) if final_moving.size else float("nan")
    raw_ripple_ok = bool(np.std(err) <= TRACKING_RAW_ERROR_STD_LIMIT_Z)
    if settling_time is not None:
        post_settling = err[wt >= settings.disturbance_start_s + settling_time]
        post_settling_raw_std = float(np.std(post_settling)) if post_settling.size else float("nan")
    else:
        post_settling_raw_std = float("nan")
    response_gate = bool(
        coverage >= 0.95 and source_window_coverage >= 1.0 - 1e-12
        and full_source_coverage and unexplained_invalid_samples == 0
        and raw_ripple_ok and settling_time is not None
        and final_moving_mae <= TRACKING_ERROR_BAND_Z
        and np.min(wr) >= -1e-12 and np.max(wr) <= max_rate_hz + 1e-12
    )
    return {
        "target_z": float(target_z), "sample_count": int(wt.size),
        "expected_sample_count": expected_count, "coverage_fraction": coverage,
        "window_complete": bool(coverage >= 0.95 and source_window_coverage >= 1.0 - 1e-12
                                and full_source_coverage and unexplained_invalid_samples == 0),
        "source_coverage_fraction": source_coverage,
        "source_window_coverage_fraction": source_window_coverage,
        "full_source_coverage": full_source_coverage,
        "source_gap_samples": int(source_gap_samples),
        "skipped_source_samples": int(skipped_source_samples),
        "expected_eligible_blocks": expected_eligible,
        "eligible_observation_blocks": eligible_observations,
        "eligible_observation_coverage_fraction": eligible_coverage,
        "known_feature_invalid_blocks": known_invalid_blocks,
        "known_feature_invalid_samples": known_invalid_samples,
        "stimulated_mask_samples": stimulated_samples,
        "feature_guard_samples": guard_samples,
        "masked_coverage_fraction": masked_coverage,
        "unexplained_feature_invalid_samples": unexplained_invalid_samples,
        "iae_z_seconds": float(np.sum(np.abs(err)) * block_s),
        "window_mean_abs_error_z": float(np.mean(np.abs(err))),
        "initial_mean_error_z": initial, "final_mean_error_z": final,
        "final_window_mean_error_z": float(np.mean(final_errors)) if final_errors.size else float("nan"),
        "final_window_raw_mae_z": float(np.mean(np.abs(final_errors))) if final_errors.size else float("nan"),
        "final_window_moving_mean_error_z": (
            float(np.mean(final_moving)) if final_moving.size else float("nan")),
        "final_window_moving_mean_abs_error_z": final_moving_mae,
        "complete_elapsed_moving_window_count": int(np.count_nonzero(finite_moving)),
        "minimum_moving_window_eligible_coverage": (
            float(np.min(moving_coverages)) if moving_coverages else 0.0),
        "moving_mean_error_max_abs_z": (
            float(np.max(np.abs(moving_errors[finite_moving]))) if np.any(finite_moving) else float("nan")),
        "raw_error_std_z": float(np.std(err)),
        "post_settling_raw_error_std_z": post_settling_raw_std,
        "raw_error_std_limit_z": TRACKING_RAW_ERROR_STD_LIMIT_Z,
        "raw_ripple_within_bounds": raw_ripple_ok,
        "mean_rate_hz": float(np.mean(wr)), "rate_std_hz": float(np.std(wr)),
        "saturation_fraction": float(np.mean(wr >= max_rate_hz - 0.05)),
        "settling_time_seconds": settling_time, "settling_band_z": float(band),
        "settling_window_seconds": TRACKING_SETTLING_WINDOW_S,
        "settled_within_band": settling_time is not None,
        "tracking_response_gate_passed": response_gate,
        "rate_direction_reversals": rate_reversals,
        "rate_within_bounds": bool(np.min(wr) >= -1e-12 and np.max(wr) <= max_rate_hz + 1e-12),
    }


def _summarize_plant_recovery(plant: ResponsiveSyntheticPlant,
                              trace: list[tuple[float, float, float]],
                              duration_s: float) -> dict[str, float | bool]:
    settings = plant.settings
    block_s = settings.block_samples / settings.sample_rate_hz
    times = (np.arange(len(plant.reduction_history_mv)) + 0.5) * block_s
    reductions = np.asarray(plant.reduction_history_mv, dtype=np.float64)
    pre_event = reductions[(times >= settings.disturbance_stop_s - 0.5)
                           & (times < settings.disturbance_stop_s)]
    final = reductions[times >= max(settings.disturbance_stop_s + 4 * settings.effect_recovery_s,
                                    duration_s - 1.0)]
    rates = np.asarray(trace, dtype=np.float64)
    if rates.size:
        rate_times, requested_rates = rates[:, 0], rates[:, 2]
        recovery_rates = requested_rates[
            (rate_times >= settings.disturbance_stop_s + 2 * settings.effect_recovery_s)
            & (rate_times < duration_s)
            & np.isfinite(requested_rates)
        ]
    else:
        recovery_rates = np.zeros(0, dtype=np.float64)
    complete = (duration_s >= settings.disturbance_stop_s
                + 4 * settings.effect_recovery_s + 1.0
                and pre_event.size > 0 and final.size > 0 and recovery_rates.size > 0)
    pre_mean = float(np.mean(pre_event)) if pre_event.size else float("nan")
    final_mean = float(np.mean(final)) if final.size else float("nan")
    residual_fraction = final_mean / pre_mean if pre_mean > 1e-12 else 0.0
    recovery_rate = float(np.mean(recovery_rates)) if recovery_rates.size else float("nan")
    return {
        "window_complete": complete,
        "pre_recovery_mean_reduction_mv": pre_mean,
        "final_recovery_mean_reduction_mv": final_mean,
        "residual_fraction": residual_fraction,
        "mean_requested_rate_after_two_recovery_constants_hz": recovery_rate,
        "bounded_recovery": bool(complete and residual_fraction <= 0.20 and recovery_rate <= 0.10),
    }


def run_responsive_case(controller: str, *, stim_enabled: bool = True, duration_s: float = 24.0,
                        lead_ms: float = 100.0, progress_path: Path | None = None,
                        recording_dir: Path | None = None, progress_interval_s: float = 10.0,
                        paced: bool = False, seed: int = 1414,
                        inject_stall_ms: float = 0.0,
                        stall_on_first_pulse: bool = False,
                        plant_settings: PlantSettings | None = None,
                        pid_settings: dict[str, float] | None = None,
                        capacity_target_z: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    if duration_s <= 0:
        raise ValueError("simulation duration must be positive")
    if recording_dir is None:
        import tempfile
        tmp = tempfile.TemporaryDirectory(prefix="echo-chamber-responsive-")
        record_dir = Path(tmp.name)
    else:
        tmp = None
        record_dir = recording_dir
        record_dir.mkdir(parents=True, exist_ok=True)
    settings = replace(plant_settings, seed=seed) if plant_settings is not None else PlantSettings(seed=seed)
    plant = ResponsiveSyntheticPlant(settings)
    config = _make_config(controller, lead_ms, record_dir, pid_settings_override=pid_settings)
    if not paced:
        # The deterministic plant advances the software AO clock in exact
        # five-millisecond virtual ticks. Size its asynchronous recorder queue
        # for this finite fixture so test execution speed cannot manufacture an
        # overflow that would not occur in paced operation.
        config = replace(config, logger_queue_blocks=max(
            config.logger_queue_blocks,
            int(np.ceil(duration_s * config.sample_rate / config.chunk_size)) + 64,
        ))
        config.validate()
    source_name = f"responsive_generic_seed_{seed}"
    metadata = {
        "application": "Echo Chamber generic responsive synthetic-plant validation",
        "controller": controller, "configuration": app.serializable_config(config),
        "plant": asdict(settings), "plant_claim": "generic synthetic plant, not tissue model",
        "plant_rate_capacity": plant_rate_capacity(
            settings, float(config.controller_settings.get("pid", {}).get("max_rate_hz", 2.0))),
        "physical_pulse_claims": False, "git_revision": git_revision(),
    }
    harness = _Harness(config, source_name, metadata, progress_path=progress_path,
                       progress_interval_s=progress_interval_s, paced=paced,
                       simulation=True, on_generated=plant.observe_generated_ao,
                       inject_stall_ms=inject_stall_ms,
                       stall_on_first_pulse=stall_on_first_pulse)
    roles, labels, ctx_index = ("CA3", "Cortex"), ("CA3", "EC"), 1
    blocks = int(np.ceil(duration_s * config.sample_rate / config.chunk_size))
    controller_trace: list[tuple[float, float, float]] = []
    feature_trace: list[dict[str, Any]] = []
    try:
        for block in range(blocks):
            first = block * config.chunk_size
            values = plant.next_block(first, config.chunk_size)
            generated = harness.process(first, values, roles, labels, ctx_index, stim_enabled=stim_enabled)
            if generated.stage == "skipped_source_block":
                controller_trace.append((first / config.sample_rate, float("nan"), float("nan")))
                feature_trace.append({"feature_valid": False,
                                      "feature_invalid_sample_count": config.chunk_size,
                                      "stimulated_sample_count": 0,
                                      "feature_recovery_samples": 0})
            else:
                diagnostics = harness.core._last_diagnostics
                z = diagnostics.get("z_score")
                rate = diagnostics.get("requested_rate_hz", 0.0)
                feature_trace.append({
                    key: diagnostics.get(key) for key in (
                        "feature_valid", "feature_invalid_sample_count",
                        "stimulated_sample_count", "feature_recovery_samples",
                        "feature_recovery_remaining_samples", "reason",
                        "baseline_std_mv",
                    )
                })
                controller_trace.append((first / config.sample_rate,
                                         float(z) if z is not None and np.isfinite(z) else float("nan"),
                                         float(rate) if rate is not None and np.isfinite(rate) else float("nan")))
        metrics = harness.finish()
        scheduler_stats = harness.scheduler.stats_snapshot()
        activity = _summarize_activity(plant)
        tracking = _summarize_target_tracking(
            controller_trace, settings,
            float(config.controller_settings.get("pid", {}).get("target_z", 1.0)),
            float(config.controller_settings.get("pid", {}).get("max_rate_hz", 2.0)),
            feature_diagnostics=feature_trace,
            source_duration_s=duration_s,
            source_samples_seen=metrics.source_samples,
            source_gap_samples=metrics.source_index_gap_samples,
            skipped_source_samples=metrics.skipped_source_samples,
        )
        target_z = float(config.controller_settings.get("pid", {}).get("target_z", 1.0))
        capacity_target = float(capacity_target_z if capacity_target_z is not None else target_z)
        baseline_std = next((float(item["baseline_std_mv"]) for item in feature_trace
                             if item.get("baseline_std_mv") is not None
                             and np.isfinite(item["baseline_std_mv"])
                             and float(item["baseline_std_mv"]) > 0), None)
        target_correction = (
            max(0.0, settings.disturbance_amplitude_mv - capacity_target * baseline_std / np.sqrt(0.5))
            if baseline_std is not None else None)
        disturbance_start = round(settings.disturbance_start_s * settings.sample_rate_hz)
        disturbance_stop = round(settings.disturbance_stop_s * settings.sample_rate_hz)
        generated_effects = [index for index in plant.command_effect_sample_indices
                             if disturbance_start <= index < disturbance_stop]
        measured_rate = (len(generated_effects)
                         / max(1e-12, settings.disturbance_stop_s - settings.disturbance_start_s))
        effective_rate_bound, effective_rate_basis = _effective_rate_bound_hz(
            lead_ms, feature_trace, settings)
        max_rate = float(config.controller_settings.get("pid", {}).get("max_rate_hz", 2.0))
        capacity = plant_rate_capacity(
            settings, max_rate, effective_max_rate_hz=effective_rate_bound)
        regulated_usage = {
            "controller_target_z": target_z,
            "generated_effect_pulse_count_in_disturbance": len(generated_effects),
            "generated_effect_rate_hz_in_disturbance": measured_rate,
            "formula_mean_reduction_at_observed_rate_mv": _steady_pulse_capacity(settings, measured_rate)[0],
            "target_correction_required_mv_for_capacity_reference_z": target_correction,
            "effective_rate_bound_basis": effective_rate_basis,
            "physical_pulse_claims": False,
        }
        tracking["baseline_std_mv"] = baseline_std
        tracking["capacity_reference_target_correction_required_mv"] = target_correction
        recovery = _summarize_plant_recovery(plant, controller_trace, duration_s)
        tracking_passed = bool(tracking["tracking_response_gate_passed"])
        report = {
            "case_id": metrics.case_id,
            "controller": controller,
            "stimulation_enabled": stim_enabled,
            "duration_seconds": duration_s,
            "lead_ms": lead_ms,
            "environment": {
                "python": __import__("platform").python_version(),
                "platform": __import__("platform").platform(),
                "git_revision": git_revision(),
                "source_checksum": hashlib_sha256_json(settings.__dict__),
                "config": app.serializable_config(config),
                "ao_scheduler": scheduler_stats,
                "physical_pulse_claims": False,
            },
            "plant": asdict(settings),
            "plant_rate_capacity": capacity,
            "regulated_controller_output_usage": regulated_usage,
            "activity": activity,
            "controller_tracking": tracking,
            "controller_tracking_trace": [
                {"time_s": float(t), "z": float(z) if np.isfinite(z) else None,
                 "error_z": float(z - tracking["target_z"]) if np.isfinite(z) else None,
                 "requested_rate_hz": float(rate) if np.isfinite(rate) else None,
                 **feature_trace[index]}
                for index, (t, z, rate) in enumerate(controller_trace)
            ],
            "plant_recovery": recovery,
            "tracking_gate_passed": tracking_passed,
            "simulation_acceptance_passed": bool(tracking_passed and recovery["bounded_recovery"]),
            "plant_effect_sample_count": plant.command_effect_sample_count,
            "plant_effect_pulse_count": plant.command_effect_pulse_count,
            "generated_nonzero_ao_command_samples": plant.generated_nonzero_samples,
            "metrics": asdict(metrics),
        }
        write_progress(progress_path, {"type": "simulation_complete", "case_id": metrics.case_id,
                                       "activity": activity, "physical_pulse_claims": False})
    except BaseException:
        try:
            harness.abort()
        except Exception:
            pass
        raise
    finally:
        if tmp is not None:
            tmp.cleanup()
    return report, {"plant": plant, "metrics": metrics}


def run_capacity_reference_case(*, duration_s: float, lead_ms: float,
                                progress_path: Path | None = None,
                                recording_dir: Path | None = None,
                                plant_settings: PlantSettings,
                                target_z: float,
                                max_rate_hz: float = 2.0,
                                paced: bool = False) -> dict[str, Any]:
    """Measure maximum modeled plant correction through the production core/AO path."""
    reference_pid = {
        **TRACKING_PID_SETTINGS,
        "kp": 1.0, "ki": 0.0, "kd": 0.0,
        "target_z": -1000.0,
        "max_rate_hz": float(max_rate_hz),
    }
    write_progress(progress_path, {
        "type": "capacity_reference_started", "controller": "pid",
        "pid_reference_settings": {key: reference_pid[key]
                                    for key in ("kp", "ki", "kd", "target_z", "max_rate_hz")},
        "duration_seconds": duration_s, "lead_ms": lead_ms,
        "physical_pulse_claims": False,
    })
    reference_report, data = run_responsive_case(
        "pid", stim_enabled=True, duration_s=duration_s, lead_ms=lead_ms,
        progress_path=progress_path,
        recording_dir=(recording_dir / "capacity_reference") if recording_dir is not None else None,
        paced=paced, plant_settings=plant_settings, pid_settings=reference_pid,
        capacity_target_z=target_z,
    )
    usage = reference_report["regulated_controller_output_usage"]
    metrics = reference_report["metrics"]
    capacity = _summarize_capacity_reference(
        data["plant"], metrics,
        duration_s=duration_s, lead_ms=lead_ms, target_z=target_z,
        target_correction_mv=usage["target_correction_required_mv_for_capacity_reference_z"],
        max_rate_hz=max_rate_hz,
        feature_diagnostics=reference_report["controller_tracking_trace"],
        scheduler_stats=reference_report["environment"]["ao_scheduler"],
    )
    result = {
        **capacity,
        "case_id": reference_report["case_id"],
        "environment": reference_report["environment"],
        "plant": reference_report["plant"],
        "metrics": metrics,
        "controller_output_usage": usage,
        "physical_pulse_claims": False,
    }
    write_progress(progress_path, {
        "type": "capacity_reference_complete", "case_id": result["case_id"],
        "capacity_reference_reaches_target": result["capacity_reference_reaches_target"],
        "steady_mean_reduction_mv": result["steady_mean_reduction_mv"],
        "target_correction_required_mv": result["target_correction_required_mv"],
        "generated_effect_rate_hz_in_steady_window": result["generated_effect_rate_hz_in_steady_window"],
        "physical_pulse_claims": False,
    })
    return result


def hashlib_sha256_json(value: dict[str, Any]) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def run_feedback_comparison(controller: str, *, duration_s: float = 24.0, lead_ms: float = 100.0,
                            progress_path: Path | None = None, recording_dir: Path | None = None,
                            paced: bool = False, plant_settings: PlantSettings | None = None,
                            pid_settings: dict[str, float] | None = None,
                            inject_stall_ms: float = 0.0,
                            stall_on_first_pulse: bool = False,
                            include_capacity_reference: bool = False) -> dict[str, Any]:
    enabled, enabled_data = run_responsive_case(
        controller, stim_enabled=True, duration_s=duration_s, lead_ms=lead_ms,
        progress_path=progress_path, recording_dir=recording_dir, paced=paced,
        plant_settings=plant_settings, pid_settings=pid_settings,
        inject_stall_ms=inject_stall_ms, stall_on_first_pulse=stall_on_first_pulse,
    )
    disabled, disabled_data = run_responsive_case(
        controller, stim_enabled=False, duration_s=duration_s, lead_ms=lead_ms,
        progress_path=progress_path, recording_dir=recording_dir, paced=paced,
        plant_settings=plant_settings, pid_settings=pid_settings,
        inject_stall_ms=0.0, stall_on_first_pulse=False,
    )
    capacity_reference = None
    if include_capacity_reference:
        if plant_settings is None:
            raise ValueError("a plant_settings fixture is required for a capacity reference")
        capacity_reference = run_capacity_reference_case(
            duration_s=duration_s, lead_ms=lead_ms, progress_path=progress_path,
            recording_dir=recording_dir, plant_settings=plant_settings,
            target_z=float((pid_settings or TRACKING_PID_SETTINGS).get("target_z", 1.0)),
            max_rate_hz=float((pid_settings or TRACKING_PID_SETTINGS).get("max_rate_hz", 2.0)),
            paced=paced,
        )
    active_rms = enabled["activity"]["disturbance_ca3_block_rms_mv"]
    disabled_rms = disabled["activity"]["disturbance_ca3_block_rms_mv"]
    observed_change = (disabled_rms - active_rms
                       if np.isfinite(active_rms) and np.isfinite(disabled_rms) else None)
    tracking_improvement = (
        disabled["controller_tracking"]["iae_z_seconds"]
        - enabled["controller_tracking"]["iae_z_seconds"]
        if disabled["controller_tracking"]["window_complete"]
        and enabled["controller_tracking"]["window_complete"] else None
    )
    tracking_iae_improved = bool(tracking_improvement is not None and tracking_improvement > 0)
    tracking_passed = bool(enabled["tracking_gate_passed"])
    recovery_passed = bool(enabled["plant_recovery"]["bounded_recovery"])
    capacity_passed = (bool(capacity_reference["capacity_reference_reaches_target"])
                       if capacity_reference is not None else True)
    reported_capacity = capacity_reference if capacity_reference is not None else enabled["plant_rate_capacity"]
    return {
        "schema": "echo-chamber-responsive-simulation-v1",
        "controller": controller,
        "stimulated": enabled,
        "no_stimulation_control": disabled,
        "observed_disturbance_rms_reduction_mv": observed_change,
        "feedback_changed_observed_activity": bool(observed_change is not None and observed_change > 0),
        "tracking_error_reduction_z_seconds": tracking_improvement,
        "tracking_iae_improved": tracking_iae_improved,
        "plant_rate_capacity": reported_capacity,
        "capacity_reference": capacity_reference,
        "regulated_controller_output_usage": enabled["regulated_controller_output_usage"],
        "tracking_gate_passed": tracking_passed,
        "plant_recovery_passed": recovery_passed,
        "simulation_acceptance_passed": bool(
            capacity_passed and tracking_passed and recovery_passed and tracking_iae_improved),
        "accepted_generated_command_samples": enabled["generated_nonzero_ao_command_samples"],
        "no_stimulation_generated_command_samples": disabled["generated_nonzero_ao_command_samples"],
        "plant_is_tissue_model": False,
        "physical_pulse_claims": False,
    }
