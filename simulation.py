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
    disturbance_amplitude_mv=0.010,
    disturbance_start_s=12.0,
    disturbance_stop_s=20.0,
    effect_delay_ms=20.0,
    effect_recovery_s=0.50,
    amplitude_reduction_per_pulse_mv=0.002,
    max_amplitude_reduction_mv=0.020,
    seed=1414,
)
TRACKING_PID_SETTINGS = {"kp": 0.03, "ki": 0.001, "kd": 0.0, "target_z": 1.0,
                         "direction": 1, "max_rate_hz": 2.0, "update_ms": 5.0,
                         "derivative_tau_s": 0.020, "band_hz": (100.0, 250.0)}


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
                              target_z: float, max_rate_hz: float) -> dict[str, float | bool | None]:
    if not trace:
        return {"target_z": target_z, "sample_count": 0, "iae_z_seconds": float("nan"),
                "expected_sample_count": 0, "coverage_fraction": 0.0, "window_complete": False,
                "window_mean_abs_error_z": float("nan"), "initial_mean_error_z": float("nan"),
                "final_mean_error_z": float("nan"), "mean_rate_hz": 0.0, "rate_std_hz": 0.0,
                "saturation_fraction": 0.0, "settling_time_seconds": None,
                "settling_band_z": float("nan"), "rate_direction_reversals": 0,
                "rate_within_bounds": True}
    data = np.asarray(trace, dtype=np.float64)
    time_s, z, rate = data.T
    valid = np.isfinite(z) & np.isfinite(rate)
    window_start = settings.disturbance_start_s + 0.5
    window_end = settings.disturbance_stop_s
    block_s = settings.block_samples / settings.sample_rate_hz
    expected_count = max(0, int(round((window_end - window_start) / block_s)))
    expected_window = (time_s >= window_start) & (time_s < window_end)
    window = valid & expected_window
    coverage = float(np.count_nonzero(window) / expected_count) if expected_count else 0.0
    if not np.any(window) or coverage < 0.95:
        return {"target_z": target_z, "sample_count": 0, "iae_z_seconds": float("nan"),
                "expected_sample_count": expected_count, "coverage_fraction": coverage,
                "window_complete": False,
                "window_mean_abs_error_z": float("nan"), "initial_mean_error_z": float("nan"),
                "final_mean_error_z": float("nan"), "mean_rate_hz": 0.0, "rate_std_hz": 0.0,
                "saturation_fraction": 0.0, "settling_time_seconds": None,
                "settling_band_z": float("nan"), "rate_direction_reversals": 0,
                "rate_within_bounds": True}
    wt, wz, wr = time_s[window], z[window], rate[window]
    err = wz - target_z
    initial = float(np.mean(err[wt < settings.disturbance_start_s + 1.5]))
    final = float(np.mean(err[wt >= max(settings.disturbance_start_s + 1.5,
                                        settings.disturbance_stop_s - 2.0)]))
    band = max(0.25, abs(initial) * 0.25)
    settling_time = None
    # Require a full one-second running average inside an error band derived
    # from the actual perturbation response; no instantaneous one-SD claim.
    width = max(1, round(1.0 / block_s))
    abs_error = np.abs(err)
    for end in range(width - 1, len(abs_error)):
        start = max(0, end - width + 1)
        if end - start + 1 == width and float(np.mean(abs_error[start:end + 1])) <= band:
            settling_time = float(wt[end] - settings.disturbance_start_s)
            break
    rate_delta = np.diff(wr)
    rate_direction = np.sign(np.where(np.abs(rate_delta) >= 0.02, rate_delta, 0.0))
    rate_direction = rate_direction[rate_direction != 0]
    rate_reversals = int(np.count_nonzero(rate_direction[1:] != rate_direction[:-1])) if rate_direction.size > 1 else 0
    return {
        "target_z": float(target_z), "sample_count": int(wt.size),
        "expected_sample_count": expected_count, "coverage_fraction": coverage,
        "window_complete": True,
        "iae_z_seconds": float(np.sum(np.abs(err)) * block_s),
        "window_mean_abs_error_z": float(np.mean(np.abs(err))),
        "initial_mean_error_z": initial, "final_mean_error_z": final,
        "mean_rate_hz": float(np.mean(wr)), "rate_std_hz": float(np.std(wr)),
        "saturation_fraction": float(np.mean(wr >= max_rate_hz - 0.05)),
        "settling_time_seconds": settling_time, "settling_band_z": float(band),
        "rate_direction_reversals": rate_reversals,
        "rate_within_bounds": bool(np.min(wr) >= -1e-12 and np.max(wr) <= max_rate_hz + 1e-12),
    }


def run_responsive_case(controller: str, *, stim_enabled: bool = True, duration_s: float = 24.0,
                        lead_ms: float = 100.0, progress_path: Path | None = None,
                        recording_dir: Path | None = None, progress_interval_s: float = 10.0,
                        paced: bool = False, seed: int = 1414,
                        inject_stall_ms: float = 0.0,
                        stall_on_first_pulse: bool = False,
                        plant_settings: PlantSettings | None = None,
                        pid_settings: dict[str, float] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
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
        "plant_setpoint_reachable_by_max_reduction": bool(
            settings.disturbance_amplitude_mv <= settings.max_amplitude_reduction_mv),
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
    try:
        for block in range(blocks):
            first = block * config.chunk_size
            values = plant.next_block(first, config.chunk_size)
            generated = harness.process(first, values, roles, labels, ctx_index, stim_enabled=stim_enabled)
            if generated.stage == "skipped_source_block":
                controller_trace.append((first / config.sample_rate, float("nan"), float("nan")))
            else:
                diagnostics = harness.core._last_diagnostics
                z = diagnostics.get("z_score")
                rate = diagnostics.get("requested_rate_hz", 0.0)
                controller_trace.append((first / config.sample_rate,
                                         float(z) if z is not None and np.isfinite(z) else float("nan"),
                                         float(rate) if rate is not None and np.isfinite(rate) else float("nan")))
        metrics = harness.finish()
        scheduler_stats = harness.scheduler.stats_snapshot()
        activity = _summarize_activity(plant)
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
            "plant_setpoint_reachable_by_max_reduction": bool(
                settings.disturbance_amplitude_mv <= settings.max_amplitude_reduction_mv),
            "activity": activity,
            "controller_tracking": _summarize_target_tracking(
                controller_trace, settings,
                float(config.controller_settings.get("pid", {}).get("target_z", 1.0)),
                float(config.controller_settings.get("pid", {}).get("max_rate_hz", 2.0)),
            ),
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


def hashlib_sha256_json(value: dict[str, Any]) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def run_feedback_comparison(controller: str, *, duration_s: float = 24.0, lead_ms: float = 100.0,
                            progress_path: Path | None = None, recording_dir: Path | None = None,
                            paced: bool = False, plant_settings: PlantSettings | None = None,
                            pid_settings: dict[str, float] | None = None,
                            inject_stall_ms: float = 0.0,
                            stall_on_first_pulse: bool = False) -> dict[str, Any]:
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
    active_rms = enabled["activity"]["disturbance_ca3_block_rms_mv"]
    disabled_rms = disabled["activity"]["disturbance_ca3_block_rms_mv"]
    observed_change = (disabled_rms - active_rms
                       if np.isfinite(active_rms) and np.isfinite(disabled_rms) else None)
    return {
        "schema": "echo-chamber-responsive-simulation-v1",
        "controller": controller,
        "stimulated": enabled,
        "no_stimulation_control": disabled,
        "observed_disturbance_rms_reduction_mv": observed_change,
        "feedback_changed_observed_activity": bool(observed_change is not None and observed_change > 0),
        "tracking_error_reduction_z_seconds": (
            disabled["controller_tracking"]["iae_z_seconds"]
            - enabled["controller_tracking"]["iae_z_seconds"]
            if disabled["controller_tracking"]["window_complete"]
            and enabled["controller_tracking"]["window_complete"] else None
        ),
        "plant_setpoint_reachable_by_max_reduction": bool(
            (plant_settings or PlantSettings()).disturbance_amplitude_mv
            <= (plant_settings or PlantSettings()).max_amplitude_reduction_mv),
        "accepted_generated_command_samples": enabled["generated_nonzero_ao_command_samples"],
        "no_stimulation_generated_command_samples": disabled["generated_nonzero_ao_command_samples"],
        "plant_is_tissue_model": False,
        "physical_pulse_claims": False,
    }
