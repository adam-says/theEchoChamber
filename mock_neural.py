"""Small deterministic E/I neural-mass fixture for closed-loop software demos.

This is an engineering LFP proxy, not a calibrated neural or tissue model.
Its oscillatory content comes from the two coupled population states.  The
plant accepts only AO samples that the simulated scheduler has generated.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class NeuralMassParameters:
    sample_rate_hz: int = 20_000
    integration_substeps: int = 1
    tau_e_s: float = 0.0012
    tau_i_s: float = 0.0020
    w_ee: float = 1.5
    w_ei: float = 1.5
    w_ie: float = 3.0
    w_ii: float = 0.5
    noise_sd: float = 0.03
    noise_tau_s: float = 0.001
    baseline_drive: float = 0.05
    lfp_projection_e: float = 1.0
    lfp_projection_i: float = -1.0
    lfp_scale_mV: float = 5.0
    ao_effect_delay_ms: float = 20.0
    ao_effect_decay_s: float = 0.5
    ao_effect_gain_per_v_s: float = 100.0
    max_effect_state: float = 2.0

    def validate(self) -> None:
        numeric = (
            self.tau_e_s, self.tau_i_s, self.w_ee, self.w_ei, self.w_ie,
            self.w_ii, self.noise_sd, self.noise_tau_s, self.baseline_drive,
            self.lfp_projection_e, self.lfp_projection_i, self.lfp_scale_mV,
            self.ao_effect_delay_ms, self.ao_effect_decay_s,
            self.ao_effect_gain_per_v_s, self.max_effect_state,
        )
        if (self.sample_rate_hz < 1_000 or self.integration_substeps < 1
                or not all(math.isfinite(x) for x in numeric)):
            raise ValueError("neural-mass parameters must be finite at a supported sample rate")
        if min(self.tau_e_s, self.tau_i_s, self.noise_tau_s, self.ao_effect_decay_s,
               self.lfp_scale_mV, self.max_effect_state) <= 0:
            raise ValueError("time constants, LFP scale, and state bound must be positive")
        if self.noise_sd < 0 or self.ao_effect_delay_ms < 0 or self.ao_effect_gain_per_v_s < 0:
            raise ValueError("noise and AO effect parameters must be non-negative")
        dt = 1.0 / self.sample_rate_hz
        substep_dt = dt / self.integration_substeps
        if substep_dt / self.tau_e_s > 0.1 or substep_dt / self.tau_i_s > 0.1:
            raise ValueError("fixed sample step is too large for the configured population time constants")
        if self.ao_effect_delay_ms * self.sample_rate_hz / 1_000.0 % 1:
            raise ValueError("AO effect delay must resolve to a whole sample")


class TwoPopulationNeuralMass:
    """Euler-integrated normalized E/I activity deviations and LFP proxy."""

    def __init__(self, parameters: NeuralMassParameters | None = None, *, seed: int = 1) -> None:
        self.parameters = parameters or NeuralMassParameters()
        self.parameters.validate()
        p = self.parameters
        self.dt = 1.0 / p.sample_rate_hz
        self.substep_dt = self.dt / p.integration_substeps
        self.e = 0.0
        self.i = 0.0
        self.inhibitory_state = 0.0
        self.noise_e = 0.0
        self.noise_i = 0.0
        self._noise_e_rng, self._noise_i_rng = (
            np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)
        )
        self._noise_alpha = -math.expm1(-self.dt / p.noise_tau_s)
        self._effect_decay = math.exp(-self.substep_dt / p.ao_effect_decay_s)
        self._effect_delay_samples = round(p.ao_effect_delay_ms * p.sample_rate_hz / 1_000.0)
        self._effect_delay_line = np.zeros(max(1, self._effect_delay_samples), dtype=np.float64)
        self._effect_delay_cursor = 0
        self.next_sample_index: int | None = None

    def step_block(
        self,
        first_sample_index: int,
        excitatory_drive: float | np.ndarray,
        generated_ao_v: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Advance the plant using only generated AO samples for this interval."""
        ao = np.asarray(generated_ao_v, dtype=np.float64).reshape(-1)
        if ao.size == 0 or not np.all(np.isfinite(ao)):
            raise ValueError("generated AO must be a non-empty finite sample vector")
        if np.max(np.abs(ao)) > 10.0:
            raise ValueError("generated AO exceeds the mock plant's bounded input range")
        if self.next_sample_index is not None and int(first_sample_index) != self.next_sample_index:
            raise ValueError("neural-mass sample indices must be contiguous and increasing")
        if np.isscalar(excitatory_drive):
            drive = np.full(ao.size, float(excitatory_drive), dtype=np.float64)
        else:
            drive = np.asarray(excitatory_drive, dtype=np.float64).reshape(-1)
            if drive.size != ao.size:
                raise ValueError("drive and generated AO must have matching sample counts")
        if not np.all(np.isfinite(drive)) or np.max(np.abs(drive)) > 2.0:
            raise ValueError("excitatory drive must be finite and bounded")

        p = self.parameters
        out_e = np.empty(ao.size, dtype=np.float64)
        out_i = np.empty(ao.size, dtype=np.float64)
        out_lfp = np.empty(ao.size, dtype=np.float64)
        out_effect = np.empty(ao.size, dtype=np.float64)
        alpha_e = self.substep_dt / p.tau_e_s
        alpha_i = self.substep_dt / p.tau_i_s
        for index, ao_sample in enumerate(ao):
            if self._effect_delay_samples:
                cursor = self._effect_delay_cursor
                delayed_ao = self._effect_delay_line[cursor]
                self._effect_delay_line[cursor] = abs(float(ao_sample))
                self._effect_delay_cursor = (cursor + 1) % self._effect_delay_line.size
            else:
                delayed_ao = abs(float(ao_sample))
            self.noise_e += self._noise_alpha * (
                p.noise_sd * float(self._noise_e_rng.standard_normal()) - self.noise_e
            )
            self.noise_i += self._noise_alpha * (
                p.noise_sd * float(self._noise_i_rng.standard_normal()) - self.noise_i
            )
            for _ in range(p.integration_substeps):
                self.inhibitory_state = min(
                    p.max_effect_state,
                    max(0.0, self.inhibitory_state * self._effect_decay
                        + p.ao_effect_gain_per_v_s * delayed_ao * p.ao_effect_decay_s
                        * (1.0 - self._effect_decay)),
                )
                effective_recurrent_e = p.w_ee / (1.0 + self.inhibitory_state)
                e_input = (effective_recurrent_e * self.e - p.w_ei * self.i
                           + drive[index] + self.noise_e)
                i_input = p.w_ie * self.e - p.w_ii * self.i + self.noise_i
                target_e = math.tanh(e_input)
                target_i = math.tanh(i_input)
                self.e += alpha_e * (target_e - self.e)
                self.i += alpha_i * (target_i - self.i)
                if abs(self.e) > 1.0 + 1e-12 or abs(self.i) > 1.0 + 1e-12:
                    raise RuntimeError("normalized E/I state escaped its mathematical bound")
            out_e[index] = self.e
            out_i[index] = self.i
            out_lfp[index] = p.lfp_scale_mV * (
                p.lfp_projection_e * self.e + p.lfp_projection_i * self.i
            )
            out_effect[index] = self.inhibitory_state
        self.next_sample_index = int(first_sample_index) + ao.size
        return out_lfp, {
            "e_activity_deviation": out_e,
            "i_activity_deviation": out_i,
            "inhibitory_effect_state": out_effect,
        }


@dataclass(frozen=True)
class ElectrodeArtifactParameters:
    sample_rate_hz: int = 20_000
    direct_mV_per_ao_v: float = 80.0
    tail_mV_per_ao_v: float = 20.0
    tail_tau_s: float = 0.002
    blanking_ms: float = 20.0  # post-pulse blanking; the pulse interval is included separately

    def validate(self) -> None:
        if self.sample_rate_hz < 1_000:
            raise ValueError("artifact sample rate must be at least 1 kHz")
        values = (self.direct_mV_per_ao_v, self.tail_mV_per_ao_v,
                  self.tail_tau_s, self.blanking_ms)
        if not all(math.isfinite(v) for v in values) or min(values) < 0:
            raise ValueError("artifact parameters must be finite and non-negative")
        if self.tail_tau_s == 0 and self.tail_mV_per_ao_v > 0:
            raise ValueError("a nonzero artifact tail requires a positive decay constant")
        if self.blanking_ms * self.sample_rate_hz / 1_000.0 % 1:
            raise ValueError("blanking duration must resolve to a whole sample")


class SyntheticElectrodeObservation:
    """Adds a declared AO-coupled artifact without changing latent LFP state."""

    def __init__(self, parameters: ElectrodeArtifactParameters | None = None) -> None:
        self.parameters = parameters or ElectrodeArtifactParameters()
        self.parameters.validate()
        self._tail = 0.0
        self._tail_decay = (math.exp(-1.0 / (self.parameters.sample_rate_hz
                                            * self.parameters.tail_tau_s))
                            if self.parameters.tail_tau_s else 0.0)
        self._blank_until = 0
        self.next_sample_index: int | None = None

    def observe(
        self,
        first_sample_index: int,
        true_lfp_mV: np.ndarray,
        generated_ao_v: np.ndarray,
    ) -> dict[str, np.ndarray]:
        truth = np.asarray(true_lfp_mV, dtype=np.float64).reshape(-1)
        ao = np.asarray(generated_ao_v, dtype=np.float64).reshape(-1)
        if truth.size == 0 or truth.size != ao.size or not np.all(np.isfinite(truth)):
            raise ValueError("true LFP and generated AO must be equal, non-empty finite vectors")
        if not np.all(np.isfinite(ao)) or np.max(np.abs(ao)) > 10.0:
            raise ValueError("generated AO must be finite and within the mock plant's bounded input range")
        if self.next_sample_index is not None and int(first_sample_index) != self.next_sample_index:
            raise ValueError("electrode observation indices must be contiguous and increasing")
        p = self.parameters
        artifact = np.empty(truth.size, dtype=np.float64)
        blanking = np.zeros(truth.size, dtype=bool)
        blank_samples = round(p.blanking_ms * p.sample_rate_hz / 1_000.0)
        for j, ao_sample in enumerate(ao):
            sample_index = int(first_sample_index) + j
            magnitude = abs(float(ao_sample))
            if magnitude > 0.0:
                self._blank_until = max(self._blank_until, sample_index + blank_samples + 1)
            self._tail = self._tail * self._tail_decay + (
                (1.0 - self._tail_decay) * magnitude * p.tail_mV_per_ao_v
            )
            artifact[j] = float(ao_sample) * p.direct_mV_per_ao_v + self._tail
            blanking[j] = sample_index < self._blank_until
        self.next_sample_index = int(first_sample_index) + truth.size
        return {
            "true_lfp_mV": truth.copy(),
            "artifact_mV": artifact,
            "electrode_mV": truth + artifact,
            "artifact_oracle_mask": blanking,
        }
