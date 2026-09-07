"""Corrected ESN readout training for paired CTX -> CA3 recordings.

The trained reservoir topology and weights are preserved. Only independent
input/target scaling and the ridge readout are fitted. Training and runtime use
the same causal 25 Hz model-band filter with an explicit zero state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from .artifact import InferenceArtifact, load_inference_artifact, save_inference_artifact
from .preprocessing import build_lowpass_fir


@dataclass(frozen=True)
class RecordingPair:
    recording_id: str
    ctx_path: Path
    ca3_path: Path


@dataclass(frozen=True)
class AffineScaler:
    scale: float
    offset: float
    data_min: float
    data_max: float

    @classmethod
    def from_range(cls, data_min: float, data_max: float) -> "AffineScaler":
        if not np.isfinite(data_min) or not np.isfinite(data_max) or data_max <= data_min:
            raise ValueError(f"invalid scaler range [{data_min}, {data_max}]")
        scale = 2.0 / (data_max - data_min)
        return cls(scale, -1.0 - data_min * scale, data_min, data_max)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale + self.offset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_stem(path: Path) -> tuple[str, int]:
    match = re.fullmatch(r"(.+?)_[^_]+_(\d+)", path.stem)
    if not match:
        raise ValueError(f"cannot parse experiment/electrode from {path.name}")
    return match.group(1), int(match.group(2))


def pair_recordings(ctx_dir: Path, ca3_dir: Path, *, role: str) -> list[RecordingPair]:
    grouped: dict[str, dict[str, list[tuple[int, Path]]]] = {}
    for modality, directory in (("ctx", ctx_dir), ("ca3", ca3_dir)):
        files = sorted(directory.glob("*.mat"))
        if not files:
            raise ValueError(f"no .mat files found in {directory}")
        for path in files:
            group, electrode = _parse_stem(path)
            grouped.setdefault(group, {"ctx": [], "ca3": []})[modality].append((electrode, path))

    pairs: list[RecordingPair] = []
    for group in sorted(grouped):
        ctx_items = sorted(grouped[group]["ctx"])
        ca3_items = sorted(grouped[group]["ca3"])
        if len(ctx_items) != len(ca3_items):
            raise ValueError(
                f"unequal CTX/CA3 counts for {group}: {len(ctx_items)} != {len(ca3_items)}"
            )
        # CONFIRMED PENDING TEST: files from the same experiment are paired by
        # ascending electrode number because no explicit electrode-pair map is
        # present in the source dataset. Verify this rule against lab notes.
        for (ctx_electrode, ctx_path), (ca3_electrode, ca3_path) in zip(ctx_items, ca3_items):
            recording_id = f"{role}__{group}__ctx{ctx_electrode}__ca3{ca3_electrode}"
            pairs.append(RecordingPair(recording_id, ctx_path, ca3_path))
    return pairs


def _load_mat_signal(
    path: Path, *, missing_fs_hz: float | None = None
) -> tuple[np.ndarray, float]:
    try:
        import mat73

        loaded = mat73.loadmat(str(path))
    except Exception:
        from scipy.io import loadmat

        loaded = loadmat(path)
    if "data" not in loaded:
        raise KeyError(f"{path} must contain data")
    values = np.asarray(loaded["data"], dtype=np.float64).reshape(-1)
    if "fs" in loaded:
        sample_rate = float(np.asarray(loaded["fs"]).squeeze())
    elif missing_fs_hz is not None:
        # The five legacy validation X_data/Y_data files intentionally omit
        # ``fs``. Paper_figures.py and the project pipeline audit identify
        # those files as already sampled at the 2 kHz common/model rate.
        sample_rate = float(missing_fs_hz)
    else:
        raise KeyError(f"{path} does not contain fs and no explicit assumption was supplied")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains empty or non-finite data")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(f"{path} has invalid sample rate {sample_rate}")
    return values, sample_rate


def preprocess_signal(values: np.ndarray, source_rate: float, artifact: InferenceArtifact) -> np.ndarray:
    """Apply the deployment-equivalent causal path and return model-rate data."""
    from scipy.signal import lfilter, resample_poly

    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    model_rate = artifact.fs_model_hz
    if not np.isclose(source_rate, model_rate):
        ratio = source_rate / model_rate
        if np.isclose(ratio, round(ratio)):
            q = int(round(ratio))
            aa = build_lowpass_fir(source_rate, artifact.aa_cutoff_hz, artifact.aa_numtaps)
            signal = lfilter(aa, [1.0], signal)[::q]
        else:
            # The notebook called resample without assigning its return value.
            # Polyphase resampling here is assigned and deterministic.
            signal = resample_poly(signal, model_rate, int(round(source_rate)))
    model_fir = build_lowpass_fir(
        float(model_rate), artifact.model_cutoff_hz, artifact.model_numtaps
    )
    return np.asarray(lfilter(model_fir, [1.0], signal), dtype=np.float64)


def _pair_values(pair: RecordingPair, artifact: InferenceArtifact) -> tuple[np.ndarray, np.ndarray]:
    ctx, ctx_rate = _load_mat_signal(
        pair.ctx_path, missing_fs_hz=artifact.fs_model_hz
    )
    ca3, ca3_rate = _load_mat_signal(
        pair.ca3_path, missing_fs_hz=artifact.fs_model_hz
    )
    if not np.isclose(ctx_rate, ca3_rate):
        raise ValueError(f"sampling-rate mismatch in {pair.recording_id}: {ctx_rate} != {ca3_rate}")
    ctx = preprocess_signal(ctx, ctx_rate, artifact)
    ca3 = preprocess_signal(ca3, ca3_rate, artifact)
    if ctx.size != ca3.size:
        raise ValueError(f"length mismatch in {pair.recording_id}: {ctx.size} != {ca3.size}")
    return ctx, ca3


def fit_scalers(pairs: Sequence[RecordingPair], artifact: InferenceArtifact) -> tuple[AffineScaler, AffineScaler]:
    ctx_min, ctx_max = np.inf, -np.inf
    ca3_min, ca3_max = np.inf, -np.inf
    for pair in pairs:
        ctx, ca3 = _pair_values(pair, artifact)
        ctx_min, ctx_max = min(ctx_min, float(np.min(ctx))), max(ctx_max, float(np.max(ctx)))
        ca3_min, ca3_max = min(ca3_min, float(np.min(ca3))), max(ca3_max, float(np.max(ca3)))
    return AffineScaler.from_range(ctx_min, ctx_max), AffineScaler.from_range(ca3_min, ca3_max)


def _stats_numpy(
    x: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    Win: np.ndarray,
    bias: np.ndarray,
    leak: float,
    washout: int,
) -> tuple[np.ndarray, np.ndarray]:
    units = W.shape[0]
    gram = np.zeros((units + 1, units + 1), dtype=np.float64)
    target = np.zeros(units + 1, dtype=np.float64)
    state = np.zeros(units, dtype=np.float64)
    phi = np.empty(units + 1, dtype=np.float64)
    phi[0] = 1.0
    for sample in range(x.size):
        state = (1.0 - leak) * state + leak * np.tanh(
            W @ state + Win[:, 0] * x[sample] + bias
        )
        if sample >= washout:
            phi[1:] = state
            gram += np.outer(phi, phi)
            target += phi * y[sample]
    return gram, target


def _stats_runner():
    try:
        from numba import njit

        return njit(cache=True, fastmath=False)(_stats_numpy)
    except ImportError:
        return _stats_numpy


def refit_readout(
    pairs: Sequence[RecordingPair],
    artifact: InferenceArtifact,
    input_scaler: AffineScaler,
    target_scaler: AffineScaler,
    *,
    ridge: float,
    washout_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative")
    washout = int(round(washout_seconds * artifact.fs_model_hz))
    runner = _stats_runner()
    units = artifact.W.shape[0]
    gram = np.zeros((units + 1, units + 1), dtype=np.float64)
    target = np.zeros(units + 1, dtype=np.float64)
    for pair in pairs:
        ctx, ca3 = _pair_values(pair, artifact)
        if ctx.size <= washout:
            raise ValueError(f"{pair.recording_id} is shorter than the washout")
        pair_gram, pair_target = runner(
            input_scaler.transform(ctx),
            target_scaler.transform(ca3),
            artifact.W,
            artifact.Win,
            artifact.reservoir_bias,
            artifact.leak_rate,
            washout,
        )
        gram += pair_gram
        target += pair_target
    # Match a conventional fit-intercept ridge readout: reservoir weights are
    # regularized, while the explicit intercept is not.
    penalty = np.eye(units + 1, dtype=np.float64)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(gram + penalty * ridge, target)
    return coefficients[1:].reshape(units, 1), coefficients[:1]


def _predict_scaled(
    x: np.ndarray, artifact: InferenceArtifact, Wout: np.ndarray, readout_bias: np.ndarray
) -> np.ndarray:
    state = np.zeros(artifact.W.shape[0], dtype=np.float64)
    result = np.empty(x.size, dtype=np.float64)
    for index, value in enumerate(x):
        state = (1.0 - artifact.leak_rate) * state + artifact.leak_rate * np.tanh(
            artifact.W @ state + artifact.Win[:, 0] * value + artifact.reservoir_bias
        )
        result[index] = float(Wout[:, 0] @ state + readout_bias[0])
    return result


def _metrics(target: np.ndarray, prediction: np.ndarray, washout: int) -> dict[str, float]:
    target = target[washout:]
    prediction = prediction[washout:]
    error = prediction - target
    return {
        "pearson_r": float(np.corrcoef(target, prediction)[0, 1]),
        "nrmse": float(np.sqrt(np.mean(error * error)) / max(float(np.std(target)), 1e-12)),
    }


def validate_readout(
    pairs: Sequence[RecordingPair],
    artifact: InferenceArtifact,
    input_scaler: AffineScaler,
    target_scaler: AffineScaler,
    Wout: np.ndarray,
    readout_bias: np.ndarray,
    *,
    washout_seconds: float,
) -> dict[str, object]:
    washout = int(round(washout_seconds * artifact.fs_model_hz))
    rows: list[dict[str, object]] = []
    for pair in pairs:
        ctx, ca3 = _pair_values(pair, artifact)
        corrected_scaled = _predict_scaled(
            input_scaler.transform(ctx), artifact, Wout, readout_bias
        )
        corrected_physical = (corrected_scaled - target_scaler.offset) / target_scaler.scale

        # The historical readout must be evaluated using the historical
        # scaler coefficients on both sides. Comparing it in the corrected
        # target-scaled domain would penalize it for a coordinate mismatch
        # rather than prediction error.
        legacy_input = ctx * artifact.input_scale[0] + artifact.input_offset[0]
        legacy_scaled = _predict_scaled(
            legacy_input, artifact, artifact.Wout, artifact.readout_bias
        )
        legacy_physical = (
            legacy_scaled - artifact.target_offset[0]
        ) / artifact.target_scale[0]
        rows.append(
            {
                "recording_id": pair.recording_id,
                "corrected": _metrics(ca3, corrected_physical, washout),
                "legacy_readout": _metrics(ca3, legacy_physical, washout),
            }
        )
    corrected_r = float(np.mean([dict(row["corrected"])["pearson_r"] for row in rows]))
    legacy_r = float(np.mean([dict(row["legacy_readout"])["pearson_r"] for row in rows]))
    corrected_nrmse = float(np.mean([dict(row["corrected"])["nrmse"] for row in rows]))
    legacy_nrmse = float(np.mean([dict(row["legacy_readout"])["nrmse"] for row in rows]))
    passed = corrected_r >= legacy_r and corrected_nrmse <= legacy_nrmse
    return {
        "passed": passed,
        "gate": "mean Pearson r not lower and mean nRMSE not higher than legacy readout",
        "corrected_mean_pearson_r": corrected_r,
        "legacy_mean_pearson_r": legacy_r,
        "corrected_mean_nrmse": corrected_nrmse,
        "legacy_mean_nrmse": legacy_nrmse,
        "recordings": rows,
    }


def train_from_pairs(
    *,
    training_pairs: Sequence[RecordingPair],
    validation_pairs: Sequence[RecordingPair],
    seed_artifact_path: Path,
    output_path: Path,
    ridge: float = 4.4065269894699847e-10,
    washout_seconds: float = 1.0,
) -> tuple[Path, Path]:
    if not training_pairs or not validation_pairs:
        raise ValueError("both training and held-out validation pairs are required")
    seed = load_inference_artifact(seed_artifact_path)
    input_scaler, target_scaler = fit_scalers(training_pairs, seed)
    Wout, readout_bias = refit_readout(
        training_pairs, seed, input_scaler, target_scaler,
        ridge=ridge, washout_seconds=washout_seconds,
    )
    validation = validate_readout(
        validation_pairs, seed, input_scaler, target_scaler, Wout, readout_bias,
        washout_seconds=washout_seconds,
    )
    arrays = {
        "W": seed.W,
        "Win": seed.Win,
        "reservoir_bias": seed.reservoir_bias,
        "Wout": Wout,
        "readout_bias": readout_bias,
        "initial_state": np.zeros_like(seed.initial_state),
        "input_scale": np.asarray([input_scaler.scale]),
        "input_offset": np.asarray([input_scaler.offset]),
        "target_scale": np.asarray([target_scaler.scale]),
        "target_offset": np.asarray([target_scaler.offset]),
    }
    all_pairs = list(training_pairs) + list(validation_pairs)
    recording_hashes: dict[str, str] = {}
    for pair in all_pairs:
        role = pair.recording_id.split("__", 1)[0]
        ctx_folder, ca3_folder = (
            ("CTX", "CA3") if role == "training" else ("X_data", "Y_data")
        )
        recording_hashes[f"{role}/{ctx_folder}/{pair.ctx_path.name}"] = _sha256(pair.ctx_path)
        recording_hashes[f"{role}/{ca3_folder}/{pair.ca3_path.name}"] = _sha256(pair.ca3_path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "production_closed_loop_inference",
        "production_eligible": bool(validation["passed"]),
        "model": {
            "activation": "tanh", "units": int(seed.W.shape[0]),
            "leak_rate": seed.leak_rate, "readout": "ridge_linear_with_bias",
        },
        "preprocessing": seed.manifest["preprocessing"],
        "scaling": {
            "formula": "scaled = physical * scale + offset",
            "separate_input_target_scalers": True,
            "fit_scope": "all paired training recordings only",
            "input_data_min": input_scaler.data_min, "input_data_max": input_scaler.data_max,
            "target_data_min": target_scaler.data_min, "target_data_max": target_scaler.data_max,
        },
        "training": {
            "preserved_reservoir_weights": True, "ridge": ridge,
            "washout_seconds": washout_seconds,
            "pairing_rule": "ascending electrode rank within each experiment group",
            "training_recording_ids": [pair.recording_id for pair in training_pairs],
            "validation_recording_ids": [pair.recording_id for pair in validation_pairs],
        },
        "validation": validation,
        "sources": {
            "seed_artifact": {"name": seed_artifact_path.name, "sha256": _sha256(seed_artifact_path)},
            "missing_fs_assumption_hz": seed.fs_model_hz,
            "missing_fs_basis": (
                "legacy validation X_data/Y_data omit fs; Paper_figures.py and "
                "the pipeline audit identify them as the 2 kHz common-rate data"
            ),
            "recordings": dict(sorted(recording_hashes.items())),
        },
        "pulse": {"waveform": "sine_half_cycle", "duration_ms": 5.0, "frequency_hz": 100.0},
    }
    return save_inference_artifact(output_path, arrays=arrays, manifest=manifest)


def train_from_mat(
    *,
    x_dir: str,
    y_dir: str,
    validation_x_dir: str,
    validation_y_dir: str,
    seed_artifact_path: str,
    out_path: str,
    **_: object,
) -> str:
    artifact, _ = train_from_pairs(
        training_pairs=pair_recordings(Path(x_dir), Path(y_dir), role="training"),
        validation_pairs=pair_recordings(
            Path(validation_x_dir), Path(validation_y_dir), role="validation"
        ),
        seed_artifact_path=Path(seed_artifact_path), output_path=Path(out_path),
    )
    return str(artifact)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-ctx", required=True, type=Path)
    parser.add_argument("--training-ca3", required=True, type=Path)
    parser.add_argument("--validation-ctx", required=True, type=Path)
    parser.add_argument("--validation-ca3", required=True, type=Path)
    parser.add_argument("--seed-artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ridge", type=float, default=4.4065269894699847e-10)
    parser.add_argument("--washout-seconds", type=float, default=1.0)
    args = parser.parse_args()
    artifact, manifest = train_from_pairs(
        training_pairs=pair_recordings(args.training_ctx, args.training_ca3, role="training"),
        validation_pairs=pair_recordings(args.validation_ctx, args.validation_ca3, role="validation"),
        seed_artifact_path=args.seed_artifact, output_path=args.output,
        ridge=args.ridge, washout_seconds=args.washout_seconds,
    )
    print(json.dumps({"artifact": str(artifact), "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
