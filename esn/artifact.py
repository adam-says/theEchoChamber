"""Safe, versioned ESN inference artifacts.

Production code deliberately never unpickles a model.  A model consists of a
compressed NumPy archive and a JSON manifest with a SHA-256 digest of that
archive.  Legacy pickle migration lives in an explicit offline tool.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FORMAT_TAG = "echo-chamber-esn-inference-v1"
REQUIRED_ARRAYS = (
    "W",
    "Win",
    "reservoir_bias",
    "Wout",
    "readout_bias",
    "initial_state",
    "input_scale",
    "input_offset",
    "target_scale",
    "target_offset",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class InferenceArtifact:
    W: np.ndarray
    Win: np.ndarray
    reservoir_bias: np.ndarray
    Wout: np.ndarray
    readout_bias: np.ndarray
    initial_state: np.ndarray
    input_scale: np.ndarray
    input_offset: np.ndarray
    target_scale: np.ndarray
    target_offset: np.ndarray
    leak_rate: float
    fs_input_hz: int
    fs_model_hz: int
    preferred_chunk_size: int
    aa_cutoff_hz: float
    aa_numtaps: int
    model_cutoff_hz: float
    model_numtaps: int
    manifest: dict[str, Any]

    def validate(self) -> None:
        arrays = {
            name: np.asarray(getattr(self, name), dtype=np.float64)
            for name in REQUIRED_ARRAYS
        }
        units = arrays["W"].shape[0]
        expected = {
            "W": (units, units),
            "Win": (units, 1),
            "reservoir_bias": (units,),
            "Wout": (units, 1),
            "readout_bias": (1,),
            "initial_state": (units,),
            "input_scale": (1,),
            "input_offset": (1,),
            "target_scale": (1,),
            "target_offset": (1,),
        }
        if units <= 0:
            raise ValueError("artifact reservoir must contain at least one unit")
        for name, value in arrays.items():
            if value.shape != expected[name]:
                raise ValueError(
                    f"artifact array {name} has shape {value.shape}; expected {expected[name]}"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"artifact array {name} contains non-finite values")
        if arrays["input_scale"][0] == 0 or arrays["target_scale"][0] == 0:
            raise ValueError("artifact scaler coefficients must be non-zero")
        if not np.isfinite(self.leak_rate) or not 0 < self.leak_rate <= 1:
            raise ValueError("artifact leak_rate must be in (0, 1]")
        if self.fs_input_hz <= 0 or self.fs_model_hz <= 0:
            raise ValueError("artifact sample rates must be positive")
        if self.fs_input_hz % self.fs_model_hz:
            raise ValueError("artifact input rate must be an integer multiple of model rate")
        if self.preferred_chunk_size <= 0:
            raise ValueError("artifact preferred chunk size must be positive")
        if self.preferred_chunk_size % (self.fs_input_hz // self.fs_model_hz):
            raise ValueError("artifact chunk size must align with the decimation factor")
        for name, cutoff, rate, taps in (
            ("anti-alias", self.aa_cutoff_hz, self.fs_input_hz, self.aa_numtaps),
            ("model-band", self.model_cutoff_hz, self.fs_model_hz, self.model_numtaps),
        ):
            if not 0 < cutoff < rate / 2:
                raise ValueError(f"{name} cutoff must be between zero and Nyquist")
            if taps < 3:
                raise ValueError(f"{name} FIR must contain at least three taps")


def _manifest_path(npz_path: Path) -> Path:
    return npz_path.with_suffix(".json")


def load_inference_artifact(path: str | Path) -> InferenceArtifact:
    npz_path = Path(path).expanduser().resolve()
    if npz_path.suffix.lower() != ".npz":
        raise ValueError("production ESN artifacts must be .npz files; pickle is not accepted")
    manifest_path = _manifest_path(npz_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_tag") != FORMAT_TAG:
        raise ValueError(f"unsupported ESN artifact format: {manifest.get('format_tag')!r}")
    expected_digest = str(manifest.get("npz_sha256", "")).lower()
    actual_digest = sha256_file(npz_path)
    if not expected_digest or actual_digest != expected_digest:
        raise ValueError("ESN artifact checksum does not match its manifest")

    with np.load(npz_path, allow_pickle=False) as stored:
        missing = [name for name in REQUIRED_ARRAYS if name not in stored.files]
        extras = sorted(set(stored.files) - set(REQUIRED_ARRAYS))
        if missing or extras:
            raise ValueError(f"invalid artifact arrays; missing={missing}, unexpected={extras}")
        arrays = {name: np.asarray(stored[name], dtype=np.float64) for name in REQUIRED_ARRAYS}

    preprocessing = manifest.get("preprocessing", {})
    model = manifest.get("model", {})
    artifact = InferenceArtifact(
        **arrays,
        leak_rate=float(model["leak_rate"]),
        fs_input_hz=int(preprocessing["fs_input_hz"]),
        fs_model_hz=int(preprocessing["fs_model_hz"]),
        preferred_chunk_size=int(preprocessing.get("preferred_chunk_size", 100)),
        aa_cutoff_hz=float(preprocessing["anti_alias"]["cutoff_hz"]),
        aa_numtaps=int(preprocessing["anti_alias"]["numtaps"]),
        model_cutoff_hz=float(preprocessing["model_lowpass"]["cutoff_hz"]),
        model_numtaps=int(preprocessing["model_lowpass"]["numtaps"]),
        manifest=manifest,
    )
    artifact.validate()
    return artifact


def save_inference_artifact(
    path: str | Path,
    *,
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    """Write an artifact and its checksum manifest.

    This helper is for offline migration/training tools.  It refuses object
    arrays so the resulting archive remains loadable with ``allow_pickle=False``.
    """

    npz_path = Path(path).expanduser().resolve()
    if npz_path.suffix.lower() != ".npz":
        raise ValueError("artifact output path must end in .npz")
    missing = [name for name in REQUIRED_ARRAYS if name not in arrays]
    extras = sorted(set(arrays) - set(REQUIRED_ARRAYS))
    if missing or extras:
        raise ValueError(f"invalid artifact arrays; missing={missing}, unexpected={extras}")
    clean: dict[str, np.ndarray] = {}
    for name in REQUIRED_ARRAYS:
        value = np.asarray(arrays[name], dtype=np.float64)
        if value.dtype.hasobject:
            raise ValueError(f"object array is forbidden: {name}")
        clean[name] = value

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **clean)
    output_manifest = dict(manifest)
    output_manifest["format_tag"] = FORMAT_TAG
    output_manifest["npz_sha256"] = sha256_file(npz_path)
    manifest_path = _manifest_path(npz_path)
    manifest_path.write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return npz_path, manifest_path
