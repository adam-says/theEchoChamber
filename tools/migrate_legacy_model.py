"""Offline migration of a trusted ReservoirPy pickle to a numeric artifact.

Pickle can execute code while loading.  Run this tool only on the historical
Echo State Network files from the project authors.  The live application never
imports this module and never accepts pickle files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from esn.artifact import save_inference_artifact  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dense(value: Any) -> np.ndarray:
    return np.asarray(value.toarray() if hasattr(value, "toarray") else value, dtype=np.float64)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as source:
        return pickle.load(source)


def migrate(model_path: Path, secondary_path: Path, output_path: Path) -> tuple[Path, Path]:
    primary = _load_pickle(model_path)
    secondary = _load_pickle(secondary_path)
    if not isinstance(primary, (list, tuple)) or len(primary) < 3:
        raise ValueError("primary pickle does not contain model/reservoir/readout")
    if not isinstance(secondary, (list, tuple)) or len(secondary) < 6:
        raise ValueError("secondary pickle does not contain the historical scaler")

    reservoir = primary[1]
    readout = primary[2]
    scaler = secondary[5]
    input_scale = np.asarray(scaler.scale_, dtype=np.float64).reshape(1)
    input_offset = np.asarray(scaler.min_, dtype=np.float64).reshape(1)
    arrays = {
        "W": _dense(reservoir.W),
        "Win": _dense(reservoir.Win),
        "reservoir_bias": _dense(reservoir.bias).reshape(-1),
        "Wout": _dense(readout.Wout),
        "readout_bias": _dense(readout.bias).reshape(1),
        # The serialized node state reflects whichever notebook cell ran last,
        # not a defined deployment initial condition.  Reset-to-zero is the
        # reproducible ReservoirPy behavior used during fitting.
        "initial_state": np.zeros(_dense(reservoir.W).shape[0], dtype=np.float64),
        # The notebook refit one scaler on CA3 before serialization, so its CTX
        # coefficients are unrecoverable from the pickle.  These duplicated
        # values preserve legacy behavior for replay comparison only.
        "input_scale": input_scale,
        "input_offset": input_offset,
        "target_scale": input_scale.copy(),
        "target_offset": input_offset.copy(),
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "legacy_replay_and_numpy_equivalence_only",
        "production_eligible": False,
        "model": {
            "activation": "tanh",
            "units": int(arrays["W"].shape[0]),
            "leak_rate": float(reservoir.lr),
            "readout": "linear_with_bias",
        },
        "preprocessing": {
            "fs_input_hz": 20_000,
            "fs_model_hz": 2_000,
            "preferred_chunk_size": 100,
            "anti_alias": {"family": "fir", "cutoff_hz": 500.0, "numtaps": 201},
            "model_lowpass": {"family": "fir", "cutoff_hz": 25.0, "numtaps": 401},
            "filter_state": "zeros",
            "upsampling": "causal_linear",
        },
        "scaling": {
            "formula": "scaled = physical * scale + offset",
            "separate_input_target_scalers": False,
            "warning": (
                "Historical notebook serialized only the CA3-fitted scaler; "
                "its coefficients are duplicated for CTX solely to reproduce legacy behavior."
            ),
        },
        "training": {
            "status": "requires_corrected_refit_from_paired_source_recordings",
            "washout_seconds": 1.0,
            "ridge": float(getattr(readout, "ridge", 4.4065269894699847e-10)),
        },
        "pulse": {
            "waveform": "sine_half_cycle",
            "duration_ms": 5.0,
            "frequency_hz": 100.0,
        },
        "sources": {
            "primary_pickle": {"name": model_path.name, "sha256": _sha256(model_path)},
            "secondary_pickle": {"name": secondary_path.name, "sha256": _sha256(secondary_path)},
        },
    }
    return save_inference_artifact(output_path, arrays=arrays, manifest=manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--secondary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    npz_path, manifest_path = migrate(args.model, args.secondary, args.output)
    print(json.dumps({"artifact": str(npz_path), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
