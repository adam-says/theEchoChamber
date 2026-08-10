from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Tuple

import numpy as np

from .io import save_artifact
from .model import DEFAULT_HYPERPARAMS, ESNModel, build_esn


def _load_mat_signal(path: str) -> np.ndarray:
    try:
        import mat73

        mat = mat73.loadmat(path)
        data = mat.get("data")
        if data is None:
            raise KeyError("Missing 'data' key")
        return np.asarray(data, dtype=np.float64).reshape(-1, 1)
    except Exception:
        import scipy.io

        mat = scipy.io.loadmat(path)
        data = mat.get("data")
        if data is None:
            raise KeyError("Missing 'data' key")
        return np.asarray(data, dtype=np.float64).reshape(-1, 1)


def _resample_if_needed(x: np.ndarray, fs_src: float, fs_dst: float) -> np.ndarray:
    if fs_src == fs_dst:
        return x
    from scipy.signal import resample

    n_dst = int(round(x.shape[0] * fs_dst / fs_src))
    y = resample(x[:, 0], n_dst).reshape(-1, 1)
    return y


def train_from_mat(
    *,
    x_dir: str,
    y_dir: str,
    fs_native: float,
    common_fs: int = 2000,
    cutoff_hz: float = 25.0,
    fir_numtaps: int = 401,
    out_path: str = "esn_artifact.pkl",
    hyperparams: Dict[str, Any] | None = None,
) -> str:
    """
    Training pipeline matching the notebook:
    - load .mat files from x_dir (CTX) and y_dir (CA3)
    - resample to common_fs if needed
    - causal FIR lowpass
    - MinMaxScaler(-1,1)
    - reservoirpy ESN fit
    - save artifact
    """
    from scipy.signal import firwin, lfilter
    from sklearn.preprocessing import MinMaxScaler

    hp = dict(DEFAULT_HYPERPARAMS)
    if hyperparams:
        hp.update(hyperparams)

    x_files = sorted([f for f in os.listdir(x_dir) if f.lower().endswith(".mat")])
    y_files = sorted([f for f in os.listdir(y_dir) if f.lower().endswith(".mat")])
    if not x_files or not y_files:
        raise ValueError("No .mat files found in x_dir or y_dir.")

    # For now: train on first pair (keeps behavior close to ESN_test_working.ipynb which used first channel).
    x = _load_mat_signal(os.path.join(x_dir, x_files[0]))
    y = _load_mat_signal(os.path.join(y_dir, y_files[0]))

    x = _resample_if_needed(x, fs_native, common_fs)
    y = _resample_if_needed(y, fs_native, common_fs)

    b = firwin(fir_numtaps, cutoff_hz, window="hamming", pass_zero="lowpass", scale=True, fs=common_fs)
    x_f = lfilter(b, [1.0], x[:, 0]).reshape(-1, 1)
    y_f = lfilter(b, [1.0], y[:, 0]).reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(x_f)
    x_s = scaler.transform(x_f)
    scaler.fit(y_f)
    y_s = scaler.transform(y_f)

    model_graph = build_esn(
        units=int(hp["units"]),
        sr=float(hp["sr"]),
        lr=float(hp["lr"]),
        iss=float(hp["iss"]),
        ridge=float(hp["ridge"]),
        seed=int(hp["seed"]),
    )
    esn = ESNModel(model=model_graph, fs_train=common_fs)
    esn.fit(x_s, y_s, reset=True)

    config = {
        "fs_in": 20000,
        "fs_train": common_fs,
        "chunk_size": 100,
        "cutoff_hz": cutoff_hz,
        "fir_numtaps": fir_numtaps,
        "up_method": "linear",
        "stim_mode": "passthrough",
        "stim_gain": 1.0,
        "hyperparams": hp,
    }
    save_artifact(out_path, esn_model=esn.model, scaler=scaler, config=config)
    return out_path


def _main() -> None:
    p = argparse.ArgumentParser(description="Train ESN artifact from .mat LFP data.")
    p.add_argument("--x-dir", required=True, help="Directory with CTX .mat files (key: data).")
    p.add_argument("--y-dir", required=True, help="Directory with CA3 .mat files (key: data).")
    p.add_argument("--fs-native", required=True, type=float, help="Native sampling rate of .mat files.")
    p.add_argument("--common-fs", default=2000, type=int, help="Training sampling rate after resampling.")
    p.add_argument("--out", default="esn_artifact.pkl", help="Output artifact path.")
    args = p.parse_args()

    out = train_from_mat(
        x_dir=args.x_dir,
        y_dir=args.y_dir,
        fs_native=args.fs_native,
        common_fs=args.common_fs,
        out_path=args.out,
    )
    print(out)


if __name__ == "__main__":
    _main()

