from __future__ import annotations

import pickle
from dataclasses import asdict
from typing import Any, Dict, Optional

from .model import ESNModel
from .streaming import ESNStreamer, ThresholdPulseConfig


def save_artifact(
    path: str,
    *,
    esn_model: Any,
    scaler: Any,
    config: Dict[str, Any],
) -> None:
    """
    Save a lean artifact: {esn_model, scaler, config}.
    """
    payload = {
        "esn_model": esn_model,
        "scaler": scaler,
        "config": dict(config),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_artifact(path: str) -> ESNStreamer:
    """
    Load artifact and return a configured ESNStreamer.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    esn_model = payload["esn_model"]
    scaler = payload["scaler"]
    cfg = payload.get("config", {})

    fs_in = int(cfg.get("fs_in", 20000))
    fs_train = int(cfg.get("fs_train", 2000))
    chunk_size = int(cfg.get("chunk_size", 100))
    cutoff_hz = float(cfg.get("cutoff_hz", 25.0))
    fir_numtaps = int(cfg.get("fir_numtaps", 8000))
    up_method = cfg.get("up_method", "linear")

    stim_mode = cfg.get("stim_mode", "passthrough")
    stim_gain = float(cfg.get("stim_gain", 1.0))

    pulse_cfg_dict = cfg.get("pulse_cfg", None)
    pulse_cfg = ThresholdPulseConfig(**pulse_cfg_dict) if isinstance(pulse_cfg_dict, dict) else ThresholdPulseConfig()

    esn = ESNModel(model=esn_model, fs_train=fs_train)
    streamer = ESNStreamer(
        esn=esn,
        scaler=scaler,
        fs_in=fs_in,
        fs_train=fs_train,
        chunk_size=chunk_size,
        cutoff_hz=cutoff_hz,
        fir_numtaps=fir_numtaps,
        up_method=up_method,
        stim_mode=stim_mode,
        stim_gain=stim_gain,
        pulse_cfg=pulse_cfg,
    )
    return streamer


def extract_from_notebook_pkl(src: str = "objs.pkl", dst: str = "esn_artifact.pkl") -> None:
    """
    Extract only the pieces needed for streaming from the large notebook pickle.

    The notebook stores: [esn_model, reservoir, readout, states, Y_pred, scaler]
    This function writes {esn_model, scaler, config} to dst.
    """
    with open(src, "rb") as f:
        objs = pickle.load(f)

    if isinstance(objs, (list, tuple)) and len(objs) >= 6:
        esn_model = objs[0]
        scaler = objs[5]
    elif isinstance(objs, dict) and "esn_model" in objs:
        esn_model = objs["esn_model"]
        scaler = objs.get("scaler")
    else:
        raise ValueError(f"Unrecognized pickle format in {src}")

    config = {
        "fs_in": 20000,
        "fs_train": 2000,
        "chunk_size": 100,
        "cutoff_hz": 25.0,
        "fir_numtaps": 8000,
        "up_method": "linear",
        "stim_mode": "passthrough",
        "stim_gain": 1.0,
        "pulse_cfg": asdict(ThresholdPulseConfig()),
    }
    save_artifact(dst, esn_model=esn_model, scaler=scaler, config=config)

