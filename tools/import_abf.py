"""Offline, calibrated ABF2-to-Echo-Chamber replay importer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "ca3_slow_waves_sources.json"
EXPECTED_FILES = ["2026_05_29_0001.abf", "2026_08_05_0000.abf", "2026_08_06_0000.abf"]
EXPECTED_BYTES = 144_018_432


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_abf(path: Path) -> tuple[np.ndarray, float, dict[str, Any]]:
    try:
        import pyabf
    except ImportError as exc:
        raise RuntimeError("ABF import requires the optional dependency; install requirements-import.txt") from exc
    if getattr(pyabf, "__version__", None) != "2.3.8":
        raise RuntimeError(f"ABF importer requires pyABF 2.3.8, found {getattr(pyabf, '__version__', '<unknown>')}")
    abf = pyabf.ABF(str(path))
    version = getattr(abf, "abfVersion", {})
    major_version = int(version.get("major", 0)) if isinstance(version, dict) else int(version)
    if major_version < 2:
        raise ValueError(f"unsupported ABF version (ABF2 required): {path.name}")
    if int(abf.channelCount) != 2:
        raise ValueError(f"expected exactly 2 channels, found {abf.channelCount}: {path.name}")
    if int(abf.dataRate) != 20_000:
        raise ValueError(f"expected 20000 Hz, found {abf.dataRate}: {path.name}")
    labels = tuple(str(x).strip() for x in abf.adcNames[:2])
    units = tuple(str(x).strip().casefold() for x in abf.adcUnits[:2])
    if units != ("mv", "mv"):
        raise ValueError(f"expected ABF stored units mV for both channels; found {units}: {path.name}")
    operation_mode = int(getattr(getattr(abf, "_protocolSection", None), "nOperationMode", -1))
    if operation_mode != 3 or int(abf.sweepCount) != 1:
        raise ValueError(f"expected one continuous gap-free sweep (ABF operation mode 3), found mode={operation_mode}, sweeps={abf.sweepCount}: {path.name}")
    adc_numbers = tuple(int(x) for x in getattr(getattr(abf, "_adcSection", None), "nADCNum", ()))
    if adc_numbers != (0, 2):
        raise ValueError(f"expected physical ADC mapping position 0=ADC0 and position 1=ADC2: {path.name}")
    # pyABF's sweepY values are already scaled to adcUnits; do not apply gain.
    abf.setSweep(0, channel=0)
    ch0 = np.asarray(abf.sweepY, dtype=np.float32).copy()
    abf.setSweep(0, channel=1)
    ch1 = np.asarray(abf.sweepY, dtype=np.float32).copy()
    values = np.vstack((ch0, ch1))
    expected = 20_000 * 1_800
    if values.shape != (2, expected):
        raise ValueError(f"expected 2 x 36000000 samples (30 minutes), found {values.shape}: {path.name}")
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite calibrated signal values in {path.name}")
    return values, float(abf.dataRate), {
        "abf_version": str(version), "data_format": str(getattr(abf, "dataFormat", "")),
        "channel_labels": labels, "stored_units": units,
        "physical_adc_numbers": list(adc_numbers), "operation_mode": operation_mode,
        "sweep_count": int(abf.sweepCount),
        "sweep_length_seconds": float(abf.sweepLengthSec),
        "data_point_count": int(abf.dataPointCount),
    }


def write_replay(source: Path, destination: Path, *, allow_overwrite: bool = False) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size != EXPECTED_BYTES:
        raise ValueError(f"source file size mismatch: expected {EXPECTED_BYTES}, got {source.stat().st_size}")
    if destination.exists() and not allow_overwrite:
        raise FileExistsError(f"refusing to overwrite existing replay: {destination}")
    values, fs, header = read_abf(source)
    source_hash = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": {"filename": source.name, "size_bytes": source.stat().st_size,
                   "sha256": source_hash, "header": header},
        "calibration": {"signal_unit": "mV", "method": "pyABF calibrated ADC values in ABF stored units",
                        "applied_amplifier_gain": False},
        "channel_mapping": [
            {"position": 0, "label": "CA3", "region": "CA3", "role": "CA3", "physical_adc": 0},
            {"position": 1, "label": "EC", "region": "EC", "role": "Cortex", "physical_adc": 2},
        ],
        "segments": [{"start_sample": 0, "sample_count": int(values.shape[1]), "continuous": True}],
        "gaps": [],
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "importer": "tools/import_abf.py", "pyabf_version": __import__("pyabf").__version__,
    }
    fd, temp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        with h5py.File(temp_name, "w") as out:
            out.attrs["schema"] = "echo-chamber-replay"
            out.attrs["schema_version"] = 1
            out.attrs["sample_rate_hz"] = fs
            out.attrs["committed_samples"] = values.shape[1]
            out.attrs["complete"] = np.uint8(1)
            out.attrs["metadata_json"] = json.dumps(metadata, separators=(",", ":"))
            group = out.create_group("signals")
            ds = group.create_dataset("electrode_mV", data=values, dtype="float32", chunks=(2, min(200_000, values.shape[1])), compression="gzip", compression_opts=4, shuffle=True)
            ds.attrs["unit"] = "mV"
            ds.attrs["channel_labels"] = np.asarray(["CA3", "EC"], dtype=h5py.string_dtype("utf-8"))
            ds.attrs["channel_roles"] = np.asarray(["CA3", "Cortex"], dtype=h5py.string_dtype("utf-8"))
            ds.attrs["channel_regions"] = np.asarray(["CA3", "EC"], dtype=h5py.string_dtype("utf-8"))
            out.flush()
        if allow_overwrite:
            os.replace(temp_name, destination)
        else:
            # Same-volume rename is atomic and fails if another importer won
            # the race to create this destination.
            os.rename(temp_name, destination)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return {"source": str(source), "destination": str(destination), "source_sha256": source_hash,
            "sample_count": int(values.shape[1]), "sample_rate_hz": fs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source ABF2 file")
    parser.add_argument("destination", type=Path, help="new canonical .h5 output")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    args = parser.parse_args()
    result = write_replay(args.source, args.destination, allow_overwrite=args.overwrite)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
