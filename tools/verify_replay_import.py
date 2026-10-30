"""Verify canonical replay files against their original ABF2 sources."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMPORT_SPEC = importlib.util.spec_from_file_location("echo_chamber_import_abf", ROOT / "tools" / "import_abf.py")
if IMPORT_SPEC is None or IMPORT_SPEC.loader is None:
    raise RuntimeError("could not load tools/import_abf.py")
IMPORTER = importlib.util.module_from_spec(IMPORT_SPEC)
sys.modules[IMPORT_SPEC.name] = IMPORTER
IMPORT_SPEC.loader.exec_module(IMPORTER)
sys.path.insert(0, str(ROOT))
from replay import load_replay  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_one(source: Path, replay_path: Path, expected_sha256: str, chunk_samples: int) -> dict[str, Any]:
    actual_source_sha = file_sha256(source)
    if actual_source_sha != expected_sha256:
        raise ValueError(f"source SHA-256 does not match manifest: {source.name}")
    expected, fs, header = IMPORTER.read_abf(source)
    replay = load_replay(replay_path)
    if replay.sample_rate_hz != fs:
        raise ValueError(f"sample-rate mismatch for {replay_path.name}")
    if replay.channel_labels != ("CA3", "EC") or replay.channel_roles != ("CA3", "Cortex"):
        raise ValueError(f"channel mapping mismatch for {replay_path.name}")
    if (replay.ctx_index != 1 or replay.metadata["calibration"]["signal_unit"] != "mV"
            or replay.metadata.get("channel_regions") != ["CA3", "EC"]):
        raise ValueError(f"role or unit mismatch for {replay_path.name}")
    if replay.metadata["source"]["sha256"] != actual_source_sha:
        raise ValueError(f"canonical provenance SHA-256 mismatch for {replay_path.name}")
    if not np.isfinite(replay.electrode_mV).all():
        raise ValueError(f"canonical replay contains non-finite electrode mV values: {replay_path.name}")

    expected_cursor = 0
    chunk_count = 0
    chunks = iter(replay.iter_chunks(chunk_samples))
    for absolute_start, chunk in chunks:
        if absolute_start != expected_cursor:
            raise ValueError(f"unexpected gap or repeated sample at {absolute_start} in {replay_path.name}")
        expected_chunk = expected[:, expected_cursor : expected_cursor + chunk.shape[1]]
        if not np.array_equal(chunk, expected_chunk):
            raise ValueError(f"ABF-to-HDF5 values differ at sample {expected_cursor}: {replay_path.name}")
        expected_cursor += chunk.shape[1]
        chunk_count += 1
    if expected_cursor != expected.shape[1] or expected_cursor != replay.sample_count:
        raise ValueError(f"replay did not reach the expected finite EOF: {replay_path.name}")
    try:
        next(chunks)
    except StopIteration:
        pass
    else:
        raise ValueError(f"replay unexpectedly yielded samples after EOF: {replay_path.name}")
    if replay.metadata.get("gaps") not in (None, []):
        raise ValueError(f"unexpected source gaps in supplied gap-free ABF: {replay_path.name}")
    expected_segments = [{"start_sample": 0, "sample_count": int(expected.shape[1]), "continuous": True}]
    if replay.metadata.get("segments") != expected_segments:
        raise ValueError(f"segment metadata mismatch for {replay_path.name}")

    return {
        "source": source.name,
        "source_sha256": actual_source_sha,
        "canonical_replay": str(replay_path),
        "canonical_sha256": file_sha256(replay_path),
        "sample_rate_hz": fs,
        "sample_count_per_channel": int(expected.shape[1]),
        "duration_seconds": int(expected.shape[1]) / fs,
        "channel_labels": list(replay.channel_labels),
        "channel_roles": list(replay.channel_roles),
        "channel_regions": ["CA3", "EC"],
        "unit": "mV",
        "values_exactly_equal_to_pyabf": True,
        "max_absolute_difference_mV": 0.0,
        "finite_eof": True,
        "gap_count": 0,
        "segment_count": len(replay.segments),
        "verified_chunk_count": chunk_count,
        "source_header": header,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="folder containing the three ABF sources")
    parser.add_argument("--replay-dir", type=Path, required=True, help="folder containing canonical <source-stem>.h5 outputs")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "ca3_slow_waves_sources.json")
    parser.add_argument("--json-output", type=Path, help="optional path to save the verification summary")
    parser.add_argument("--chunk-samples", type=int, default=200_000)
    args = parser.parse_args()
    if args.chunk_samples <= 0:
        parser.error("--chunk-samples must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = []
    for item in manifest["files"]:
        source = args.source_dir / item["name"]
        replay_path = args.replay_dir / (source.stem + ".h5")
        results.append(verify_one(source, replay_path, item["sha256"], args.chunk_samples))
    report = {"status": "verified", "schema": "echo-chamber-replay", "schema_version": 1,
              "results": results}
    encoded = json.dumps(report, indent=2)
    if args.json_output:
        args.json_output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
