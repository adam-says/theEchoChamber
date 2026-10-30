"""Finite, calibrated replay input for offline and recorded electrophysiology data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np


CANONICAL_SCHEMA = "echo-chamber-replay"
CANONICAL_VERSION = 1
LEGACY_TAGS = {"echoChamber_H5_v1", "echoChamber_H5_v2", "echoChamber_H5_v3", "echoChamber_H5_v4", "echoChamber_H5_v5"}


@dataclass(frozen=True)
class ReplaySegment:
    """Contiguous samples and their source-absolute starting index."""

    sample_offset: int
    sample_count: int
    absolute_start: int


@dataclass(frozen=True)
class ReplayData:
    """A finite replay in calibrated electrode millivolts.

    ``electrode_mV`` is channels-by-samples. ``raw_V`` is available only for
    legacy Echo Chamber recordings that stored acquired voltages.
    """

    electrode_mV: np.ndarray
    sample_rate_hz: float
    channel_labels: tuple[str, str]
    channel_roles: tuple[str, str]
    ctx_index: int
    metadata: dict[str, Any]
    raw_V: np.ndarray | None = None
    segments: tuple[ReplaySegment, ...] = ()

    @property
    def sample_count(self) -> int:
        return int(self.electrode_mV.shape[1])

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate_hz

    def iter_chunks(self, chunk_samples: int) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (absolute sample start, data) chunks exactly once; never wraps."""
        if chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive")
        segments = self.segments or (ReplaySegment(0, self.sample_count, 0),)
        for segment in segments:
            stop = segment.sample_offset + segment.sample_count
            for start in range(segment.sample_offset, stop, chunk_samples):
                chunk_stop = min(start + chunk_samples, stop)
                absolute_start = segment.absolute_start + start - segment.sample_offset
                yield absolute_start, self.electrode_mV[:, start:chunk_stop]


def load_replay(path: str | Path) -> ReplayData:
    """Load canonical replay HDF5 or Echo Chamber HDF5 v1-v4."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        schema = _text(f.attrs.get("schema", ""))
        if schema == CANONICAL_SCHEMA:
            return _load_canonical(f, path)
        if schema != "echo-chamber-recording":
            raise ValueError(f"unsupported replay schema in {path}: {schema or '<missing>'}")
        return _load_legacy(f, path)


def _load_canonical(f: h5py.File, path: Path) -> ReplayData:
    version = int(f.attrs.get("schema_version", -1))
    if version != CANONICAL_VERSION:
        raise ValueError(f"unsupported canonical replay version {version}: {path}")
    signals = f.get("signals/electrode_mV")
    if signals is None or signals.ndim != 2 or signals.shape[0] != 2:
        raise ValueError(f"canonical replay requires signals/electrode_mV with shape (2,N): {path}")
    if _text(signals.attrs.get("unit", "")) != "mV":
        raise ValueError("canonical electrode_mV unit must be exactly mV")
    count = min(int(f.attrs.get("committed_samples", signals.shape[1])), signals.shape[1])
    if count <= 0:
        raise ValueError("replay contains no committed samples")
    labels = tuple(_text(x) for x in signals.attrs.get("channel_labels", ()))
    roles = tuple(_text(x) for x in signals.attrs.get("channel_roles", ()))
    regions = tuple(_text(x) for x in signals.attrs.get("channel_regions", ()))
    if len(labels) != 2 or len(roles) != 2 or len(regions) != 2:
        raise ValueError("canonical replay requires two channel labels, roles, and regions")
    _validate_roles(roles)
    fs = float(f.attrs.get("sample_rate_hz", 0))
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sample_rate_hz must be positive")
    values = np.asarray(signals[:, :count], dtype=np.float32)
    _validate_values(values)
    metadata = _metadata(f)
    metadata["channel_regions"] = list(regions)
    ctx_indices = [i for i, role in enumerate(roles) if role.casefold() in {"cortex", "ctx", "ec"}]
    segments = _segments_from_metadata(metadata, count)
    return ReplayData(values, fs, labels, roles, ctx_indices[0], metadata, segments=segments)


def _load_legacy(f: h5py.File, path: Path) -> ReplayData:
    tag = _text(f.attrs.get("format_tag", ""))
    if tag and tag not in LEGACY_TAGS:
        raise ValueError(f"unsupported Echo Chamber format tag: {tag}")
    metadata = _metadata(f)
    config = metadata.get("configuration", {})
    sample_ids = None
    if "signals/electrode_mV" in f and tag == "echoChamber_H5_v5":
        ds = f["signals/electrode_mV"]
        if _text(ds.attrs.get("unit", "")) != "mV":
            raise ValueError("Echo Chamber v5 electrode_mV unit must be exactly mV")
        labels = tuple(_text(x) for x in ds.attrs.get("channel_labels", ()))
        roles = tuple(_text(x) for x in ds.attrs.get("channel_roles", ()))
        regions = tuple(_text(x) for x in ds.attrs.get("channel_regions", ()))
        _validate_roles(roles)
        if len(regions) != 2:
            raise ValueError("Echo Chamber v5 electrode_mV requires two channel_regions")
        metadata["channel_regions"] = list(regions)
        ctx_index = next(i for i, role in enumerate(roles) if role.casefold() == "cortex")
        fs = float(f.attrs.get("sample_rate_hz", 0))
        first = int(f.attrs.get("first_ai_sample_index", 0))
        count = min(int(f.attrs.get("committed_samples", ds.shape[1])), ds.shape[1])
        mv = np.asarray(ds[:, :count], dtype=np.float32)
        raw_ds = f.get("signals/ai_raw_V")
        raw = None if raw_ds is None else np.asarray(raw_ds[:, :count], dtype=np.float32)
        if raw is not None and not np.isfinite(raw).all():
            raw = None
    elif "signals/ai_raw_V" in f:
        ds = f["signals/ai_raw_V"]
        labels = tuple(_text(x) for x in ds.attrs.get("channel_labels", ()))
        gains = tuple(float(x) for x in config.get("amplifier_gain", ()))
        if len(labels) != 2 or len(gains) != 2 or any(not np.isfinite(x) or x <= 0 for x in gains):
            raise ValueError("legacy v3-v4 requires two labels and positive amplifier_gain metadata")
        fs = float(f.attrs.get("sample_rate_hz", config.get("sample_rate", 0)))
        first = int(f.attrs.get("first_ai_sample_index", 0))
        count = min(int(f.attrs.get("committed_samples", ds.shape[1])), ds.shape[1])
        raw = np.asarray(ds[:, :count], dtype=np.float32)
        mv = raw * (1000.0 / np.asarray(gains, dtype=np.float32)[:, None])
    else:
        if "data" not in f or "row_names" not in f:
            raise ValueError(f"legacy HDF5 recording is missing signal datasets: {path}")
        rows = tuple(_text(x) for x in f["row_names"][:])
        labels_found = [x[: -len("_raw_V")] for x in rows if x.endswith("_raw_V")]
        if len(labels_found) != 2:
            raise ValueError("legacy v1-v2 requires exactly two *_raw_V rows")
        gains = tuple(float(x) for x in config.get("amplifier_gain", ()))
        if len(gains) != 2 or any(not np.isfinite(x) or x <= 0 for x in gains):
            raise ValueError("legacy v1-v2 requires amplifier_gain metadata")
        indexes = [rows.index(f"{label}_raw_V") for label in labels_found]
        count = min(int(f.attrs.get("committed_samples", f["data"].shape[1])), f["data"].shape[1])
        raw = np.asarray(f["data"][indexes, :count], dtype=np.float32)
        mv = raw * (1000.0 / np.asarray(gains, dtype=np.float32)[:, None])
        fs = float(config.get("sample_rate", 0))
        if "ai_sample_index" in rows:
            sample_ids = np.asarray(f["data"][rows.index("ai_sample_index"), :count], dtype=np.int64)
            first = int(sample_ids[0]) if count else 0
        else:
            first = 0
        labels = tuple(labels_found)
    if len(labels) != 2:
        raise ValueError("recording requires exactly two channel labels")
    if count <= 0:
        raise ValueError("replay contains no committed samples")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("legacy sample rate must be positive")
    _validate_values(mv)
    sidecar_path = path.with_suffix(".json")
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("format_tag") != "echo-chamber-replay-sidecar-v1":
            raise ValueError(f"unsupported replay sidecar format: {sidecar_path}")
        if sidecar.get("data_sha256") != _sha256_file(path):
            raise ValueError(f"replay sidecar checksum mismatch: {sidecar_path}")
        sidecar_labels = tuple(str(x) for x in sidecar.get("channel_labels_by_index", ()))
        ctx_index = int(sidecar.get("ctx_index", -1))
        if len(sidecar_labels) != 2 or ctx_index not in (0, 1):
            raise ValueError(f"invalid authoritative channel mapping in {sidecar_path}")
        labels = sidecar_labels
        roles = tuple("Cortex" if i == ctx_index else "CA3" for i in range(2))
        metadata["replay_sidecar"] = sidecar
    elif tag != "echoChamber_H5_v5":
        roles = tuple("CA3" if "ca3" in label.casefold() else "Cortex" if any(k in label.casefold() for k in ("ctx", "cortex", "ec")) else "unknown" for label in labels)
        ctx_index = next((i for i, role in enumerate(roles) if role == "Cortex"), -1)
    _validate_roles(roles)
    segments, gaps = _legacy_segments(f, first, count, sample_ids)
    if gaps:
        metadata["gaps"] = gaps
    metadata["segments"] = [segment.__dict__ for segment in segments]
    metadata.update({"legacy_format_tag": tag, "first_ai_sample_index": first})
    if tag != "echoChamber_H5_v5":
        metadata.update({"legacy_amplifier_gain": gains,
                         "calibration_note": "legacy stored volts converted using recorded amplifier_gain"})
    return ReplayData(mv, fs, labels, roles, ctx_index, metadata, raw, segments)


def _validate_roles(roles: tuple[str, ...]) -> None:
    normalized = tuple(role.casefold() for role in roles)
    if len(roles) != 2 or normalized.count("ca3") != 1 or normalized.count("cortex") != 1:
        raise ValueError("replay requires exactly one CA3 role and one Cortex role")


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_segments(f: h5py.File, first: int, count: int,
                     sample_ids: np.ndarray | None = None) -> tuple[tuple[ReplaySegment, ...], list[dict[str, int]]]:
    gaps: list[dict[str, int]] = []
    events = f.get("events/sample_discontinuities")
    if events is not None:
        previous_offset = 0
        previous_absolute = first
        for event in events[:]:
            offset = int(event["sample_offset"])
            expected = int(event["expected_ai_sample_index"])
            actual = int(event["actual_ai_sample_index"])
            expected_from_prior = previous_absolute + offset - previous_offset
            if (offset <= previous_offset or offset >= count or actual == expected
                    or expected != expected_from_prior):
                raise ValueError("invalid sample discontinuity metadata")
            gaps.append({"sample_offset": offset, "expected_index": expected, "actual_index": actual})
            previous_offset, previous_absolute = offset, actual
    if not gaps and sample_ids is not None:
        if sample_ids.shape != (count,):
            raise ValueError("legacy AI sample index row length does not match signal length")
        for offset in np.flatnonzero(np.diff(sample_ids) != 1) + 1:
            gaps.append({"sample_offset": int(offset), "expected_index": int(sample_ids[offset - 1] + 1),
                         "actual_index": int(sample_ids[offset])})
    if not gaps:
        return (ReplaySegment(0, count, first),), []
    starts = [(0, first)] + [(gap["sample_offset"], gap["actual_index"]) for gap in gaps]
    segments = tuple(
        ReplaySegment(offset, (starts[i + 1][0] if i + 1 < len(starts) else count) - offset, absolute)
        for i, (offset, absolute) in enumerate(starts)
    )
    return segments, gaps


def _segments_from_metadata(metadata: dict[str, Any], count: int) -> tuple[ReplaySegment, ...]:
    specs = metadata.get("segments")
    if not specs:
        return (ReplaySegment(0, count, int(metadata.get("first_sample_index", 0))),)
    segments = tuple(ReplaySegment(int(s["start_sample"]), int(s["sample_count"]),
                                   int(s.get("absolute_start", s["start_sample"]))) for s in specs)
    cursor = 0
    for segment in segments:
        if segment.sample_offset != cursor or segment.sample_count <= 0:
            raise ValueError("canonical segment metadata must cover samples contiguously without overlap")
        cursor += segment.sample_count
    if cursor != count:
        raise ValueError("canonical segment metadata sample counts do not match signal length")
    return segments


def _metadata(f: h5py.File) -> dict[str, Any]:
    value = f.attrs.get("metadata_json", "{}")
    try:
        return json.loads(_text(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("metadata_json is invalid") from exc


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _validate_values(values: np.ndarray) -> None:
    if values.ndim != 2 or values.shape[0] != 2 or not np.isfinite(values).all():
        raise ValueError("replay signals must be finite with shape (2,N)")
