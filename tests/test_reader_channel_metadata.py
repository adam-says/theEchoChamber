from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from readEchoChamberData import load_recording


def make_calibrated_recording(path: Path, *, format_tag: str,
                              roles: tuple[str, ...] | None = None,
                              regions: tuple[str, ...] | None = None) -> Path:
    text = h5py.string_dtype(encoding="utf-8")
    mode_dtype = np.dtype([("sample_offset", "<i8"), ("mode", "u1")])
    with h5py.File(path, "w") as recording:
        recording.attrs.update(
            schema="echo-chamber-recording",
            format_tag=format_tag,
            committed_samples=3,
            sample_rate_hz=20_000.0,
            first_ai_sample_index=0,
            metadata_json=json.dumps({"configuration": {"sample_rate": 20_000.0, "ctx_index": 1}}),
        )
        signals = recording.create_group("signals")
        electrode = signals.create_dataset(
            "electrode_mV", data=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        )
        electrode.attrs["unit"] = "mV"
        electrode.attrs["channel_labels"] = np.asarray(["CA3", "EC"], dtype=text)
        if roles is not None:
            electrode.attrs["channel_roles"] = np.asarray(roles, dtype=text)
        if regions is not None:
            electrode.attrs["channel_regions"] = np.asarray(regions, dtype=text)
        signals.create_dataset("ao_command_V", data=np.zeros((1, 3), dtype=np.float32))
        events = recording.create_group("events")
        events.create_dataset("mode_changes", data=np.asarray([(0, 0)], dtype=mode_dtype))
    return path


def test_v5_requires_explicit_channel_roles_and_regions(tmp_path: Path) -> None:
    path = make_calibrated_recording(tmp_path / "missing-v5-metadata.h5", format_tag="echoChamber_H5_v5")

    with pytest.raises(ValueError, match="v5 electrode_mV must declare channel roles and regions"):
        load_recording(path)


def test_v4_without_channel_metadata_keeps_legacy_context_fallback(tmp_path: Path) -> None:
    path = make_calibrated_recording(tmp_path / "legacy-v4.h5", format_tag="echoChamber_H5_v4")

    loaded = load_recording(path)

    assert loaded.metadata["channel_roles"] == ["CA3", "Cortex"]
    assert loaded.metadata["channel_regions"] == ["CA3", "EC"]
    assert loaded.data[1:3].tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_v5_preserves_explicit_roles_and_ec_region(tmp_path: Path) -> None:
    path = make_calibrated_recording(
        tmp_path / "valid-v5.h5",
        format_tag="echoChamber_H5_v5",
        roles=("CA3", "Cortex"),
        regions=("CA3", "EC"),
    )

    loaded = load_recording(path)

    assert loaded.metadata["channel_roles"] == ["CA3", "Cortex"]
    assert loaded.metadata["channel_regions"] == ["CA3", "EC"]
