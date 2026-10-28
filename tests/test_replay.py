import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from replay import load_replay


def test_legacy_v4_volts_are_calibrated_once(tmp_path):
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = "echo-chamber-recording"
        f.attrs["format_tag"] = "echoChamber_H5_v4"
        f.attrs["sample_rate_hz"] = 20_000
        f.attrs["committed_samples"] = 3
        f.attrs["metadata_json"] = json.dumps({"configuration": {"amplifier_gain": [10, 20]}})
        ds = f.create_dataset("signals/ai_raw_V", data=np.array([[0.001, 0.002, 0.003], [0.002, 0.004, 0.006]], dtype=np.float32))
        ds.attrs["channel_labels"] = np.array(["CA3", "EC"], dtype=h5py.string_dtype())
    replay = load_replay(path)
    np.testing.assert_allclose(replay.electrode_mV, [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    assert replay.raw_V is not None
    assert replay.ctx_index == 1


def test_canonical_units_must_be_millivolts(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as f:
        f.attrs.update(schema="echo-chamber-replay", schema_version=1, sample_rate_hz=20_000, committed_samples=2)
        ds = f.create_dataset("signals/electrode_mV", data=np.ones((2, 2)))
        ds.attrs.update(unit="V", channel_labels=["CA3", "EC"], channel_roles=["CA3", "Cortex"])
    with pytest.raises(ValueError, match="unit must be exactly mV"):
        load_replay(path)


def test_unknown_schema_is_rejected(tmp_path):
    path = tmp_path / "unknown.h5"
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = "other"
    with pytest.raises(ValueError, match="unsupported replay schema"):
        load_replay(path)


def test_legacy_discontinuities_preserve_absolute_indices(tmp_path):
    path = tmp_path / "gapped.h5"
    events_dtype = np.dtype([("sample_offset", "<u8"), ("expected_ai_sample_index", "<u8"),
                             ("actual_ai_sample_index", "<u8")])
    with h5py.File(path, "w") as f:
        f.attrs.update(schema="echo-chamber-recording", format_tag="echoChamber_H5_v4",
                       sample_rate_hz=1000, first_ai_sample_index=50, committed_samples=6)
        f.attrs["metadata_json"] = json.dumps({"configuration": {"amplifier_gain": [10, 10]}})
        ds = f.create_dataset("signals/ai_raw_V", data=np.ones((2, 6), dtype=np.float32))
        ds.attrs["channel_labels"] = np.array(["CA3", "Cortex"], dtype=h5py.string_dtype())
        f.create_dataset("events/sample_discontinuities", data=np.array([(3, 53, 60)], dtype=events_dtype))
    replay = load_replay(path)
    chunks = list(replay.iter_chunks(4))
    assert [(start, data.shape[1]) for start, data in chunks] == [(50, 3), (60, 3)]
    assert replay.metadata["gaps"] == [{"sample_offset": 3, "expected_index": 53, "actual_index": 60}]


def test_v5_reads_calibrated_mv_without_applying_gain_again(tmp_path):
    path = tmp_path / "v5.h5"
    with h5py.File(path, "w") as f:
        f.attrs.update(schema="echo-chamber-recording", format_tag="echoChamber_H5_v5",
                       schema_version=5, sample_rate_hz=1000, first_ai_sample_index=4,
                       committed_samples=2, metadata_json=json.dumps({"configuration": {"amplifier_gain": [100, 100]}}))
        ds = f.create_dataset("signals/electrode_mV", data=np.array([[1, 2], [3, 4]], dtype=np.float32))
        ds.attrs["unit"] = "mV"
        ds.attrs["channel_labels"] = np.array(["CA3", "EC"], dtype=h5py.string_dtype())
        ds.attrs["channel_roles"] = np.array(["CA3", "Cortex"], dtype=h5py.string_dtype())
        ds.attrs["channel_regions"] = np.array(["CA3", "EC"], dtype=h5py.string_dtype())
    replay = load_replay(path)
    np.testing.assert_array_equal(replay.electrode_mV, [[1, 2], [3, 4]])
    assert replay.raw_V is None


def test_sample_sidecar_overrides_conflicting_embedded_channel_labels():
    sample = Path(__file__).resolve().parents[1] / "sample" / "test_AI0_CA3_AI1_CTX.h5"
    replay = load_replay(sample)
    assert replay.channel_labels == ("CA3", "Cortex")
    assert replay.channel_roles == ("CA3", "Cortex")
    assert replay.ctx_index == 1
    assert "AI0=Cortex" in replay.metadata["replay_sidecar"]["provenance"]["embedded_metadata_conflict"]
