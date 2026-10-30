import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("import_abf", ROOT / "tools" / "import_abf.py")
import_abf = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = import_abf
SPEC.loader.exec_module(import_abf)
sys.path.insert(0, str(ROOT))
from replay import load_replay


def test_canonical_import_output_and_finite_chunks(tmp_path, monkeypatch):
    source = tmp_path / "source.abf"
    with source.open("wb") as stream:
        stream.truncate(import_abf.EXPECTED_BYTES)
    signal = np.arange(40, dtype=np.float32).reshape(2, 20)
    monkeypatch.setattr(import_abf, "read_abf", lambda p: (signal, 20_000.0, {"stored_units": ("mv", "mv")}))
    monkeypatch.setattr(import_abf, "sha256", lambda p: "a" * 64)
    fake = type("PyABF", (), {"__version__": "2.3.8"})
    monkeypatch.setitem(sys.modules, "pyabf", fake)
    target = tmp_path / "out.h5"
    import_abf.write_replay(source, target)
    loaded = load_replay(target)
    assert np.array_equal(loaded.electrode_mV, signal)
    assert loaded.channel_labels == ("CA3", "EC")
    assert loaded.channel_roles == ("CA3", "Cortex")
    chunks = list(loaded.iter_chunks(7))
    assert [start for start, _ in chunks] == [0, 7, 14]
    assert sum(chunk.shape[1] for _, chunk in chunks) == 20
    with pytest.raises(FileExistsError):
        import_abf.write_replay(source, target)


def test_abf_source_size_is_validated(tmp_path):
    source = tmp_path / "bad.abf"
    source.write_bytes(b"bad")
    with pytest.raises(ValueError, match="size mismatch"):
        import_abf.write_replay(source, tmp_path / "out.h5")


def test_unsupported_unit_or_channel_count_is_rejected(tmp_path):
    class Fake:
        abfVersion = 2
        channelCount = 2
        dataRate = 20_000
        adcNames = ["CA3", "EC"]
        adcUnits = ["V", "mV"]
        adcChannelNum = [0, 2]
        sweepCount = 1
        sweepLengthSec = 1800
        dataPointCount = 72_000_000
        def __init__(self, path): pass
        def setSweep(self, *args, **kwargs): pass
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setitem(sys.modules, "pyabf", type("pyabf", (), {"ABF": Fake, "__version__": "2.3.8"}))
        with pytest.raises(ValueError, match="stored units mV"):
            import_abf.read_abf(tmp_path / "fake.abf")
    finally:
        monkeypatch.undo()
