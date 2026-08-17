from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import echoChamber as app
from readEchoChamberData import load_recording


class EchoChamberRecorderTests(unittest.TestCase):
    @staticmethod
    def make_block(start: int, count: int, value: float, mode: float, fired: bool = False) -> app.RecordBlock:
        pair = np.full((2, count), value, dtype=np.float64)
        single = np.full((1, count), value, dtype=np.float64)
        return app.RecordBlock(
            sample_index=start,
            ai=pair,
            calibrated_lfp=pair * 100.0,
            raw_esn=single,
            model_esn=single + 1.0,
            pulse_threshold=single + 2.0,
            pulse_peak=single + 3.0,
            pulse_fired=np.full((1, count), float(fired)),
            safe_ao=single + 4.0,
            actual_stim=None,
            mode_value=mode,
        )

    def test_compact_v3_layout_scaling_and_sparse_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = app.AppConfig(
                record_dir=Path(directory), processing_block_ms=10.0,
                recorder_batch_s=0.02, recorder_flush_s=0.02,
                amplifier_gain=(10.0, 20.0),
            )
            recorder = app.H5Recorder(
                config, {"application": "test", "configuration": app.serializable_config(config)}
            )
            path = recorder.start("compact")
            recorder.submit(self.make_block(100, config.chunk_size, 1.0, 2.0))
            recorder.submit(self.make_block(100 + config.chunk_size + 5, config.chunk_size, 2.0, 3.0, True))
            recorder.stop_recording()
            recorder.close()

            with h5py.File(path, "r") as recording:
                self.assertEqual(recording.attrs["format_tag"], "echoChamber_H5_v3")
                self.assertEqual(int(recording.attrs["schema_version"]), 3)
                self.assertNotIn("data", recording)
                ai = recording["signals/ai_raw_V"]
                self.assertEqual(ai.dtype, np.dtype("float32"))
                self.assertEqual(ai.compression, "gzip")
                self.assertTrue(ai.shuffle)
                self.assertEqual(recording["events/mode_changes"].shape[0], 2)
                self.assertEqual(recording["events/pulses"].shape[0], 1)
                self.assertEqual(recording["events/sample_discontinuities"].shape[0], 1)
                self.assertEqual(recording["diagnostics/blocks"].shape[0], 2)
                metadata = json.loads(recording.attrs["metadata_json"])
                self.assertEqual(metadata["format_tag"], "echoChamber_H5_v3")

            loaded = load_recording(path)
            self.assertEqual(
                loaded.rows,
                ("ai_sample_index", "CA3_electrode_mV", "Cortex_electrode_mV",
                 "safe_ao_command_V", "mode"),
            )
            np.testing.assert_allclose(loaded.data[1, :config.chunk_size], 100.0)
            np.testing.assert_allclose(loaded.data[2, :config.chunk_size], 50.0)
            np.testing.assert_array_equal(
                loaded.data[4],
                np.r_[np.full(config.chunk_size, 2.0), np.full(config.chunk_size, 3.0)],
            )
            self.assertEqual(loaded.data[0, config.chunk_size], 100 + config.chunk_size + 5)


if __name__ == "__main__":
    unittest.main()
