from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import h5py

import benchmarkEchoChamberProcessing as benchmark


class ProcessingBenchmarkTests(unittest.TestCase):
    def test_processing_converts_raw_volts_to_electrode_millivolts(self) -> None:
        class Bridge:
            received = None

            def process(self, values, *, ctx_index):
                self.received = values.copy()
                return np.zeros((1, values.shape[1]))

        class Safety:
            @staticmethod
            def process(command):
                return command, None

        bridge = Bridge()
        source = benchmark.BlockSource(np.ones((2, 100)), 100)
        benchmark.process_once(
            bridge, Safety(), source, ctx_index=1,
            amplifier_gain=(10.0, 20.0), injected_delay_ms=0.0,
        )
        np.testing.assert_allclose(bridge.received[0], 100.0)
        np.testing.assert_allclose(bridge.received[1], 50.0)

    def test_synthetic_source_returns_repeatable_two_channel_blocks(self) -> None:
        first = benchmark.synthetic_lfp(20_000, 0.1, seed=7)
        second = benchmark.synthetic_lfp(20_000, 0.1, seed=7)
        np.testing.assert_array_equal(first, second)
        source = benchmark.BlockSource(first, 100)
        self.assertEqual(source.next().shape, (2, 100))

    def test_hdf5_loader_selects_named_electrode_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.h5"
            with h5py.File(path, "w") as recording:
                recording.attrs["schema"] = "echo-chamber-recording"
                recording.attrs["committed_samples"] = 5
                recording.create_dataset(
                    "row_names", data=np.asarray(
                        ["Cortex_raw_V", "CA3_raw_V"], dtype=h5py.string_dtype("utf-8")
                    )
                )
                recording.create_dataset("data", data=np.arange(10).reshape(2, 5))
            values = benchmark.load_hdf5(path, 5, ("CA3", "Cortex"))
            np.testing.assert_array_equal(values[0], np.arange(5, 10))
            np.testing.assert_array_equal(values[1], np.arange(5))

    def test_timing_summary_counts_deadline_misses(self) -> None:
        result = benchmark.summarize([1.0, 2.0, 6.0], deadline_ms=5.0)
        self.assertEqual(result.count, 3)
        self.assertEqual(result.over_deadline, 1)
        self.assertEqual(result.maximum_ms, 6.0)


if __name__ == "__main__":
    unittest.main()
