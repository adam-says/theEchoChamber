from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import tools.ni_staged_controller_test as ni_stage


class FakeOutStream:
    def __init__(self):
        self.regen_mode = None
        self.output_buf_size = 0
        self.queued_samples = 0
        self.started_at = None
        self.sample_rate = 20_000

    @property
    def space_avail(self):
        elapsed_generated = 0 if self.started_at is None else int((time.monotonic() - self.started_at) * self.sample_rate)
        return max(0, self.output_buf_size - max(0, elapsed_generated - self.queued_samples))

    @property
    def total_samp_per_chan_generated(self):
        if self.started_at is None:
            return 0
        return min(self.queued_samples, int((time.monotonic() - self.started_at) * self.sample_rate))


class FakeNiApi:
    def __init__(self, fail_writer_call: int | None = None):
        self.tasks = []
        self.zero_writes = []
        self.fail_writer_call = fail_writer_call
        self.writer_calls = 0

        owner = self

        class Task:
            def __init__(self, name):
                self.name = name
                self.out_stream = FakeOutStream()
                self.timing = SimpleNamespace(cfg_samp_clk_timing=self._timing)
                self.ao_channels = SimpleNamespace(add_ao_voltage_chan=lambda *a, **k: None)
                self.ai_channels = SimpleNamespace(add_ai_voltage_chan=lambda *a, **k: None)
                self.in_stream = SimpleNamespace()
                self.triggers = SimpleNamespace(start_trigger=SimpleNamespace(
                    term="/Dev1/ai/StartTrigger", cfg_dig_edge_start_trig=lambda *a, **k: None))
                self.started = False
                owner.tasks.append(self)

            def _timing(self, rate, **kwargs):
                self.out_stream.sample_rate = rate
                self.timing_args = (rate, kwargs)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def start(self):
                self.started = True
                self.out_stream.started_at = time.monotonic()

            def stop(self):
                self.started = False

            def write(self, value, **kwargs):
                owner.zero_writes.append((self.name, value, kwargs))

        class SingleWriter:
            def __init__(self, stream, **kwargs):
                self.stream = stream

            def write_many_sample(self, samples, **kwargs):
                owner.writer_calls += 1
                if owner.fail_writer_call == owner.writer_calls:
                    raise RuntimeError("injected AO writer failure")
                self.stream.queued_samples += np.asarray(samples).shape[-1]

        class MultiReader:
            def __init__(self, stream):
                self.stream = stream

            def read_many_sample(self, buffer, **kwargs):
                buffer.fill(0.0)

        nidaqmx = SimpleNamespace(
            Task=Task,
            system=SimpleNamespace(System=SimpleNamespace(local=lambda: SimpleNamespace(
                devices=[SimpleNamespace(name="Dev1")]))),
        )
        constants = SimpleNamespace(
            AcquisitionType=SimpleNamespace(CONTINUOUS="continuous"),
            RegenerationMode=SimpleNamespace(DONT_ALLOW_REGENERATION="no-regeneration"),
            TerminalConfiguration=SimpleNamespace(DIFF="diff", NRSE="nrse", RSE="rse"),
        )
        self.apis = (nidaqmx, constants.AcquisitionType, constants.RegenerationMode,
                     constants.TerminalConfiguration, MultiReader,
                     SingleWriter, SingleWriter)


def make_args(stage: str):
    return ni_stage.build_parser().parse_args([
        "--stage", stage, "--duration-seconds", "0.05", "--device", "Dev1",
    ])


class NiStageExecutionTests(unittest.TestCase):
    def test_all_three_stage_implementations_complete_with_fake_ni_api(self):
        for stage in ni_stage.STAGES:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                fake = FakeNiApi()
                args = make_args(stage)
                result = ni_stage._base_result(args)
                start = time.monotonic()
                with patch.object(ni_stage, "_load_ni_apis", return_value=fake.apis):
                    final = ni_stage._hardware_run(args, result, Path(directory) / "stage.jsonl")
                self.assertEqual(final["status"], "completed")
                self.assertTrue(final["hardware_accessed"])
                self.assertTrue(final["ao_zero_cleanup_succeeded"])
                self.assertGreaterEqual(final["ai_samples_read"], 1_000 if stage == "ai-zero" else 0)
                if stage == "ao-waveform":
                    self.assertGreaterEqual(fake.tasks[0].out_stream.total_samp_per_chan_generated, 1_000)
                    self.assertGreaterEqual(time.monotonic() - start, 0.045)
                self.assertTrue(fake.zero_writes)
                self.assertEqual(fake.zero_writes[-1][1], 0.0)

    def test_writer_failure_propagates_and_still_runs_ao_zero_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = FakeNiApi(fail_writer_call=2)
            args = make_args("ao-waveform")
            result = ni_stage._base_result(args)
            with patch.object(ni_stage, "_load_ni_apis", return_value=fake.apis):
                with self.assertRaisesRegex(RuntimeError, "injected AO writer failure"):
                    ni_stage._hardware_run(args, result, Path(directory) / "failed.jsonl")
            self.assertTrue(result["ao_zero_cleanup_succeeded"])
            self.assertEqual(fake.zero_writes[-1][1], 0.0)


if __name__ == "__main__":
    unittest.main()
