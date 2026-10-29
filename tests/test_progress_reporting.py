from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
import unittest
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import numpy as np

import benchmarkControllers as benchmark
from benchmarkControllers import (
    RunMetrics,
    _Harness,
    _HarnessProgressWriter,
    HarnessOwnerShutdownError,
    acceptance_failures,
    run_replay_case,
)
import tools.validate_controllers as validate_cli


class HarnessProgressWriterTests(unittest.TestCase):
    def test_blocked_writer_does_not_block_producer_and_keeps_fifo_order(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        writes: list[tuple[int, int, dict]] = []

        def blocked_write(_path, payload):
            if not entered.is_set():
                entered.set()
                if not release.wait(5.0):
                    raise TimeoutError("test writer was not released")
            writes.append((int(payload["progress_sequence"]),
                           threading.get_ident(), payload))

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(benchmark, "write_progress", blocked_write):
                writer = _HarnessProgressWriter(Path(directory) / "progress.jsonl", capacity=2)
                first = {"type": "source_event", "source_sample_index": 100, "value": [1, 2]}
                self.assertTrue(writer.submit(first))
                self.assertTrue(entered.wait(2.0))
                second = {"type": "ao_clock_fault", "source_sample_index": 200}
                third = {"type": "run_complete", "source_sample_index": 300}
                self.assertTrue(writer.submit(second))
                self.assertTrue(writer.submit(third))
                second["source_sample_index"] = 999
                self.assertFalse(writer.submit({"type": "overflow"}))
                release.set()
                status = writer.close()
                self.assertEqual(status, writer.close())

        self.assertEqual([row[0] for row in writes], [1, 2, 3])
        self.assertTrue(all(row[1] != threading.get_ident() for row in writes))
        self.assertEqual(writes[1][2]["source_sample_index"], 200)
        self.assertEqual(status["written"], 3)
        self.assertEqual(status["dropped"], 0)
        self.assertEqual(status["rejected"], 1)
        self.assertIn("overflow", status["error"] or "")
        self.assertFalse(status["owner_alive"])
        self.assertEqual(status["accepted"], 3)
        self.assertEqual(status["pending"], 0)

    def test_writer_computes_bounded_recent_core_window_on_owner_thread(self) -> None:
        writes: list[tuple[int, dict]] = []
        producer_thread = threading.get_ident()
        window = ((0.4, 10_000, 10_100), (0.2, 10_100, 10_200),
                  (0.8, 10_200, 10_300))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                benchmark, "write_progress",
                side_effect=lambda _path, payload: writes.append((threading.get_ident(), payload)),
            ):
                writer = _HarnessProgressWriter(Path(directory) / "progress.jsonl")
                self.assertTrue(writer.submit(
                    {"type": "run_progress", "processed_blocks": 3},
                    recent_core_window=window,
                ))
                status = writer.close()
        thread_id, payload = writes[0]
        self.assertNotEqual(thread_id, producer_thread)
        self.assertEqual(payload["recent_core_window_count"], 3)
        self.assertEqual(payload["recent_core_window_start_sample_index"], 10_000)
        self.assertEqual(payload["recent_core_window_end_sample_index"], 10_300)
        self.assertEqual(payload["recent_core_window_range_convention"], "half-open [start, end)")
        self.assertEqual(payload["recent_core_p99_ms"], float(np.percentile([0.4, 0.2, 0.8], 99)))
        self.assertNotIn("latency_p99_ms_so_far", payload)
        self.assertEqual(status["written"], 1)

    def test_writer_failure_latches_without_raising_and_fails_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(benchmark, "write_progress", side_effect=OSError("disk unavailable")):
                writer = _HarnessProgressWriter(Path(directory) / "progress.jsonl", capacity=2)
                self.assertTrue(writer.submit({"type": "ao_clock_fault"}))
                self.assertTrue(writer.submit({"type": "run_complete"}))
                status = writer.close()
        self.assertIn("disk unavailable", status["error"] or "")
        self.assertEqual(status["written"], 0)
        self.assertGreaterEqual(status["dropped"], 1)
        self.assertEqual(status["pending"], 0)
        metrics = RunMetrics(
            case_id="diag-error", source_name="fixture", controller="rule",
            sample_rate_hz=20_000, block_samples=100, block_ms=5.0,
            ao_lead_ms=100.0, deadline_ms=5.0,
            progress_diagnostic_error=status["error"],
        )
        self.assertTrue(any("progress diagnostics failed" in item
                            for item in acceptance_failures(metrics)))

    def test_close_timeout_is_reported_but_waits_for_accepted_events_and_join(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        writes: list[int] = []

        def blocked_write(_path, payload):
            if not entered.is_set():
                entered.set()
                if not release.wait(5.0):
                    raise TimeoutError("test writer was not released")
            writes.append(int(payload["progress_sequence"]))

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(benchmark, "write_progress", blocked_write):
                writer = _HarnessProgressWriter(
                    Path(directory) / "progress.jsonl", capacity=1, close_timeout_s=0.02)
                self.assertTrue(writer.submit({"type": "first"}))
                self.assertTrue(entered.wait(2.0))
                self.assertTrue(writer.submit({"type": "second"}))
                self.assertFalse(writer.submit({"type": "rejected-before-close"}))
                first_status = writer.close()
                self.assertTrue(writer._close_timed_out)
                self.assertTrue(first_status["owner_alive"])
                self.assertEqual(first_status["pending"], 2)
                self.assertEqual(first_status["accepted"], 2)
                self.assertEqual(first_status["rejected"], 1)
                release.set()
                final_status = writer.close()
        self.assertEqual(writes, [1, 2])
        self.assertTrue(first_status["close_timed_out"])
        self.assertTrue(final_status["close_timed_out"])
        self.assertFalse(final_status["owner_alive"])
        self.assertEqual(final_status["accepted"], 2)
        self.assertEqual(final_status["written"], 2)
        self.assertEqual(final_status["pending"], 0)
        self.assertEqual(final_status["rejected"], 1)
        # The earlier overflow remains the first diagnostic; the timeout is
        # independently represented by close_timed_out.
        self.assertIn("overflow", final_status["error"] or "")

    def test_submit_rejects_promptly_while_close_waits_to_queue_stop_marker(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        close_finished = threading.Event()
        close_status: list[dict] = []

        def blocked_write(_path, _payload):
            entered.set()
            if not release.wait(5.0):
                raise TimeoutError("test writer was not released")

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(benchmark, "write_progress", blocked_write):
                writer = _HarnessProgressWriter(
                    Path(directory) / "progress.jsonl", capacity=1, close_timeout_s=0.5)
                self.assertTrue(writer.submit({"type": "first"}))
                self.assertTrue(entered.wait(2.0))
                self.assertTrue(writer.submit({"type": "queued"}))

                def close_writer():
                    close_status.append(writer.close())
                    close_finished.set()

                closer = threading.Thread(target=close_writer)
                closer.start()
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    with writer._state_lock:
                        if writer._closed:
                            break
                    time.sleep(0.001)
                with writer._state_lock:
                    self.assertTrue(writer._closed)
                started = time.perf_counter()
                self.assertFalse(writer.submit({"type": "after-close"}))
                elapsed = time.perf_counter() - started
                self.assertLess(elapsed, 0.1)
                release.set()
                closer.join(timeout=3.0)
                self.assertTrue(close_finished.is_set())
                self.assertFalse(closer.is_alive())

        self.assertEqual(close_status[0]["accepted"], 2)
        self.assertEqual(close_status[0]["rejected"], 1)
        self.assertEqual(close_status[0]["pending"], 0)
        self.assertFalse(close_status[0]["owner_alive"])

    def test_actual_harness_keeps_recording_but_fails_case_on_progress_write_error(self) -> None:
        trace = Path(benchmark.app.__file__).resolve().parent / "sample" / "test_AI0_CA3_AI1_CTX.h5"
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(benchmark, "write_progress", side_effect=OSError("progress disk failure")):
                metrics, _environment = run_replay_case(
                    trace, "rule", limit_seconds=0.25, paced=True,
                    progress_path=Path(directory) / "progress.jsonl", progress_interval_s=0.0,
                    recording_dir=Path(directory) / "recordings",
                )
        self.assertTrue(metrics.recording_completed)
        self.assertEqual(metrics.recording_committed_samples, 5_000)
        self.assertIsNone(metrics.runtime_fault)
        self.assertIn("progress disk failure", metrics.progress_diagnostic_error or "")
        self.assertTrue(any("progress diagnostics failed" in item
                            for item in acceptance_failures(metrics)))

    def test_browser_startup_writer_failure_aborts_and_joins_worker(self) -> None:
        config = benchmark.app.AppConfig(controller="rule", processing_block_ms=5.0,
                                         ao_target_lead_ms=100.0)
        config.validate()
        started_writers: list[_HarnessProgressWriter] = []
        original_init = _HarnessProgressWriter.__init__

        def capture_writer(writer, *args, **kwargs):
            original_init(writer, *args, **kwargs)
            started_writers.append(writer)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(benchmark, "write_progress", side_effect=OSError("cannot write URL event")):
                with patch.object(_HarnessProgressWriter, "__init__", capture_writer):
                    with self.assertRaisesRegex(RuntimeError, "progress diagnostics failed during browser startup"):
                        _Harness(
                            config, "progress-startup-failure", {"configuration": {"test": True}},
                            progress_path=Path(directory) / "progress.jsonl",
                            progress_interval_s=60.0, paced=True, simulation=False,
                            on_generated=None, browser_ui=True, external_browser=True,
                        )
        self.assertEqual(len(started_writers), 1)
        self.assertFalse(started_writers[0]._thread.is_alive())
        self.assertIn("cannot write URL event", started_writers[0].error or "")

    def test_unresolved_ao_owner_leaves_dependencies_open_until_retry(self) -> None:
        trace = Path(benchmark.app.__file__).resolve().parent / "sample" / "test_AI0_CA3_AI1_CTX.h5"
        entered = threading.Event()
        release = threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            config = benchmark.app.AppConfig(
                controller="rule", processing_block_ms=5.0, ao_target_lead_ms=100.0,
                record_dir=Path(directory),
            )
            config.validate()
            harness = _Harness(
                config, "ao-owner-timeout", {"configuration": {"test": True}},
                progress_path=Path(directory) / "progress.jsonl", progress_interval_s=60.0,
                paced=False, simulation=False, on_generated=None,
            )
            harness._ao_stop_timeout_s = 0.02

            def blocked_advance(_elapsed):
                entered.set()
                if not release.wait(5.0):
                    raise TimeoutError("test AO owner was not released")
                return ()

            harness._advance_virtual = blocked_advance
            current_generation = harness.state.snapshot().generation
            harness.core.enqueue_ao_outcome(benchmark.app.PulseAoOutcome(
                pulse_id="abort-tail", generation=current_generation,
                start_sample_index=2_000, end_sample_index=2_100,
                status="cancelled", reason="queued for shutdown drain",
                source_onset_sample_index=0, target_onset_sample_index=2_000,
                latest_sample_index=99, controller_id="rule",
            ))
            harness._write_progress({"type": "before_abort"})
            harness._start_ao_clock()
            self.assertTrue(entered.wait(2.0))
            order: list[str] = []
            original_drain = harness.core.drain_ao_notifications
            original_record_stage = harness.recorder.record_action_stage
            original_stop_recording = harness.recorder.stop_recording
            original_recorder_close = harness.recorder.close

            def drain_notifications():
                order.append("drain")
                return original_drain()

            def record_stage(*args, **kwargs):
                order.append("stage")
                return original_record_stage(*args, **kwargs)

            def stop_recording(*args, **kwargs):
                order.append("stop_recording")
                return original_stop_recording(*args, **kwargs)

            def close_recorder(*args, **kwargs):
                order.append("close_recorder")
                return original_recorder_close(*args, **kwargs)

            core_close = patch.object(harness.core, "close_ui", wraps=harness.core.close_ui)
            with (
                core_close as core_close_mock,
                patch.object(harness.core, "drain_ao_notifications", side_effect=drain_notifications),
                patch.object(harness.recorder, "record_action_stage", side_effect=record_stage),
                patch.object(harness.recorder, "stop_recording", side_effect=stop_recording),
                patch.object(harness.recorder, "close", side_effect=close_recorder),
            ):
                with self.assertRaises(HarnessOwnerShutdownError) as first:
                    harness.abort()
                self.assertEqual(first.exception.owner_kind, "simulated_ao")
                self.assertTrue(first.exception.owner_unresolved)
                self.assertTrue(harness.recorder._thread.is_alive())
                self.assertTrue(harness._progress_writer._thread.is_alive())
                core_close_mock.assert_not_called()
                self.assertEqual(order, [])

                release.set()
                ao_thread = harness._ao_clock_thread
                ao_thread.join(timeout=2.0)
                self.assertFalse(ao_thread.is_alive())
                with self.assertRaises(HarnessOwnerShutdownError) as retry:
                    harness.abort()
                self.assertIs(retry.exception, first.exception)
                self.assertEqual(retry.exception.owner_kind, "simulated_ao")
                self.assertFalse(retry.exception.owner_unresolved)
                self.assertEqual(order[0:2], ["drain", "stage"])
                self.assertLess(order.index("stage"), order.index("stop_recording"))
                self.assertLess(order.index("stage"), order.index("close_recorder"))
                core_close_mock.assert_called_once()
            self.assertFalse(harness._progress_writer._thread.is_alive())
            progress_rows = [
                json.loads(line)
                for line in (Path(directory) / "progress.jsonl").read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual([row["type"] for row in progress_rows], ["before_abort"])
        self.assertIn("did not stop", retry.exception.original_error)

    def test_unresolved_ao_without_explicit_recording_dir_retains_temp_path(self) -> None:
        trace = Path(benchmark.app.__file__).resolve().parent / "sample" / "test_AI0_CA3_AI1_CTX.h5"
        entered = threading.Event()
        release = threading.Event()
        captured: list[_Harness] = []
        results: list[BaseException | tuple] = []
        original_init = _Harness.__init__
        original_process = _Harness.process
        original_advance = _Harness._advance_virtual

        def capture_init(harness, *args, **kwargs):
            original_init(harness, *args, **kwargs)
            harness._ao_stop_timeout_s = 0.02
            captured.append(harness)

        def start_ao_before_process(harness, *args, **kwargs):
            harness._start_ao_clock()
            return original_process(harness, *args, **kwargs)

        def block_only_ao_owner(harness, elapsed):
            if threading.current_thread() is harness._ao_clock_thread:
                entered.set()
                if not release.wait(5.0):
                    raise TimeoutError("test AO owner was not released")
                return ()
            return original_advance(harness, elapsed)

        def run_case():
            try:
                results.append(run_replay_case(
                    trace, "rule", limit_seconds=0.005, paced=False,
                    progress_path=None, progress_interval_s=60.0, recording_dir=None,
                ))
            except BaseException as exc:
                results.append(exc)

        with (
            patch.object(_Harness, "__init__", capture_init),
            patch.object(_Harness, "process", start_ao_before_process),
            patch.object(_Harness, "_advance_virtual", block_only_ao_owner),
        ):
            runner = threading.Thread(target=run_case)
            runner.start()
            self.assertTrue(entered.wait(3.0))
            runner.join(timeout=5.0)
            self.assertFalse(runner.is_alive())
        self.assertEqual(len(captured), 1)
        failure = results[0]
        self.assertIsInstance(failure, HarnessOwnerShutdownError)
        assert isinstance(failure, HarnessOwnerShutdownError)
        self.assertEqual(failure.owner_kind, "simulated_ao")
        self.assertTrue(failure.owner_unresolved)
        retained = Path(failure.failure_fields()["retained_temporary_path"])
        self.assertTrue(retained.is_dir())
        harness = captured[0]
        release.set()
        harness._ao_clock_thread.join(timeout=2.0)
        with self.assertRaises(HarnessOwnerShutdownError) as retry:
            harness.abort()
        self.assertFalse(retry.exception.owner_unresolved)
        self.assertEqual(retry.exception.retained_temporary_path, str(retained))
        shutil.rmtree(retained)

    def test_runtime_initializer_failure_cleans_started_progress_writer_and_ui_loop(self) -> None:
        config = benchmark.app.AppConfig(controller="rule", processing_block_ms=5.0,
                                         ao_target_lead_ms=100.0)
        config.validate()
        captured: list[_Harness] = []

        def fail_runtime_initialization(harness, *_args, **_kwargs):
            captured.append(harness)
            raise RuntimeError("scheduler construction failed")

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(_Harness, "_initialize_harness_runtime", fail_runtime_initialization):
                with patch.object(benchmark._CountingUiHub, "client_count", new_callable=PropertyMock,
                                  return_value=2):
                    with self.assertRaisesRegex(RuntimeError, "scheduler construction failed"):
                        _Harness(
                            config, "progress-runtime-init-failure", {"configuration": {"test": True}},
                            progress_path=Path(directory) / "progress.jsonl",
                            progress_interval_s=60.0, paced=True, simulation=False,
                            on_generated=None, browser_ui=True, external_browser=True,
                        )
        self.assertEqual(len(captured), 1)
        harness = captured[0]
        self.assertIsNotNone(harness._progress_writer)
        self.assertFalse(harness._progress_writer._thread.is_alive())
        self.assertFalse(harness.loop_thread.is_alive())
        self.assertTrue(harness.loop.is_closed())

    def test_unresolved_owner_stops_collection_and_avoids_shared_jsonl_writes(self) -> None:
        status = {
            "error": "timed out joining progress writer",
            "accepted": 3, "written": 1, "dropped": 0, "pending": 2,
            "queue_high_water": 2, "closed": True, "owner_alive": True,
            "close_timed_out": True,
        }
        failure = HarnessOwnerShutdownError(
            "progress_writer", status, progress_path=Path("progress.jsonl"))
        with tempfile.TemporaryDirectory() as directory:
            output = StringIO()
            error = StringIO()
            with patch.object(validate_cli, "run_replay_case", side_effect=failure) as run_case:
                with redirect_stdout(output), redirect_stderr(error):
                    code = validate_cli.main([
                        "--matrix", "--trace", "synthetic-fixture.h5",
                        "--controller", "rule", "--controller", "pid",
                        "--duration-seconds", "0.25", "--output-dir", directory,
                    ])
            self.assertEqual(run_case.call_count, 1)
            progress_path = Path(directory) / "progress.jsonl"
            progress_events = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["type"] for event in progress_events], ["validation_started"])
            report = json.loads(next(Path(directory).glob("controller_matrix_*.json")).read_text(encoding="utf-8"))
            self.assertEqual(report["case_failures"][0]["owner_unresolved"], True)
            self.assertEqual(report["case_failures"][0]["progress_events_pending"], 2)
            self.assertIn("stopping collection", error.getvalue())
            self.assertIn("Progress JSONL is incomplete", output.getvalue())
            self.assertEqual(code, 1)

    def test_unresolved_simulated_ao_owner_stops_collection_and_skips_shared_jsonl(self) -> None:
        failure = HarnessOwnerShutdownError(
            "simulated_ao",
            {"error": "AO clock stop timed out", "owner_alive": True,
             "owner_thread_name": "blocked-ao", "stop_requested": True,
             "stop_timed_out": True},
            progress_path=Path("progress.jsonl"),
        )
        with tempfile.TemporaryDirectory() as directory:
            stderr = StringIO()
            with patch.object(validate_cli, "run_replay_case", side_effect=failure) as run_case:
                with redirect_stdout(StringIO()), redirect_stderr(stderr):
                    code = validate_cli.main([
                        "--matrix", "--trace", "synthetic-fixture.h5",
                        "--controller", "rule", "--controller", "pid",
                        "--duration-seconds", "0.25", "--output-dir", directory,
                    ])
            self.assertEqual(run_case.call_count, 1)
            progress_path = Path(directory) / "progress.jsonl"
            rows = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["type"] for row in rows], ["validation_started"])
            report = json.loads(next(Path(directory).glob("controller_matrix_*.json")).read_text())
            failed = report["case_failures"][0]
        self.assertEqual(failed["shutdown_owner_kind"], "simulated_ao")
        self.assertTrue(failed["owner_unresolved"])
        self.assertTrue(failed["shutdown_owner_status"]["stop_timed_out"])
        self.assertIn("stopping collection", stderr.getvalue())
        self.assertEqual(code, 1)

    def test_run_and_abort_reraising_same_owner_error_does_not_self_chain(self) -> None:
        trace = Path(benchmark.app.__file__).resolve().parent / "sample" / "test_AI0_CA3_AI1_CTX.h5"
        failure = HarnessOwnerShutdownError(
            "simulated_ao",
            {"owner_alive": True, "error": "AO stop timed out", "stop_timed_out": True},
        )

        class FailedHarness:
            recording_path = Path("failed-recording.h5")

            def process(self, *_args, **_kwargs):
                raise failure

            def abort(self):
                raise failure

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(benchmark, "_Harness", return_value=FailedHarness()):
                with self.assertRaises(HarnessOwnerShutdownError) as caught:
                    run_replay_case(
                        trace, "rule", limit_seconds=0.005,
                        recording_dir=Path(directory) / "recordings",
                    )
        self.assertIs(caught.exception, failure)
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNot(caught.exception.__context__, caught.exception)
        self.assertFalse(any("Run also failed" in note
                             for note in getattr(caught.exception, "__notes__", ())))

    def test_resolved_timeout_retains_failure_and_allows_normal_checkpoints(self) -> None:
        status = {
            "error": "timed out joining progress writer; owner later stopped",
            "accepted": 3, "written": 3, "dropped": 0, "pending": 0,
            "queue_high_water": 2, "closed": True, "owner_alive": False,
            "close_timed_out": True,
        }
        failure = HarnessOwnerShutdownError(
            "progress_writer", status, progress_path=Path("progress.jsonl"))
        completed = RunMetrics(
            case_id="second-case", source_name="fixture", controller="pid",
            sample_rate_hz=20_000, block_samples=100, block_ms=5.0,
            ao_lead_ms=100.0, deadline_ms=5.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(validate_cli, "run_replay_case", side_effect=[failure, (completed, {})]) as run_case:
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    code = validate_cli.main([
                        "--matrix", "--trace", "synthetic-fixture.h5",
                        "--controller", "rule", "--controller", "pid",
                        "--duration-seconds", "0.25", "--output-dir", directory,
                    ])
            self.assertEqual(run_case.call_count, 2)
            report = json.loads(next(Path(directory).glob("controller_matrix_*.json")).read_text(encoding="utf-8"))
            self.assertIn("timed out", report["case_failures"][0]["message"])
            progress_types = [
                json.loads(line)["type"]
                for line in (Path(directory) / "progress.jsonl").read_text(encoding="utf-8").splitlines()
            ]
        self.assertIn("case_failed", progress_types)
        self.assertIn("case_checkpoint", progress_types)
        self.assertIn("validation_complete", progress_types)
        self.assertEqual(code, 1)

    def test_recent_core_window_is_bounded_without_truncating_final_latency_history(self) -> None:
        harness = _Harness.__new__(_Harness)
        harness.config = SimpleNamespace(chunk_size=100)
        harness.latencies_ms = []
        harness._recent_core_latencies = deque(maxlen=2048)
        for block in range(2_050):
            harness._record_core_latency(block * 100, float(block))
        self.assertEqual(len(harness.latencies_ms), 2_050)
        self.assertEqual(len(harness._recent_core_latencies), 2_048)
        self.assertEqual(harness._recent_core_latencies[0], (2.0, 200, 300))
        self.assertEqual(harness._recent_core_latencies[-1], (2_049.0, 204_900, 205_000))

    def test_short_harness_writes_recent_progress_but_preserves_final_quantiles(self) -> None:
        trace = Path(benchmark.app.__file__).resolve().parent / "sample" / "test_AI0_CA3_AI1_CTX.h5"
        original_finish = _Harness.finish
        observed: list[float] = []
        observed_window: list[tuple[float, int, int]] = []

        def capture_final_latency_input(harness):
            observed.extend(harness.latencies_ms)
            observed_window.extend(harness._recent_core_latencies)
            return original_finish(harness)

        with tempfile.TemporaryDirectory() as directory:
            progress_path = Path(directory) / "progress.jsonl"
            with patch.object(_Harness, "finish", capture_final_latency_input):
                metrics, _environment = run_replay_case(
                    trace, "rule", limit_seconds=0.25, paced=True,
                    progress_path=progress_path, progress_interval_s=0.0,
                    recording_dir=Path(directory) / "recordings",
                )
            events = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
        progress = [event for event in events if event.get("type") == "run_progress"]
        self.assertTrue(progress)
        latest = progress[-1]
        self.assertEqual(latest["recent_core_window_count"], metrics.processed_blocks)
        self.assertLessEqual(latest["recent_core_window_count"], 2048)
        self.assertEqual(latest["recent_core_window_start_sample_index"], observed_window[0][1])
        self.assertEqual(latest["recent_core_window_end_sample_index"], observed_window[-1][2])
        self.assertEqual(latest["recent_core_window_range_convention"], "half-open [start, end)")
        self.assertEqual(latest["recent_core_p99_ms"], float(np.percentile(observed, 99)))
        self.assertNotIn("latency_p99_ms_so_far", latest)
        self.assertEqual(metrics.process_p99_ms, float(np.percentile(observed, 99)))
        self.assertEqual(metrics.progress_events_written, len(events))
        self.assertEqual(metrics.progress_events_dropped, 0)
        self.assertIsNone(metrics.progress_diagnostic_error)


if __name__ == "__main__":
    unittest.main()
