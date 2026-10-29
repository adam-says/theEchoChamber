from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from controllers import PulseEvent

import benchmarkControllers as benchmark


def _row(source_index: int, *, lag_ms: float | None = None) -> benchmark._CoreTimingContext:
    return benchmark._CoreTimingContext(
        source_sample_index=source_index,
        start_perf_counter_ns=source_index * 1_000,
        end_perf_counter_ns=source_index * 1_000 + 500,
        wall_ms=0.5,
        thread_cpu_ms=0.0,
        process_cpu_ms=15.625,
        gc_overlap=False,
        source_availability_lag_ms=lag_ms,
        source_availability_lag_sample_index=source_index if lag_ms is not None else None,
        mode="closed-loop",
        acquiring=True,
        armed=True,
        transition_pending=False,
        generation=7,
        core_processed_generation=7,
        state_observation="after_core_call",
        controller="pid",
        controller_ready=True,
        controller_calibrated=True,
        fault=None,
        controller_reason="active",
        pulse_events=(
            PulseEvent("g7-p1", source_index - 20, source_index - 20,
                       source_index + 1_980, "rendered_for_ao", 7,
                       rendered_sample_count=20, latest_sample_index=source_index - 1,
                       controller_id="pid"),
        ),
    )


class _StopAfterOneTick:
    def __init__(self) -> None:
        self.stopped = False

    def is_set(self) -> bool:
        return self.stopped

    def wait(self, _seconds: float) -> bool:
        self.stopped = True
        return True


class FirstFaultContextTests(unittest.TestCase):
    def test_gc_callback_can_reenter_the_short_gc_snapshot_lock(self) -> None:
        harness = benchmark._Harness.__new__(benchmark._Harness)
        harness._gc_lock = threading.RLock()
        harness._gc_starts = {}
        harness._gc_pauses_ns = []
        harness._gc_events = []

        # Collection callbacks can fire on the same thread while diagnostic
        # tuple snapshots allocate. The lock must permit that reentrant path.
        with harness._gc_lock:
            harness._collect_gc_timing("start", {"generation": 0})
        harness._collect_gc_timing("stop", {"generation": 0})
        self.assertEqual(len(harness._gc_pauses_ns), 1)
        self.assertEqual(len(harness._gc_events), 1)

    def test_recent_core_history_is_fixed_capacity_and_preserves_absolute_indices(self) -> None:
        harness = benchmark._Harness.__new__(benchmark._Harness)
        harness._recent_core_rows = deque(maxlen=benchmark.AO_FAULT_CORE_HISTORY_CAPACITY)
        harness._recent_core_lock = threading.Lock()
        harness._active_core_source_index = None
        harness._last_core_call_started_source_index = 500_000
        harness._latest_source_sample_index_seen = 500_000
        harness._active_core_source_lag_ms = None
        harness._active_core_source_lag_index = None
        harness.last_source_index = 500_000

        for index in range(40):
            harness._recent_core_rows.append(_row(496_000 + index * 100))

        context = harness._snapshot_recent_core_context()
        rows = context["recent_core_history"]
        self.assertEqual(len(rows), 32)
        self.assertEqual(rows[0]["source_sample_index"], 496_800)
        self.assertEqual(rows[-1]["source_sample_index"], 499_900)
        self.assertEqual(context["latest_completed_core_source_sample_index"], 499_900)
        self.assertEqual(context["latest_source_sample_index_seen"], 500_000)
        self.assertEqual(context["latest_core_call_started_source_sample_index"], 500_000)
        self.assertIsNone(context["core_call_started_not_yet_published_source_sample_index"])
        with self.assertRaises((AttributeError, TypeError)):
            harness._recent_core_rows[-1].source_sample_index = 0

    def test_ao_fault_event_captures_service_gap_source_lag_state_and_recent_core_rows(self) -> None:
        harness = benchmark._Harness.__new__(benchmark._Harness)
        harness.config = SimpleNamespace(chunk_size=100, sample_rate=20_000)
        harness._ao_clock_stop = _StopAfterOneTick()
        harness._ao_clock_error = None
        harness._observed_ao_fault = None
        harness._ao_clock_max_gap_s = 0.0
        harness._ao_clock_gap_count = 0
        harness.pacing_epoch = 99.0
        harness.last_source_index = 480_200
        harness._active_core_source_index = 480_200
        harness._last_core_call_started_source_index = 480_200
        harness._latest_source_sample_index_seen = 480_200
        harness._active_core_source_lag_ms = 66.7837
        harness._active_core_source_lag_index = 480_200
        harness._recent_core_rows = deque((_row(480_000, lag_ms=42.0),
                                           _row(480_100, lag_ms=50.0)), maxlen=32)
        harness._recent_core_lock = threading.Lock()
        harness.state = SimpleNamespace(snapshot=lambda: SimpleNamespace(
            fault="simulated AO host refill stalled past committed output",
            mode="closed-loop", acquiring=True, armed=True, transition_pending=False,
            generation=7, controller="pid", controller_ready=True, controller_calibrated=True,
        ))
        harness.scheduler = SimpleNamespace(
            _next_write_block=4_822,
            stats_snapshot=lambda: {
                "generated_sample_count": 482_000,
                "active_demand_misses": 1,
                "queue_depth": 0,
            },
        )
        harness._advance_virtual = lambda _elapsed: None

        with tempfile.TemporaryDirectory() as directory:
            harness.progress_path = Path(directory) / "progress.jsonl"
            harness._progress_writer = None
            harness._progress_writer_init_lock = threading.Lock()
            harness._progress_diagnostic_lock = threading.Lock()
            harness._progress_diagnostic_error = None
            with patch.object(benchmark.time, "perf_counter", side_effect=(100.0, 100.066)):
                harness._ao_clock_loop()
            harness._close_progress_writer()
            event = json.loads(harness.progress_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(event["type"], "ao_clock_fault")
        self.assertFalse(event["observation_is_fault_onset"])
        self.assertGreaterEqual(event["ao_service_gap_ms"], 65.0)
        self.assertTrue(event["ao_service_gap_over_block_period"])
        self.assertEqual(event["latest_source_sample_index_seen"], 480_200)
        self.assertEqual(event["latest_core_call_started_source_sample_index"], 480_200)
        self.assertEqual(event["core_call_started_not_yet_published_source_sample_index"], 480_200)
        self.assertEqual(event["latest_completed_core_source_sample_index"], 480_100)
        self.assertEqual(event["source_availability_lag_sample_index"], 480_200)
        self.assertAlmostEqual(event["source_availability_lag_ms"], 66.7837)
        self.assertEqual(event["state_observed_after_ao_advance"]["generation"], 7)
        self.assertTrue(event["state_observed_after_ao_advance"]["armed"])
        self.assertEqual(event["recent_core_history"][-1]["state_observation"], "after_core_call")
        self.assertEqual(event["recent_core_history"][-1]["core_processed_generation"], 7)
        self.assertEqual(event["latest_completed_pulse_events"][0]["pulse_id"], "g7-p1")
        self.assertEqual(event["latest_completed_pulse_events"][0]["rendered_sample_count"], 20)
        self.assertEqual(event["ao"]["active_demand_misses"], 1)

    def test_fault_snapshot_does_not_wait_for_an_inflight_real_core_call(self) -> None:
        app = benchmark.app
        config = app.AppConfig(controller="rule", processing_block_ms=5.0,
                               ao_target_lead_ms=100.0)
        config.validate()
        entered_ui = threading.Event()
        release_ui = threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = benchmark._Harness(
                config, "first-fault-context", {"configuration": app.serializable_config(config)},
                progress_path=root / "progress.jsonl", progress_interval_s=60.0,
                paced=True, simulation=False, on_generated=None,
                browser_ui=False, external_browser=False,
            )
            worker: threading.Thread | None = None
            metrics = None
            try:
                harness.process(
                    480_000, np.zeros((2, config.chunk_size), dtype=np.float64),
                    ("CA3", "Cortex"), ("CA3", "EC"), 1,
                )
                self.assertEqual(harness._recent_core_rows[-1].source_sample_index, 480_000)

                original_publish = harness.core._publish_ui

                def blocked_publish(*args, **kwargs):
                    entered_ui.set()
                    if not release_ui.wait(5.0):
                        raise TimeoutError("test Core call was not released")
                    return original_publish(*args, **kwargs)

                harness.core._publish_ui = blocked_publish
                worker = threading.Thread(
                    target=harness.process,
                    args=(480_100, np.zeros((2, config.chunk_size), dtype=np.float64),
                          ("CA3", "Cortex"), ("CA3", "EC"), 1),
                    daemon=True,
                )
                worker.start()
                self.assertTrue(entered_ui.wait(5.0))
                harness.state.fault("runtime AO fault observed")
                ao_worker = harness._ao_clock_thread
                self.assertIsNotNone(ao_worker)
                progress_path = root / "progress.jsonl"
                deadline = time.monotonic() + 2.0
                lines = []
                while time.monotonic() < deadline:
                    if progress_path.exists():
                        lines = progress_path.read_text(encoding="utf-8").splitlines()
                        if any(json.loads(line).get("type") == "ao_clock_fault" for line in lines):
                            break
                    time.sleep(0.002)
                fault_rows = [json.loads(line) for line in lines
                              if json.loads(line).get("type") == "ao_clock_fault"]
                self.assertEqual(len(fault_rows), 1, "AO fault context was not captured promptly")
                self.assertTrue(worker.is_alive(), "Core should still be blocked at snapshot time")

                event = fault_rows[0]
                self.assertEqual(event["type"], "ao_clock_fault")
                self.assertEqual(event["core_call_started_not_yet_published_source_sample_index"], 480_100)
                self.assertEqual(event["latest_completed_core_source_sample_index"], 480_000)
                completed_row = event["recent_core_history"][-1]
                self.assertEqual(completed_row["source_sample_index"], 480_000)
                self.assertGreaterEqual(completed_row["end_perf_counter_ns"],
                                        completed_row["start_perf_counter_ns"])
                self.assertAlmostEqual(
                    completed_row["wall_ms"],
                    (completed_row["end_perf_counter_ns"] -
                     completed_row["start_perf_counter_ns"]) / 1e6,
                )
                self.assertGreaterEqual(completed_row["thread_cpu_ms"], 0.0)
                self.assertGreaterEqual(completed_row["process_cpu_ms"], 0.0)
                self.assertIsInstance(completed_row["gc_overlap"], bool)
                self.assertEqual(completed_row["state_observation"], "after_core_call")
                state_at_observation = event["state_observed_after_ao_advance"]
                self.assertFalse(state_at_observation["acquiring"])
                self.assertFalse(state_at_observation["armed"])
                self.assertEqual(state_at_observation["fault"], "runtime AO fault observed")
                self.assertIn("sampled sequentially, not atomically", event["snapshot_semantics_note"])
                self.assertIn("outside the recorded Core interval", event["history_overhead_note"])
            finally:
                release_ui.set()
                if worker is not None:
                    worker.join(timeout=5.0)
                metrics = harness.finish()
                if worker is not None:
                    self.assertFalse(worker.is_alive())

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.runtime_fault, "runtime AO fault observed")
        self.assertEqual(metrics.processed_blocks, 2)
        acceptance = benchmark.evaluate_case_acceptance(
            metrics, "matrix", requested_scope="full_replay_case")
        self.assertFalse(acceptance["full_acceptance_passed"])


if __name__ == "__main__":
    unittest.main()
