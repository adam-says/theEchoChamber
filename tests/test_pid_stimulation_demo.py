from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from tools.audit_pid_stimulation_demo import (
    EXPECTED_PID,
    PULSE_AMPLITUDE_V,
    PULSE_FREQUENCY_HZ,
    PULSE_SAMPLES,
    _capacity_reference_failures,
    _check_runtime_metrics,
    _collect_range_coverage,
    _read_h5,
)


ACTION_DTYPE = np.dtype([
    ("request_id", "S32"), ("generation", "<u8"),
    ("source_sample_index", "<u8"), ("latest_sample_index", "<u8"),
    ("target_sample_index", "<u8"), ("sample_index", "<u8"),
    ("range_end_sample_index", "<i8"), ("stage", "S24"),
    ("controller", "S16"), ("requested_rate_Hz", "<f4"), ("reason", "S96"),
])


def _action(stage: str, pulse_id: str = "", *, start: int = 0, end: int = -1,
            source: int = 0, target: int = 0) -> tuple:
    return (pulse_id.encode(), 2, source, source + 99, target, start, end,
            stage.encode(), b"pid", 0.0, b"fixture")


def _metrics(*, pulse: bool = True) -> dict:
    ids = ["g2-p1"] if pulse else []
    return {
        "case_id": "fixture", "source_samples": 300, "recording_committed_samples": 300,
        "recording_completed": True, "ao_drain_complete": True, "ao_final_queue_depth": 0,
        "runtime_fault": None, "ao_active_demand_misses": 0, "ao_late_commands": 0,
        "ao_rejected_stale": 0, "ao_noncommitted_nonzero_blocks": 0,
        "source_index_gap_samples": 0, "skipped_source_samples": 0,
        "progress_events_dropped": 0, "progress_events_rejected": 0, "progress_events_pending": 0,
        "physical_pulse_claims": False, "pulse_request_ids": ids, "pulse_commit_ids": ids.copy(),
        "requested_pulse_events": len(ids), "rendered_pulse_events": len(ids),
        "rejected_pulse_events": 0, "unclassified_pulse_events": 0,
        "pulse_commit_samples": PULSE_SAMPLES if pulse else 0,
        "ao_nonzero_generated_samples": 99 if pulse else 0,
    }


def _write_recording(path: Path, *, mode: str = "valid") -> None:
    config = {
        "sample_rate": 20_000, "chunk_size": 100, "ao_target_lead_ms": 100.0,
        "controller": "pid", "controller_settings": {"pid": EXPECTED_PID},
        "safety": {"max_command_v": 5.0, "max_slew_v_per_s": 2_000.0,
                    "max_abs_area_v_s": 0.010},
    }
    metadata = {
        "configuration": config, "controller": "pid",
        "plant_claim": "generic synthetic plant, not tissue model",
        "plant": {"effect_delay_ms": 20.0}, "physical_pulse_claims": False,
    }
    with h5py.File(path, "w") as h5:
        h5.attrs.update({
            "complete": 1 if mode != "truncated" else 0,
            "committed_samples": 300 if mode != "truncated" else 200,
            "format_tag": "echoChamber_H5_v5", "sample_rate_hz": 20_000,
            "metadata_json": json.dumps(metadata),
        })
        signals = h5.create_group("signals")
        signals.create_dataset("ai_raw_V", shape=(2, 300), dtype="f4")
        signals.create_dataset("electrode_mV", shape=(2, 300), dtype="f4")
        ao = np.zeros((1, 300), dtype=np.float32)
        if mode not in {"request_only", "unknown_commit"}:
            wave = PULSE_AMPLITUDE_V * np.sin(
                2.0 * np.pi * PULSE_FREQUENCY_HZ * np.arange(PULSE_SAMPLES) / 20_000)
            ao[0, 100:200] = wave
        signals.create_dataset("ao_command_V", data=ao)

        diagnostics = h5.create_group("diagnostics")
        block_dtype = np.dtype([("sample_offset", "<u8"), ("ai_sample_index", "<u8"),
                                ("ao_target_sample_index", "<u8"), ("sample_count", "<u4")])
        block_rows = np.zeros(3, dtype=block_dtype)
        for index in range(3):
            block_rows[index] = (index * 100, index * 100, index * 100 + 2_000, 100)
        diagnostics.create_dataset("blocks", data=block_rows)
        ctrl_dtype = np.dtype([("sample_offset", "<u8"), ("controller_id", "S8"),
                               ("z_score", "<f4"), ("envelope_mV", "<f4"),
                               ("target_z", "<f4"), ("feature_valid", "u1")])
        ctrl = np.zeros(3, dtype=ctrl_dtype)
        for index in range(3):
            ctrl[index] = (index * 100, b"pid", 0.5, 0.1, 1.0, 1)
        diagnostics.create_dataset("controller_blocks", data=ctrl)

        events = h5.create_group("events")
        events.create_dataset("sample_discontinuities", shape=(0,), dtype="i8")
        if mode == "valid":
            rows = [
                _action("rendered_for_ao", "g2-p1", start=2100, source=100, target=2100),
                _action("ao_reserved", "g2-p1", start=2100, end=2200, source=100, target=2100),
                _action("ao_committed", "g2-p1", start=2100, source=100, target=2100),
                _action("ao_commit", "g2-p1", start=2100, end=2130, source=100, target=2100),
                _action("ao_commit", "g2-p1", start=2130, end=2200, source=100, target=2100),
                _action("drain_acknowledged"),
            ]
        elif mode == "wrong_origin":
            rows = [
                _action("rendered_for_ao", "g2-p1", start=2000, source=0, target=2000),
                _action("ao_reserved", "g2-p1", start=2000, end=2100, source=0, target=2000),
                _action("ao_committed", "g2-p1", start=2000, source=0, target=2000),
                _action("ao_commit", "g2-p1", start=2000, end=2100, source=0, target=2000),
            ]
        elif mode == "request_only":
            rows = [_action("rendered_for_ao", "g2-p1", start=2100, source=100, target=2100)]
        elif mode == "unknown_commit":
            rows = [
                _action("rendered_for_ao", "g2-p1", start=2100, source=100, target=2100),
                _action("ao_reserved", "g2-p1", start=2100, end=2200, source=100, target=2100),
                _action("ao_write_unknown", "g2-p1", start=2100, end=2200, source=100, target=2100),
            ]
        else:
            rows = [_action("drain_acknowledged")]
        events.create_dataset("action_stages", data=np.asarray(rows, dtype=ACTION_DTYPE))


class PidStimulationDemoAuditTests(unittest.TestCase):
    def test_complete_h5_proves_proposal_to_full_commit_and_command_waveform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.h5"
            _write_recording(path)
            failures, data = _read_h5("stimulated", {"metrics": _metrics()}, path, expected_samples=300)
        self.assertEqual(failures, [])
        self.assertIsNotNone(data)
        assert data is not None
        self.assertEqual(data["ao_nonzero_samples"], 99)
        self.assertEqual(len(data["pulse_rows"]), 1)
        self.assertEqual(data["pulse_rows"][0]["committed_samples"], 100)

    def test_positive_request_without_commit_and_generated_waveform_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request-only.h5"
            _write_recording(path, mode="request_only")
            failures, _ = _read_h5("stimulated", {"metrics": _metrics()}, path, expected_samples=300)
        text = " ".join(failures)
        self.assertIn("AO outcome IDs", text)
        self.assertIn("committed AO ranges", text)
        self.assertIn("nonzero count differs", text)

    def test_unknown_write_cannot_count_as_a_committed_pulse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.h5"
            _write_recording(path, mode="unknown_commit")
            failures, _ = _read_h5("stimulated", {"metrics": _metrics()}, path, expected_samples=300)
        self.assertTrue(any("unknown pulse-output ranges" in failure for failure in failures))

    def test_committed_pulse_with_wrong_source_origin_does_not_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-origin.h5"
            _write_recording(path, mode="wrong_origin")
            failures, _ = _read_h5("stimulated", {"metrics": _metrics()}, path, expected_samples=300)
        self.assertTrue(any("safe sine pulse" in failure for failure in failures))

    def test_truncated_h5_is_rejected_even_if_report_claims_a_complete_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.h5"
            _write_recording(path, mode="truncated")
            failures, _ = _read_h5("stimulated", {"metrics": _metrics()}, path, expected_samples=300)
        self.assertTrue(any("complete flag" in failure for failure in failures))
        self.assertTrue(any("committed source extent" in failure for failure in failures))

    def test_phase_runtime_gate_rejects_a_fault_and_rejects_zero_pulse_demo(self) -> None:
        good = _metrics()
        self.assertEqual(_check_runtime_metrics("stimulated", {"metrics": good},
                                               expected_samples=300, requires_pulses=True), [])
        zero = _metrics(pulse=False)
        failures = _check_runtime_metrics("stimulated", {"metrics": zero},
                                          expected_samples=300, requires_pulses=True)
        self.assertTrue(any("no PID pulse proposals" in failure for failure in failures))
        faulted = _metrics()
        faulted["runtime_fault"] = "simulated AO safety fault"
        failures = _check_runtime_metrics("stimulated", {"metrics": faulted},
                                          expected_samples=300, requires_pulses=True)
        self.assertTrue(any("runtime fault" in failure for failure in failures))

    def test_range_coverage_rejects_gaps_and_overlaps(self) -> None:
        dtype = ACTION_DTYPE
        gaps = np.asarray([
            _action("ao_commit", "pulse", start=100, end=140),
            _action("ao_commit", "pulse", start=141, end=200),
        ], dtype=dtype)
        _, errors = _collect_range_coverage(gaps, "pulse", 100)
        self.assertTrue(any("gap or overlap" in error for error in errors))
        overlap = np.asarray([
            _action("ao_commit", "pulse", start=100, end=150),
            _action("ao_commit", "pulse", start=140, end=200),
        ], dtype=dtype)
        _, errors = _collect_range_coverage(overlap, "pulse", 100)
        self.assertTrue(any("gap or overlap" in error for error in errors))

    def test_healthy_capacity_reference_still_rejects_changed_preset(self) -> None:
        capacity = {
            "capacity_reference_reaches_target": True,
            "generated_effect_pulses_in_steady_window": 20,
            "pid_reference_settings": {
                "kp": 1.0, "ki": 0.0, "kd": 0.0,
                "target_z": -999.0, "max_rate_hz": 2.0,
            },
        }
        failures = _capacity_reference_failures(capacity)
        self.assertEqual(len(failures), 1)
        self.assertIn("target_z differs", failures[0])


if __name__ == "__main__":
    unittest.main()
