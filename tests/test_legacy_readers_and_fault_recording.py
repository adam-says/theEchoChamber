from __future__ import annotations

import json
from pathlib import Path
import threading

import h5py
import numpy as np
import pytest

import benchmarkControllers as benchmark
import echoChamber as app
from readEchoChamberData import load_recording, preferred_role_lfp_row
from replay import load_replay


def _write_v1_v2(path: Path, tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_ids = np.array([500, 501, 502, 550, 551], dtype=np.float64)
    cortex_v = np.array([0.001, 0.002, 0.003, 0.004, 0.005], dtype=np.float32)
    ca3_v = np.array([0.002, 0.004, 0.006, 0.008, 0.010], dtype=np.float32)
    ao_v = np.array([0.0, 0.1, 0.0, -0.1, 0.0], dtype=np.float32)
    mode = np.array([0, 1, 1, 0, 1], dtype=np.float32)
    rows = ("ai_sample_index", "Cortex_raw_V", "CA3_raw_V", "safe_ao_command_V", "mode")
    matrix = np.vstack((sample_ids, cortex_v, ca3_v, ao_v, mode))
    # v1/v2 have no roles dataset; this fixture explicitly declares channel 0
    # as Cortex in configuration rather than inferring an ambiguous mapping.
    metadata = {"configuration": {
        "sample_rate": 1_000, "amplifier_gain": [10.0, 20.0], "ctx_index": 0,
    }}
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = "echo-chamber-recording"
        h5.attrs["format_tag"] = tag
        h5.attrs["committed_samples"] = matrix.shape[1]
        h5.attrs["metadata_json"] = json.dumps(metadata)
        h5.create_dataset("row_names", data=np.asarray(rows, dtype=h5py.string_dtype()))
        h5.create_dataset("data", data=matrix)
    return sample_ids, cortex_v, ca3_v


@pytest.mark.parametrize(
    ("tag", "expected_modes"),
    [
        ("echoChamber_H5_v1", [0, 1, 1, 0, 1]),
        ("echoChamber_H5_v2", [0, 1, 2, 3, 3]),
    ],
)
def test_v1_v2_viewer_and_replay_keep_units_indices_modes_and_channel_mapping(
    tmp_path: Path, tag: str, expected_modes: list[int]
) -> None:
    path = tmp_path / f"{tag}.h5"
    sample_ids, cortex_v, ca3_v = _write_v1_v2(path, tag)
    if tag.endswith("v2"):
        with h5py.File(path, "r+") as h5:
            data = h5["data"][:]
            data[-1] = expected_modes
            h5["data"][:] = data

    viewed = load_recording(path)
    assert viewed.sample_rate == 1_000
    assert viewed.rows == ("ai_sample_index", "Cortex_raw_V", "CA3_raw_V",
                           "safe_ao_command_V", "mode")
    np.testing.assert_array_equal(viewed.data[0], sample_ids)
    np.testing.assert_allclose(viewed.data[1], cortex_v)
    np.testing.assert_allclose(viewed.data[2], ca3_v)
    np.testing.assert_array_equal(viewed.data[-1], expected_modes)
    np.testing.assert_allclose(viewed.time_seconds, [0.0, 0.001, 0.002, 0.05, 0.051])

    replay = load_replay(path)
    assert replay.sample_rate_hz == 1_000
    assert replay.channel_labels == ("Cortex", "CA3")
    assert replay.channel_roles == ("Cortex", "CA3")
    assert replay.ctx_index == 0
    assert replay.raw_V is not None
    np.testing.assert_allclose(replay.raw_V, np.vstack((cortex_v, ca3_v)))
    # Cortex gain=10, CA3 gain=20: convert V to mV exactly once per row.
    np.testing.assert_allclose(replay.electrode_mV[0], cortex_v * 100.0)
    np.testing.assert_allclose(replay.electrode_mV[1], ca3_v * 50.0)
    assert [(start, block.shape[1]) for start, block in replay.iter_chunks(3)] == [
        (500, 3), (550, 2),
    ]

    # Legacy viewer rows have no roles metadata. Their recorded labels still
    # identify Cortex and preserve the original raw-voltage unit.
    (row, unit), _region = preferred_role_lfp_row(viewed.rows, "Cortex")
    assert viewed.rows[row] == "Cortex_raw_V"
    assert unit == "V"


def _write_v3(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.array([[0.001, 0.002, 0.003, 0.004],
                    [0.002, 0.004, 0.006, 0.008]], dtype=np.float32)
    ao = np.array([[0.0, 0.1, 0.2, 0.0]], dtype=np.float32)
    mode_dtype = np.dtype([("sample_offset", "<u8"), ("mode", "u1")])
    gap_dtype = np.dtype([("sample_offset", "<u8"),
                          ("expected_ai_sample_index", "<u8"),
                          ("actual_ai_sample_index", "<u8")])
    metadata = {"configuration": {"sample_rate": 2_000,
                                   "amplifier_gain": [10.0, 20.0], "ctx_index": 0}}
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = "echo-chamber-recording"
        h5.attrs["format_tag"] = "echoChamber_H5_v3"
        h5.attrs["sample_rate_hz"] = 2_000
        h5.attrs["first_ai_sample_index"] = 1_000
        h5.attrs["committed_samples"] = raw.shape[1]
        h5.attrs["metadata_json"] = json.dumps(metadata)
        ai = h5.create_dataset("signals/ai_raw_V", data=raw)
        ai.attrs["channel_labels"] = np.asarray(["EC", "CA3"], dtype=h5py.string_dtype())
        h5.create_dataset("signals/ao_command_V", data=ao)
        h5.create_dataset("events/mode_changes", data=np.asarray([(0, 0), (2, 3)], dtype=mode_dtype))
        h5.create_dataset(
            "events/sample_discontinuities",
            data=np.asarray([(2, 1_002, 1_100)], dtype=gap_dtype),
        )
    return raw, ao[0], np.array([0, 0, 3, 3], dtype=np.int64)


def test_v3_viewer_and_replay_scale_volts_once_and_keep_absolute_timeline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-v3.h5"
    raw, ao, expected_modes = _write_v3(path)

    viewed = load_recording(path)
    assert viewed.sample_rate == 2_000
    assert viewed.rows == ("ai_sample_index", "EC_electrode_mV", "CA3_electrode_mV",
                           "safe_ao_command_V", "mode")
    np.testing.assert_array_equal(viewed.data[0], [1_000, 1_001, 1_100, 1_101])
    np.testing.assert_allclose(viewed.data[1], raw[0] * 100.0)
    np.testing.assert_allclose(viewed.data[2], raw[1] * 50.0)
    np.testing.assert_array_equal(viewed.data[-2], ao)
    np.testing.assert_array_equal(viewed.data[-1], expected_modes)
    assert viewed.metadata["channel_labels"] == ["EC", "CA3"]

    replay = load_replay(path)
    assert replay.channel_labels == ("EC", "CA3")
    assert replay.channel_roles == ("Cortex", "CA3")
    assert replay.ctx_index == 0
    assert replay.raw_V is not None
    np.testing.assert_array_equal(replay.raw_V, raw)
    np.testing.assert_allclose(replay.electrode_mV[0], raw[0] * 100.0)
    np.testing.assert_allclose(replay.electrode_mV[1], raw[1] * 50.0)
    assert [(start, values.shape[1]) for start, values in replay.iter_chunks(3)] == [
        (1_000, 2), (1_100, 2),
    ]


def test_benchmark_harness_records_finite_prefix_after_fault_and_fails_full_acceptance(
    tmp_path: Path,
) -> None:
    config = benchmark._make_config("rule", 100.0, tmp_path)
    metadata = {
        "configuration": app.serializable_config(config),
        "draft_fixture": "finite-source-fault-prefix",
    }
    harness = benchmark._Harness(
        config, "finite-source-fault-prefix", metadata,
        progress_path=None, progress_interval_s=3600.0,
        paced=False, simulation=False, on_generated=None,
        browser_ui=False, external_browser=False,
    )
    completed = False
    total_blocks = 8
    prefix_blocks = 2
    first_source_index = 1_000_000
    chunk = config.chunk_size
    samples = np.arange(total_blocks * chunk, dtype=np.float64)
    trace = np.vstack((0.05 * np.sin(samples / 83.0), 0.03 * np.cos(samples / 107.0)))
    try:
        for block_index in range(total_blocks):
            start = block_index * chunk
            harness.process(
                first_source_index + start, trace[:, start:start + chunk].copy(),
                ("CA3", "Cortex"), ("CA3", "Cortex"), 1, stim_enabled=False,
            )
            if block_index == prefix_blocks - 1:
                harness.state.fault("injected latched runtime fault")

        # The finite benchmark source continues through the real Core after
        # fault; Core stops admission, so H5 contains only the acquired prefix.
        assert harness.metrics.source_samples == total_blocks * chunk
        assert harness.metrics.processed_blocks == total_blocks
        assert harness.state.snapshot().fault == "injected latched runtime fault"
        assert harness.state.snapshot().acquiring is False
        processed_before_finish = harness.metrics.processed_blocks
        core_sample_index_before_finish = harness.core.sample_index
        core_last_source_end = harness.core._last_source_end

        metrics = harness.finish()
        completed = True

        assert metrics.runtime_fault == "injected latched runtime fault"
        assert metrics.source_samples == total_blocks * chunk
        assert metrics.processed_blocks == total_blocks
        assert metrics.ao_drain_complete is True
        assert metrics.recording_completed is True
        assert metrics.recording_committed_samples == prefix_blocks * chunk
        # finish drains AO notifications without inventing source blocks.
        assert harness.metrics.processed_blocks == processed_before_finish
        assert harness.core._last_source_end == first_source_index + prefix_blocks * chunk
        assert harness.core._last_source_end == core_last_source_end
        assert harness.core.sample_index == core_sample_index_before_finish

        with h5py.File(harness.recording_path, "r") as h5:
            assert int(h5.attrs["committed_samples"]) == prefix_blocks * chunk
            assert int(h5.attrs["complete"]) == 1
            np.testing.assert_array_equal(
                h5["signals/electrode_mV"][:], trace[:, :prefix_blocks * chunk].astype(np.float32),
            )
            assert h5["signals/electrode_mV"].shape[1] == prefix_blocks * chunk

        acceptance = benchmark.evaluate_case_acceptance(
            metrics, "matrix", requested_scope="full_replay_case",
        )
        assert acceptance["scope"] == "full_replay_case"
        assert acceptance["runtime_checks_passed"] is False
        assert acceptance["full_acceptance_passed"] is False
        assert any("runtime fault was latched" in item for item in acceptance["runtime_failures"])
        assert any("recorded sample count does not match" in item
                   for item in acceptance["runtime_failures"])
        assert metrics.recording_committed_samples != metrics.source_samples
    finally:
        if not completed:
            harness.abort()


class _NoUiCore(app.ProcessingCore):
    def _publish_ui(self, *args, **kwargs):
        pass


def test_mock_daq_latched_fault_stops_source_cursor_and_dummy_frames_do_not_record(
    tmp_path: Path,
) -> None:
    """Distinct from the Harness test: live-like MockDaq stops source reads."""
    from replay import ReplayData

    replay = ReplayData(
        electrode_mV=np.vstack((np.arange(400, dtype=np.float32),
                                -np.arange(400, dtype=np.float32))),
        sample_rate_hz=20_000.0,
        channel_labels=("CA3", "EC"),
        channel_roles=("CA3", "Cortex"),
        ctx_index=1,
        metadata={"fixture": "fault-stop-prefix"},
    )
    config = app.AppConfig(
        sample_rate=20_000, processing_block_ms=5.0, ao_target_lead_ms=100.0,
        record_dir=tmp_path, mock_replay=tmp_path / "finite-source.h5",
        mock_replay_data=replay, open_browser=False,
    )
    config.validate()
    state = app.RuntimeState(
        esn_ready=False, start_paused=False, electrode_labels=replay.channel_labels,
        ctx_index=replay.ctx_index, pulse_threshold_std=3.0, pulse_window_sec=10.0,
        controller="rule", controller_ready={"esn": False, "rule": True, "pid": True},
    )
    state.set_recording(True)
    recorder = app.H5Recorder(config, {"configuration": app.serializable_config(config)})
    recording_path = recorder.start("fault-prefix")
    core = _NoUiCore(config, state, app.EsnRuntime(None, config), recorder,
                      object(), None, app.Telemetry())
    daq = app.MockDaq(config, state, core, core.telemetry)
    acquired_block = threading.Event()
    paused_display_frame = threading.Event()
    original_process = core.process

    def fault_after_first_acquired_block(samples, **kwargs):
        result = original_process(samples, **kwargs)
        if state.snapshot().acquiring and np.isfinite(samples).all():
            if not acquired_block.is_set():
                acquired_block.set()
                state.fault("injected runtime fault after first acquired block")
        elif state.snapshot().fault is not None and not state.snapshot().acquiring:
            paused_display_frame.set()
        return result

    core.process = fault_after_first_acquired_block
    daq_thread = threading.Thread(target=daq.run, name="draft-fault-mock-daq", daemon=True)
    daq_thread.start()
    try:
        assert paused_display_frame.wait(timeout=5.0)
        assert acquired_block.is_set()
        faulted = state.snapshot()
        assert faulted.fault == "injected runtime fault after first acquired block"
        assert faulted.acquiring is False
        assert faulted.armed is False
        assert daq.replay_index == config.chunk_size
    finally:
        state.stop()
        daq_thread.join(timeout=5.0)
        assert not daq_thread.is_alive()
        # Drain scheduler notifications on its owner without synthesizing AI.
        source_index_before_flush = core.sample_index
        core.drain_ao_notifications()
        assert core.sample_index == source_index_before_flush
        state.set_recording(False)
        recorder.stop_recording()
        recorder.close()

    viewed = load_recording(recording_path)
    assert viewed.data.shape[1] == config.chunk_size
    np.testing.assert_array_equal(viewed.data[0], np.arange(config.chunk_size))
    electrode_row = viewed.rows.index("CA3_electrode_mV")
    np.testing.assert_array_equal(viewed.data[electrode_row], replay.electrode_mV[0, :config.chunk_size])
    with h5py.File(recording_path, "r") as h5:
        assert int(h5.attrs["committed_samples"]) == config.chunk_size
        assert int(h5.attrs["complete"]) == 1
