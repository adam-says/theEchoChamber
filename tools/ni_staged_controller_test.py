#!/usr/bin/env python3
"""Run an explicit NI AO/AI bench stage or print its dry-run plan.

This tool never imports NI-DAQmx or contacts hardware unless ``--execute`` is
present. Connect AO only to a scope or rated dummy load for the two AO stages.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGES = ("ao-zero", "ao-waveform", "ai-zero")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--execute", action="store_true", help="open NI hardware and run the selected stage")
    parser.add_argument("--device", default="Dev1")
    parser.add_argument("--ai", nargs=2, default=("ai0", "ai1"), metavar=("AI0", "AI1"))
    parser.add_argument("--ao", default="ao0")
    parser.add_argument("--terminal-config", choices=("DIFFERENTIAL", "RSE", "NRSE"), default="RSE")
    parser.add_argument("--sample-rate", type=int, default=20_000)
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--amplitude-v", type=float, default=0.1)
    parser.add_argument("--frequency-hz", type=float, default=100.0)
    parser.add_argument("--log", type=Path, help="JSON Lines evidence log; defaults to a timestamped file under work/validation_runs")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_rate != 20_000:
        raise ValueError("staged NI tests use the accepted 20 kHz sample rate")
    if args.block_size != 100:
        raise ValueError("staged NI tests use the accepted 100-sample block size")
    if not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0:
        raise ValueError("duration must be finite and positive")
    if not math.isfinite(args.amplitude_v) or not 0 < args.amplitude_v <= 0.1:
        raise ValueError("AO waveform amplitude must be in (0, 0.1] V for the scope or dummy-load stage")
    if not math.isfinite(args.frequency_hz) or not 0 < args.frequency_hz < args.sample_rate / 2:
        raise ValueError("waveform frequency must be positive and below Nyquist")


def _base_result(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "echo-chamber-ni-staged-test-v1",
        "stage": args.stage,
        "device": args.device,
        "ai_channels": list(args.ai),
        "ao_channel": args.ao,
        "sample_rate_hz": args.sample_rate,
        "block_samples": args.block_size,
        "block_ms": 1_000 * args.block_size / args.sample_rate,
        "duration_seconds": args.duration_seconds,
        "amplitude_v": args.amplitude_v if args.stage == "ao-waveform" else 0.0,
        "frequency_hz": args.frequency_hz if args.stage == "ao-waveform" else None,
        "regeneration": "disabled",
        "physical_pulse_claims": False,
        "hardware_accessed": False,
        "status": "planned",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_log(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, allow_nan=False, separators=(",", ":")) + "\n")
        stream.flush()


def _default_log_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    directory = root / "work" / "validation_runs"
    return directory / f"ni_stage_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl"


def _dry_run(args: argparse.Namespace, result: dict[str, Any]) -> int:
    result["dry_run"] = True
    result["required_connection"] = (
        "AO to scope or rated dummy load" if args.stage in {"ao-zero", "ao-waveform"}
        else "two AI inputs plus AO zero output; do not connect to a preparation"
    )
    result["planned_actions"] = {
        "ao-zero": ["configure AO continuous sample clock", "disable regeneration", "queue zero blocks", "measure output externally", "return AO to zero"],
        "ao-waveform": ["precompute finite low-amplitude changing waveform", "configure AO continuous sample clock", "disable regeneration", "refill AO blocks", "return AO to zero"],
        "ai-zero": ["configure two AI channels at 20 kHz", "configure AO zeros from the AI sample clock", "disable regeneration", "read AI while refilling AO zeros", "return AO to zero"],
    }[args.stage]
    result["evidence_required"] = [
        "scope or dummy-load recording", "device model and serial", "driver version",
        "software Git revision", "measured AO zero and onset", "late blocks or DAQ errors",
    ]
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.log:
        _write_log(args.log, {"type": "dry_run", **result})
    return 0


def _physical_name(device: str, channel: str) -> str:
    return channel if "/" in channel else f"{device}/{channel}"


def _force_zero(nidaqmx: Any, ao_channel: str) -> None:
    with nidaqmx.Task("echo-chamber-stage-zero") as task:
        task.ao_channels.add_ao_voltage_chan(ao_channel, min_val=-1.0, max_val=1.0)
        task.write(0.0, auto_start=True)


def _load_ni_apis() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Load NI APIs only after the caller explicitly selects hardware execution."""
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, RegenerationMode, TerminalConfiguration
    from nidaqmx.stream_readers import AnalogMultiChannelReader
    from nidaqmx.stream_writers import AnalogMultiChannelWriter, AnalogSingleChannelWriter
    return (nidaqmx, AcquisitionType, RegenerationMode, TerminalConfiguration,
            AnalogMultiChannelReader, AnalogMultiChannelWriter, AnalogSingleChannelWriter)


def _hardware_run(args: argparse.Namespace, result: dict[str, Any], log_path: Path) -> dict[str, Any]:
    # This import and every hardware lookup stay behind the explicit execute flag.
    (nidaqmx, AcquisitionType, RegenerationMode, TerminalConfiguration,
     AnalogMultiChannelReader, AnalogMultiChannelWriter, AnalogSingleChannelWriter) = _load_ni_apis()

    ai_channels = tuple(_physical_name(args.device, channel) for channel in args.ai)
    ao_channel = _physical_name(args.device, args.ao)
    device_names = [device.name for device in nidaqmx.system.System.local().devices]
    if args.device not in device_names:
        raise RuntimeError(f"NI device {args.device!r} not found; available devices: {device_names}")
    terminal_name = {"DIFFERENTIAL": "DIFF", "NRSE": "NRSE", "RSE": "RSE"}[args.terminal_config]
    terminal = getattr(TerminalConfiguration, terminal_name)
    chunk = args.block_size
    preload_blocks = 8
    preload_samples = preload_blocks * chunk
    zero_block = np.zeros(chunk, dtype=np.float64)
    started = time.monotonic()
    next_report = started
    written_samples = 0
    read_samples = 0
    min_space_available: int | None = None
    block_write_max_ms = 0.0
    result.update({"hardware_accessed": True, "status": "running", "dry_run": False,
                   "ai_physical_channels": list(ai_channels), "ao_physical_channel": ao_channel,
                   "terminal_config": args.terminal_config})
    _write_log(log_path, {"type": "stage_started", **result})

    def note_queue(task: Any) -> None:
        nonlocal min_space_available
        try:
            space = int(task.out_stream.space_avail)
            min_space_available = space if min_space_available is None else min(min_space_available, space)
        except Exception:
            pass

    def write_one(writer: Any, task: Any, values: np.ndarray) -> None:
        nonlocal written_samples, block_write_max_ms
        started_write = time.perf_counter_ns()
        writer.write_many_sample(values, timeout=2.0)
        block_write_max_ms = max(block_write_max_ms, (time.perf_counter_ns() - started_write) / 1e6)
        written_samples += int(values.shape[-1])
        note_queue(task)

    def log_progress() -> None:
        nonlocal next_report
        now = time.monotonic()
        if now < next_report:
            return
        sample = {
            "type": "stage_progress", "stage": args.stage,
            "elapsed_seconds": now - started, "ai_samples_read": read_samples,
            "ao_samples_written": written_samples, "ao_space_available_min": min_space_available,
            "max_block_write_ms": block_write_max_ms,
            "hardware_accessed": True, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_log(log_path, sample)
        print(json.dumps(sample, separators=(",", ":")), flush=True)
        next_report = now + 1.0

    def wait_for_ao_generated(task: Any, target_samples: int) -> None:
        deadline = time.monotonic() + args.duration_seconds + 5.0
        while int(task.out_stream.total_samp_per_chan_generated) < target_samples:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"AO generated {task.out_stream.total_samp_per_chan_generated} samples; "
                    f"expected at least {target_samples}"
                )
            log_progress()
            time.sleep(0.01)

    try:
        if args.stage in {"ao-zero", "ao-waveform"}:
            with nidaqmx.Task("echo-chamber-stage-ao") as ao_task:
                ao_task.ao_channels.add_ao_voltage_chan(ao_channel, min_val=-1.0, max_val=1.0)
                ao_task.timing.cfg_samp_clk_timing(
                    args.sample_rate, sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=preload_samples,
                )
                ao_task.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION
                ao_task.out_stream.output_buf_size = preload_samples
                writer = AnalogSingleChannelWriter(ao_task.out_stream, auto_start=False)
                if args.stage == "ao-zero":
                    preload = np.zeros(preload_samples, dtype=np.float64)
                    next_values = lambda offset: zero_block
                    total_samples = max(preload_samples, int(round(args.duration_seconds * args.sample_rate)))
                else:
                    total_samples = max(preload_samples, int(round(args.duration_seconds * args.sample_rate)))
                    t = np.arange(total_samples, dtype=np.float64) / args.sample_rate
                    pulse_phase = np.mod(t, 0.5)
                    pulse = pulse_phase < 0.005
                    waveform = np.zeros(total_samples, dtype=np.float64)
                    waveform[pulse] = args.amplitude_v * np.sin(2 * np.pi * args.frequency_hz * pulse_phase[pulse])
                    preload = np.pad(waveform[:preload_samples], (0, max(0, preload_samples-waveform.size)))[:preload_samples]
                    next_values = lambda offset: np.pad(
                        waveform[offset:min(offset + chunk, waveform.size)],
                        (0, max(0, offset + chunk - waveform.size)),
                    )
                writer.write_many_sample(preload, timeout=5.0)
                written_samples = preload_samples
                ao_task.start()
                blocks = max(0, int(math.ceil(max(0, total_samples-preload_samples) / chunk)))
                for block in range(blocks):
                    if args.stage == "ao-zero":
                        values = zero_block
                    else:
                        offset = preload_samples + block * chunk
                        values = next_values(offset)
                    write_one(writer, ao_task, values)
                    log_progress()
                wait_for_ao_generated(ao_task, total_samples)
                with contextlib.suppress(Exception):
                    ao_task.stop()
        else:
            with nidaqmx.Task("echo-chamber-stage-ai") as ai_task, nidaqmx.Task("echo-chamber-stage-ai-zero-ao") as ao_task:
                for channel in ai_channels:
                    ai_task.ai_channels.add_ai_voltage_chan(channel, terminal_config=terminal,
                                                            min_val=-10.0, max_val=10.0)
                ao_task.ao_channels.add_ao_voltage_chan(ao_channel, min_val=-1.0, max_val=1.0)
                ai_task.timing.cfg_samp_clk_timing(
                    args.sample_rate, sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=preload_samples,
                )
                ao_task.timing.cfg_samp_clk_timing(
                    args.sample_rate, source=f"/{args.device}/ai/SampleClock",
                    sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=preload_samples,
                )
                ao_task.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION
                ao_task.out_stream.output_buf_size = preload_samples
                ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(ai_task.triggers.start_trigger.term)
                writer = AnalogSingleChannelWriter(ao_task.out_stream, auto_start=False)
                reader = AnalogMultiChannelReader(ai_task.in_stream)
                reader_buffer = np.empty((2, chunk), dtype=np.float64)
                writer.write_many_sample(np.zeros(preload_samples, dtype=np.float64), timeout=5.0)
                written_samples = preload_samples
                ao_task.start()
                ai_task.start()
                blocks = max(1, int(math.ceil(args.duration_seconds * args.sample_rate / chunk)))
                for _ in range(blocks):
                    reader.read_many_sample(reader_buffer, number_of_samples_per_channel=chunk, timeout=2.0)
                    read_samples += chunk
                    write_one(writer, ao_task, zero_block)
                    log_progress()
                wait_for_ao_generated(ao_task, read_samples)
                with contextlib.suppress(Exception):
                    ao_task.stop()
                with contextlib.suppress(Exception):
                    ai_task.stop()
    except BaseException as exc:
        result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        _write_log(log_path, {"type": "stage_failed", **result,
                              "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        try:
            _force_zero(nidaqmx, ao_channel)
            result["ao_zero_cleanup_succeeded"] = True
        except Exception as exc:
            result["ao_zero_cleanup_succeeded"] = False
            result["ao_zero_cleanup_error"] = f"{type(exc).__name__}: {exc}"
    result.update({
        "status": "completed", "elapsed_seconds": time.monotonic() - started,
        "ai_samples_read": read_samples, "ao_samples_written": written_samples,
        "ao_space_available_min": min_space_available,
        "max_block_write_ms": block_write_max_ms,
        "block_deadline_ms": 1_000 * chunk / args.sample_rate,
        "block_deadline_misses": None,
        "ao_underruns_observed": 0,
        "daq_errors_observed": 0,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    _write_log(log_path, {"type": "stage_complete", **result})
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    result = _base_result(args)
    if not args.execute:
        return _dry_run(args, result)
    log_path = args.log or _default_log_path()
    final = _hardware_run(args, result, log_path)
    print(json.dumps(final, indent=2, allow_nan=False))
    print(f"Evidence log: {log_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. AO zero cleanup runs when a hardware stage is active.", file=sys.stderr)
        raise SystemExit(130)
