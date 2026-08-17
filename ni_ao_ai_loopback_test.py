"""Verify an AO0-to-AI0/AI1 loopback on a National Instruments DAQ.

Connect AO0 directly to AI0 and AI1 before running.  The script writes a
low-amplitude sine wave, continuously acquires both inputs, and reports their
amplitude, frequency, and error relative to the generated waveform.

Example:
    python ni_ao_ai_loopback_test.py --device Dev1

Use Ctrl+C to stop early.  AO0 is explicitly returned to 0 V on every exit
path.
"""

from __future__ import annotations

import argparse
import contextlib
import time

import numpy as np

try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, RegenerationMode, TerminalConfiguration
    from nidaqmx.stream_readers import AnalogMultiChannelReader
    from nidaqmx.stream_writers import AnalogSingleChannelWriter
except (ImportError, OSError) as exc:
    raise SystemExit(
        "NI-DAQmx Python support is unavailable. Install nidaqmx and the NI-DAQmx driver. "
        f"Details: {exc}"
    ) from exc


def physical(device: str, channel: str) -> str:
    return channel if "/" in channel else f"{device}/{channel}"


def validate_hardware(device_name: str, ai_channels: tuple[str, str], ao_channel: str) -> None:
    system = nidaqmx.system.System.local()
    device_names = [device.name for device in system.devices]
    if device_name not in device_names:
        raise RuntimeError(f"NI device {device_name!r} not found; available devices: {device_names}")
    device = system.devices[device_name]
    available_ai = {channel.name for channel in device.ai_physical_chans}
    available_ao = {channel.name for channel in device.ao_physical_chans}
    missing_ai = set(ai_channels) - available_ai
    if missing_ai:
        raise RuntimeError(f"AI channel(s) not available: {sorted(missing_ai)}")
    if ao_channel not in available_ao:
        raise RuntimeError(f"AO channel not available: {ao_channel}")
    print(f"Connected to {device.name}: {device.product_type}")
    print(f"  output: {ao_channel}; inputs: {', '.join(ai_channels)}")


def estimate_frequency(samples: np.ndarray, sample_rate: float) -> float:
    """Estimate frequency from positive-going zero crossings."""
    centered = samples - np.mean(samples)
    crossings = np.flatnonzero((centered[:-1] < 0) & (centered[1:] >= 0))
    if crossings.size < 2:
        return float("nan")
    return sample_rate / float(np.mean(np.diff(crossings)))


def force_output_zero(ao_channel: str) -> None:
    """Set AO to 0 V after any timed task using it has been closed."""
    with nidaqmx.Task("loopback-ao-zero") as zero_task:
        zero_task.ao_channels.add_ao_voltage_chan(ao_channel, min_val=-1.0, max_val=1.0)
        zero_task.write(0.0, auto_start=True)


def run(args: argparse.Namespace) -> None:
    if args.sample_rate <= 0 or args.block_size <= 0 or args.duration <= 0:
        raise ValueError("sample rate, block size, and duration must be positive")
    if not 0 < args.amplitude <= 1.0:
        raise ValueError("amplitude must be in (0, 1.0] V for this safe loopback test")
    if not 0 < args.frequency < args.sample_rate / 2:
        raise ValueError("frequency must be greater than 0 and below the Nyquist frequency")

    ai_channels = tuple(physical(args.device, channel) for channel in args.ai)
    ao_channel = physical(args.device, args.ao)
    validate_hardware(args.device, ai_channels, ao_channel)
    terminal_name = {"DIFFERENTIAL": "DIFF", "PSEUDODIFFERENTIAL": "PSEUDO_DIFF"}.get(
        args.terminal_config, args.terminal_config
    )
    terminal = getattr(TerminalConfiguration, terminal_name, None)
    if terminal is None:
        raise ValueError(f"unsupported terminal configuration: {args.terminal_config}")

    total_samples = 0
    captured: list[np.ndarray] = []
    started = time.monotonic()
    report_at = started

    with nidaqmx.Task("loopback-ai") as ai_task, nidaqmx.Task("loopback-ao") as ao_task:
        for channel in ai_channels:
            ai_task.ai_channels.add_ai_voltage_chan(
                channel, terminal_config=terminal, min_val=-10.0, max_val=10.0
            )
        ao_task.ao_channels.add_ao_voltage_chan(ao_channel, min_val=-1.0, max_val=1.0)

        ai_task.timing.cfg_samp_clk_timing(
            args.sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=args.block_size * 8,
        )
        ao_task.timing.cfg_samp_clk_timing(
            args.sample_rate,
            source=f"/{args.device}/ai/SampleClock",
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=args.block_size * 8,
        )
        # This is a fixed waveform connectivity test, not closed-loop control.
        # Preload and regenerate the waveform in hardware so host scheduling
        # cannot create an AO underrun and obscure the wiring result.
        ao_task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
        ao_task.out_stream.output_buf_size = args.block_size * 8
        ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(ai_task.triggers.start_trigger.term)

        reader = AnalogMultiChannelReader(ai_task.in_stream)
        writer = AnalogSingleChannelWriter(ao_task.out_stream, auto_start=False)
        read_buffer = np.empty((2, args.block_size), dtype=np.float64)

        waveform_samples = args.block_size * 8
        waveform_indices = np.arange(waveform_samples)
        waveform = args.amplitude * np.sin(
            2 * np.pi * args.frequency * waveform_indices / args.sample_rate
        )
        # AO must contain samples before either synchronized task starts.
        writer.write_many_sample(waveform, timeout=5.0)
        ao_task.start()
        ai_task.start()
        print(
            f"Running {args.duration:g}s: {args.amplitude:g} V peak, "
            f"{args.frequency:g} Hz sine. Press Ctrl+C to stop."
        )

        try:
            while time.monotonic() - started < args.duration:
                reader.read_many_sample(read_buffer, number_of_samples_per_channel=args.block_size, timeout=2.0)

                centered = read_buffer - read_buffer.mean(axis=1, keepdims=True)
                rms = np.sqrt(np.mean(centered**2, axis=1))
                peak_to_peak = np.ptp(read_buffer, axis=1)
                total_samples += args.block_size
                captured.append(read_buffer.copy())

                now = time.monotonic()
                if now >= report_at:
                    print(
                        f"  {now - started:5.1f}s | "
                        f"AI0: {peak_to_peak[0]:.4f} Vpp, {rms[0]:.4f} Vrms | "
                        f"AI1: {peak_to_peak[1]:.4f} Vpp, {rms[1]:.4f} Vrms"
                    )
                    report_at = now + 1.0
        finally:
            # Do not write while regenerating: changing an active regeneration
            # buffer can interleave old and new samples.  A separate on-demand
            # task writes zero after this task has been released.
            with contextlib.suppress(Exception):
                ao_task.stop()
            with contextlib.suppress(Exception):
                ai_task.stop()

    data = np.hstack(captured) if captured else np.empty((2, 0))
    if data.shape[1] == 0:
        print("No samples acquired.")
        return
    print("\nLoopback summary")
    for index, channel in enumerate(ai_channels):
        freq = estimate_frequency(data[index], args.sample_rate)
        print(
            f"  {channel}: mean={np.mean(data[index]):+.5f} V, "
            f"Vpp={np.ptp(data[index]):.5f} V, estimated frequency={freq:.2f} Hz"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="Dev1", help="NI-DAQmx device name")
    parser.add_argument("--ai", nargs=2, default=("ai0", "ai1"), metavar=("AI0", "AI1"))
    parser.add_argument("--ao", default="ao0")
    parser.add_argument("--terminal-config", default="RSE", choices=("DIFFERENTIAL", "RSE", "NRSE"))
    parser.add_argument("--sample-rate", type=float, default=20_000.0, help="samples/s per channel")
    parser.add_argument("--block-size", type=int, default=200, help="samples per read/write block")
    parser.add_argument("--frequency", type=float, default=100.0, help="test sine frequency in Hz")
    parser.add_argument("--amplitude", type=float, default=0.1, help="sine peak amplitude in V (max 1.0)")
    parser.add_argument("--duration", type=float, default=10.0, help="test duration in seconds")
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    try:
        run(parsed_args)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        try:
            force_output_zero(physical(parsed_args.device, parsed_args.ao))
        except Exception as exc:
            print(f"WARNING: could not explicitly return AO0 to 0 V: {exc}")
        else:
            print("AO0 was returned to 0 V.")
        print("AO0 was returned to 0 V.")
