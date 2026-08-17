"""Benchmark Echo Chamber processing and a board-free AO consumer simulation.

This tool does not open the WebSocket UI and does not require NI-DAQmx. It uses
the real ESN artifact and application bridge, either with deterministic fake
LFP or with one explicitly selected Echo Chamber HDF5 input file.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import h5py

from echoChamber import SafetyConfig, StimulusSafetyAdapter
from esn_bridge import EchoChamberEsnBridge


@dataclass(frozen=True)
class TimingSummary:
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    maximum_ms: float
    deadline_ms: float
    over_deadline: int


def summarize(values_ms: list[float], deadline_ms: float) -> TimingSummary:
    values = np.asarray(values_ms, dtype=np.float64)
    return TimingSummary(
        count=int(values.size),
        mean_ms=float(np.mean(values)),
        median_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)),
        maximum_ms=float(np.max(values)),
        deadline_ms=deadline_ms,
        over_deadline=int(np.count_nonzero(values > deadline_ms)),
    )


class BlockSource:
    def __init__(self, values: np.ndarray, chunk_size: int) -> None:
        data = np.asarray(values, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < chunk_size:
            raise ValueError(f"input must have shape (at least 2, at least {chunk_size}); got {data.shape}")
        self.values = data[:2]
        self.chunk_size = chunk_size
        self.index = 0

    def next(self) -> np.ndarray:
        indices = (np.arange(self.chunk_size) + self.index) % self.values.shape[1]
        self.index = int((self.index + self.chunk_size) % self.values.shape[1])
        return self.values[:, indices].copy()


def synthetic_lfp(sample_rate: int, seconds: float, seed: int) -> np.ndarray:
    count = max(1, int(sample_rate * seconds))
    samples = np.arange(count, dtype=np.float64)
    time_s = samples / sample_rate
    rng = np.random.default_rng(seed)
    common = 0.05 * np.sin(2 * np.pi * 8 * time_s)
    faster = 0.02 * np.sin(2 * np.pi * 35 * time_s)
    bursts = ((time_s % 4.0) > 3.4) * 0.12 * np.sin(2 * np.pi * 18 * time_s)
    noise = rng.normal(0.0, 0.01, (2, count))
    return np.vstack((common + faster + bursts, 0.8 * common + bursts)) + noise


def load_hdf5(path: Path, maximum_samples: int, electrode_labels: tuple[str, str]) -> np.ndarray:
    with h5py.File(path, "r") as source:
        if source.attrs.get("schema") != "echo-chamber-recording":
            raise ValueError(f"not an Echo Chamber HDF5 recording: {path}")
        rows = tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in source["row_names"][:]
        )
        requested = tuple(f"{label}_raw_V" for label in electrode_labels)
        if not all(name in rows for name in requested):
            raise ValueError(f"recording does not define rows {requested}")
        committed = min(
            int(source.attrs.get("committed_samples", source["data"].shape[1])),
            source["data"].shape[1], maximum_samples,
        )
        return np.vstack([
            np.asarray(source["data"][rows.index(name), :committed], dtype=np.float64)
            for name in requested
        ])


def make_source(args: argparse.Namespace, required_samples: int) -> tuple[BlockSource, str]:
    if args.input is None:
        values = synthetic_lfp(args.sample_rate, max(10.0, required_samples / args.sample_rate), args.seed)
        description = "deterministic synthetic LFP"
    else:
        path = args.input.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix in {".h5", ".hdf5"}:
            values = load_hdf5(path, required_samples, tuple(args.electrode_labels))
        else:
            raise ValueError("--input must be one Echo Chamber .h5 or .hdf5 file")
        description = str(path)
    return BlockSource(values, args.chunk_size), description


def make_processor(args: argparse.Namespace) -> tuple[EchoChamberEsnBridge, StimulusSafetyAdapter]:
    bridge = EchoChamberEsnBridge.load(
        args.artifact.expanduser().resolve(),
        runtime_chunk_size=args.chunk_size,
        sample_rate=args.sample_rate,
        passthrough_dc_block_hz=args.passthrough_dc_block_hz,
        pulse_threshold_std=args.pulse_threshold_std,
        pulse_window_sec=args.pulse_window_sec,
    )
    bridge.configure(stim_mode=args.stim_mode, stim_gain=args.stim_gain)
    safety = StimulusSafetyAdapter(
        SafetyConfig(
            max_command_v=args.max_command_v,
            max_slew_v_per_s=args.max_slew_v_per_s,
            allow_sustained_output_for_dry_test=True,
        ),
        args.sample_rate,
    )
    return bridge, safety


def process_once(
    bridge: EchoChamberEsnBridge,
    safety: StimulusSafetyAdapter,
    source: BlockSource,
    ctx_index: int,
    injected_delay_ms: float,
) -> np.ndarray:
    command = bridge.process(source.next(), ctx_index=ctx_index)
    safe, reason = safety.process(command)
    if reason:
        raise RuntimeError(f"benchmark safety trip: {reason}")
    if injected_delay_ms > 0:
        time.sleep(injected_delay_ms / 1_000.0)
    return safe


def throughput_benchmark(
    process: Callable[[], np.ndarray], blocks: int, deadline_ms: float
) -> TimingSummary:
    timings: list[float] = []
    for _ in range(blocks):
        started = time.perf_counter_ns()
        process()
        timings.append((time.perf_counter_ns() - started) / 1e6)
    return summarize(timings, deadline_ms)


@dataclass(frozen=True)
class PacedResult:
    timing: TimingSummary
    duration_s: float
    produced_blocks: int
    consumed_blocks: int
    consumer_underruns: int
    minimum_queued_blocks: int
    maximum_queued_blocks: int
    producer_schedule_lag_max_ms: float


def paced_consumer_benchmark(
    process: Callable[[], np.ndarray],
    *,
    duration_s: float,
    period_s: float,
    lead_blocks: int,
    chunk_size: int,
) -> PacedResult:
    command_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=lead_blocks * 3)
    for _ in range(lead_blocks):
        command_queue.put_nowait(np.zeros((1, chunk_size), dtype=np.float64))

    stop = threading.Event()
    consumer_underruns = 0
    consumed = 0
    minimum_queued = command_queue.qsize()
    maximum_queued = command_queue.qsize()
    stats_lock = threading.Lock()

    def consume() -> None:
        nonlocal consumer_underruns, consumed, minimum_queued
        deadline = time.perf_counter()
        while not stop.is_set():
            deadline += period_s
            try:
                command_queue.get_nowait()
            except queue.Empty:
                with stats_lock:
                    consumer_underruns += 1
            else:
                with stats_lock:
                    consumed += 1
                    minimum_queued = min(minimum_queued, command_queue.qsize())
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                deadline = time.perf_counter()

    consumer = threading.Thread(target=consume, name="simulated-ao-consumer", daemon=True)
    consumer.start()
    deadline = time.perf_counter()
    finish = deadline + duration_s
    timings: list[float] = []
    produced = 0
    max_lag_ms = 0.0
    try:
        while time.perf_counter() < finish:
            deadline += period_s
            started = time.perf_counter_ns()
            command = process()
            timings.append((time.perf_counter_ns() - started) / 1e6)
            try:
                command_queue.put(command, timeout=period_s)
            except queue.Full:
                # A full queue is harmless for underrun testing; discard the
                # newest simulated command while retaining the lead.
                pass
            else:
                produced += 1
                with stats_lock:
                    maximum_queued = max(maximum_queued, command_queue.qsize())
            lag = time.perf_counter() - deadline
            max_lag_ms = max(max_lag_ms, lag * 1_000.0)
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                deadline = time.perf_counter()
    finally:
        stop.set()
        consumer.join(timeout=2.0)

    with stats_lock:
        return PacedResult(
            timing=summarize(timings, period_s * 1_000.0),
            duration_s=duration_s,
            produced_blocks=produced,
            consumed_blocks=consumed,
            consumer_underruns=consumer_underruns,
            minimum_queued_blocks=minimum_queued,
            maximum_queued_blocks=maximum_queued,
            producer_schedule_lag_max_ms=max_lag_ms,
        )


def print_timing(name: str, timing: TimingSummary) -> None:
    print(
        f"{name}: mean={timing.mean_ms:.3f} ms, median={timing.median_ms:.3f} ms, "
        f"p95={timing.p95_ms:.3f} ms, p99={timing.p99_ms:.3f} ms, "
        f"max={timing.maximum_ms:.3f} ms, over {timing.deadline_ms:.3f} ms="
        f"{timing.over_deadline}/{timing.count}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path(__file__).resolve().parent / "esn_artifact.pkl")
    parser.add_argument("--input", type=Path, help="one Echo Chamber .h5 recording")
    parser.add_argument("--sample-rate", type=int, default=20_000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--ctx-index", type=int, choices=(0, 1), default=1)
    parser.add_argument("--electrode-labels", nargs=2, default=("CA3", "Cortex"))
    parser.add_argument("--stim-mode", choices=("off", "passthrough", "threshold_pulse"), default="passthrough")
    parser.add_argument("--stim-gain", type=float, default=1.0)
    parser.add_argument("--passthrough-dc-block-hz", type=float, default=0.5)
    parser.add_argument("--pulse-threshold-std", type=float, default=1.0)
    parser.add_argument("--pulse-window-sec", type=float, default=10.0)
    parser.add_argument("--max-command-v", type=float, default=1.0)
    parser.add_argument("--max-slew-v-per-s", type=float, default=2_000.0)
    parser.add_argument("--warmup-blocks", type=int, default=100)
    parser.add_argument("--throughput-blocks", type=int, default=2_000)
    parser.add_argument("--paced-seconds", type=float, default=30.0)
    parser.add_argument("--ao-target-lead-ms", type=float, default=100.0)
    parser.add_argument("--inject-delay-ms", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path, help="write complete results as JSON")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.sample_rate <= 0 or args.chunk_size <= 0:
        raise ValueError("sample rate and chunk size must be positive")
    if args.warmup_blocks < 0 or args.throughput_blocks <= 0 or args.paced_seconds <= 0:
        raise ValueError("benchmark counts and duration must be positive")
    period_s = args.chunk_size / args.sample_rate
    lead_blocks_float = args.ao_target_lead_ms / (period_s * 1_000.0)
    if not math.isclose(lead_blocks_float, round(lead_blocks_float)) or lead_blocks_float < 2:
        raise ValueError("AO target lead must resolve to at least two complete processing blocks")
    lead_blocks = int(round(lead_blocks_float))
    paced_blocks = math.ceil(args.paced_seconds / period_s)
    required_samples = max(args.throughput_blocks, paced_blocks, args.warmup_blocks) * args.chunk_size
    source, source_description = make_source(args, required_samples)
    bridge, safety = make_processor(args)

    process = lambda: process_once(
        bridge, safety, source, args.ctx_index, args.inject_delay_ms
    )
    print(f"Input: {source_description}")
    print(
        f"Runtime: {args.sample_rate:,} Hz, {args.chunk_size} samples/block, "
        f"deadline={period_s * 1_000:.3f} ms, simulated AO lead={args.ao_target_lead_ms:.1f} ms"
    )
    for _ in range(args.warmup_blocks):
        process()
    bridge.reset()
    safety.reset()

    throughput = throughput_benchmark(process, args.throughput_blocks, period_s * 1_000.0)
    print_timing("Unpaced processing", throughput)
    bridge.reset()
    safety.reset()
    paced = paced_consumer_benchmark(
        process,
        duration_s=args.paced_seconds,
        period_s=period_s,
        lead_blocks=lead_blocks,
        chunk_size=args.chunk_size,
    )
    print_timing("Paced processing", paced.timing)
    print(
        f"Simulated AO: underruns={paced.consumer_underruns}, "
        f"queued min/max={paced.minimum_queued_blocks}/{paced.maximum_queued_blocks} blocks, "
        f"maximum producer schedule lag={paced.producer_schedule_lag_max_ms:.3f} ms"
    )

    result = {
        "input": source_description,
        "configuration": {
            "sample_rate": args.sample_rate,
            "chunk_size": args.chunk_size,
            "block_deadline_ms": period_s * 1_000.0,
            "ao_target_lead_ms": args.ao_target_lead_ms,
            "stim_mode": args.stim_mode,
            "stim_gain": args.stim_gain,
            "injected_delay_ms": args.inject_delay_ms,
        },
        "throughput": asdict(throughput),
        "paced": asdict(paced),
    }
    if args.json:
        destination = args.json.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote results: {destination}")
    return 1 if paced.consumer_underruns or paced.timing.mean_ms >= paced.timing.deadline_ms else 0


if __name__ == "__main__":
    raise SystemExit(main())
