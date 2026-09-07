"""Load and plot Echo Chamber HDF5 recordings."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np


SUPPORTED_H5_FORMAT_TAGS = {
    "echoChamber_H5_v1", "echoChamber_H5_v2", "echoChamber_H5_v3", "echoChamber_H5_v4"
}


@dataclass(frozen=True)
class Recording:
    data_path: Path
    metadata: dict[str, Any]
    rows: tuple[str, ...]
    data: np.ndarray
    sample_rate: float

    @property
    def time_seconds(self) -> np.ndarray:
        """Time relative to the first recorded AI sample."""
        sample_row = self.rows.index("ai_sample_index") if "ai_sample_index" in self.rows else 0
        sample_ids = self.data[sample_row]
        return (sample_ids - sample_ids[0]) / self.sample_rate


def resolve_recording(path: Path) -> Path:
    """Resolve one explicitly selected recording file."""
    resolved = path.expanduser().resolve()
    if resolved.suffix not in {".h5", ".hdf5"}:
        raise ValueError(f"expected a .h5 or .hdf5 file, got: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def load_recording(data_path: Path) -> Recording:
    """Load and validate an Echo Chamber HDF5 recording."""
    return load_hdf5_recording(data_path)


def load_hdf5_recording(data_path: Path) -> Recording:
    """Load the committed portion of an Echo Chamber HDF5 recording."""
    with h5py.File(data_path, "r") as source:
        if source.attrs.get("schema") != "echo-chamber-recording":
            raise ValueError(f"not an Echo Chamber HDF5 recording: {data_path}")
        format_tag = source.attrs.get("format_tag")
        if format_tag is not None and str(format_tag) not in SUPPORTED_H5_FORMAT_TAGS:
            raise ValueError(f"unsupported Echo Chamber HDF5 format tag: {format_tag}")
        if str(format_tag) in {"echoChamber_H5_v3", "echoChamber_H5_v4"}:
            return load_hdf5_v3_recording(data_path)
        if "data" not in source or "row_names" not in source:
            raise ValueError(f"HDF5 recording is missing required datasets: {data_path}")
        rows = tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in source["row_names"][:]
        )
        committed = min(int(source.attrs.get("committed_samples", source["data"].shape[1])),
                        source["data"].shape[1])
        if committed <= 0:
            raise ValueError(f"recording contains no committed samples: {data_path}")
        data = np.asarray(source["data"][:, :committed])
        metadata = json.loads(str(source.attrs.get("metadata_json", "{}")))
        if format_tag is not None:
            metadata.setdefault("format_tag", str(format_tag))

    configuration = metadata.get("configuration", {})
    sample_rate = float(configuration.get("sample_rate", 0))
    if len(rows) != data.shape[0]:
        raise ValueError("HDF5 row definitions do not match the data matrix")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("metadata configuration.sample_rate must be positive")
    if not np.all(np.diff(data[0]) > 0):
        raise ValueError("AI sample indices are not strictly increasing")
    return Recording(data_path, metadata, rows, data, sample_rate)


def load_hdf5_v3_recording(data_path: Path) -> Recording:
    """Materialize the compact v3 signals needed by the standard viewer."""
    with h5py.File(data_path, "r") as source:
        ai = source.get("signals/ai_raw_V")
        ao = source.get("signals/ao_command_V")
        modes = source.get("events/mode_changes")
        if ai is None or ao is None or modes is None:
            raise ValueError(f"HDF5 v3 recording is missing required datasets: {data_path}")
        committed = min(int(source.attrs.get("committed_samples", ai.shape[1])), ai.shape[1])
        if committed <= 0:
            raise ValueError(f"recording contains no committed samples: {data_path}")
        metadata = json.loads(str(source.attrs.get("metadata_json", "{}")))
        metadata.setdefault("format_tag", str(source.attrs.get("format_tag", "echoChamber_H5_v3")))
        configuration = metadata.get("configuration", {})
        sample_rate = float(source.attrs.get("sample_rate_hz", configuration.get("sample_rate", 0)))
        labels = tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in ai.attrs["channel_labels"]
        )
        gains = tuple(float(value) for value in configuration.get("amplifier_gain", (1.0, 1.0)))
        if len(labels) != 2 or len(gains) != 2:
            raise ValueError("HDF5 v3 requires two channel labels and two amplifier gains")
        raw = np.asarray(ai[:, :committed], dtype=np.float64)
        scaled = raw * (1000.0 / np.asarray(gains, dtype=np.float64)[:, None])
        sample_ids = np.arange(
            int(source.attrs.get("first_ai_sample_index", 0)),
            int(source.attrs.get("first_ai_sample_index", 0)) + committed,
            dtype=np.float64,
        )
        gaps = source.get("events/sample_discontinuities")
        if gaps is not None:
            for gap in gaps[:]:
                offset = int(gap["sample_offset"])
                sample_ids[offset:] += int(gap["actual_ai_sample_index"]) - int(gap["expected_ai_sample_index"])
        mode_values = np.zeros(committed, dtype=np.float64)
        changes = modes[:]
        for index, change in enumerate(changes):
            start = int(change["sample_offset"])
            stop = int(changes[index + 1]["sample_offset"]) if index + 1 < len(changes) else committed
            mode_values[start:stop] = int(change["mode"])
        rows = (
            "ai_sample_index",
            *(f"{label}_electrode_mV" for label in labels),
            "safe_ao_command_V",
            "mode",
        )
        data = np.vstack((sample_ids, scaled, np.asarray(ao[0, :committed]), mode_values))
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("HDF5 v3 sample rate must be positive")
    return Recording(data_path, metadata, rows, data, sample_rate)


def select_time_window(
    recording: Recording, start: float, duration: float | None
) -> tuple[np.ndarray, np.ndarray]:
    time = recording.time_seconds
    if start < 0:
        raise ValueError("--start must be non-negative")
    if duration is not None and duration <= 0:
        raise ValueError("--duration must be positive")

    stop = np.inf if duration is None else start + duration
    mask = (time >= start) & (time < stop)
    if not np.any(mask):
        raise ValueError(
            f"selected window contains no samples; recording duration is {time[-1]:.3f} s"
        )
    return time[mask], recording.data[:, mask]


def display_indices(count: int, max_points: int) -> np.ndarray:
    if max_points <= 0:
        raise ValueError("--max-points must be positive")
    step = max(1, int(np.ceil(count / max_points)))
    return np.arange(0, count, step)


def preferred_lfp_row(rows: Sequence[str], electrode: str) -> tuple[int, str]:
    """Return the best available electrode-referred LFP row and its unit."""
    candidates = (
        (f"{electrode}_electrode_mV", "mV"),
        (f"{electrode}_raw_V", "V"),
    )
    for name, unit in candidates:
        if name in rows:
            return rows.index(name), unit
    raise ValueError(f"recording has no {electrode} LFP row")


def required_display_rows(rows: Sequence[str]) -> tuple[tuple[int, str], tuple[int, str], int, int]:
    """Resolve the four traces in the standard recording view."""
    ca3 = preferred_lfp_row(rows, "CA3")
    cortex = preferred_lfp_row(rows, "Cortex")
    missing = [name for name in ("mode", "safe_ao_command_V") if name not in rows]
    if missing:
        raise ValueError(f"recording is missing required row(s): {', '.join(missing)}")
    return ca3, cortex, rows.index("mode"), rows.index("safe_ao_command_V")


def plot_recording(
    recording: Recording,
    *,
    start: float = 0.0,
    duration: float | None = None,
    max_points: int = 200_000,
    save: Path | None = None,
    show: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires matplotlib; install requirements.txt") from exc

    time, window = select_time_window(recording, start, duration)
    indices = display_indices(time.size, max_points)
    plot_time = time[indices]
    plot_data = window[:, indices]
    ca3, cortex, mode_row, ao_row = required_display_rows(recording.rows)
    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(14, 10), squeeze=False)
    axes_1d = axes[:, 0]

    for axis, title, (row_index, unit) in zip(
        axes_1d[:2], ("CA3", "CTX"), (ca3, cortex)
    ):
        axis.plot(plot_time, plot_data[row_index], linewidth=0.8)
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
        axis.set_ylabel(unit)
        axis.grid(True, alpha=0.25)

    axes_1d[2].step(plot_time, plot_data[mode_row], where="post", linewidth=0.9)
    axes_1d[2].set_title("Mode", loc="left", fontsize=10, fontweight="bold")
    if recording.metadata.get("format_tag") in {
        "echoChamber_H5_v2", "echoChamber_H5_v3", "echoChamber_H5_v4"
    }:
        mode_values = [0, 1, 2, 3]
        mode_labels = [
            "open-loop", "closed-loop off", "closed-loop passthrough", "closed-loop pulse"
        ]
        axes_1d[2].set_ylim(-0.35, 3.35)
    else:
        # Version 1 recorded only the broad control/closed-loop state.
        mode_values = [0, 1]
        mode_labels = ["open-loop", "closed-loop (mode not recorded)"]
        axes_1d[2].set_ylim(-0.2, 1.2)
    axes_1d[2].set_yticks(mode_values, labels=mode_labels)
    axes_1d[2].grid(True, alpha=0.25)

    ao_delay = float(recording.metadata.get("ao_pipeline_delay_seconds", 0.0) or 0.0)
    axes_1d[3].plot(plot_time + ao_delay, plot_data[ao_row], linewidth=0.8)
    title = "AO command"
    if ao_delay:
        title += f" (target shifted +{ao_delay * 1_000:.1f} ms)"
    axes_1d[3].set_title(title, loc="left", fontsize=10, fontweight="bold")
    axes_1d[3].set_ylabel("V")
    axes_1d[3].grid(True, alpha=0.25)

    axes_1d[-1].set_xlabel("Time from recording start (s)")
    figure.suptitle(recording.data_path.name, fontsize=12)
    figure.tight_layout()

    if save is not None:
        destination = save.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160, bbox_inches="tight")
        print(f"Saved plot: {destination}")
    if show:
        plt.show()
    else:
        plt.close(figure)


def plot_lfp_only(
    recording: Recording,
    *,
    start: float = 0.0,
    duration: float | None = None,
    max_points: int = 200_000,
) -> None:
    """Interactive two-panel LFP view: CA3 above Cortex."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires matplotlib; install requirements.txt") from exc

    time, window = select_time_window(recording, start, duration)
    indices = display_indices(time.size, max_points)
    names = ("CA3", "Cortex")
    rows: list[tuple[str, int, str]] = []
    for name in names:
        row_index, unit = preferred_lfp_row(recording.rows, name)
        rows.append((name, row_index, unit))

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 7), squeeze=False)
    for axis, (name, row_index, unit) in zip(axes[:, 0], rows):
        axis.plot(time[indices], window[row_index, indices], linewidth=0.7)
        axis.set_title(name, loc="left", fontweight="bold")
        axis.set_ylabel(unit)
        axis.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("Time from recording start (s)")
    figure.suptitle(f"{recording.data_path.name} — use the toolbar to zoom/pan", fontsize=12)
    figure.tight_layout()
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recording",
        type=Path,
        help="the single .h5 or .hdf5 recording file to load",
    )
    parser.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    parser.add_argument("--duration", type=float, help="duration to plot in seconds")
    parser.add_argument(
        "--max-points", type=int, default=200_000, help="maximum displayed points per trace"
    )
    parser.add_argument("--save", type=Path, help="save the figure (for example, plot.png)")
    parser.add_argument("--no-show", action="store_true", help="do not open an interactive window")
    parser.add_argument("--list-rows", action="store_true", help="print recorded row names and exit")
    parser.add_argument(
        "--lfp-only", action="store_true",
        help="interactive CA3/Cortex view only, with CA3 on top",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        data_path = resolve_recording(args.recording)
        recording = load_recording(data_path)
        duration = recording.time_seconds[-1]
        print(
            f"Loaded {data_path.name}: {recording.data.shape[1]:,} samples, "
            f"{recording.sample_rate:g} Hz, {duration:.3f} s"
        )
        if args.list_rows:
            for index, name in enumerate(recording.rows):
                print(f"{index:2d}: {name}")
            return 0
        if args.lfp_only:
            plot_lfp_only(
                recording,
                start=args.start,
                duration=args.duration,
                max_points=args.max_points,
            )
            return 0
        plot_recording(
            recording,
            start=args.start,
            duration=args.duration,
            max_points=args.max_points,
            save=args.save,
            show=not args.no_show,
        )
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
