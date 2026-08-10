"""Load and plot Echo Chamber ``.npyseq`` recordings.

The recorder writes a sequence of ordinary NumPy arrays to one file.  This
reader concatenates those blocks, uses the adjacent JSON sidecar to recover
row names and sampling information, and plots the recorded signals without
depending on the acquisition or ESN runtime.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


INDEX_ROWS = {"ai_sample_index", "ao_target_sample_index"}


@dataclass(frozen=True)
class Recording:
    data_path: Path
    metadata_path: Path
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
    if resolved.suffix != ".npyseq":
        raise ValueError(f"expected a .npyseq file, got: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def metadata_path_for(data_path: Path) -> Path:
    return data_path.with_suffix(".json")


def load_recording(data_path: Path) -> Recording:
    """Load and validate all arrays in an Echo Chamber recording."""
    metadata_path = metadata_path_for(data_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata sidecar not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = tuple(metadata.get("rows", ()))
    if not rows:
        raise ValueError(f"metadata contains no row definitions: {metadata_path}")

    blocks: list[np.ndarray] = []
    with data_path.open("rb") as stream:
        while True:
            try:
                block = np.load(stream, allow_pickle=False)
            except EOFError:
                break
            except ValueError as exc:
                if stream.tell() == data_path.stat().st_size:
                    break
                raise ValueError(f"invalid or truncated NumPy block near byte {stream.tell()}") from exc

            block = np.asarray(block)
            if block.ndim != 2:
                raise ValueError(f"recording block must be 2-D, got shape {block.shape}")
            if block.shape[0] != len(rows):
                raise ValueError(
                    f"recording has {block.shape[0]} rows but metadata defines {len(rows)}"
                )
            blocks.append(block)

    if not blocks:
        raise ValueError(f"recording contains no data blocks: {data_path}")

    data = np.concatenate(blocks, axis=1)
    configuration = metadata.get("configuration", {})
    sample_rate = float(configuration.get("sample_rate", 0))
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("metadata configuration.sample_rate must be positive")
    if not np.all(np.diff(data[0]) > 0):
        raise ValueError("AI sample indices are not strictly increasing")

    return Recording(data_path, metadata_path, metadata, rows, data, sample_rate)


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


def signal_groups(rows: Sequence[str], data: np.ndarray) -> list[tuple[str, list[int]]]:
    raw = [index for index, name in enumerate(rows) if name.endswith("_raw_V")]
    calibrated = [
        index
        for index, name in enumerate(rows)
        if index not in raw
        and name not in INDEX_ROWS
        and name not in {"raw_esn_mapped_output", "safe_ao_command_V", "actual_stim_unavailable", "mode"}
        and not name.startswith("actual_stim_monitor")
    ]
    outputs = [
        index
        for index, name in enumerate(rows)
        if name in {"raw_esn_mapped_output", "safe_ao_command_V"}
        or name.startswith("actual_stim_monitor")
    ]
    outputs = [index for index in outputs if not np.all(np.isnan(data[index]))]
    modes = [index for index, name in enumerate(rows) if name == "mode"]

    groups = [
        ("Raw electrode inputs", raw),
        ("Calibrated LFP", calibrated),
        ("ESN and stimulation outputs", outputs),
        ("Experiment mode", modes),
    ]
    return [(title, indices) for title, indices in groups if indices]


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
    groups = signal_groups(recording.rows, plot_data)

    figure, axes = plt.subplots(
        len(groups), 1, sharex=True, figsize=(14, max(3.0, 2.6 * len(groups))), squeeze=False
    )
    axes_1d = axes[:, 0]
    for axis, (title, row_indices) in zip(axes_1d, groups):
        for row_index in row_indices:
            values = plot_data[row_index]
            if recording.rows[row_index] == "mode":
                axis.step(plot_time, values, where="post", label="mode")
                axis.set_yticks([0, 1], labels=["control", "closed-loop"])
            else:
                axis.plot(plot_time, values, linewidth=0.8, label=recording.rows[row_index])
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", fontsize=8)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recording",
        type=Path,
        help="the single .npyseq recording file to load",
    )
    parser.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    parser.add_argument("--duration", type=float, help="duration to plot in seconds")
    parser.add_argument(
        "--max-points", type=int, default=200_000, help="maximum displayed points per trace"
    )
    parser.add_argument("--save", type=Path, help="save the figure (for example, plot.png)")
    parser.add_argument("--no-show", action="store_true", help="do not open an interactive window")
    parser.add_argument("--list-rows", action="store_true", help="print recorded row names and exit")
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
