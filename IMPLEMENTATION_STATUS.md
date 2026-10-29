# Implementation and validation status

Status date: 2026-10-29

Current implementation: `88469989` ESN bridge revision

The reviewed software paths listed below are implemented. The current full test
suite passes on Python 3.12.14 and Python 3.14.0. The latest nine-case replay
matrix completed every 30-minute recording but failed strict timing limits.
That matrix predates the current ESN moment-only rebase. Full replay timing and
ESN parity on the current bridge, plus the three paced cases, remain pending.
No hardware requirement has been verified.

## Implemented software behavior

- The runtime loads the checksummed numeric `echo-chamber-esn-inference-v1`
  artifact. It does not load pickle files.
- NumPy and Numba ESN inference reset state between recordings and produce
  chunk-invariant output. The supplied recurrence matches ReservoirPy 0.3.11
  exactly over 10,000 random samples.
- The reviewed moment-only rebase recalculates centered rolling moments while
  preserving the live peak queue and its absolute sample sequence. Contraction
  and drift callers reset scale high-water to the live peak. Window resizing
  still rebuilds peak metadata. The current `esn_bridge.py` SHA-256 is
  `88469989b45429ec270f55e37139c96a1565ae91193dd2113c7b332d4e9c4749`.
- The corrected training path fits separate input and target scalers, uses a
  one-second washout, and resets state for each recording.
- The replay path imports ABF recordings offline and writes calibrated
  electrode values in mV. It preserves channel roles, provenance, finite EOF,
  and discontinuity metadata. It does not apply amplifier gain a second time.
- The controller paths share the processing core, pulse actuator, safety
  adapter, AO scheduler, recorder, and UI packet serializer.
- AO pulse outcomes carry immutable pulse origins and absolute half-open sample
  ranges. The runtime retains plan metadata until AO ownership and local safety
  accounting resolve. Unknown writes remain unresolved rather than becoming
  cancellations.
- PID and rule control require baseline calibration and AO drain before arming.
  Calibration updates are generation and feature-epoch tagged. Prepared
  controller settings apply at a drained processing boundary, and settings
  revisions remain independent of unrelated mode or stimulation changes.
  Faults latch and disarm output.
- HDF5 v5 stores calibrated AI values, raw AI values when available, AO data,
  sample indices, calibration metadata, controller diagnostics, and sparse
  event and gap records.
- Finite-source EOF and application shutdown flush queued AO outcomes after the
  AO producer stops. The core does not create a synthetic source block to flush
  them. Shutdown keeps the recorder open until the DAQ owner and AO producer
  stop.

## Current test status

The full suite passes with 240 tests and 15 subtests under both supported
interpreters. Python 3.12.14 completed in 53.27 seconds. Python 3.14.0 completed
in 47.75 seconds. Run `python -m pytest -q` with each interpreter to reproduce
these counts.

The frozen baseline `6cc4035` passed 189 tests and 6 subtests under both Python
3.12 and 3.14. Its counts are historical and do not describe the current
implementation.

## Previously measured software evidence

The 2026-08-28 benchmarks are short historical measurements. Two synthetic
runs processed 5,000 unpaced blocks each, and two paced runs processed 1,000
blocks each.

| Historical synthetic measure | Result |
|---|---:|
| Unpaced mean | 0.677 to 0.754 ms |
| Unpaced p99 | 2.205 to 2.312 ms |
| Unpaced maximum | 3.798 to 3.800 ms |
| Paced mean | 0.882 to 1.030 ms |
| Paced p99 | 2.371 to 2.375 ms |
| Paced maximum | 2.912 to 3.881 ms |
| Paced simulated AO underruns | 0 |

Both synthetic runs met the p99 target below 2.5 ms and the maximum target
below 5 ms. The recorded replay smoke processed about 2,000 unpaced and 1,000
paced blocks. Its unpaced p99 and maximum were 1.886 and 3.162 ms. Its paced
p99 and maximum were 2.103 and 4.048 ms. These short runs do not establish
full-duration replay acceptance or NI timing behavior. The source JSON files
remain under `docs/benchmark_*_python314_2026-08-28.json`.

The corrected ESN passed its held-out software comparison. Mean Pearson
correlation was `0.260` versus `-0.260` for the historical readout, and mean
nRMSE was `1.007` versus `2.062`. One of five corrected held-out recordings had
weak negative correlation (`-0.034`), so research runs still require
per-recording review.

The incremental ESN rolling-threshold path matched the previous NumPy reference
across all three complete baseline recordings. Trigger indices matched exactly,
and recorded float32 model means and thresholds reconstructed with no
mismatches. That report used the earlier `f6147a59` ESN bridge. The current
moment-only rebase has SHA-256
`88469989b45429ec270f55e37139c96a1565ae91193dd2113c7b332d4e9c4749`. The old
result remains previous-revision evidence. Rerun parity on the current bridge.
Neither result establishes real-time timing or hardware acceptance.

## Synthetic tracking evidence

The final deterministic 64-second PID case uses `kp=0.2`, `ki=0.06`, `kd=0`,
and `target_z=1`. Tracking, plant recovery, and pair acceptance passed in the
software fixture. The separate max-request reference measured a mean reduction
of `0.003076 mV` over source seconds 22 to 42, against a `0.002616 mV` target
correction. Its margin was `0.000460 mV`. The regulated PID generated 32 effect
pulses during the 30-second disturbance, at `1.067 Hz`; this is controller
usage, not the capacity reference. Tracking settled `14.16 s` after the
disturbance began. Final moving-mean absolute error was `0.0532 z`, full-window
raw error SD was `1.9045 z`, and post-settling raw error SD was `1.0168 z`.
All 4,672 eligible observations were present. Source coverage was complete,
with no gaps or skipped samples.

This is software-only evidence from a deterministic plant. It does not prove
physical output, hardware safety, or biological efficacy.

## Full replay matrix: failed timing gates

The `e965d808` matrix completed all nine full replay cases. Each processed
360,000 blocks and all 36,000,000 samples. All nine cases failed the strict
maximum process-time limit of `<5 ms>`, with maxima from 9.959 to 199.559 ms.
The three ESN cases also failed the p99 limit of `<2.5 ms>`, with p99 values
from 2.700 to 3.022 ms. These results predate the current moment-only ESN
rebase. They establish full input and recording coverage for that revision,
not timing acceptance for the current bridge. See
[`docs/validation/replay-e965d80/`](docs/validation/replay-e965d80/).

The frozen `6cc4035` baseline also completed all nine cases. All three PID
cases had positive requested-rate blocks: 80,123 for May 29, 79,272 for Aug 5,
and 32,025 for Aug 6. Each had zero pulse proposals. Positive rate requests
therefore did not become pulses on those traces.

The earlier baseline had different timing results. It exceeded the maximum
limit with maxima from 7.095 to 34.104 ms. Its three ESN p99 values ranged from
4.745 to 4.830 ms. The current matrix results above supersede those numbers for
the `e965d808` revision.

The matrix report separates case coverage from timing acceptance. Full matrix
acceptance requires all nine trace/controller cases. Each case must meet both
timing limits, process every source sample without gaps or skips, report no late
or stale AO commands, active-demand misses, or noncommitted nonzero AO blocks,
complete recording, consume UI packets, and finish AO drain with an empty queue.
The collection cannot pass if any individual full case fails its coverage or
runtime checks.

The separate paced acceptance set remains pending: it requires one 30-minute
run per controller with the real browser page connected. Any browser disconnect
fails the case, even if the page reconnects. The run must meet the same source,
timing, AO, recorder, UI, and drain requirements. The short smoke and synthetic
simulations do not count toward either full-duration set.

## Hardware status: ALL UNVERIFIED

No NI bench run is recorded. The working hardware profile contains assumptions
marked `confirmed_pending_test`; that label does not mean that anyone measured
or confirmed the values. Keep all hardware evidence `UNVERIFIED` until a lab
operator records and reviews it.

Before connecting AO to an isolator or biological preparation, verify the
following stages with a scope or rated dummy load:

1. Confirm continuous AO zero output.
2. Measure the programmed waveform and its return to zero.
3. Calibrate AI gain and channel mapping.
4. Run synchronized AI with zero AO.
5. Add processing and recording.
6. Add the ESN bridge.
7. Add the WebSocket UI and verify every safety trip.
8. Run the complete configuration for at least four hours. Require zero NI AO
   underruns, no unexplained sample gaps, and retained AO headroom.

Record the DAQ, amplifier, and isolator identifiers, calibration values,
operator, date, scope or dummy-load evidence, and application revision. A
software scheduler commit is not evidence of voltage, current, charge, or
delivered stimulation. Follow [the NI staged controller protocol](docs/ni_controller_protocol.md).
