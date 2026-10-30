# Implementation and validation status

Status date: 2026-10-30

## User-selected biological task

The user accepts the architecture as the basis for the methodological paper
and selects regulation of the CA3 100–250 Hz RMS envelope toward a target. The software
computes a causal envelope with 10 ms smoothing and normalizes it against the
frozen valid baseline. The numeric target, direction, response window,
biological tolerances, and sample-size analysis remain for the experimenters
to prespecify. This design choice does not establish biological validation.
See the [biological task qualification plan](docs/biological_task_qualification.md).

## Neural mock circuit closure

The paired 24-second virtual Rule-off/on R4 run passed all ten declared
circuit-closure gates. Both arms recorded 480,000 samples. The feedback arm
made 22 pulse requests and generated 2,178 nonzero AO samples. Its first
generated output affected the neural-mass state after the configured 20 ms
sample delay. A later valid Rule measurement changed after the actual Core
blanking and feature-recovery path. This confirms virtual circuit closure,
delay, response, and basic software blanking only. It does not establish
biological target tracking, physical artifact recovery, or NI latency. See the
[R4 report](docs/validation/neural-mock/README.md) and the
[mock circuit closure guide](docs/mock_neural_test.md).

The separate [PID stimulation demonstration](docs/validation/pid-stimulation-demo/README.md)
uses a generic responsive plant with a preset 150 Hz carrier. Its stimulated
phase produced 32 unique pulse proposals and all three 64-second phases
completed. It is not a neural model or tissue result.

The nine-case unpaced matrix ran at `bb65477`; the three paced runs used
`c16c0eee0174c7c016d05a5ce617186a3c4c42d8`. The matrix is included because
later changes were outside its measured timed-component scope. The combined
evidence bundle was assembled at c16. Per-run source revisions remain recorded
in the reports. The 286-test suite ran on an uncommitted candidate based on
`996dd1ef7e3a355a2aa98921dea2ac718939219e`, later committed as isolated
`4967720` and cherry-picked to main as `ef81a2348bbf927da10da086a10c555359616f16`.
The runtime-input proof verifies that all 69 captured non-documentation inputs
from that tested suite match the c16 main snapshot, with line-ending-only
differences in 30 files. The final docs/evidence commits are separate; they do
not define a new runtime build.

The reviewed software paths listed below are implemented. The latest full suite
passed 305 tests and 20 subtests on Python 3.12.14 and Python 3.14.0. It ran on
an uncommitted candidate based on isolated `c2e41199191d42b55298baaf46e9c868576d1b2b`.
The 74 captured runtime, model, configuration, UI, and test inputs were
unchanged through both runs. Documentation was outside the capture. The neural
mock plotting postprocessor was checked separately. See the [unit-test
snapshot](docs/validation/neural-mock/unit-tests/).
All 12 required full-duration software runs have now completed: the nine-case
unpaced matrix on `bb65477` and three 30-minute paced runs on `c16c0ee` with an
actual browser. The
combined result is **REQUIRED FULL RUNS FAILED**; see
[`docs/validation/required-runs-c16c0ee/`](docs/validation/required-runs-c16c0ee/).
Rule and PID paced recordings cover the full 36 million samples; the ESN paced
run stopped acquisition and recording at 5,564,500 samples after a simulated AO
active-demand fault. All nine matrix cases and all three paced cases failed at
least one strict timing limit. Hardware remains **ALL UNVERIFIED**.

The c16 ESN fault analysis links all 17 over-deadline Core calls to the bounded
log and conditionally associates the 52.1041 ms UI-submission maximum with the
53.3083 ms call at sample 5,564,400. The timing narrows the location, not the
cause. Two six-minute probes and a scheduler-boundary check completed without
reproducing the fault or establishing a fix. The boundary check confirms the
software 50 ms reserve behavior, not a physical underrun. See the
[fault analysis](docs/validation/esn-fault-investigation-c16c0ee/README-analysis.md)
and [probe results](docs/validation/esn-fault-probes-c2e4119/README.md).

A separate runtime-input proof verified all 69 captured code, test, config,
model, and UI inputs against the unit-test manifest. It found no content
mismatches between the tested inputs and main; 30 files differed only in line
endings. This establishes source/input provenance, not timing or hardware
behavior. The proof is included in the required-runs evidence bundle.

Earlier paced ESN and synthetic tracking results on `6657c88` predate the
progress-reporting fix and do not describe the final full-runtime runs. The
`e965d808` matrix is historical; the `bb65477` matrix is the matrix component
of the final required-run bundle. Per-run provenance and the reason for
combining the revisions are documented in that bundle.
Full-recording ESN parity on the current bridge passed all three canonical recordings; see
[`docs/validation/esn-full-parity-88469989/`](docs/validation/esn-full-parity-88469989/).
No hardware requirement has been verified.

The software benchmark retains its diagnostic limits of p99 `<2.5 ms` and maximum `<5 ms`. These values are not universal biological acceptance thresholds. The biological task, measured output latency, tissue response window, and pass limits remain to be set and measured under the [biological task qualification plan](docs/biological_task_qualification.md).
The current full matrix completed and failed timing acceptance. The 45-second
unpaced lead sweep with virtual AO has a successful 18-case coverage audit, but its
timing gates failed in 15 cases and passed in 3. Full-duration timing
acceptance was not evaluated.
Historical plan and benchmark statements describe their original runs, not
current acceptance.

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
  event and gap records. The writer prepares metadata in source block order
  and appends each populated metadata dataset once per existing recorder batch.
  It preserves source offsets, gaps, mode changes, controller rows, pulse rows,
  and sparse action-stage insertion order. The v5 schema is unchanged.
- The v3/v5 reader reads the file-level format tag once. It uses that value for
  the metadata default and v5 channel-role validation.
- Recorder shutdown preserves the first writer or stop error across cleanup.
  Close retries reuse the same shutdown sentinel. A failed metadata or event
  append leaves the file incomplete and keeps `committed_samples` at the last
  complete batch. Normal stop drains admitted queue tasks before closing the
  file. If stop fails before ownership resolves, cleanup waits for the writer
  to exit before closing the file.
- Finite-source EOF and application shutdown flush queued AO outcomes after the
  AO producer stops. The core does not create a synthetic source block to flush
  them. Shutdown keeps the recorder open until the DAQ owner and AO producer
  stop.
- Benchmark progress reporting uses a bounded queue and one serialization/file
  writer. Interim p99 describes the last at most 2,048 completed Core calls;
  final metrics and acceptance continue to use every call. Accepted, written,
  dropped, rejected, and pending progress observations are accounted
  separately. Unresolved progress-writer or simulated-AO shutdown ownership is
  an explicit case failure; the CLI stops collection and does not write more
  rows to a JSONL file whose owner may still be active. If AO ownership times
  out, dependent resources remain open and any owned temporary recording path
  is retained until a later cleanup retry confirms the owner stopped.
- Responsive simulation summaries report post-settling tracking spread as null
  when coverage is insufficient; the CSV writer also tolerates older summary
  objects that lack this optional metric. This is post-run serialization and
  does not change runtime, Core, recorder, AO, or plant behavior.

## Current test status

The latest full suite passed 305 tests and 20 subtests under Python 3.12.14 and
Python 3.14.0. It ran on the uncommitted `c2e4119` candidate described above.
The frozen logs and manifest are in
[`docs/validation/neural-mock/unit-tests/`](docs/validation/neural-mock/unit-tests/).
The neural-mock plotting postprocessor was checked separately and is not part
of the 74-input capture. The previous full suite passed 286 tests and 17
subtests on both interpreters. The earlier c16 runtime-input proof in
[`docs/validation/required-runs-c16c0ee/`](docs/validation/required-runs-c16c0ee/)
checks 69 code, test, config, model, and UI inputs against the historical
286-test manifest. It found no content mismatches; 30 files differ only in
line endings. That proof does not describe the newer candidate or establish
timing or hardware performance.

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
result remains previous-revision evidence.

The current bridge parity rerun passed on all three full recordings: May 29,
Aug 5, and Aug 6. Each covered 360,000 blocks and 36,000,000 samples. Fired and
trigger sample indices matched exactly; there were no recorded float32 model
mean or threshold reconstruction mismatches. Maximum threshold errors were
`3.410605e-13`, `5.400125e-13`, and `7.212009e-13`, below the unchanged
`max(2e-12, 4 ULP)` gate. The model recurrence and artifact were unchanged. See
[`docs/validation/esn-full-parity-88469989/`](docs/validation/esn-full-parity-88469989/).
The earlier `f6147a59` parity bundle remains historical evidence for the prior
bridge revision. Neither parity result establishes real-time timing or hardware
acceptance.

## Synthetic tracking diagnostic before the progress-reporting fix

The 64-second `6657c88` pre-progress-fix diagnostic used the Core and recorder
identified by its recorded source hashes. Its simulated tracking response,
recovery, and simulation acceptance gates passed, with all 4,673 eligible
target observations and full source coverage. This evidence applies to that
tested revision, not the later progress-reporting fix or current full-runtime
behavior. It is a diagnostic, not real-time acceptance. All three actual-Core
segments exceeded the strict `<5 ms` maximum: stimulated p99/max were
1.483109/7.612 ms, the
no-stimulation control 1.416319/8.4815 ms, and the capacity reference
1.522206/11.0244 ms. The 64-second run is shorter than the 1,800-second timing
requirement. A compact report is in
[`docs/validation/tracking-6657c88/`](docs/validation/tracking-6657c88/).
This software fixture does not prove physical output, hardware safety, or
biological efficacy.

A separate 32-second diagnostic on committed tree `bb65477` processed and
recorded 640,000 samples per controller. Rule measured p99/max of
2.190158/4.2422 ms with zero deadline misses. PID measured 2.439674/5.7394 ms
with one miss. ESN measured 2.776818/5.401 ms with one miss. These short cases
do not establish timing acceptance. The full matrix later completed and
failed timing acceptance. The post-calibration lead sweep used unpaced
processing and virtual AO. It is a 45-second diagnostic prefix, not a full
timing run. Its independent audit verified all
18 controller/lead cases and found no structural issues. Actual eligible
coverage was 28.25 seconds for Rule, 33.99 seconds for PID, and 44.995 seconds
for ESN. Runtime timing gates failed in 15 cases and passed in 3; full-duration
timing acceptance was not evaluated. See
[`docs/validation/lead-sweep-650c4a4/`](docs/validation/lead-sweep-650c4a4/).
The exact short-prefix figures are in
[`docs/validation/reporting-prefix-bb65477/summary.json`](docs/validation/reporting-prefix-bb65477/summary.json).

## Default responsive-simulation reporting attempt

The six-segment default simulation attempt on `2a3a1fb` wrote its aggregate
JSON, then failed while exporting CSV because an incomplete ESN tracking
summary lacked an optional post-settling metric. The retained audit found five
full 480,000-sample recordings and one PID no-stimulation recording truncated
at 273,100 samples by a separate latched simulated-AO fault. The faulted case
reported one active-demand miss, ten late commands, and a 211.9717 ms maximum.
The CLI/reporting failure and the AO/runtime fault are distinct observations;
the CSV null-handling fix does not resolve or explain the AO fault. The audit
also found the functional acceptance gates failed, with modeled capacity
unreachable. The reporting fix is post-run only; it does not change the
runtime processing components measured by the prior full matrix or lead sweep.
The post-fix retry wrote all three CSV rows, but the ESN stimulated segment
had a separate simulated-AO refill fault and all functional acceptance gates
remained false. See the
[`2a3a1fb audit`](docs/validation/default-simulation-2a3a1fb/) and
[`ef81a23 retry audit`](docs/validation/default-simulation-ef81a23/).

## Post-fix synthetic PID tracking diagnostic

The 64-second `ef81a23` tracking run passed the modeled stimulated tracking,
recovery, and capacity-reference checks. It is not full simulation acceptance:
the PID control segment faulted at 1,193,300 of 1,280,000 recorded samples
(93.23%), with one active-demand miss and five late commands. The stimulated,
control, and capacity-reference maxima were 24.0277, 280.6627, and 5.1051 ms,
with 8, 11, and 1 deadline misses respectively. All p99 values were below
2.5 ms, but all maxima exceeded the 5 ms limit. This 64-second run does not
meet the 1,800-second timing requirement. See
[`docs/validation/tracking-ef81a23/`](docs/validation/tracking-ef81a23/).

## Injected-stall fail-safe proof

A targeted 150 ms simulated producer stall on `ef81a23` passed its expected
fault-behavior checks: one pulse proposal was rejected and canceled before
commit; no pulse samples, nonzero AO samples, plant effects, or final rate phase
remained; final drain was empty. The faulted recording is the expected
250,000/320,000-sample prefix, while the uninjected control completed all
320,000 samples. The strict runtime acceptance result is false by design for
this fault-injection case. This is a software check, not hardware proof. See
[`docs/validation/stall-150ms-ef81a23/`](docs/validation/stall-150ms-ef81a23/).

## Full-recording reader check

The v5 reader loaded a complete earlier Rule recording and compared it with
the canonical mV source. It verified 36,000,000 samples from source index zero,
360,000 block rows, 360,000 controller rows, channel roles and EC region,
absolute indices, mode data, raw-AI extent, and AO command samples. All recorded
mismatch counts were zero. The reader and recording hashes stayed unchanged.
This check validates reader coverage for the earlier recording; it does not
establish current timing or physical output. See
[`docs/validation/full-reader-98b6ed/`](docs/validation/full-reader-98b6ed/).

## HDF5 batching evidence

The deterministic recorder fixture compared the prior and updated HDF5 writers
on the same input. All ten datasets matched exactly, including signal arrays,
diagnostic rows, mode and gap rows, pulse rows, and ordered action-stage rows.
The fixture covers a 3-block batch, a 2-block partial final batch, and a source
gap at the batch boundary. Its output is
[the fixture proof](docs/validation/recorder-batching/h5-batching-fixture-parity.json).

Failure tests cover a metadata write error, a one-shot action-stage flush error,
and a failure in the second batch after the first 300 samples commit. They check
that the first error survives close cleanup, the owner stops, the queue drains,
and the file remains incomplete at the prior commit boundary.

An earlier 60-second `f999` recorder pair measured p99 at 2.823 ms with the
writer active and 1.937 ms when metadata-batch processing and HDF5 writes were
discarded while the queue-draining worker remained active. That comparison
motivated the metadata batching change, but it does not isolate the
batched append itself. A separate balanced 10-versus-50 diagnostic improved
p99 but worsened backlog and runtime; the default remains 50. The diagnostic
does not establish a performance benefit from metadata batching.

## Full replay matrix: failed timing gates

The `e965d808` matrix completed all nine full replay cases. Each processed
360,000 blocks and all 36,000,000 samples. All nine cases failed the strict
maximum process-time limit of `<5 ms>`, with maxima from 9.959 to 199.559 ms.
The three ESN cases also failed the p99 limit of `<2.5 ms>`, with p99 values
from 2.700 to 3.022 ms. These results predate the current moment-only ESN
rebase. They establish full input and recording coverage for that revision,
not timing acceptance for the current bridge. See
[`docs/validation/replay-e965d80/`](docs/validation/replay-e965d80/).

The current bridge parity rerun is separate from the matrix: all three
recordings passed exact event and trigger-index checks and the threshold gate.
The `bb65477` full nine-case matrix completed all cases at the 100 ms AO lead.
Every case processed and recorded 36,000,000 samples with no source gaps or
skips. The independent closed-file audit found no structural issues. All nine
failed the strict maximum-time limit, and the three ESN cases also failed p99.
All cases reported zero AO active-demand misses, late commands, stale
rejections, and completed AO drain. The PID matrix cases had positive
requested-rate blocks but zero pulse proposals, commits, and nonzero AO
samples. The detailed nine-case results are in
[`docs/validation/replay-bb65477/`](docs/validation/replay-bb65477/). The
45-second unpaced lead sweep with virtual AO has an audited 18/18 coverage result,
with 15 runtime timing failures and 3 passes; full-duration timing was not
evaluated. The final nine-case matrix and three actual-browser paced runs are
complete and failed strict timing acceptance; see the c16 required-runs bundle.
The parked process-recorder prototype did not show a repeatable timing benefit
and does not isolate GIL contention.

The frozen `6cc4035` baseline also completed all nine cases. All three PID
cases had positive requested-rate blocks: 80,123 for May 29, 79,272 for Aug 5,
and 32,025 for Aug 6. Each had zero pulse proposals. Positive rate requests
therefore did not become pulses on those traces.

The earlier baseline had different timing results. It exceeded the maximum
limit with maxima from 7.095 to 34.104 ms. Its three ESN p99 values ranged from
4.745 to 4.830 ms. The current matrix results above supersede those numbers for
the `e965d808` revision.

The final required-run bundle contains the completed nine-case matrix and
three actual-browser paced results. Together, the 12 required runs failed
timing qualification. The matrix report separates case coverage from timing acceptance. Full matrix
acceptance requires all nine trace/controller cases. Each case must meet both
timing limits, process every source sample without gaps or skips, report no late
or stale AO commands, active-demand misses, or noncommitted nonzero AO blocks,
complete recording, consume UI packets, and finish AO drain with an empty queue.
The collection cannot pass if any individual full case fails its coverage or
runtime checks.

### Paced browser attempts: current ESN failed; earlier Rule and PID attempts are historical

The completed Rule and PID attempts each processed 36,000,000 replay samples,
but a latched AO fault stopped acquisition and further `RecordBlock` submission.
The recorder therefore contains only the clean prefix; the UI remaining open or
recording-enabled does not establish full recording coverage. The summary's
all-call p99 includes calls after the fault, when active control had stopped,
and is not evidence that the paced timing gate passed.

| Controller | Recorded samples | Deadline misses | AO active-demand misses | Browser disconnects | All-call p99 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Rule | 16,608,200 | 578 | 1 | 1 | 1.774801 ms | 428.0297 ms |
| PID | 480,300 | 103 | 1 | 0 | 0.9009 ms | 66.7837 ms |

These Rule and PID results are failed pre-fix attempts from the `09ebf1e` run. Their saved reports are in
[`docs/validation/paced-browser-09ebf1e-attempt1/`](docs/validation/paced-browser-09ebf1e-attempt1/)
and [`docs/validation/paced-pid-09ebf1e/`](docs/validation/paced-pid-09ebf1e/).
The `6657c88` pre-progress-fix ESN attempt faulted near source second 1,080 and
stopped recording at 21,606,700 samples. The replay loop continued and later
processed 36,000,000 source samples. It had 237 processing deadline misses,
p99 2.1158 ms, and maximum 38.7395 ms. The recorder closed cleanly, but the
file contains only the committed prefix. These results do not describe current
full-runtime behavior. The compact report is in
[`docs/validation/paced-esn-6657c88/`](docs/validation/paced-esn-6657c88/).
A full paced acceptance requires one 30-minute run per controller with the real browser
connected and passing source, timing, AO, recorder, UI, and drain gates. Any
browser disconnect fails that case, even if the page reconnects. Short smoke
runs and synthetic simulations do not count.

Use 45 seconds per controller and lead for the post-calibration lead sweep.
Thirty-two seconds is a nominal floor based on an expected 11-second
calibration and settling period; it does not guarantee 20 seconds of eligible
data. Verify at least 400,000 eligible post-calibration samples and contiguous
source coverage in each HDF5 file. A readiness banner or elapsed duration alone
does not establish post-calibration coverage.

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
