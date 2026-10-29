# Validate the controller paths

Use this guide to run the controller smoke, synthetic simulations, replay
matrix, or paced soak. These checks use software AO scheduling. They do not
measure physical stimulation. Hardware work remains **ALL UNVERIFIED**. Follow
[the NI staged controller protocol](ni_controller_protocol.md) before connecting
AO to an isolator or preparation.

The replay commands use the canonical recordings in
`recordings/ca3_slow_waves/`. Import the ABF files first with the steps in
[the replay import guide](replay_import.md). Each canonical replay contains
36,000,000 samples at 20 kHz. The importer stores the calibrated electrode
signals in mV, preserves CA3 and EC region labels, and applies no second
amplifier-gain conversion.

## Run a short controller smoke

Run one second for every selected trace and controller pair:

```powershell
.\.venv\Scripts\python.exe tools\validate_controllers.py --smoke
```

By default, the smoke selects all three canonical traces. It fails if a
required trace is missing instead of silently omitting that input. The smoke
uses the shared processing core, pulse actuator, safety adapter, recorder, UI
serializer, and AO scheduler. It checks a short real input path. It does not
count toward the 30-minute matrix or paced acceptance.

Pass `--trace PATH` once per trace to select other replay files. Pass
`--controller rule --controller pid` to restrict the controller set.
`--duration-seconds` limits the source prefix for a short diagnostic.

## Run a synthetic simulation

Compare each controller with stimulation enabled and disabled:

```powershell
.\.venv\Scripts\python.exe tools\validate_controllers.py --simulation
```

The deterministic plant is a software fixture, not a tissue model. It applies
an effect only when the simulated AO scheduler generates a nonzero command.
This command uses short scenario durations. It does not count toward replay
acceptance.

Run the PID target-tracking case with the final software-only preset:

```powershell
.\.venv\Scripts\python.exe tools\validate_controllers.py --tracking-simulation
```

The default case runs for 64 seconds with `kp=0.2`, `ki=0.06`, `kd=0`, and
`target_z=1`. It uses a 0.003 mV disturbance from source seconds 12 to 42, a 20 ms
effect delay, a 5 s exponential recovery, and a 0.0005 mV effect per generated
pulse. The tracking case reports source coverage, known feature exclusions,
eligible observations, target error, settling, ripple, recovery, and IAE against
the no-stimulation control.

The case also runs a separate max-request capacity reference through the actual
processing core. It reports the measured steady reduction and target correction
separately from regulated PID output usage. The reference uses `kp=1`, `ki=0`,
`kd=0`, `target_z=-1000`, and a 2 Hz limit. Do not treat reference capacity as
proof that the regulated PID reached its target. Check
`tracking_gate_passed`, `plant_recovery_passed`, and
`simulation_acceptance_passed` separately.

The tracking response gate uses fixed ±0.5 z limits on one-second moving means
over the full 5 ms source-time grid. After settling, each complete moving
window must stay within the band through the disturbance end. Require at least
95% eligible target-window observations, complete source coverage, no source
gaps or skipped samples, and no unexplained invalid feature samples. Count
pulse and software-settling masks as ineligible only when the controller marks
them explicitly. Those masks do not excuse a raw acquisition gap. The gate
also requires full-window raw error SD at or below 2 z. The stimulation-enabled
pair must improve IAE over the disabled control.

The recovery gate is separate. Inspect
`stimulated.plant_recovery.bounded_recovery` in the run JSON. Recovery must
retain a complete window, leave no more than 20% of the pre-recovery effect,
and keep mean requested rate after two recovery constants at or below 0.10 Hz.
The pair summary reports `plant_recovery_passed` and
`simulation_acceptance_passed` alongside `tracking_gate_passed`.

An earlier pre-progress-fix software pair passed these gates. The reference generated 25 pulse
effects from seconds 22 to 42, at 1.25 Hz. It measured a 0.003076 mV mean reduction
against a 0.002616 mV target correction, for a 0.000460 mV margin. The regulated
PID generated 32 effects during the disturbance, at 1.067 Hz. Its formula-based
usage estimate was 0.002667 mV. Tracking settled 14.16 seconds after the
disturbance began. Final moving-mean absolute error was 0.0532 z. Full-window
raw error SD was 1.9045 z, and post-settling raw error SD was 1.0168 z. All
4,672 eligible observations were present, and the source had no gaps or skipped
samples. This result is software-only evidence, not a hardware or biological
claim. The 64-second `6657c88` pre-progress-fix run also passed its simulation
tracking and recovery gates with 4,673 eligible observations and complete
source coverage. Its actual-Core timing gates failed: all three maximum times
exceeded 5 ms. These results do not establish current full-runtime behavior.
See [the compact tracking report](validation/tracking-6657c88/tracking-summary.json).
The older JSON and CSV are under
`work/validation_runs/tracking-kp02-ki006-capacity-reference/`.

The post-fix 64-second tracking run on `ef81a23` passed the modeled stimulated
tracking, recovery, and capacity-reference checks, but not full simulation
acceptance. Its PID control segment faulted at 1,193,300/1,280,000 recorded
samples (93.23%), with one AO active-demand miss and five late commands. The
stimulated/control/capacity maxima were 24.0277/280.6627/5.1051 ms, with
8/11/1 deadline misses. All p99 values remained below 2.5 ms, but every maximum
exceeded 5 ms. The run is shorter than the 1,800-second timing requirement. See
the [post-fix tracking audit](validation/tracking-ef81a23/README.md).

A separate 150 ms simulated producer-stall test on `ef81a23` passed its
expected fail-safe checks. One pulse proposal was canceled before commit; no
pulse samples, nonzero AO samples, or plant effects occurred; final AO drain
was empty. The faulted recording was a contiguous 250,000/320,000-sample
prefix, while its uninjected control completed all 320,000 samples. Runtime
acceptance correctly failed for the injected-fault case. This is not hardware
evidence. See the [stall proof audit](validation/stall-150ms-ef81a23/README.md).

## Run the finite replay matrix

Run all three controllers on each of the three full finite recordings:

```powershell
.\.venv\Scripts\python.exe tools\validate_controllers.py --matrix --keep-recordings
```

The matrix contains nine cases. The runner stops at finite EOF, disarms output,
and drains the simulated AO queue. It then flushes queued typed AO outcomes
without creating another source block. It does not wrap a recording. A run
longer than the available source is rejected. A shorter `--duration-seconds`
run is diagnostic and cannot pass the full-duration gate.

The replay matrix loads the nonzero synthetic-demo PID configuration. It uses
`kp=0.4`, `ki=0.1`, `kd=0.02`, `target_z=1`, and a 2 Hz rate limit. It does not
use the zero-gain hardware PID defaults. In the earlier `e965d808` matrix,
all three PID cases had positive requested-rate blocks but zero pulse
proposals: May 29 had 80,123 requested-rate blocks, phase high-water 0.118351397,
and 29,210 phase resets; Aug 5 had 79,272 blocks, phase high-water 0.146280455,
and 32,764 resets; Aug 6 had 32,025 blocks, phase high-water 0.12974451823226668,
and 8,816 resets. The phase never reached one, so these runs do not establish
effective SWR pulse control. The replay cases did not have equal stimulation
exposure. Compare requested rates and generated pulse counts before comparing
controller outcomes. The 64-second actual-core tracking fixture below provides
separate software actuation and feedback evidence.

The frozen `6cc4035` baseline showed the same requested-rate block counts on
these three traces and also produced zero PID pulse proposals.

Each JSON and CSV case reports its `acceptance_scope`, runtime checks, and
reference coverage. A short run—including `--matrix --duration-seconds 20` or
`--paced --duration-seconds 20`—is labeled `diagnostic_prefix`; full-duration
acceptance is `NOT EVALUATED` and its pass value is null. The report still
records whether runtime checks passed, and a runtime-check failure still makes
the command exit nonzero. Full replay acceptance requires exactly 1,800 source
seconds, 360,000 blocks, 36,000,000 samples, 20 kHz, 100 samples per block, and
the 5 ms processing deadline.
The required and observed values are included in both case JSON and CSV.
Per-case full passes do not stand in for collection completeness: the aggregate
matrix gate requires all nine trace/controller pairs across three traces, and
the paced collection gate requires one full 30-minute case for each of the
three controllers. A deliberately selected subset remains collection-incomplete
even when its individual full cases pass. A malformed full-intent case remains
in full scope and fails coverage or runtime acceptance. It is not relabeled as a
diagnostic. The matrix or paced collection fails if any required case fails.

The unpaced recorder uses a bounded queue. When backlog reaches its configured
threshold, the producer waits between blocks for the writer to drain. This wait
does not advance the AO clock, change source indices, or enter
`AsyncUiProcessingCore.process` timing. A 30-second wait with no queue progress
fails the case. Paced runs do not use this wait policy; recorder overflow fails
the case.

## Sweep AO target lead

The default lead sweep tests 100, 75, 50, 40, 30, and 20 ms:

```powershell
.\.venv\Scripts\python.exe tools\validate_controllers.py --smoke --lead-sweep
```

Replace `--smoke` with `--matrix` to repeat the full matrix at every lead. The
100 ms lead remains the reference setting. The command reports each lead and
does not change a hardware setting. The one-second smoke ends before the
one-second settling and ten-second baseline calibration complete. Treat a
smoke lead sweep as scheduler timing diagnostics, not evidence of active Rule
or PID output.

To inspect controller activity after calibration, pace for 45 seconds at each
lead and controller. Thirty-two seconds is only a nominal floor based on
11 seconds of expected settling and calibration; it may not leave 20 seconds of
eligible data. Start the local UI server, then pass one trace explicitly:

```powershell
python -m http.server 8000 --directory .
.\.venv\Scripts\python.exe tools\validate_controllers.py --paced --external-browser `
  --browser-base-url http://127.0.0.1:8000 `
  --trace recordings\ca3_slow_waves\2026_08_06_0000.h5 `
  --lead-sweep --duration-seconds 45 --keep-recordings
```

Verify that each case records at least 400,000 eligible post-calibration source
samples, with contiguous absolute indices and no gaps. A readiness banner or
elapsed duration does not prove post-calibration coverage. This sweep does not
pass the 30-minute acceptance gate.

## Run the paced 30-minute acceptance set

Serve the local UI, then run one paced case per controller:

```powershell
python -m http.server 8000 --directory .
.\.venv\Scripts\python.exe tools\validate_controllers.py --paced --external-browser `
  --browser-base-url http://127.0.0.1:8000 --paced-seconds 1800 --keep-recordings
```

The runner waits for each real browser page before it starts. Open the URL from
the `browser_required` record in the progress JSONL. By default, the CLI writes
that file to `work/validation_runs/progress.jsonl`. If you set `--output-dir`,
the default progress file is `<output-dir>/progress.jsonl`. Pass `--progress-log`
to choose another path. A browser disconnect fails the run even if the page
reconnects. The report records
disconnect count, disconnected duration, and browser presence at completion.
The check confirms the browser connection and consumed UI packets. It does not
confirm rendered pixels or that a person saw them. Use `--browser-ui` instead
of `--external-browser` to open the page automatically.

Use `--paced-seconds 10` for a pacing smoke. It still waits on real elapsed
time and requires a browser page. It does not count as a 30-minute pass.

## Check fault handling with a producer stall

Inject a 150 ms stall at 12 seconds of source time:

```powershell
.\.venv\Scripts\python.exe tools\validate_controllers.py --smoke --trace PATH_TO_TRACE.h5 `
  --controller pid --duration-seconds 20 --inject-stall-ms 150 --stall-at-second 12
```

The scheduler keeps its own clock. A command that misses its scheduled AO
sample must fault and must not change the synthetic plant. Check the fault,
active-demand misses, pulse request IDs, commit IDs, and final drain state. AO
samples already accepted by the simulated output buffer may play at their
scheduled indices while the queue drains.

## Read the result files

Each run writes JSON and CSV under `work/validation_runs`. The files contain
source hashes, controller settings, timing, source continuity, recorder
completion, UI packet counts, scheduler counters, pulse outcomes, and AO drain
state. The progress JSONL records the process ID, run state, and case progress.

The [full-recording ESN threshold parity report](validation/esn-full-parity/README.md)
documents the historical result for the earlier `f6147a59` bridge. The current
bridge has SHA-256
`88469989b45429ec270f55e37139c96a1565ae91193dd2113c7b332d4e9c4749` and includes
the reviewed moment-only rebase. That rebase recalculates count, origin, mean,
and M2 while preserving the live peak queue and absolute sequence. Contraction
and drift callers reset scale high-water to the current live peak. Window resize
still rebuilds peak metadata. The [current-bridge parity evidence](validation/esn-full-parity-88469989/README.md)
covers all three full recordings, with 360,000 blocks and 36,000,000 samples
per recording. Fired and trigger sample indices matched exactly, and recorded
float32 model means and thresholds reconstructed without mismatches. Maximum
threshold errors were `3.410605e-13`, `5.400125e-13`, and `7.212009e-13`, below
the unchanged `max(2e-12, 4 ULP)` gate. The model recurrence and artifact were
unchanged. The older bundle remains historical evidence. Neither parity result
establishes real-time timing or hardware acceptance.

Earlier suite runs on `09ebf1e`, `c32d151`, and the fault-event candidate are
historical and superseded by the current suite below.

The final suite passed 286 tests and 17 subtests on Python 3.12.14 and Python
3.14.0. It ran on an uncommitted candidate based on `996dd1e`, later committed
as isolated `4967720` and cherry-picked as main `ef81a23`. The root-run suite
logs and source-hash manifest are in [progress-reporting validation](validation/progress-reporting/).
The independent runtime-input proof in
[`validation/required-runs-c16c0ee/`](validation/required-runs-c16c0ee/)
verifies those 69 captured code, test, config, model, and UI inputs against the
c16 main snapshot. It found no content mismatches; 30 files differed only in
line endings. This confirms tested-input provenance, not that the suite ran at
c16, nor timing or hardware performance.
Coverage includes AO ownership and terminal outcomes, calibration and settings
transactions, EOF and shutdown drains, bounded progress reporting, legacy
reader behavior, and optional post-run simulation-metric serialization. The
suite does not establish timing acceptance.

The HDF5 v5 writer prepares block metadata in source order and appends each
populated metadata dataset once per existing batch. It keeps absolute sample
indices, gaps, mode changes, diagnostics, controller rows, pulse rows, and
sparse action-stage order. The file schema does not change. Close retries reuse
one shutdown sentinel. A failed write preserves the first error and leaves the
recording incomplete at the last fully committed batch. Normal stop drains
admitted queue tasks before closing the file. If stop fails before ownership
resolves, cleanup waits for the writer to exit before closing the file.

The reader uses the HDF5 file-level `format_tag` for metadata defaults and v5
channel-role validation. This fixes an undefined `format_tag` lookup that
raised `NameError` on the legacy fallback and malformed-v5 validation paths.

The deterministic old/new fixture matched all ten signal and metadata datasets
exactly. It covers three blocks in one batch, two blocks in the partial final
batch, and a sample gap at the batch boundary. The [fixture proof](validation/recorder-batching/h5-batching-fixture-parity.json)
records the comparison. Failure tests include a second-batch
metadata error after the first 300 samples commit; the incomplete file retains
that boundary.

The recorder also bounds fault-zero event storage. It merges confirmed
pulse-free zero AO writes only when their half-open sample ranges touch and
their generation, controller, and reason match. The AO writer reports an extent
only after it confirms the whole zero refill. Partial or unknown writes cannot
become full ranges. Pulse outcomes and other action stages remain individual
rows in insertion order.

The recorder services event-only data on a timed queue wait, even after source
acquisition stops. One open range, queued rows, and rows in an active HDF5
append share a 4,096-row limit. Overflow latches the first error, stops
admission, and leaves the file incomplete. A writer-side overflow aborts the
current batch before it advances `committed_samples`. The writer checks for a
latched error immediately before that boundary decision. It releases the error
lock before writing the HDF5 attribute. An AO error after that decision applies
to later batches. These changes bound event memory and preserve the committed
source boundary. They do not explain or fix the paced AO deadline misses below,
and they have no measured latency benefit yet.

An earlier 60-second `f999` diagnostic measured p99 at 2.823 ms with the writer
active and 1.937 ms when metadata-batch processing and HDF5 writes were
discarded while the queue-draining worker remained active. This motivated
metadata batching but does not measure the candidate's performance. The
candidate has data-equivalence evidence only. No timing benefit is claimed.

Treat the timing limits as strict: process p99 must be below 2.5 ms, and the
maximum process time must be below the 5 ms block deadline. The process interval
includes feature extraction, the controller, waveform rendering, safety checks,
recorder submission, and UI submission. Component timings are diagnostic
subsets of that interval. Recorder waits between unpaced blocks are excluded.

For each full replay case, require all 36,000,000 source samples, no source
gaps or skipped samples, no deadline misses, no late or stale AO commands, no
active-demand misses, no noncommitted nonzero AO blocks, complete recording,
consumed UI packets, and a completed drain with an empty queue. Require every
proposed pulse to be classified: `requested_pulse_events` must equal the sum of
`rendered_pulse_events` and `rejected_pulse_events`, and
`unclassified_pulse_events` must be zero.

Report `ao_zero_substitutions` separately. The scheduler may insert zeros for
scheduled transitions or during drain. A zero substitution count of zero is
not an acceptance requirement. Check active-demand misses, late commands,
stale rejections, noncommitted nonzero blocks, and the final drain state as
separate measures.

## Current replay status

The `e965d808` full matrix completed all nine 30-minute cases on October 29,
2026. Each processed 360,000 blocks and 36,000,000 samples. All nine failed the
strict maximum process-time limit. Maxima ranged from 9.959 to 199.559 ms,
against the `<5 ms>` limit. The three ESN cases also failed the p99 limit, at
2.700 to 3.022 ms against `<2.5 ms>`. The exact case results are in
[`validation/replay-e965d80/`](validation/replay-e965d80/).

This matrix predates the reviewed moment-only ESN rebase. It is not timing or
parity evidence for the current bridge. All three current-bridge full-recording
parity cases passed; see
[`validation/esn-full-parity-88469989/`](validation/esn-full-parity-88469989/).
The `bb65477` full nine-case matrix completed at the 100 ms AO lead. All nine
cases processed and recorded 36,000,000 samples. The independent closed-file
audit checked the retained recordings, block and controller rows, headers, and
provenance. It found no structural issues. Every case failed the strict maximum
time gate; all three ESN cases also failed the p99 gate. Rule and PID p99 passed.

| Trace | Controller | p99 (ms) | Maximum (ms) | Deadline misses |
|---|---|---:|---:|---:|
| May 29 | Rule | 2.244601 | 16.7774 | 47 |
| May 29 | PID | 2.176102 | 240.7780 | 54 |
| May 29 | ESN | 2.884603 | 148.2125 | 224 |
| Aug 5 | Rule | 2.185002 | 297.8926 | 41 |
| Aug 5 | PID | 2.012201 | 28.2999 | 44 |
| Aug 5 | ESN | 2.628504 | 447.2756 | 133 |
| Aug 6 | Rule | 2.118502 | 15.6181 | 29 |
| Aug 6 | PID | 2.122004 | 17.1542 | 37 |
| Aug 6 | ESN | 2.770607 | 486.6503 | 168 |

All cases reported zero source gaps and skipped samples, AO active-demand misses,
late commands, and stale rejections. AO drain completed with an empty queue,
and the run consumed all submitted WebSocket packets without a rendered
browser. PID had positive requested-rate blocks but zero pulse proposals,
commits, and nonzero AO samples, so its p99 results do not demonstrate PID
stimulation on these traces. The
detailed case summary and closed-file audit are in
[`validation/replay-bb65477/`](validation/replay-bb65477/). Hardware delivery
was not measured.

The post-calibration lead sweep completed as a 45-second unpaced diagnostic
with virtual AO.
Its independent audit verified all 18 controller/lead cases with no structural
issues and at least 400,000 valid post-calibration feature samples per case.
Eligible source coverage was 28.25 seconds for Rule, 33.99 seconds for PID,
and 44.995 seconds for ESN. Runtime timing gates failed in 15 cases and passed
in 3. Full-duration timing acceptance was not evaluated. See the
[lead-sweep report and audit](validation/lead-sweep-650c4a4/README.md).

## Final required full runs: timing qualification failed

All 12 required full-duration software runs are complete. The nine-case
unpaced replay matrix ran on `bb65477`; the three separate 30-minute paced
runs with an actual browser ran on `c16c0ee`. The combined bundle was assembled
at c16, with per-run source revisions preserved. The matrix is included because
subsequent changes were outside its measured timed-component scope. The combined
result is **REQUIRED FULL RUNS FAILED**. Every case exceeded at least one
strict timing gate (p99 `<2.5 ms`, maximum `<5 ms`). The full metrics and
independent HDF5 audit are in
[`validation/required-runs-c16c0ee/`](validation/required-runs-c16c0ee/README.md).

| Paced controller | Recorded / expected samples | p99 (ms) | Maximum (ms) | Deadline misses | AO active-demand misses | AO late | Browser disconnects |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule | 36,000,000 / 36,000,000 | 1.499801 | 35.1682 | 49 | 0 | 0 | 0 |
| PID | 36,000,000 / 36,000,000 | 1.399002 | 27.5681 | 46 | 0 | 0 | 0 |
| ESN | 5,564,500 / 36,000,000 | 1.312801 | 53.3083 | 17 | 1 | 0 | 0 |

The ESN p99 shown above is an all-call aggregate that includes calls after
acquisition stopped; it does not represent post-fault active-control timing.

The ESN replay processed all 36 million source samples, but its latched AO
active-demand fault stopped acquisition and recording at 5,564,500 samples
(278.225 seconds). Its cleanly closed HDF5 file contains only that prefix.
All paced runs drained the AO queue and remained browser-connected. The
independent closed-file audit found no structural or provenance issues. The
matrix also failed all nine cases: every maximum exceeded 5 ms, and every ESN
p99 exceeded 2.5 ms. PID had positive requested-rate blocks but no pulse
proposals, commits, or nonzero AO samples in the matrix or paced run. Hardware
remains **ALL UNVERIFIED**. Full case details are in the linked report.

The final runtime-input proof checked 69 captured code, test, config, model,
and UI inputs against the 286-test manifest. It found no content mismatch
between tested inputs and main; 30 files differed only in line endings. This
is provenance evidence, not a timing or hardware result.

A paced Rule run and a separate paced PID run on
`09ebf1e` processed all 36,000,000 source samples, but both failed. The Rule
recording contains only 16,608,200 samples. Its fault stopped acquisition and
`RecordBlock` submission near source second 830.41. It had 578 processing
deadline misses, one AO active-demand miss, one browser disconnect, a 1.774801
ms all-call p99, and a 428.0297 ms maximum. The PID recording contains only
480,300 samples. It had 103 deadline misses, one AO active-demand miss, no
browser disconnect, a 0.9009 ms all-call p99, and a 66.7837 ms maximum.

Both files closed cleanly with `complete=1`, but each is only the prefix recorded
before the fault. `complete=1` means the writer closed that file cleanly. It
does not mean the file contains all 36,000,000 samples. The fault stops
acquisition, and `ProcessingCore` stops submitting `RecordBlock` values while
the finite replay loop continues through its remaining source data. The all-call
p99 values include inactive calls after acquisition stopped. They do not show
active controller timing after the fault.

The original combined PID and ESN startup attempts timed out after 120 seconds
while waiting for browser connections, before producing case metrics. They are
not completed runs. The separate PID attempt completed and failed. These Rule
and PID results are historical pre-fix evidence from `09ebf1e`.

The `6657c88` pre-progress-fix paced ESN attempt faulted near source second
1,080 and stopped acquisition and recording at 21,606,700 samples. The replay
loop continued afterward and processed the remaining input, reaching
36,000,000 source samples. It had 237 processing deadline misses, p99 2.1158 ms,
and maximum 38.7395 ms. The browser remained connected, and AO drain completed,
but full recording and runtime timing gates failed. These numeric Core and
recorder results apply only to revision `6657c88`; final full-runtime runs are
reported above. The compact report is in
[`validation/paced-esn-6657c88/`](validation/paced-esn-6657c88/). The
[historical Rule attempt](validation/paced-browser-09ebf1e-attempt1/summary.json)
and [historical PID attempt](validation/paced-pid-09ebf1e/controller_paced_20261029T125918Z.json)
retain the earlier reports. Those runs predate the fault-event recorder change.

A 32-second diagnostic on `bb65477` processed and recorded 640,000 samples for
each controller. Rule p99/max were 2.190158/4.2422 ms with zero deadline
misses. PID measured 2.439674/5.7394 ms with one miss. ESN measured
2.776818/5.401 ms with one miss. These short cases are diagnostics, not timing
acceptance. The full nine-case matrix later completed and failed timing
acceptance.
The figures are summarized in
[`validation/reporting-prefix-bb65477/summary.json`](validation/reporting-prefix-bb65477/summary.json).

For the post-calibration lead sweep, use 45 seconds per controller and lead.
Thirty-two seconds is a nominal floor based on an expected 11-second calibration
and settling period; it does not guarantee 20 seconds of eligible data. Verify
at least 400,000 eligible post-calibration samples and contiguous HDF5 source
coverage in each case. A readiness banner or elapsed duration does not prove
post-calibration coverage.

The default six-segment simulation on `2a3a1fb` wrote JSON before CSV export
failed on an unavailable ESN post-settling tracking field. The audit found five
full recordings and a PID no-stimulation recording truncated at 273,100 of
480,000 samples after a latched simulated-AO fault (one active-demand miss, ten
late commands, 211.9717 ms maximum). These are separate post-run reporting and
runtime failures. The retry wrote all three CSV rows but faulted in the ESN
stimulated segment after 136,800 of 480,000 samples; the other five segments
recorded fully. All controller-level simulation acceptance gates remained
false. The reporting fix is post-run only and does not change the processing
path measured in the prior matrix or lead sweep. It does not explain or resolve
either run's AO fault. See the [first audit](validation/default-simulation-2a3a1fb/README.md)
and [post-fix retry audit](validation/default-simulation-ef81a23/README.md).

The benchmark keeps a bounded history of the 32 most recent Core calls for
fault diagnostics. Each row separates call start from completion. The snapshot
reports the latest completed source index separately from the latest-started
and active source indices. The AO-clock observer captures the history after it
sees a fault. This observation is not the fault onset. History capture occurs
outside the measured Core interval, though it adds harness work that may affect
later scheduling. These diagnostics do not change the strict p99 `<2.5 ms` or
maximum `<5 ms` limits.

The historical
`6cc4035` matrix is available in
[`validation/baseline-6cc4035/`](validation/baseline-6cc4035/); its maxima were
7.095 to 34.104 ms, and its ESN p99 values were 4.745 to 4.830 ms. See
[implementation and validation status](../IMPLEMENTATION_STATUS.md) before
making a release claim.

## Hardware status

Hardware status is **ALL UNVERIFIED**. Software checks and the synthetic plant
do not establish NI timing, physical output, charge, or biological efficacy.
Keep AO disconnected from an isolator or preparation until a lab operator
completes the staged checks and four-hour soak in [the NI protocol](ni_controller_protocol.md).
