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

The final software pair passed these gates. The reference generated 25 pulse
effects from seconds 22 to 42, at 1.25 Hz. It measured a 0.003076 mV mean reduction
against a 0.002616 mV target correction, for a 0.000460 mV margin. The regulated
PID generated 32 effects during the disturbance, at 1.067 Hz. Its formula-based
usage estimate was 0.002667 mV. Tracking settled 14.16 seconds after the
disturbance began. Final moving-mean absolute error was 0.0532 z. Full-window
raw error SD was 1.9045 z, and post-settling raw error SD was 1.0168 z. All
4,672 eligible observations were present, and the source had no gaps or skipped
samples. This result is software-only evidence, not a hardware or biological
claim. The JSON and CSV are under
`work/validation_runs/tracking-kp02-ki006-capacity-reference/`.

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
use the zero-gain hardware PID defaults. In the current full replay results,
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

To inspect controller activity after calibration, pace a 20-second run at each
lead and controller. Start the local UI server, then pass one trace explicitly:

```powershell
python -m http.server 8000 --directory .
.\.venv\Scripts\python.exe tools\validate_controllers.py --paced --external-browser `
  --browser-base-url http://127.0.0.1:8000 `
  --trace recordings\ca3_slow_waves\2026_08_06_0000.h5 `
  --lead-sweep --duration-seconds 20 --keep-recordings
```

The 20-second duration leaves about nine seconds after the 11-second filter
settling and baseline calibration. This short paced sweep checks scheduler
behavior after readiness. It does not pass the 30-minute acceptance gate.

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
documents exact trigger-index parity and float32 reconstruction on all three
30-minute ESN recordings for the earlier `f6147a59` bridge. The current bridge
has SHA-256
`88469989b45429ec270f55e37139c96a1565ae91193dd2113c7b332d4e9c4749` and includes
the reviewed moment-only rebase. That rebase recalculates count, origin, mean,
and M2 while preserving the live peak queue and absolute sequence. Contraction
and drift callers reset scale high-water to the current live peak. Window resize
still rebuilds peak metadata. Full three-recording parity on the current bridge
is pending. Neither parity result establishes real-time timing or hardware
acceptance.

The full current Python test suite passes with 240 tests and 15 subtests on both
Python 3.12.14 and Python 3.14.0. The 3.12 run took 53.27 seconds; the 3.14 run
took 47.75 seconds. It includes controller readiness transactions, AO range
attribution, unresolved-write retention, shutdown outcome flushing, and the
moment-only ESN rebase. Run `python -m pytest -q` with each interpreter to
reproduce these counts. These tests do not establish full replay timing or
hardware acceptance.

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
parity evidence for the current bridge. Full replay timing and three paced
30-minute runs on the current implementation remain pending. The historical
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
