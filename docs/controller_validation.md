# Validate the controller paths

Use these commands to measure controller processing against the canonical CA3 recordings or the deterministic synthetic plant. Every benchmark uses `AsyncUiProcessingCore`, the shared pulse actuator, the safety adapter, `H5Recorder`, the UI serializer, and `SimulatedAoScheduler`. The simulator commits only queued commands at their scheduled AO sample indices. It does not claim that an output reached hardware.

## Run a short core smoke

Run the nine trace and controller combinations for one second each:

```powershell
.\.venv\Scripts\python.exe tools\validate_controllers.py --smoke
```

By default, the command reads the three canonical files from `recordings/ca3_slow_waves/`. Import them first as described in [the replay import guide](replay_import.md). If a default file is missing, the command reports its path and stops. Add `--trace PATH` once per trace to choose other replay files. Add `--controller rule --controller pid` to select controller paths. The smoke mode processes real canonical input through the core. It does not shorten a block or skip the scheduler.

## Run the finite replay matrix

Process each of the three full, finite 30-minute recordings once with each controller:

```powershell
.\.venv\Scripts\python.exe tools\validate_controllers.py --matrix
```

The command does not wrap at end of file. It stops source input, disarms output, and advances the shared simulated AO clock to record the zero drain. Use `--duration-seconds 60` for a shorter run. That option reads only the requested prefix.

Pass three `--trace PATH` options to use replay files outside the default directory. Each full canonical input must contain 36,000,000 samples. The matrix rejects missing or invalid files instead of skipping a case.

The unpaced producer pauses between blocks when the bounded HDF5 queue reaches 80% of capacity. It waits for the writer to drain before processing another block. The wait does not advance the virtual AO clock or change source indices, and it stays outside `AsyncUiProcessingCore.process` timing. A 30-second wait with no queue progress fails the case with queue diagnostics. JSON and CSV report the wait count, duration, queue limit, and policy. Paced runs never use this wait policy. A paced recorder overflow remains a failed case.

## Sweep AO target lead

Use the default leads of 100, 75, 50, 40, 30, and 20 ms:

```powershell
.\.venv\Scripts\python.exe tools\validate_controllers.py --smoke --lead-sweep
```

For a full finite sweep, replace `--smoke` with `--matrix`. The 100 ms lead is the reference setting. The command reports every lead as a separate result and never lowers a hardware setting automatically.

## Run the paced controller soak

Run each controller for the finite 30-minute source duration at the 20 kHz, 100-sample, 5 ms block rate. A paced run longer than its finite source is rejected; the tool never wraps the source implicitly. The paced producer waits until each input block's original five-millisecond acquisition interval has completed, then processes every source block in order. If processing falls behind, it catches up from the buffered source and reports availability lag; it does not seek forward. The recorder, UI serializer, WebSocket client, and browser page stay active:

```powershell
python -m http.server 8000 --directory .
.\.venv\Scripts\python.exe tools\validate_controllers.py --paced --external-browser --browser-base-url http://127.0.0.1:8000
```

For each controller the progress JSONL emits a `browser_required` record with the current WebSocket port and full UI URL. Open that URL in the browser while the run waits; the run will not begin until the UI connects as an external WebSocket client. The runner tracks the external connection through the run and fails acceptance if it drops, even if it reconnects. Reports include disconnect count, disconnected duration, and presence at completion. The WebSocket check does not verify pixels or browser rendering. `--browser-ui` is available when Python should open the page automatically. Use `--paced-seconds 10` for a pacing smoke. This still waits for real elapsed time and requires the browser page to connect. Use `--keep-recordings` to retain the HDF5 files. By default, temporary recordings are deleted after the summaries are written.

For a controlled producer-overrun check, inject one 150 ms stall at 12 seconds of source time. The AO scheduler keeps its independent clock; a missed active demand should fault and substitute zeros rather than commit the late command:

```powershell
.\.venv\Scripts\python.exe tools\validate_controllers.py --smoke --trace PATH_TO_TRACE.h5 --controller pid --duration-seconds 20 --inject-stall-ms 150 --stall-at-second 12
```

The simulator records the stall, active-demand misses, fault, pulse request IDs, and committed pulse IDs. A request that misses its target must not alter the synthetic plant.

On a fault, the scheduler cancels commands that have not reached the AO writer. Samples already accepted into the NI output buffer may still play at their scheduled indices. The simulator models that residual output. Keep the scope or dummy load connected until the committed output drains.

## Compare the responsive synthetic plant

Run every controller once with stimulation enabled and once with stimulation disabled:

```powershell
.\.venv\Scripts\python.exe tools\validate_controllers.py --simulation
```

The deterministic plant generates a 150 Hz CA3 carrier, a disturbance, and noise. It applies a delayed amplitude reduction only after `SimulatedAoScheduler` generates a non-zero AO command. The plant is a software fixture. It is not a tissue model and does not establish biological efficacy.

The default plant has a 0.18 mV disturbance but allows at most 0.10 mV reduction. It is an explicit actuator-saturation scenario; it cannot reach the original signal amplitude even at maximum output. Use the separate PID tracking case to measure target regulation and its rate-limited plant capacity:

```powershell
.\.venv\Scripts\python.exe tools\validate_controllers.py --tracking-simulation
```

The tracking trial uses a 0.003 mV disturbance, 20 ms effect delay, 5 s exponential recovery, 0.0005 mV reduction per generated pulse, and a 0.020 mV reduction cap. The configured 2 Hz controller limit would imply 0.005 mV theoretical mean capacity, but the software recovery guard and pulse timing constrain the ideal cadence to about 1.27 Hz (`1 / (100 ms lead + 5 ms pulse + 181.9 ms feature guard + 500 ms refractory)`), or roughly 0.00318 mV mean capacity. The tracking command runs an additional PID reference through the actual `ProcessingCore` and AO scheduler, with `kp=1`, `ki=0`, `kd=0`, `target_z=-1000`, and the same 2 Hz cap. It measures actual modeled reduction from 22–42 s, after two plant recovery constants, and compares it with the target correction estimated from the frozen baseline. The requested-rate formula, guard-limited formula, and measured reference reduction are separate values. Regulated-controller pulse usage is reported separately and is not used as the plant capacity ceiling. The current synthetic tracking preset is `kp=0.2`, `ki=0.06`, `kd=0`, `target_z=1`; hardware PID gains remain zero.

The default 64-second source has a 30-second disturbance from 12 to 42 seconds, then 22 seconds of recovery. Tracking acceptance uses a fixed ±0.5 z band on one-second moving means over the full 5 ms source grid, requires at least 95% eligible target-window coverage, and requires every complete moving window after settling to stay inside the band through the disturbance end. Stimulated and post-mask guard samples count as feature-ineligible only when the controller explicitly reports them. The report keeps full source coverage and source gaps/skips separate; known feature masks never excuse a source gap, runtime fault, or unexplained invalid feature. It also limits raw target-error ripple to 2 z standard deviations over the target window. The report includes post-settling raw ripple as a diagnostic; it does not replace the fixed full-window ripple gate. The final moving-mean error, raw absolute error and IAE, ripple, and full controller trace are retained. The pair also requires lower IAE than the no-stimulation control. Recovery is measured separately from the last half-second of the disturbance through the final output window; the report includes residual effect and requested output after two recovery constants. The case reports `tracking_gate_passed`, `plant_recovery.bounded_recovery`, and `simulation_acceptance_passed`. Capacity alone does not pass tracking; the command exits nonzero if measured tracking or recovery gates fail. Results are synthetic software measurements, not biological efficacy.

The feature-aware 64-second PID pair with `kp=0.05` processed all 1,280,000 source samples with no gap or skip. It marked 1,225 target-window blocks (119,616 samples) as explicit stimulation or feature-guard masks; all 4,675 expected eligible blocks were observed. The simulator generated 32 pulse effects during the 30-second disturbance (1.067 Hz); the controller's formula-based usage estimate was 0.002667 mV against a 0.002616 mV estimated target correction. This is usage evidence, not a capacity reference. The one-second moving mean settled 28.51 s after the disturbance began (source time 40.51 s); the final moving-mean error was −0.367 z. IAE improved over the disabled control and recovery passed. The tracking gate failed because full-window raw error SD was 2.455 z, above its fixed 2.0 limit. Post-settling raw error SD was 1.005 z, retained as context only. The result is in `work/validation_runs/feature-aware-tracking-64s-final/result.json` with a compact `result.csv`.

One theory-led trial with `kp=0.2`, `ki=0.05` and the same plant/preset limits passed the tracking gate. Its regulated run had a formula-based usage estimate of 0.002583 mV against the 0.002616 mV correction and requested output after two recovery constants of 0.10018 Hz, just over the fixed 0.10 Hz limit. The independent max-request capacity reference had not yet been added, so this run does not establish capacity. Its result is in `work/validation_runs/tracking-kp02-64s/`.

The final `kp=0.2`, `ki=0.06` pair passed software tracking, max-request capacity, and recovery gates. Its capacity reference generated 25 pulse effects in the 22–42 s window (1.25 Hz) and measured 0.003076 mV mean reduction against the 0.002616 mV estimated target correction (0.000460 mV margin). The regulated PID run generated 32 pulse effects during the 12–42 s disturbance (1.067 Hz); this is reported as usage, not capacity. Tracking settled 14.16 s after disturbance start, final moving-mean absolute error was 0.053 z, whole-window raw error SD was 1.904 z, and all 4,672 eligible observations were present. Recovery passed; source coverage was 100% with no gaps or skips. The result is in `work/validation_runs/tracking-kp02-ki006-capacity-reference/result.json` and `.csv`. It remains a software-only synthetic result with no physical pulse claim.

## Read the result files

Each run writes a JSON summary and a CSV row per case under `work/validation_runs`. The summary includes the Python and OS versions, Git revision, source checksum, controller settings, AO lead, block deadline, median, p95, p99, maximum processing time, thread and process CPU timing, GC pause timing, inclusive component timings, deadline misses, source availability lag and index gaps, recorder backlog and completion, UI packet counts, AO clock mode and service-gap counts, scheduler queue health, requested actions, scheduler commits, zero substitutions, late commands, stale rejections, cancellations, runtime fault, and AO drain-barrier completion. Slow core calls also produce a bounded set of progress records correlating wall time, thread CPU, process CPU, GC pauses, and source index. Component timings are diagnostic measurements inside the same `AsyncUiProcessingCore.process` interval; they are not subtracted from the block time. The default runner starts the application's WebSocket handler and a live client that parses every UI packet. The result names this transport and marks browser rendering as false. `--browser-ui` opens the page automatically; `--external-browser` waits for the real page to connect to the emitted URL.

`progress.jsonl` contains a process ID, start and completion records, and periodic case progress. Keep the process ID and log path when a long run needs to be monitored. Processing latency measures the synchronous `AsyncUiProcessingCore.process` path, including feature extraction, controller, rendering, safety, recorder submission, and UI submission. The scheduler commits each AO block separately and reports its queue health.

`pulse_commit_samples` and `ao_nonzero_generated_samples` describe commands accepted by the software AO simulator. They are not physical pulse measurements. The JSON field `physical_pulse_claims` is always `false`.

## Acceptance checks

Treat the reference timing targets as evidence to inspect, not as an automatic guarantee: p99 processing time below 2.5 ms and maximum below the 5 ms block deadline on the measured workstation. Check for zero unexplained late commands, stale commands, or queue growth. Confirm every requested pulse event is categorized as rendered or rejected, and every generated non-zero command has an `ao_committed` scheduler stage.

If a case raises an exception, the command writes its exception type and message to the JSON `case_failures` array and to a failed CSV row. The matrix then exits with status 1. Completed cases retain their individual timing gates.

Do not call a simulated run a hardware pass. Follow [the NI staged protocol](ni_controller_protocol.md) before connecting AO to the isolator or preparation.

## Evidence checklist

Record the run directory and mark each item `UNVERIFIED`, `PASS`, or `FAIL`. A passing unit-test suite does not mark any hardware or long-duration acceptance item as passed.

| Requirement | Evidence to retain | Status |
| --- | --- | --- |
| Canonical recordings are finite, 20 kHz, 36,000,000-sample files with CA3/Cortex roles | Source path and SHA-256 in the JSON summary; import-verification result for every supplied canonical file | UNVERIFIED per run |
| All three controller implementations use the real application processing core | Nine-case finite matrix; JSON and CSV; application Git revision | UNVERIFIED |
| AO commands use the shared target-index scheduler and configured lead | AO reference and lead sweep reports; scheduler queue and late/substitution counters | UNVERIFIED |
| Stalls cannot make a missed command appear committed | Injected-stall report with fault, demand-miss, request-ID, and commit-ID fields | UNVERIFIED |
| Recorder, UI transport, and UI packet parsing stay active | HDF5 completion attributes and progress/JSON metrics; browser connection/rendering fields | UNVERIFIED per run |
| Process timing meets the stated workstation targets | Median/p95/p99/max and missed-deadline count under the measured run configuration | UNVERIFIED per machine |
| Responsive synthetic feedback changes generated activity | Paired stimulation-enabled/disabled reports and generated AO scheduler stages | UNVERIFIED per controller |
| NI output and input checks pass at the lab rig | Signed stage forms, scope captures, calibrated channel/gain values, and recordings | UNVERIFIED; no hardware run performed |
| Four-hour on-device soak has no unexplained AO underruns or recording gaps | Application log, scheduler counters, recorder gap report, and operator review | UNVERIFIED; requires physical lab execution |

Reproduce the software sequence in this order: `--smoke`, `--simulation`, `--matrix`, `--lead-sweep --smoke`, then `--paced --external-browser` (or `--paced --browser-ui`). Matrix and paced runs write JSON/CSV evidence first, then return exit status 1 if any case misses a software acceptance limit. The short smoke is diagnostic and does not override the release gate. Run the longer software soak with `--paced-seconds 1800`. The staged NI sequence and the separate four-hour hardware command are in [the NI protocol](ni_controller_protocol.md); they require a lab operator and are not implied by these software results.
