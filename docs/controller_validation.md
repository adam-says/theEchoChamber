# Validate the controller paths

Use these commands to measure controller processing against the canonical CA3 recordings or the deterministic synthetic plant. Every benchmark uses `AsyncUiProcessingCore`, the shared pulse actuator, the safety adapter, `H5Recorder`, the UI serializer, and `SimulatedAoScheduler`. The simulator commits only queued commands at their scheduled AO sample indices. It does not claim that an output reached hardware.

## Run a short core smoke

Run the nine trace and controller combinations for one second each:

```powershell
.\.venv\Scripts\python.exe tools\validate_controllers.py --smoke
```

Add `--trace PATH` once per trace to choose other replay files. Add `--controller rule --controller pid` to select controller paths. The smoke mode processes real canonical input through the core. It does not shorten a block or skip the scheduler.

## Run the finite replay matrix

Process each of the three full, finite 30-minute recordings once with each controller:

```powershell
.\.venv\Scripts\python.exe tools\validate_controllers.py --matrix
```

The command does not wrap at end of file. It stops source input, disarms output, and advances the shared simulated AO clock to record the zero drain. Use `--duration-seconds 60` for a shorter run. That option reads only the requested prefix.

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

For each controller the progress JSONL emits a `browser_required` record with the current WebSocket port and full UI URL. Open that URL in the browser while the run waits; the run will not begin until the real UI page connects as a second WebSocket client. `--browser-ui` is available when Python should open the page automatically. Use `--paced-seconds 10` for a pacing smoke. This still waits for real elapsed time and requires the browser page to connect. Use `--keep-recordings` to retain the HDF5 files. By default, temporary recordings are deleted after the summaries are written.

For a controlled producer-overrun check, inject one 150 ms stall at 12 seconds of source time. The AO scheduler keeps its independent clock; a missed active demand should fault and substitute zeros rather than commit the late command:

```powershell
.\.venv\Scripts\python.exe tools\validate_controllers.py --smoke --trace PATH_TO_TRACE.h5 --controller pid --duration-seconds 20 --inject-stall-ms 150 --stall-at-second 12
```

The simulator records the stall, active-demand misses, fault, pulse request IDs, and committed pulse IDs. A request that misses its target must not alter the synthetic plant.

## Compare the responsive synthetic plant

Run every controller once with stimulation enabled and once with stimulation disabled:

```powershell
.\.venv\Scripts\python.exe tools\validate_controllers.py --simulation
```

The deterministic plant generates a 150 Hz CA3 carrier, a disturbance, and noise. It applies a delayed amplitude reduction only after `SimulatedAoScheduler` generates a non-zero AO command. The plant is a software fixture. It is not a tissue model and does not establish biological efficacy.

The default plant has a 0.18 mV disturbance but allows at most 0.10 mV reduction. It is an explicit actuator-saturation scenario; it cannot reach the original signal amplitude even at maximum output. Use a separate PID tracking case to test target regulation with an attainable disturbance:

```powershell
.\.venv\Scripts\python.exe tools\validate_controllers.py --tracking-simulation
```

That case uses the documented generic plant settings in the JSON report: 0.010 mV disturbance, 20 ms effect delay, 0.5 s recovery, 0.002 mV reduction per generated pulse, and 0.020 mV maximum reduction. Its PID demo preset (`kp=0.03`, `ki=0.001`, `kd=0`, `target_z=1`) is separate from hardware gains. The default 40-second trace contains an 8-second disturbance after baseline calibration, followed by recovery. The report includes z-score target, IAE, disturbance-window mean error, initial/final error, mean and variation of requested rate, saturation fraction, rate-direction reversals, and settling time using a one-second average band derived from the observed disturbance error. It also records target-window coverage and marks tracking metrics incomplete if the window has less than 95% valid controller observations, so a runtime fault cannot look like an improvement by shortening the trace. Both enabled and disabled results are compared; none of these measures indicate biological efficacy.

## Read the result files

Each run writes a JSON summary and a CSV row per case under `work/validation_runs`. The summary includes the Python and OS versions, Git revision, source checksum, controller settings, AO lead, block deadline, median, p95, p99, maximum processing time, thread and process CPU timing, GC pause timing, inclusive component timings, deadline misses, source availability lag and index gaps, recorder backlog and completion, UI packet counts, AO clock mode and service-gap counts, scheduler queue health, requested actions, scheduler commits, zero substitutions, late commands, stale rejections, cancellations, runtime fault, and AO drain-barrier completion. Slow core calls also produce a bounded set of progress records correlating wall time, thread CPU, process CPU, GC pauses, and source index. Component timings are diagnostic measurements inside the same `AsyncUiProcessingCore.process` interval; they are not subtracted from the block time. The default runner starts the application's WebSocket handler and a live client that parses every UI packet. The result names this transport and marks browser rendering as false. `--browser-ui` opens the page automatically; `--external-browser` waits for the real page to connect to the emitted URL.

`progress.jsonl` contains a process ID, start and completion records, and periodic case progress. Keep the process ID and log path when a long run needs to be monitored. Processing latency measures the synchronous `AsyncUiProcessingCore.process` path, including feature extraction, controller, rendering, safety, recorder submission, and UI submission. The scheduler commits each AO block separately and reports its queue health.

`pulse_commit_samples` and `ao_nonzero_generated_samples` describe commands accepted by the software AO simulator. They are not physical pulse measurements. The JSON field `physical_pulse_claims` is always `false`.

## Acceptance checks

Treat the reference timing targets as evidence to inspect, not as an automatic guarantee: p99 processing time below 2.5 ms and maximum below the 5 ms block deadline on the measured workstation. Check for zero unexplained late commands, stale commands, or queue growth. Confirm every requested pulse event is categorized as rendered or rejected, and every generated non-zero command has an `ao_committed` scheduler stage.

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
