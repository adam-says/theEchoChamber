# ESN AO-miss diagnostic probes (c2e4119)

## Purpose

These bounded diagnostics compare one 360 source-second ESN replay with and without an external browser connection, and separately verify the simulated AO scheduler’s configured missing-target guard boundary. They investigate the earlier paced-run active-demand miss. They do not change the application, controller, model artifact, safety limits, scheduler limits, or acceptance gates.

## Results

Both ESN probes processed and recorded 7,200,000 samples (360 source seconds at 20 kHz). Both completed their AO drain with zero active-demand misses, zero late commands, and no runtime fault. The first run used the internal WebSocket consumer. The second run kept an external browser connection active through completion and recorded no disconnect. Rendering of browser frames was not verified. Neither run reproduced the earlier miss.

The bounded phase summaries and exact hash provenance are in [comparison-summary.json](comparison-summary.json). The full phase-ring reports remain in the task’s ignored `work/diagnostics/esn-first-fault-c2e4119/` and `work/diagnostics/esn-first-fault-browser-c2e4119/` directories; this bundle keeps curated summaries instead of copying the multi-megabyte full reports.

| Run | Browser connected at finish | Core p99 / max | Calls over 5 ms | Scheduler active misses / late commands | Recorded samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1, internal WebSocket consumer | No browser requested | 2.340 / 28.648 ms | 56 | 0 / 0 | 7,200,000 |
| V2, external browser connected | Yes; zero disconnects | 2.155 / 37.479 ms | 49 | 0 / 0 | 7,200,000 |

These Core timings include the probe wrappers. The selected phase samples are not a complete distribution of all 72,000 Core calls.

The deterministic scheduler probe uses 20 kHz samples, 100-sample chunks, and 100 ms AO lead. Its derived safe-command deadline is 1,000 samples, or 50 ms. The scheduler evaluates remaining committed queue headroom at the write frontier; the first unavailable target index is recorded as context. With the producer held back and simulated AO clock advancing, the fixture reports no miss with 55 ms of headroom, then records an active-demand miss and substitutes zeros at 50 ms. This verifies the scheduler condition only. It does not reproduce a physical hardware underrun or explain the paced-run miss.

## How to reproduce

Use the isolated checkout at the recorded revision `c2e41199191d42b55298baaf46e9c868576d1b2b`, Python 3.14, Git on `PATH`, and the project’s installed runtime dependencies (`numpy`, `scipy`, `h5py`, `numba`, and `websockets`). Place the frozen probe snapshots from this directory in the checkout’s `work/` directory. The probes resolve the repository root from that location. The replay H5 and `tmp` directory are in the sibling `work/` directory above the checkout, so use `..\tmp\...` paths from the checkout root. Use a fresh output directory for every run.

From the isolated checkout root, run the no-browser diagnostic:

```powershell
$env:TEMP = (Resolve-Path ..\tmp).Path
$env:TMP = $env:TEMP
python work/probe_esn_ao_fault_v1.py --trace ..\tmp\ca3_import_validation\2026_05_29_0001.h5 --output-dir work/diagnostics/esn-fault-v1-repro --duration-seconds 360 --lead-ms 100
```

For the browser-connected diagnostic, start a static server from the checkout root in a second terminal and leave it running through the probe:

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

Start the probe in the first terminal. Connect the browser to the printed URL and WebSocket port within the 120-second startup window. Keep the browser and static server running until the probe finishes. Stop the static server afterward.

```powershell
$env:TEMP = (Resolve-Path ..\tmp).Path
$env:TMP = $env:TEMP
python work/probe_esn_ao_fault_browser_v2.py --trace ..\tmp\ca3_import_validation\2026_05_29_0001.h5 --output-dir work/diagnostics/esn-fault-v2-repro --duration-seconds 360 --lead-ms 100 --external-browser
```

Run the deterministic boundary check in a scratch checkout. It needs no replay or browser and writes directly to `work/diagnostics/ao-deadline-boundary.json`. Do not redirect its console output to a JSON file; stdout contains both the output path and a short summary, not the full result. Preserve any existing result before rerunning.

```powershell
python work/probe_ao_deadline_boundary.py
```

The archived scripts are the exact executed versions. Their SHA-256 values are in [snapshot-hashes.json](snapshot-hashes.json). V1 is also preserved at `work/diagnostics/esn-fault-probes/probe_esn_ao_fault_v1.py`; its archived hash was checked against the executed copy. The output report records code, source, and artifact hashes before and after each run.

## Limits

The probes wrap per-call phases with wall and thread-CPU timers. Those wrappers can change scheduling. The phase ring has a fixed 50,000-row capacity and captures selected near-deadline activity, so the phase percentiles describe those retained rows, not all calls. The summary reports the retained-row count for each phase.

Phase wall spans are nested and can include OS or thread scheduling delays. Do not add them as CPU time. A small or quantized thread-CPU delta does not prove that a thread was blocked. The queue-lock timer covers mutex wait and hold intervals, but a long wait can include a thread that the OS did not schedule. The observer does not prove which operation caused a delay.

The 5 ms boundary is reported as diagnostic context; these runs are not acceptance results. AO records are software commands. No physical AO delivery, electrical response, or latency was measured. The probes do not establish regulation of the CA3 ripple-band RMS biological measure.
