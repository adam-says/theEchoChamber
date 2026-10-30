# ESN paced AO-fault evidence package

This package preserves a bounded investigation of the failed 30-minute ESN paced run. The run used source revision `c16c0eee0174c7c016d05a5ce617186a3c4c42d8`; the saved recording and inputs remain in the task workspace and are not copied here.

## Files

- [Analysis](README-analysis.md) explains the observations and their limits.
- [Bounded extraction JSON](esn-paced-fault-analysis-c16c0ee-v2.json) contains the source-indexed timeline, H5 metadata/diagnostics/action-stage summaries, UI timing cross-check, and comparisons.
- [Extractor snapshot](extractor/analyze_paced_fault_timeline.py) is the Python script used to produce bounded timeline summaries.
- [.gitattributes](.gitattributes) disables text conversion in this package so the extractor snapshot bytes remain stable.
- [SHA-256 manifest](SHA256SUMS.txt) records the packaged analysis, JSON, and extractor bytes.

## Reproduce from the original task workspace

Run this from the project task root. The command writes a new JSON file and refuses to overwrite an existing one:

```powershell
& .\work\venv314\Scripts\python.exe `
  .\work\control-validation-fixes\docs\validation\esn-fault-investigation-c16c0ee\extractor\analyze_paced_fault_timeline.py `
  --case `
    .\work\theEchoChamber\work\validation_runs\acceptance-c16c0ee\paced-esn-30min\controller_paced_20261029T225815Z.json `
    .\outputs\paced-esn-c16c0ee-live-fault.json `
  --case `
    .\work\theEchoChamber\work\validation_runs\acceptance-6657c88\paced-esn-30min\controller_paced_20261029T144840Z.json `
    .\outputs\paced-esn-6657c88-live-fault.json `
  --simulation-report `
    .\work\theEchoChamber\work\validation_runs\acceptance-2a3a1fb\default-simulations\responsive_simulation_20261029T164720Z.json `
  --simulation-report `
    .\work\theEchoChamber\work\validation_runs\acceptance-ef81a23\default-simulations\responsive_simulation_20261029T165702Z.json `
  --window-samples 20000 `
  --output .\work\esn-paced-fault-reproduction.json
```

Each input case report and its `progress.jsonl` must remain together in the original case directory because the extractor reads the adjacent progress log. Each fault snapshot is at the path above. The reports name the recording H5 files. The extractor reads H5 metadata, block/controller diagnostic rows, source-discontinuity count, and sparse AO action-stage rows in chunks. It does **not** read raw AI/electrode signal payloads or hash the full H5 file. Its simulation summaries read compact JSON reports only.

Dependencies are Python 3.14, `h5py`, and the Python standard library. The packaged snapshot was captured from `work/analyze_paced_fault_timeline.py`; the SHA manifest identifies the exact archived bytes. This is a rerun of the bounded analysis against retained local artifacts, not a full replay or a hardware measurement.

## Scope and interpretation

The 30-minute case processed the full 36 million source samples, but its closed H5 contains a clean 5,564,500-sample prefix after a simulated AO active-demand safety fault. “Complete” means the file closed cleanly. It does not mean the requested recording duration was captured. `physical_pulse_claims` is false.

The fault snapshot is explicitly a post-advance observation rather than an atomic fault-onset record. Its counters were sampled sequentially, so they do not form one simultaneous queue state. The UI cross-check is conditional: the bounded slow-call log contains all 17 deadline events and remains below its 20-event cap; the only Core call at least as long as the 52.1041 ms UI-submission maximum is the same-index 53.3083 ms call at source 5,564,400. Since the measured UI wrapper is synchronous within Core processing, this identifies a timing overlap under those premises. It does not identify the delay’s cause. In particular, it does not establish a 52 ms WebSocket wait.

The scheduler recorded a missing active-generation command at its safety refill point and replaced output with a safe zero range. This is evidence of the software safety fault, not proof of an underrun at physical hardware. The 5 ms processing interval is diagnostic per the 2026-10-30 direction. This analysis does not justify changing the 100 ms output lead, 50 ms AO reserve, or safety behavior. Runtime’s deterministic boundary reproduction and source-indexed phase probe are still in preparation; no result is included here.

Historical run classifications are preserved as reported. The older 6657c888 snapshot is also post-advance, its slow-call log reached its 20-event cap despite 237 deadline misses, and its cause cannot be inferred from this package.
