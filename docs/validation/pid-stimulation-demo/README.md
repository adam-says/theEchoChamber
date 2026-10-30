# Synthetic PID stimulation demonstration

This bundle records a 64-second software-only run of the real `ProcessingCore`, safety checks, simulated AO scheduler, and H5 recorder. It demonstrates that the PID path can produce and commit safe pulse commands and that the delayed synthetic plant responds to them. The plant is a deterministic 150 Hz CA3 carrier with noise and a preset amplitude disturbance. It is a generic software plant, not a tissue model.

## Evidence

The run used revision `4e9cb5b8e375a6b4fb56f9e9d68fc4c0aec7f892`, Python 3.14.0, 20 kHz sampling, 100-sample blocks, and a 100 ms AO lead. PID settings were `kp=0.2`, `ki=0.06`, `kd=0`, target `z=1`, and a 2 Hz rate limit. The synthetic plant used seed 1414, a 0.003 mV disturbance from 12 to 42 seconds, a 20 ms effect delay, 5 second recovery, and a 0.0005 mV amplitude reduction per generated pulse.

The stimulated phase recorded all 1,280,000 source samples and closed with the AO queue drained. It produced 32 unique pulse proposals, each matched to a committed AO range, for 3,200 committed pulse samples and 3,168 nonzero generated command samples. There were no source gaps, runtime faults, AO active-demand misses, or late commands. The separate no-stimulation control also recorded all 1,280,000 samples, drained cleanly, and had zero pulse proposals or nonzero generated command samples.

The report’s disturbance reduction was 0.001626696841 mV. Recomputing the disturbance comparison from the saved H5 data gave 0.001626696840 mV. The tracking and recovery criteria passed: the 1-second moving-mean error settled within the predefined ±0.5 z band, raw error standard deviation stayed below 2 z, and the recovery residual was 1.29% of the pre-recovery reduction. The 67-pulse forced-capacity reference is reported separately; it measures available plant capacity and is not regulated PID output.

Processing time is diagnostic evidence, not a real-time acceptance result. The three phases had p99 values of 1.417, 1.404, and 1.900 ms; maxima of 6.783, 12.308, and 11.785 ms; and 3, 2, and 4 blocks over the 5 ms deadline. No physical pulse delivery or biological efficacy is established.

## Files

- [`audit.json`](audit.json): report and H5 consistency checks, pulse IDs, ranges, coverage, gates, and timing diagnostics.
- [`tracking-series.csv`](tracking-series.csv): curated per-block plant activity and tracking traces for the stimulated and control phases.
- [`pulse-events.csv`](pulse-events.csv): pulse source and AO target ranges for the stimulated and capacity-reference phases.
- [`pid-stimulation-demo.png`](pid-stimulation-demo.png): activity, controller tracking, and AO pulse timeline.

The source report and three H5 recordings remain in the ignored validation output folder:
`work/validation_runs/pid-stimulation-demo-4e9cb5b-attempt1/`.
The audit checks their completed H5 metadata, committed extents, block indexing, feature-valid coverage, AO range identities, and source-indexed command waveform. The bundle does not copy those full recordings.

## Historical negative control

The same auditor was run against the earlier `ef81a23` tracking report, which had top-level functional flags set even though its no-stimulation control encountered an AO fault and its recorded source prefix was incomplete. The auditor rejected it with 11 failures, including the full-recording/runtime checks. This demonstrates that the artifact audit does not accept report-level tracking flags in place of closed-H5 completeness and scheduler evidence. The retained negative-audit output is `outputs/pid-stimulation-demo-historical-negative-r1/` in the task workspace.

## Reproduce the audit

From the repository root, using the same Windows task workspace paths and a Python environment with the project dependencies installed:

```powershell
python tools/audit_pid_stimulation_demo.py `
  --report work/validation_runs/pid-stimulation-demo-4e9cb5b-attempt1/responsive_simulation_20261030T171013Z.json `
  --output-dir work/validation_runs/pid-stimulation-demo-4e9cb5b-attempt1/pid-audit-reproduced
```

The command requires the source report and its sibling `recordings/` and `recordings/capacity_reference/` directories. It creates a new output directory and exits nonzero if any identity, coverage, safety, pulse-range, or functional gate fails. To run the small auditor tests:

```powershell
python -m pytest -q tests/test_pid_stimulation_demo.py
```

The focused fixture suite passed on Python 3.14.0: 7 tests. These tests check that a pulse request alone cannot pass as output, and reject unknown writes, source mismatches, truncated recordings, runtime faults, and broken AO range coverage.
