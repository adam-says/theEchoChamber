# Implementation and validation status

Status date: 2026-08-28

## Completed in software

- Safe, checksummed `echo-chamber-esn-inference-v1` artifact format.
- Stateful NumPy and Numba inference backends with zero-state reset and
  chunk-invariant output.
- Exact recurrence comparison with the original ReservoirPy 0.3.11 graph:
  maximum absolute difference `0.0` over 10,000 random input samples.
- NumPy/Numba comparison: maximum absolute difference `3.55e-15`, within
  `1e-10` tolerance.
- Causal 20 kHz to 2 kHz preprocessing and causal output reconstruction.
- Corrected paired-recording training path with separate input/target scaling,
  one-second washout, per-recording state reset, and held-out acceptance gate.
- Corrected `esn_corrected_v1.npz` trained from 14 paired recordings and
  evaluated on five held-out pairs. The mean comparison passed, so it is now
  the application default.
- Explicit arm/disarm state, latched faults, safe-zero behavior, single AO unit
  conversion, with explicit warnings for pending-test profiles and overrides.
- Threshold decisions use prior baseline history, exclude fired blocks, detect
  actual crossings, and support absolute/positive/negative polarity.
- Gap-aware HDF5 v4 recording and plotting with AO pipeline-delay alignment.
- Offline browser assets, backend-authoritative controls, hardware sample-axis
  plotting, fault visibility, and calibration display.
- Bundled replay sidecar with checksum and explicit channel mapping.

## Automated checks

- 31 tests pass under Python 3.12.
- 31 tests pass under Python 3.14.
- Python compilation succeeds under both versions.
- Mock replay starts, processes the bundled recording, and exits cleanly.

Python 3.14, 100-sample/5 ms corrected-artifact synthetic benchmark (two runs
of 5,000 unpaced blocks and 1,000 paced blocks):

| Measure | Result |
|---|---:|
| Unpaced mean | 0.677-0.754 ms |
| Unpaced p99 | 2.205-2.312 ms |
| Unpaced maximum | 3.798-3.800 ms |
| Blocks over 5 ms | 0 / 5,000 in both runs |
| Paced mean | 0.882-1.030 ms |
| Paced p99 | 2.371-2.375 ms |
| Paced maximum | 2.912-3.881 ms |
| Simulated AO underruns | 0 in both runs |
| Simulated queue range | 19-21 blocks |

Both runs pass the p99 target below 2.5 ms and maximum target below 5 ms;
neither paced simulation missed its processing deadline or reported an AO-
consumer underrun. These results do not prove NI timing behavior.

The bundled whole-slice replay was also benchmarked with 2,000 unpaced and
approximately 1,000 paced blocks: unpaced p99/max were 1.886/3.162 ms, paced
p99/max were 2.103/4.048 ms, and the simulated AO consumer reported zero
underruns. A separate gain-1 passthrough replay also completed with p99 below
2.1 ms and zero underruns. The complete machine-readable results are in
`docs/benchmark_*_python314_2026-08-28.json`.

## Corrected model result

The downloaded MAT data were processed successfully. Validation files omit an
`fs` field; their 2 kHz rate is taken from `Paper_figures.py` and the pipeline
audit and recorded in the artifact manifest.

| Held-out mean | Corrected | Historical |
|---|---:|---:|
| Pearson correlation | 0.260 | -0.260 |
| nRMSE | 1.007 | 2.062 |

The software acceptance rule passed. The fifth held-out recording has weak
negative corrected correlation (`-0.034`), so per-recording performance should
still be inspected during research testing rather than relying only on the mean.

## Confirmed pending test

For now, NI runs are allowed before bench confirmation. The working profile and
the relevant code comments mark each assumption as `confirmed_pending_test`.
The application prints a clear warning but does not prevent the run.

The following must be performed on the actual Windows/NI rig with a scope or
dummy load before any biological preparation:

1. AO-only continuous, non-regenerating zeros.
2. AO-only precomputed changing waveform.
3. Synchronized AI plus zero AO, without UI, recorder, or ESN.
4. Add processing queue and recorder.
5. Add ESN bridge.
6. Add WebSocket UI.
7. Confirm channel identity, amplifier gain, AO-to-isolator conversion, polarity,
   pulse amplitude/duration, AO monitoring, and every safety trip.
8. Run the complete configuration for at least four hours; require zero NI AO
   underruns, no unexplained sample gaps, and retained AO headroom.

After testing, replace the working profile values with measurements and record
the reviewer, instrument identifiers, date, and scope/dummy-load evidence.
