# Echo Chamber stabilization plan

This plan records the agreed implementation goal: stabilize and optimize the
current NI architecture, replace the unsafe/slow model boundary, preserve the
current pulse waveform, and make pending-test assumptions explicit without
blocking research hardware runs.

## 1. Model boundary

- [x] Inspect the supplied notebook and serialized model.
- [x] Export reservoir/readout arrays, leak rate, scaler coefficients, and zero
  initial state into a versioned numeric artifact.
- [x] Reject pickle in the live application; isolate trusted-pickle conversion
  in an offline migration tool.
- [x] Implement deterministic stateful NumPy and Numba inference.
- [x] Verify the recurrence sample-for-sample against ReservoirPy 0.3.11.
- [x] Add causal, stateful anti-aliasing, decimation, model-band filtering, and
  output reconstruction.
- [x] Implement corrected refitting with separate CTX/CA3 scalers and one-second
  washout while preserving the reservoir.
- [x] Run corrected refitting over 14 pairs and acceptance on five held-out
  pairs; make the accepted corrected artifact the default.

## 2. Stimulation and safety

- [x] Default to stimulation off, gain zero, and unarmed.
- [x] Require explicit arming; disarm on control mode, acquisition stop,
  stimulation off/zero, and any fault.
- [x] Latch faults and require restart after the cause is corrected.
- [x] Use one explicit ESN-unit-to-AO-voltage conversion.
- [x] Make pulse polarity selectable (`absolute` default).
- [x] Detect threshold crossings against prior history and exclude event blocks
  from the adaptive baseline.
- [x] Preserve the explicit 5 ms, 100 Hz sine half-cycle across block boundaries.
- [x] Include a `confirmed_pending_test` hardware profile, allow research runs
  and explicit overrides with prominent warnings, and retain explicit arming.
- [ ] Measure the amplifier, isolator, electrode, and safety profile on a
  scope/dummy load, then replace the pending values.

## 3. Timing, recording, and UI

- [x] Use 100-sample/5 ms processing blocks by default.
- [x] Retain non-regenerating, hardware-clocked NI AO with proactive queued
  writes and safe-zero deadline substitution.
- [x] Bind the server before starting DAQ work and keep UI serialization off the
  acquisition thread.
- [x] Record compact HDF5 v4 files with true hardware sample indices, AO target
  indices, arm state, calibration, diagnostics, and sparse events.
- [x] Show gaps on the hardware time axis and align AO plots by recorded delay.
- [x] Make the browser UI offline, responsive, and backend-authoritative.
- [x] Verify Python 3.12/3.14 tests, mock replay, and 5 ms software benchmarks.
- [ ] Execute the staged NI test matrix and a four-hour final soak with zero AO
  underruns and no unexplained gaps.

Detailed evidence and pending hardware tests are in `IMPLEMENTATION_STATUS.md`.
