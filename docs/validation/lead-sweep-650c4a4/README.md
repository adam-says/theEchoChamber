# Post-calibration lead sweep: coverage passed, timing not accepted

The 45-second diagnostic sweep ran unpaced with a virtual AO scheduler on revision `650c4a456019b03f14f88e8c54667874a5361c71` with the May 29 trace, three controllers, and six AO leads (20, 30, 40, 50, 75, and 100 ms). It is not paced or hardware evidence.

The independent r2 evidence audit verified all 18 expected cases, with no structural issues and at least 400,000 eligible post-calibration feature samples in each case. Actual eligible source coverage was 28.25 seconds for Rule, 33.99 seconds for PID, and 44.995 seconds for ESN. These are measured eligible durations, not estimates from the nominal replay duration.

Runtime timing gates passed in 3 cases and failed in 15. The strict limits remain p99 below 2.5 ms and maximum below 5 ms. **Full-duration timing acceptance was not evaluated:** these were 45-second prefixes. This evidence establishes coverage for the sweep, not a timing pass.

The run completed the 45-second HDF5 recordings. The audit reported zero structural issues. The results make no claim of physical pulse delivery; hardware remains unverified. The audit also recorded that the ESN readiness model is not baseline calibration, while Rule and PID use baseline calibration.

## Evidence files

- [Original lead-sweep report (JSON)](controller_lead_sweep_20261029T163915Z.json) and [summary (CSV)](controller_lead_sweep_20261029T163915Z.csv).
- [Independent r2 coverage audit (JSON)](lead_sweep_evidence_audit.json) and [case table (CSV)](lead_sweep_evidence_audit.csv).
- [Audit tests with corrected temporary directory](audit-tests-correct-temp.log): 9 passed and 3 subtests; three h5py fixture deprecation warnings.
- [Initial audit-test attempt](audit-tests-initial-temp-permission-failure.log) retained for provenance; it failed because the temporary directory was outside the configured writable location and is not a test result for the sweep.

The r1 audit naming-contract failure is historical; it did not change the recorded case metrics. The r2 audit is the corrected coverage result. The original metrics and report are preserved unchanged.
