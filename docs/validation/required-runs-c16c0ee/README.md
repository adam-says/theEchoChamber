# Required full software runs: qualification failed

The combined bundle was assembled at revision `c16c0eee0174c7c016d05a5ce617186a3c4c42d8`; the runs were not all executed at that revision. The nine-case unpaced replay matrix was executed at `bb65477`, and the three separate 30-minute paced runs with an actual browser were executed at `c16c0ee`. The matrix is included because the subsequent changes were scoped to post-run/reporting behavior, outside the matrix's measured timed-component scope. Each report retains its own source revision. This is a combined evidence bundle, not a claim that all 12 runs used one identical checkout. The combined status is **REQUIRED FULL RUNS FAILED** (matrix 9/9 fail; paced 3/3 fail). Strict limits remain p99 below 2.5 ms and maximum below 5 ms. These results do not support a real-time-ready qualification.

## Paced results and recording audit

The [timing summary plot](controller_timing.png) orders the nine unpaced matrix
cases first and the three paced cases last; the report table identifies each
run type. Its ESN p99 values are all-call aggregates, including inactive calls
after acquisition stopped in the paced ESN case.

| Controller | Recorded samples | p99 (ms) | Maximum (ms) | Deadline misses | AO active-demand misses | AO late commands | Browser disconnects |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule | 36,000,000 / 36,000,000 | 1.499801 | 35.1682 | 49 | 0 | 0 | 0 |
| PID | 36,000,000 / 36,000,000 | 1.399002 | 27.5681 | 46 | 0 | 0 | 0 |
| ESN | 5,564,500 / 36,000,000 | 1.312801 | 53.3083 | 17 | 1 | 0 | 0 |

The ESN p99 is an all-call aggregate that includes calls after acquisition
stopped; it is not a post-fault active-control p99.

The Rule and PID recordings contain the full source. The ESN finite replay processed all 36,000,000 source samples, but a latched simulated AO active-demand fault stopped acquisition and recording at 5,564,500 samples (278.225 seconds). Its HDF5 file closed cleanly but contains only that prefix. All three runs drained the AO queue and had no browser disconnects. The independent bounded-chunk audit checked the v5 structures, indices, controller diagnostics, AO events, source/provenance hashes, and committed extents; it found no structural or provenance issues. A clean closed file does not imply complete input coverage.

The ESN file contains 5,767 coalesced fault-zero ranges totaling 30,439,400 accepted zero AO samples through output sample 36,005,800. These are output-history records and do not extend the ESN input recording past 5,564,500 samples. All three paced maxima exceed 5 ms, so all paced cases fail timing acceptance.

The nine-case matrix also failed 9/9. Every maximum exceeded 5 ms; all three ESN p99 values exceeded 2.5 ms. Its PID cases had positive requested-rate blocks but zero pulse proposals, commits, and nonzero AO samples. The paced PID case also had zero pulse proposals and output samples. The 45-second lead sweep, default simulations, synthetic tracking diagnostic, and injected-stall proof are separate evidence bundles and do not turn these required runs into a pass.

Progress events were fully written (201/201 Rule, 201/201 PID, 200/200 ESN) with no progress-writer error. Browser connectivity and packet consumption do not prove that a person saw rendered frames.

## Evidence

- [Combined twelve-run report (Markdown)](controller_validation_summary.md)
- [Combined twelve-run report (JSON)](controller_validation_summary.json)
- [Timing-only plot](controller_timing.png)
- [Unpaced matrix report and closed-file audit](../replay-bb65477/README.md)
- [Independent paced closed-file audit (JSON)](paced_30min_evidence.json)
- [Paced segment table (CSV)](paced_30min_segments.csv)
- [Original paced audit notes](paced-audit-source.md)
- [Final tested-input provenance proof](final-runtime-input-proof.json)
- [Audit reproduction instructions and scripts](audit-reproduction/README.md)

The preserved provenance proof covers 69 captured code, test, config, model,
and UI inputs outside the documentation. It compares the tested input snapshot
with the final main source snapshot against the 286-test manifest. It reports
no content mismatches and 30 main-versus-tested-worktree differences that are
line endings only. A separate Git-blob check found 15 exact matches and 54
files equivalent after LF/CRLF conversion. The reconstruction notes explain
why line-ending counts depend on checkout settings. A scoped `.gitattributes`
file preserves the included audit snapshots byte-for-byte in Git. This proves
input provenance only, not timing or hardware behavior.

Hardware status remains **ALL UNVERIFIED**. Simulated AO and NI scheduling data are not physical delivery or safety evidence. No platform selection or timing-threshold change is inferred here.
