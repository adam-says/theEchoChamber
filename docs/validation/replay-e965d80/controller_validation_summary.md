# Controller validation evidence

**Overall status:** REQUIRED FULL RUNS FAILED

**Short diagnostic status:** NOT REPORTED

Physical hardware and monitor behavior remain **unverified**. Software AO scheduling and synthetic-plant commands are reported separately and do not establish physical delivery or biological effect.

Coverage reference: 1800 seconds, 360,000 blocks, 36,000,000 samples per full recording.

Per-block process timing covers the controller, renderer, safety adapter, recorder submission, and UI submission. Asynchronous health checks remain separate. Unpaced recorder backpressure waits occur between blocks and are excluded from that timing. Browser connection and packet consumption do not prove that a person saw rendered frames.

## Required full-duration runs

- Nine-case unpaced matrix: **FAIL** (9/9 cases).
- Three-controller paced runs: **NOT REPORTED** (0/3 controllers).

| Status | Run | Source | Controller | Blocks | Samples | p99 ms | Max ms | Gaps | Skipped | AO misses, late, stale, zero | Recorder | Browser lifecycle |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| FAIL | matrix @ 100.0 ms | 2026_05_29_0001 | rule | 360,000 / 360,000 | 36000000 / 36,000,000 | 2.260 | 40.112 | 0 | 0 | 0, 0, 0, 20 | complete=True; committed=36,000,000/source=36,000,000/processed=36,000,000 | not required |
| FAIL | matrix @ 100.0 ms | 2026_05_29_0001 | pid | 360,000 / 360,000 | 36000000 / 36,000,000 | 2.248 | 9.959 | 0 | 0 | 0, 0, 0, 20 | complete=True; committed=36,000,000/source=36,000,000/processed=36,000,000 | not required |
| FAIL | matrix @ 100.0 ms | 2026_05_29_0001 | esn | 360,000 / 360,000 | 36000000 / 36,000,000 | 2.759 | 12.779 | 0 | 0 | 0, 0, 0, 20 | complete=True; committed=36,000,000/source=36,000,000/processed=36,000,000 | not required |
| FAIL | matrix @ 100.0 ms | 2026_08_05_0000 | rule | 360,000 / 360,000 | 36000000 / 36,000,000 | 2.307 | 11.855 | 0 | 0 | 0, 0, 0, 20 | complete=True; committed=36,000,000/source=36,000,000/processed=36,000,000 | not required |
| FAIL | matrix @ 100.0 ms | 2026_08_05_0000 | pid | 360,000 / 360,000 | 36000000 / 36,000,000 | 2.342 | 26.139 | 0 | 0 | 0, 0, 0, 20 | complete=True; committed=36,000,000/source=36,000,000/processed=36,000,000 | not required |
| FAIL | matrix @ 100.0 ms | 2026_08_05_0000 | esn | 360,000 / 360,000 | 36000000 / 36,000,000 | 2.700 | 25.035 | 0 | 0 | 0, 0, 0, 20 | complete=True; committed=36,000,000/source=36,000,000/processed=36,000,000 | not required |
| FAIL | matrix @ 100.0 ms | 2026_08_06_0000 | rule | 360,000 / 360,000 | 36000000 / 36,000,000 | 2.312 | 12.267 | 0 | 0 | 0, 0, 0, 20 | complete=True; committed=36,000,000/source=36,000,000/processed=36,000,000 | not required |
| FAIL | matrix @ 100.0 ms | 2026_08_06_0000 | pid | 360,000 / 360,000 | 36000000 / 36,000,000 | 2.322 | 15.064 | 0 | 0 | 0, 0, 0, 20 | complete=True; committed=36,000,000/source=36,000,000/processed=36,000,000 | not required |
| FAIL | matrix @ 100.0 ms | 2026_08_06_0000 | esn | 360,000 / 360,000 | 36000000 / 36,000,000 | 3.022 | 199.559 | 0 | 0 | 0, 0, 0, 20 | complete=True; committed=36,000,000/source=36,000,000/processed=36,000,000 | not required |

## Short timing diagnostics

No smoke or lead-sweep diagnostics were supplied.

## Failures and limits

- **20261029_091358_20261029T081358Z_2026_05_29_0001_rule_100ms (FAIL):** maximum process time reached or exceeded the block deadline; one or more blocks exceeded the processing deadline; maximum process time 40.112 ms is not below 5.0 ms; deadline-exceeded blocks: 73
- **20261029_091938_20261029T081938Z_2026_05_29_0001_pid_100ms (FAIL):** maximum process time reached or exceeded the block deadline; one or more blocks exceeded the processing deadline; maximum process time 9.959 ms is not below 5.0 ms; deadline-exceeded blocks: 53
- **20261029_092500_20261029T082500Z_2026_05_29_0001_esn_100ms (FAIL):** process p99 is not below 2.5 ms; maximum process time reached or exceeded the block deadline; one or more blocks exceeded the processing deadline; process p99 2.759 ms is not below 2.5 ms; maximum process time 12.779 ms is not below 5.0 ms; deadline-exceeded blocks: 102
- **20261029_093129_20261029T083129Z_2026_08_05_0000_rule_100ms (FAIL):** maximum process time reached or exceeded the block deadline; one or more blocks exceeded the processing deadline; maximum process time 11.855 ms is not below 5.0 ms; deadline-exceeded blocks: 46
- **20261029_093715_20261029T083715Z_2026_08_05_0000_pid_100ms (FAIL):** maximum process time reached or exceeded the block deadline; one or more blocks exceeded the processing deadline; maximum process time 26.139 ms is not below 5.0 ms; deadline-exceeded blocks: 59
- **20261029_094254_20261029T084254Z_2026_08_05_0000_esn_100ms (FAIL):** process p99 is not below 2.5 ms; maximum process time reached or exceeded the block deadline; one or more blocks exceeded the processing deadline; process p99 2.700 ms is not below 2.5 ms; maximum process time 25.035 ms is not below 5.0 ms; deadline-exceeded blocks: 125
- **20261029_094914_20261029T084914Z_2026_08_06_0000_rule_100ms (FAIL):** maximum process time reached or exceeded the block deadline; one or more blocks exceeded the processing deadline; maximum process time 12.267 ms is not below 5.0 ms; deadline-exceeded blocks: 58
- **20261029_095457_20261029T085457Z_2026_08_06_0000_pid_100ms (FAIL):** maximum process time reached or exceeded the block deadline; one or more blocks exceeded the processing deadline; maximum process time 15.064 ms is not below 5.0 ms; deadline-exceeded blocks: 27
- **20261029_100038_20261029T090038Z_2026_08_06_0000_esn_100ms (FAIL):** process p99 is not below 2.5 ms; maximum process time reached or exceeded the block deadline; one or more blocks exceeded the processing deadline; process p99 3.022 ms is not below 2.5 ms; maximum process time 199.559 ms is not below 5.0 ms; deadline-exceeded blocks: 342

## Synthetic simulations

No simulation reports were supplied.

## Import verification

ABF import verification status: **verified** across 3 source files.
See the JSON summary for source and canonical replay hashes.

## Timing plot

![Controller timing evidence](controller_timing.png)
