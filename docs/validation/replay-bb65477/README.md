# Full nine-case replay matrix on `bb65477`

The matrix ran all three controllers on all three canonical traces at the 100 ms AO lead. Every case processed 360,000 blocks and 36,000,000 source samples. All nine missed the strict maximum-time limit. The three ESN cases also missed the strict p99 limit; Rule and PID p99 values passed.

| Trace | Controller | p99 (ms) | Maximum (ms) | Deadline misses |
|---|---|---:|---:|---:|
| May 29 | Rule | 2.244601 | 16.7774 | 47 |
| May 29 | PID | 2.176102 | 240.7780 | 54 |
| May 29 | ESN | 2.884603 | 148.2125 | 224 |
| Aug 5 | Rule | 2.185002 | 297.8926 | 41 |
| Aug 5 | PID | 2.012201 | 28.2999 | 44 |
| Aug 5 | ESN | 2.628504 | 447.2756 | 133 |
| Aug 6 | Rule | 2.118502 | 15.6181 | 29 |
| Aug 6 | PID | 2.122004 | 17.1542 | 37 |
| Aug 6 | ESN | 2.770607 | 486.6503 | 168 |

All nine case reports record complete 36,000,000-sample files, zero source gaps or skipped samples, zero AO active-demand misses, late commands, or stale rejections, and completed AO drain. The independent closed-file audit checked all nine retained recordings, block and controller rows, headers, and provenance with no structural issues. See `full_matrix_h5_audit.json` and its CSV. The audit validates file structure and reported coverage, not the timing gates.

The PID matrix cases had positive requested-rate blocks but zero pulse proposals, commits, and nonzero AO samples. Their p99 results do not demonstrate PID stimulation on those traces. The run used a WebSocket client without browser rendering; no physical-delivery claim follows from software AO commands.

The detailed case summary JSON, audit JSON and CSV, and timing-only plot are stored beside this README. The strict limits remain p99 below 2.5 ms and maximum below 5 ms. The full matrix failed acceptance.
