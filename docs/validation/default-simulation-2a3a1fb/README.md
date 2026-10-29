# Default responsive-simulation export and runtime evidence

This six-segment 24-second run is retained as a failed diagnostic from revision `2a3a1fb052ade0ea57057af52bcd573523a9a626`. It predates the CSV null-handling fix.

The CLI wrote the aggregate JSON, then failed while exporting CSV because the incomplete-coverage ESN tracking result omitted the optional post-settling raw-error spread. The saved CSV therefore contains only the first two controller rows. This is a post-run serialization failure; it does not itself indicate that a controller or recording failed.

The independent HDF5 audit found five complete 480,000-sample recordings. The PID no-stimulation control recording stopped at 273,100 of 480,000 samples after a latched simulated-AO fault. It reported one active-demand miss, ten late commands, and a 211.9717 ms maximum. The fault and shortened recording are a separate runtime failure; the CSV fix does not resolve or explain them. The audit found all functional acceptance gates failed, and modeled capacity was not reachable. These generated AO values are simulated commands; no physical output claim is made.

The reporting fix changes only post-run summary/CSV handling and does not alter
the runtime components measured in the earlier matrix or lead-sweep runs. The
post-fix retry wrote all three CSV rows but had a separate ESN stimulated AO
refill fault and still failed functional acceptance. See the
[retry audit on `ef81a23`](../default-simulation-ef81a23/README.md).

## Preserved audit files

- [Audit summary](default_simulation_evidence.json)
- [Per-segment table](default_simulation_segments.csv)
- [Original audit notes](audit-source.md)

No HDF5 recording or raw trace is included here. The source report and recordings remain at the paths recorded in the audit for local review.
