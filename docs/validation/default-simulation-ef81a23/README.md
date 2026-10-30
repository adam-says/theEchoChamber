# Default responsive-simulation retry after CSV fix

This six-segment, 24-second retry ran on revision `ef81a2348bbf927da10da086a10c555359616f16`. The optional tracking metric now serializes as null when unavailable, and the CSV contains all three controller rows. The earlier JSON-success/CSV-error reporting failure did not recur.

The audit found five complete 480,000-sample recordings. The ESN stimulated segment faulted after 136,800 of 480,000 samples (28.5% coverage) with `simulated AO host refill stalled past committed output`. It reported one AO active-demand miss, two late commands, five deadline misses, and a 302.4315 ms maximum. The other five segments, including the PID no-stimulation control, recorded their full 480,000 samples. The finite replay source continued to its endpoint after the fault stopped recording.

All three controller-level simulation acceptance gates failed, and the full simulation acceptance result is false. ESN generated no plant effects; its observed activity did not change. The AO host-refill fault moved from the earlier PID control segment to ESN stimulation in this separate run. This change in observed segment does not establish a cause or show that the fault is fixed. The earlier run is preserved at [revision `2a3a1fb`](../default-simulation-2a3a1fb/README.md).

The reporting correction is post-run serialization only. It does not explain or resolve the AO fault. The six-run audit separately reports 69 coalesced HDF5 fault-zero ranges totaling 336,900 AO samples and 3,471 scheduler zero-substitution blocks (347,100 samples); the difference remains an unreconciled accounting discrepancy. Reported AO values are simulated commands, not physical pulse delivery.

## Preserved audit files

- [Audit summary](default_simulation_evidence.json)
- [Per-segment table](default_simulation_segments.csv)
- [Original audit notes](audit-source.md)

No raw trace or HDF5 recording is copied into this evidence directory.
