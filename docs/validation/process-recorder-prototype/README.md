# Process-recorder prototype diagnostic

A single 60-second-source A/B/A ESN diagnostic compared the threaded recorder,
a parked process-recorder prototype, and the threaded recorder again. All three
runs processed and recorded 1,200,000 samples with complete files, no runtime
faults, no source gaps, clean AO drain, and no AO late/active-demand errors.
The measured Core p99 values were 3.423 ms (thread), 3.533 ms (process), and
3.495 ms (thread); maxima were 6.855, 8.898, and 9.286 ms. All exceeded the
strict `<2.5 ms` p99 and `<5 ms` maximum gates. The process arm also took longer
wall time in this one diagnostic.

This is not evidence of a timing improvement or of a unique GIL cause. The
process prototype remains parked and is not the application default. The
prototype used bounded IPC and explicit commit acknowledgments, but its IPC
copies and transient queue occupancy were not identical to the threaded
recorder. No full-duration matrix, paced acceptance, or hardware result follows
from these runs.
