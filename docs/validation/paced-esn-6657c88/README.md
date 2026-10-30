# Pre-progress-fix paced ESN diagnostic

Revision `6657c888de432b92ca4bfab56d8d43ae59fb4665`, before the progress-
reporting fix, ran the May 29 trace with the browser connected. Near source
second 1,080, a simulated AO active-demand fault stopped acquisition and
recording at 21,606,700 committed samples. The replay loop continued afterward
and processed all 36,000,000 source samples. The file closed cleanly; it is a
prefix, not complete recording coverage.

The Core p99 was 2.1158 ms, under the strict 2.5 ms gate. The maximum was
38.7395 ms, with 237 processing deadline misses, so the case failed timing. The
AO drain completed with queue depth zero, but one active-demand miss and the
runtime fault fail acceptance. No late or stale AO commands were reported.

The compact case values are in `case-summary.json`. The original reports and
recording remain in the task outputs, outside this repository. These numeric
Core and recorder results apply only to this pre-progress-fix revision; current
full-runtime paced reruns remain pending. This is not physical-output or
hardware evidence.
