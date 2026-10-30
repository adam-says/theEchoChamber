# Fault-event recording evidence

This note records the fault-event writer change and the paced attempts that
motivated it. The paced reports use revision `09ebf1e`. They predate the
isolated recorder change, so they do not measure its performance or behavior.

## What the recorder records

After an AO fault, the scheduler can keep submitting known zero commands while
AI acquisition has stopped. The recorder now merges only confirmed,
pulse-free zero writes whose half-open AO ranges are contiguous and whose
generation, controller, and reason match. It keeps pulse outcomes and unrelated
action stages as separate rows in insertion order.

The writer services pending action rows on a timed queue wait, even when no
`RecordBlock` arrives. The pending-row limit includes queued rows, the open
range, and rows currently being appended. Attempting to exceed 4,096 rows makes
the recorder latch the first error, stop admission, and leave the file
incomplete. A writer-side overflow aborts the batch before it advances `committed_samples`. The AO
producer does not perform HDF5 writes or wait for the file lock.

The NI path reports a zero range only after the writer accepts the full
zero-filled refill. A partial or unknown write does not receive a full-block
range. A missing waveform alone does not prove that a zero write succeeded.
The simulator records only its own scheduled AO command ranges. These rows do
not claim physical output.

## Paced attempts on revision `09ebf1e`

The Rule and PID jobs each processed the full 36,000,000-sample finite replay.
Both latched one active AO deadline fault and failed the pacing gates.

| Controller | Processed samples | Recorded samples | Deadline misses | AO active-demand misses | Browser disconnects | All-call p99 | Maximum call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule | 36,000,000 | 16,608,200 | 578 | 1 | 1 | 1.774801 ms | 428.0297 ms |
| PID | 36,000,000 | 480,300 | 103 | 1 | 0 | 0.9009 ms | 66.7837 ms |

The Rule recorder contains 830.41 seconds of source data. The PID recorder
contains 24.015 seconds. Each file reports `complete=1` because the writer
closed the recorded prefix cleanly. Neither file contains the full replay.

The fault stops acquisition and `ProcessingCore` stops submitting
`RecordBlock` values. The replay loop continues through the remaining finite
input. The all-call p99 includes inactive calls after acquisition stopped. It
does not measure active controller timing after the fault.

The original combined PID and ESN startup attempts timed out after 120 seconds
waiting for browser connections and produced no completed case metrics. A
separate PID run completed and failed as shown above. A separate 30-minute ESN
run remains pending.

## Software test results

The full `python -m pytest -q` suite passed on the isolated candidate based on
`c32d151`. The candidate was uncommitted during both runs. Python 3.12.14
reported 269 passed and 17 subtests in 38.79 seconds. Python 3.14.0 reported
269 passed and 17 subtests in 36.84 seconds. The input hashes were unchanged
through both runs.

The suite covers legacy v1-v3 recordings, actual-core fault and EOF outcome
classification, and recorder overflow and commit-boundary cases, along with the
other project tests. The earlier focused test selections differed between
Python versions. These full-suite results replace those focused counts for the
current candidate. The 252-test, 17-subtest results on published `09ebf1e` are
historical and do not describe this candidate.

The [test manifest](unit-test-manifest.json) records each interpreter, duration,
exit code, input hash, and summary. Read the [Python 3.12 log](unit-tests-python312.txt)
and [Python 3.14 log](unit-tests-python314.txt) for the full output.

The benchmark also keeps a 32-row history of recent Core calls for fault
diagnostics. Each row separates call start from completion. The fault snapshot
reports active and latest-started source indices separately from the latest
completed index. The AO-clock observer captures the history after it sees a
fault. That observation is not the fault onset. History capture occurs outside
the measured Core interval, though it adds harness work that may affect later
scheduling. It does not change the strict p99 `<2.5 ms` and maximum `<5 ms`
limits.

## Limits and source reports

These results do not identify the cause of the AO deadline fault. They do not
show that fault-zero coalescing improves latency. The strict limits remain
`p99 < 2.5 ms` and maximum process time `< 5 ms`. Hardware status remains
**ALL UNVERIFIED**.

- [Rule summary and case report](../paced-browser-09ebf1e-attempt1/summary.json)
- [Rule raw case JSON](../paced-browser-09ebf1e-attempt1/controller_case_001_20261029_120911_20261029T110911Z_2026_05_29_0001_rule_100ms.json)
- [PID collection report](../paced-pid-09ebf1e/controller_paced_20261029T125918Z.json)
- [PID raw case JSON](../paced-pid-09ebf1e/controller_case_001_20261029_132852_20261029T122852Z_2026_05_29_0001_pid_100ms.json)
