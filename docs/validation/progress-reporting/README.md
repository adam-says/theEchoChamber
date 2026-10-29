# Progress reporting and frozen unit-suite evidence

The isolated candidate's full test suite passed on both supported interpreters:

- Python 3.12.14: 285 passed, 17 subtests passed (40.47 s).
- Python 3.14.0: 285 passed, 17 subtests passed (40.44 s).

The exact captured outputs are in `unit-tests-python312.txt` and
`unit-tests-python314.txt`. `unit-test-manifest.json` records the commands,
completion times, source/input hashes, interpreter results, and limitations. It
explicitly makes no timing-acceptance or hardware claim.

## Provenance mapping

The tests ran from base commit `95603cf056b9c7b09fafc560ab56b32c76c4fa66`
with the reviewed candidate changes still uncommitted; the manifest preserves
that fact as `tested_uncommitted_candidate: true`. After the run, the isolated
candidate was committed as `0d57f3c` and cherry-picked to the main checkout as
`bb65477dfb79b5372013d9f3ec52cfa2fd263401`. The resulting source tree hash was
`8524f3b3b450d5cc0732743269f69cf9a17b16e5`. These commits map the tested
candidate to the published working tree without rewriting the original test
provenance. The manifest's source hash list is the authoritative record of the
actual tested files.

A later independent mapping check verified that all 93 original tested-source
hashes still match the isolated working tree. It found only line-ending
differences in 31 corresponding files between the isolated and main checkouts;
their Git trees were identical. The original commit IDs and uncommitted-test
status remain unchanged in the manifest. See
`tested-source-mapping-650c4a4.json`.

## What the progress-writer changes cover

The harness sends detached observations through a bounded queue to one
serialization and file-writing owner. Interim progress contains a p99 computed
from the last at most 2,048 completed Core calls, with the exact window's source
bounds. Final metrics and strict timing gates still use the complete latency
history; the interim window does not replace or truncate it.

A progress-writer failure is a case failure. Admission rejections, accepted but
discarded events, written events, and pending accepted events are counted
separately. An unresolved writer owner stops the CLI collection, prevents any
further writes to its JSONL file, and leaves independent JSON/CSV failure
results. Simulated-AO ownership uses the same typed shutdown boundary: abort
first disarms, waits only for its bounded stop interval, and does not close
resources an unresolved AO thread could still use. After the owner stops, retry
flushes final Core AO notifications before recorder closure. A timeout remains a
failed case even if a later cleanup retry resolves the owner.

These tests validate reporting, ownership, and cleanup behavior. They do not
show that asynchronous reporting improves processing latency or that the full
replay/paced gates pass.

## Post-run simulation-reporting correction

A later full suite covered one additional post-run-only correction. An
incomplete-coverage tracking summary now includes an unavailable
`post_settling_raw_error_std_z` value, which strict JSON writes as `null`; CSV
serialization also accepts older summaries without that optional key. This
does not modify live processing, AO, plant, or HDF5 recording.

The run used base `996dd1ef7e3a355a2aa98921dea2ac718939219e` plus uncommitted
changes. They were then committed as isolated `4967720` and main cherry-pick
`ef81a2348bbf927da10da086a10c555359616f16` (tree
`440255a110cd240c7b399df145fc0a360669b2b4`). The preserved root manifest is
[`unit-test-manifest-reporting-fix.json`](unit-test-manifest-reporting-fix.json),
with outputs in
[`unit-tests-reporting-fix-python312.txt`](unit-tests-reporting-fix-python312.txt)
and [`unit-tests-reporting-fix-python314.txt`](unit-tests-reporting-fix-python314.txt).
It records 286 tests and 17 subtests passing on each interpreter, unchanged
inputs, and no timing or hardware claim. This is separate from the earlier
285-test evidence above; neither test run establishes runtime performance.
