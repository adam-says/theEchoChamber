# Synthetic PID tracking evidence

Source revision: `ef81a2348bbf927da10da086a10c555359616f16`; Python `3.14.0`.
Recorded scenario: 64.0 s at 20000 Hz, 1280000 samples per segment; physical pulse claims: **false**.

This report separates simulated tracking metrics from actual-core timing checks. It does not establish real-time acceptance, physical output, or biological efficacy.

## Segment recording and runtime checks

| Segment | Samples committed | Complete | Diagnostic rows | p99 ms | Max ms | Deadline misses | AO misses | AO drain | Gaps |
|---|---:|:---:|---:|---:|---:|---:|---:|:---:|---:|
| stimulated | 1280000 | True | 12800 | 1.609446000000001 | 24.0277 | 8 | 0 | True | 0 |
| no_stimulation_control | 1193300 | True | 11933 | 1.496250000000001 | 280.6627 | 11 | 1 | True | 0 |
| capacity_reference | 1280000 | True | 12800 | 1.6814240000000005 | 5.1051 | 1 | 0 | True | 0 |

## Tracking metrics reported by the synthetic simulation

The stimulated segment reports IAE `35.32679305879348 z·s`, source coverage `1.0`, eligible observation coverage `1.0`, raw error SD `1.9044885806274352 z` (limit `2.0 z`), settling time `14.16 s`, and bounded-recovery result `True`.
These are scenario metrics, not an independent functional or real-time pass claim.

## Capacity and regulated controller usage

The separate maximum-request capacity fixture reports an effective cadence bound of `1.2708095056551023 Hz`, a 22.0–42.0 s steady-window reduction of `0.0030760099316544972 mV`, and target reach `True`.
The regulated stimulated segment instead generated `32` effect pulses during the disturbance (1.0666666666666667 Hz). Capacity is a separate reference case; it is not substituted for observed controller usage.

## Runtime timing evaluation

Runtime failure messages are calculated with the existing `RunMetrics` and `acceptance_failures` implementation. The 64-second run is diagnostic scope and does not satisfy a long-duration real-time acceptance gate.

- `stimulated` runtime checks: maximum process time reached or exceeded the block deadline; one or more blocks exceeded the processing deadline
- `no_stimulation_control` runtime checks: runtime fault was latched: simulated AO host refill stalled past committed output; maximum process time reached or exceeded the block deadline; one or more blocks exceeded the processing deadline; AO active-demand deadline miss occurred; late AO command was reported; recorded sample count does not match submitted processing blocks
- `capacity_reference` runtime checks: maximum process time reached or exceeded the block deadline; one or more blocks exceeded the processing deadline

## Evidence identity

Report SHA-256: `1bd96065ee582e2c87c650fcc32373d1673ae48c4dfa8adc25927556eccbe17b`.
CSV SHA-256: `5ea20e4b1aff26c6d9a807d6e3214bb785b7186452da84f0a162c9c9a8aafd3a`.
HDF5 and source file hashes are listed in `tracking_evidence.json`.
