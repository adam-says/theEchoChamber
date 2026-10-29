# 64-second synthetic PID tracking diagnostic

This diagnostic ran three actual-Core software segments on revision `ef81a2348bbf927da10da086a10c555359616f16`: stimulated PID, no-stimulation PID control, and a separate capacity reference. Each requested 1,280,000 source samples. This 64-second run is not the required 1,800-second timing acceptance test; hardware status remains unverified.

## Tracking behavior and recording coverage

The stimulated segment and capacity reference recorded all 1,280,000 samples. The PID control recording stopped at 1,193,300 of 1,280,000 samples (93.23%) after a simulated-AO host-refill fault observed near source time 59.665 seconds. Its AO scheduler reported one active-demand miss and five late commands. The finite source loop completed, but the recorder held only the prefix.

The stimulated run's synthetic tracking/recovery measurements passed their functional checks: 4,673 eligible observations, 100% source and eligible-observation coverage, 1.90449 z raw-error standard deviation against a 2.0 z limit, and 14.16-second settling time. Bounded recovery passed. The separate capacity reference reached its modeled target. These modeled functional results do not override the control segment's runtime fault or shortened recording, and do not constitute full-simulation acceptance.

## Runtime timing

| Segment | p99 (ms) | Maximum (ms) | Deadline misses | Recording |
|---|---:|---:|---:|---:|
| Stimulated | 1.609446 | 24.0277 | 8 | 1,280,000 / 1,280,000 |
| No-stimulation control | 1.496250 | 280.6627 | 11 | 1,193,300 / 1,280,000 |
| Capacity reference | 1.681424 | 5.1051 | 1 | 1,280,000 / 1,280,000 |

All three p99 values are below 2.5 ms, but every maximum exceeds the strict 5 ms limit. The control segment also faulted and missed full recording coverage. Thus this diagnostic is not a clean runtime trial despite its passing functional tracking measures.

## Evidence files

- [Audited summary](tracking_evidence.json)
- [Per-segment measurements](tracking_segments.csv)
- [Original audit notes](audit-source.md)

AO values represent simulated command data, not physical pulse delivery. The synthetic plant is not a tissue model.
