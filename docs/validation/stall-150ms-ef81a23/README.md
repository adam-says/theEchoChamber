# Injected 150 ms AO stall proof

This targeted software fault-injection case ran on revision `ef81a2348bbf927da10da086a10c555359616f16`. The expected fail-safe behavior checks passed; runtime acceptance correctly failed because the stimulated recording stopped at a latched AO fault. This is not a clean timing trial or physical hardware evidence.

The test injected exactly one 150 ms producer stall. One pulse proposal (`g2-p1`) was made at source sample 249,999 for AO target 251,999. The injected AO fault rejected and canceled that request before any pulse samples committed. There were zero committed pulse IDs/samples, zero nonzero AO samples, zero plant effects, and zero final rate phase. The final AO drain completed with an empty queue.

The stimulated HDF5 file recorded a contiguous 250,000/320,000-sample prefix (78.125%) before the fault. The uninjected control recorded all 320,000 samples without a fault. Both finite source loops processed their full inputs. The faulted run reported one active-demand miss, with no AO late command in this specific test. The strict runtime acceptance result remained false for the intentionally faulted case.

## Audit files

- [Expected-fault audit](stall_evidence.json)
- [Per-segment summary](stall_segments.csv)
- [Original audit notes](audit-source.md)

All AO values are simulated commands; this test makes no physical pulse or hardware safety claim.
