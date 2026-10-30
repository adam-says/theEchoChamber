# Full-recording reader round trip

The actual v5 reader and plotter loaded one complete earlier Rule recording and
compared it against its canonical calibrated mV source. The reader report is
`reader-roundtrip.json`; it records all mismatch counts and pre/post hashes.

The fixture covered 36,000,000 source samples at 20 kHz from absolute source
index 0 with no gaps or discontinuities. It verified 360,000 block-diagnostic
rows and 360,000 controller rows, CA3/Cortex roles with the EC region preserved,
raw-AI extent, AO command values, mode data, and source indices. Signal,
absolute-index, block, controller, and mode mismatch counts were all zero.
The report distinguishes sparse action-stage rows from pulse-event rows and
marks physical delivery and timing claims false. The recording and plot image
are intentionally not copied into this repository.

This reader proof predates the final progress-reporting candidate and validates
reader coverage only. It is not evidence of current runtime timing or hardware
acceptance.
