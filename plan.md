# Biomimetic Closed-Loop LFP

## Summary:
    This project aims to implement a closed-loop reservoir computing algorithm to prevent epileptic seizure-like activity in rodent brain slices.
    The algorithm will be implemented in Python and will use the NI USB-6343 board for digital interface.
    On the biological side, brain slices (hippocampal-cortical) will be incubated with 4-AP to increase excitability. Seizure-like activity will be triggered by cutting the Schaffer collaterals, disrupting the hipp-ctx loop.
    This will be monited using extracellular LFP recordings: one recording electrode in the CA3 region, and a second one in the entorhinal cortex.
    A stimulation electrode will be placed adjecent to the cortex, after the cut.
    
## Software-side plan:
    1. Connect to the NI board
    2. Set up the board for simultaneous recording and stimulation (two AI channels/one AO channel, FS = 20000 Hz)
    3. Create a visual monitor of the data stream, using Python-based websockets (so we have a web interface to monitor the experiment, where we can see the recorded activity, and the eventual stimulation in realtime)
        3.1. This visual interface should offer the possibility of start/stop acquisition, start/stop gap-free recording, and the choice between control recording (no stimulation), and the closed-loop recording (stimulation is applied - stimulation defined by the RC algorithm)
    4. Implement a buffer system to store incoming data and feed it into Angel's control algorithm - should be in realtime
    5. Channel the RC algorithm output to the AO channel to stimulate the brain slice
    6. Be sure that the data is saved to disk in a format that can be used to analyze the experiment

## Backlog / decisions to implement

### User interface

- Restore project branding in the WebSocket UI.
- Replace the `The Echo Chamber` text in the top bar with `assets/echoChamberLogo.png`.
- Remove the duplicate logo from the body of the interface and preserve a compact top bar on smaller screens.

### Recording format

- Use a documented, chunked HDF5 scientific-data format.
- Preferred target: NWB/HDF5, with chunking and lossless compression.
- Store time as `starting_time + sample_rate`, rather than saving two floating-point sample-index rows for every sample.
- Store continuously sampled AI and commanded/monitored AO as compressed datasets.
- Store block-level ESN diagnostics, mode changes, faults, and detected pulses as lower-rate tables/events rather than repeating scalar values at 20 kHz.
- Avoid duplicating raw and calibrated LFP arrays when calibration can be represented by metadata.
- Use `float32` for derived continuous traces unless an analysis requirement demonstrates that `float64` is necessary.
- Provide an optional CSV export for a selected time window; CSV is not suitable as the primary gap-free recording format.

### AO underrun investigation

- Treat NI error `-200290` as an implementation/configuration fault, not an expected consequence of ESN computation. One AO channel at 20 kHz is a very small data rate for this hardware.
- [x] Use 100-sample processing blocks and one user-facing AO timing setting (`ao_target_lead_ms`).
- [x] Remove fixed refill-chunk and mandatory write-batch parameters. The application derives its deadline from the target lead and dynamically sizes each NI write.
- [x] Ensure missing commands become safe zeros at the derived half-lead deadline rather than waiting for the final buffered block.
- Record total generated/written samples, calculated queued samples, minimum queue depth, DAQmx write duration, writer-loop scheduling gaps, and command-ready backlog.
- Build an incremental Windows test matrix:
  1. AO-only, non-regenerating, continuously generated zero blocks.
  2. AO-only with a precomputed changing waveform supplied continuously.
  3. Synchronized AI acquisition plus zero AO, without UI, recording, or ESN.
  4. Add the processing queue and recorder.
  5. Add the ESN bridge.
  6. Add the WebSocket UI.
- Run every stage as a prolonged soak test and identify the first stage that loses AO headroom.
- [x] Add `benchmarkEchoChamberProcessing.py` for artifact-backed synthetic/recorded-input throughput testing and a board-free paced AO-consumer simulation.
- [x] Benchmark recording `20260812_163821_876650`: paced mean 2.630 ms, p99 3.091 ms, one block over the 5 ms deadline, no simulated AO underruns, and at least 18/20 lead blocks retained. Processing is not the leading explanation for NI `-200290`.
- [x] Compare chunk sizes on `20260812_152531_423641`: 100 samples gave paced mean/p99 2.536/4.768 ms, 200 gave 4.342/5.255 ms, and 400 gave 7.862/8.975 ms. All completed without simulated AO underruns; 200 samples is the current latency/headroom compromise.
- Keep regeneration disabled for closed-loop stimulation; repeating an old command is not an acceptable recovery mechanism.
- After stable operation, reduce the deliberately conservative 400 ms AO lead and measure the achievable closed-loop latency.

### ESN boundary

- [x] Introduce an application-owned `esn_bridge.py` between the frozen collaborator ESN and `echoChamber_v4.py`.
- [x] Adapt DAQ chunks to the artifact chunk size and move stimulation mapping, DC blocking, adaptive pulse thresholds, pulse generation, and diagnostics into the bridge.
- [x] Keep the collaborator-owned `esn/` package unchanged.
