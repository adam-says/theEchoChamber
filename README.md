# The Echo Chamber

<p align="center">
  <img src="assets/echoChamberLogo.png" alt="The Echo Chamber logo" width="640">
</p>

This project aims to implement a closed-loop reservoir computing algorithm to prevent epileptic seizure-like activity in rodent brain slices.
The algorithm will be implemented in Python and will use the NI USB-6343 board for digital interface.
On the biological side, brain slices (hippocampal-cortical) will be incubated with 4-AP to increase excitability. Seizure-like activity will be triggered by cutting the Schaffer collaterals, disrupting the hipp-ctx loop.
This will be monited using extracellular LFP recordings: one recording electrode in the CA3 region, and a second one in the entorhinal cortex.
A stimulation electrode will be placed adjecent to the cortex, after the cut.

## Development status

> **Work in progress:** Echo Chamber is research software under active
> development. The current application has not yet been validated for
> unattended experiments or direct use with a biological preparation.

The following implementation details still require experimental validation:

- A low-pass filter was required to obtain a useful ESN signal, but its frequency
  response, delay, and suitability for the final preparation must still be
  verified end to end.
- `esn_bridge.py` is a compatibility layer. It was created while testing different DAQ chunk sizes and now adapts runtime chunks to the ESN artifact, without changing the ESN itself. It also contains the current application-side output mapping and pulse diagnostics.
- The stimulation safety configuration is unfinished. Its voltage, slew,
  command-area, duty-cycle, and continuous-output limits are provisional and
  must be selected and validated for the actual stimulator or stimulus
  isolator, its command conversion, the electrode, and the intended waveform.
  These software limits do not replace electrical isolation or the hardware's
  own current and charge protections.

`echoChamber_v4.py` is retained as the last version tested with the NI board.
`echoChamber.py` is the current development version.

## Contributors
[Adam Armada-Moreira](https://github.com/adam-says)

[Angel Canal-Alonso](https://github.com/AngelCanal)

[Alessio Di Clemente](https://github.com/alediclemente)

[Laura Monni](https://github.com/LauraMonni1)

[Michele Giugliano](https://github.com/mgiugliano)

## Usage
If connected to the NI board
```
python echoChamber.py
```

Replay the bundled two-channel sample in test mode (no NI device required):
```
python echoChamber.py --mock
```

Replay a different Echo Chamber recording:

```
python echoChamber.py --mock-replay recordings/your_recording.h5
```

Currently, the "normal" script has strict safety conditions for stimulation, which
prevents the Closed-Loop Passthrough mode from running. To test this mode, only in **dry test with no stimulation hardware connected**, run as:
```
python echoChamber.py --mock --dry-test-allow-sustained-ao
```

Echo Chamber records directly to a compact `.h5` file. A dedicated writer thread
stores compressed `float32` AI and AO signals, calibration metadata, block-rate
ESN diagnostics, and sparse mode, pulse, and sample-discontinuity events.
Files carry a versioned format identifier both as a root HDF5 attribute and
inside the embedded metadata. Existing recordings use `echoChamber_H5_v1`;
files created by the current application use `echoChamber_H5_v3`.

The monitor's optional recording-name field produces
`YYYYMMDD_HHMMSS_CUSTOMNAME.h5`; if left empty it produces
`YYYYMMDD_HHMMSS_echo.h5`. A same-second Echo Chamber recording with less than
one second of committed data is treated as a premature start and replaced;
longer or unrecognized files are never overwritten.

### Plot a recording

Select the specific recording to load:

```
python readEchoChamberData.py sample/test_AI0_CA3_AI1_CTX.h5
```

Plot a time window, or save it without opening a window:

```
python readEchoChamberData.py sample/test_AI0_CA3_AI1_CTX.h5 --start 10 --duration 30
python readEchoChamberData.py sample/test_AI0_CA3_AI1_CTX.h5 --save echo-plot.png --no-show
```

The reader accepts Echo Chamber `.h5` recordings; metadata are embedded in the file.

## ESN closed-loop setup

Install the ESN dependencies:

```
python -m pip install -r requirements.txt
```

Files:
- `esn_artifact.pkl`: pre-trained ESN + scaler + config (loaded by `echoChamber.py`)
- `esn/`: streaming ESN runtime package used by the closed-loop loop

