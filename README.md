# The Echo Chamber

<p align="center">
  <img src="assets/echoChamberLogo.png" alt="The Echo Chamber logo" width="640">
</p>

The Echo Chamber is research software for low-latency, closed-loop LFP
acquisition and stimulation with an NI USB-6343. It records CA3 and cortical
signals, runs a stateful Echo State Network (ESN), and can convert the model
output into either a passthrough command or a threshold-triggered pulse.

> **Research status:** the corrected ESN is now the default and passed the
> repository's held-out software comparison. Hardware calibration, channel
> mapping, pulse shape, and safety values are marked **confirmed pending test**.
> NI runs are allowed with prominent warnings; stimulation still starts off,
> at gain zero, and requires explicit arming.

`echoChamber.py` is the supported development application.
`echoChamber_v4.py` is retained only as the previous NI-tested reference.

## What changed

- The live runtime uses a checksummed numeric `.npz` artifact and never loads
  pickle files.
- The ReservoirPy graph was replaced by an inference-only NumPy/Numba loop. Its
  recurrence was verified exactly against ReservoirPy 0.3.11 for the supplied
  model.
- Input and target scaling are separate in the corrected training path.
- The 20 kHz input is anti-aliased, decimated to 2 kHz, causally low-pass
  filtered, evaluated by the ESN, and causally returned to 20 kHz.
- Stimulation starts off, at gain zero, and unarmed. Arming is explicit and is
  cleared by control mode, acquisition stop, stimulation off/zero, or a fault.
- Pulse detection supports absolute, positive, and negative polarity. Absolute
  is the default. The current pulse is an explicit 5 ms, 100 Hz sine half-cycle.
- Real-hardware startup uses the included confirmed-pending-test profile by
  default and warns until its values are replaced with measured ones.
- Recording uses compact, compressed `echoChamber_H5_v4` files with true NI
  sample indices, calibration metadata, AO target indices, arm state, faults,
  and sparse events.
- The browser monitor is self-contained and does not require internet access.

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for completed
verification and the remaining pending tests.

## Install

Python 3.12 and 3.14 are supported and tested.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

For development tests:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest -q
```

## Mock replay

Run the bundled whole-slice 4-AP recording without NI hardware:

```powershell
python echoChamber.py --mock
```

The application starts in control mode, stimulation off, gain zero, and
unarmed. The sidecar beside the sample makes its AI0=CA3, AI1=Cortex mapping and
the conflict with its embedded metadata explicit.

Use another Echo Chamber recording:

```powershell
python echoChamber.py --mock-replay recordings\recording.h5
```

Passthrough can exceed the pulse-oriented sustained-output safety rules. It may
be bypassed only for a disconnected scope or dummy-load test:

```powershell
python echoChamber.py --mock --dry-test-allow-sustained-ao
```

The same explicit flag is allowed for NI bench testing but emits a prominent
warning. Use it only with a scope or dummy load because it bypasses charge,
duty-cycle, and continuous-output trips.

## Real-hardware testing

The default command now uses the corrected artifact and the included
`hardware_profiles/lab_pending_test.json` working profile:

```powershell
python echoChamber.py
```

The profile records these confirmed-pending-test assumptions: AI0=CA3,
AI1=Cortex, amplifier gain 10, one AO volt per ESN physical output unit, absolute
pulse polarity, and the current ISO-Flex-oriented safety values. The application
warns but does not refuse the run. Command-line calibration/safety overrides are
also permitted and logged.

Select another profile or artifact explicitly when needed:

```powershell
python echoChamber.py `
  --hardware-profile hardware_profiles\lab_rig_reviewed.json `
  --model-artifact artifacts\esn_corrected_v1.npz
```

The explicit arm step, software limits, and warnings do not replace electrical
isolation, current/charge protection, an emergency stop, or bench observation.

## Corrected ESN training

The supplied notebook reused one scaler for CTX and CA3 and serialized only its
last, CA3-fitted coefficients. The corrected default model was therefore refit
from the paired source recordings; the historical pickle could not recover the
missing CTX scaling.

The held-out result passed: mean Pearson correlation was `0.260` versus
`-0.260` for the historical readout, and mean nRMSE was `1.007` versus `2.062`.
Fourteen training pairs were matched by ascending electrode rank within each
experiment group; that pairing rule is documented as confirmed pending test.

To reproduce the artifact from the source folders:

```powershell
python -m esn.train `
  --training-ctx "PATH\connected\CTX" `
  --training-ca3 "PATH\connected\CA3" `
  --validation-ctx "PATH\validation\X_data" `
  --validation-ca3 "PATH\validation\Y_data" `
  --seed-artifact artifacts\esn_legacy_compat_v1.npz `
  --output artifacts\esn_corrected_v1.npz
```

Training preserves the supplied reservoir weights, input weights, bias, leak
rate, and topology. It fits separate training-only input and target scalers,
refits the ridge readout after a one-second washout, resets state per recording,
and compares held-out correlation and normalized RMSE with the legacy readout.

## Legacy migration

Only the isolated migration tool accepts trusted project-author pickle files.
Never use it on an untrusted pickle.

```powershell
python -m pip install -r requirements-legacy-migration.txt
python tools\migrate_legacy_model.py `
  --model PATH\model.pkl `
  --secondary PATH\secondary_objs.pkl `
  --output artifacts\esn_legacy_compat_v1.npz
```

The result is retained for numerical compatibility and comparison. If selected
for an NI run, the application warns that its validation is pending but does not
block the run.

## Benchmark and recordings

Run the 5 ms processing benchmark against synthetic or recorded data:

```powershell
python benchmarkEchoChamberProcessing.py --stim-mode off
python benchmarkEchoChamberProcessing.py `
  --input sample\test_AI0_CA3_AI1_CTX.h5 --stim-mode off
```

Plot a recording:

```powershell
python readEchoChamberData.py recordings\recording.h5
python readEchoChamberData.py recordings\recording.h5 `
  --start 10 --duration 30 --save echo-plot.png --no-show
```

HDF5 v4 stores compressed `float32` AI/AO signals, calibration, block-level ESN
diagnostics, AO pipeline delay, arm state, and discontinuity/mode/pulse events.
The reader also supports prior Echo Chamber HDF5 versions.

## Contributors

- [Adam Armada-Moreira](https://github.com/adam-says)
- [Angel Canal-Alonso](https://github.com/AngelCanal)
- [Alessio Di Clemente](https://github.com/alediclemente)
- [Laura Monni](https://github.com/LauraMonni1)
- [Michele Giugliano](https://github.com/mgiugliano)
