# Mock neural circuit closure test

This paired 24-second virtual-clock test checks that a Rule decision passes through the shared actuator and AO scheduler, changes an excitatory/inhibitory (E/I) model after a configured delay, creates a separate observation artifact, passes through the real Core blanking path, and changes a later valid controller measurement. It tests circuit closure, sample-index timing, response, and basic blanking. It does not test biological target tracking or physical output.

## Model and signal boundary

The fixture uses normalized E/I activity deviations with tanh responses and explicit coupling, bounded states, seeded correlated noise, and forward Euler updates at a 20 kHz sample rate (50 microseconds per step). Its reduced Wilson–Cowan-inspired engineering equations are:

```text
E_next = E + dt/tau_E * (tanh(w_EE_effective*E - w_EI*I + drive + noise_E) - E)
I_next = I + dt/tau_I * (tanh(w_IE*E - w_II*I + noise_I) - I)
w_EE_effective = w_EE / (1 + inhibitory_effect_state)
```

Only AO samples generated for a source block by the virtual scheduler drive the model and artifact generator. A pulse proposal or commit callback does not drive either component. The delivered generated AO enters a 20 ms delay line before it changes the inhibitory-effect state. The effect decays after the delay.

The latent LFP-like proxy is `5 mV * (E - I)`. The mV label is an arbitrary fixed observation scale, not electrode calibration. The observed signal adds a synthetic AO-coupled artifact at 80 mV/V and a 20 mV/V exponential tail with a 2 ms time constant. The actual processing Core receives the observed signal and applies its own 20 ms post-pulse blanking and feature-recovery masks. The artifact oracle is used only to compare expected artifact intervals with the Core's invalid samples. The oracle is not passed to the Core. HDF5 retains the complete observed signal.

The current engineering parameters use `tau_E=1.2 ms`, `tau_I=2 ms`, `w_EE=1.5`, `w_EI=1.5`, `w_IE=3`, and `w_II=0.5`. Euler uses one 50 microsecond step per sample. The model has no exogenous ripple carrier. These chosen values yield an approximately 198 Hz linearized mode; the seeded run's proxy has an approximately 201 Hz dominant oscillation, with 87.8% of its non-DC power in 100–250 Hz. A half-step comparison changed the checked output by 3.08%. These are software-fixture choices and numerical checks, not values from Wilson and Cowan or validated CA3 properties.

## R4 result

R4 passed all ten declared closure gates. The paired Rule-off and Rule-feedback conditions each processed and recorded all 480,000 input samples at 20 kHz. The off condition generated no nonzero AO. The feedback condition made 22 pulse requests, committed 2,200 pulse samples, and generated 2,178 nonzero AO samples. Both conditions drained with no pending commands, latched fault, or unexplained output.

| Event | Sample index | Interpretation |
| --- | ---: | --- |
| First Rule decision | 220,214 | Source onset |
| Latest source sample used | 220,299 | End of the 100-sample input block |
| Target AO onset | 222,214 | 100 ms after source onset by virtual sample indices |
| First nonzero generated AO | 222,215 | The waveform starts with a zero at its target index |
| First latent model effect | 222,615 | 400 samples, or 20 ms, after first nonzero AO |
| First later valid Rule measurement change | 226,399 | After the actual Core blanking and feature recovery |

The latest source sample used to the first target AO onset spans 95.75 ms by sample indices. The interval to the first nonzero generated AO sample is 95.8 ms. These values come from the virtual sample clock. They are not measured host, NI, or physical latencies.

The 0–12 second interval is a baseline-drive and calibration window, not 12 unstimulated seconds. The Rule controller calibrated at sample 220,000 and armed at sample 220,100. Its first decision occurred at sample 220,214, before the 12-second disturbance boundary at sample 240,000. Early feedback therefore overlaps the nominal baseline-drive window.

All HDF5 observed input samples matched the generated input in chunked comparisons. The 4,800 diagnostic block offsets were contiguous. Observed maximum absolute E/I deviations were 0.265/0.372 off and 0.239/0.336 with feedback; all remained inside the model's unit bounds. The Core-invalid feature mask covered every artifact-oracle interval, while the oracle was not supplied to the Core. This checks software masking on the fixture. It does not measure physical electrode artifact or recovery.

The result shows that a decision closes the virtual loop and causes a delayed model response that later reaches a valid controller measurement. The Rule thresholds (`upper_z=1`, `lower_z=0.5`) are fixture settings. The result does not establish tracking to a biological target, model fidelity, NI timing, or tissue effect.

## Run the paired test

Run from the repository root with the project dependencies installed. Choose a fresh output directory:

```powershell
python tools/run_neural_mock.py `
  --output-dir C:\path\to\fresh\neural-mock-run `
  --duration-s 24 --seed 7 --upper-z 1 --lower-z 0.5
```

The command writes paired HDF5 recordings, compressed model-trace sidecars, and `report.json`. It exits nonzero if a closure gate fails. Review the [R4 report](validation/neural-mock/report.json), [source-run manifest](validation/neural-mock/source-run-manifest.json), and [attempt history](validation/neural-mock/failure-history.json). The original HDF5 recordings and model traces remain in the task's output directory and are not copied into the evidence folder.

The latest full unit suite passed 305 tests and 20 subtests under Python 3.12.14 and 3.14.0. It ran on an uncommitted candidate based on `c2e4119`; all 74 captured code, model, configuration, UI, and test inputs remained unchanged. See the [frozen logs and manifest](validation/neural-mock/unit-tests/). The plotting postprocessor was checked separately and is outside that capture.

## Literature basis and limits

Wilson and Cowan's original two-population work models coarse-grained excitatory and inhibitory activity with time constants, recurrent coupling, external drive, sigmoid responses, and refractory-availability factors. Its local-population equations include terms of the form `tau_E*dE/dt = -E + (k_E-r_E*E)*S_E(c1*E-c2*I+P)` and `tau_I*dI/dt = -I + (k_I-r_I*I)*S_I(c3*E-c4*I+Q)`. It reports limit-cycle activity for some stimuli. The mock replaces those response and refractory terms with a bounded engineering variant using `tanh`, Euler updates, seeded noise, and AO-dependent recurrent-excitation damping. It is Wilson–Cowan-inspired, not an exact reproduction.

The original [1972 paper](https://doi.org/10.1016/S0006-3495(72)86068-5) studies localized populations. The [1973 follow-up](https://doi.org/10.1007/BF00288786) extends the model to spatially distributed cortical and thalamic tissue. Neither source validates these mock parameters, the approximately 198–201 Hz fixture oscillation, its arbitrary mV projection, electrode-artifact mapping, or biological efficacy.
