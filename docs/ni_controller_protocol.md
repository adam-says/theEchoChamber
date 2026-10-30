# NI staged controller protocol

This procedure moves from a zero-output check to a four-hour soak. It provides commands and evidence forms. It does not run hardware. No rig check is verified by this document. Keep AO disconnected from an isolator, electrode, or biological preparation until the lab has signed off each preceding stage.

Until the isolator output conversion is calibrated, monitor NI AO command volts only. Do not describe AO voltage as electrode current, delivered stimulation, or charge.

## Current unresolved hardware facts

| Item | Current assumption | Evidence status |
| --- | --- | --- |
| MultiClamp output gain | The runtime profile says 10. The reported setting says 100 mV/mV. | UNVERIFIED. Measure and record the actual amplifier display and a known input. |
| Electrode input conversion | ABF replay data is already in mV. Live volts convert with `1000 / amplifier_gain`. | UNVERIFIED on the NI rig. Do not apply live gain to imported ABF data. |
| AO voltage conversion | The software command is AO volts. Isolator output conversion is unresolved. | UNVERIFIED. Calibrate with a scope and dummy load before connecting an isolator. |
| Channel mapping | AI0 is CA3 and AI1 is Cortex or EC in the imported files. | UNVERIFIED on the rig. Confirm with a known signal before enabling feedback. |
| Physical pulse waveform | The command is a 5 ms, 100 Hz sine half-cycle with a manual 0.5 V demo amplitude and 0.5 s refractory interval. | Software setting only. Scope the actual output at the configured load. |
| Delivered stimulation | A scheduler commit records a software AO command. | No physical delivery claim. Confirm onset, amplitude, and duration on the scope. |
| Charge per phase | Voltage-area is not charge. A calibrated current measurement and pulse duration are needed. A voltage-driven load also needs an impedance measurement to derive current. | Not established. Do not infer charge from a voltage trace. |

## Prepare the evidence folder

1. Copy the application revision, hardware profile, and calibration sheet into a dated lab folder.
2. Record the NI device model and serial number, driver version, application Git revision, Python version, amplifier model, and operator.
3. Keep the AO cable disconnected from the isolator and preparation.
4. Set the output amplitude to zero before each stage.
5. Save the application JSON summary, progress log, HDF5 recording, and scope screenshots for each software stage.

Use the form at the end of this document for every stage. Enter `UNVERIFIED` until an operator captures and reviews the evidence.

## Stage 1: verify AO zero

1. Connect the NI AO channel to a scope input or a rated dummy load.
2. Run the AO-zero stage explicitly:

   ```powershell
   .\.venv\Scripts\python.exe tools\ni_staged_controller_test.py --stage ao-zero --execute --duration-seconds 10
   ```

   The command creates a non-regenerating 20 kHz AO task and keeps refilling zeros. Do not run it with AO connected to an isolator or preparation.
3. Start the application in control mode with feedback disarmed:

   ```powershell
   .\.venv\Scripts\python.exe echoChamber.py --hardware-profile hardware_profiles\lab_pending_test.json --start-paused --no-browser
   ```

4. Measure AO after startup, after stop, after a simulated fault, and after closing the application.
5. Record the scope range, probe attenuation, measured offset, peak-to-peak voltage, and time to reach the lab's zero tolerance.
6. Stop the application and confirm the output returns to zero.

**Pass:** The scope shows zero within the lab's signed tolerance in all four conditions. Record that tolerance before the test.

## Stage 2: scope the programmed waveform

1. Keep the output connected to the scope or dummy load.
2. Run the precomputed changing-waveform stage at 0.1 V maximum:

   ```powershell
   .\.venv\Scripts\python.exe tools\ni_staged_controller_test.py --stage ao-waveform --execute --duration-seconds 10 --amplitude-v 0.1
   ```

3. Measure onset, half-cycle duration, peak voltage, and the return-to-zero delay.
4. Stop the staged test and record AO zero after the timed task closes.
5. Use the local UI to start acquisition, choose a controller, enter closed-loop mode, select the demo pulse mode, and arm only after calibration and output drain complete.
6. Capture at least ten scope events at a sample rate that resolves the 100 Hz sine and its 5 ms duration.
7. Measure onset, half-cycle duration, peak voltage, inter-event interval, and the delay from disarm or fault until AO reaches the lab's zero tolerance. A waveform tail already committed to AO can finish during the drain barrier.

**Pass:** Each new scope event matches the software command within the lab's signed amplitude and timing tolerances. Record any committed tail after disarm or fault. Do not infer current, charge, or biological effect.

## Stage 3: measure AI conversion and channel mapping

1. Apply a known, isolated calibration signal to one input at a time.
2. Confirm that the physical CA3 input appears on the channel labeled CA3 and that EC appears on the Cortex role.
3. Record the MultiClamp display setting, expected gain, measured volts at the DAQ, and reported electrode mV.
4. Repeat the measurement at two amplitudes and at the second channel.
5. Resolve the reported gain of 10 versus 100 mV/mV with the amplifier operator before enabling a controller.

**Pass:** The measured live conversion and channel labels agree with the calibrated values recorded in the hardware profile. Imported ABF values remain in their recorded mV scale and receive no second gain conversion.

Before adding the controller, run synchronized AI with zero AO into a scope or rated dummy load:

```powershell
.\.venv\Scripts\python.exe tools\ni_staged_controller_test.py --stage ai-zero --execute --duration-seconds 10 --ai ai0 ai1 --ao ao0
```

The AO task follows the AI sample clock, uses a start trigger from AI, and disables regeneration. This stage checks synchronization while AI is read and AO is refilled with zeros.

## Stage 4: check AI and AO together

1. Keep the AO load disconnected from the preparation.
2. Record AI while the scope monitors AO.
3. Enable a low-rate controller command with the synthetic demo PID preset only when the previous stages pass.
4. Compare the AI sample clock, AO onset, scheduler target index, and scope onset.
5. Check that each command starts at the configured lead and that a late command becomes a zero substitute.

**Pass:** Every software target maps to the measured scope onset within the lab's signed tolerance. The run has no unaccounted command, late block, or unexplained sample gap.

## Stage 5: add recording and the UI

1. Start HDF5 recording and open the browser UI.
2. Repeat the simultaneous AI and AO check with rule, PID, and ESN selected in separate runs.
3. Confirm that the recording contains source sample indices, calibrated input mV, AO command V, controller name, configuration generation, sparse action stages, and recorder completion metadata.
4. Compare the UI sample index and diagnostics with the HDF5 block records.
5. Review the JSON and CSV summaries from the software benchmark before starting a physical paced run.

**Pass:** The UI remains responsive, the recorder drains without errors, and the output returns to zero on stop, disarm, or fault.

## Stage 6: calibrate the output conversion

1. Keep the preparation disconnected.
2. Use a scope and a rated dummy load to measure each AO command level across the intended operating range.
3. Measure the isolator output separately only after the isolator is approved for the dummy-load setup.
4. Record the complete conversion, load, measurement uncertainty, and calibration date.
5. Update the hardware profile only after a second operator reviews the evidence.

**Pass:** The measured AO and isolator conversions match the profile within documented uncertainty. Treat software voltage-area as command evidence only. Report charge only when the lab has a calibrated current measurement and pulse duration. For a voltage-driven load, measure impedance before deriving current. For a validated constant-current source, use its measured current and pulse duration.

## Run the software prerequisites

Run the full nine-case finite replay matrix and the responsive synthetic comparison before any hardware stage:

```powershell
.\.venv\Scripts\python.exe tools\validate_controllers.py --matrix
.\.venv\Scripts\python.exe tools\validate_controllers.py --simulation
```

These commands use the shared simulated AO scheduler. They do not contact NI hardware and do not count as a hardware soak.

## Stage 7: run the four-hour hardware soak

1. Connect AO only to the approved scope or dummy load. Start the real application with recording enabled through the browser UI. Use the pending profile only after the earlier calibration checks pass:

   ```powershell
   .\.venv\Scripts\python.exe echoChamber.py --controller rule --run-seconds 14400 --hardware-profile hardware_profiles\lab_pending_test.json
   ```

2. Open the browser UI and start recording. Select the approved controller, complete its baseline calibration, wait for the AO drain barrier, and arm only when the UI reports the controller calibrated and ready.
3. Keep the scope or approved dummy load connected throughout the run. Repeat the four-hour run for another controller when the lab approves it.
4. Save the application log, HDF5 recording path, scope capture, process ID, and UI status at the start.
5. Check the application telemetry at least every 15 minutes. Record the current AI and AO sample indices, scheduler queue, recorder backlog, late commands, zero substitutes, and fault state.
6. At four hours, disarm output, stop acquisition, wait for the AO drain and recorder to finish, and measure the time until the scope reaches the lab's zero tolerance.
7. Inspect the complete HDF5 file for unexplained index gaps and the application log for queue growth, AO write errors, or unaccounted late commands.

**Pass:** The NI task reports no AO underrun or write error. Every zero substitution has an explained boundary or intentional safe-output cause. The recording has no unexplained sample gap. The browser UI and recorder remain active for the entire run. Any failed or missing check blocks the next lab stage.

## Hardware evidence form

Copy one form for each stage. Do not replace `UNVERIFIED` with `PASS` without a saved measurement.

| Field | Entry |
| --- | --- |
| Stage | UNVERIFIED |
| Date and UTC time | UNVERIFIED |
| Operator and reviewer | UNVERIFIED |
| NI device, model, and serial | UNVERIFIED |
| Amplifier model and displayed gain | UNVERIFIED |
| AO load and scope setup | UNVERIFIED |
| Application Git revision and profile checksum | UNVERIFIED |
| Evidence files and recording path | UNVERIFIED |
| Measured input conversion and channel mapping | UNVERIFIED |
| Measured AO onset, duration, and peak | UNVERIFIED |
| Scheduler late, substitute, and rejection counts | UNVERIFIED |
| Recorder gaps and completion state | UNVERIFIED |
| AO zero after stop and fault | UNVERIFIED |
| Result: UNVERIFIED / PASS / FAIL | UNVERIFIED |
| Reviewer notes and sign-off | UNVERIFIED |

## Software checks before lab use

Create the repository virtual environment and install the pinned development dependencies as described in [the README](../README.md). Then run tests in the supported Python versions:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run the software benchmark and responsive synthetic plant separately. They validate software behavior and do not replace any stage above.
