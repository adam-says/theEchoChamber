# Qualify biological activity regulation

This plan defines the evidence needed to claim that the application regulates a measured activity signal toward a lab-selected target in an ex vivo slice. It is a qualification plan, not a result. Hardware and biological effect remain **UNVERIFIED** until the stages below produce reviewed evidence.

The software's 5 ms maximum processing-time threshold is a diagnostic benchmark. It is not a universal biological acceptance threshold. A biological latency limit depends on the chosen signal, detector, response window, actuator, and preparation. The existing timing runs retain their recorded values and pass or fail classifications. Reinterpreting the benchmark does not clear the recorded simulated AO active-demand deadline fault or other runtime faults.

## Choose the activity measure and target

The software-supported measure for this task is the causal CA3 100–250 Hz RMS envelope, smoothed over 10 ms and normalized against the frozen valid baseline. This defines the feature computation, not the biological target or evidence of a tissue effect. The experiment lead must complete and sign the remaining fields before collecting qualification data. Do not infer missing values from the software preset or a paper.

A CA1 recording-site variant needs a separate protocol, revised feature provenance, and validation.

| Decision | Prespecified value |
| --- | --- |
| Preparation and region | TBD: species, age or source, acute or organotypic hippocampal slice. This protocol fixes the recording site to CA3. |
| Activity measure | CA3 100–250 Hz RMS envelope, causal filter, 10 ms smoothing, normalized against the frozen valid baseline. Record channel identity, filter implementation, feature window, sample rate, artifact mask, and offline reference method. |
| Target and direction | TBD: target value or range, direction of regulation, practical biological effect, and rationale for this preparation |
| Evaluation window | TBD: trial duration, response time, settling interval, and recovery interval |
| Disturbance or state change | TBD: trigger, timing, size, duration, or explicit reason to test spontaneous variation only |
| Measurement quality | TBD: valid-signal criteria, minimum baseline coverage, offline reference for envelope and artifact validation, and allowable disagreement. Add offline SWR event labels only if the claim concerns SWR events. |
| Biological limits | TBD: acceptable tracking error, response tolerance, recovery, and adverse activity limits, justified by the lab |
| Stimulation limits | TBD: calibrated current or voltage at the preparation, pulse shape, charge or other safety limits, duty cycle, and stop rules |
| Exclusions | TBD: prospective slice-quality and recording exclusions. Report every excluded slice and reason. |

## Match controller claims to controller objectives

The controllers do not share one target objective. PID uses a `target_z` setpoint on the normalized envelope, but the hardware PID preset has zero gains and therefore requests no stimulation. The nonzero PID preset is for software demonstrations, not tissue tuning. The rule controller triggers above its upper envelope threshold and rearms below its lower threshold; it does not track a target. The existing ESN artifact drives passthrough output or threshold pulses from its model output. It was not trained to track this CA3 envelope target.

The experimenters can compare each controller's effect on the same prespecified envelope outcome. That common outcome does not mean each controller implements target regulation. Before making a same-target regulation claim for rule or ESN, document the mapping from the chosen envelope target to controller settings or model behavior, then validate that mapping on the preparation. A software framework demonstration does not establish target tracking in tissue.

The completed [PID stimulation demonstration](validation/pid-stimulation-demo/README.md) used a responsive plant with a preset 150 Hz carrier. Its stimulated phase produced 32 unique pulse proposals, each matched to a committed AO range, and all three 64-second software phases completed; its separate forced-capacity reference used 67 pulses. It is not a neural model or tissue evidence. The paired [neural mock R4](validation/neural-mock/README.md) passed ten circuit-closure gates in virtual time: each Rule-off/on arm recorded 480,000 samples, the feedback arm generated 2,178 nonzero AO samples from 22 pulse requests, and the delayed model effect was followed by a later valid measurement after actual Core blanking. This checks software circuit closure, delay, response, and basic blanking. It does not show biological target tracking or real hardware latency. The lab must measure physical artifact and recovery on the rig. See the [mock neural circuit closure guide](mock_neural_test.md).

A ripple-band envelope can track fast-band energy, but band power alone does not identify an SWR. If the claim concerns SWR rate or event disruption, validate the online detections against blinded offline labels and report sensitivity, positive predictive value, false detections per minute, and detection delay. Preserve all samples and mark invalid or masked intervals explicitly. Do not count masked data as evidence that the target was reached.

## Separate software, instrument, and tissue evidence

Report these times separately for every stage that can measure them:

- **Decision time:** source sample time or index that completed the feature window, detector decision time, and software command time.
- **Electrical output time:** NI write or buffer acceptance, and independently measured output onset at the scope or calibrated monitor. Report latency, jitter distribution, missed or late outputs, extra outputs, and output identity. Software AO commitment is not physical output evidence.
- **Biological response time:** onset and trajectory of change in the prespecified activity measure after measured output. Report measurement uncertainty and the response window chosen before the experiment.

Record sample-clock alignment, filter and detector group delay, block size, acquisition buffering, NI scheduling, output monitoring resolution, and timestamp clock source. Use a shared clock or a documented synchronization check. Estimate end-to-end latency from the feature's source samples to measured electrical output. A detector can intentionally wait for evidence across several oscillation cycles; its delay is part of the biological protocol, not a substitute for output timing.

Electrical stimulation can corrupt the same recording used for feedback. Characterize the artifact at the recording electrode with the preparation disconnected or with a suitable dummy load, then with the preparation under an approved low-risk stage. Prespecify the artifact interval and recovery rule. Keep original samples. Mark masked intervals, quantify lost coverage, and inspect raw and filtered traces around every pulse. Compare online detections with offline detections on the unmasked signal. A mask that removes a failed response or event cannot count as successful regulation.

## Compare regulation with controls

Use within-slice conditions where feasible and counterbalance their order. Compare closed-loop regulation with at least a sham condition and a control that separates feedback timing from stimulation exposure. Suitable choices include an open-loop replay of the same pulse sequence or a yoked, matched-dose schedule. State which comparison answers the experiment's question before data collection.

Match or report pulse count, amplitude, duration, polarity, inter-pulse interval, and total dose. If the closed-loop and control conditions differ in delivered exposure, do not attribute the difference to feedback alone. Include a no-stimulation baseline and recovery period when the preparation supports them. Stop on prespecified tissue, signal, electrical, or runtime faults.

Treat the slice as the experimental unit for slice-level outcomes. Slices from the same animal or donor are clustered observations, not automatically independent replicates. Treat repeated windows, events, and pulses within a slice as nested observations. Choose the sample-size calculation and statistical model with the experimenters, using the animal or donor grouping where relevant. Prespecify the primary endpoint and analysis. Report per-slice trajectories and uncertainty, along with the aggregate estimate. State all exclusions and missing intervals.

For each slice and condition, report at least:

- baseline level and variation of the activity measure;
- target error over time, response latency, settling, and recovery;
- valid and masked coverage, source gaps, detector misses, and false detections;
- commanded and measured output counts, pulse parameters, and all AO faults;
- the corresponding sham, open-loop, or yoked control result;
- slice metadata, protocol deviations, and reasons for exclusions.

## Run the qualification in stages

1. **Freeze the protocol.** Complete the decision table, analysis, exclusions, control schedule, and lab-specific biological and electrical limits. Save the signed protocol with the application revision and configuration hashes.
2. **Qualify measurement offline.** Apply the feature to representative recordings and verify it against the prespecified offline reference. Check channel identity, units, filter response, baseline stability, artifact handling, and coverage before using closed loop. If the claim concerns SWR events, compare detections with blinded offline labels.
3. **Measure the software path with replay.** Run the existing replay and paced diagnostics. Preserve historical timing metrics and case status. Report runtime faults, deadline misses, and recording completeness separately from biological criteria. Replay does not prove hardware output or tissue response.
4. **Qualify the instrument with no preparation connected.** Follow the [NI staged controller protocol](ni_controller_protocol.md). Measure AO zero, waveform, onset, duration, jitter, missed and extra outputs, and the full acquisition-to-output delay at the scope or calibrated monitor. Keep every hardware result marked unverified until the lab records and reviews it.
5. **Check recording artifacts and safety on an approved dummy load.** Measure artifacts and recovery at every relevant input. Confirm masks, valid coverage, synchronized timestamps, output bounds, fault stops, and recovery after disarm. Do not route output to a preparation until the lab approves this stage.
6. **Run a small ex vivo feasibility series.** After institutional and lab approvals, verify spontaneous activity, feature validity, stimulation artifact handling, and a measurable response with the chosen preparation. Use conservative lab-approved settings and the planned controls. This checks feasibility; it does not establish efficacy.
7. **Run the prespecified within-slice comparison.** Use the signed protocol and full logging. Include sham and matched-dose or yoked control conditions. Analyze at the slice level and report all slices, failures, missing data, exclusions, and uncertainty.
8. **Make a bounded claim.** Claim regulation only if the prespecified activity measure moved toward its target within its signed response window, controls support a closed-loop effect, physical output was independently measured, and safety and data-quality limits passed. Otherwise report the measured outcome and the failed, unresolved, or unmeasured stage.

## Literature basis and limits

| Primary source | Evidence used here | Limit for this project |
| --- | --- | --- |
| Dutta, Ackermann, and Kemere, [Analysis of an open source, closed-loop, realtime system for hippocampal sharp-wave ripple disruption](https://doi.org/10.1088/1741-2552/aae90e), 2019 | Measures acquisition and algorithm contributions to ripple-detection delay, with explicit tradeoffs between delay and false detections. | In vivo detection-system characterization, not this NI device, slice preparation, actuator, or a universal latency criterion. |
| Monteiro de Sousa et al., [An open-source, ready-to-use and validated ripple detector plugin for the Open Ephys GUI](https://doi.org/10.1088/1741-2552/ac857b), 2022 | Reports detector accuracy against simulated ripple recordings and measures round-trip latency on synthetic data. | Simulation and synthetic loopback do not establish performance on this rig or biological efficacy. |
| Jiang et al., [Pacing hippocampal sharp-wave ripples with weak electric stimulation](https://doi.org/10.3389/fnins.2018.00164), 2018 | Demonstrates electrical modulation of spontaneous and evoked SWR abundance in mouse hippocampal slices; the methods address stimulation artifacts and compare activity with and without pacing. | A specific slice model and pacing protocol. Its settings and effects are not transferable acceptance thresholds for this project. |

These studies support measuring detector performance, full-loop latency, artifact effects, and activity change with controls. They do not supply the missing target, acceptable response window, safety limits, or success thresholds for this preparation. Those values remain lab decisions.
