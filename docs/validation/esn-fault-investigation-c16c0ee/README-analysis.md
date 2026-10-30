# ESN paced AO-fault timeline investigation

This is a bounded read-only analysis of retained run artifacts. It does not reread raw AI/electrode samples, recalculate full-file hashes, or classify simulated AO commands as physical output. The extractor snapshot is [analyze_paced_fault_timeline.py](extractor/analyze_paced_fault_timeline.py); its JSON output is [bounded extraction data](esn-paced-fault-analysis-c16c0ee-v2.json).

## Current c16c0ee ESN run

The 30-minute ESN case is full-intent failed: 36,000,000 source samples were processed, but the recording contains a clean 5,564,500-sample prefix. It stopped recording after one simulated AO active-demand miss. The closed H5 has 55,645 block and controller rows, no source discontinuities, and retains the source block at 5,564,400 with target 5,566,400.

The fault snapshot is a post-advance observation (`observation_is_fault_onset=false`), not an atomic capture of the miss decision. It records source block 5,564,400 in progress (last completed block 5,564,300), source-availability lag 0.298 ms, and AO counter 5,565,400. The expected target 5,566,400 was 1,000 samples / 50 ms ahead of that counter. In the final H5, the first fault-zero interval is `[5,566,400, 5,567,400)`, matching that source block’s target and spanning 50 ms. The snapshot’s post-service write frontier is 5,567,400, or 100 ms ahead of the sampled AO counter; these counters were sampled sequentially and must not be treated as a simultaneous queue snapshot.

The same source block later logged a 53.3083 ms Core call. Its slow-call record was written about 7.67 ms after the AO-fault observation, consistent with the call still being in progress when the fault was observed. All 17 deadline misses have corresponding `slow_core_process` records. The log cap is 20, so the list is complete for this run. The UI timer wraps `core._publish_ui`, which `ProcessingCore.process` calls once synchronously before returning. The largest UI-submission time was 52.1041 ms. Only one Core call was at least that long: source 5,564,400 at 53.3083 ms. Under the complete-log and one-call-per-block premises, the UI-submission maximum therefore belongs to this same source block and accounts for at least 52.1041 ms of its 53.3083 ms Core interval. Its stored thread-CPU and process-CPU deltas are both 0.0 ms.

Under those premises, the faulting source block’s `_publish_ui` interval reached the aggregate UI maximum. This correlation narrows where that interval occurred, but does not establish why it took 52.1041 ms. The inspected method appends display data to its local ring and enqueues a `UiJob` with nonblocking queue operations. Packet serialization and WebSocket work run in the separate UI worker. The evidence does not support attributing the interval to a 52 ms socket wait. Thread preemption, GIL scheduling, queue-lock contention, or local display preparation remain possible. GC did not overlap the saved slow Core call.

At the AO-fault observation, the current service gap was 7.4435 ms and the largest AO service gap observed so far was 25.1181 ms. The final report recorded 115,697 service gaps longer than one 5 ms block over the run. The observed gap is not the fault onset and cannot establish the host’s exact state at the miss.

## Earlier comparison points

The earlier 6657c888 ESN paced run also failed with one active-demand miss. Its post-advance snapshot marks source 21,606,600 in progress; the current AO counter was 21,607,600 and the expected target was 21,608,600. Its H5’s first fault-zero interval starts at 21,608,400, two source blocks before that snapshot marker’s expected target. The snapshot had 59.2266 ms source-availability lag; the bounded recent Core history showed a maximum lag of 64.3266 ms at source 21,606,400, while its largest Core call in that 32-row window was 2.0252 ms. It also reports a 69.4154 ms maximum AO service gap over the run, but the gap is not timestamp-linked to the miss. This run is consistent with late source processing and host scheduling jitter; it does not prove which caused the miss.

The short responsive simulations show that AO host faults were not exclusive to ESN: revision 2a3a1fb had a PID no-stimulation control fault with 273,100 of 480,000 samples recorded and a 211.9717 ms maximum call; revision ef81a23 had an ESN stimulated fault with 136,800 of 480,000 samples recorded and a 302.4315 ms maximum call. These are different short synthetic runs with different fault text (`simulated AO host refill stalled past committed output`) and are not direct replications of the paced fault.

In the c16c0ee paced set, Rule and PID completed their recordings without an AO miss; ESN did not. PID had 80,123 positive-rate blocks but no pulse proposals or committed/nonzero samples. These three runs do not establish an ESN-specific cause.

## What the artifacts support

The repeated fact is a missing active-generation AO command at the scheduler’s safety refill point. That is a software safety fault for a missing future command; it does not prove that the physical device experienced an output underrun. In c16c0ee, the same-index Core call took longer than 50 ms, and the missing command’s target was 50 ms beyond the sampled AO counter. In 6657c888, the source producer was substantially behind the source-availability schedule and the run also had large AO service gaps. The first snapshot is not the fault-onset record, and the aggregate component timings cannot identify a causal subsystem. OS scheduling, producer backlog, UI submission blocking, and AO-service delay remain candidates; none is established as the sole cause.

Per the 2026-10-30 direction, the 5 ms processing interval is now a diagnostic measure in this investigation, not a release-pass claim. This evidence does not justify changing the 100 ms output lead, 50 ms AO reserve, or safety behavior. Historical report classifications remain as recorded. All H5 `complete` flags mean the saved file closed cleanly; they do not mean the requested 30-minute source was fully recorded. All output records continue to state `physical_pulse_claims: false`.

Runtime is preparing a deterministic scheduler-boundary reproduction and source-indexed phase probe. Those checks are still pending; this package contains no result from them.

## Next diagnostic capture

For a short controlled rerun, capture a bounded record at the exact scheduler decision that increments `active_demand_misses`: AO generated counter, first missing target, contiguous available command prefix, queued samples/time, configured deadline threshold, refill size, source index and availability lag, active Core index and elapsed time, scheduler service interval, and the resulting zero-range bounds. Add source-indexed component timings only for slow Core calls, retaining wall, thread-CPU, process-CPU, and GC-overlap values. This should separate slow producer calls from AO-thread service gaps without inferring onset from a later snapshot.

Use two deterministic software checks independently: delay a source Core call while the AO consumer continues on time, then delay AO host service while the producer remains fast. Preserve the current gate and verify fault/zero behavior. Do not run another long paced case until these diagnostics are reviewed.
