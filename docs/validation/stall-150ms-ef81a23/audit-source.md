# Injected 150 ms stall evidence

- Report: `work\theEchoChamber\work\validation_runs\acceptance-ef81a23\stall-150ms\responsive_simulation_20261029T171043Z.json`
- Revision: `ef81a2348bbf927da10da086a10c555359616f16`
- This is a deliberate software fault-injection case, not a runtime-acceptance pass and not physical output evidence.

## Expected fault behavior

- exactly one 150ms injected stall: **PASS**
- fault latched and active miss: **PASS**
- single request has no commit or nonzero generation: **PASS**
- faulted request has rejection and cancellation without commit: **PASS**
- no plant effect: **PASS**
- no rate debt or postfault pulse burst: **PASS**
- final drain and empty queue: **PASS**
- uninjected control full 320k and faultfree: **PASS**
- both h5 indices and metadata consistent: **PASS**

The stimulated run made one pulse proposal (`g2-p1`) at source 249,999 for AO target 251,999. H5 records `rendered_for_ao`, then fault rejection and AO cancellation; no AO commit ID or sample commit followed. The recorded AO waveform contains zero nonzero samples, no pulse effect was applied, and final actuator phase returned to zero.
The injected run recorded 250,000/320,000 samples before the fault. The control recorded 320,000/320,000 samples with no injected stall or AO fault. Both finite replay loops processed their complete 320,000-sample inputs; the faulted H5 is only a contiguous prefix.

## AO accounting
The faulted H5 records 15 safe-zero AO ranges covering 73,700 AO samples. These absolute AO ranges are pipeline output history; they do not extend the recorded AI/AO sample arrays past the 250,000-sample source prefix.

Both reports keep `physical_pulse_claims=false`.