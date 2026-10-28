# Offline ABF import and replay

The offline importer converts the supplied gap-free ABF2 recordings to a compressed, finite canonical HDF5 replay. The live application does not import `pyabf`; install the optional importer dependency separately:

```powershell
python -m pip install -r requirements-import.txt
python tools/import_abf.py ..\data\CA3_slow_waves\2026_05_29_0001.abf recordings\2026_05_29_0001.h5
```

The importer accepts ABF2 only, and validates the supplied acquisition contract: two channels, 20 kHz, 30 minutes, and stored mV units. Position 0 is labeled CA3 (physical ADC0); position 1 retains regional label EC and has Cortex role (physical ADC2). `pyABF` returns calibrated values in the ABF stored units. The importer copies those mV values as-is and records explicitly that no amplifier gain was applied.

To import the three supplied files, then reproduce the full source-to-output check:

```powershell
$source = "..\data\CA3_slow_waves"
$replays = "..\replays\CA3_slow_waves"
Get-ChildItem $source -Filter *.abf | ForEach-Object {
    python tools\import_abf.py $_.FullName (Join-Path $replays ($_.BaseName + ".h5"))
}
python tools\verify_replay_import.py --source-dir $source --replay-dir $replays `
  --json-output ..\replays\CA3_slow_waves\verification.json
```

The verifier checks every output sample against `pyABF`'s calibrated channel values, validates source hashes against the manifest, confirms mapping and unit metadata, and checks finite EOF and segment coverage. It reports both source and compressed replay hashes.

The output uses `echo-chamber-replay` schema version 1 and stores `signals/electrode_mV` as compressed float32, with sample rate, labels, roles, regions, source SHA-256, source header summary, calibration method, and continuous segment/gap metadata. It refuses to overwrite an existing file unless `--overwrite` is supplied. Source inventory and expected properties are recorded in `data/ca3_slow_waves_sources.json`; source recordings remain outside Git.

`replay.load_replay(path)` accepts canonical replay and Echo Chamber HDF5 v1-v5. Legacy v1-v4 stored volts are converted to electrode mV using their own recorded amplifier gain. V5 and canonical mV data are never rescaled. The supplied 4-AP sample's adjacent sidecar is checksum-verified and takes precedence over conflicting embedded channel labels. Recorded sample discontinuities remain segment boundaries with their absolute sample indices. `ReplayData.iter_chunks(n)` yields absolute sample starts, splits at gaps, reaches EOF once, and never loops the recording. Callers must treat EOF as finite; a new run must explicitly reopen the source.
