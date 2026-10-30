# Full-recording ESN threshold parity

The incremental rolling-threshold implementation matched the previous NumPy
reference on all three 30-minute ESN recordings from baseline `6cc4035`. Each
run processed 360,000 blocks and 36,000,000 samples. All event decisions and
trigger sample indices matched exactly.

| Recording | Triggers | Trigger-index SHA-256 | Max threshold error | Fallbacks | Rebuilds |
|---|---:|---|---:|---:|---:|
| `2026_05_29_0001` | 482 | `4e9b3fc964fe16fadb60d7852f7f2a42e149fee23b1122a7fdd54f03de70531a` | 3.410605e-13 | 1,979 | 23 |
| `2026_08_05_0000` | 318 | `63e6e7ff34394e00ecf27e4379c94a82c3e0d2abf1fc8e511c82a3a0448ddbb1` | 5.400125e-13 | 1,979 | 29 |
| `2026_08_06_0000` | 513 | `7cef9475d1c6e83064a24f71810cba4154fc8a6fa7b9d5e70117a3d8204503bb` | 7.212009e-13 | 1,979 | 23 |

The threshold comparison covered 359,798 blocks per recording. It used the
tolerance `max(2e-12, 4 * abs(spacing(old_numpy_threshold)))`. All three maximum
errors were below the 2e-12 floor. Each run also reconstructed 359,998 stored
float32 model means and 359,798 float32 thresholds with zero mismatches.

This is algorithm-correctness evidence for replayed float32 input. The HDF5
files do not store full model predictions, and this result does not prove
bitwise reproduction of live float64 DAQ input. It is not a real-time timing or
hardware acceptance result. Final runtime pulse-lifecycle validation remains
pending during the Astra review.

## Reproduce

Run the probe without `--limit-seconds` to process all committed samples:

```powershell
$recordings = '..\..\work\theEchoChamber\work\validation_runs\acceptance-6cc4035\full-matrix\recordings'
python tools\compare_esn_thresholds.py `
  --artifact artifacts\esn_corrected_v1.npz `
  --json-output work\validation_logs\esn_full_parity\rerun.json `
  "$recordings\20261028_231556_20261028T221556Z_2026_05_29_0001_esn_100ms.h5" `
  "$recordings\20261028_233842_20261028T223842Z_2026_08_05_0000_esn_100ms.h5" `
  "$recordings\20261029_000231_20261028T230231Z_2026_08_06_0000_esn_100ms.h5"
```

The source manifest and three per-recording JSON results are included beside
this summary. The manifest records the SHA-256 hashes before and after the
probe. All six hashes were unchanged.

| Input or code file | SHA-256 |
|---|---|
| `2026_05_29_0001` recording | `a791c6a548de1eb7a4f27920e01b7163a868fc0a352a05835df870871d52a3e2` |
| `2026_08_05_0000` recording | `c3a268ee086ecd1b0dc94f069284dafa5d517d9a60400f75430131c943112ae7` |
| `2026_08_06_0000` recording | `eb5c3e882c98da005c504069657d07b2c4e7c7c22ff6a34ec3a3080b96df03c3` |
| `tools/compare_esn_thresholds.py` | `1de7fa763fe15c13c5bcefcea462fb829d0d23a334dde158d5cdc32b29f6e760` |
| `esn_bridge.py` | `f6147a59cc0e754e8fdcce53bb62999214daceed0de1bce24ce893e49f5d9644` |
| `artifacts/esn_corrected_v1.npz` | `4908a423689067669f4b18e45948027e8577278b8def1fcae38c107dd5aac103` |
