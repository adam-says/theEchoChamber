# ESN full-recording parity: moment-only rebase

This diagnostic compares the current ESN rolling-threshold path with the prior
NumPy reference on the three canonical 30-minute recordings. Each case covered
all 360,000 blocks and 36,000,000 samples. There was no duration limit.

All three reports passed exact fired-event and trigger-sample-index parity.
The recorded float32 model-mean and threshold reconstruction checks had zero
mismatches. Maximum threshold errors were below the unchanged
`max(2e-12, 4 * abs(spacing(reference_threshold)))` tolerance:

| Recording | Maximum threshold error |
|---|---:|
| May 29, 2026 | `3.410605e-13` |
| Aug 5, 2026 | `5.400125e-13` |
| Aug 6, 2026 | `7.212009e-13` |

The candidate bridge SHA-256 is
`88469989b45429ec270f55e37139c96a1565ae91193dd2113c7b332d4e9c4749`. The
moment-only rebase leaves the ESN model recurrence and numeric model artifact
unchanged. The artifact SHA-256 is
`4908a423689067669f4b18e45948027e8577278b8def1fcae38c107dd5aac103`.

The reconstruction checks use the captured float32 input and stored per-block
means and thresholds; the HDF5 recordings do not contain full model
predictions. This does not establish bitwise replay of the original live
float64 DAQ input. These parity results are not timing or hardware acceptance
evidence. The earlier `f6147a59` bridge bundle remains available at
[`../esn-full-parity/`](../esn-full-parity/).

## Reports

- [May 29, 2026](20261028_231556_20261028T221556Z_2026_05_29_0001_esn_100ms.json)
- [Aug 5, 2026](20261028_233842_20261028T223842Z_2026_08_05_0000_esn_100ms.json)
- [Aug 6, 2026](20261029_000231_20261028T230231Z_2026_08_06_0000_esn_100ms.json)
- [Run manifest](manifest.json)
