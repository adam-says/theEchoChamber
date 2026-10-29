# Final paced 30-minute evidence audit

Source revision: `c16c0eee0174c7c016d05a5ce617186a3c4c42d8`. Provenance H5 and ABF hashes are recorded in `paced_30min_evidence.json`.
All three reports are full-intent paced cases. Each report classifies full acceptance as **FAIL** because the unchanged strict timing gate is not met; this bundle does not combine them into a passing collection.

| Controller | Recorded samples / 36M | p99 ms | Max ms | Deadline misses | AO miss / late | Browser disconnects | Pulse requests / commits / nonzero samples |
|---|---:|---:|---:|---:|---:|---:|---:|
| rule | 36,000,000 | 1.499801000000001 | 35.1682 | 49 | 0 / 0 | 0 | 2229 / 158000 samples / 156420 samples |
| pid | 36,000,000 | 1.3990020000000019 | 27.5681 | 46 | 0 / 0 | 0 | 0 / 0 samples / 0 samples |
| esn | 5,564,500 | 1.312801000000001 | 53.3083 | 17 | 1 / 0 | 0 | 81 / 8100 samples / 8019 samples |

## Recording and event verification

All three H5 files were checked in bounded chunks for report/controller/source metadata, v5 schema, committed extent, shapes, every block offset/AI index/sample count, controller diagnostic offsets, generation IDs, per-block feature count bounds, source discontinuities, AO command nonzero sample counts, pulse-ID partitions, and committed AO pulse ranges. Canonical H5 source checksum: `282827974028af3f22375fba78cc3d969e228e39e5de23458b0a9ddccb44422b`; underlying ABF checksum: `eaef7545161022a46bd0b07360a7bbad90f977e8d0938347e4252f0a49f387af`.

## Browser and writer health

The per-case metrics and progress logs preserve browser connection/disconnect state, progress accepted/written/dropped/pending counters, scheduler drain, and first AO fault events. See JSON for exact records.

No claim of physical output is made.
