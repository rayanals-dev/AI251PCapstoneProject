# P25: UAVDT site-level analysis (Track 1 classifier)

**What this is:** a post-hoc breakdown of the **saved** test predictions (`2026-10-23_TestPredictions_v1.csv.gz`, model `fullFT` lr 0.002, raw, seeds 42/43/44). No new inference ran on the test set, and no model or threshold choice was changed. The script is `scripts/p25_uavdt_site_analysis.py`.

**Site definition:** the first 3 characters of the UAVDT sequence name (M0206 -> M02), so 14 sites (M01-M14).

## Site overlap in the current split (crops per site)

| Site | train | val | test |
|---|---|---|---|
| M02 | 2114 | 443 | 332 |
| M04 | 948 | 416 | 508 |
| M06 | 3093 | 520 | 893 |
| M08 | 0 | 0 | 925 |
| M10 | 2490 | 1658 | 0 |
| M13 | 1923 | 1036 | 1324 |
| M01, M03, M05, M07, M09, M11, M12, M14 | train only | - | - |

4 of the 5 test sites (M02, M04, M06, M13) also appear in train, which is 3,057 of 3,982 test crops (77%). Only **M08** is unseen (2 sequences, 925 crops). All 5 val sites also appear in train. This confirms the Phase 1 site-leakage finding.

## Results (macro-F1 over car/truck/bus, mean of 3 seeds)

| Group | Crops | Seqs | macro-F1 | Acc | F1 car | F1 truck | F1 bus |
|---|---|---|---|---|---|---|---|
| UAVDT test (all) | 3982 | 9 | 0.545 | 0.847 | 0.917 | 0.452 | 0.266 |
| Seen sites | 3057 | 7 | 0.464 | 0.815 | 0.899 | 0.370 | 0.124 |
| **Unseen site (M08)** | 925 | 2 | **0.876** | 0.953 | 0.974 | 0.744 | 0.911 |
| M02 (seen) | 332 | 3 | 0.647 | 0.821 | 0.903 | - | 0.391 |
| M04 (seen) | 508 | 1 | 0.739 | 0.894 | 0.940 | 0.537 | - |
| M06 (seen) | 893 | 1 | 0.289 | 0.619 | 0.770 | 0.096 | 0.000 |
| M13 (seen) | 1324 | 2 | 0.689 | 0.916 | 0.957 | 0.422 | - |

Sequence-level bootstrap, 95% CI (2,000 resamples): all UAVDT 0.398-0.776; seen sites 0.338-0.634. There is no CI for the unseen group because it has only 1 site and 2 sequences.

## Interpretation

- The leakage is real in the split, but in this data it **does not inflate** the UAVDT number: the only clean site scores higher than the seen sites.
- This **does not prove** that leakage is harmless. The clean group is a single site with 2 sequences and only 25 bus crops. Differences in difficulty between sites dominate: M06 alone drops to 0.289, and its bus F1 is 0.
- The honest claim is: "We found no evidence that site overlap inflated the reported UAVDT score. The current split cannot measure generalisation to unseen sites. The fix is a site-grouped split (planned grouped CV)."
- The metric here is 3-class macro-F1 on the UAVDT subset. A prediction of a sea class counts as an error. The number is not directly comparable to the headline 7-class test macro-F1.
