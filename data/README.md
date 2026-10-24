# data/: splits and audits (no images)

This folder holds only the metadata needed to rebuild the Track 1 data: the split definitions, the per-group counts, the cleaning log and the label audits. Images, crops and videos stay on the team Drive. `.gitignore` blocks them.

## splits/

| File | What it is |
|---|---|
| `2026-10-19_Splits_v1.json` | **The split.** Seed 42, target 70/15/15. `video_to_split` covers SeaDronesSee and `seq_to_split` covers UAVDT. Detection and segmentation must use the same mapping, so that no flight or sequence lands in two splits across tracks. |
| `2026-10-19_SplitClassTable_v1.csv` | Crops per class in each split, with the share and the number of groups. Totals: train 66,925, val 13,800, test 13,551. |
| `2026-10-18_SDSGroups_v1.csv` | SeaDronesSee groups (one per video) with class counts, image count and median altitude. |
| `2026-10-18_UAVDTGroups_v1.csv` | UAVDT groups (one per sequence) with box and track counts per class. |

**How the split was made:** by group, never by image, so that consecutive frames of one video cannot appear in both train and test.

**Known limitations:**
- UAVDT is split by sequence, not by site prefix (M01…M14). Sequences that share a prefix may come from the same site, so val and test scores on UAVDT may be optimistic.
- The SeaDronesSee groups `rgb` and `multispectral` are images without a video id. Each one is treated as a single group, and `rgb` (1,338 images, 5,549 crops, 41% of test) is in **test**. A check on 24 Oct found no image-level leakage:
    - no MD5 is shared across splits;
    - the `rgb` images are 20 MP still photos (5456×3632, some portrait), while every train and val image is a video frame (3840×2160 or 1920×1080) or a multispectral frame.

  The `rgb` images have no timestamps, so a same-session overlap (the same boats on the same day) cannot be ruled out. Test results are therefore reported both with and without `rgb`. See `reports/track1/README.md`.

## audit/

| File | What it is |
|---|---|
| `2026-10-18_CleaningLog_v1.csv` | Everything removed during cleaning: 23 duplicate SeaDronesSee boxes and 2 exact-duplicate UAVDT frames (found by MD5). |
| `2026-10-19_LabelAudit_UAVDTTracks_v1.csv` | UAVDT tracks whose class flips between frames. Each track gets its majority label, and the number of relabelled boxes is recorded. |
| `2026-10-21_AuditResults_v1.csv` | A manual audit of 300 random crops (about 37 per class). Weighted label-error rate is 4.4%, with a 95% upper bound of 8.5%. |
