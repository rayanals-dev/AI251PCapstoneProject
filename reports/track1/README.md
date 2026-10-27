# Track 1 results: aerial vehicle and vessel classification (Phase 1)

The selected model is **ResNet18 (ImageNet), fully fine-tuned at lr 0.002, on 64 px crops**. It was chosen on the validation set before the test set was opened. Every opening of the test set is logged in `TestLog.csv`: once on 23 Oct for the scores below, and once on 25 Oct, report-only, to add top-k accuracy (same checkpoints, nothing tuned).

**Headline test score: raw macro-F1 0.679 ± 0.013 without the SeaDronesSee `rgb` stills (group-bootstrap 95% CI 0.523–0.874).** The `rgb` stills share a drone (`trinity`) with a train group, so the full-test score (0.733) is reported as secondary. See the sensitivity section below.

Every number below is the mean ± std over 3 seeds (42, 43, 44). The Adjusted column applies a per-class logit bias. That bias is fitted on val and applied to test unchanged.

## Test (13,551 crops: 4 SeaDronesSee videos, 9 UAVDT sequences)

| Model | Macro-F1 (raw) | Macro-F1 (adjusted) | Accuracy | ECE |
|---|---|---|---|---|
| **M4 Full fine-tune, lr 0.002 (selected)** | **0.733 ± 0.020** | **0.738 ± 0.025** | 0.934 ± 0.003 | 0.020 ± 0.009 |
| M4 Full fine-tune, lr 0.001 | 0.719 ± 0.008 | 0.738 ± 0.004 | 0.932 ± 0.003 | 0.024 ± 0.007 |
| M2 SmallCNN from scratch | 0.702 ± 0.020 | 0.717 ± 0.018 | 0.892 ± 0.012 | 0.020 ± 0.006 |
| M3 ResNet18 frozen | 0.652 ± 0.002 | 0.660 ± 0.005 | 0.897 ± 0.002 | 0.079 ± 0.002 |

The test set has only 13 independent groups, so the raw macro-F1 of the selected model has a wide 95% group-bootstrap CI: **0.570–0.837**. With that width, the ranking between M4 and M2 is suggestive, not conclusive.

### Sensitivity: test without the SeaDronesSee `rgb` stills

41% of the test crops (5,549) come from one SeaDronesSee group, `rgb`. It holds 20 MP still photos, a capture type that appears nowhere in train or val, and it shares no images with them (checked 24 Oct, see `data/README.md`). These stills turn out to be the easiest part of the test, so we also report the test score without them.

This table was computed from the saved Step 8 predictions. The models were not re-run, and nothing was tuned on test.

| Model (raw macro-F1, 3 seeds) | All test | Test without `rgb` | Sea: `rgb` stills only | Sea: test videos only |
|---|---|---|---|---|
| **M4 Full fine-tune, lr 0.002** | 0.733 | **0.679** | 0.887 | 0.770 |
| M4 Full fine-tune, lr 0.001 | 0.719 | 0.666 | 0.885 | 0.770 |
| M2 SmallCNN from scratch | 0.702 | 0.655 | 0.863 | 0.797 |
| M3 ResNet18 frozen | 0.652 | 0.635 | 0.775 | 0.753 |

- Without `rgb` the test has 12 groups. The selected model's 95% group-bootstrap CI is 0.523–0.874 (`2026-10-25_TestNoRGB_v1.csv`, same bootstrap as Step 8, which reproduces 0.570–0.837 on the full test).
- `rgb` (1,338 stills, test) and `multispectral` (475 frames, train) were taken by the same drone (`trinity`) and carry no timestamps. No image is shared, but a same-session overlap cannot be ruled out. That is why the headline excludes `rgb`.
- Without `rgb`, jetski, life-saving appliances and buoy each come from a single test video. Life-saving appliances drop from F1 0.40 to 0.00 on all three seeds.
- The model ranking is the same with or without `rgb`, but the intervals overlap, so the differences between models are not statistically confirmed on test.
- On sea video frames alone, the from-scratch SmallCNN matches or beats full fine-tuning (0.797 vs 0.770 raw, 0.824 vs 0.777 adjusted). The ImageNet advantage at sea comes mainly from the high-resolution stills.

### Top-k accuracy and calibration (report-only opening, 25 Oct)

`step8b_topk_report` re-ran the 12 checkpoints listed in `TestLog.csv`. Every argmax matched the 23 Oct predictions before any metric was computed. Raw logits, mean of 3 seeds (`2026-10-25_TestTopK_v1.csv`):

| Model | Case | Top-1 | Top-2 | Top-3 | ECE |
|---|---|---|---|---|---|
| **M4 Full fine-tune, lr 0.002** | All test | 0.934 | 0.968 | 0.978 | 0.020 |
| **M4 Full fine-tune, lr 0.002** | Without `rgb` | **0.909** | **0.954** | **0.968** | **0.023** |
| M4 Full fine-tune, lr 0.001 | Without `rgb` | 0.904 | 0.951 | 0.967 | 0.026 |
| M2 SmallCNN from scratch | Without `rgb` | 0.853 | 0.913 | 0.939 | 0.032 |
| M3 ResNet18 frozen | Without `rgb` | 0.882 | 0.952 | 0.973 | 0.078 |

- The largest top-2 gain is truck predicted as car. A substantial share are cross-domain confusions of tiny objects (car ↔ swimmer, car → boat).
- The logit bias raises macro-F1 only within noise, but lowers top-1 (0.934 → 0.927) and worsens ECE (0.020 → 0.045). Raw scores are the headline.

## Validation (used for every choice)

| Model | Macro-F1 (raw) | Macro-F1 (adjusted) |
|---|---|---|
| M4 Full fine-tune, lr 0.002 | 0.817 ± 0.006 | 0.833 ± 0.010 |
| M4 Full fine-tune, lr 0.001 | 0.802 ± 0.009 | 0.825 ± 0.009 |
| M2 SmallCNN from scratch | 0.779 ± 0.011 | 0.811 ± 0.005 |
| M3 ResNet18 frozen | 0.707 ± 0.001 | 0.719 ± 0.003 |

**Selection rule**, fixed before seeds 43 and 44 were run: switch the learning rate only if the new one beats the default by more than the seed std (0.009). lr 0.002 beat 0.001 on all three seeds.

## Where the model fails

- **Bus (test F1 0.26):** 160 of the 249 test buses are smaller than 32 px, and recall on them is 0.
- **Truck (F1 0.45) and life-saving appliances (F1 0.40)** are the other weak classes. The remaining five classes score 0.91–0.98.
- **One sequence dominates the errors:** UAVDT M0602 holds 435 of the 1,077 errors.
- **Some errors are label noise, not model mistakes:** box trucks in M1303 are labelled car, and a billboard in M0602 is labelled truck.

## Held-out mountain road (Al-Hada, 26 Oct)

300 frames from two iPhone clips filmed from public viewpoints above a mountain road, 9,484 vehicle crops (8,885 car / 543 truck / 56 bus). Never used for training or tuning; the four models were run once, raw, and a predicted sea class counts as an error. Macro-F1 over car, truck and bus, mean ± std over 3 seeds:

| Model | Al-Hada macro-F1 |
|---|---|
| M4 Full fine-tune, lr 0.002 (selected) | 0.334 ± 0.013 |
| M3 ResNet18 frozen | 0.350 ± 0.001 |
| M4 Full fine-tune, lr 0.001 | 0.319 ± 0.002 |
| M2 SmallCNN from scratch | 0.312 ± 0.010 |

Cars transfer (F1 0.94); trucks (0.06) and buses (0.00) do not: 90% of trucks and 82% of buses are predicted car. The main cause is a class-definition shift: Al-Hada trucks are mostly pickups and its buses Hiace-type vans, which UAVDT barely contains. Details in `data/DATASET_CARD.md`.

## Files

- `2026-10-23_Track1Comparison_v2.csv`: the test table above, with the sea and urban subsets and the per-model numbers.
- `2026-10-26_WandbRuns_v1.csv`: export of every Track 1 W&B run (config and summary metrics). The W&B project itself is private (plan restriction); a read-only public report with the learning curves and run tables is linked in the main README.
- Configs: `configs/cls_scratch.yaml`, `configs/cls_frozen.yaml`, `configs/cls_full.yaml` (selected model), `configs/cls_full_lrcheck.yaml` (LR check, 23 Oct). They are the exact files the training notebook writes.
- Notebooks and cells in `notebooks/track1/`:
  - `train_cls_2026-10-25.ipynb`: training for all three models (pick the config in section 0). `VERIFY_ONLY = True` reloads the finished runs without training, logging or reading test.
  - `step8b_topk_report_2026-10-25.ipynb`: the report-only second test opening (top-k, ECE); checks every argmax against the 23 Oct predictions first.
  - `test_norgb_cell_2026-10-25.py`: test without `rgb` and its group-bootstrap CI, from the saved predictions only.
  - `rebuild_full_s42_s44_cell_2026-10-25.py`: rebuilt two finished-run summary files of cls-full from the saved checkpoint, history and val predictions (no training, no test).
  - `alhada_eval_v2_cell_2026-10-26.py`: the Al-Hada evaluation.
  - `report_figures_cell_2026-10-26.py`, `error_gallery_v2_cell_2026-10-26.py`: the report figures, from saved files only.
  - The original Step 8 test notebook is kept on the team Drive and is not re-run.
- `figures/`: the four report figures (learning curves, confusion matrix, reliability diagram, error gallery) as PDF.
- Every Drive path is set in one place (`DRIVE_PROJECT` or `R` at the top). Change it if your Drive layout differs.
- Reproducibility check: from a clean clone of this commit, `train_cls` with `CONFIG_NAME = 'cls_scratch'` and `VERIFY_ONLY = True` must reload the finished runs and print val macro-F1 0.778332 for seed 42. Result and commit hash: see the Phase 1 report, Section 4.
- Data, crops, checkpoints and the Al-Hada frames are not in this repo. They stay on the team Drive.
