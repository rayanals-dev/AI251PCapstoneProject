# Track 1 results: aerial vehicle and vessel classification (Phase 1)

The selected model is **ResNet18 (ImageNet), fully fine-tuned at lr 0.002, on 64 px crops**. It was chosen on the validation set before the test set was opened. The test set was opened once, on 23 Oct 2026, and every opening is logged in `TestLog.csv`.

Every number below is the mean ± std over 3 seeds (42, 43, 44). The Adjusted column applies a per-class logit bias. That bias is fitted on val and applied to test unchanged.

## Test (13,551 crops: 4 SeaDronesSee videos, 9 UAVDT sequences)

| Model | Macro-F1 (raw) | Macro-F1 (adjusted) | Accuracy | ECE |
|---|---|---|---|---|
| **M4 Full fine-tune, lr 0.002 (selected)** | **0.733 ± 0.020** | **0.738 ± 0.025** | 0.934 ± 0.003 | 0.020 ± 0.009 |
| M4 Full fine-tune, lr 0.001 | 0.719 ± 0.008 | 0.738 ± 0.004 | 0.932 ± 0.003 | 0.024 ± 0.007 |
| M2 SmallCNN from scratch | 0.702 ± 0.020 | 0.717 ± 0.018 | 0.892 ± 0.012 | 0.020 ± 0.006 |
| M3 ResNet18 frozen | 0.652 ± 0.002 | 0.660 ± 0.005 | 0.897 ± 0.002 | 0.079 ± 0.002 |

The test set has only 13 independent groups, so the raw macro-F1 of the selected model has a wide 95% group-bootstrap CI: **0.570–0.837**. With that width, the ranking between M4 and M2 is suggestive, not conclusive.

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

## Files

- `2026-10-23_Track1Comparison_v2.csv`: the table above, with the sea and urban subsets and the per-model numbers.
- Configs to reproduce each model: `configs/cls_full.yaml`, `configs/cls_scratch.yaml`, `configs/cls_frozen.yaml`.
- Reproducibility check: a second team member re-ran scratch seed 42 from the shared notebook and got val macro-F1 0.778332, identical to the original run.
- Data, crops and checkpoints are not in this repo. They stay on the team Drive.
