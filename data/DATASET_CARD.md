# Track 1 Dataset Card

Abdulmohsen Alharthi (Track 1, Data & Classification) · last updated 27 Oct 2026 · Phase 1

This card describes the data used for Track 1 (aerial object classification, Phase 1): two public datasets cut into object crops, plus a private mountain-road test set from Al-Hada.

## Sources and licences

| Dataset | Domain | What we used | Licence | Source |
| --- | --- | --- | --- | --- |
| SeaDronesSee, Object Detection v2 | Sea (search and rescue) | 39 video groups (flights) plus 2 still-image groups; train and val annotations (14,227 images on disk) | CC0 | [seadronessee.cs.uni-tuebingen.de](https://seadronessee.cs.uni-tuebingen.de/dataset) |
| UAVDT Benchmark-M | Urban roads | 50 sequences, 40,735 frames, sampled at 5 fps; per-sequence attributes (weather, altitude, view) | Research use, per authors | [UAVDT project page](https://sites.google.com/view/grli-uavdt/) |
| Al-Hada road (ours) | Mountain road | 2 iPhone clips from public viewpoints above the road, 312 frames at 1 fps; held-out test only, never trained on | Private, owner only | Not public |

The proposal quoted 100 UAVDT sequences and 80K frames (the full benchmark). We use Benchmark-M only: 50 sequences, 40,735 frames.

## Classes and composition

Each annotated box becomes one crop, stored at 96 px and trained at 64 px. That gives eight classes in two domains and 94,276 crops in total.

| Class | Domain | Train | Val | Test | Test groups |
| --- | --- | --- | --- | --- | --- |
| swimmer | Sea | 32,238 | 6,093 | 4,955 | 4 |
| boat | Sea | 9,707 | 2,319 | 3,206 | 4 |
| buoy | Sea | 3,476 | 659 | 810 | 2 |
| jetski | Sea | 1,793 | 443 | 414 | 2 |
| life\_saving\_appliances | Sea | 856 | 213 | 184 | 2 |
| car | Urban | 15,969 | 3,383 | 3,388 | 9 |
| truck | Urban | 1,807 | 398 | 345 | 5 |
| bus | Urban | 1,079 | 292 | 249 | 4 |
| **Total** |  | **66,925** | **13,800** | **13,551** | **13** |

Shares are 71.0 / 14.6 / 14.4% against a 70/15/15 target.

**UAVDT crops are capped per track:** at most 10 crops for each car track and 40 for each truck or bus track. One long car track would otherwise flood the set; without the cap, cars outnumbered buses 47 to 1.

**Vehicle rule** (used for UAVDT relabelling and for Al-Hada):

- a pickup is a truck;
- a Hiace-type minibus is a bus;
- a closed van is a car.

## Cleaning and label quality

Cleaning removed very little. The label audit puts the label error at **4.4%** (95% upper bound 8.5%).

| Check | Result |
| --- | --- |
| Unreadable files | 0 in both datasets |
| Duplicate boxes (SeaDronesSee, IoU >= 0.95) | 23 removed |
| Exact duplicate frames (UAVDT, MD5) | 2 removed, each inside one sequence |
| Near-identical pairs across groups (pHash) | 0. Within groups: SeaDronesSee 8,740, UAVDT 20,952. A frame-level split would therefore leak, which is why the split is by group |
| UAVDT tracks with mixed labels | 32 of 2,642 (1.2%); 516 boxes relabelled to the track majority; 2 ties |
| Manual audit, 300 crops (about 37 per class, stratified) | 7 wrong labels, 18 bad boxes, 59 unsure |

Audit detail by class (errors / judged):

- truck 4/32
- swimmer 2/29
- car 1/27
- boat 0/37
- jetski 0/31
- life-saving appliances 0/24
- buoy 0/27
- bus 0/34

Most errors are trucks labelled as cars. Unsure cases are mostly small or night crops. Counting them as errors would raise the bound to 16.3%: we report this as a limitation.

Files: `data/audit/` in the repo (cleaning log, track relabels, audit results).

## Split

The split is by group, never by image, so no flight and no sequence appears in two splits. Every cross-split leakage check returned 0: video, group, sequence, MD5 and near-duplicate.

|  | SeaDronesSee | UAVDT-M |
| --- | --- | --- |
| Group unit | Video (flight) | Sequence |
| Groups train / val / test | 30 / 5 / 4 | 34 / 7 / 9 |
| Rule | Each class within +-8% of 70/15/15 | Val and test each hold >= 5 truck and bus tracks |
| Note | Only 1 of 50,000 random assignments met every rule: the split is fragile | Grouping by site prefix (M01-M14) failed, so the split falls back to sequences |

- **Seed and file:** seed 42, `data/splits/2026-10-19_Splits_v1.json`. Detection and segmentation use the same mapping, so the test flights match across tracks.
- **The SeaDronesSee `rgb` group:** 1,338 images without a video id, kept as one group and placed in test, where it supplies 5,549 crops (41% of test). A check on 24 Oct found:
  - no MD5 is shared across splits;
  - these images are 20 MP stills (5456 x 3632, some portrait), a capture type that appears nowhere in train or val.

  The images carry no timestamps or altitude, and all 1,338 come from one platform (`trinity`). The only other `trinity` group is `multispectral` (475 frames, 740 crops), which is in **train**. No image is shared, but a same-session overlap (same boats, same day) is plausible and cannot be ruled out. The score **without** `rgb` is therefore the headline test estimate, and the full-test score is secondary:

| Model (test macro-F1, raw, 3 seeds) | All test | Without `rgb` |
| --- | --- | --- |
| ResNet18 full fine-tune, lr 0.002 (selected) | 0.733 | 0.679 |
| ResNet18 full fine-tune, lr 0.001 | 0.719 | 0.666 |
| SmallCNN from scratch | 0.702 | 0.655 |
| ResNet18 frozen | 0.652 | 0.635 |

The ranking is the same either way, but the intervals overlap, so the differences between models are not statistically confirmed on test. Without `rgb` (12 groups) the selected model scores **0.679 ± 0.013, group-bootstrap 95% CI 0.523–0.874**, against 0.733 (0.570–0.837) on the full test. Computed on 25 Oct from the saved Step 8 predictions with the same bootstrap (2,000 draws, seed 42); the full-test interval was reproduced exactly. The stills are the easiest part of the test: sea macro-F1 is 0.887 on them, against 0.770 on the test videos.

**Top-k accuracy and calibration (second opening of test, 25 Oct).** `TestPredictions_v1` stores labels only, so the 12 checkpoints listed in `TestLog.csv` were run once more on test for this report, after the opening was logged with its reason. Nothing was trained, tuned or selected. Every raw and adjusted argmax matched the 23 Oct predictions, and top-1 accuracy and ECE reproduce the earlier figures. Raw logits, mean of 3 seeds:

| Model | Case | Top-1 | Top-2 | Top-3 | ECE |
| --- | --- | --- | --- | --- | --- |
| Full fine-tune, lr 0.002 (selected) | All test | 0.934 | 0.968 | 0.978 | 0.020 |
| Full fine-tune, lr 0.002 (selected) | **Without `rgb`** | **0.909** | **0.954** | **0.968** | **0.023** |
| Full fine-tune, lr 0.001 | Without `rgb` | 0.904 | 0.951 | 0.967 | 0.026 |
| SmallCNN from scratch | Without `rgb` | 0.853 | 0.913 | 0.939 | 0.032 |
| ResNet18 frozen | Without `rgb` | 0.882 | 0.952 | 0.973 | 0.078 |

- **Where top-2 helps.** It recovers about 4.5 points. The largest single source is truck predicted as car (147-161 crops per seed). A substantial share are cross-domain confusions of tiny objects (car to swimmer, swimmer to car, car to boat), and within the sea domain, life-saving appliances to swimmer and swimmer to buoy.
- **Logit adjustment.** The per-class bias raises macro-F1 only within noise (0.733 to 0.738 on the full test, 0.679 to 0.688 without `rgb`), but lowers top-1 accuracy (0.934 to 0.927) and worsens calibration (ECE 0.020 to 0.045). Raw scores are therefore the headline.
- **Calibration.** The frozen backbone is the worst calibrated model (ECE about 0.08). The selected model's confidence tracks its accuracy closely (ECE 0.02).

## Known limitations

1. **Small vehicles.** 63.6% of UAVDT cars are under 32 px, and so are 47.4% of SeaDronesSee swimmers. In the test, 160 of the 249 buses are under 32 px, and recall on them is 0.
2. **One sequence holds most of the bus failure.** M0602 contains those same 160 buses: 64% of all test buses. Its test error rate is 38%, against 2.5-12.5% for most other sequences. Its UAVDT attributes (daylight, medium altitude, front view) match three sequences that score about 10%, and the night sequence M0206 scores 9%. The cause is object size, not weather or light.
3. **Small test at group level.** The test holds 4 SeaDronesSee videos and 9 UAVDT sequences. Group-bootstrap 95% CI for the selected model's macro-F1: 0.570-0.837 on the full test, and 0.523-0.874 without rgb (12 groups). Crop-level intervals are too narrow, because crops from one video are correlated.
4. **Rare classes come from few groups.** Jetski, life-saving appliances and buoy each come from only 2 videos in val and 2 in test, and one of the two test groups is rgb. Without rgb each rests on a single video, so its interval means nothing. Life-saving appliances fall from F1 0.40 to 0.00 on all 3 seeds: every correct prediction for this class came from the stills. This is a shift between flights inside SeaDronesSee, not label noise: what the model learned for this class from its training flights did not transfer to the other test video.
5. **UAVDT site leakage risk.** Sequences that share a site prefix can land in different splits, so UAVDT val and test scores may be optimistic.
6. **UAVDT label noise in test.** Some box trucks are labelled car (M1303), and a billboard is labelled truck (M0602). These count as model errors in our scores. A blind audit of the 50 most confident test errors (25 Oct, 2 raters, 94% agreement) found the label wrong and the model right in 13 of them (26%, 6 distinct objects). Another 21 (42%) could not be judged by either rater, and most of those are under 15 px. Only 16 (32%) were clear model errors. The reported test score is therefore a lower bound. Sea classes were judged by one rater with the class definitions, because the second rater lacked domain familiarity.
7. **The SeaDronesSee `rgb` group in test.** No image is shared across splits, but it comes from the same platform as the train group multispectral, and it is 41% of the test and easier than the rest. The headline test score is therefore the one without it: 0.679 (CI 0.523-0.874). See Split.

## Al-Hada mountain test set

Al-Hada is a held-out test of a domain none of the training data covers: a steep mountain road filmed with an iPhone from public viewpoints above it, looking down on passing traffic. It is used for evaluation only and never for training or tuning.

| Item | Value |
| --- | --- |
| Clips | CLIP01 and CLIP02, filmed for the team with an iPhone (elevated public viewpoints, not a drone) |
| Frames | 164 + 148 = 312, sampled at 1 fps, 1920 x 1080 |
| Capture | HEVC in HDR (HLG). Tone-mapped to SDR on extraction, which leaves colours washed out |
| Classes | car, truck, bus (same vehicle rule as UAVDT) |
| Privacy | Raw videos stay in an owner-only folder. Annotators see extracted frames only, never the clips, and only after an automatic blur step (YOLOv8s at 1280 px). That step blurred 5 person boxes in CLIP01, and it blurs the plate zone of any vehicle 40 px or larger. The largest annotated vehicles are about 70 px long (CLIP02; 61 px in CLIP01), so a plate spans only a few pixels and no plates or faces are readable |
| Tool | CVAT, exported as COCO |
| Rule | Label every vehicle with half or more of it visible, edges included. No empty frames on export |
| Speed | About 30 s per frame (about 4 s per box) |

**Annotation (closed 25 Oct)**

| Part | Frames | Boxes (car / truck / bus) | Status |
| --- | --- | --- | --- |
| CLIP02 t0075-t0148 | 74 | 595 (541 / 54 / 0) | Done, approved after one revision |
| CLIP02 t0001-t0074 | 69 | 829 (733 / 87 / 9) | Done, approved |
| CLIP01 | 157 (t0001-t0164, 7 excluded) | 8,060 (7,611 / 402 / 47) | Done 25 Oct; checked (no missing frames, no duplicates) |
| **Total** | **300** | **9,484 (8,885 / 543 / 56)** |  |

**Exclusions:** 5 frames of CLIP02 (t0023-t0027) and 7 frames of CLIP01 (t0002-t0008) were dropped because a phone blocked the camera.

**Review:** every export is checked for empty frames, boxes outside the image, near-duplicate boxes and missing frames. Sample frames are also inspected with the boxes drawn on. The first CLIP02 export had 3 empty frames and missed vehicles at the image edges. Both were fixed in the second version, which added 66 boxes.

**Limitations of this set:**

- traffic was light;
- trucks are few and buses rare;
- the HDR footage loses contrast.

**Results (v2, 26 Oct, final).** Both clips complete: 300 frames, 9,484 vehicle crops (8,885 car / 543 truck / 56 bus), cut exactly as for UAVDT. A partial run on 25 Oct (6,649 crops) gave the same picture (selected model 0.336). The four Track 1 models were run once on them, with no tuning. A predicted sea class counts as an error. Macro-F1 over car, truck and bus, raw, mean ± std over 3 seeds:

| Model | Al-Hada macro-F1 |
| --- | --- |
| Full fine-tune, lr 0.002 (selected) | 0.334 ± 0.013 |
| ResNet18 frozen | 0.350 ± 0.001 |
| Full fine-tune, lr 0.001 | 0.319 ± 0.002 |
| SmallCNN from scratch | 0.312 ± 0.010 |

- **Cars transfer; trucks and buses do not.** Selected model: car F1 0.94, truck 0.06, bus 0.00. 90% of trucks and 82% of buses are predicted car, and 5% of all crops are predicted as a sea class (mostly car to swimmer, the same tiny-object confusion seen on the test set). Truck recall is 0.29 even above 64 px. Frozen scores highest (0.350), but the gap to the selected model is small and the ranking is not meaningful at this level.
- **The main cause is a class-definition shift, not only new terrain.** Al-Hada trucks are mostly white pickups and its buses white Hiace-type vans, labelled by our vehicle rule. UAVDT trucks are cargo trucks and its buses are city buses, so the training data holds almost no pickup labelled truck. A visual check of the misclassified crops confirmed this.
- **Many CLIP01 trucks are only 18-25 px**, the size range where even the test audit raters could not decide.
- These are the final Phase 1 scores. The models were not changed between v1 and v2. A frame-bootstrap CI (0.328-0.342) is reported only as a lower bound on uncertainty, because frames of one clip are correlated.

## Data-improvement plan

The plan has seven actions, ordered by expected gain. Every one starts from a measured problem. The first two need no new data and can be reported in Phase 1. None of them uses test results to tune the model: test scores are only re-reported under corrected labels or explicit size limits, next to the original score.

| # | Problem (evidence) | Action | When |
| --- | --- | --- | --- |
| 1 | **Label errors in test.** 13 of the 50 most confident errors are wrong labels: trucks labelled car in M0208 and M1303, a jetski labelled boat, a boat labelled swimmer | Relabel these 6 objects. Audit every test truck and bus track blind to predictions. Report the score with original and with cleaned labels | Phase 1 (report both); full audit Phase 2 |
| 2 | **Crops too small to judge.** 21 of 50 errors could not be classified by either rater, most under 15 px; 160/249 test buses are under 32 px with recall 0 | Report test metrics by size bin, plus a score that excludes crops under 16 px as an "ignore" region (standard practice in aerial benchmarks). Keep all crops in training | Phase 1 |
| 3 | **Bus confused with truck even when large** (M0207 buses over 100 px predicted truck); only 46 bus tracks in all of UAVDT | Add bus and truck examples from another aerial dataset with these classes, grouped by sequence. Weight bus and truck by track, not by crop | Phase 2 |
| 4 | **Small objects lose detail at 64 px.** 96 px input did not help (-0.016 for full fine-tune) | Try context crops: the box plus a margin, so road and scale are visible. Coordinate with detection (tiling), where small objects are found first | Phase 2 |
| 5 | **Test is small** (4 SeaDronesSee videos, 9 UAVDT sequences; CI 0.570-0.837) | Add Al-Hada as a held-out test. Report grouped cross-validation over train+val as a second estimate | Al-Hada: Phase 1. CV: Phase 2 |
| 6 | **SeaDronesSee `rgb` stills** are 41% of test and easier (sea F1 0.887 vs 0.770 on video) | Report test with and without them (done). In Phase 2, give stills their own split, or split them by session if metadata can be recovered | Done; Phase 2 |
| 7 | **Rare sea classes come from few flights** (life-saving appliances: 2 test videos, F1 0.40, and 0.00 without the stills) | Collect or borrow more varied examples. Until then, report per-class CIs and do not over-read this class | Phase 2 |

**Action 2, done 25 Oct.** Computed from the saved Step 8 predictions; no model was re-run. Test macro-F1, raw, mean of 3 seeds:

| Model | All test (13,551) | Box >= 16 px (12,515) | Box >= 16 px, no `rgb` (6,986) |
| --- | --- | --- | --- |
| Full fine-tune, lr 0.002 (selected) | 0.733 | 0.738 | 0.688 |
| Full fine-tune, lr 0.001 | 0.719 | 0.726 | 0.675 |
| SmallCNN from scratch | 0.702 | 0.717 | 0.680 |
| ResNet18 frozen | 0.652 | 0.658 | 0.643 |

- **Dropping crops under 16 px barely moves the score (+0.005).** Only 19 truck and bus crops are that small. The real gap is the 16-32 px bin, where 149 buses have recall 0.
- **Size is not the whole story.** Above 64 px, truck recall is still 0.57 and bus recall 0.49, while car stays at 0.94-0.98 at every size. This matches the audit: buses are confused with trucks even when large.
- **Without the stills and tiny crops, full fine-tuning leads scratch by only 0.008,** which is within seed noise. The claim that ImageNet pretraining helps should be stated with that caveat.

**Action 8 (Phase 2): domain-aware prediction.** Top-2 analysis shows tiny cars predicted as swimmer or boat, and swimmers as cars. In deployment the domain is known (the drone flies over sea or over a road), so predictions can be restricted to that domain's classes. This rests on an explicit assumption, that the domain is known at inference time; it is a deployment rule, not an improvement of the model itself. It must be designed and evaluated on val, not on test.

**Al-Hada.** A second iPhone shoot is planned for November, with a written brief: SDR not HDR, 4K, no zoom, several vantage points at different heights and angles, busy hours, and a target of 100 or more trucks and buses. It will extend the mountain test set and add the large-vehicle examples the current clips lack.
