# Model card: Cross-domain aerial detection, classification and segmentation (AI251P Capstone)

Last updated 17 Nov 2026. Every number below comes from a result file named in [Sources](#sources). Nothing was tuned on test.

## 1. Summary

The system finds small objects (people in the water, vessels and road vehicles) in aerial or elevated imagery. It returns **candidates ranked by confidence** for a human operator to confirm or reject. One call per image returns, for each object, the class, confidence, box, mask and size.

| Stage | Model | Weights (team Drive) | Input |
|---|---|---|---|
| Detector (T2) | YOLO11n, *Joint_StrongerAug* (sea + urban, stronger augmentation) | `03_Detection/2026-11-04_P08_Baseline/YOLO11n_Joint_StrongerAug/weights/best.pt`; ONNX `03_Detection/Export/2026-11-16_YOLO11n_JointStrongerAug_v1/model.onnx` | 640×640 letterbox |
| Classifier (T1) | ResNet18, full fine-tune, lr 0.002, seed 43, raw logits | `02_Classification/Export/2026-10-31_ClsResNet18_s43_v1/model.onnx` | 64 px crop |
| Segmentation (T3) | YOLO11n-seg, 3 vehicle classes | `04_Segmentation/seg/yolo11n_seg_1280/weights/last.pt` | 1280 px |
| Service (T4) | FastAPI wrapper of the pipeline | `04_Integration/Service/2026-11-16_Service_v1/app.py`; notebook `notebooks/integration/service.ipynb` | image or video ≤ 10 s |

**Classes (8):** sea: swimmer, boat, jetski, life_saving_appliance, buoy; urban: car, truck, bus. The detector was selected on validation only (`P09_Model_Freeze.json`). Its selection note says no statistically significant superiority over the other two training regimes is claimed.

## 2. Intended use

- **Decision support** for an operator who reviews aerial or elevated imagery, for example:
  - maritime search and rescue (people in the water, small craft);
  - road-traffic observation (cars, trucks, buses).
- The output is a **ranked list of suggestions**. Every candidate starts as `pending_human_confirmation`, and a person confirms or rejects it (`POST /confirm/{request_id}`). **Final decisions always stay with the human operator.**
- Users: trained operators and researchers who know the limits in Section 5.

## 3. Domains and data

| Domain | Data | Role | Notes |
|---|---|---|---|
| Sea | SeaDronesSee ODv2: 37 videos + 2 still-image groups | train / val / test | split by video (`data/splits/2026-10-19_Splits_v1.json`) |
| Urban roads | UAVDT Benchmark-M: 50 sequences, sampled at 5 fps | train / val / test | split by sequence, **not by site** (Section 5) |
| Mountain road | Al-Hada: 2 iPhone clips from public viewpoints, 300 labelled frames | held-out test for detector and classifier; **training data for segmentation** | private, not public; blurred for privacy |
| Urban (external) | VSAI: 221 images | held-out diagnostic for the detector | classes merged into one "vehicle" class |

Details on sources, licences, cleaning and the label audit (label error about 4.4%, 95% upper bound 8.5%) are in `data/DATASET_CARD.md`. **Desert scenes, named in the project title, were not evaluated.** No desert data was used.

## 4. Performance

### 4.1 Detector, official test (conf 0.001, NMS IoU 0.7, 640 px)

| Domain | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Sea (SeaDronesSee test) | 0.630 | 0.517 | 0.528 | 0.282 |
| Urban (UAVDT test) | 0.503 | 0.373 | 0.339 | 0.218 |

Per-class AP50 (AP50-95):

| Sea class | AP50 (AP50-95) | Urban class | AP50 (AP50-95) |
|---|---|---|---|
| boat | 0.891 (0.513) | car | 0.679 (0.420) |
| buoy | 0.792 (0.483) | truck | 0.178 (0.119) |
| jetski | 0.713 (0.352) | bus | 0.160 (0.116) |
| swimmer | 0.199 (0.054) | | |
| life_saving_appliance | 0.043 (0.010) | | |

Recall by object size (test):

| Domain | Small | Medium | Large |
|---|---|---|---|
| Sea | 0.134 (361 of 2,702) | 0.627 (3,103 of 4,946) | 0.926 (1,778 of 1,921) |
| Urban | 0.746 (8,770 of 11,763) | 0.968 (5,779 of 5,968) | 0.254 (100 of 394) |

Out-of-domain (diagnostic, not comparable with the 8-class test mAP):
- **Al-Hada** (car/truck/bus, 300 images): precision 0.629, **recall 0.109**, mAP50 0.078, mAP50-95 0.015.
- **VSAI** (all vehicles merged): recall 0.521, precision 0.055, AP50 0.343, mAP50-95 0.142.

Validation sweep (after test, test unchanged): over conf 0.001–0.25 and NMS IoU 0.5–0.7, val recall barely moves: 0.611–0.614 (sea) and 0.334–0.370 (urban). **No recall-oriented operating point was adopted.** The service runs at conf 0.25 / IoU 0.7, the Ultralytics default, which was **not chosen on validation**.

### 4.2 Classifier (T1), raw logits, mean of 3 seeds

| Set | Macro-F1 |
|---|---|
| Validation | 0.817 ± 0.006 |
| Test without the SeaDronesSee `rgb` stills (headline) | 0.679 ± 0.013 (group-bootstrap 95% CI 0.523–0.874) |
| Test, all | 0.733 ± 0.020 |
| Al-Hada (car/truck/bus) | 0.334 ± 0.013 |

Car transfers to Al-Hada (F1 0.94); truck (0.06) and bus (0.00) do not. Source: `reports/track1/README.md`.

### 4.3 Segmentation (T3), Al-Hada test (`yolo11n_seg_1280/last.pt`, 1,448 objects)

| Mask mAP50-95 | Mask mAP50 | Box recall | Mask recall | Per-object IoU | Per-object Dice |
|---|---|---|---|---|---|
| 0.286 | 0.557 | 0.483 | 0.483 | 0.775 | 0.869 |

Only Al-Hada is reported. **No sea or urban segmentation results exist**, because the model was trained on vehicles from Al-Hada clips only. A segmentation model trained on another dataset (`yolo11n_seg_fixed/last.pt`) scored mask mAP50-95 0.002 on Al-Hada.

### 4.4 Speed and deployment (Tesla T4)

| Measure | Result | Target |
|---|---|---|
| Detector only, ONNX Runtime CUDA, 640×640, batch 1 (preprocess + inference + NMS) | p50 13.7 / p95 17.8 / p99 19.7 ms, 70.6 img/s, 188 MiB GPU | p95 ≤ 200 ms: **MET** |
| Detector only, PyTorch fp32 CUDA, same setup | p50 20.9 / p95 34.7 / p99 50.1 ms, 42.4 img/s, 242 MiB GPU | p95 ≤ 200 ms: **MET** |
| **Full pipeline** (detector + classifier + segmentation), 30 frames 1920×1080, decode excluded | p50 170.6 / **p95 279.8** / p99 310.0 ms | p95 ≤ 200 ms: **NOT MET** |
| ONNX vs PyTorch raw output | max abs diff 4.07e-3 (CPU), 3.75e-3 (CUDA) | ≤ 1e-3: **FAIL** (reported as is) |
| ONNX vs PyTorch after NMS, same input tensor | 977 of 977 boxes matched, same class; max conf diff 4e-6 | — |
| Masks returned per object (pipeline) | 512 of 592 (86%) | all objects: **NOT MET** |

In the full pipeline the classifier stage is the slowest at p95 (131.8 ms), because cropping runs on the CPU for every box.

## 5. Limitations

1. **Small objects.** Sea recall on small objects is 0.13. Swimmers, the main search-and-rescue target, have AP50 0.20. Many small swimmers will be missed, so **an empty result does not mean the water is clear.**
2. **Weak classes.** Life-saving appliances (AP50 0.04, recall 0.03), trucks (0.18) and buses (0.16) are unreliable. The classifier confuses truck→car and small bus→car (test F1: bus 0.26, truck 0.45).
3. **False positives.** At low confidence the detector produces many false positives (for example 43,219 swimmer FPs and 68,242 car FPs on test at conf 0.001). Sea classes are also predicted on roads and car on water (cross-domain FPs).
4. **Domain shift.** On the mountain road (Al-Hada) the detector's recall is 0.109. Classifier macro-F1 is 0.33: Al-Hada trucks are mostly pickups and its buses Hiace-type vans, which UAVDT barely contains. VSAI precision is 0.055. Performance outside SeaDronesSee- and UAVDT-like imagery should be assumed poor until measured.
5. **Segmentation scope.** Masks come from a model trained only on Al-Hada vehicles (3 classes). Sea objects and UAVDT-style vehicles often get no mask. Its test frames come from the same two clips as its training frames, so the reported scores are likely optimistic.
6. **Size estimates.** Sizes are in pixels. Metres are given only when altitude and camera field of view are supplied, and they assume a near-nadir view. No metre-level accuracy has been validated.
7. **Evaluation caveats.**
   - UAVDT is split by sequence, not by site; 4 of 5 test sites also appear in train.
   - The SeaDronesSee `rgb` stills (41% of classifier test crops) may share a session with a training group.
   - Test sets hold few independent groups, so confidence intervals are wide.
   - About 4.4% of labels are wrong (audit), so test scores are a lower bound.
8. **Operating point.** The service threshold (conf 0.25) is a default, not a validated choice. Recall and precision at that point were not tuned for the use case.
9. **Latency.** The full pipeline misses the 200 ms p95 budget on a T4 (279.8 ms). The detector alone meets it.
10. **Capture conditions not covered:** night, thermal or infrared, heavy weather, oblique ground-level views, and desert terrain.

## 6. Out-of-scope uses

- **Any automatic action without a human confirming each candidate.** This includes dispatch, alerts that trigger action on their own, enforcement and any safety-critical control loop.
- **Identifying, tracking or profiling individual people,** reading licence plates, or any surveillance of identifiable persons. The system does not and must not do face or person re-identification.
- **Military, weapon-targeting or law-enforcement evidence use.**
- **Declaring an area clear of people or vessels** based on an empty result (see limitation 1).
- **Domains or sensors not evaluated:** desert, night, thermal, satellite, ground-level cameras, indoor scenes.
- **Official traffic statistics or counts.** Truck and bus counts are not reliable.
- **Measuring object size in metres** for decisions without validated altitude and camera data.

## 7. Ethical and privacy considerations

- **Privacy.** Al-Hada footage was filmed from public viewpoints. Annotators saw only frames after an automatic blur of people and plate zones, and the raw clips stay in an owner-only folder. At the resolution used, no faces or plates are readable.
- **Errors are not equal.** In search and rescue a missed person (false negative) is far more costly than a false alarm. The current model's low small-object recall makes it a **supplement to, not a replacement for,** a human search.
- **Human oversight.** The service always returns ranked suggestions and records each human decision (`decisions.jsonl`).

## 8. Sources

| What | File |
|---|---|
| Detector selection | `03_Detection/2026-11-06_P09_Evaluation/P09_Model_Freeze.json` |
| Detector test, per-class AP, recall by size, errors, val sweep, VSAI, Al-Hada | `03_Detection/2026-11-06_P09_Evaluation/P09_Final_Consolidated_Results.json`, `P09_Recall_By_Size.json`, `P09_AlHada_Independence_Audit.json` |
| Segmentation, Al-Hada | `04_Segmentation/alhada_FINAL_test_imgsz.txt`, `alhada_results3_imgsz.txt`, `alhada_FINAL_test.txt` |
| Classifier | `reports/track1/README.md`, `data/DATASET_CARD.md` |
| ONNX parity | `03_Detection/Export/2026-11-16_YOLO11n_JointStrongerAug_v1/parity_report.json` |
| Pipeline latency and masks | `04_Integration/2026-11-16_Pipeline_v1/manifest.json`; `reports/integration/2026-11-16_Section4_Note.md` |
| Detector benchmark | `03_Detection/Benchmark/2026-11-17_T4_v1/benchmark_report.json` |
| Service | `04_Integration/Service/2026-11-16_Service_v1/` (`app.py`, `EXAMPLE_REQUEST.md`) |
