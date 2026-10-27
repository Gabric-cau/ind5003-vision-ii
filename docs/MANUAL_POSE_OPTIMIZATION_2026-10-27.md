# Manual Pose Development Experiments

The new manual joint delivery is usable and was used in real inference and two
masked-supervision training runs. Neither trained candidate improved the fixed
10-pixel metric on the validation video over the original model at 1920 input.
The original deployed pose weights were therefore retained. This is a measured
failure to transfer within this experiment family, not proof that the supplied
annotations are useless or that every possible strategy has been exhausted.

## New Source Evidence

The received report and JSON describe one reviewer's targeted correction of
previously observed shooter errors. The JSON and CSV contain exactly the same
139 records across four videos and 47 frames: 112 manually positioned joints,
24 human-confirmed original model joints, and three occluded joints with null
coordinates. There are 136 visible coordinate targets, not 139 independent
frames or 139 complete human skeletons.

| Video | Reviewed frames | Joint records | Corrected | Confirmed | Occluded |
| --- | ---: | ---: | ---: | ---: | ---: |
| 05-04 | 10 | 35 | 30 | 4 | 1 |
| 10-01 | 13 | 42 | 27 | 14 | 1 |
| 17-08 | 4 | 14 | 8 | 5 | 1 |
| 19-08 | 20 | 48 | 47 | 1 | 0 |

The source files were read without modification:

- Manual JSON SHA256: `0b72d4d9f50952798dec5ad58c6a9109b7525ac748a45901da5457025651eb4a`.
- Manual CSV SHA256: `6646cf552165b194b846c3979d38e7b3b6d399a92855c035823a41455cb52f58`.
- Report PDF SHA256: `dd012ad4da1012679dfb8ce137e7c66e76d1350fc6f9d33d579bca72623908c4`.

All four video hashes match the existing pose-pack MP4 files. Every annotation
time matches the source per-frame PTS exactly; frame/FPS was not substituted for
the variable-rate source timeline. All coordinate bounds and COCO17 joint names
match original image dimensions. The available lab YOLO11m-pose weights match
the declared SHA256
`29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2`.

The delivery now provides exact reviewed pose-output hashes. Those four original
JSON files have not been located and hash-verified locally. The importer records
this partial binding honestly: source video and model verified, exact historical
inference output file pending. New predictions have new frame-local candidate
IDs; old track IDs are not joined to new predictions by number alone.

## Import and Evaluation Contract

`scripts/import_manual_pose_joints.py` creates 47 lossless source PNGs, preserves
all raw manual records, and emits 17-slot masks per reviewed frame. The 136
visible coordinates can supervise position. The three occluded joints carry
visibility state but no invented coordinates. All 660 unreviewed joint slots
are excluded from both coordinate and visibility supervision.

This is a targeted development set, not an independent blind test or a random
sample of overall pose quality. Training videos are 05-04, 10-01 and 17-08
(27 frames, 88 visible joints); validation is 19-08 (20 frames, 48 visible
joints). Frames or individual joints were not randomly mixed across this split.

Primary metric: visible human joint within 10 original-image pixels. Missing
people and low-confidence joints remain failures in the denominator. Secondary
metric: error within 2 percent of image height. This secondary normalization is
not standard torso-based PCK and does not replace the primary metric.

Predictions are saved before manual target coordinates are loaded. Full-frame
and automatic person-ROI inference receive only images and model-produced boxes.
The ROI is each predicted person box expanded by 20 percent. Evaluation associates
a reference person using human-point containment in an expanded predicted box,
then detection confidence; it never chooses the candidate with the lowest joint
error. This association uses ground truth for diagnostic correspondence, so the
reported metric does not establish automatic shooter-role identification.

## Actual Runs

All YOLO runs used the existing RTX 4090 environment. Three baseline modes were
compared: full frame at 1280, full frame at 1920, and automatic person ROI at 1280.
MediaPipe heavy33 was also run on the same frozen automatic person boxes, with
the existing task asset and CPU environment. MediaPipe used IMAGE mode on each
static frame; no temporal tracking claim was made for sparsely sampled frames.
Its shoulder, elbow, wrist, hip, knee and ankle outputs were mapped to COCO17.

Two 30-epoch sparse training runs updated only the YOLO `model.23.cv4` keypoint
branch. The installed loss implementation was inspected and a gradient check
confirmed that visibility-zero joints contribute no coordinate gradient. The
keypoint visibility, box, classification and DFL loss gains were set to zero.
This prevents the 660 unreviewed joints from becoming false negative labels.

Person boxes used for assignment were frozen-detector pseudo boxes, not human
box labels. Only training boxes were expanded to include supplied visible
points. No validation coordinates were used to construct inference crops.
Both runs use the fixed last epoch; native YOLO AP/OKS on sparse labels was not
used to select a checkpoint.

- Run 1: AdamW learning rate 0.0001; non-keypoint branches and their BatchNorm
  statistics frozen. Training pose loss fell from 4.38464 to 1.29340.
- Run 2: learning rate 0.00002 and all BatchNorm running statistics frozen,
  including the keypoint branch. This tested the observed small-sample BN drift.

After accounting for checkpoint FP16 storage rounding, run 1 changed 26 tensors,
all within the keypoint branch. Run 2 changed 13 tensors, all within that branch,
and changed no BatchNorm running mean or variance. These were actual parameter
updates, not only threshold or report changes.

## Fixed-Metric Results

Every entry below uses the same 10-pixel threshold and counts missing outputs
as failures. The pooled column mixes fitted and validation data and therefore
must not be called generalization accuracy.

| Model and input | Training points /88 | Validation points /48 | Pooled /136 |
| --- | ---: | ---: | ---: |
| Original, full1280 | 37 | 9 | 46 |
| Original, full1920 | 42 | 16 | 58 |
| Original, person ROI1280 | 23 | 5 | 28 |
| Run1, full1280 | 67 | 8 | 75 |
| Run1, full1920 | 60 | 12 | 72 |
| Run1, person ROI1280 | 28 | 4 | 32 |
| Run2, full1280 | 50 | 8 | 58 |
| Run2, full1920 | 50 | 15 | 65 |
| Run2, person ROI1280 | 23 | 5 | 28 |
| MediaPipe heavy33, automatic ROI | 38 | 11 | 49 |

The best validation result is original full1920: 16/48, or 33.33 percent, on
these deliberately selected difficult joint targets. Run2 full1920 improves
validation mean error from 17.05 to 16.31 pixels and secondary image-height
coverage, but 15/48 does not beat the fixed primary 16/48. Neither candidate is
promoted. A complete pose or entire project 90-percent claim is unsupported.

The 24 human-confirmed model points are 24/24 within 10 pixels under original full1920,
while the 112 corrected points are 34/112. This confirms strong difficulty
selection in this set; those confirmed-point percentages are not overall pose
accuracy. Run1 full1280 reaches 52/112 on corrected points and 23/24 on confirmed
points, showing the new annotations did support real targeted improvement.

## Error Inspection and Decision

All 47 reviewed frames obtained a reference-person box match in full-frame
inference. Two 05-04 frames had more than one eligible person box, retained as
association ambiguity metadata. Five largest-error frames were rendered with
human, original and trained points. The inspected images show high-kick lower
limb collapse onto the support leg, anatomical left/right confusion, and
crossing limbs or arm motion. No coordinate scaling, frame-number or COCO-index
mismatch was found in these examples.

The present comparison supports retaining the original weights and using the
manual set as a reproducible error-regression set. It does not justify repeatedly
adding epochs until the same development labels reach 90 percent. A further
generalization claim needs additional independent players or action sequences,
especially the failure poses, with event/person-separated evaluation. Existing
annotations should be reused, not redone; no new arbitrary sample quota is
imposed. Exact historical pose JSONs would close the remaining ID-binding audit,
but their absence does not block new image-based development.

## Artifacts

- `outputs/manual_pose_joints_2026-10-27/`: verified import, masks, 47 PNG frames,
  all source records, and extracted PDF text.
- `outputs/manual_pose_optimization_2026-10-27/`: three baselines, first training
  run, three trained evaluations, loss/source checks, parameter audit and
  `diagnostics/largest_errors.png`.
- `outputs/manual_pose_optimization_2026-10-27/mediapipe_static/`: full MediaPipe
  prediction records and fixed-metric evaluation.
- `outputs/manual_pose_conservative_2026-10-27/`: second training/evaluation run.
- First last-checkpoint SHA256:
  `6a0bbe0c480a9a16e342f57ddc100ef8b9e0eb18b84ae4db13fd69f6ced8ca09`.
- Second last-checkpoint SHA256:
  `2aa93eb2db0a05875519401996a3332ba384df01cbdfd0a75bbfb025bf417445`.

The 15 focused importer/evaluator tests passed. Default pipeline pose weights were
not replaced. All candidate weights and original inference records are retained.
