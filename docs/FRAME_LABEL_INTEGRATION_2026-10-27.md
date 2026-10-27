# Frame Annotation Integration and Detector Pilot

The new workbooks are usable. Their remaining uncertainties do not require a
global pause or another round of generic video collection. This run imported
the source records, trained an experimental ball detector on the lab RTX 4090,
and compared its predictions with the original pretrained detector. It did not
replace the default pipeline or establish final unseen-video accuracy.

## Source Records

`球路逐帧补标_总表.xlsx`, SHA256
`c7e8f631165cdff4843a195e23cc77d39a15b576396226402f8b32fb54e698e5`,
contains 170 rows from four events: `core_01`, `core_02`, `core_03`, and
`match_01`. There are 150 centers, 144 complete boxes, six center-only rows,
and 20 rows without localization. Boxes were obtained through reported human
localization plus automatic bright-spot refinement; they are assisted training
candidates, not independent consensus references.

The importer preserves raw cells, sheet/row references, workbook and source
video hashes. Duplicate frames, dimensions, frame/time correspondence, geometry
and decoded video frame counts were checked. It does not rewrite the workbook.

- `core_03` frame 161 is explicitly marked interpolated, despite the source
  visibility cell saying clear. It is excluded from detection fitting.
- Frame 193 is marked truncated. Explicit out-of-view notes are distinguished
  from unknown visibility and camera cuts; ambiguous notes remain unknown.
- Partial visibility stays separate. No missing coordinate becomes `(0, 0)`,
  and no unlocalized frame is automatically a negative detection example.
- Contact references remain adjacent-frame intervals. Last visible ball
  coordinates are not relabelled as final resting positions.

The strict default selects 122 clear/blur frames with complete boxes: 87 from
the three Colombia-England core events for training, and 35 from the German Cup
`match_01` for development validation. Frames are not randomly split. Partial
boxes and center-only records remain available for later diagnostic work.

`人工复核反馈_2026-10-27.xlsx`, SHA256
`752c0676583f1565619572f63b8d7d882ab1855e5c1d06485a0516f980764465`,
was imported as eight keeper/person observations and twelve shooter issue
ranges across eight pose-pack videos. All twenty regression checks remain
pending. The observations describe YOLO11m-pose outputs, whereas the old pack
contains YOLOv8-derived coordinates. Exact reviewed inference output binding is
pending; track IDs cannot be joined across those models. These records are not
corrected joint coordinates or new striking-foot/contact labels.

## Actual Training

`scripts/train_ball_frame_detector.py` ran a fixed 25-epoch experiment using
the existing YOLO11m pretrained weights. Settings were 1280-pixel input, batch
4, AdamW at initial learning rate 0.0005, frozen first ten layers, seed 27,
no mosaic/mixup, and no AMP. The run used the existing lab environment:
PyTorch 2.7.0+cu128 and Ultralytics 8.4.175. Detector training and comparisons
took 73.39 seconds, excluding local preparation and file transfer.

Evaluation selects the highest-confidence ball detection at confidence 0.05;
a hit requires box IoU of at least 0.5 with the assisted reference. Missing
predictions remain in the denominator. Any-candidate oracle recall is stored
separately and is not substituted for the top-ranked prediction.

| Model | Training localization hits | Development validation hits |
| --- | --- | --- |
| Original pretrained YOLO11m | 16/87 | 15/35 (42.86%) |
| Best validation checkpoint | 84/87 | 20/35 (57.14%) |
| Last checkpoint, epoch 25 | 86/87 | 15/35 (42.86%) |

The best checkpoint improves this development comparison, but the last
checkpoint does not. Strong fitting of the three training events does not
establish transfer. The best checkpoint was selected using the same validation
match, and 35 correlated frames are not 35 independent penalty events. These
numbers are positive-frame localization recall, not goal/outcome accuracy,
complete trajectory accuracy, or the overall project accuracy. Background
false positives and difficult excluded frames are not comprehensively scored.

The candidate uses class 0 (`ball`); the deployed COCO detector uses class 32.
It has not been substituted into that pipeline. Before any promotion, automatic
full-video selection, identity continuity and unsupported views must be checked.

## Evaluation Boundaries

No new core05/core06/core09 frames were loaded by this importer or detector
pilot. Nevertheless these videos are previously inspected development material:
core05/core06 were used by an earlier ball-height model, and the deployed
contact fit used all ten core events. They must not be described as completely
untouched final tests of the existing system. A new module-specific evaluation
must state its exclusions and prior exposure; a final generalization claim
needs independent material and frozen predictions.

Core07/core08 interpretation conflicts, selected pre-movement timing questions,
and external match09 outcome identity remain unresolved by these two workbooks.
They can stay quarantined while uncontested records support development.

## Later Test Reference Delivery

The subsequently supplied `测试集_球路逐帧_core05_06_09.xlsx` has been read and
audited: 130 rows, 117 centers, 112 complete boxes, five center-only rows and
13 unlocalized rows from the three requested videos. This reference delivery is
now available; it must no longer be listed as missing. It was not used for
training or checkpoint selection in the detector pilot above. No detector
evaluation on these three videos has been run in this round.

Before scoring, separate the interpolated core09 frame 89, annotated repeated
frames, and uncertain ball identity after core09 frame 109. The source describes
the original shot as wide, but cannot confirm that the ball re-entering the view
is the same ball. A local ball detection can be scored without asserting a
continuous identity or a confirmed final position.

The latest user-supplied chat also contains the core annotator's confirmation
and screenshots supporting their newer core07/core08 endpoint descriptions.
These are additional review evidence; differences between a flight category,
save category and endpoint description still need distinct field semantics.
Subsequent chat screenshots provide additional observations: core01 keeper
movement at about 1.452 s labelled right; core02 movement at about 3.562 s and
contact at about 3.658 s; core03 movement at about 5.136 s labelled right and
contact at about 5.246 s. These are observed-frame evidence, not certified
first-onset instants; their direction convention must remain explicit. The
annotator also confirms core09 is wide. These answers must not be requested
again. Only external match09 event/result adjudication remains pending in this
round of semantic review, and that single label stays quarantined. The existing ten
core IDs were checked against the 39-event reference: all ten are subsets,
not additional independent events. This mapping needs no further human work.

## Artifacts and Reproduction

- `outputs/ball_frame_annotations_2026-10-27/`: source manifest, all raw and
  normalized annotations, summary, exported images and YOLO labels.
- `outputs/pose_review_feedback_2026-10-27/`: source-bound observations,
  shooter ranges, and pending regression checklist.
- `outputs/ball_frame_detector_2026-10-27/`: frozen protocol, original and
  trained predictions, comparison report, training log/CSV and both weights.
- Remote run: `/home/boris/workspace/ind5003-vision-ii-experiments/2026-10-27-ball-frames-v1`.

```bash
python scripts/prepare_ball_frame_annotations.py --export-yolo
python scripts/import_pose_review_feedback.py --help
python scripts/train_ball_frame_detector.py \
  --data DATA/yolo/dataset.yaml --manifest DATA/manifest.json \
  --weights MODELS/yolo11m.pt --output NEW_OUTPUT \
  --epochs 25 --imgsz 1280 --batch 4 --device 0
python -m pytest -q tests/test_ball_frame_annotations.py \
  tests/test_pose_review_feedback.py tests/test_ball_frame_detector.py
```

The three focused test files passed 30 tests. Training verifies exported image
and label hashes against the import manifest. Source workbooks and videos were
not edited, and previous outputs were preserved.
