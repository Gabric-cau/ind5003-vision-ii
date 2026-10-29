# External Ball Detector Round 1

## Scope

One fixed 30-epoch detector training experiment was authorized using the new
Leverkusen-Stuttgart and Arsenal-Manchester City material. This experiment
measures ball-box localization. It does not establish complete video analysis,
trajectory identity, kicking-foot accuracy, or 90% overall project accuracy.

A subsequent local [M5 Pro training probe](MPS_TRAINING_PROBE_2026-10-28.md)
verified actual two-epoch training and checkpoint inference at 1920 pixels.
Batch 1 is the preferred local fallback based on measured memory use. This
probe does not complete the 30-epoch experiment described below.

## Recovering Clean Input

Both delivered review-video archives contain burned-in ball boxes, center
circles and coordinate text. They are evidence for review, not valid training
or evaluation pixels. The corresponding clean video streams were recovered
from the source URLs already recorded in the workbooks:

- Leverkusen-Stuttgart, 2024-08-17: `https://www.youtube.com/watch?v=G2Gnu4J4a3I`,
  1280x720, 25 fps, video format 136.
- Arsenal-Manchester City, 2023-08-06: `https://www.youtube.com/watch?v=LAt-9jOgwtY`,
  1920x1080, 25 fps, video format 137.

The recovered MP4 hashes differ from the annotator's original container hashes.
Consequently the importer verifies decoded scene correspondence instead of
assuming that the same URL implies matching frame indices. All 20 shots use a
zero-frame offset; 447 scene comparisons, covering every exported sample and
each shot's start/middle/end, passed. The maximum mean pixel error was 5.014/255.
Alignment masks exclude review text and annotation colors and do not use the
reference box coordinates to select an offset. Exported model inputs are the
unmodified clean frames, with no review overlays or masked answer locations.

Recovered source SHA256 values:

```text
lev_stu_20240817.mp4 252ad7592bde5439eaa68afea0028db1c2edc15f771bffecaa1ff40526af4c84
ars_mci_20230806.mp4 c790a49e44c85ce76af6d4dc3d07797601b0aa515c027080272f1ca91edba4fd
```

## Frozen Data And Protocol

The import preserves all 704 source annotation rows. Only clear or
motion-blurred frames with complete source boxes are exported; uncertain,
center-only and unlocalized records are not turned into negative examples.

| Partition | Source | Positive Frames |
| --- | --- | ---: |
| Training | Previous core01/core02/core03 and match01 | 122 |
| Training | Leverkusen-Stuttgart, nine events across 13 views | 320 |
| New-match evaluation | Arsenal-Manchester City, seven events | 88 |
| Development regression | Previously inspected core05/core06/core09 | 101, or 91 after existing duplicate grouping |

The new matches have source-specific identifiers to avoid collisions with the
older three-match package's unrelated `match_03` and `match_10` filenames.
Every view of an event stays in the same match partition. No exported image
hash crosses training and evaluation.

Training starts from the same original YOLO11m weights used by the previous
four-event refit. Its recipe is retained: 30 epochs, 1920-pixel input, batch 2,
AdamW with initial learning rate 0.0002, first ten layers frozen, seed 27,
no AMP, mosaic or mixup. The last epoch is selected before evaluating Arsenal;
Arsenal does not select a checkpoint or tune the hyperparameters. Internal
Ultralytics validation uses training images only and is not reported as
generalization performance.

The original model, previous four-event refit, and new checkpoint are compared
at identical confidence 0.05, NMS IoU 0.6, and 1280-pixel inference in both full
frame and tiled modes. The primary metric is highest-confidence-box IoU >= 0.5
over annotated positive frames. Missing detections count as misses. Per-event
counts, duplicate-adjusted counts and inference elapsed time are retained.
Background precision is not measured because the reference is not exhaustive.

The contradictory kicking-foot descriptions in Arsenal's two workbooks do not
enter this ball-box experiment. Existing source files and model defaults remain
separate from the candidate experiment.

## Reference Quality Review

After training started and before inspecting any model predictions, a fixed
18-frame visual check sampled the first, middle and last penalty of each new
match, using the last pre-contact and first/middle post-contact exported frames.
The clean inputs contained no residual annotation overlays. Two reference boxes
visibly target the player's footwear instead of the adjacent ball:

- Arsenal P1, original frame 194 (7.76 s), box `[531, 685, 555, 709]`.
- Leverkusen P5, original frame 4499 (179.96 s), box `[654, 558, 674, 578]`.

These are source-label review issues, not source-video frame-offset errors.
Evidence is in `outputs/external_ball_frames_2026-10-28_qa/`. No training labels
or evaluation denominators were changed during this round. The observations
need human adjudication; the small targeted sample does not estimate the
dataset-wide annotation error rate. Results below therefore measure agreement
with the frozen, imperfect references, not independently established truth.

## Artifacts

- `outputs/external_ball_frames_2026-10-28/manifest.json`: image/label hashes,
  source identities, partitions and clean-input verification.
- `outputs/external_ball_frames_2026-10-28/alignment.json`: frame correspondence
  and all candidate offset scores.
- `outputs/external_ball_frames_2026-10-28/alignment_samples/`: clean/review
  comparisons for visual inspection.
- `outputs/external_ball_training_2026-10-28/`: frozen protocol and run results.

Preparation and partition checks passed 18 tests. Real training started on the
lab RTX 4090 with PyTorch 2.7.0+cu128 and Ultralytics 8.4.175. The last verified
checkpoint progress was epoch 19/30, with 460.563 seconds of training elapsed.
The SSH connection then failed; direct and Tailscale-proxied reconnection,
IPv6 and peer ping attempts timed out. The local network and another tailnet
peer were verified working. The target's last handshake remained at
2026-10-28 16:31:07 Singapore time.

Remote training may have continued: its completion, final weights and evaluation
results are not yet verified. No replacement training process was started and
no new accuracy result is available. See `connection_interruption.json` beside
the local protocol. After connectivity returns, inspect the existing remote
process and files before resuming or fetching:

```text
/home/boris/workspace/ind5003-vision-ii-experiments/2026-10-28-external-ball-round1/train.log
/home/boris/workspace/ind5003-vision-ii-experiments/2026-10-28-external-ball-round1/run/fit/weights/last.pt
/home/boris/workspace/ind5003-vision-ii-experiments/2026-10-28-external-ball-round1/run/report.json
```

## Reproduction

Use a fresh output directory and the source files identified by the import
manifest. The new importer does not read frame pixels from the review videos
for model inputs.

```bash
python scripts/prepare_external_ball_frames_round1.py \
  --source-root /path/to/IND5003_delivery_20261028 \
  --output-dir NEW_IMPORT

python scripts/train_external_ball_round1.py \
  --old-manifest outputs/ball_frame_annotations_2026-10-27/manifest.json \
  --new-manifest NEW_IMPORT/manifest.json \
  --regression-manifest outputs/ball_frame_evaluation_2026-10-27/manifest.json \
  --pretrained outputs/core_contact_training_2026-10-26/models/yolo11m.pt \
  --previous outputs/ball_detector_four_event_refit_2026-10-27/last.pt \
  --output NEW_RUN --device 0
```
