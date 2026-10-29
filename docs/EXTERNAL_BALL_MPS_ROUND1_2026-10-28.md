# External Ball Round 1 On M5 Pro

## Frozen Protocol

This is the authorized full local training round after the
[MPS compatibility probe](MPS_TRAINING_PROBE_2026-10-28.md). The separate 4090
run was last verified at epoch 19/30; its final state is still unavailable.
The local experiment starts from the original pretrained weights, not from
the short-probe checkpoints or an unverified remote intermediate checkpoint.

- Training: 442 positive frames, comprising the existing 122-frame four-event
  training set and 320 clean-source Leverkusen-Stuttgart frames.
- Fixed comparison: 88 Arsenal-Manchester City frames from seven events.
- Development regression: 101 existing core05/core06/core09 frames, 91 after
  the existing duplicate grouping.
- Model and fit: YOLO11m, 1920 pixels, batch 1, 30 fixed epochs, freeze 10,
  AdamW, initial learning rate 0.0002, nominal accumulated batch 64, seed 27.
- Runtime: Apple M5 Pro 24 GB, MPS FP32, AMP disabled, no implicit MPS CPU
  fallback, workers 0, isolated Python 3.12/PyTorch 2.7.0/Ultralytics 8.4.175.
- Selection: last epoch, frozen before the comparison. No checkpoint or
  confidence threshold is selected using the Arsenal results.
- Comparison: original pretrained weights, previous four-event refit, and
  the new last checkpoint; each uses full-frame and tiled 1280-pixel inference,
  evaluation batch 1, confidence 0.05 and NMS IoU 0.6 on this same machine.

The batch and device differ from the disconnected CUDA experiment. Batch 1
was chosen from measured hardware compatibility and memory usage, without
using held-out predictions. It is a separate experiment, not a numerical
continuation of the remote run.

## Completed Run

All 30 epochs and all 12 comparisons completed successfully. Training took
5,700.51 seconds (95.01 minutes), including the framework's final fit validation
and checkpoint saving. The complete training/evaluation call took 5,964.81
seconds (99.41 minutes).

The receipt confirms 252 optimizer steps and changes in 201 of 331 parameter
tensors, compared on the same FP32 model after head initialization. The tracked
non-bias parameter's maximum observed change was 0.00895994. Frozen backbone
layers and DFL parameters did not change. There was no OOM retry, batch-size
change, or non-finite training loss. MPS driver allocation at saved checkpoints
was approximately 6.13 GB; this is not a measurement of total system RAM.

Final candidate: `outputs/external_ball_training_mps_2026-10-28/fit/weights/last.pt`.
SHA256: `f929de5b1c10aec4b892e3278e9360e423f85c4af88dfd93858a4b81583a42d1`.

## Comparison Results

The table uses deduplicated positive frames and the frozen top-ranked ball-box
IoU >= 0.5 metric. Missing predictions remain misses. All models were evaluated
on the same MPS environment with the same settings within each mode.

| Group | Mode | Original pretrained | Previous refit | New round | Change from previous |
| --- | --- | ---: | ---: | ---: | ---: |
| Arsenal-Manchester City, 7 events | Full | 37/88 (42.05%) | 40/88 (45.45%) | 42/88 (47.73%) | +2.27 pp |
| Arsenal-Manchester City, 7 events | Tiles | 39/88 (44.32%) | 51/88 (57.95%) | 62/88 (70.45%) | +12.50 pp |
| Core development retest, 3 events | Full | 50/91 (54.95%) | 49/91 (53.85%) | 53/91 (58.24%) | +4.40 pp |
| Core development retest, 3 events | Tiles | 48/91 (52.75%) | 65/91 (71.43%) | 63/91 (69.23%) | -2.20 pp |

The previous refit's historical CUDA core result was 66/91. The current
same-environment MPS comparison is 65/91; do not substitute the historical
number into this table or attribute that device/runtime difference to training.

For tiled inference on Arsenal, 13 previously incorrect frames became correct
and 2 previously correct frames regressed. The largest net gains were P1
(+4 frames) and P4 (+6): these two events account for 10 of the 11 net gains,
and only 3/7 events have a positive net change. For the core retest, 1 frame improved and 3 regressed:
core06 lost two frames; core09 had one improvement and one regression.

The new tiled model produced a candidate in 87/88 Arsenal frames and 80/91
deduplicated core frames. Output availability is not correctness. Its 88-frame
Arsenal tiled run took 43.60 seconds versus 44.39 seconds for the previous
refit. Core timing was 18.95 versus 19.15 seconds for the complete 101-frame
run before deduplication. Timings include model loading and are single runs,
not repeated throughput measurements or complete-video pipeline latency.

**Decision:** keep the new checkpoint as an experimental candidate and retain
the previous default. The new-match gain is useful, but there is a core tiled
regression and neither group reaches 90%. This does not establish a global
optimum or an improvement in all assignment outputs.

## Error Review

The generated [summary](../outputs/external_ball_training_mps_2026-10-28/summary/summary.md)
contains all-frame and deduplicated results, per-event paired changes, and the
training-loss plot. Deterministic post-hoc review sheets show up to three frames
per error category, balanced across events:

- [Arsenal review](../outputs/external_ball_training_mps_2026-10-28/error_review_tiles/arsenal_new_match_contact_sheet.png)
- [Core review](../outputs/external_ball_training_mps_2026-10-28/error_review_tiles/core_development_retest_contact_sheet.png)

Visual review confirms successful recoveries of small balls, but also core06
misses near the goalkeeper (frames 155 and 156) and a wrong top-ranked target
in core09 frame 87. On Arsenal, 16 of the 26 strict misses have IoU between
0.3 and 0.5: the ball-box extent/alignment needs attention as well as detection.
This diagnostic does not change the primary IoU threshold.

Even selecting the best-overlapping candidate using the references would give
only 64/88 hits on Arsenal (72.73%) and 66/91 on the core retest. This is an
oracle diagnostic of the already generated boxes, not a deployable result or
a ceiling on future models. Reranking these same candidates alone cannot
reach 90% under the frozen metric.

Arsenal `ars_mci_20230806_p01_s0003_f000186` is an additional post-hoc reference
review candidate: the displayed reference box appears displaced to the right
of the visible ball, while both models box the ball to its left. Verify the
original annotation and exact frame alignment before deciding whether it is a
label error. This is an observation, not an adjudicated correction. The two
pre-existing reference issues also remain unchanged. No evaluation labels or
metrics were edited after seeing predictions.

Before another optimization round, audit precise box boundaries/frame alignment
and use separate development data for any changed recipe, including experiments
on old/new training balance and motion-blurred balls. These are hypotheses to
test, not proven causes of the current misses. The observed Arsenal results
must not be reused for repeated tuning and then described as a fresh final test.

## Recovery

Each model-save callback writes a separate temporary recovery copy and then
atomically replaces `last_recoverable.pt`. A copy every five epochs is also
retained by the framework. Progress records bind the recovery file's SHA256,
epoch, actual device/batch, finite training losses and optimizer-update evidence.
Recovery files retain optimizer state. Normal final inference checkpoints may
be stripped by the framework and should not be assumed to support resumption.
The runner has no automatic resume mode; any recovery must preserve the original
data and epoch budget and record its source checkpoint.

The completed job ran under `caffeinate -i` while connected to AC power.
Original model defaults are retained. The final checkpoint was independently
loaded on CPU and verified to match the recovery checkpoint's EMA tensors;
all parameters were finite.

## Artifacts

All artifacts are under `outputs/external_ball_training_mps_2026-10-28/`:

- `protocol.json`: frozen sources, hashes, configuration and comparison scope.
- `dataset/manifest.json`: exact training image and label inventory.
- `source_snapshot.tar.gz`: the source code used to launch this run.
- `requirements-mps-installed.txt`: complete installed dependency versions.
- `training_progress.json` and `last_recoverable.pt`: saved training progress.
- `fit/weights/last.pt`: completed final candidate.
- `selected_candidate.json`: final selection and actual parameter-update receipt.
- `evaluation_progress.json` and `report.json`: all 12 completed comparisons.
- `summary/`: checked summary JSON/Markdown and training-loss plot.
- `error_review_tiles/`: selected example identities and two review sheets.

The adjacent `outputs/external_ball_training_mps_2026-10-28.log` contains the
console log. Source preparation and training guards passed 20 focused tests;
summary and visualization checks passed another 14. All 442 training images,
442 labels, source manifests, launch-code hashes and baseline-weight hashes
were checked again after training.
An independent CPU review recomputed all 1,134 prediction rows across 12 files,
including source identities, reference boxes, IoUs, deduplication flags,
weight hashes and summary counts; all matched.

## Interpretation

The primary metric is top-ranked ball-box IoU >= 0.5 on annotated positive
frames, with missing detections counted as misses. It is not whole-project
accuracy. Report per-event counts and output availability alongside localization
recall; the old core set remains a development retest. Negative-frame precision
is unavailable because these references are not exhaustive.

The two known reference-box review issues described in
[the data report](EXTERNAL_BALL_ROUND1_2026-10-28.md) remain frozen for this
comparison. Training-subset validation metrics in the framework log are fit
diagnostics, not generalization evidence. This round trained the ball detector;
it did not retrain the pose, run-up-step, event-output, or offline Q&A models.
