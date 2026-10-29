# Final Augmentation Round

Protocol fixed on 2026-10-29, Singapore time, before this round's development
scores. The user requested one final training round with appropriate data
augmentation. This is a bounded comparison, not a claim of a globally optimal
model or a promise of 90% overall assignment accuracy.

## Changes And Rationale

The ball experiment aligns training with the existing tiled inference path:
half of the augmented arm's source presentations may use a fixed image-grid
crop containing the complete ball box. The grid is dimension-based, not
centered on the reference ball. Original full images remain in the mixture.
Moderate photometric variation and a small in-plane affine rotation accompany
the crop intervention. This is a combined augmentation treatment, not an
isolated causal test of contrast alone.

The keeper experiment uses all 835 already-delivered training boxes to fit a
single active_keeper class, meaning the defending goalkeeper for the current
attempt. The earlier 149-frame export was a sparse pose subset. Full-box
training addresses localization before pose estimation. It does not turn
unlabeled people into negatives for the generic COCO person class.

Brightness, contrast, horizontal reflection, and modest scaling can be
label-preserving here when boxes are transformed consistently. Horizontal
reflection does not supervise a left/right event label in these detector
experiments. No new 3D camera views, vertical flips, strong projective warps,
mosaic scenes, fabricated body joints, or invented ball trajectories are used.
The previous ball runs already included HSV, scale, translation, and horizontal
flip; this is not the first use of augmentation in the project.

## Frozen Data And Budgets

| Experiment | Training pool | Comparison groups | Fixed fit |
| --- | --- | --- | --- |
| Ball | 122 old frames and 320 Leverkusen frames | Arsenal 88; deduplicated core 91 | 2 arms, 10 epochs, 442 presentations/epoch, old/new 221 each |
| Active keeper | 835 frames from six short clips | Subject 02 and Argentina-France remain separate | 2 arms, 12 epochs, 835 presentations/epoch |

Both arms of each experiment share initialization, source presentation order,
optimizer configuration, and actual optimizer-update budget. The ball starts
from the completed round-2 balanced checkpoint. The keeper starts from the
local COCO YOLO11m checkpoint. Ball uses 1920 pixels, AdamW 0.00005, nominal
batch 16, and freeze 10. Keeper uses 1280 pixels, AdamW 0.0001, nominal batch
16, and freeze 10. Bias warmup learning rates equal the respective base rates.
Both use MPS, FP32, batch 1, workers 0; GPU jobs run sequentially. OOM or
non-finite updates are failures, not permission to silently change the fit.

Checkpoints are the fixed final epoch, with both arms frozen before external
development scoring. Source video/image hashes, frame PTS, transformed labels,
actual loaded examples, parameter changes, and optimizer steps are recorded.
Augmentation occurs in the training partition only. Sibling crops and all
frames/views of an event retain their original partition.

Keeper development frames include the existing pose-reference frames and a
fixed uniform source-frame sample per clip. The final exporter manifest fixes
that inventory before fitting. Unlocalizable keeper boxes are excluded, not
converted to empty-background negatives.

## Selection And Claims

For ball promotion, a candidate must reach at least 65/91 core and 62/88 Arsenal
top-confidence box hits at IoU >=0.5 under the same tiled inference settings.
Choose the qualifying candidate with the higher equal-group mean; ties prefer
the simpler control. Otherwise preserve existing defaults and report the
candidate without promotion. Report per-event metrics and any-candidate recall
separately. These positive-frame measurements do not establish precision on
background footage or complete-video accuracy.

For active keeper, choose the augmented candidate only if neither development
group's per-clip macro top-confidence IoU >=0.5 score regresses against its
matched control, and the equal-group mean improves. Otherwise retain the
control for the experimental detector. Report generic-person candidate
coverage separately: it does not solve the same keeper-role selection task.
Detector output is a new optional diagnostic, not an automatic replacement for
the existing movement, direction, and height pipeline.

These development groups have been exposed during earlier work. A frozen
protocol prevents this round's adaptive checkpoint search but does not make
them independent final tests. The 24-row preparation package updates existing
events; 14 populated labels do not add 24 independent events or supply new
box/joint coordinates. Pending semantic and timing issues retain masks.

## Delivery Checks

Inspect augmentation examples, run focused data/geometry tests, verify actual
weight updates, score both frozen arms, and run a label-free video inference
check with the chosen experimental keeper weights. Preserve original weights
and publish a local reproducibility receipt with exact paths and hashes.

## Completed Results

The ball control arm completed 10/10 epochs and 388 optimizer updates. Its
checkpoint is kept as the best ball result in this round because it reached
61/88 Arsenal frames and 65/91 deduplicated core frames under the frozen tiled
evaluation. The new ball control and mixed-augmentation arms both scored
58/88 and 62/91, so neither passed the promotion gates. The existing ball
initializer therefore remains the default; the augmentation did not improve
the measured ball detector.

The active-keeper control and photometric arms both completed 12/12 epochs and
838 optimizer updates from the same YOLO11m initialization and frozen sample
schedule. The photometric arm resumed from its saved epoch-7 optimizer
checkpoint after a process interruption and completed the remaining five
epochs. On 455 frozen development frames, control scored 60/294 (20.47%
clip-macro top-1) on `development_external_match` and 92/98 (93.02%) on
`development_subject_02`. Photometric scored 44/294 (15.09%) and 92/98
(93.02%), respectively. Because it regressed on the external-match group,
the control checkpoint is the selected standalone active-keeper detector.

The selected keeper detector was then run without labels on two complete
videos. `02-06.mp4` produced 45 sampled frames, 39 frames with a detection,
and 39 frames with pose candidates in 13.67 seconds. `match_24.mp4` produced
61 sampled frames, 20 frames with a detection, and 8 frames with pose
candidates in 17.38 seconds. These are runtime/output checks only; both
reports set `accuracy_measured=false`, and the detector is not wired into the
default event, direction, height, or whole-project accuracy path.

The final local artifacts are in
`outputs/keeper_detector_final_summary_2026-10-29`,
`outputs/keeper_final_smoke_2026-10-29`, and
`outputs/ball_final_augmentation_2026-10-29/release_review`. Focused final
tests passed: 53 tests in 3.48 seconds. The original full test run also
remains recorded as 1543 passed, 4 skipped, and 40 subtests.
