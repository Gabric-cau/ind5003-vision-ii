# Continued Optimization, 27 October 2026

The user authorized continued training and engineering until the available
evidence supports a useful stopping point, a material data gap is reached, or
the relevant accuracy exceeds 90%. A finite experiment cannot certify the
global optimum. A 90% training fit, one-class majority score, conditional
accuracy after dropping unknowns, and a detector score are not interchangeable
with 90% performance on all assignment outputs.

## Frozen Comparisons

Ball detection training uses core01/core02/core03. Development selection uses
match01. The new module evaluation uses core05/core06/core09 only after candidate
selection; those events were exposed to older contact/height components and
therefore are not a fresh whole-system final test. Source workbooks are unchanged.

The evaluation importer retains all 130 source rows, selects 101 clear/blur
complete-box frames, and marks 91 primary frames after source-declared duplicate
grouping. Interpolated/truncated positions are excluded from strict localization;
later core09 detections do not establish continuous identity after re-entry.
The target remains highest-confidence ball localization at IoU >= 0.5 and
confidence >= 0.05, with absent predictions counted as misses. Per-video,
all-frame, and duplicate-adjusted results must all remain available.

## Changes Explained Before Implementation

The first ball detector fit strongly on three training events but transferred
poorly to the development match. Compare conservative fine-tuning, adapting the
backbone, higher-resolution training, and shifted crop augmentation using only
the existing training events. Compare full-frame and overlapping-tile inference
without a reference-driven ROI. Crop augmentation does not add independent events.

Keeper processing was coupled to successful strict ball-contact analysis.
Evaluate a separately guarded recovery using an existing source-bound automatic
contact estimate, bounded time window, camera stabilization, and actor checks.
This is an inference repair, not new human truth or an unconstrained prediction.

Shoe processing propagated early camera-chain failures through every later
frame. Evaluate independent reliable camera segments, selecting the automatic
contact segment and prohibiting velocity/landing comparisons across breaks.
This does not establish a complete run-up count if only a later segment is usable.

## Completed Detector Training

Nine new ball-detector training configurations completed on the existing lab
RTX 4090: five initial alternatives, three follow-up high-resolution alternatives,
and one fixed 30-epoch refit. The existing 25-epoch pilot was retained as a
control. Learning rate, frozen backbone depth, resolution, crop augmentation,
seed, checkpoint and label-free tiled inference were compared. No evaluation
event entered training.

The best development result was 26/35 annotated match01 frames, or 74.29 percent,
using high-resolution training and tiled inference. The three follow-up
configurations did not exceed that development score. The frozen comparison on
91 duplicate-adjusted positive evaluation frames was:

| Detector | Correct top-ranked boxes /91 | Localization recall |
| --- | ---: | ---: |
| Original pretrained model | 50 | 54.95% |
| Previous pilot | 63 | 69.23% |
| New development-selected checkpoint | 58 | 63.74% |

The new development-selected checkpoint was not promoted. A subsequent fixed
refit included all four development events: core01, core02, core03 and match01,
122 training frames. It achieved 66/91, or 72.53 percent, on the same evaluation
frames: core05 21/24, core06 30/37 and core09 15/30. Its all-frame result is
73/101. Because the previous evaluation had already informed this follow-up,
this is explicitly a **development retest**, not a fresh final test.

These are localization recalls on annotated positive frames at top-1 box
IoU >= 0.5. They do not measure false-positive precision, correct shot identity,
goal outcome, or complete video accuracy. The post-re-entry core09 ball can be
localized without establishing that it is the same shot ball.

The fixed-refit weights are retained at
`outputs/ball_detector_four_event_refit_2026-10-27/last.pt`, SHA256
`12d4b79bf610ae1297732a733455a420c632707ca7a9b7534811432ec5e2f8e3`.
Comparison receipts, all predictions and metrics are in
`outputs/ball_detector_evaluation_2026-10-27/` and
`outputs/ball_detector_refit_evaluation_2026-10-27/`.

The optional `--ball-specialist` interface keeps the original COCO model for
people and adds the trained single-class ball model with an explicit class
mapping. On a real wc20 video, the specialist was called successfully, but
visual inspection found incorrect downstream motion-blob associations to the
shooter. Additional detections and 19 returned candidate points therefore do
not establish an improved ball trajectory. The specialist remains opt-in.

## New Manual Pose Delivery

The PDF, JSON and CSV were actually read and checked. JSON and CSV agree on
139 joint records across four videos and 47 frames: 136 visible coordinates
and three occluded joints. All four source video hashes and per-frame timestamps
match. Unreviewed joints are masked, not turned into negative labels.

Two real 30-epoch masked keypoint-head training runs completed. Full-frame
1280/1920, automatic person crops, and MediaPipe heavy33 were compared. The
first fit improved training-point hits from 37/88 to 67/88, but neither fit
beat the original full1920 model on the held-out video: original 16/48,
first fit 12/48, conservative fit 15/48; MediaPipe 11/48. The fixed metric is
10 original-image pixels, including missing outputs as failures. This deliberately
selected difficult-joint set is not representative of overall pose accuracy.

The original pose weights are retained. Exact parameter-change checks prove
both fine-tuning runs updated the intended keypoint branch. Existing annotations
were useful and do not need to be redone. Full evidence and weight locations:
`docs/MANUAL_POSE_OPTIMIZATION_2026-10-27.md`.

## Contact and Integration Repairs

The old contact extractor stopped after 12 seconds, excluding actual late kicks
in two external videos. The replacement inspects the first automatic camera
shot, capped at 60 seconds, with matched retrained model metadata. Automatic
actor recovery handles an unsupported short track and rejects stale or
post-candidate stationary-ball evidence. No human contact times enter inference.

The selected causal v3 protocol changes wc18 from 2.32 to 16.20 seconds,
wc19 from 6.72 to 4.32 seconds, and wc20 from 3.68 to 17.72 seconds. Source-image
inspection supports those corrections as development diagnostics. Grouped
validation on the original ten course events remains MAE 0.08 seconds and
10/10 within 0.2 seconds of their coarse timing references. This is not
90-percent external video accuracy. An interval-constrained v4 ablation made
wc19 worse and was rejected; all variants are preserved.

Real complete main-entry runs finished on wc18 (317.00 seconds), wc19
(73.52 seconds) and wc20 with the specialist (479.75 seconds). The latter two
shared the GPU, and their configurations differ; these are execution receipts,
not controlled speed comparisons. wc18 correctly returns left foot. The first
wc19 run still abstained because the pose stage required visible goal geometry,
even though the contact stage had already measured the shooter. A subsequent
source-bound shoe recovery reuses that automatic actor without inventing goal
qualification and correctly returns right foot. Its ten landing candidates
are not a verified total run-up step count.

The final consistent eight-event shoe replay improves strict foot accuracy from
3/8 to 5/8 (62.5 percent) and output coverage from 5/8 to 7/8 (87.5 percent).
Conditional accuracy is 5/7, macro F1 is 0.6136, and balanced accuracy is 0.5833.
wc18 and wc19 change from unknown to correct; no previously available foot
decision becomes worse. wc20 remains unknown, while wc22 and wc24 remain wrong.
All eight bindings and decoded source pixels were rechecked; exactly matching
pose caches were reused with zero new neural calls. The 114.57-second replay
time is not fresh end-to-end latency. Full records are in
`outputs/pipeline_v4_2026-10-27/shoes_default_causal_v3/`.

A final fresh main-entry run, with the selected contact model and the latest
default shoe fallback, completes wc19 in 102.96 seconds and returns right foot
consistently in both required outputs and the final feature row. Its complete
receipts are in `outputs/pipeline_fullshot_2026-10-27/wc19_latest/`.

The source-ID bookkeeping fix removes false duplicate-view ambiguity in keeper
analysis. An optional cross-view identity check requires multiple simultaneous
box, torso and appearance matches, rejects separating observations and excludes
the shooter. It recovers only wc17's early image-left dive evidence. Direction
coverage is 1/8; direction accuracy is not scored because the reference
coordinate convention is unresolved. Incomplete observation or unverified
later identity switches still reject height. Final ordinal-height coverage and
strict accuracy remain 0/8. ROI/alias recovery is not enabled by default. The
evidence does not support blaming an unsubmitted teammate deliverable.

The short-contact-ROI shoe ablation restores wc20's paired pose features but
returns the wrong foot. It is rejected. The camera-segment alternative also
fails to recover a qualified foot answer at the corrected contact time and
remains disabled by default. Neither failure is hidden by changing thresholds.

Output summaries now exclude post-contact points from the run-up trajectory and
separate the final selected feature row from the legacy intermediate rule row.
Unsupported non-MP4 inputs fail early rather than after expensive inference.
The original stages, predictions and source files remain available for audit.

Detailed contact and keeper records are in
`docs/CONTACT_FULLSHOT_2026-10-27.md` and
`docs/KEEPER_RECOVERY_2026-10-27.md`.

## Completed Local VLM Comparison

The already installed local `qwen3-vl:8b` was actually run on all eight videos,
using 111 fixed automatically sampled frames. It received no reference answers,
player names or manually selected frames. The installed model is the Thinking
variant; the first two token-budget pilots were truncated without an answer.
After checking the installed renderer and official generation settings, a new
protocol was frozen before the completed eight-event run. All raw responses,
sample hashes, failures and predictions are retained.

All eight final responses satisfy the JSON schema. Foot, goal outcome and
qualitative keeper height each score 6/8, or 75 percent. However, it predicts
right foot for every event and goal for every event, matching the two majority
baselines exactly; left-foot and no-goal recalls are both 0/2. Its height result
is worse than always predicting low (7/8). It is therefore **not promoted**.
The local model's median inference time is 104.64 seconds, excluding contact
extraction. This is development retesting of a previously inspected match,
not evidence of unseen-match performance. See
`docs/LOCAL_QWEN3VL_BASELINE_2026-10-27.md` and
`outputs/local_qwen3vl_baseline_v3_2026-10-27/evaluation.json`.

## Delivery and Current Boundary

The new usable bundle is
`outputs/offline_deployment_2026-10-27/ind5003_offline_video_v4_20261027.zip`.
It contains the selected causal full-shot contact model, measured-shooter
fallback, corrected summaries and the original retained pose/shoe/keeper/
category weights. The optional detector/ROI/VLM experiments do not replace
default decisions. All 363 packaged files and model/code bindings passed
verification. The ZIP SHA256 is
`c42f6b545c85a4bce99c764b54f9203d56be1712f28388922d3a717b5d4f0c1f`.
The lab's existing environments were used for real inference; a clean dependency
installation has not been tested. The previous bundle remains intact. Final
repository verification passed 1,406 tests and 40 subtests; `git diff --check`
also passed. Packaged source files match the final local source byte for byte.

The existing material was sufficient for all the actual experiments above.
Repeated detector configurations reached a development plateau, pose fine-tuning
did not transfer to the held-out video, and the VLM did not beat simple class
priors. This establishes a stopping point for these measured experiment
families, not a proof that all possible algorithms have been exhausted.

Remaining requirements are specific rather than a request to redo the supplied
packs. The new manual joints supervise shooters, not a continuously identified
keeper through a dive. The ball sets supply localized positive frames but not
exhaustive distractor/background labels or a fully verified same-ball track
through all occlusions. No high-keeper class is represented in the current
event references. Independent top-down path/terminal-position references and
the external direction coordinate convention remain incomplete. These limits
prevent establishing every teacher output at 90 percent. All already inspected
external matches are now development material; a final generalization claim
requires a new match with a fixed, independently reviewed evaluation
definition. More epochs on the current references cannot supply that evidence.

External match09 remains quarantined pending conflict review. Its delayed review
has not blocked this work. The pose coordinates, ball-frame workbooks and core
reviews are already integrated; no teammate needs to repeat their original
delivery. All training and inference jobs for this round have finished.
