# Contact-time training on the ten core clips

## Latest result: recovered visual evidence and retraining

The current deployable contact head is `model_roi_v5`, with frozen evidence in
`features_roi_v5`. Earlier experiments below remain as the experiment history.
The final head uses the same fixed logistic-regression settings and four match
folds. It outputs a timestamp for all ten events: MAE **0.0800 seconds**, maximum
error **0.1667 seconds**, and **10/10 within 0.2 seconds** of the coarse human
viewing references. The existing rule on the identical final evidence has MAE
0.1133 seconds. These are grouped development results, not an untouched test.

The improvement follows repairs to visual evidence: full-frame pose resolution
1280 and confidence 0.10, association in camera-stabilized coordinates, and
additional real pose inference in automatically requested actor crops when the
original rule abstains. Crop boxes are interpolated in the stabilized plane and
mapped back using each frame's camera transform. Missing keypoints remain
missing; interpolated boxes are search regions, never fabricated observations.
Existing usable events retain their original feature evidence.

All ten events now contribute positive and negative examples to the separate
deployment fit. Versions v3 through v5 preserve the development result while
correcting nonlinear camera mapping and invalid keypoint handling. The deployed
extractor checks video, checkpoint, preprocessing and source-code hashes. The
v5 result supersedes the seven-event coverage and six-event training-fit limits
of the initial experiment described below.

To reproduce from the final cached evidence:

```sh
python scripts/train_core_contact_model.py train \
  --features-dir outputs/core_contact_training_2026-10-26/features_roi_v5 \
  --labels outputs/core_contact_training_2026-10-26/data/labels.json \
  --output outputs/core_contact_training_2026-10-26/model_roi_v5_repeat
```

Direct new-video inference uses `scripts/predict_video_contact.py` with
`--model outputs/core_contact_training_2026-10-26/model_roi_v5`, the same two
YOLO checkpoints and a new output directory. No offset-calibration argument is
needed for this head. The complete video entry point and portable deployment
package additionally execute the other course-output modules.

This experiment learns a contact-time selector from automatically extracted
ball and player-pose evidence. The output is one candidate contact timestamp
and source frame per penalty clip. It changes only an isolated contact module;
the original project pipeline and its saved predictions remain unchanged.
It does not train ball direction, physical ball height, full trajectories or a
new detector.

## References and evaluation split

The source workbook is `罚球补标工作表.xlsx` in
`/Users/gabric/Downloads/IND5003_资料审阅_20261026/extracted/小皮_10段核心罚球_人工补标交付`.
Its `补标10段` sheet supplies contact times from cells G6 through G15. These are
human viewing times in seconds, not frame-exact physical-contact ground truth.
No uncertainty interval was supplied, and none is invented. The workbook's
keeper-movement times and manually recorded camera-cut times are retained only
for provenance and audit; they are not inputs to feature extraction or training
features.

The preparation script checks every video SHA-256 against the workbook's current
video hash, and checks frame rate, resolution, duration and reference bounds.
It writes separate `inputs.json`, `labels.json` and `audit.json`. `inputs.json`
contains video identity and technical metadata only. The extractor never opens
`labels.json`. Old contact frames are not transferred through an assumed offset
between different video files.

| Match group | Events | Validation fold size |
| --- | ---: | ---: |
| `col_eng` | 4 | 4 |
| `cro_den` | 2 | 2 |
| `esp_rus` | 2 | 2 |
| `rus_cro` | 2 | 2 |

The fixed evaluation is four-fold leave-one-match-out. Each fold fits on three
matches and predicts the remaining match. Frames of the same event, and events
from the same match, cannot cross that fold's training/validation boundary.
All ten events are a subset of the original 39 events that previously informed
project development. These are development-validation results, not a new
untouched test. The separate eight World Cup 2022 events remain unused.

The references retain their limitations. For example, `core_01` and `core_03`
have equal recorded keeper/contact times, and empty uncertainty notes do not
prove that a reference is exact. Such limits restrict the interpretation of
fine timing errors; they do not prevent this isolated experiment from running.

## Frozen video evidence

`src/contact_video_features.py` selects the first camera shot automatically with
the existing shot scanner, up to a maximum of 12 seconds and scene-cut threshold
30.0. It samples every second source frame. Selection uses neither the human
contact time nor the manually recorded shot boundary. A later label comparison
does not expand an automatic validation window if the reference falls outside
it.

The pretrained YOLO11m COCO detector runs at image size 1280 with sports-ball
class 32 and confidence threshold 0.04. The pretrained YOLO11m-pose model runs
at image size 640 with confidence threshold 0.15. Both are frozen. The current
execution uses two CPU threads, batches of four sampled frames and the original
video bytes; no detector or pose weights are updated.

The pipeline associates person detections into tracks, compensates camera
motion, and derives 24 geometric/evidence features per sampled frame. These
include ankle visibility/confidence, ankle-to-stationary-ball distance, ankle
separation and speed, hip speed, knee extension, body lean, ball visibility and
confidence, ball movement, ball-to-ankle distance, camera reliability and body
scale. Coordinates and distances are normalized where defined by the extractor.
Missing evidence is represented with its corresponding visibility/availability
features. Pose and ball predictions are model evidence, not manual ground truth.

Actor and stationary-ball seed selection reuse the existing contact-candidate
scoring where available, with a track-displacement fallback when it is not.
This is an offline method: frames later than a proposed contact time are allowed
as evidence. It is not a causal real-time alarm.

## Learned selector

For each candidate time, the model concatenates features at lags
`-0.4, -0.2, 0, +0.2, +0.4` seconds and five availability masks, for 125 values.
Candidates without the full lag context are excluded from fitting and cannot
win a prediction. Evidence additionally requires a visible pose plus either a
visible ball or an available stationary-ball seed, for at least three consecutive
sampled observations. Absolute timestamp, event identifier and match identifier
are not model features.

Training samples within 0.15 seconds of a coarse contact reference are positive;
samples at least 0.35 seconds away are negative. The intermediate band is not
fitted. These fixed tolerances are training choices, not measured uncertainty
bounds on the human references. Each event contributes total weight 0.5 to its
positive samples and 0.5 to its negative samples. If an automatic window has no
eligible positive or negative samples, the event is recorded as unused in that
fold's training audit.

A weighted standardizer and a binary logistic-regression head are fitted on
training events only. The fixed head uses `C=0.1`, `lbfgs`, up to 2,000
iterations and seed `20261026`. There is no hyperparameter search on these
validation folds. Scores are smoothed in a 0.1-second neighborhood, then the
highest eligible score supplies the event prediction. Scores are uncalibrated;
the model abstains when no complete temporal context exists.

The four fold models produce out-of-fold predictions. A separate deployment
fit then considers all ten events, using only events that satisfy the evidence
and positive/negative sample requirements. Its in-sample results are not used
as validation scores. In this run only six events actually contribute to that
fit: four `col_eng` and two `esp_rus` events. The distinction between events
considered and events fitted is recorded in `deployment_training_audit`.

## Comparisons and measurements

The learned selector is compared with three controls: the existing causal
first-departure contact rule applied to exactly the same stride-two ball/pose
evidence, minimum ankle-to-stationary-ball distance, and the median reference
timestamp from the training portion of each fold. The first comparison is not
a rerun of the complete original pipeline: sparse-to-dense refinement is not
performed in this experiment.

The report includes output coverage, mean/median/maximum absolute timing error
and fractions within 0.1, 0.2 and 0.5 seconds of the supplied references. Because
some methods can abstain, it also compares methods on the same available events
and on pairwise intersections. Errors against coarse reference times must not
be described as exact physical-contact accuracy. Frame count measures processing
volume; the independent data remain ten events in four matches.

## Reproduction

Run from the project root. Data preparation needs openpyxl and OpenCV; feature
extraction additionally needs PyTorch and Ultralytics; head fitting needs NumPy
and scikit-learn. Set the checkpoint arguments to the frozen local YOLO11m and
YOLO11m-pose files used by this run; their hashes are saved in feature provenance.

```sh
CORE_DATA='/Users/gabric/Downloads/IND5003_资料审阅_20261026/extracted/小皮_10段核心罚球_人工补标交付'
python scripts/prepare_core_contact_training.py \
  --workbook "$CORE_DATA/罚球补标工作表.xlsx" \
  --videos "$CORE_DATA/videos" \
  --output outputs/core_contact_training_2026-10-26/data

BALL_CHECKPOINT=outputs/core_contact_training_2026-10-26/models/yolo11m.pt
POSE_CHECKPOINT=outputs/core_contact_training_2026-10-26/models/yolo11m-pose.pt
python scripts/extract_core_contact_features.py \
  --inputs outputs/core_contact_training_2026-10-26/data/inputs.json \
  --ball-model "$BALL_CHECKPOINT" \
  --pose-model "$POSE_CHECKPOINT" \
  --output outputs/core_contact_training_2026-10-26/features_v1 \
  --threads 2 --device cpu

python scripts/train_core_contact_model.py train \
  --features-dir outputs/core_contact_training_2026-10-26/features_v1 \
  --labels outputs/core_contact_training_2026-10-26/data/labels.json \
  --output outputs/core_contact_training_2026-10-26/model_v1
```

The extractor refuses to replace existing event feature files, and the trainer
requires a new output directory. Use new directory names for a repeat run. When
executing on another machine, `--video-root` can point the extractor at an
identical-hash copy of the ten videos without editing source annotations.

The model directory contains per-fold parameter archives, `oof_predictions.json`,
`report.json`, `deployment_weights.npz` and `deployment_metadata.json`. The report
binds references, features and code by SHA-256. Existing label-free feature
records can be predicted without opening an annotation file:

```sh
python scripts/train_core_contact_model.py predict \
  --features outputs/core_contact_training_2026-10-26/features_v1/core_01.json \
  --model outputs/core_contact_training_2026-10-26/model_v1 \
  --output outputs/core_contact_training_2026-10-26/deployment_smoke_prediction.json
```

This example exercises a known event and is a deployment smoke check, not a new
test result. Direct video inference needs no manual annotation or contact-time
argument:

```sh
NEW_EVENT_VIDEO=/path/to/penalty_event.mp4
python scripts/predict_video_contact.py \
  --video "$NEW_EVENT_VIDEO" \
  --model outputs/core_contact_training_2026-10-26/model_v1 \
  --calibration outputs/core_contact_training_2026-10-26/calibration_v1 \
  --ball-model outputs/core_contact_training_2026-10-26/models/yolo11m.pt \
  --pose-model outputs/core_contact_training_2026-10-26/models/yolo11m-pose.pt \
  --output outputs/core_contact_training_2026-10-26/new_video_prediction \
  --device cpu --threads 2
```

This writes prediction, feature and evidence JSON files into a new output
directory. It checks weight, extractor-code and preprocessing bindings against
the trained model before predicting. With `--calibration`, the selected output
uses the separately fitted rule-offset model described below; the temporal
classifier prediction remains available for comparison. Omitting that argument
selects the temporal classifier.

## Temporal-classifier results

The completed four-fold run in `model_v1/report.json` evaluates all ten events.
Its learned selector is not promoted: the existing rule applied to the same
visual evidence has lower timing error on their shared available events.

| Method | Outputs / all events | MAE on own available events (s) | Within 0.2 s / all events |
| --- | ---: | ---: | ---: |
| Learned temporal classifier | 7/10 | 0.3838 | 4/10 |
| Existing rule on identical evidence | 7/10 | 0.0995 | 7/10 |
| Ankle-proximity control | 8/10 | 0.0933 | 8/10 |
| Training-fold median time | 10/10 | 1.2390 | 0/10 |

The two 7/10 rows are not the same seven events. On the six events where all
methods produce an output, the learned classifier MAE is 0.4283 seconds and
the existing rule and ankle-proximity MAEs are each 0.1050 seconds. The learned
selector reaches the 0.2-second tolerance on 3/6 of those events; the rule reaches
6/6. Unavailable events remain in the denominator of the all-event hit rates
above.

Feature extraction processed 973 sampled frames. Automatic windows contained
all ten supplied contact references, but usable temporal evidence was missing
for several events. `core_05`, `core_06` and `core_10` do not have three
consecutive supported temporal observations; `core_09` has positive observations
but no eligible negative ones for fitting. Consequently the deployment fit uses
only `core_01` through `core_04`, `core_07` and `core_08`. In the fold validating
`col_eng`, just the two `esp_rus` events actually fit the head; in the fold
validating `esp_rus`, just the four `col_eng` events do. This is an automatic
tracking/evidence limitation, not missing human contact labels.

The independent `feature_audit.json` records the video/reference/code integrity
checks and per-event evidence counts. Camera compensation also failed for some
adjacent frame pairs, notably in `core_05` and `core_08`. The audit does not turn
those detections or the original coarse timestamps into verified physical truth.

CPU feature extraction with two threads took 28.47 to 56.31 seconds per clip,
median 37.91 seconds, excluding model loading. Total extraction was 392.84
seconds. Head fitting and cross-validation took 0.089 seconds after extraction.
The direct video-wrapper smoke check in `video_smoke_core01/prediction.json`
took 30.78 seconds end to end on the remote CPU with two threads, including model
loading; visual extraction took 29.30 seconds. It received the `core_01` video
without any manual time or annotation input. The unchanged rule and temporal
classifier each returned 1.7333 seconds; the separately fitted offset returned
1.8329 seconds. This is a deployment smoke check on a known training event,
not an unseen-video accuracy test or a general speed benchmark.

## Offset-calibration experiment

A separate one-parameter model fits the mean residual
`human_reference_seconds - existing_rule_seconds` on each fold's available
training events. It adds that learned offset to the unchanged rule timestamp.
It preserves an original rule abstention and abstains if the adjusted time would
fall outside the automatic shot; it does not clip a result into the window.
The implementation uses `DummyRegressor(strategy="mean")`, with no neural
weight update and no change to visual detection. Its deployment fit uses the
seven events for which the rule outputs a timestamp, learning an offset of
`+0.0995238095` seconds. This is separate from the per-fold offsets used below.

`calibration_v1/report.json` records the out-of-fold comparison on the same seven
available events:

| Method | Outputs / all events | MAE (s) | Median error (s) | Within 0.1 s / all events | Within 0.2 s / all events |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing rule | 7/10 | 0.0995 | 0.0667 | 4/10 | 7/10 |
| Rule with fitted offset | 7/10 | 0.0742 | 0.0900 | 7/10 | 7/10 |

Mean error decreases by about 0.0253 seconds, while median error increases.
Coverage and the 0.2-second hit rate do not change. This supports retaining the
offset as an experimental option, not a claim that the full original pipeline
or exact physical-contact accuracy improved. The small difference is measured
against coarse human viewing times and may include annotation-timing bias.

This calibration was proposed after inspecting the temporal classifier's
development results. Match-wise fitting excludes each fold's references from
its offset estimate, but does not remove experiment-selection bias on this same
cohort. It is not a new untouched test and is not promoted to the default
pipeline. The external World Cup set remains unused.

Reproduce the calibration from the already extracted evidence, without running
the visual models again:

```sh
python scripts/train_contact_offset_calibration.py train \
  --features-dir outputs/core_contact_training_2026-10-26/features_v1 \
  --labels outputs/core_contact_training_2026-10-26/data/labels.json \
  --visual-metadata outputs/core_contact_training_2026-10-26/model_v1/deployment_metadata.json \
  --output outputs/core_contact_training_2026-10-26/calibration_v1
```

Use a new output directory for a repeat run. The calibration directory contains
the fitted `deployment_calibration.json`, fold estimates, out-of-fold predictions
and source/feature/code bindings in its report. The fit and fold evaluation took
0.0057 seconds after visual features were available.
