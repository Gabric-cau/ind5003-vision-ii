# Supervised shoe landing and striking-foot experiment

The new heads improve observed-window landing counts and foot classification
over the project's frozen shoe rules in grouped development evaluation. They
operate on the same saved paired MediaPipe 33-point poses. The pose network is
frozen; this experiment fits small supervised heads and does not train a new
pose estimator. It does not establish complete run-up counts or physical contact.

## Data and leakage controls

The reference is
`outputs/fresh_course_experiment/human_evaluation_2026-10-22/reference_source/penalty_annotations_39_human_v040.json`.
It contains 39 striking-foot labels and 223 landing observations inside the
recorded review windows, supplied by one human reviewer. Five additional
outside-window observations are not fitted or evaluated here. The 39 events
were already inspected during project development; results are not an untouched
test or a two-reviewer consensus.

The source-only evidence is
`outputs/fresh_course_experiment/shoe_approach33_cohort_v1/events/<clip_id>`.
Exactly 32 events have `inputs.json` and paired `poses33.json`. The other seven
remain unavailable. The runner verifies pose/input hashes against the frozen
summary, source hashes against references, and automatic window identity. It
does not transfer timing labels to the different core-ten video files.

Feature extraction reads only automatic approach/contact windows, camera
transforms, source-pixel hashes and model poses. Human reported contact frames,
landing frames, event outcome, actor identity strings and reference foot labels
are not inference features. Four leave-one-match-out folds fit on three matches
and predict the fourth. All frames/views of an event stay together. The
standardizer is fitted within each training fold, and each fitted landing event
contributes equal total weight. A separate deployment model fits all 32 usable
events; its in-sample predictions are not the validation results.

The foot target is highly imbalanced: 36 right and three left. Only two left-foot
events have dense inputs. Always predicting right would score 36/39 overall
accuracy while recalling none of the left-foot events. Therefore per-class
recall, output coverage and macro F1 accompany raw accuracy.

## Models

Each frame requires accepted normal/reflected actor poses, matching decoded
pixel hashes and a usable original camera transform. Anatomical channels are
already inverse-mapped by the existing MediaPipe producer. The feature extractor
uses confidence at least 0.35, body height at least 30 pixels and maximum paired
view disagreement 0.15 body heights. These are fixed experiment settings, not
newly certified visibility thresholds. It does not invent a pose through a gap.

For each anatomical foot, 19 features describe ankle/heel/toe position relative
to hips, knee angle, heel-to-toe length, distance to the box bottom, pose
confidence, mirror disagreement, and stabilized/hip-relative motion before and
after the frame. Positions and motion use body-height normalization. Features
at offsets -0.12, -0.06, 0, +0.06 and +0.12 seconds are concatenated with five
availability masks, giving 100 values. At least three observations are needed.
The method can use later frames and is an offline estimator.

Landing training marks samples within 0.05 seconds of a human same-foot landing
positive and samples at least 0.13 seconds away negative; the middle band is
ignored. Each event assigns total weight 0.5 to positive samples and 0.5 to
negative samples. A histogram gradient-boosted classifier uses 80 iterations,
seven leaves, maximum depth three, minimum leaf size 20, L2 regularization 5 and
seed 20261026. Scores are averaged within 0.05 seconds; candidates need score
at least 0.5, with same-foot non-maximum suppression at 0.20 seconds. Negative
samples mean absent from the supplied landing list. List completeness was not
independently certified for every event.

The striking-foot head summarizes each side's 19 features by quartiles in the
automatic contact bracket extended 0.4 seconds before and 0.25 seconds after.
It subtracts left summaries from right, yielding 57 features. A standardized
logistic head uses `C=0.1` and no intercept. Training includes the negated feature
vector with the opposite label, enforcing anatomical-side swap symmetry without
using an always-right prior. Both sides need at least three observations. Its
score is uncalibrated, not a verified probability of contact.

No model/threshold search was performed on the reported folds. The second
artifact directory, `model_v2_audit`, reruns the same settings and predictions as
`model_v1`; it adds the latest foot comparator and deployment provenance.

## Results

The authoritative report is
`outputs/shoe_supervised_2026-10-26/model_v2_audit/report.json`, with all 39
out-of-fold predictions retained in `oof_predictions.json`.

| Striking foot | Outputs | Correct / outputs | Correct / all 39 | Macro F1 including abstentions |
| --- | ---: | ---: | ---: | ---: |
| Latest frozen foot supplement | 14 | 12/14 | 12/39 | 0.5031 |
| Learned foot head | 32 | 30/32 | 30/39 | 0.7232 |
| Always-right control | 39 | 36/39 | 36/39 | 0.4800 |

Both available left-foot events are correct. Aspas has no dense input and
remains unknown. Eriksen and Jorgensen are predicted left instead of right.
The learned head improves usable coverage but does not meet the engineering
goal of at least 90% coverage and macro F1 at least 0.85.

For a directly comparable landing evaluation, the following rows use the same
27 windows and 156 human landings. Matching is one-to-one within 0.08 seconds.
It preserves the existing evaluator's temporal matching; anatomical agreement
is measured separately, not silently required by this timing metric.

| Landing method | Precision | Recall | F1 | Observed-window count MAE |
| --- | ---: | ---: | ---: | ---: |
| Frozen whole-shoe rule | 0.5909 | 0.0833 | 0.1461 | 4.9630 |
| Frozen fixed-heel/toe supplement | 0.6308 | 0.2628 | 0.3710 | 3.3704 |
| Learned landing head | 0.7353 | 0.8013 | 0.7669 | 0.8889 |

The learned head matches 125/156 landings on these 27 windows; 123 of its 125
temporal matches also have the right anatomical side. On all 39 events,
including unavailable cases, it outputs counts for 32, matches 133/223 human
landings, has 64 unmatched proposals and recalls 0.5964 of the supplied list.
The 32 available-window count MAE is 0.9375; 26/39 total events are within one
step. A good average count can hide incorrect landing times, so both results
must be shown. The full-cohort engineering target is not met.

Cached evidence extraction and head training take roughly one to two seconds
on the current local CPU; this excludes the earlier neural pose inference and
is not new-video latency. Seven focused tests cover held-out-label isolation,
anatomical swap symmetry, abstention, the ambiguous timing band and event
weights, all-event denominators, missing automatic video windows and model
integrity before deserialization.

## Reproduction and video inference

Run from the project root using NumPy 2.4.6, scikit-learn 1.8.0 and joblib 1.6.0.
The original dense pose producer used the heavy MediaPipe checkpoint with SHA-256
`64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b`.
The trained artifact's `deployment_metadata.json` records these dependencies,
feature-code hash, checkpoint binding and model hash.

```sh
python scripts/train_shoe_supervised.py \
  --output outputs/shoe_supervised_2026-10-26/model_reproduction
python -m pytest tests/test_shoe_supervised.py -q
```

The output directory must not already exist. Training produces four fold models,
the deployment model, metadata, reference/evidence bindings, per-event
predictions and metrics. It never edits the reference or earlier predictions.

`scripts/predict_video_shoes.py` adds the learned heads to a completed automatic
video run. Its only prerequisites are the video and its automatically produced
`source`, `pose`, `ball` and `phase` artifacts from
`scripts/analyze_penalty_video.py`. It reads no annotation workbook or answer
file. Set `POSE33_CHECKPOINT` to the identical-hash heavy checkpoint on the
execution machine:

```sh
python scripts/predict_video_shoes.py \
  --analysis-dir outputs/core_contact_training_2026-10-26/full_video_v1_core01 \
  --model outputs/shoe_supervised_2026-10-26/model_v2_audit \
  --pose33-model "$POSE33_CHECKPOINT" \
  --output outputs/shoe_supervised_2026-10-26/video_shoe_smoke_core01 \
  --threads 2
```

The adapter reuses the existing source/actor/contact binding and paired MediaPipe
inference functions, and checks the model and checkpoint before prediction.
Missing contact, actor or approach evidence produces a recorded unavailable
result rather than a fabricated zero count. It preserves the partial-window
scope and reports its own inference time separately from prerequisite stages.
The verified core-01 smoke artifact reports three candidate landings at frames
33, 40 and 50 and a right-foot strike. Its automatic approach window is frames
15-51. It performs 92 paired pose calls across 46 decoded frames, with 89 accepted
actor poses; model loading and this pose/inference stage take 5.1513 seconds on
the lab execution machine. Earlier pose/ball/phase stages are excluded from that
time. The core-01 event was seen during deployment training; this checks the
label-free execution path and is not an independent accuracy test.
