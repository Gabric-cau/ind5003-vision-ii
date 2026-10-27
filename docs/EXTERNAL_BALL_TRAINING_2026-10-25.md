# External ball-category training experiment, 2026-10-25

This isolated experiment trains two small event classifiers from the three-match
release. Its input is a penalty event clip starting with the shooting view. Its
outputs are `ball_direction` (`left`, `center`, `right`) and
`ball_height_category` (`low`, `middle`, `high`). Height is the observer's coarse
category, without a calibrated physical-height or exact arrival-position claim.
The original football pipeline, its saved predictions and its development
metrics are not replaced by this experiment.

## Data and split

Source: [three-match release](https://github.com/RX422040/vision-ii-penalty-analysis/releases/tag/dataset-three-matches-20261025).
The source `metadata/annotations.json` SHA-256 is
`ae88061bca8e3a269b51b592447a462de5dfa59759e4c6b6e46eec69694816f6`.
The preparation script preserves that file and exports a separate manifest with
source hashes, declared label provenance, selection candidates and exclusions.

| Match | Role | Independent events used |
| --- | --- | ---: |
| German Cup 2022 | Training | 8 |
| Coppa Italia 2025 | Development validation | 7 |
| World Cup 2022 | Locked future test group | 0 of 8 |

`match_09.mp4` is quarantined as an entire event because its recorded human and
external goal outcomes conflict. Other labels from that event are not silently
accepted while its identity and result remain under review. World Cup clips,
targets and segments are excluded from training and validation samples; this
experiment does not decode, evaluate or select settings on them. Reading the
release's overall metadata during its earlier audit was not a model test.

The supervision comes from human observations recorded in chat and transcribed
into the release. External match results are separate sources and do not supply
the ball direction or height targets. The 16 club-match direction labels were
already normalized by the observer to the view from behind the kicker; they are
not mirrored again. No horizontal-flip augmentation is used. The World Cup
direction convention remains unconfirmed.

Each event is one training example even though it supplies many frames. All
views, replay shots and frames of an event belong to the same match split. This
first run uses the primary action view only, without the optional behind-kicker
replay augmentation supported by the preparation script. The short 3.32-second
`match_08.mp4` retains its incomplete-action review flag. No proposed AI visual
correction becomes a human label, and kicking-foot labels are not training
targets.

## Automatic input processing

The extractor reads the video directly and reuses
`src.step_pipeline._scan_camera_shots` to find the first shot, with a 20-second
maximum selected-shot duration and scene-cut threshold 30.0. The scanner decodes
the complete event clip to detect boundaries. A first shot shorter than
0.75 seconds is rejected. It does not use human labels or the manual shot ledger
to choose the input time range. This assumes that an already isolated penalty
event clip starts with its shooting view; it is not a general event detector for
an untrimmed match.

The extractor uniformly samples 64 frame indices from that first shot, resizes
them to 512 by 288 pixels and stores JPEGs at quality 95 with individual hashes.
Standalone prediction uses the same processing. Native video frame rates remain
part of the provenance; 64 samples are not 64 independent events. A subsequent
audit found that all 15 automatically detected first-shot intervals matched the
manual primary-shot intervals exactly. This boundary comparison does not verify
ball visibility, a physical endpoint or the completeness of the action.

## Model and controls

A pretrained ImageNet ResNet18 supplies frozen convolutional features; no
ResNet18 weights receive updates. Each frame is pooled into a 2 by 2 spatial
grid, giving 2,048 values. The 64 frames are divided into four ordered temporal
bins, each averaged and then concatenated into an 8,192-value event descriptor.

Standardization and PCA are fitted only on the eight training events. PCA uses
seven components. Two separate balanced logistic-regression classifiers use
fixed `C=0.1`, the `lbfgs` solver, a maximum of 2,000 iterations and seed
`20261025`. The fitted parameters are the scaler, PCA and classification heads.
The validation match does not fit these transformations or select a regularizer.

The report compares each head with a training-majority-class baseline and a
separately fitted first-frame-only control. The latter helps reveal whether
static appearance can explain predictions as well as the temporal descriptor.
Results include event counts, accuracy, macro F1, confusion matrices and per-event
predictions. Probabilities are uncalibrated. Seven validation events support a
development diagnostic, not a reliable generalization estimate.

This is not a ball-detector fine-tune or a trajectory model: the release has no
ball boxes, exact contact frames, continuous ball coordinates or 3D targets.
Match appearance, camera placement, players and scoreboard content remain
possible shortcut signals. The training and validation matches also differ in
their primary camera viewpoints. The original 39-event data are not mixed into
this fit; their outside-goal `unknown` policy differs from this release's raw
direction and height categories.

## Reproduction

Run the commands below from the project root with a Python environment containing
NumPy, OpenCV and PyTorch, plus the existing video pipeline's pandas, SciPy and
PyYAML dependencies. Training additionally needs scikit-learn and torchvision.
The initial run uses two CPU threads on the remote lab machine;
the existing 4090 workload is not used for this experiment. CPU execution is
sufficient for frozen-feature extraction and these small classification heads.

Set the data location to the extracted release, then prepare the manifest and
verify the selected media bytes:

```sh
DATASET_ROOT=/path/to/Penalty_3_Matches_Export_20261025
python scripts/prepare_external_ball_training_2026_10_25.py \
  --annotations "$DATASET_ROOT/metadata/annotations.json" \
  --view-policy primary_action \
  --verify-media-sha256 \
  --output outputs/external_ball_training_2026-10-25/manifest.json

python scripts/train_external_ball_classifier.py extract \
  --manifest outputs/external_ball_training_2026-10-25/manifest.json \
  --output outputs/external_ball_training_2026-10-25/frames_v2
```

Use the pretrained ResNet18 checkpoint supplied for this run, not an arbitrary
checkpoint with the same architecture. The saved checkpoint SHA-256 is
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.

```sh
python scripts/train_external_ball_classifier.py train \
  --frames outputs/external_ball_training_2026-10-25/frames_v2 \
  --backbone outputs/external_ball_training_2026-10-25/resnet18-f37072fd.pth \
  --output outputs/external_ball_training_2026-10-25/model_v1 \
  --threads 2
```

Extraction and training refuse to overwrite an existing output directory. For a
new reproduction, choose new output-directory names and pass the matching frame
directory to training. The source annotation file is never rewritten.

The model directory contains `classifier_weights.npz`, `encoder.pt`,
`feature_cache.npz` and `report.json`. The frozen TorchScript `encoder.pt` permits
offline prediction with PyTorch without importing torchvision. The numerical
classifier export also avoids requiring scikit-learn for prediction:

```sh
NEW_EVENT_VIDEO=/path/to/new_penalty_event.mp4
python scripts/train_external_ball_classifier.py predict \
  --video "$NEW_EVENT_VIDEO" \
  --model outputs/external_ball_training_2026-10-25/model_v1 \
  --backbone outputs/external_ball_training_2026-10-25/model_v1/encoder.pt \
  --output outputs/external_ball_training_2026-10-25/new_event_prediction.json \
  --threads 2
```

New-video prediction takes no manual annotations. Its timing includes video
selection/decoding and model loading, with feature-extraction time also reported
separately. Do not use the locked World Cup clips for this demonstration before
the test protocol and unresolved label conventions are finalized.

## Observed results

The completed `model_v1/report.json` records the following development-validation
results. Both trained heads fit all eight training events correctly, but neither
improves on its training-majority-class baseline in the seven-event validation
match. This experiment is not promoted to the default pipeline.

| Output | Temporal model correct | Temporal macro F1 | Majority correct | Majority macro F1 | First-frame control correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ball direction | 1/7 (14.3%) | 0.0952 | 3/7 (42.9%) | 0.2000 | 3/7 (42.9%) |
| Ball height category | 1/7 (14.3%) | 0.0833 | 5/7 (71.4%) | 0.2778 | 2/7 (28.6%) |

These comparisons concern the two controls in this experiment, not the original
project pipeline. The small training set and change in match/camera view are
plausible limitations, but this run does not establish which caused the failure.
The high training score is not evidence of a reliable video-understanding model.
No second experiment or World Cup test has been run as part of these results.

On the remote CPU with two threads, feature extraction took 2.72 to 2.94 seconds
per 64-frame event. Fitting the scaler, PCA and two heads took 0.128 seconds after
features were available. These are not end-to-end video inference timings.
The run used Python 3.11.15,
PyTorch 2.7.0, torchvision 0.22.0 and scikit-learn 1.9.0.

Local offline prediction was tested on the 24.84-second `match_10.mp4` with the
exported encoder, without torchvision or manual timing input. Python 3.13.15
completed video decoding, shot selection, model loading and prediction in
3.777 seconds, including 0.980 seconds for feature extraction. A separate
Python 3.14 run took 4.148 seconds. These are individual timing observations,
not a hardware benchmark. The selected frames and predicted classes matched
the remote run; the largest probability difference was below 0.000001.
The predictions remained incorrect against this event's coarse reference labels.
Current local PyTorch emits a TorchScript deprecation warning; the saved model
passed these actual runs, and the original training environment remains recorded.

The 25 focused tests pass: match isolation, conflict/unknown exclusions,
training-only preprocessing, frozen encoder state, source-selection guards and
exported prediction parity. The original project algorithm was not changed.

The report retains the checkpoint, frame-manifest, source-annotation, classifier
and training-script hashes, plus confusion matrices and every event prediction.
The trained classifier archive SHA-256 is
`33aea669a492e8524cbe384fd5457ced91451309a47ddb375f0e6c3bdc1eb09d`.
