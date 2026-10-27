# Human-Supervised Spatial Camera Pilot

This experiment trains a small spatial-view classifier from the supplied human
camera labels. It does not complete the teacher's four requested camera/playback
classes, and it does not treat missing annotations as negative examples.

## Data and Meaning

After excluding the locked external World Cup match and conflicting
`match_09.mp4`, 32 exactly bounded human camera intervals remain: 18 from the
German Cup and 14 from Coppa Italia, belonging to 15 penalty videos. They are not
32 independent matches or penalties. Replays and repeated camera setups within
a match are correlated.

Only `human_user_observation_in_chat` labels with exact clip-local start/end
seconds enter training. Partial labels use their exact labelled sub-interval;
an unresolved partial label is excluded. Original source-video timestamps are
not substituted for clip-local time.

The auxiliary classes are deliberately broader than the course targets:

| Auxiliary Class | Human Meaning | Course Interpretation |
| --- | --- | --- |
| `behind_kicker` | Behind the kicker, including a distant view | Behind-kicker candidate only |
| `goal_area` | Behind, beside or inside the goal | Does not establish behind-keeper slow motion |
| `other_side` | Wide side view or a view beside the kicker | Does not establish right 45 degrees or slow-motion close-up |

Playback labels in these admitted segments are 28 unknown, three slow-motion,
and one normal-speed. A playback head is not trained from this distribution.
Unknown speed stays unknown; encoded FPS is not used as a slow-motion label.

## Actual Training

Eight equally spaced frames are sampled strictly within each labelled interval,
resized to 512x288 and encoded by the existing frozen ResNet18. Their spatial
2x2 features are averaged into 2,048 dimensions. StandardScaler, PCA7 and a
balanced multinomial logistic classifier with fixed `C=0.1` are fitted inside
each fold. The encoder is not fine-tuned.

Two folds hold out an entire match. No event, replay, source video or frame from
the held-out match enters fitting or preprocessing in its fold. A separate
deployment head is fitted on all 32 intervals after evaluation; its training
predictions are not reported as held-out predictions.

| Evaluation | Correct | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Three auxiliary classes, whole-match OOF | 17/32 | 53.1% | 0.514 |
| Fold training-majority class | 7/32 | 21.9% | 0.166 |
| Behind-kicker vs other, same OOF predictions | 20/32 | 62.5% | 0.624 |
| Always predict other, binary control | 20/32 | 62.5% | 0.385 |

The three-class confusion matrix has human classes as rows and predicted classes
as columns, in `behind_kicker`, `goal_area`, `other_side` order:

```text
9  1  2
1  6  1
8  2  2
```

Side views remain the main failure: the two matches contain different side-view
types. This is a development pilot across only two matches. Segment boundaries
are supplied human labels during OOF evaluation, so these results do not measure
automatic boundary detection or end-to-end four-class accuracy.

## Artifacts and Inference

Model, fold weights, features, source hashes, the raw-to-auxiliary mapping and
per-segment OOF predictions are in:

`outputs/external_camera_training_2026-10-26/model_v1_final/`

```sh
python scripts/train_external_camera_classifier.py predict \
  --video /path/to/new_video.mp4 \
  --model outputs/external_camera_training_2026-10-26/model_v1_final \
  --backbone outputs/external_ball_training_2026-10-25/model_v1/encoder.pt \
  --output /path/to/camera_predictions.json
```

This new-video path automatically detects cuts, subdivides each detected interval
into windows no longer than two seconds, and applies the all-data deployment head.
These are bounded prediction windows, not verified camera-shot boundaries. It requires no human labels, returns
unverified spatial class candidates and uncalibrated probabilities, and keeps
playback speed unknown. The final smoke result is `video_smoke_match10_sequential.json` in
the model folder. The initial unbounded smoke, retained under `model_v1/`, took 5.39 seconds but missed cuts
inside 9.2-24.84 seconds, motivating the fixed inference-window bound. No classifier
weights or hyperparameters were changed for this repair. The external World Cup
videos remain unopened.

The two-second-window path processes 13 windows covering all 24.84 seconds of
`match_10.mp4`. Sequential frame decoding reduced its local CPU time from 14.40
to 4.36 seconds, with exactly identical sampled frame IDs, predicted labels and
probabilities. The original training script is archived as `training_source.py`
and its SHA256 matches the training report; inference records its current source
hash separately.

Focused tests cover locked-match exclusion, human partial-window handling,
unknown-label exclusion, whole-match grouping, source-video overlap guards,
train-only preprocessing and numerical export parity.
