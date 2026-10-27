# External Ball Category Development, 2026-10-26

The frozen ResNet18 descriptor is improved by subtracting its first-frame
features from each of the four temporal bins. The direction head improves from
1/7 to 4/7 on the same Coppa Italia development match. This is a small development
comparison, not an independent-test or full-pipeline accuracy result.

## Fixed Data Boundaries

- German Cup 2022: eight training events.
- Coppa Italia 2025: seven validation events, previously inspected during v1.
- Conflicting `match_09.mp4`: excluded.
- External World Cup 2022: eight videos remain unopened and unevaluated.
- Additional height training: eight existing core events inherit the old human
  ordinal height labels by verified event identity. `core_07` is excluded for a
  new/old height conflict; `core_09` is excluded because its human height is
  unknown. These are existing events, not eight new independent annotations.
- Old direction labels use image-space left/right. They are not mixed with the
  external behind-kicker labels.
- Every input window is the automatically selected first shot. No old/new human
  contact timestamp or manual shot boundary enters feature extraction.

## Results

All rows use the same seven validation events. Macro-F1 always includes all three
classes, including classes a model never predicts.

| Candidate | Direction Correct | Direction Macro-F1 | Height Correct | Height Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Prior temporal appearance | 1/7 | 0.095 | 1/7 | 0.083 |
| Subtract first-frame appearance | 4/7 | 0.556 | 4/7 | 0.383 |
| Dense-motion grid control | 3/7 | 0.357 | 3/7 | 0.302 |
| Original-resolution ball/goal geometry control | 3/7 | 0.200 | 1/7 | 0.083 |
| Add eight core human heights, height only | N/A | N/A | 4/7 | 0.444 |
| Training-majority baseline | 3/7 | 0.200 | 5/7 | 0.278 |

Height has improved relative to v1, but its accuracy remains below the majority
baseline. The augmented height head is retained as an experimental three-class
candidate because its macro-F1 is higher. No claim of reliable height estimation
or a calibrated 3D ball trajectory is supported.

## Training and Export

The encoder remains frozen. Parameters actually fitted are train-only
StandardScaler, train-only PCA with seven components, and balanced multinomial
logistic heads with fixed `C=0.1`. No validation-based hyperparameter search was
performed. The winning feature/head combination is selected using the already
inspected development match and must therefore be treated as development-selected.

The three initial descriptors were declared together. The later ball/goal
geometry control and additional-height-data experiment are separately retained,
including their failed or limited results.

Final candidate artifacts:

- `outputs/external_ball_training_2026-10-26/model_v2/report.json`
- `outputs/external_ball_training_2026-10-26/model_v2/first_frame_difference_weights.npz`
- `outputs/external_ball_training_2026-10-26/height_core_augmented_v2/report.json`
- `outputs/external_ball_training_2026-10-26/height_core_augmented_v2/height_weights.npz`
- Frozen encoder: `outputs/external_ball_training_2026-10-25/model_v1/encoder.pt`
- Failed geometry control: `outputs/external_ball_training_2026-10-26/geometry_head_v2/report.json`

Each model report records source hashes, feature settings, fitted weight hashes,
training/validation metrics and per-event predictions. Numeric NPZ exports use
`allow_pickle=False`; inference verifies encoder and classifier hashes.

## New-Video Entry

```sh
python scripts/train_external_ball_v2.py predict \
  --video /path/to/new_event.mp4 \
  --model outputs/external_ball_training_2026-10-26/model_v2 \
  --backbone outputs/external_ball_training_2026-10-25/model_v1/encoder.pt \
  --candidate first_frame_difference \
  --height-model outputs/external_ball_training_2026-10-26/height_core_augmented_v2 \
  --output /path/to/prediction.json
```

This expects an event clip beginning with the shooting view. It does not replace
the separate camera/shot segmentation of an arbitrary complete match. The
probabilities are uncalibrated. Inference needs no human annotation.

The first local CPU smoke run on `match_10.mp4` took 3.74 seconds including model
load. Its original height/direction predictions were wrong; the smoke verifies
the video interface, not correctness. Probabilities agreed with cache-based
evaluation to within 3e-6. The final combined-model smoke took 3.44 seconds, with
correct low height but incorrect left direction for that example. It is stored
beside `model_v2/report.json`.

Focused tests cover source/holdout guards, train-only preprocessing, numerical
export parity, encoder/head hash guards, human-height provenance, conflicting
event exclusions, temporal-background subtraction and missing geometric evidence.
