# Goal/No-Goal Supervised Pilot

Two training cohorts were declared before fitting and evaluated on the same
seven Italian development events. The German-eight head gets 7/7 there, but
predicts goal for all ten core events. Adding the core ten improves their
training fit while reducing Italian validation accuracy to 3/7. Neither head
is promoted or used to override an existing outcome rule.

## Data and Fixed Method

The external split remains German Cup eight training and Coppa Italia seven
validation. Conflicting `match_09.mp4` remains excluded. The World Cup videos,
their outcome targets and their previously evaluated predictions are not used
for training, candidate selection or a new test.

The German training set contains six goals and two no-goals. The Italian
validation set contains six goals and only one no-goal. Their human chat labels
agree with the supplied external-result records and are bound to video hashes.
The ten core workbook outcomes contain four goals and six no-goals; existing
reference loading verifies the workbook, source row, event and current video
hash. These are actual human outcome labels, not inferred model targets.

Both cohorts use the same frozen ResNet18 automatic-first-shot descriptor:
four temporal spatial-feature means minus the first-frame appearance. The
classifier is balanced logistic regression with fixed `C=0.1`, preceded by
train-only StandardScaler and PCA7. The second cohort is exactly the German
eight plus the core ten. No feature or hyperparameter search was performed.

Twenty-three feature rows were reused from existing caches. Only core 07 and
09 needed new encoding, using the same frozen encoder and automatic first-shot
selection. Manual contact times and manual camera boundaries do not enter the
features. First-shot visibility does not prove that the terminal result is
visible: `match_08.mp4` retains its incomplete-action flag, and no independent
terminal-visibility annotation exists for the remaining first-shot windows.

## Results

| Training Cohort | Italian Validation | Italian Macro-F1 | Core Ten Model | Core Interpretation |
| --- | ---: | ---: | ---: | --- |
| German eight | 7/7 | 1.000 | 4/10 | Not fitted on core labels; predicts goal for all ten |
| German eight + core ten | 3/7 | 0.417 | 9/10 | In-sample training diagnostic, not validation |
| Training-majority goal | 6/7 | 0.462 | 4/10 | Constant baseline |

The available existing core rule emits four outcomes, all correct, and abstains
on six. The German-eight model contradicts three of those four known outcomes.
Keeping every available rule output and filling only its unknowns with the
German-eight model would produce 7/10 correct, but filling every unknown with
the fixed training-majority label `goal` produces exactly the same ten outputs
and 7/10. There is no demonstrated learned fallback advantage on the core ten.

The augmented cohort's hypothetical fallback is 9/10, but the same core outcomes
were training targets. Its poor 3/7 league-validation result prevents treating
that training fit as a generalization improvement. Both fallback comparisons
remain analysis only; no existing rule output is replaced.

No existing full-rule outputs are available for the seven Italian videos, so
the report leaves that same-video rule comparison null. It does not substitute
the core rule's conditional accuracy as an Italian validation baseline.

## Artifacts

`outputs/external_outcome_training_2026-10-26/model_v1/` contains the pre-fit
protocol, both numeric NPZ heads, all feature rows, sample/reference provenance,
train/validation predictions, same-core rule comparisons and the complete report.
The supplemental `core10_constant_fallback_control.json` makes the fixed-label
fallback comparison explicit without modifying either fitted model.

The isolated new-video entry is:

```sh
python scripts/train_external_outcome_classifier.py predict \
  --video /path/to/event.mp4 \
  --model outputs/external_outcome_training_2026-10-26/model_v1 \
  --backbone outputs/external_ball_training_2026-10-25/model_v1/encoder.pt \
  --cohort german8 \
  --output /path/to/outcome_candidate.json
```

This emits an experimental auxiliary candidate, an automatic window and
uncalibrated probabilities. It neither verifies a goal-plane crossing nor
certifies terminal-result visibility. Four focused tests verify train-only
preprocessing, numerical binary export parity, holdout exclusion, reference
provenance/conflict rejection and unknown-rule handling. The actual video smoke
is saved as `video_smoke_match15.json`; it is a previously used development
video and is not a new independent test.
