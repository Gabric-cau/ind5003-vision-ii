# Keeper Supervised Development Experiment

The fixed logistic heads use automatic stabilized keeper geometry from the original 39-event bundle. They do not use player names, match IDs, absolute time, existing categorical outputs, or human labels during prediction. Each head has seven numerical inputs. The four validation folds hold out one entire match at a time; scaling and model fitting use the other matches only.

All 39 annotation clip hashes and frame clocks match the frozen source mappings. There are 29 usable keeper tracks and ten abstentions. Original video files were not re-decoded for this cached-evidence experiment. The references contain one human reviewer's qualitative source-image judgments, including 19 left, 13 right, seven center directions, 29 low heights and ten middle heights. There are no high-height references.

## Results

These are development results on previously inspected matches, not an untouched test or general accuracy claim. Missing predictions remain in the 39-event denominator.

| Direction method | Outputs | Correct / all 39 | Covered accuracy | All-event macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Existing rule | 23 | 19 / 39 | 82.6% | 0.514 |
| Learned head | 29 | 20 / 39 | 69.0% | 0.456 |
| Retain rule, fill missing | 29 | 22 / 39 | 75.9% | 0.538 |
| Training-fold majority | 39 | 19 / 39 | 48.7% | 0.218 |

The raw direction head must not replace supported existing decisions: on the same original 23 cases it gets 17 correct versus the rule's 19. The fallback preserves all 23 decisions and adds six outputs, three correct and three incorrect. This policy was proposed after inspecting the raw-head results, so its result is exploratory.

| Qualitative height method | Outputs | Correct / all 39 | Covered accuracy | All-event macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Existing low/mid rule | 14 | 11 / 39 | 78.6% | 0.262 |
| Learned low/middle head | 29 | 18 / 39 | 62.1% | 0.506 |
| Training-fold majority low | 39 | 29 / 39 | 74.4% | 0.426 |

On the 14 cases covered by both height methods, both get 11 correct. The learned head recognizes five middle references across the 39 events, but is less accurate than the majority baseline. It remains a separate model candidate; it does not replace strict keeper height, estimate metres, or support a high class. Partial saved tracks may end before the decisive save phase.

## Artifacts and Use

`outputs/keeper_supervised_2026-10-26/model_v1/` retains the initial result. `model_v2_hybrid_report/` adds the fallback comparison; deployment model bytes are identical. The latter contains `report.json`, `oof_predictions.json`, four fold models and `deployment.json`. Deployment SHA-256 is `3b3ae7cdc452d3c73b9f4c1b4125fae07b5d98994c2c94204f2d21bd007b6fb3`.

```sh
python scripts/train_keeper_supervised.py --output outputs/keeper_new_run
python scripts/predict_video_keeper.py \
  --analysis-dir outputs/my_video_analysis \
  --model outputs/keeper_supervised_2026-10-26/model_v1/deployment.json \
  --output outputs/my_video_keeper
```

The prediction command consumes the automatic single-video pipeline's `source/`, `pose/` and `ball/results/` artifacts. It verifies actual copied video bytes, pose provenance, ball-to-pose hashes, and feature-code identity. It requires no test-video labels and writes both raw learned candidates and the direction fallback to `prediction.json`. The Python interface is `predict_keeper(keeper_dict, model_dict)` in `src/keeper_supervised_model.py`.

The 24 focused tests cover geometry invariance, missing-evidence abstention, fold separation, NumPy/JSON prediction parity, source hash rejection, absence of label inputs, and preservation of existing direction decisions. Cached feature extraction and training took about 0.55 seconds locally; this excludes the upstream video/pose inference cost. No new neural inference or external test-video evaluation was performed for this training run.

A real cached-video CLI smoke run is saved at `outputs/keeper_supervised_2026-10-26/video_core01_smoke/prediction.json`. It rehashed the copied source video and verified both producer bindings. The raw learned direction was center while the rule was image-left; the fallback retained image-left. The height candidate was low. Head inference plus source/hash checks took 0.039 seconds, excluding all earlier visual processing. This is an integration check, not a new held-out accuracy result.

## Further Evidence Needed

More independent matches are needed to test transfer. Complete keeper tracks through the save and independently reviewed direction labels would help distinguish actor/tracking failures from classifier failures. Height requires a consistent phase-based low/middle/high definition, reviewed examples of every class, and ideally a second reviewer. Raising the head's capacity without these observations cannot resolve missing geometry or the absent high class.
