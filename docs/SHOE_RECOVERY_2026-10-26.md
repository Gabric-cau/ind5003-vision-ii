# Recovering the Seven Previously Unavailable Shoe Events

The seven missing events were an automatic-pipeline failure, not a missing-video or missing-human-label problem. All seven original MP4 files remain at `/home/boris/workspace/ind5003-vision-ii-experiments/2026-10-12/project/data/raw_clips` on `lab4090`. Each file's SHA256 matches its original dense-experiment receipt and the human 39-event reference. The original 32 usable dense records and all human review windows remain unchanged.

Six events had `independent_contact_unavailable`; Denmark Rakitic had `visible_approach_onset_unavailable`. All seven recorded zero MediaPipe calls. All seven already have human striking-foot and landing observations. Aspas is the third left-foot reference.

## Automatic Recovery

`scripts/recover_shoe_evidence.py` uses no human labels, timings, or manually supplied actor boxes. It takes the original source-bound pose model's automatically selected shooting shot, writes a lossless FFV1 intermediate clip, and checks every decoded BGR frame against the original. The saved offset and hashes preserve the original MP4 frame clock.

The frozen contact model and its visual detectors run on this automatically selected shot. Its actor selection is based on measured visual geometry, not the learned contact classifier's label or the human foot label. The actor is associated back to the original dense pose tracks through at least three simultaneous measured boxes with median IoU at least 0.5. A unique match is required, except for two independently supported track fragments with at most 0.15 seconds of overlap and compatible overlapping boxes. Original observations can fill gaps; new measured observations take precedence. No interpolated pose is treated as a measurement.

The recovered MediaPipe extraction covers the complete automatic shot. Normal and reflected predictions are restored to the original image coordinates. Dense background camera motion is remeasured; transforms from the first failed camera link onward are unavailable. The original source hash and exact original frame indices remain attached to every output. These are automatic candidates, not newly certified human observations.

Two specific recoveries explain the former failures:

- Aspas: the original pose selector chose track 8. The new contact actor matches original track 3 in 40 overlapping frames, with median box IoU 0.965. Track 3 covers all 188 shot frames. The final recovery obtains 373 accepted calls out of 376.
- Denmark Rakitic: tracks 0 and 11 cover frames 165–266 and 264–305. Both independently match the new actor; the three-frame overlap passes the fragment guard. The recovery obtains 266 accepted calls out of 282, instead of retaining only eight sparse actor observations.

The final seven-event evidence is `outputs/shoe_supervised_2026-10-26/recovery_v3`. It combines six unchanged v2 extractions with the separately rerun Rakitic fragment recovery. Earlier attempts are preserved. `recovery_input.json`, `contact/`, and `shoes/{inputs,poses33,summary}.json` retain the complete bindings. Lossless intermediate videos remain in the original remote `shoe_recovery_v1` directory.

## Evaluation Scope

`scripts/train_recovered_shoes.py` keeps the four match groups separate. For outer validation match H, **every recovered training event and validation event** uses contact weights trained without H. The whole-shot pose cache is independent of that contact prediction, so each fold can reconstruct its contact neighborhood without selecting neural evidence with a model trained on H. The contact fold weight hashes and training core IDs are verified against the saved contact report.

The original 32 events retain their original rule-based automatic evidence. The new seven use their automatic shot and fold-specific contact neighborhood. A source-hash match does not imply that an automatic window equals a human review window: those fields are retained separately.

Landing training uses only rows inside the original human review window. Frames outside that window are not negative labels. Evaluation clips predicted landings to that same review window and retains all its human reference landings, including those missed by an automatic contact window. Thus all 39 events and all 223 original in-window reference landings remain in the denominator. Raw predictions outside the review window are retained but unscored. Counts refer to the reviewed interval, not necessarily an entire run-up.

These are previously inspected development matches. Grouped validation is not a new untouched test, and four groups cannot establish broad football-video generalization. These recovery and augmentation experiments do not access the external World Cup holdout.

## Results

| Method | Foot Outputs | Correct / 39 | Foot Macro F1 | Left Correct / 3 |
|---|---:|---:|---:|---:|
| Original 32-event supervised evidence | 32 | 30 | 0.723 | 2 |
| Original frozen heads with recovered inputs | 39 | 35 | 0.721 | 2 |
| Recovered 39, unchanged 57-feature head | 39 | 34 | 0.686 | 2 |
| Recovered 39 plus external foot supervision | 39 | 34 | 0.686 | 2 |
| Contact-aligned 95-feature head | 37 | 34 | 0.721 | 2 |
| Fixed final-support-foot control | 36 | 33 | 0.788 | 3 |
| Support control, original head when unavailable | 39 | 34 | 0.735 | 3 |

The 95-feature head preserves five contact-relative time bins rather than summarizing a whole interval with quantiles. It is a fixed separate experiment, not a parameter sweep. Its checkpoint requires the corresponding temporal feature adapter and must not be loaded into the original 57-feature inference wrapper.

An always-right control would already achieve 36/39 accuracy because the cohort has only three left-foot examples, but its macro F1 is approximately 0.48 and left-foot recall is zero. Overall accuracy alone is therefore insufficient for selecting a foot model.

The support-foot control proposes the opposite of the latest predicted landing within 0.25 seconds before automatic contact. It contains no player-specific rule or human-label routing. It recovers all three left-foot cases, but errors on right-foot cases prevent a net improvement. Its outputs remain an experimental control.

The predefined availability fallback, keeping the original head whenever its original automatic binder succeeds, has exactly the same 34/39 foot score as the recovered head. None of the original 32 foot classifications changed. The new seven alone are 4/7 correct: Kjaer, Spain Ignashevich, Russia Ignashevich and Kuzyaev are correct; Denmark Modric, Denmark Rakitic and Aspas are incorrect. `model_recovery_v3/fallback_audit.json` records this comparison.

This differs from the stronger **frozen-head control**: the original 32-trained shoe fold heads themselves process all recovered inputs, without any shoe parameter updates. That control scores 35/39 overall and 5/7 on the new events. Original 32 foot classifications and full landing lists are checked for exact equality with their original saved predictions. The control is recorded in `frozen_head_recovery_control_v1`.

| Landing Metric, Same Human Review Scope | Original 32 | Recovered 39 |
|---|---:|---:|
| Output events | 32/39 | 39/39 |
| Matched / false positive / missed | 133 / 64 / 90 | 153 / 76 / 70 |
| Precision / recall | 0.675 / 0.596 | 0.668 / 0.686 |
| Temporal matching F1 | 0.633 | 0.677 |
| Count MAE on available outputs | 0.938 | 0.872 |
| Count within one, all 39 events | 26/39 | 29/39 |

Matching retains the original 0.08-second temporal tolerance. Of 153 temporal matches, 150 have the correct anatomical foot. The recovered landing model improves coverage and recall, but these figures do not meet a high-accuracy acceptance claim. The newly trained foot heads are not promoted to the default pipeline.

With recovered inputs and **frozen** landing weights, the corresponding results are 154 matches, 79 false positives, 69 misses, F1 0.675, count MAE 1.026 and 30/39 counts within one. Thus retraining slightly reduces count MAE, but does not demonstrate a clear overall advantage: temporal F1 is nearly unchanged and the within-one rate is lower by one event. Default selection is the original shoe weights plus source-only automatic recovery when the original binder is unavailable. The retrained landing head remains an explicitly separate candidate for count-error evaluation.

## External Foot Augmentation

The existing external collection supplies 15 eligible human foot labels: eight German Cup events and seven Coppa Italia events, excluding conflicted `match_09`. The three left-foot examples are `match_04`, `match_06` and `match_14`. This augmentation run neither uploads nor infers the eight World Cup holdout videos. Labels are separated from the source-only input manifest by `scripts/prepare_external_foot_data.py`.

All 15 original MP4 files, 817,366,107 bytes in total, were uploaded without reencoding and hash-checked. `scripts/extract_external_shoe_evidence.py` runs automatic first-shot contact extraction and normal/reflected MediaPipe inference on measured actor anchors only. There is no manual timing, camera boundary, or actor selection. The full automatic shot is cached, so each original-match outer fold can apply contact weights excluding that same held-out match to every external event as well as every recovered original event.

Extraction completed for all 15 with 3,596 MediaPipe calls and about 365 seconds of measured processing, excluding upload waits. Fourteen events have adequate paired foot evidence; `match_05` remains unusable. In every outer fold, all three additional left-foot events and eleven right-foot events actually enter training. The deployment model consequently fits 53 foot events and 39 landing events. External coarse total-step labels are never converted to frame-level landing supervision or negative examples.

The fixed original 57-feature head and regularization remain unchanged. Augmentation fixes Denmark Modric and Jorgensen but introduces errors on Kramaric and Russia Ignashevich, leaving 34/39 correct and macro F1 0.686. Aspas remains wrong. This is an executed negative result, not a proposal or evidence that more epochs are needed. The existing external left-foot material has now been used, so a lack of such examples is not a justified explanation for stopping the current engineering work.

Artifacts are `external15_data`, `external15_evidence_v1`, and `model_external15_v5` below `outputs/shoe_supervised_2026-10-26`. The model report preserves per-fold actual training IDs, source hashes, and the original-39-only comparison. `external15_usage_audit.json` separately records the actual 14 events and all three left-foot labels used in every outer fold, the unavailable event, and the bound source/report hashes. The augmented checkpoint is not selected for deployment.

## Aspas Failure Audit

The automatic contact maps to original frame 282. The existing human context says frame 281, with the final reported right-foot landing at 278. That context is not independently validated exact ball-contact truth. Inspection of frames 275, 277, 279, 281, 283 and 285 shows the kicking sequence and ball departure in the expected order; there is no evidence of a large contact-time shift.

`outputs/shoe_supervised_2026-10-26/aspas_contact_audit.png` shows the actual decoded pixels. The body and both feet are generally in view, without another player blocking them. Rapid leg crossing and motion blur occur near contact. MediaPipe normal/reflected foot positions disagree by approximately 0.68–0.74 body heights at frames 277 and 279 despite confidence values above 0.8. The strict paired evidence therefore drops several contact-adjacent observations. This is evidence of unstable pose estimates, not proof that the original video lacks the action or that more human labels are required. No visual audit was fed back as a timing or actor correction.

## Reproduction

Run the recovery script with the project runtime, source manifest, source clips, original automatic pose directory and bundle config. The actual remote experiment used the frozen `deployment_v2` runtime and its `lab_config.json`, with the original project as the source. `--reuse-contact` reuses verified contact artifacts while rerunning only the geometric association and MediaPipe stage. A fresh output directory is mandatory.

The local, fully repeatable training/evaluation commands are:

```bash
python scripts/train_recovered_shoes.py --recovered outputs/shoe_supervised_2026-10-26/recovery_v3 --contact-model outputs/core_contact_training_2026-10-26/model_roi_v5 --output outputs/shoe_supervised_2026-10-26/model_recovery_repeat
python scripts/train_recovered_shoes.py --recovered outputs/shoe_supervised_2026-10-26/recovery_v3 --contact-model outputs/core_contact_training_2026-10-26/model_roi_v5 --output outputs/shoe_supervised_2026-10-26/model_temporal_repeat --temporal-foot
python scripts/evaluate_support_foot.py --predictions outputs/shoe_supervised_2026-10-26/model_recovery_v3/oof_predictions.json --output outputs/shoe_supervised_2026-10-26/support_control_repeat.json
python scripts/evaluate_recovered_frozen_shoes.py --recovered outputs/shoe_supervised_2026-10-26/recovery_v3 --contact-model outputs/core_contact_training_2026-10-26/model_roi_v5 --shoe-model outputs/shoe_supervised_2026-10-26/model_v2_audit --output outputs/shoe_supervised_2026-10-26/frozen_control_repeat
python scripts/train_recovered_shoes.py --recovered outputs/shoe_supervised_2026-10-26/recovery_v3 --contact-model outputs/core_contact_training_2026-10-26/model_roi_v5 --external-data outputs/shoe_supervised_2026-10-26/external15_data --external-evidence outputs/shoe_supervised_2026-10-26/external15_evidence_v1 --output outputs/shoe_supervised_2026-10-26/external_foot_repeat
python -m pytest tests/test_shoe_recovery.py
```

Final experiment artifacts are `model_recovery_v3`, `model_recovery_temporal_v4`, and `model_recovery_v3/support_control.json`. Existing datasets, annotations and deployment snapshots are unchanged. The raw-video fallback adapter is a separate integration task and must activate from automatic binding failure, not a hard-coded list of event IDs.
