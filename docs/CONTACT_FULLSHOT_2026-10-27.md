# Complete First-Shot Contact Protocol

The old contact extractor inspected the first automatic camera shot with a 12-second cap. In two external clips the actual kicking action occurs after that cap. Increasing training iterations could not recover visual evidence that was never supplied to the contact classifier.

## Changes

- `src/contact_fullshot.py` wraps the unchanged, hash-bound extractor with a declared 60-second maximum for the first automatic camera shot. It retains source frame IDs, automatic cuts, model hashes and feature hashes. It does not search later shots or entire matches.
- `src/contact_actor_support.py` handles the narrow case where the originally selected actor produces zero eligible temporal samples. It checks the existing automatic candidates in their existing order and accepts the first actor with the unchanged temporal support gate inside that candidate's interval. It does not use human timing or outcome labels.
- `src/contact_causal_actor.py` also checks that the stationary-ball cluster contains an observation strictly before the candidate, within the original scorer's 0.65-second approach horizon. This prevents old observations and observations after the candidate from retroactively supporting a different person's later movement. A replacement still needs ankle approach, ball departure and temporal support. Unsupported replacements abstain.
- `scripts/predict_video_contact_fullshot.py` supports a single video or a source-only input manifest. `--single-output` writes the same three files as the original contact CLI. Old 12-second weights require an explicit `--allow-window-transfer`; matched new metadata validates normally.
- `scripts/replay_contact_actor_support.py` creates source-bound feature variants and runs the matched trained head on them. Old predictions and models are preserved.
- Experimental foot camera recovery now requires `predict_video_shoes.py --recover-camera-segments`; it is disabled by default until the input contact window is valid.

## Actual Runs

All ten core videos have automatic first camera cuts before 12 seconds. Rescanning with the 60-second cap yields exactly the same frame IDs and windows in all ten. Their frozen ROI-v5 features were reused only after source SHA, visual model, original feature code, settings, FPS and selected-window checks. Both new protocols were trained with matching metadata using the existing ten labels.

Leave-one-match-out development validation remains 10/10 available, MAE 0.08 seconds, and 10/10 within 0.2 seconds of the existing coarse human timing. This is the previously seen course cohort, not independent external accuracy. The support fallback does not trigger for any of those ten events.

| External event | Original prediction (s) | Selected causal protocol v3 (s) |
| --- | ---: | ---: |
| wc_17 | 3.80 | 3.80 |
| wc_18 | 2.32 | 16.20 |
| wc_19 | 6.72 | 4.32 |
| wc_20 | 3.68 | 17.72 |
| wc_21 | 9.56 | 9.56 |
| wc_22 | 7.44 | 7.44 |
| wc_23 | 5.88 | 5.88 |
| wc_24 | 4.12 | 4.12 |

The six unchanged external windows also reuse exact existing sampled evidence. wc_18 and wc_20 were re-extracted on the 4090, taking 36.4 and 30.2 seconds respectively including model loading. These figures are contact-stage timings, not full-pipeline latency.

For wc_20, the complete window initially selected a short late actor track with zero eligible samples. The next automatically ranked candidate belongs to the persistent shooter track; recovery produces 17.72 seconds without specifying a time externally. The real CLI was smoke-tested in single-output mode and reproduced that result with matching metadata.

Visual inspection of frames around the automatically predicted times confirms that wc_18 and wc_20 now fall within the kicking action. The support-only v2 prediction for wc_19 remained wrong: 6.72 seconds shows the goalkeeper celebrating. The original goal-frame detector abstains near every candidate in that clip, so its thresholds were not relaxed to manufacture a role decision. Instead, inspection of the automatic evidence found that the two late goalkeeper candidates had no preceding stationary-ball observation for 1.16 and 1.64 seconds. The automatically detected shooter candidate has a preceding observation only 0.20 seconds earlier. The causal protocol selects that actor and the unchanged head predicts 4.32 seconds.

Native-resolution crops were generated *after freezing those predictions*. They show the foot approaching the ball at 4.36 seconds and the kicking follow-through by 4.42 seconds, so 4.32 seconds is a plausible coarse estimate. This is a development diagnostic, not an independently annotated timing test. None of these inspections establishes 90% external timing accuracy, and the external reference file does not contain independent frame-level contact labels.

## Rejected Candidate-Window Ablation

An additional v4 experiment in `src/contact_candidate_window.py` confined the final head decision to the recovered automatic ball/ankle candidate's interval while retaining full temporal context. For wc_19, that automatic interval was frames 235-252, or 4.70-5.04 seconds. It produced 4.92 seconds, which the subsequent native crop review clearly places after the kick. Thus the automatic interval itself is late and cannot be treated as reliable supervision. **Do not adopt v4; retain it as a failed ablation.** No thresholds or offsets were subsequently adjusted to this clip.

The selected v3 protocol preserves all ten original core feature arrays and all seven other external predictions relative to v2. Both new actor checks remain inactive on the ten core events; their unchanged grouped validation establishes a regression check, not broad validation of the new branches. The eight external clips have now informed diagnosis and are development examples for these changes.

## Artifacts

- Selected deployment candidate: `outputs/contact_fullshot_2026-10-27/model_fullshot_causal_v3/`
- Eight matched predictions: `outputs/contact_fullshot_2026-10-27/predictions_causal_v3/wc_XX/{prediction,features,evidence}.json`
- Support-only v2 control: `outputs/contact_fullshot_2026-10-27/model_fullshot_actor_v2/` and `predictions_final/`
- Rejected bounded v4: `outputs/contact_fullshot_2026-10-27/model_fullshot_bounded_v4/` and `predictions_bounded_v4/`
- Full-shot-only control: `outputs/contact_fullshot_2026-10-27/predictions/`
- Core training features: `outputs/contact_fullshot_2026-10-27/core10_causal_features/`
- Visual inspection sheets: `outputs/contact_fullshot_2026-10-27/visual_audit/`
- Native wc_19 v3/v4 comparison: `outputs/contact_fullshot_2026-10-27/visual_audit_wc19_native/`

## Full-Pipeline And Shoe Check

The real main-entry wc_19 run used v3, without a ball-specialist change, and completed all stages successfully in 73.52 seconds. It reproduced 4.32 seconds on freshly extracted video features. This run overlapped another GPU inference job, so its latency is not a controlled speed comparison. All downstream required outputs initially remained unavailable because the classical goal detector rejected every shot before the pose stage could select an actor.

`src/shoe_video_recovery.py` now handles only the `selected_person_not_qualified` case by binding the measured contact-stage tracks to the same source SHA, FPS and automatic frame window. It does not assert that the goal scene is qualified, does not bypass source/identity errors or a positive goalkeeper-role rejection, and does not claim the same track is an independently agreeing old pose track. Existing measured-ROI, no-interpolation, actor-rival and camera-chain checks still apply.

A separate fresh wc_19 shoe run completed 358 actual MediaPipe inference calls in 23.54 seconds. It changed the unavailable foot output to **right**, agreeing with the pre-existing foot reference. Its 10 landing candidates describe the automatic pre-contact window only; they are not a validated complete run-up step count. Optional camera-segment recovery remained disabled. The old full-pipeline output is preserved alongside `outputs/pipeline_v4_2026-10-27/wc19_contact_pose_shoes/`.

The new wc_20 full run uses the correct automatic contact at 17.72 seconds. Its shoe stage made 1,936 MediaPipe calls but still abstained. A separate CPU-only replay reused those exact poses and the frozen shoe head, remeasuring camera transforms without reading labels. The only failed camera pairs were 0-1 and 1-2; the contact bracket 885-887 lies wholly inside the automatically chosen continuous segment 2-967. The replay took 27.08 seconds and restored 387 left-side and 424 right-side supported frames across that segment, but the local contact feature remained unavailable.

The frozen head's local foot window is frames 865-899. Thirty of those 35 frames lack a pair of original/reflected poses that both pass actor identity checks. The remaining five frames, 895-899, support the left side only: right-side minimum joint confidences are 0.143-0.175, below the unchanged 0.35 requirement. Consequently there are zero qualified right-side frames near contact, while the fixed head requires at least three for each side. Camera recovery cannot repair those neural pose estimates, and more head-training iterations cannot supply the missing features. The output remains unknown; no thresholds or contact times were changed. This is a measured inference-evidence failure, not proof that a teammate failed to supply labels. The report is `outputs/pipeline_v4_2026-10-27/wc20_contact_camera_segments/report.json`; the cached poses are in the neighboring `wc20_specialist/trained_shoes/` directory.

A final source-only crop ablation used exactly that existing automatic 0.4-second-before/0.25-second-after window and the same measured actor, original 0.45-body-height padding, checkpoint, mirror checks and identity/confidence thresholds. `scripts/evaluate_shoe_contact_crop.py` narrowed the fixed ROI from `[375,208,1573,926]` to `[876,348,1332,756]`; 70 fresh CPU MediaPipe calls took 5.97 seconds. Local supported frames increased to 35 left/19 right, and a foot feature became available. However, the frozen head predicted **left**, contrary to the existing **right** reference. This demonstrates restored pose evidence, not improved foot accuracy. The ablation explicitly records an inference-distribution change and `matched_training_protocol=false`; it is not enabled in the default entry. No further thresholds or time offsets were tuned on this clip. Preserve `outputs/pipeline_v4_2026-10-27/wc20_contact_local_crop/report.json` as a failed classification ablation.

The selected contact protocol's frozen weight SHA256 is `99f3fbd89d2181249aa1b3097fbcc5c07bd4bcadb8906f1fe45446231f7b5182`. The core feature values and fitted head coefficients are unchanged; the gains come from making the real action and correct measured actor available to that head and to later stages. This does not establish independent 90% accuracy or complete teacher-output coverage.

## Complete Eight-Video Default Foot Replay

`scripts/replay_frozen_shoe_protocol.py` replayed the current default shoe entry on all eight original source/pose/ball/phase artifact sets with the selected v3 contact artifacts. Both optional camera segmentation and the failed local crop ablation were disabled. The default policy still prefers an independently bound ball contact and phase when available: wc_22 and wc_23 used that path; the other six used the measured contact actor fallback. Thus the table reports the actual default entry policy, not a forced replacement of every contact source.

The new optional `predict_video_shoes.py --pose-cache` compares source mapping, automatic windows, actor, all ROI-plan entries, camera transforms, contact/phase values, pose checkpoint, original extractor hash, cached pose hash and every decoded frame pixel hash. All eight matched their existing measured caches exactly, so the replay made zero additional neural calls. For wc_18/wc_19/wc_20 these are the newly extracted correct-window caches from this round, not their original wrong-window caches. The 114.57-second total covers remeasurement, decoding and frozen-head replay; it is not fresh end-to-end latency.

| Event | Previous foot | Current default foot | Existing reference |
| --- | --- | --- | --- |
| wc_17 | right | right | right |
| wc_18 | unknown | left | left |
| wc_19 | unknown | right | right |
| wc_20 | unknown | unknown | right |
| wc_21 | right | right | right |
| wc_22 | right | right | left |
| wc_23 | right | right | right |
| wc_24 | left | left | right |

Strict foot agreement, counting unknown as incorrect, increases from 3/8 (37.5%) to 5/8 (62.5%). Coverage increases from 5/8 (62.5%) to 7/8 (87.5%). Accuracy among available outputs is 5/7 (71.4%); this must not replace the strict 5/8 result. Macro F1 over left/right is 0.6136, and balanced accuracy with unknown counted as a false negative is 0.5833. No existing foot decision changed. These are development results against already inspected labels, not a new independent test or whole-project accuracy.

The complete source-hashed replay, per-event predictions and cache receipts are in `outputs/pipeline_v4_2026-10-27/shoes_default_causal_v3/`. Its `scores.json` was generated only after freezing all predictions with `scripts/score_shoe_protocol.py`; it records reference, original prediction and replay hashes. Sixty-two relevant contact/shoe/main-entry tests passed after the default changes.
