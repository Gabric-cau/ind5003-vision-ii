# Keeper Recovery Development Audit

The saved pipeline couples keeper analysis to the strict stationary-ball
departure rule. On a failed ball contact, keeper analysis is skipped even when
the separate trained contact branch has an automatic time candidate. A new
optional `--recover-from-contact` branch in `predict_video_keeper.py` removes
that coupling while retaining source hashes, clock bounds, goal-line role
selection, camera stabilization and observation guards. The default is unchanged.

The existing ROI experiment also requires an already identified keeper and
only replaces detections at already observed frames. It cannot recover a
keeper absent from the original full-frame pose output. The new isolated
`recover_keeper_roi_pose.py` instead selects a crop from coherent automatic
goal geometry and performs fresh YOLO11m-pose inference at 960 pixels with
0/90/270 degree views. New detections are mapped back to source coordinates,
assigned distinct track IDs and passed through the retained role/identity
checks. Human boxes, contact times and result labels are not inference inputs.

## Completed Diagnostics

- CPU contact-only replay: eight previous World Cup development videos,
  zero keeper classification outputs. Six had no pre-contact goal-line keeper
  identity, one lacked a supported pose shot and one lacked post-contact
  trajectory coverage.
- ROI inference: five automatic crops produced new pose observations. Four
  clips initially yielded low/middle head classifications, all `middle`.
  The legacy reference comparison was 1/8 correct, with four unknown outputs.
  These are preserved diagnostic results, **not evidence of successful dive
  recovery or a deployable accuracy improvement**.
- The contact-window audit subsequently identified the main fault: the old
  contact extraction can stop at 600 frames (12 seconds at 50 fps), while
  some actual shots occur after that interval. The recovered people in those
  windows are preparing, not diving. No amount of fitting their preparation
  geometry establishes the eventual dive height.

The four ROI classification attempts all have
`pose_direction_evidence.supported=false`. The updated recovery-only guard
therefore rejects them. Sideways preparation movement alone does not establish
a dive: a supported sustained body transition is required before a recovered
head prediction can be returned. Existing deployed predictions and frozen
training weights are unchanged. This conservative recovery branch currently
supports observed lateral dives; it does not establish vertical high saves.

## Artifacts

`outputs/keeper_contact_recovery_2026-10-27_run2/` preserves the contact-only
predictions, hashes, summary and separate reference evaluation.
`outputs/keeper_roi_recovery_2026-10-27/` preserves raw ROI detections, source
pixel receipts, old-window classification attempts and their evaluation. Those
files predate the sustained-dive guard and must not be quoted as current
guarded outputs. Re-evaluation should first use repaired automatic contact
windows that include the action, then repeat source-only ROI inference and
score frozen predictions. A new match is still required for a final unseen-match
claim because these eight events have been examined during development.

## Full-Shot Recovery and Source-Identity Fix

The completed full-shot run uses
`outputs/contact_fullshot_2026-10-27/predictions_final/`, including automatic
contact candidates at 16.20 seconds for `wc_18` and 17.72 seconds for `wc_20`.
The optional `--contact-predictions` argument verifies the prediction, feature
and evidence video hashes, the automatic camera-shot window, FPS and matching
training protocol. The analysis window is recorded separately from the old
pose file; old observations and their hash are preserved. In these two clips
the original pose shot was already complete; the contact extractor was the
truncated branch.

New automatic goal-ROI inference ran on six clips and generated 3,422 pose
observations. A source-image contact sheet is saved as
`outputs/keeper_roi_fullshot_2026-10-27/source_window_diagnostics.jpg`.
It shows actual dives in the repaired windows. It also reveals that the
appearance-stitched `wc_22` trajectory switches onto the shooter. These images
were used for failure diagnosis, not supplied as boxes or labels to inference.

Four clips initially failed the body check with
`overlapping_selected_source_tracks`. The stitcher selected one observation
per frame but discarded that choice's source ID; the body check then read all
observations from every ID used anywhere in the stitched track. Duplicate
full-frame/ROI detections consequently appeared as multiple selected people.
`goalkeeper_pipeline.py` now preserves `source_track_id_by_frame`, and
`goalkeeper_pose_direction.py` reads only that frame's selected source. Records
without this field retain the old strict behavior. True identity ambiguity,
source-ID changes, gaps and keypoint thresholds are still enforced.

The fix was replayed on the exact cached ROI detections on Linux. A first
macOS replay was correctly rejected because automatic goal geometry differed
by 1-3 pixels from Linux; it is not a completed result. The completed replay is
`outputs/keeper_roi_fullshot_lineage_linux_2026-10-27/`, with frozen prediction
hashes and a separate `evaluation.json` that loads references only after
prediction verification.

| Events | Remaining rejection after the source-ID fix |
| --- | --- |
| `wc_17`, `wc_18`, `wc_20` | Source-ID switch before sustained lateral body evidence |
| `wc_22` | Identity switch; visual diagnostic shows the shooter being followed |
| `wc_23` | Only 0.20 seconds of onset-connected observations; no supported sustained dive |
| `wc_21` | No qualified continuous keeper trajectory or complete camera chain |
| `wc_19`, `wc_24` | No coherent automatic goal ROI; `wc_19` uses the later causal contact rerun below |

**Decision: do not enable ROI recovery by default.** Both before and after
have 0/8 valid ordinal-height outputs and 0/8 all-event height accuracy, counting
unknown as incorrect. Direction output coverage is also 0/8; direction accuracy
is not computed because the reference coordinate convention is unresolved.
The five unguarded head classifications are diagnostics, not valid predictions.
This is a development regression, not an independent test result.

The remaining obstacle is stable keeper identity and pose observation through
the dive, not proof that the existing videos or annotations are missing.
Increasing head-training epochs cannot recover an absent continuous trajectory.
The source-ID bookkeeping fix is retained; the optional recovery experiment is
not promoted.

After the eight-event replay, the causal contact experiment supplied an
automatic `wc_19` candidate at 4.32 seconds. This removed the post-window
failure but still did not produce coherent automatic goal geometry, so no
new ROI pose network inference was attempted for that event. The later
4.92-second interval-constrained candidate was not used after source-image
review exposed its timing problem. The `wc_19` rerun is preserved in
`outputs/keeper_roi_wc19_causal_v3_2026-10-27/`.

`outputs/keeper_roi_fullshot_latest_2026-10-27/` composes the unchanged seven
frozen replay predictions with this latest `wc_19` result. Its protocol links
every original prediction and protocol hash, and its separate evaluation
retains the 0/8 result above. This is an explicit composition of completed
runs, not a claim that all eight were inferred again under a single new run.

Validation: 150 focused tests and five subtests passed across keeper recovery,
source identity, pose/body guards, ball integration and the existing fresh
event analysis. The source-ID fix removes false duplicate-view ambiguity but
does not claim that a detector ID change proves the same physical person.

## Final Cross-View Alias Experiment

A final source-pixel audit found that the first `wc_17` identity change is
duplicate observation of the same goalkeeper by two inference views.
The optional `--allow-cross-view-aliases` flag applies only within ROI recovery.
`keeper_view_aliases.py` requires at least three shared frames spanning 0.04
seconds near the switch, bbox IoU at least 0.60, torso confidence at least 0.50,
torso disagreement at most 0.15 body heights, and torso color similarity at
least 0.90. Any confidently simultaneous separated-body observation vetoes
the pair. Different view tags are required; the selected shooter cannot be
aliased. Only direct aliases of the goal-line seed are considered. Original
tracks and native selected IDs remain preserved, and `identity_verified`
remains false. No reference labels or fitted threshold search were used.

Linux CPU replay reused the exact original ROI detections and source bindings:
`outputs/keeper_roi_fullshot_alias_guarded_2026-10-27/`. Only `wc_17` accepts an
alias, native track 4 to seed 54. Frames 195, 196, 197, 198 and 201 supply the
cross-view evidence. Its onset-connected body sequence now supports
`image_left` at frames 196-201. It still stops at a new, unverified source ID
at frame 202. Thus this is an early direction candidate, not a complete dive
trajectory or a verified physical identity.

| Event | Final alias result |
| --- | --- |
| `wc_17` | One evidenced duplicate-view alias; early `image_left` candidate |
| `wc_18` | Seed 36 and next source 69 have no shared-frame evidence; unknown |
| `wc_20` | Seed 42 and next source 68 are separate people in 14 shared frames; unknown |
| `wc_22` | Shooter ID 0 explicitly excluded; other pair has only one supporting frame; unknown |
| `wc_21`, `wc_23`, `wc_24` | Prior observation/ROI failures remain; unknown |
| `wc_19` | Cache replay retains original contact; separate causal-v3 rerun also remains unknown |

The first alias replay exposed a recovery-wrapper bug: a learned `middle`
candidate overrode the existing whole-window height rejection. The actual
window contains only 21/51 expected observations and ends at frame 213;
its later source IDs are unverified. The recovery wrapper now requires both
the existing complete-window check and one evidenced canonical identity
through the head's selected trajectory before returning a height candidate.
It preserves a supported early body direction independently. The pre-fix
artifact `outputs/keeper_roi_fullshot_alias_linux_2026-10-27/` is retained as
a failed diagnostic: its incidental 1/8 height-label match is not a valid
accuracy improvement.

Final guarded direction coverage is **1/8**, versus 0/8 before. Ordinal-height
coverage and all-event accuracy remain **0/8**, with unknown counted as a miss.
Direction accuracy is not computed because the reference coordinate
convention has not been established. The later `wc_19` causal-v3 timing
attempt is recorded separately above and does not change these counts; this
report does not imply that its new timing was substituted into old caches.
The inference run was frozen before the separate reference evaluation.

Keep recovery and aliases opt-in. This closes the narrow identity-repair
experiment: the source-ID bookkeeping and height guard fixes are retained,
but the evidence does not support promoting the full recovery pipeline or
claiming high accuracy. Validation: **160 tests and five subtests passed**,
including same-person aliases, earlier separation despite later overlap,
insufficient overlap, shooter exclusion and censored-height rejection.

## Optional Ball Specialist Interface

`run_fresh_ball_experiment.py` accepts `--ball-specialist WEIGHTS`,
`--specialist-mode full|tiles` and `--specialist-imgsz 1280`. The original COCO
detector continues to provide people and ball proposals. The specialist adds
only ball proposals: native class 0 is recorded as `native_class_id=0` and
mapped to canonical class 32, with people disabled. Existing candidate gates,
contact search and final decision rules remain active. The shared label-free
helper is `src/ball_specialist.py`; optimizer imports remain compatible.
Both model hashes, inference code hashes, dimensions, tile parameters and
thresholds are recorded. This is an opt-in experiment interface, not a claim
of improved full-video accuracy or a change to the main analysis entry point.

```bash
python -m pytest -q tests/test_keeper_recovery_integration.py tests/test_keeper_supervised.py
python scripts/recover_keeper_roi_pose.py --batch BATCH --model KEEPER_JSON \
  --pose-model YOLO11M_POSE --contact-predictions FULLSHOT_PREDICTIONS \
  --output NEW_OUTPUT --device cuda:0 --imgsz 960
python scripts/recover_keeper_roi_pose.py --batch BATCH --model KEEPER_JSON \
  --reuse-roi-cache NEW_OUTPUT --contact-predictions FULLSHOT_PREDICTIONS \
  --output NEW_CPU_REPLAY --device cpu --imgsz 960 --allow-cross-view-aliases
python scripts/evaluate_keeper_recovery.py --predictions NEW_OUTPUT \
  --references REFERENCES --output NEW_OUTPUT/evaluation.json
```
