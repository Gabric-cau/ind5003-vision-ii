# Targeted Missing Evidence, 26 October 2026

Later runs have recovered usable shoe evidence for all 39 original events and
completed the ten-video batch. The original seven unavailable windows were an
engineering failure, not missing source videos or labels. The tables below
retain the bound earlier comparison snapshot; current experiments and package
status are in `TRAINING_AND_DEPLOYMENT_2026-10-26.md`. The World Cup eight were
subsequently used once under a frozen test protocol. They must not be reused
to select later models; fresh independent test material is now needed.

The existing 39 event reviews, 156 qualitative shot/save entries, landing lists and ten-event workbook are already useful supervision. They should not be redone. The targeted forms below cover only missing coordinate evidence, a few unresolved meanings and teacher-specific camera fields. No request has been sent to anyone, no new human answer has been filled, and no original annotation was changed.

The actual brief, `ind5003_vision_ii_project_requirements.pdf`, physical page 5, requests nine event features and timings for four camera types. It specifies neither a numeric accuracy target nor compulsory model training, 3D reconstruction or metric keeper height. The form fields are an evaluation protocol proposal, not additional marking criteria. Final-position and height definitions must be stated before evaluating them; a first impact, last visible point and play-end location are different observations.

## What Is Established Now

The snapshot is bound to report hashes in `outputs/next_data_requirements_2026-10-26/verified_metrics_snapshot.json`. Contact uses **v5**, not the earlier ROI snapshot. The ten-video full-pipeline batch is a separate ongoing engineering run; its future outputs are not counted as already verified here.

| Required feature | Current evidence and measured result | Data boundary versus work still possible |
| --- | --- | --- |
| Top-down kicker run-up | Frozen 39 have sparse projected candidates, not independently checked continuous paths. | Missing independent ground correspondences and planted-foot reference points prevents a geometric accuracy claim. Continuous tracking and camera calibration can still improve with existing pixels. |
| Kicking foot | Learned model: 30/32 decided correct, 32/39 coverage; two of three left-foot references correct. | Existing human labels suffice for further pose/actor and coverage work. Reuse available external left-foot labels before requesting new ones. |
| Number of steps | 133/223 window landings matched; 26/39 counts within one; count MAE 0.9375 on 32 available windows. | Existing landing labels support improvement. Do not compare window landing counts to the ten full-approach step counts without matching scope and final-support convention. |
| Keeper movement before kick | Ten human onset times/directions already exist; some times equal coarse contact. | Keep developing the motion measurement. Only three onset/precision cases are selected for clarification; do not request all ten again. |
| Keeper dive direction | Preserve-rule fallback: 22 correct, 29 outputs, all 39 retained in denominator. Raw head is weaker on old-rule cases. | More actor continuity and usable dive-phase pixels can help now. Existing direction labels are not missing. |
| Keeper dive height | Low/middle head: 18/29 covered correct; no high-class reference in old39 or external24. | Improve phase/track coverage now. A high-inclusive categorical claim needs genuine high examples and a consistent definition; ball shot height is not keeper dive height. |
| Ball trajectory | Source-bound detector/trajectory candidates exist; external class experiments remain weak. | No independent centre sequences/visibility truth exist. Coarse direction/height labels cannot measure trajectory error or train a box detector. Tracking can still improve without new labels. |
| Ball final position | Core ten contain endpoint descriptions; exact terminal coordinates/visibility are not annotated. | Inspect complete existing clips and available originals first. Only if the state is absent from all available views is longer footage a genuine input gap. |
| Goal/no goal | Frozen39 fusion has 35 outputs, all 35 agreeing with its source-attributed outcomes. | Coverage and full-video inference remain engineering work. External match09 alone needs event/result adjudication before leaving quarantine. |

Supporting contact v5 has 10/10 outputs, MAE 0.0800 s and 10/10 within 0.2 s of the coarse human times, versus rule MAE 0.1133 s. This meets that development timing goal; it is not exact-frame contact validation or new-video population accuracy.

The missing-high statement is specifically about `goalkeeper_dive_height_label`: the club16 have 14 low/two middle/zero high, and the locked World Cup8 have seven low/one middle/zero high. In contrast, World Cup `ball_height_label` has six low/one middle/one high, and club16 ball height already has five high examples. This is an aggregate metadata check only; no held-out pixels, predictions or training were used. Ball-high observations cannot become keeper-high training examples.

External ball experiments have already compared frozen appearance, frame differences, flow-grid motion, detector/goal geometry and added core-height examples. On the same seven-event validation match, frame differences reach 4/7 for direction and height, geometry reaches 3/7 and 1/7, and core-height augmentation reaches 4/7. These results do not justify calling ball understanding solved. They also do not prove that all possible engineering improvements are exhausted.

The camera model currently measures three broad spatial classes: 17/32 correctly classified intervals, macro F1 0.514. This is not the teacher's four-class task. Of these 32 intervals, speed is unknown in 28, slow in three and normal in one. Existing broad spatial labels and exact boundaries remain valid and will be reused.

## Small Existing-Video Packet

All forms are in `outputs/next_data_requirements_2026-10-26/`. The manifest rechecks exact source bytes for 13 unique existing videos. These are not 13 newly requested videos. Several tasks reuse the same short clip. Coordinates, visibility decisions and adjudication answers remain empty.

| Form | Selected existing material | New work only |
| --- | --- | --- |
| `ball_geometry_training.json` | core01 goal, core02 save, core03 crossbar, external match01 | Ball centres and, only for detector fitting, tight boxes across the short departure-to-interaction sequence; explicit occlusion/identity and later terminal visibility. Ground correspondences on core01 and match01 share the same viewing effort. |
| `geometry_evaluation_only.json` | core05 goal, core06 save, core09 miss | Blind per-frame ball/visibility references; ground/sole checks on core05 and core09. These groups must not fit or tune the proposed new geometric model. |
| `camera_targeted_review.json` | Seven intervals listed below | Missing teacher-specific view/speed membership; two unlabelled intervals are checked for possible none-of-four membership, never assumed negative. |
| `targeted_semantic_adjudication.json` | core07, core08, onset cases core01/02/03, external match09 | Resolve only the specified phase, coordinate, timing or event-identity ambiguity. Preserve both old and new descriptions until resolution. |
| `additional_class_inventory.json` | Empty inventory; no existing video falsely labelled high | Collect keeper-high examples only if claiming support for that class; collect more slow-motion views only if targeted review cannot establish the necessary cases. |

Proposed new ball-geometry training groups are Colombia-England and German Cup. Geometry evaluation-only groups are Croatia-Denmark and Russia-Croatia. Camera training/validation keeps German Cup versus Coppa Italia. These videos have already been inspected during development, so the additional labels provide an independent geometric reference, not an untouched final test.

For training, first annotate every second frame in the short flight/interaction sequence, plus the stationary seed and ambiguous/negative detections. A box is needed for detector training; centre-only labels support tracking work. For evaluation, annotate every source frame in the chosen action span, recording invisible states rather than inventing coordinates. Original video frame IDs/PTS define the clock; slowed playback copies do not.

For top-down checks, annotate visible ground landmarks with a documented pitch coordinate convention. Use non-collinear correspondences to fit a reference homography and additional withheld landmarks to measure its error. Mark planted sole points rather than projecting the hip or an airborne ball as ground truth. If too little ground geometry is visible, record that failure instead of manufacturing four points.

## Exact Camera Gaps

| Existing video / interval in seconds | Already recorded | Only missing decision |
| --- | --- | --- |
| match01, 26.12-27.64 | Inside-goal right-side view | Whether it is behind goalkeeper, and whether it is slow motion. |
| match07, 15.08-25.80 | Kicker right-side close view | Slow motion and whether it meets the teacher close-up/right-side category. |
| match10, 14.04-19.32 | Kicker right-side view; slow=true | Close-up and right-45 category membership. Preserve the existing speed label. |
| match13, 14.94-21.66 | Side view; slow=true | Kicker close-up/right-side category membership. |
| match10, 19.32-24.84 | Behind-goal close view, right side | Behind-goalkeeper relation and slow-motion status. |
| match01, 13.88-16.72 | Exact interval, no class label | Review class membership, including explicit none-of-four if appropriate. |
| match10, 5.78-7.68 | Exact interval, no class label | Same targeted negative-class check; unknown is not negative. |

The known missed cuts in match10 at 14.04 and 19.32 seconds are a code/inference-window issue. Those boundaries already exist in the annotations; asking a person to mark them again is unnecessary. Low apparent movement alone is not proof of slow playback. When possible, bind a replay to the same normal-speed action and record the timing evidence.

## Specific Meaning Checks

Core07 has older mid shot/save categories and newer endpoint text `左低扑出`. First determine which object, action phase and view each phrase describes; they may be compatible. Core08 has older screen-right/mid ball labels and newer centre/mid endpoint text. Preserve both until their phase/view convention is clear. Core09's outside-goal unknown category policy is intentional and must not be rewritten into an inside-goal class for convenience.

Core01 and core03 movement starts equal their coarse contact times; core02 differs by 0.05 s. Clarify whether no earlier movement is visible or whether the timestamps are too coarse. A plausible frame bracket is more honest than forcing an exact earlier frame. Contact labels themselves need not be redone for the current 0.2-second development target.

External match09 remains quarantined because the recorded human goal conflicts with the source-attributed missed attempt. Verify that the clip, player and attempt agree before resolving the result. Its other fields are not automatically corrected from a match report.

## Continue Engineering Now

Complete and inspect the ten-video batch; improve actor/ball identity continuity, automatic cut handling, shoe visibility, full-action phase coverage and terminal censoring using current pixels. Reuse completed categories/counts for fixed baseline comparisons, and generate evidence overlays before allocating manual coordinate work. The available data does not justify a global stop or waiting for more labels before doing these steps.

The genuinely blocked claims are narrower: measured ball/endpoint/top-down coordinate accuracy without corresponding independent coordinates; keeper-high performance without high references; a complete four-class camera evaluation without playback/class truth; and a broad unseen-video high-accuracy claim without a frozen independent test. More CPU/GPU inference cannot create those missing reference facts.

The frozen World Cup height and spatial-camera test is now complete; direction was not scored because the annotator's convention is still unspecified. Confirm that convention independently of the stored model predictions, and use fresh independent footage for later model-selection claims. The small packet here is diagnostic; no fixed small number of new labels proves high accuracy across all outputs/classes. Report class counts, coverage and uncertainty, and expand evaluation only according to the final frozen target definitions rather than an invented teacher threshold.
