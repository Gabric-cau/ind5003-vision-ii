Vision II — goal-right camera kicking-foot re-detection, 2026-10-11
================================================================

Open index.html or refresh http://127.0.0.1:8886/index.html while the local
read-only gallery server is running. A server is not needed to inspect JSON.

Scope
-----
Exactly 10 Croatia-Denmark primary shots selected by AI inspection of all 39
source-frame samples, matching the user's example of the goal on image right.
This is not identification of horizontally mirrored videos. Do not infer
permanent player footedness from the kicking foot in one observed penalty.

User reference
--------------
The user reviewed the gallery and corrected the earlier all-right statement:
Pivaric is left; the other 9 kicks are right. The direct quote and screenshot
identifier are in user_review_reference.json and the project annotation file.
The user's labels are loaded only in the reporting/evaluation stage, not by
fresh pose inference, approach extraction, or the camera-scoped predictor.

Experiments
-----------
1. Fresh YOLO11m pose on dense, enlarged cached-actor ROI source frames.
2. Horizontal reflection ablation, undoing both pixel x and anatomical channels.
3. Hip-relative motion around independently detected stationary-ball departure.
4. Foot/ball proximity and post-departure projected foot extension.
5. Screen-horizontal approach and projected ball-to-goal-axis approach priors.

The approach shortcut is provisional: run-up geometry does not guarantee the
striking foot. No shoe-ball contact is certified. Goal coordinates are image
geometry candidates, not calibrated ground coordinates; the goal source is an
earlier frame, without an independently validated goal-plane registration.
Ball seeds, camera transforms and cached actor identity also remain candidates.

Comparison against these 10 user references
-------------------------------------------
Old:                     5 matching, 4 nonmatching, 1 unknown.
Previous experiment:     3 matching, 7 nonmatching, 0 unknown.
Fresh normal max motion: 4 matching, 3 nonmatching, 3 unknown.
Strict contact fusion:   0 matching, 0 nonmatching, 10 unknown.
Screen x shortcut:       9 matching, 1 nonmatching, 0 unknown.
Projected goal axis:     9 matching, 0 nonmatching, 1 unknown.
Pose/axis cross-check:   2 matching, 0 nonmatching, 8 unknown.
Camera-scoped shortcut: 10 matching, 0 nonmatching, 0 unknown.

Badelj has no selected automatic goal geometry: its final shortcut is the
weaker screen-horizontal fallback, explicitly labelled in the report.
Kramaric's followthrough pose consensus disagrees with the approach prior;
the report retains that conflict and requires review. It is not silently fixed
by swapping anatomical keypoints. All 10 shortcut candidates require review.

These videos were used to explore the methods after the user's feedback.
The 10/10 candidate match is NOT a held-out result, not global 100% accuracy,
and not evidence that every other angle is solved. Do not use the references
for training and then present this same comparison as an independent test.

Preservation and evidence
-------------------------
All canonical automatic predictions, videos and other feature modules remain
unchanged. camera_scoped_predictions.json is a separate experimental output.
The gallery now separately shows user reference and automatic candidate.
Its previous page is index_before_foot_retest.html in the gallery directory.
The earlier aborted first run is retained; run_v2 is the complete 10-clip run.

protocol.json records model/code hashes and fresh inference configuration.
poses/ stores all original and inverse-reflected source-coordinate observations.
evidence/ shows source-pixel crops with candidate anatomical channels.
results/ and summary.json were written before reference evaluation.
followthrough_analysis.json and camera_scoped_predictions.json are label-free
reanalyses. evaluation.json keeps every attempted method, including abstentions.
validation.json confirms 10 source hashes, 50 input hashes, 10 deterministic
approach replays, exact inference-source snapshots and 53 local links per page.
parameter_sensitivity.json probes +/- 2 contact-anchor frames and +/- 20 px
goal-position shifts. No axis labels changed in this bounded probe; this is
not a confidence interval or proof of broader stability.

Reproduction from the original project root
-------------------------------------------
Use the existing C:/Users/30605/Virtual_Environment/myenv/Scripts/python.exe
environment, with project .venv/Lib/site-packages appended, not prepended.
Fresh inference requires source raw clips, cached actor tracks, upstream ball
and camera measurements, automatic goal geometry, and yolo11m_pose.pt.

python -X utf8 scripts/run_goal_right_foot_redetection.py --output outputs/NEW_UNIQUE_RUN

The later analysis/finalization scripts currently target run_v2. They are
project-local reproducibility tools, not a standalone portable installation.
Inference refuses to overwrite a new run unless --resume is explicitly used.
14 targeted unit tests passed; no claim about the entire project test suite.
The HTML links and source hashes are checked, and videos were decoded in the
gallery build. Browser button interaction is not claimed as visually tested.
