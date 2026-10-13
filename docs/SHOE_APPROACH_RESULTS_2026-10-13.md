# Continuous shoe observations: what changed on the original recordings

This run uses the original four course recordings and the same 39 attempts.
No additional footage, player-footedness lookup or human reference label enters
prediction. The previous 39-row table remains frozen for comparison.

## Why extend the observation window?

The earlier shoe experiment started near contact. It could inspect a kick but
could not inspect most of the visible approach. The new run begins at the
existing proposed approach onset, keeps independent normal/mirror VIDEO
trackers, and uses one fixed person ROI within the selected shot.

All 39 input chains were checked. Six events lack an independent ball-contact
bracket and one lacks a supported approach onset. The other 32 yield 2,121
decoded frames and 4,242 new neural calls. Of those calls, 4,158 pass the original
actor-identity guard. No exact repeated source pictures were found in these
intervals. This does not establish native capture rate or real-time playback.

The [protocol](SHOE_APPROACH_PROTOCOL_2026-10-13.md) defines the frozen landing
settings. Shoe observations are compensated through supported camera chains.
A proposed landing requires a measured swing followed by several low-motion
observations of the same foot, paired normal/mirror agreement, body progression
and uninterrupted source evidence. Initial planted support is not a new event.

## Results

The experiment retains 12 visible stance-transition proposals in seven events.
Twenty processed windows contain no accepted proposal; five abstain because
anatomical channels are temporally ambiguous; seven lack an eligible input
window. A zero proposal count means the rule found nothing meeting its criteria,
not that the player took zero steps. Missing output remains null.

| Event | Previous ankle extrema | New stance proposals |
|---|---:|---:|
| Colombia-England / Cuadrado | 7 | 1 |
| Colombia-England / Rashford | 15 | 3 |
| Colombia-England / Trippier | 4 | 1 |
| Croatia-Denmark / Eriksen | 5 | 1 |
| Croatia-Denmark / Pivaric | 13 | 3 |
| Spain-Russia / Smolov | 8 | 1 |
| Russia-Croatia / Smolov | 6 | 2 |

Every proposal has exactly three shared stance observations: 0.0667 seconds at
30 FPS. That satisfies the engineering minimum, but low image motion may also
occur during an airborne change of direction. Rashford frame 275 and Pivaric
frames 245 and 310 remain visually ambiguous. Neither column is a verified
total step count; the new column is a conservative subset within the visible
window, and the onset itself has not been independently verified.

The dense strike rule returns five foot candidates: Dier, Spain-Russia Smolov,
Dzagoev and Vida as right, Pivaric as left. Four agree with the existing table;
Dzagoev is new. Rashford's earlier right candidate is not reproduced by this
longer-history run. This sensitivity matters: more frames do not consistently
improve coverage. The region check uses median actor height over the approach,
whereas the earlier short-window experiment used preparation height. This is an
image-space scale, not a metric body measurement.

For Dzagoev, the paired shoe-region support is frame 262, between stationary-ball
frame 261 and observed departure frame 263. Its 66 decoded frames, actor/rival
anchors, restored coordinates, camera chain and contact producer were separately
rechecked against the saved inputs. The evidence supports an unverified right
candidate, not an observed physical touch.

The source-bound
[foot-only supplement](../outputs/fresh_course_experiment/shoe_approach_foot_supplement_v2/foot_candidates_39.csv)
retains the earlier thirteen candidates and adds Dzagoev: ten right, four left
and twenty-five unknown, with no decided-method conflicts. Fusion independently
redecodes all eligible source intervals, reconstructs mirrored points and
duplicate flags, checks original actor/rival boxes and frozen contact/camera
inputs, and binds each result to both producers and its selection protocol.
The earlier main table and its step counts are preserved. This is a one-event
increase in candidate availability, not an established accuracy improvement.

## Inspect and Evaluate

[All 39 comparisons](../outputs/fresh_course_experiment/shoe_approach_analysis_v2/comparison_39.csv),
[source-frame boards](../outputs/fresh_course_experiment/shoe_approach_analysis_v2/review_v2/index.html)
and [frozen predictions](../outputs/fresh_course_experiment/shoe_approach_analysis_v2/prediction_freeze.json)
are retained. Each displayed frame is checked against the exact decoded-pixel
hash used for inference. Crops are enlarged for inspection, without claiming
additional visual detail.

Genuine reviewers should first use the
[blind original-video viewer](../outputs/fresh_course_experiment/shoe_approach_analysis_v2/blind_review/index.html).
It requires the permitted original context clips locally. It shows no model
overlays, predicted counts or predicted striking feet. Reviewers independently
record observable intervals and landing frames in separate copies of the
[reference template](../outputs/fresh_course_experiment/shoe_approach_analysis_v2/human_reference_template.json),
then freeze those labels before inspecting prediction boards.

The evaluator matches events one-to-one within an explicit time tolerance.
It reports count error only for matching observed intervals and retains
reviewer disagreements. At this release, **zero events have independent human
review**: agreement, precision, recall and step-count error are null in the
[pending evaluation](../outputs/fresh_course_experiment/shoe_approach_analysis_v2/evaluation_pending.json).
This is a development comparison on previously inspected videos, not a held-out
test or evidence of a particular course grade.

## Reproduction

The fresh neural run uses `requirements-pose33.txt` and the official Heavy task
model. The model ran on CPU; no training or fine-tuning was performed. Original
video/model bytes are excluded from Git, while hashes and source snapshots are
retained in the experiment output.

```sh
python scripts/run_fresh_shoe_approach_experiment.py \
  --clips-dir data/course_context_clips_2026-10-12 \
  --pose-dir outputs/fresh_course_experiment/pose_v7 \
  --phase-dir outputs/fresh_course_experiment/phase_selected_v1 \
  --ball-dir outputs/fresh_course_experiment/ball_selected_v1 \
  --model models/weights/pose_landmarker_heavy.task \
  --output-dir outputs/my_dense_shoe_run

python scripts/analyze_shoe_approach.py \
  --dense-dir outputs/my_dense_shoe_run \
  --phase-dir outputs/fresh_course_experiment/phase_selected_v1 \
  --baseline-bundle outputs/fresh_course_experiment/final_bundle_v3_shoe_fusion \
  --output-dir outputs/my_shoe_analysis
```

Use a new output directory on each run. `scripts/evaluate_shoe_landings.py`
provides `freeze`, `template` and `evaluate` commands; prediction generation never
reads human reference rows. The local historical analysis-v1 and released
analysis-v2 retain their own code snapshots; numerical proposals are unchanged.
