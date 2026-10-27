# Ball Specialist Full-Video Audit

Decision: retain the trained detector as an **opt-in experiment**, not the
default pipeline. The specialist was called correctly and supplied useful ball
candidates, but the final associated trajectory follows the shooter rather
than the ball. More trajectory points are not an accuracy improvement.

## Compared Runs

- Candidate: remote `2026-10-27-pipeline-v4/wc20_specialist/`.
- Baseline: remote `2026-10-26-contact-v1/full_worldcup8_v3/events/wc_20/attempt_001/pipeline/`.
- Source SHA-256: `860936c4642808cb8fc2c0b09feca2ce72fb5e13607d44e5a007e1bb927fa376`.
- Original COCO model SHA-256: `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`.
- Four-event specialist SHA-256: `12d4b79bf610ae1297732a733455a420c632707ca7a9b7534811432ec5e2f8e3`.
- Both runs decode the same 968-frame automatic first shot at 50 FPS.
  The specialist uses full-frame plus overlapping tiles at `imgsz=1280`.

The compared artifacts are `ball/protocol.json`,
`ball/detections/unseen_penalty.json`, `ball/contact_search/unseen_penalty.json`
and `ball/results/unseen_penalty.json`. No model or decision threshold was
changed during this audit.

## Integration Verification

| Evidence | Baseline | Specialist candidate |
| --- | ---: | ---: |
| COCO full-frame raw detections | 135,737 | 135,737 |
| COCO goal-ROI raw detections | 117,566 | 117,566 |
| COCO ankle-ROI raw detections | 88,586 | 88,586 |
| Additional specialist raw ball detections | 0 | 1,868 |
| Specialist detections incorrectly entering people | 0 | 0 |
| Final associated trajectory points | 0 | 19 |
| Ball-stage elapsed seconds | 86.249 | 215.055 |

All 1,868 specialist detections retain `native_class_id=0` and use canonical
`class_id=32`. Both model hashes and the shared inference-helper hash appear in
the protocol and inference identity. The original COCO output counts are
identical. These facts establish actual integration, not merely a declared CLI
flag. Runtime is one development observation, not a controlled speed benchmark.

## Incorrect Association

The baseline abstains with `no_independent_stationary_ball_departure`. The
candidate enters `candidate_event`, with an automatic contact bracket of
frames 888-892 and representative frame 890 (17.80 seconds). However, 18 of
its 19 selected trajectory points come from `motion_blob`, not neural ball
observations. The single initial neural point is frame 886 at (971.9, 589.6).

Source-image verification shows the selected green trajectory marker on the
shooter's torso or shoulder at frames 890, 896 and 905. In particular, the
selected frame-896 point is approximately (1028.1, 473.1), while the merged
neural ball candidate is (969.3, 476.0). The detector's useful proposal is not
the one followed by the final tracker.

![Source-frame diagnostic: green markers are selected trajectory points](../outputs/ball_specialist_video_audit_2026-10-27/wc20_specialist_contact_sheet.jpg)

The contact-search departure bootstrap also pairs frame 892's ball candidate
near (967.4, 520.8) with a low-confidence background detection at frame 896
near (1030.7, 289.4). A more plausible neural ball candidate exists in frame
896 near (969.3, 476.0). This is a detection-association failure, not evidence
that the source video lacks the ball.

The final outcome rule remains `unknown`, final position remains `null`, and
whole-ball goal crossing is not established. These abstentions must be kept.
The separate old `trained_ball` branch still predicts `left/high` identically
in both runs; those labels do not demonstrate any benefit from the specialist.

## Scale-Dependent Departure Threshold

`src/ball_continuity_v4.py` calculates velocity as stabilized pixel
displacement divided by frame-index difference and requires
`departure_min_speed_px_per_frame=15`. `goalward_departure_support` constructs
the default configuration directly. Neither this gate nor its 160-pixel/frame
upper bound is normalized by FPS or image dimensions. Associated displacement
and backward-miss thresholds are also pixel based.

This is a confirmed resolution/FPS-dependent heuristic, not a demonstrated
unit-conversion typo: the variable explicitly says pixels per frame. A source
at twice the FPS has half the pixel displacement per frame for the same motion.
Source-image coordinates also change with resolution and camera scale. The raw
892-to-896 plausible-ball pair moves about 11.2 pixels/frame, but the bootstrap
uses stabilized coordinates, so that raw number alone does not prove which
individual gate rejected it. No threshold was tuned to make this one event
pass, and no new tracker was implemented.

This inspected event is development evidence. Detector localization metrics,
automatic contact candidates, trajectory length and complete event accuracy
remain separate claims.
