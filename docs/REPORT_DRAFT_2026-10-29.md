# IND5003 Vision II: Penalty Video Analysis (Review Draft 2026-10-29)

**Group:** IND5003 Vision II group 4
**Status:** Review draft. Replace the placeholders for group members, contributions, references and final submission links after group review.

This report analyses penalty-kick videos from four 2018 FIFA World Cup shoot-outs. The system combines source-bound video slicing, pose estimation, object detection, temporal tracking, camera-aware geometry and scoreboard evidence. The result is an evidence-first feature table: a value is emitted only when the source frames and actor identity support it; otherwise the system keeps an explicit unknown or abstention.

## 1. Literature Review

Penalty analysis is a temporal computer-vision problem rather than a single image classification task. The requested outputs depend on the order of events: the kicker approaches the ball, contacts it, the ball travels, the goalkeeper reacts and the play ends. A useful system therefore needs person and ball localisation, identity continuity, temporal phase detection and a coordinate convention for image-space or field-space measurements.

Pose estimation provides body landmarks for the kicker and goalkeeper, but broadcast footage contains occlusion, viewpoint changes, motion blur and incomplete bodies. Object detection can localise the ball and people, but a frame-level detection does not prove that the same ball or person is tracked through the whole event. We therefore use temporal continuity checks, duplicate-view agreement and source-frame evidence before reporting a feature.

The project also uses camera-aware reasoning. White-line and goal geometry provide a partial field reference, while optical-flow and RANSAC checks identify camera motion. These are useful for image-plane comparisons and sparse ground projections, but they do not create a calibrated 3-D measurement when the required reference is not visible.

Our design follows an abstention principle. A missing endpoint, an ambiguous dive phase or an unresolved replay interval remains unknown instead of being converted into a guessed class. This makes the coverage limits visible and allows the report to distinguish model failure from insufficient evidence.

## 2. Data Overview and Preparation

The main cohort contains 39 penalty events from four 2018 World Cup shoot-outs: Colombia–England (10), Croatia–Denmark (10), Spain–Russia (9) and Russia–Croatia (10). Each event is represented by a source-bound clip or saved result record with frame numbers, frame rate, clip offset and SHA256 provenance where available. The group also supplied core-event manual reviews, a 24-row preparation package for existing events, six short clips with 835 goalkeeper boxes, and additional match footage used for development checks.

The labels cover contact time, kicking foot, visible approach steps, goalkeeper movement and direction, qualitative ball direction/height, outcome and selected camera observations. The 24-row package updates existing events; it does not create 24 new independent penalties. The 835 goalkeeper boxes are valid training boxes for an `active_keeper` detector, but their development videos were already used in model development and are not an independent final test.

The pipeline keeps event-level partitions. Frames and views from one penalty stay in the same partition to avoid leakage. The eight-event World Cup check is a frozen held-out check for the primary pipeline, while the final ball and goalkeeper detector experiments were evaluated on exposed development groups. Raw recordings, weights and personal-account data are not committed to the public repository; the reproduction package documents the required local inputs and environment.

## 3. Exploratory Data Analysis

The feature table shows that observability varies more than the nominal number of labels suggests. Current candidate coverage over the 39-event development cohort is:

| Output | Candidate coverage | Interpretation |
| --- | ---: | --- |
| Contact time | 33/39 | A supported contact window exists for most events. |
| Kicking foot | about 13–14/39 | Anatomical foot evidence is sparse and often occluded. |
| Approach/step candidates | 32/39 | Candidate counts exist, but comparable landing validation is much smaller. |
| Goalkeeper pre-kick movement | 29/39 | Mostly image-space movement with source-time support. |
| Goalkeeper dive direction | 23/39 | Screen-left/right/centre convention is retained. |
| Ball trajectory candidate | 33/39 | Tracked observations exist, but independent ball-centre truth is absent. |
| Score/outcome fusion | 35/39 | Development agreement is high, but this is not held-out accuracy. |
| Strict dive height | 0/39 | Required body/ground/contact evidence is not sufficiently observable. |
| Strict event-final ball position | 0/39 | Last-visible observations are censored before a verified terminal state. |

The camera audit contains 320 candidate intervals. Only a small number have role-specific view candidates, and automated slow-motion status remains unknown. Sparse ground projections cover some events but do not establish a continuous calibrated top-down trajectory.

The final training audit also provides useful negative evidence. A ball detector control and an augmented mixed arm both completed 10 epochs, but the new arms fell below the initializer on the Arsenal and core development groups. An active-keeper control and photometric arm both completed 12 epochs; photometric augmentation regressed on the external-match group. The selected models therefore remain the existing initializer for the ball path and the control detector for the standalone active-keeper experiment.

## 4. Research Questions

### RQ1. Which requested features are observable from broadcast views?

Contact, pre-kick movement, ball-track candidates and outcome fusion have the broadest coverage. Dive height and event-final ball position are limited by missing planted-body references, occlusion, camera cuts, source boundaries and loss of ball identity. The result is a measurement boundary, not simply a classifier threshold problem.

### RQ2. Do pose, temporal rules and augmentation improve the requested outputs?

Temporal continuity and paired normal/mirrored pose checks improve auditability and add some candidates, but they do not create evidence where the source frame is missing. The final augmentation comparison did not improve the ball or goalkeeper development metrics, so the new arms were not promoted. This prevents a visually plausible but unsupported model change from entering the default pipeline.

### RQ3. Can the system combine heterogeneous evidence while remaining reproducible?

Yes for the saved-result workflow: source hashes, frame timestamps, producer manifests, component records and abstentions can be replayed locally. The eight-event held-out primary-pipeline check completed 8/8 runs, but its strict kicking-foot accuracy was 3/8, and goal outcome and keeper height had zero valid coverage. The latest detector experiments still need a genuinely unseen event cohort before their accuracy can be claimed.

## 5. Overview of New Techniques

The pipeline uses YOLO pose and object detection, MediaPipe Pose Landmarker, temporal identity association, ball continuity checks, camera-motion estimation, RANSAC/field geometry, goalkeeper phase and direction rules, scoreboard icon transitions, and a review UI with source-frame evidence. Each module preserves its input provenance and returns an explicit abstention when its contract is not satisfied.

The October 18 landing replay repaired an implementation error that truncated a supported stance phase after three low-motion frames. The replay preserved the old candidates and added candidates on several clips, but the remaining landing mismatch shows that this is an engineering repair rather than a solved step-count model.

The final ball experiment compared the existing initializer, a full-image control and a mixed crop/photometric/affine augmentation arm. The initializer remained selected. The final goalkeeper experiment trained an `active_keeper` detector from the 835 reviewed boxes and compared control against photometric augmentation. The control arm remained selected. A separate provisional coverage view exposes 17/39 legacy image-plane height candidates and 27/39 last-visible ball observations; these values are explicitly not strict accuracy and do not overwrite the main feature table.

## 6. Code Structure and Reproducibility

Reusable components are in `src/`; experiment entry points and audit scripts are in `scripts/`; regression tests are in `tests/`; configurations and dependency files are in the repository root and `configs/`. Saved-result EDA can be reproduced with `requirements-analysis.txt`. The portable Dell RTX 2060 package documents a fresh Windows installation and a label-free video smoke run. The final training summaries include fixed sampling manifests, optimizer-step counts, weight hashes and selection rules.

The focused final audit passed 53 tests. The repository does not include original match videos, private chat attachments, model weights or ZIP archives. A teaching-team reproduction package must place the permitted course data and any required model files in the documented local paths or SharePoint location. The current public checkout supports saved-result analysis and review pages; it is not a claim that every raw-video extraction stage can run without those external inputs.

The principal limitations are incomplete raw-video coverage, single-reviewer labels for parts of the 39-event reference, no independent ball endpoint coordinates, no independent metric dive-height reference, unresolved camera playback labels and exposed development cohorts. The final video and report should show one supported example and one abstention, explain the denominators, and never convert provisional coverage into accuracy.

**AI assistance disclosure for review:** AI assistance was used to help inspect existing artifacts, write and test code, organise the report and generate this draft. The group must verify every claim, add the real member contributions and references, and describe any additional assistance before submission.
