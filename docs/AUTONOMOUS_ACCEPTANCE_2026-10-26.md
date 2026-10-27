# Vision II requirements, references and acceptance

This audit binds the next experiments to the actual course brief and the supplied
annotations. It is an engineering protocol, not a marking prediction. The
machine-readable companion is `AUTONOMOUS_ACCEPTANCE_2026-10-26.json`.

## Authoritative scope

`ind5003_vision_ii_project_requirements.pdf`, physical page 5, specifies a penalty
event from the kicker walking to the spot until play ends. The script should aim
to extract the nine features below. A separate script must return intervals for
four camera views: behind the kicker, 45 degrees on the kicker's right,
slow-motion kicker close-up, and slow-motion behind the goalkeeper. Manual event
slicing is permitted; subsequent processing must be scripted and reproducible.
Basic EDA follows feature extraction.

The brief gives **no numeric accuracy target**, prescribed model, requirement to
train a model, metre-based height definition, or compulsory 3D reconstruction.
It does not specify a hidden new-video evaluation for Vision II. New-video
inference is a useful engineering objective requested by the group, not a
separate grading rule inferred from the Vision I section. All nine outputs and
the separate camera task remain in scope; an unknown output is an honest failure
or partial observation, not evidence that the feature is solved.

Pages 5-7 require a ten-minute presentation, a PDF report of at most 25 pages
excluding cover/contents/references/appendices, font at least 10-point Times New
Roman, and all code/data on SharePoint. The report has six explicit sections:
literature review, data overview/preparation, EDA, two or three research
questions, overview of new techniques, and code structure. It should use proper
English prose, visuals, iteration and discussion of limitations. The video should
show findings upfront, use visuals, avoid artificial speed-up and avoid an AI
voice. The teaching team must be able to reproduce results without the group's
personal accounts. The brief states 30 October at 23:59 as the submission
deadline. These are document requirements, not confirmation of any submission.

The saved teacher-update note dated 24 October also emphasizes understanding,
validation and reproducibility. That note is a secondary saved summary, not a
fresh email/Canvas verification in this audit. The brief is the primary source.
Existing `README.md` and `SUBMISSION_READINESS.md` contain earlier snapshots;
their older statements about training or available media must not override the
new experiment artifacts. This audit does not edit those files.

## Output contracts and available supervision

| Required output | Evaluation contract | References actually available |
| --- | --- | --- |
| Kicker run-up trajectory, top-down | Actor identity, observation interval and ground-plane transformation must be explicit. Image-space hip motion alone is not a measured top-down ground path. | Pose tracks and automatic projections exist. No independent top-down path coordinates or surveyed calibration are present in the audited packs. |
| Right/left kicking foot | Anatomical striking foot for this attempt, not preferred foot or screen side. | 39 human event labels; ten earlier user-reviewed foot labels overlap those events. External 24 also have human chat labels. Pose pack has eight right-foot examples, not balanced independent pose truth. |
| Number of steps | Declare approach onset/end and whether final support placement counts. Keep partial observed counts separate from full approach counts. | 228 human landing events in the old 39, with 223 inside their recorded review windows. Core ten provide full-approach counts; these disagree with the old window counts on six events because scope differs. External 24 have counts, without landing frames. |
| Keeper movement before kick | Correct actor, time basis and pre-contact interval; categories and continuous displacement are different targets. | Core ten have direction and movement-start seconds. Two starts equal recorded contact; one is only 0.05 s earlier. Three older user-reviewed actions concern specific displayed views, not a complete 39-event movement truth set. |
| Keeper dive direction | Preserve screen-direction versus actor-direction convention and the actual dive phase. | 39 human screen-direction labels; 24 external human labels with viewpoint issues described below. Core ten movement directions must not be silently substituted for final dive direction. |
| Keeper dive height | Declare an ordinal visual category or a reproducible image-normalized measure. Physical height needs additional geometry if claimed. | 39 ordinal human labels, all low/mid; external 24 also all low/mid. These do not validate metric height or the existing engineered torso-clearance thresholds. No high-class reference is present in either cohort. |
| Ball trajectory | Same-ball identity through supported contact-to-arrival observations; retain gaps and coordinate system. | Qualitative direction/height labels exist for 39 and 24 events, but no independently annotated ball centre sequence or bounding boxes. A direction class is not a trajectory. |
| Ball final position | Define the terminal event, time and same-ball endpoint. Last visible detection is a separate, censored observation. | Ten core free-text endpoint descriptions; old/external direction-height labels supply coarse location context. No independent endpoint pixel coordinates or terminal timestamps. Core 03 describes crossbar impact, not the later resting position. |
| Goal/no goal | Automatic outcome and external reference must remain separate; abstentions stay in the denominator. | 39 source-attributed FIFA outcomes; ten core manual outcomes overlap them. External 24 have human and external-result records, with one disagreement quarantined. |

The 39-event reference's 156 qualitative entries have no evidence frame attached.
Four values marked unknown across two events reflect an explicit outside-goal
policy. They are completed review decisions, not automatically missing cells.
The 228 landing events are not 228 independent videos. The old 27-window replay
comparison used 156 comparable landings, a different denominator from 223 in all
review windows. Contact frames in the old reference provide context and are not
independently verified frame-exact contact annotations.

## Cohorts and provenance

**Original 39:** `penalty_annotations_39_human_v040.json` records one reviewer,
Ruizhe Xu, with assisted transcription. Full-approach visibility was mostly
AI-assessed, not separately human certified. Both old ZIP deliveries duplicate
material already integrated into `reference_source`; they add no independent
events or reviewers. Existing `penalty_kick_frames.csv` (39),
`penalty_step_manual_reviews.csv` (39), and
`penalty_goalkeeper_track_reviews.csv` (36) explicitly mark AI-assisted,
human-unverified records. They must not become human truth because of their
filenames. The separate 39-row outcomes CSV attributes results to the FIFA
technical report; that attribution was inspected locally, not freshly checked
against the online report in this audit.

**Core ten:** the new workbook has ten filled rows in `补标10段`, rows 6-15,
with 90 newly filled cells across nine fields. Its second sheet copies the old
39 labels. These are ten of the original events, not ten new penalties. The
playback copies do not double the sample size. All current video hashes and
technical metadata were checked by the preparation script. The new video's
clock differs from the old canonical clips; observed contact offsets vary from
5.9 to 6.133 seconds, so an assumed constant offset is invalid. Contact times are
coarse human viewing observations, not exact physical-contact frames. Current
contact experiments use four match folds of 4/2/2/2 events.

Core 07 says left-low saved while the old height label is mid. Core 08 says
centre-mid while the old horizontal label is right. Preserve both descriptions
and their source/time/view; do not silently overwrite either. Empty uncertainty
notes in the workbook do not establish certainty. These issues affect their
specific targets and do not invalidate every other label in the project.

**External 24:** three matches provide eight events each, with 168 human chat
observations across seven categorical/count fields. The German Cup eight and
Coppa Italia eight have an explicit behind-kicker direction convention already
applied by the annotator; do not mirror their labels again. The World Cup eight
have no declared direction reference and require convention resolution before
direction scoring. Match 09, Vlahovic, has human goal versus external no-goal;
the earlier prototype quarantines the entire event pending identity/result
review. External mapping by chronology is not independent video-identity proof.

The release contains 127 shot/related records, 125 in the event ranges and two
supplements. There are 56 camera labels, including 54 in-range and two
supplemental labels. Two labels cover only part of a shot. Replay is explicitly
true in two records and otherwise unknown; slow motion is true in four, false
in one and unknown in 122. Occlusion is true in two and otherwise unknown.
Unknown fields cannot be used as negative examples. Camera class, shot boundary,
replay identity and playback speed need separate evaluation.

**Pose pack:** eight source videos/working copies represent eight events from
seven kickers. The 619 pose rows are upstream YOLOv8 predictions over 1,110
decoded frames, not human keypoints or motion-capture measurements. They contain
12 two-dimensional shoulder-to-ankle joints, without heel/toe keypoints,
confidence, 3D or camera calibration. The selected pose sequence ends around an
upstream estimated contact and lacks the goalkeeper/follow-through evidence
needed for other outputs. All eight shots are right-footed. True per-frame PTS
must be used because upstream container FPS was unreliable. This is useful
auxiliary data and a pipeline check, not an independent pose-accuracy benchmark.

## Evaluation and promotion

Use the old 39 for grouped development comparisons and preserve the all-39
denominator alongside eligible-window metrics. Group by match; never randomly
split frames or views of the same event. The core ten use leave-one-match-out
development evaluation with the current video hashes. Neither cohort can be
renamed an untouched test after extensive prior inspection.

For external events, preserve the existing German-Cup eight training and Coppa
seven validation split, excluding match 09. Keep the eight World Cup events
unused for model training, inference and selection until a final protocol is
frozen. Their metadata has already been audited, so this is a model-held-out
cohort, not a claim that nobody has seen any label. Pose-pack comparisons must
group by source kicker/event, not frame. Its predictions can be pseudo-labels
for experiments but not the reference used to demonstrate those experiments'
accuracy.

The following targets are **our prospective engineering goals**, not teacher
thresholds or achieved results. Freeze them before the next comparison:

| Target | Engineering goal and reporting rule |
| --- | --- |
| Categorical event outputs | At least 90% correct among all eligible referenced events and at least 90% output coverage. Also show covered accuracy, per-class counts and macro F1; aim for macro F1 at least 0.85 where all claimed classes are represented. |
| Step count | MAE at most one step and at least 90% within one step on a matched count definition. Report landing precision/recall separately and partial-window coverage. |
| Contact timing, supporting module | At least 90% of all ten events within 0.2 s of the coarse references; compare MAE on common available events and coverage. This is not frame-exact contact accuracy. |
| Camera intervals | Report class-wise temporal overlap and boundary errors on genuinely labelled intervals, with a 0.5 temporal-IoU matching threshold and target macro F1 0.85. Partial labels and unknown speed cannot establish full four-class success. |
| Trajectory/top-down/endpoint geometry | Report provenance, visibility and failures now. Quantitative accuracy is not assessable without independent corresponding coordinates; choose units and tolerance before that evaluation. |
| Runtime | Record end-to-end latency including decode/model load, execution device, video duration and processed frames; summarize median/p95 and real-time ratio. The brief sets no latency ceiling. |

A candidate must improve the relevant fixed baseline without hiding lost
coverage or selecting only easier events. Record all failed runs. Repeated
selection on these small development cohorts still needs a later model-held-out
check before a generalization claim. Exact-looking confidence scores are not
calibrated probabilities unless calibration is evaluated.

## Concrete remaining evidence boundaries

The existing material supports further foot/landing/count experiments, coarse
contact timing, ordinal categories, outcome evaluation and known-view camera
checks now. No new data is a prerequisite for those engineering steps.

Independent ball-centre/endpoint observations and visibility intervals would be
needed to *measure* ball-coordinate accuracy; bounding boxes are needed if a
new ball detector is to be supervised, not to run a frozen pretrained detector.
Independent ground-plane correspondences/path observations are needed to
validate top-down geometry. An ordinal dive-height claim must use the same
definition as its reference; no current high examples support a three-class
generalization claim. The sparse explicit slow-motion positives/negatives limit
a measured four-class camera claim. These are target-specific evidence limits,
not reasons to stop all improvements or to describe the supplied annotations as
empty.

## Final frozen external check, 27 October

After the protocol and model bindings were frozen, the complete v3 pipeline was
run on all eight World Cup events. The batch completed 8/8 events with zero
failures; event receipts, analysis hashes and source-video hashes are preserved
under `outputs/full_worldcup8_2026-10-26/batch_v3/`. The primary scoring report
is `outputs/full_worldcup8_2026-10-26/evaluation_v3_full/report.json`.

The results are not a high-accuracy pass: striking foot is 3/8 strict accuracy
with 5/8 output coverage, while goal outcome and keeper ordinal height both have
0/8 output coverage under their evidence contracts. Full inference took a
median 287.15 seconds per event and p95 468.41 seconds on the shared RTX 4090.
These outputs establish the current generalization boundary. The consumed
cohort is no longer available for model selection; a later improvement must use
a fresh match and the missing independent geometry, playback-speed and
keeper-height references described above.

Before final presentation, generate outputs from video without passing human
answers into inference, regenerate EDA from the same frozen output table, and
package weights, parameters, sources and dependencies. Report missing outputs
as limitations. Accurate supervised labels, successful code execution, and a
complete course submission are three separate acceptance checks.

## Verified development progress

The contact `model_roi_v5/report.json` now reports 10/10 outputs and 10/10 within
0.2 seconds, with MAE 0.0800 s versus 0.1133 s for the same-evidence rule. This
meets the contact engineering goal on the ten-event grouped development
evaluation only; it does not establish a new-video population accuracy.

The shoe `model_v2_audit/report.json` reports 30/32 decided foot labels correct,
but only 32/39 coverage and macro F1 0.7232. The same 27-window landing comparison
improves count MAE from the pivot rule's 3.3704 to 0.8889 and temporal recall from
0.2628 to 0.8013. Across all 39 events, landing recall is 0.5964 and 26/39 are
within one count. These are improvements, not completed overall acceptance.
See `SHOE_SUPERVISED_2026-10-26.md` for the fixed models, full denominators and
reproduction. Neither result changes the independent-coordinate evidence limits
for trajectory, endpoint or top-down geometry.
