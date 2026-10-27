# Training and Offline Video Deployment

This round provides actual supervised models, an automatic single-video entry,
source-bound batch execution, and recorded development and held-out results.
It is not evidence that all required outputs are accurate, nor a completed
course submission. The brief sets no numerical accuracy threshold.

## What Was Trained

YOLO11 detection/17-point pose, MediaPipe Heavy 33-point pose and the ResNet18
image encoder remain pretrained and frozen. Newly fitted models are the temporal
contact classifier, striking-foot classifier, landing classifier, goalkeeper
direction/ordinal-height heads, ball direction/height heads and spatial-camera
classifier. The landing model is gradient boosting; the other principal heads
are regularized linear classifiers. Outcome classification was also tested as a
separate experiment and was not enabled by default.

The small supervised heads are appropriate to the available event labels; this
round did not train or fine-tune Qwen or another video-language model. Frame
counts are not independent sample counts. Videos and views of the same penalty
stay in the same match group when evaluating the trained heads.

## Measured Results

| Experiment | Result | Scope |
| --- | --- | --- |
| Contact v5 | 10/10 within 0.2 s; MAE 0.0800 s versus same-evidence rule 0.1133 s | Four match folds, ten previously inspected core events; coarse human timestamps |
| Original shoe head | 30/39 foot labels correct, 32/39 output coverage; 30/32 covered accuracy | Four match folds; three left-foot references, one initially without usable input |
| Original landing head | MAE 0.8889 versus pivot-rule 3.3704; 125/156 matched landings | Same 27 reference windows, temporal matching within 0.08 s |
| Recovered shoe evidence | All 39 events now have measured pose evidence; fixed foot head 34/39 correct | Seven old failures were automatic contact/actor/window failures, not missing videos or labels |
| Recovered landing head | MAE 0.8718, 29/39 within one; temporal F1 0.67699 | All 39 reference windows; upstream contact also excludes the outer held-out match |
| Keeper direction hybrid | 22/39 correct, 29/39 outputs | Retains supported rules and fills missing candidates; not high overall accuracy |
| Keeper ordinal height | 18/39 correct, 29/39 outputs | Low/middle categories only; no keeper-high reference class |
| Ball height, retained World Cup test | 2/8 correct; majority-low control 6/8 | One held-out match; failed transfer, not a promoted high-accuracy model |
| Spatial camera, retained World Cup test | 80.78/85.86 annotated seconds correct, 94.08%; 14/15 interval-majority labels correct | Only behind-kicker and goal-area truth is present; not the teacher's complete four-class task |

A further fixed height fit used German8 plus all 36 admissible original-event
height labels, excluding two intentional unknowns and the unresolved Koke
height conflict. All 36 original videos were checked and encoded from their
automatic shooting windows, including 28 labels unused in the core8 experiment.
Italian7 validation decreased to 2/7 (macro F1 0.148), so the model was not
promoted. Its original-video window selector differs from the external selector;
the report states this preprocessing difference. The consumed World Cup test
was not rerun. More training examples did not by themselves solve transfer.

The fixed outcome pilot also failed to establish a reliable general replacement:
German8 fitting scored 7/7 on Italian validation but predicted goal for every
core clip. Its core fallback matched a constant-goal control exactly. Adding
core labels gave 9/10 training agreement but only 3/7 Italian validation. Neither
candidate overrides the existing automatic outcome evidence.

The recoveries preserve real observations. The code matches measured actor
boxes, reruns pose inference in automatic crops and retains source frame clocks.
It does not create interpolated pose labels. All-39 recovery improves coverage
but does not solve anatomical foot classification: temporal and last-support
foot alternatives were tested and retained as non-promoted comparisons.

The eight World Cup videos were processed only after freezing the protocol and
weights. Their direction labels have no declared reference frame, so direction
predictions were stored but not scored. Camera false positives in unannotated
time and unsupported classes remain unmeasured. That test cohort is now
consumed; subsequent models need a fresh independent evaluation cohort.

The complete v3 run finished on 27 October with 8/8 successful events and no
failures. Its frozen full-pipeline score is recorded in
`docs/WORLDCUP_FULL_PIPELINE_EVALUATION_2026-10-27.md`: striking foot 3/8
strict accuracy with 5/8 coverage, goal outcome 0/8 with no available outputs,
and keeper ordinal height 0/8 with no available outputs. The run's median
end-to-end time was 287.15 seconds and p95 was 468.41 seconds on the shared
RTX 4090. These figures supersede the earlier ball/camera-only timing for the
full-pipeline claim; the earlier 2/8 ball-height and camera measurements remain
separately identified auxiliary results.

## Complete Video Execution

The input is a manually isolated penalty attempt, starting with its shooting
view. Manual event slicing is permitted by the brief. No event name, shootout
ordinal, human contact time, manual ROI or answer label enters inference.
This entry is not an automatic detector of penalties in a full match.

`scripts/analyze_penalty_video.py` executes the source-video models and geometric
modules. `scripts/run_offline_video_batch.py` adds checked resume, immutable
attempt folders and per-video failure receipts. It binds source video, model,
configuration and code hashes. Successful execution can still return unknown
or incorrect predictions; the completion flag never establishes accuracy.

The ten core clips completed all requested stages on the lab RTX 4090, using
two CPU threads and a separate MediaPipe environment. Total original inference
time was 722.883 seconds for 212 seconds of video; median 70.072 seconds and
p95 87.383 seconds per clip. Timing includes decoding and repeated model loads.
The GPU was shared, so these are observed run timings, not a dedicated-device
benchmark or a real-time claim.

`outputs/offline_deployment_2026-10-26/core10_batch_v2` preserves those raw runs.
`core10_summary_v3` updates the output presentation without rerunning neural
models. Its 1.146 seconds of summary work is separately recorded. Original
inference receipts and runtimes remain attached. `core10_evaluation_v3` and
`core10_eda_v3` contain the corresponding checked evaluation and descriptive
CSV/plots. These ten events overlap training and cannot measure generalization.

On that deployment check, foot is correct for nine of ten events with one
abstention; keeper direction and qualitative height are each correct for six
of ten; outcome has four correct candidates and six abstentions. Eight ball
height labels used in training are all matched here, but the separate 2/8 test
result is the relevant warning about transfer. Training agreement must not
replace the held-out result.

## Meaning of the Outputs

- `keeper_dive_height` presents the human-like ordinal candidate, its supported
  classes and unknown status. Strict mask/pose criteria remain diagnostics;
  physical height in metres remains null.
- `ball_final_position` separates the event-end position from the last accepted
  observation, including its frame, time, pixel coordinates and identity status.
  Eight core clips have a last observation; none establishes the final event
  position. Camera cuts and visibility loss are recorded, not hidden.
- `camera_teacher_category_intervals` and `camera_spatial_intervals` are separate.
  The 94.08% test result applies only to the latter on labelled time.
- Step counts describe the automatically observed approach window. They are not
  substituted for a complete run-up count without establishing matching scope.
- The trained contact timestamp is explicit. Classical ball/phase modules retain
  their own source-bound contact estimate; improved contact timing does not
  automatically validate every downstream module.

The generic shoe fallback now uses measured contact-stage actors when the old
contact/phase binder cannot supply a window. A real core08 run recovered right
foot and seven observed-window landing candidates using the original frozen
shoe head; 374 MediaPipe calls took 17.786 seconds including loading and recovery.
This is a known-event recovery check, not independent accuracy evidence.

The newest full core08 run also succeeded, including all upstream and downstream
stages, in 77.899 seconds. `core10_integrated_v3` combines this new source-bound
run with the unchanged nine earlier runs; it does not claim ten new neural
executions. `core10_integrated_evaluation_v3` now reports foot 10/10, with keeper
direction and height still each 6/10, and outcome still 4/10. The integrated
original-run total is 739.963 seconds, median 73.682 seconds and p95 87.383
seconds. The separate summary replay took approximately 1.16 seconds.

## Reproduction and Evidence

The portable package under `outputs/offline_deployment_2026-10-26` contains
source, inference weights, relative configuration, launchers, requirements and
checksums. It excludes videos, personal accounts and Python environments.
Inference needs no login or model download once dependencies are installed.
Separate main and shoe environments preserve their tested model-loading
versions. Linux shoe dependencies resolve cleanly; complete clean installation
of both environments and a macOS end-to-end run have not been established.

Primary experiment documents:

- `CORE_CONTACT_TRAINING_2026-10-26.md`
- `SHOE_SUPERVISED_2026-10-26.md`
- `KEEPER_SUPERVISED_2026-10-26.md`
- `EXTERNAL_BALL_IMPROVEMENT_2026-10-26.md`
- `EXTERNAL_CAMERA_TRAINING_2026-10-26.md`
- `HELDOUT_WORLDCUP8_2026-10-26.md`
- `OUTCOME_SUPERVISED_PILOT_2026-10-26.md`
- `OLD36_HEIGHT_TRAINING_2026-10-26.md`

## Remaining Evidence

Do not redo the completed 39 reviews or 156 qualitative labels. The targeted
packet in `outputs/next_data_requirements_2026-10-26` selects existing clips for
ball-centre/visibility/endpoint coordinates, ground correspondences and missing
teacher-specific camera/speed fields. It also records a few semantic conflicts
instead of silently changing labels. Keeper-high examples and new independent
test videos remain separate collection needs.

Existing pixels can still support engineering improvements. Missing independent
coordinates prevent measured geometric accuracy; missing class examples prevent
a validated full-class claim. Repeated fitting cannot manufacture those reference
facts. A successful local package is also not evidence that the report, EDA
research questions, presentation or SharePoint submission has been completed.
