# Source-video experiment protocol

This protocol describes a real rerun of the four course recordings. It does not
replace the teaching brief or assert a guaranteed grade. The experiment uses 39
penalty events, grouped as 10 Colombia-England, 10 Croatia-Denmark, 9 Spain-Russia
and 10 Russia-Croatia. All shots and replays remain associated with their event.

## Data and clocks

The four original MP4 files were decoded and hashed before clipping. The slicing
manifest contains their SHA256 values and the exact original-frame intervals.
Context clips preserve 1280 x 720 pixels at 30 FPS. A local clip frame maps to
the original recording by adding its recorded start frame. Added context is an
observation interval, not a manually verified walk-to-spot or play-end boundary.

Every prediction records source, pose-producer, model, settings and code hashes.
The raw detections retain original source-pixel coordinates and detection IDs.
Stabilized coordinates and pitch-plane coordinates are separate fields. Numeric
missingness remains null. An occluded ball cannot become a stationary ball at
its last observed position.

## Frozen comparisons

YOLO11m-pose at 1280 pixels selects a field-supported kicker within a camera shot
overlapping the manually sliced action. Goal-line occupants are excluded from
kicker selection, including later fragmented goalkeeper tracks. The pose-motion
peak is retained as a baseline and cannot establish ball contact.

YOLO11n and YOLO11m object detectors are run on the same complete 39-event cohort,
same pose input, crops, image size, batch size, detection thresholds and temporal
rules. The source-only comparison was specified before inspecting medium-model
outcomes. A stationary ball identity must precede causal goalward flight. The
contact bracket is the last stationary observation to the first flight
observation, rather than the nearest ankle or an extrapolated ball position.

Contact selection uses source evidence alone. When both models support contact,
their actor IDs and timing brackets must agree and the stationary identities
must occupy the same spot in at least three shared source frames. Compatible
alternatives are ranked by contact score, bracket width and actual departure
confidence, with a fixed model tie-break. If only one producer passes the checks,
its original artifacts are selected. Conflicts abstain. The selector copies
existing neural outputs; its own execution performs no new neural inference.

The original nano and medium outputs remain immutable. Subsequent ball-rule
repairs use a new CPU replay version with producer receipts and cache hashes.
Reference outcomes cannot determine the selected actor, timing, model or path.

## Feature definitions

Striking foot refers to the anatomical leg used in this attempt. High-resolution
normal and mirrored pose estimates are compared over approach, contact and
follow-through. Channel ambiguity, contradictory evidence and insufficient
source observations remain visible. Permanent footedness, camera-side shortcuts
and player identity do not supply the answer.

Run-up steps are visible landing proposals during a contact-bound action phase.
Preparation adjustments and striking swings are distinguished from the approach.
Counts retain parameter and contact-bracket sensitivity and recording gaps.
An incomplete approach cannot establish the total steps of the complete action.

Ground plots use same-frame pitch homographies and visible sole candidates.
Held-out penalty-area sides and goal-bottom corners can reject a calibration;
they do not participate in fitting or selecting the best-fitting hypothesis.
Sparse ground-contact projections are distinct from a continuous measured path.
Hips and airborne ball observations are not projected as points on the grass.

Goalkeeper movement and direction use a goal-associated identity and measured
background-motion compensation. The direction convention is image-left/right.
The experimental height variable is explicitly an ordinal image-plane measure
of the first observable dive, with a stated body reference and phase. The course
does not prescribe physical height in metres; this experiment does not estimate
it. SAM2 masks are automatically prompted from a pre-contact goalkeeper role,
and identity checks stop the usable mask prefix after drift or truncation.

Ball traces are observation sequences with source IDs, gaps and branch evidence.
An endpoint record reports the last visible observation and the reason further
measurement ends, such as occlusion or a camera cut. A stationary-end proposal
needs sustained observed support. The event-final position requires observable
play end and a continuous ball identity, so it remains distinct from last visible
position. Broadcast scoreboard evidence is an independent source-image feature,
not a substitution of external match labels for ball tracking.

Camera boundaries, spatial views and playback status are separate results.
Relative slow-motion candidates require corresponding action alignment across
views; spatial close-ups alone cannot establish playback speed. Non-target views
and genuinely uncertain views are also distinguished.

## Evaluation and execution

FIFA's original technical-report tables independently support the 39 named-event
outcome references. They are joined after prediction. The audit retains table
pages, event ordinals and any naming ambiguity. Agreement against those labels
is development agreement: this cohort has already been inspected, and source
event identity has not received a separate human review in this experiment.

The actual extraction stack is documented by runtime snapshots, pinned official
model bytes, per-job source snapshots where captured, and per-output hashes.
The 4090 is used for inference, not training. CPU replay, package creation,
regression tests and EDA do not count as new model inference. Unit-test success
does not establish feature accuracy.

The report, independent saved-result analysis, source clips and raw experimental
evidence must be deliverable as ordinary files. The teaching team must not need
a student's lab login or private GitHub account to reproduce them. Student
authorship review, real human narration and coursework submission remain
separate from preparation of these artifacts.
