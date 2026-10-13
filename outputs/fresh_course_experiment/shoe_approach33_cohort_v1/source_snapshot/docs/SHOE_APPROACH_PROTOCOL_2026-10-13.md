# Continuous shoe evidence on the course recordings

This experiment keeps the same four course recordings and 39 penalty events.
It tests whether continuous heel and toe observations support visible landing
proposals better than ankle-signal extrema. The existing result table is the
comparison baseline, not an evaluation reference.

## Inputs and Timing

The hashed course-context clips, original qualified person tracks, source-only
stationary-ball/departure brackets and saved camera-motion estimates are the
inputs. All frames use the clip clock. Source-recording frames and seconds are
derived from the recorded slicing offset and FPS. An automatically proposed
approach onset is not a human-verified start of the full run-up.

Fresh pose inference extends from the proposed onset to shortly after ball
departure, within one selected camera shot. Landing analysis stops at the last
actual stationary-ball observation. The post-departure observations are retained
for diagnostics and cannot become approach steps.

## Frozen Comparison

The official MediaPipe Pose Landmarker Heavy model is reused with 33 image
landmarks. The [official VIDEO API](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)
requires a source-frame timestamp for each call. Normal and reflected inputs
have independent tracker instances and a fixed union ROI within each event.
Inverse-mapped mirrored anatomical channels follow the existing topology.
Original same-frame actor boxes and rival boxes gate identity. Missing original
actor observations do not acquire invented identities through interpolation.

Sequential decoding verifies source dimensions, FPS and actual frame numbers.
Decoded-pixel hashes identify exact repeated observations; repeated pictures
cannot provide additional independent support for a stance transition.

The landing method requires confident, agreeing bilateral shoe landmarks,
usable camera transforms and a measured swing-to-low-motion transition.
Preparation support at the beginning of the window is not counted as a new
landing. Missing chains, pose jumps and long gaps interrupt support. Settings
are frozen before inspecting the dense cohort outputs; they are engineering
thresholds, not learned probabilities or measurements of physical contact.

The baseline ankle extrema, new supported proposals, observation gaps, and
source-frame evidence are retained separately. A smaller count is not assumed
to be more correct. A shoe that looks motionless in a projected image is not
necessarily on the ground, so the output remains a visible stance-transition
proposal until independently reviewed.

## Evaluation

Predictions and their hashes are frozen before creating a blank reference
template. Human reference rows start as pending, with unknown striking foot
and missing landing events, not zero steps. Genuine reviewers must record
their identity, review time, observable interval, and actual landing frames.
Automatic output, known player footedness and prior group labels cannot fill
those fields. Independent reviewer disagreements remain explicit.

Event matching uses one-to-one timing matches within a declared tolerance.
Count error is reported only for reviewed, observable, matching windows; foot
agreement and decided coverage are separate. No reviewed labels means no
accuracy or error estimate. These 39 videos have already been used for
development, so this comparison is not an untouched held-out test.

The experiment cannot restore an unrecorded approach, an occluded contact,
or a missing final ball position. No additional video enters this run.
