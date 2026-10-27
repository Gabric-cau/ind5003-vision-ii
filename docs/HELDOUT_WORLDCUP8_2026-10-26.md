# Frozen World Cup Eight-Event Evaluation

The final selected heads were evaluated once on the eight supplied 2022
Argentina-France shootout clips, after the protocol, code, weights, inputs and
references were frozen. All eight videos completed inference. No model parameter,
feature choice or label mapping was changed after seeing these predictions.

## Results

| Output | Final Candidate | Frozen Control | Interpretation |
| --- | --- | --- | --- |
| Ordinal ball height | 2/8 correct, 25.0%; macro-F1 0.133 | V1: 1/8; always-low training majority: 6/8 | Fails to generalize reliably; below the simple majority baseline |
| Ball direction | Predictions saved; accuracy not computed | Not scored | All eight human direction coordinate conventions are unspecified |
| Camera class, labelled duration | 80.78/85.86 seconds correct, 94.08% | Always behind-kicker: 82.83% | Restricted to the explicitly labelled temporal region |
| Camera class, reference-interval majority | 14/15 correct, 93.33% | Always behind-kicker: 8/15 | Derived from automatic prediction windows, not inference on human crops |

Output coverage was 8/8 for ball height and 100% of the camera-labelled duration.
Coverage means a prediction was produced, not that it was correct. Height
references contain six low, one middle and one high event. Four low events were
predicted high; the middle and high events were both predicted low.

Camera scoring contains eight behind-kicker and seven goal-area intervals, and
no reference example of `other_side`. Time-weighted macro-F1 is 0.980 over the
two represented classes, but 0.654 over the fixed three-class output space.
These numbers do not establish all-three-class generalization, the teacher's
four camera/playback classes, or slow-motion classification.

## Frozen Scope

The protocol uses the existing training `LABEL_MAP` without adding synonyms from
test labels. Of 44 World Cup timeline records, 20 have human camera labels and
24 are unlabelled. Fifteen exact, mapped intervals enter scoring. Four labelled
texts are outside the frozen map and one partial label has no exact start/end;
they are listed but excluded. The precisely labelled partial interval in
`match_20.mp4` covers 13.00-19.36 seconds only.

The main camera inference scans the full unlabelled video, detects cuts and
classifies windows of at most two seconds. Human time boundaries are used only
after predictions are saved, to measure temporal overlap. Unlabelled time is
not treated as a negative example. No oracle human-cropped classifier run was
performed.

The auxiliary temporal-IoU comparison merges adjacent same-class intervals and
matches class runs at IoU >= 0.5, after restricting predictions to the labelled
temporal mask. It matches seven of eight behind-kicker runs and all four merged
goal-area runs. This masked comparison cannot establish camera-cut boundary
accuracy or false positives in unlabelled regions; boundary errors remain null.

The height head uses German-Cup eight plus eight inherited core-event human
height labels. The direction head uses only the German-Cup eight. The camera
deployment head uses the 32 admitted German/Italian human intervals. Exact
video hashes, canonical event IDs and source-recording hashes have no overlap
with the test set. The eight test source intervals do not overlap each other,
but all belong to one match and share its camera conditions. The audit confirms
the supplied metadata and bytes, not independently adjudicated human identities.

## Runtime and Artifacts

The local CPU run used two threads, Python 3.13.15, PyTorch 2.14.0, OpenCV 4.13.0
and NumPy 2.4.6. Total batch latency was 61.72 seconds including one shared model
load of 0.61 seconds, for 201.52 seconds of video. Median per-video processing
was 8.15 seconds; p95 was 9.24 seconds. Real-time ratio was 0.306. Per-video
latencies exclude the shared load and are not separate cold-start timings.

The original containers emitted a missing QuickTime chapter-track warning, but
all required video frames decoded and all eight predictions completed.

Artifacts are in `outputs/heldout_worldcup8_2026-10-26/`:

- `protocol.json` and `protocol_freeze.json`: pre-inference rules and binding hashes.
- `inputs.json`: eight label-free inference inputs and verified source hashes.
- `references.json`: unchanged human references, strict camera inclusion/exclusion.
- `raw_predictions/` and `raw_predictions.json`: all automatic outputs and timings.
- `predictions_frozen.json`: raw results hashed before scoring.
- `report.json`: full metrics, controls, confusion matrices and per-event results.

Protocol SHA256:
`1f62a022bf0f826fa122f9ffb6c7e38107ddb80dd88b4e88001f388c5b61b909`

Raw prediction SHA256:
`9b8a962f09a4acbb6ab32f58434d1d73492b9e8529139ec2df3c507abdc3e557`

Seven evaluator tests passed before the test run. They cover error denominators,
time-weighted camera overlap, partial-label masks, missing predictions, overlap
rejection, frozen semantic mapping and one-to-one masked interval matching.
Post-run verification confirmed that every frozen code hash remained unchanged
and that the raw prediction bytes still match the pre-scoring freeze.

The small external result supports the two represented camera-view categories.
It does not rescue the failed height head or resolve direction conventions.
These eight events are now evaluated test data and must not be reused as a fresh
untouched test or as a validation set for further model selection.
