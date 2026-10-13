# Reviewing the shoe landing experiment

The new algorithm proposes swing-to-stance transitions in a visible part of the original course footage. These are image-plane candidates, not verified physical contacts. A candidate count does not establish the total number of run-up steps.

The reference file starts with no labels. It must be filled by people who inspect the original clip frame by frame. Do not copy the candidate frames, old annotation tables, player footedness, or algorithm output into the reference. Do not invent a reviewer name or mark an AI inspection as human verification. The current development cohort is not a held-out test set.

## Freeze first

Freeze the unmodified prediction files before creating a reference sheet:

```sh
python scripts/evaluate_shoe_landings.py freeze --predictions-dir outputs/EXPERIMENT --output outputs/EXPERIMENT/prediction_freeze.json
python scripts/evaluate_shoe_landings.py template --freeze outputs/EXPERIMENT/prediction_freeze.json --output outputs/EXPERIMENT/human_reference_template.json
```

The first command records the SHA-256 of each `landing_result.json`. The second copies only the clip identity, input hash, FPS, and frame interval. It leaves the reviewer, labels, and observability pending. It does not copy the proposed landing count, proposed frames, or striking foot. Existing files are never overwritten.

The template and evaluation name the same freeze by its SHA-256. The evaluator refuses altered predictions, different source hashes, duplicate events, or a different frame clock. A hash establishes byte integrity; it cannot prove that a reviewer had not previously seen predictions. Record the actual review procedure separately.

## Prediction contract

Each `events/<clip_id>/landing_result.json` contains these evaluation fields; additional diagnostics are allowed:

```json
{
  "clip_id": "event_id",
  "source_sha256": "64 lowercase hexadecimal characters identifying the analyzed clip bytes",
  "fps": 25.0,
  "frame_clock": "clip_local_zero_based",
  "observable_window": {"start_frame": 0, "end_frame": 200},
  "full_approach_visible": false,
  "count_complete": false,
  "candidate_count": 2,
  "landings": [{"frame": 100, "foot": "left"}, {"frame": 125, "foot": "right"}],
  "striking_foot": "unknown"
}
```

Frame bounds are inclusive and refer to the analyzed clip, starting at zero. `source_sha256` identifies that clip, not a shared parent recording. `frame_clock` may be omitted by older prediction writers; it then means `clip_local_zero_based`. The reference must explicitly declare it.

`landings: null` with `candidate_count: null` means unavailable prediction. Its `observable_window` can also be `null` when no supported approach interval was found. `landings: []` with `candidate_count: 0` means a processed window produced no proposal. These states are different. An optional positive `frame_count` lets the validator check that every frame lies within the analyzed clip. Foot values are `left`, `right`, or `unknown`; landing events are ordered by frame and foot, with no duplicate frame/foot pair.

## Human reference contract

The top-level file has `schema_version: "shoe-landing-reference-v1"`, `prediction_freeze_sha256`, and an `events` list. Each event keeps its frozen `clip_id`, `source_sha256`, `fps`, `frame_clock`, and `observable_window`, and contains a `reviews` list. Two reviewers fill separate entries without consulting each other's labels. Events can be reviewed at different times; a partial file can contain only the events actually reviewed.

An untouched review is:

```json
{
  "reviewer_id": "",
  "reviewed_at": null,
  "annotator_type": null,
  "human_verified": false,
  "review_status": "pending",
  "striking_foot": "unknown",
  "review_window": {"start_frame": 0, "end_frame": 200},
  "full_approach_visible": null,
  "verified_landings": null,
  "notes": ""
}
```

After actual review, enter a real reviewer identifier, an ISO timestamp with timezone, `annotator_type: "human"`, and `human_verified: true`. These are human declarations, not identities authenticated by the script. Choose a review status:

| Status | Meaning | `verified_landings` |
| --- | --- | --- |
| `pending` | No review completed | `null` |
| `observable` | The chosen visible window can be checked for landing events | Checked list; `[]` only for an observed zero |
| `unobservable` | Foot contacts cannot be seen reliably in that window | `null` |
| `ambiguous` | Evidence admits multiple plausible landing sequences | `null` |

`review_window` may narrow the frozen interval, but it cannot extend it. If the frozen interval is unavailable (`null`), pending rows keep a null review window; a human can subsequently enter a checked clip-local interval for reference labels. That does not make the algorithm's unavailable count evaluable. If `frame_count` is supplied, the review interval must stay within the clip length. Record the same inclusive clip-local frame clock. `verified_landings` uses `{ "frame": INTEGER, "foot": "left|right|unknown" }`. Inspect the transition from swing to apparent ground support, rather than counting every ankle extremum. Mark the anatomical side unknown when it cannot be established. The event itself must still be visibly checked; ambiguous possible contacts should remain ambiguous.

`full_approach_visible` describes whether the entire approach, including its beginning and the final support before the kick, appears in the review window. For a completed review it must be a boolean. A cropped ending is `false`, even if every visible step is countable. `striking_foot` is an independent video observation; it can remain unknown even when landing events are observable. Conversely, the kick foot can be visible when earlier landings are unobservable.

## What evaluation reports

```sh
python scripts/evaluate_shoe_landings.py evaluate --freeze outputs/EXPERIMENT/prediction_freeze.json --references outputs/EXPERIMENT/human_reference_template.json --output outputs/EXPERIMENT/evaluation.json
```

The aggregate landing metrics require at least two distinct declared human reviewers. All completed reviews of that event must agree on the reviewed window, full-approach status, event count, and anatomical side labels. Their event times must match one to one within the reviewer tolerance (default 0.08 seconds). Disagreements remain in the per-event output and do not become consensus. One agreeing review is retained only as an individual diagnostic.

Prediction matching maximizes the number of one-to-one temporal matches within the timing tolerance (default 0.08 seconds), then minimizes total timing error. It reports event precision, event recall, timing error, and count MAE for jointly reviewed visible windows with available predictions. Unknown predictions are excluded from these conditional metrics and reported separately as coverage on jointly reviewed events; they do not become observed zero counts. Matching uses time, with known anatomical side agreement reported separately. Extra proposals cannot reuse one reference landing. Zero denominators produce `null`, never an invented perfect score.

Full run-up count MAE additionally requires both reviews and the prediction to establish a complete approach, and the prediction to set `count_complete: true`. The current proposal algorithm cannot establish these conditions by itself. Foot agreement requires at least two known striking-foot labels that agree; conflicting known reviewer labels exclude the event. Unknown predictions affect coverage and do not become left/right errors.

An untouched pending template therefore reports no accuracy: aggregate agreement values are `null`, reviewed events are zero, and only proposal coverage is available. All results retain `held_out_test: false`. Reviewing development clips can diagnose this algorithm; it does not measure performance on unseen footage.
