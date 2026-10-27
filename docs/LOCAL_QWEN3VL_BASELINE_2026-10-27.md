# Local Qwen3-VL Semantic Baseline

This experiment uses the existing local Ollama `qwen3-vl:8b` checkpoint to
classify qualitative penalty-video outputs. It does not train a new VLM,
replace the ball-coordinate detector, or change the default pipeline.

## Model And Runtime

The installed checkpoint has 8.8 billion parameters, Q4_K_M quantization,
and model digest
`901cae73216286ea8c5aba8b46d307ff7188f737285ec500c795a12f05225d28`.
Ollama reports version `0.40.4` and vision support. The installed Modelfile
uses `qwen3-vl-thinking` for both renderer and parser. All model inference
uses `http://localhost:11434`; no weights were downloaded and no images
were sent to an external inference service.

The first two pilot requests used greedy generation with a 384-token
budget. They took 49.5 and 46.2 seconds and exhausted that budget without
producing final JSON. A second pilot increased the context to 32768 and
the budget to 2048; it also exhausted its budget after 95.1 seconds.
These failures and the unmodified responses remain in
`outputs/local_qwen3vl_baseline_2026-10-27/` and
`outputs/local_qwen3vl_baseline_v2_2026-10-27/`.

The installed Thinking variant is not a hybrid model that can become the
Instruct variant through `think:false`. This distinction is confirmed by an
[Ollama maintainer](https://github.com/ollama/ollama/issues/13353).
The third frozen protocol follows the
[Qwen Thinking generation settings](https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction):
temperature 0.6, top-p 0.95, top-k 20, repetition penalty 1.0. It retains
fixed seed 27 and uses an 8192-token output budget with a 32768-token
context. It explicitly requests thinking and accepts only the final
schema-valid `message.content`, never answers extracted from the thinking
field. The first complete response took 64.3 seconds. This runtime change
was made before reference labels were opened; image sampling, task prompt
and output classes did not change between the pilots.

## Frozen Input Contract

The eight existing World Cup source videos are bound by SHA-256 to
`outputs/full_worldcup8_2026-10-26/inputs.json`. The only temporal anchors
come from the automatic predictions in
`outputs/contact_fullshot_2026-10-27/predictions_causal_v3/`.

Each event is sampled from 0.8 seconds before to 1.5 seconds after its
automatic contact frame. The window is truncated to the same automatic
camera shot. Thirteen uniform frame positions plus the automatic contact
frame are deduplicated and sorted chronologically. Full frames are resized
to at most 960 pixels on the longest edge and encoded as quality-92 JPEGs.
The source frame index, decoded timestamp, source pixel hash, input-image
hash and contact-record hash are saved for every input. Decoded timestamps
must agree with the contact frame clock; a clock mismatch fails instead
of silently interpreting a variable-frame-rate source as constant-rate.

The model receives only the ordered images, their relative timestamps and
the fixed task prompt. It receives no event identifier, match metadata,
player name, supplied outcome, human contact time, or human-selected crop.
All eight events use the same prompt and JSON schema. The qualitative
outputs are anatomical kicking foot, goal/no-goal, ordinal keeper dive
height, and image-coordinate dive direction, each including `unknown`.
Images are a sparse image sequence, not native continuous-video tokens.

## Evaluation Contract

`scripts/evaluate_local_video_vlm.py` has separate `prepare`, `run`, and
`score` commands. The scoring command requires eight frozen prediction
hashes and verifies video, protocol and input-record bindings before
opening `references.json`. Failed requests and unknown outputs count as
misses in strict accuracy. Coverage, conditional accuracy, represented-class
macro F1 and per-class counts are retained separately. Dive direction is
not scored because the existing reference coordinate convention is
unresolved.

This is a development retest of one previously inspected match, not a new
independent final test. A pretrained model may recognize visual identities
or have encountered the broadcast before, even though the prompt forbids
identity-based answers. Its qualitative predictions do not validate exact
ball coordinates, complete trajectories, contact timing, full run-up step
counts, physical height, or all assignment outputs.

## Completed Results

All eight requests returned schema-valid final JSON. The prediction set was
frozen before the scoring command opened the reference file. The freeze SHA
is `4ae8c007db1c61e07b9a7a2f9ae2554b3911fa026f436366a7f5ff878ffc326d`;
the reference SHA is
`51bfab8b26ae921e462904c5a477a101d932a9eed158bdc4a029dec5b25504b2`.

| Target | Coverage | Strict correct | Strict accuracy | Macro F1 | Always-majority comparison |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kicking foot | 8/8 | 6/8 | 75.0% | 0.429 | right: 6/8 |
| Goal / no goal | 8/8 | 6/8 | 75.0% | 0.429 | goal: 6/8 |
| Keeper ordinal height | 8/8 | 6/8 | 75.0% | 0.286 | low: 7/8 |

The majority comparison is a descriptive check calculated after scoring,
not a newly selected deployment rule. Conditional accuracy is also 75.0%
for all three fields because there were no abstentions. The three scores
are not combined into an overall project accuracy.

The model predicted **right foot for every event**. It missed both left-foot
references, `wc_18` and `wc_22`; left-foot recall is 0/2. It also predicted
**goal for every event**, missing both no-goal references, `wc_19` and
`wc_21`; no-goal recall is 0/2. It predicted low keeper height seven times
and high once, missing the middle-height reference on `wc_17` and assigning
high to the low-height reference on `wc_22`. There are no high-height
references in these eight clips, so high recall cannot be estimated. The
macro F1 includes the unsupported high prediction as a false-positive class.

The generated explanations assert ball entry into the net even on the two
no-goal references. Those explanations are model output, not validated visual
evidence. The raw responses remain available for inspection.

Inference took **834.76 seconds total**, with **104.64 seconds median**
and a **64.30-168.26 second** per-event range. These are local VLM request
times. They exclude the preceding automatic contact extraction and do not
measure latency of a complete replacement for all project outputs. The
failed runtime pilots are additional development cost and are not included
in that successful-run total.

**Decision: do not promote this baseline.** It produces more answers than
the conservative earlier pipeline, but its foot and outcome scores merely
match the majority class, and its keeper-height score is below that simple
comparison. This run therefore does not demonstrate useful discrimination
of the difficult classes, calibrated abstention, or 90% performance. It
does not establish that all possible VLM input or training strategies fail;
it establishes the result of this fixed source-only protocol using the
existing local checkpoint. No additional data were needed to finish this
comparison, and no repeated annotation request follows from it.

## Artifacts And Verification

`outputs/local_qwen3vl_baseline_v3_2026-10-27/` contains the frozen protocol,
Ollama model metadata, source-bound input records, 111 sampled JPEGs, saved
requests, unmodified raw responses, eight structured predictions,
`predictions_frozen.json`, and `evaluation.json` with per-class and per-event
results. The prototype pilots remain in their separate directories.

Implementation: `scripts/evaluate_local_video_vlm.py`. Five focused tests
verify shot-bound sampling, invalid contact rejection, local-only network
access, schema enforcement and unknown-as-miss scoring. An independent
source review found no reference-label path into prediction; its protocol
and input-hash consistency suggestions were included before the completed
run.
