# World Cup Full-Pipeline Evaluation, 27 October 2026

The frozen v3 deployment was run on eight penalty videos from one 2022 World
Cup match. Inference used no reference labels, human contact times, manual ROI
selection or answer values. The lab RTX 4090 ran the complete source-to-report
pipeline with CUDA and two CPU threads. The batch completed **8/8 events with
0 failures**.

The evaluation was performed only after checking the batch manifest, every
event receipt, every analysis SHA-256, and every copied source video SHA-256.
The reference file was opened after those checks. References and raw outputs
are preserved in:

- `outputs/full_worldcup8_2026-10-26/batch_v3/`
- `outputs/full_worldcup8_2026-10-26/references.json`
- `outputs/full_worldcup8_2026-10-26/evaluation_v3_full/report.json`

## Primary Metrics

| Target | Reference events | Output coverage | Strict correct / 8 | Strict accuracy | Accuracy when output exists | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Striking foot | 8 | 5/8 (62.5%) | 3/8 | 37.5% | 3/5 (60.0%) | 0.300 |
| Goal / no goal | 8 | 0/8 | 0/8 | 0% | not defined | 0.000 |
| Keeper ordinal height | 8 | 0/8 | 0/8 | 0% | not defined | 0.000 |

Unknown is a miss in strict accuracy. Conditional accuracy is shown only for
events where the model actually produced one of the reference classes. The
three targets are not averaged into one number because they have different
evidence contracts.

## Runtime

Per-video end-to-end inference includes decoding, model loading and all source
stages. The observed times were 134.05, 248.43, 64.29, 270.24, 309.34,
490.52, 427.36 and 304.06 seconds. The median was **287.15 s**, p95
**468.41 s**, and the median video-time ratio was **10.12x**. The GPU was
shared, so this is a recorded lab run rather than a dedicated-device or
real-time benchmark.

## Interpretation

The full execution path is operational, but it does not meet a high-accuracy
claim on this external check. The foot head abstains on three events and is
wrong on two of its five decisions. The default goal rule deliberately abstains
when it cannot establish a stable scoreboard transition or verified goal-plane
crossing; the eight short clips do not provide that evidence. The keeper head
also abstains because no qualified, stabilized keeper trajectory is available
under its evidence contract.

The existing goal classifier and keeper classifier are retained as development
experiments, not promoted from this result. Promoting either based on these
eight outputs would be test-set tuning. The current data also cannot establish
the teacher's full requirements for a continuous top-down trajectory, terminal
ball coordinates, four semantic camera classes with playback speed, or a
keeper-high class: those independent references are absent or unsupported.

The result is therefore a verified complete pipeline and a verified failure
boundary, not evidence that every requested output is accurate. A future model
selection round needs a fresh match after collecting the missing geometry,
camera-speed and keeper-height references.
