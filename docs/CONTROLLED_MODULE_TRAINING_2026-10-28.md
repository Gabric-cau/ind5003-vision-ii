# Controlled Ball And Keeper Development Experiments

This round follows the completed MPS ball experiment. The user authorized
using the new keeper references and investigating ball retention regressions.
Existing deployed defaults and historical results are preserved.

## Ball Protocol

Three continuation arms share 10 fixed epochs, 442 samples per epoch, 1920
pixels, MPS batch 1/workers 0, AdamW, nominal batch 64, freeze 10, seed 27,
learning rate and bias warmup rate both 0.00005. Each starts a new optimizer;
this is not checkpoint resumption. The expected optimizer budget is 114 steps.

| Arm | Initial weights | Sampling per epoch |
| --- | --- | --- |
| A | Last checkpoint of the previous 30-epoch MPS round | All original 442 frames once |
| B | Same checkpoint as A | 221 old-pool frames and 221 new Leverkusen frames |
| C | Earlier four-event ball refit | Exactly the same schedule as B |

B versus A isolates the sampling change. C versus B isolates initialization.
The balance is between the old source pool and the new match, not individual
matches/events. Fixed shuffled decks rotate through every image over the round;
the real batch filenames and per-epoch counts are checked against the saved
schedule. No Ultralytics installation files are changed.

All three last checkpoints must be frozen before any new comparison is scored.
The Arsenal and core references have already been inspected and are explicitly
development retests. They remain excluded from gradient training, but cannot
support a fresh independent test or a project-wide 90% claim. No extra epochs
are triggered by their scores.

Candidate gate: tiled core hits at least the previous refit's 65/91 and Arsenal
hits at least the latest round's 62/88. If multiple arms qualify, use the mean
of the two group recalls. Passing this module gate does not automatically
replace the video pipeline. Full-frame comparisons are also reported.

## Keeper Data

`outputs/keeper_reference_import_2026-10-28/manifest.json` binds the ZIP,
16 source videos, original frame numbers/PTS, dimensions and image hashes.
It retains 2,966 box records, 1,031 joint records, 1,020 empty references and
four box/point consistency prompts. The 752 bulk reference confirmations
retain their provenance and are not called independent blinded labels.

The exports contain 149 original images and unmarked, fixed 25% keeper crops.
Six short clips supply 63 training frame records/451 visible points. Two frames
contain only occluded or left/right-ambiguous joints without coordinates, so
the supervised loader correctly contains 61 images and all 451 visible points.
The same-shooter 02-05/02-06 group supplies 44 development images/268 points.
The eight Argentina-France 2022 clips stay together as an external-domain
development group with 42 images/294 points. Short-clip venue/keeper overlap
may remain, and the external group's point labels are assisted confirmations.

The importer preserves 78 stale derived review flags from the older CSV;
the updated JSON and actual point records determine supervision. Four points
slightly outside the manual boxes remain visible within the fixed crops and
are not silently corrected or discarded. Unknown/ambiguous coordinates are
masked; no unlabeled people become detector negatives.

The controlled pilot updates only the pose coordinate branch, with all
BatchNorm statistics frozen, and reports both manual-crop diagnostics and
automatically proposed crops. The first attempt finished 10 epochs and 47
nonzero-gradient optimizer steps but its final export guard incorrectly
required a change in the first pose scale branch alone. Eight parameter tensors
in another scale branch had changed. The failed attempt is preserved, and the
corrected guard checks every trainable parameter before a fresh rerun.
Neither manual-crop agreement nor person-pose association establishes complete
keeper identity, movement timing, or dive-height accuracy.

The pretrained full-frame model fails to propose the small goalkeeper in both
development groups. Automatic crops seeded by those proposals therefore also
fail. A separate inference experiment uses full-image grid tiles, generated
solely from image dimensions, to test this proposal bottleneck. It freezes the
same pretrained proposals for the original and fine-tuned pose models. Human
boxes are used only for scoring association; they do not select inference
tiles or candidates. The grid and confidence settings are fixed before this
comparison is run.

## Status

The corrected keeper run completed on CPU while MPS was reserved for the ball
arms. All 10 epochs and 47 optimizer steps completed. All 451 visible training
coordinates reached the 61-image loader. The largest observed parameter change
was 0.0005227923; frozen parameters and BatchNorm statistics were unchanged.
The raw final FP32 checkpoint is
`outputs/keeper_pose_reference_round1_cpu_2026-10-28/last_fp32.pt`, SHA256
`a959c06e5ef27bd0d2ddbb2808abf13ca89084e64edfad928417a48674e2610b`.

| Keeper manual-ROI diagnostic | Pretrained PCK10 | Fine-tuned PCK10 | Corrected / regressed points |
| --- | ---: | ---: | ---: |
| Six training clips | 352/451 (78.05%) | 367/451 (81.37%) | 18 / 3 |
| Subject 02 development | 154/268 (57.46%) | 154/268 (57.46%) | 8 / 8 |
| External-match development | 243/294 (82.65%) | 257/294 (87.41%) | 14 / 0 |

PCK10 counts a visible reference joint as correct when the prediction is at
most 10 original-image pixels away; missing predictions count as failures.
All 294 external-match references are bulk reference confirmations. The 116
originally corrected subject-02 points remain 35/116 before and after training.
This difference in reference strata is retained, not averaged away into a
claim of general accuracy. The fixed full-frame seed path still has no keeper
proposal in either development group. Manual-ROI results cannot be reported
as autonomous video results.

The three controlled ball arms completed on MPS. None met the frozen promotion
gate (tiles core >=65/91 and Arsenal >=62/88): A natural scored 63/91 and
60/88, B balanced scored 65/91 and 61/88, and C previous-init balanced scored
60/91 and 48/88. The balanced arm improved the core development retest by two
frames over A, but missed the external threshold by one; it remains a
development candidate only. No default ball checkpoint was replaced.

The ball review is in
`outputs/ball_controlled_round2_mps_2026-10-28/review/summary.md`, with raw
receipts, event-paired changes, candidate-oracle diagnostics and contact sheets.
All five-model comparisons are exposed development retests, not an independent
test or a whole-project accuracy estimate.

The label-free keeper tile/rotation comparison also completed. Adding rotated
grid proposals raised subject-02 candidate recall at IoU >=0.5 from 43/44 to
44/44 frames; external-match recall stayed 29/42. The fine-tuned pose head's
subject-02 PCK10 was 153/268 with the original tile/ROI control, 186/268 with
the unrotated seeds and three-angle ROI inference, and 181/268 with rotated
seeds and three-angle ROI inference. External PCK10 was 204/294 in all three
conditions. These are automatic proposals associated with human boxes for
scoring, not manual-ROI results or keeper-role accuracy. ROI rotation and
proposal rotation are separate interventions and cannot be summarized as a
single failed rotation experiment. The comparison remains preliminary and
does not change the default pipeline. Its single-run CPU timings include
proposal generation and ROI inference separately in the receipt below.

The full rotation receipt is
`outputs/keeper_rotated_proposals_cpu_2026-10-29/report.json`.

Correction on 2026-10-29: the 43 KB training preparation ZIP was present in
WeChat's protected folder and has now been copied through Finder and audited.
A failed shell search was not evidence that the user had not downloaded it.
See `TRAINING_PREPARATION_AUDIT_2026-10-29.md` for its event-label coverage.

The earlier keeper ZIP already contains the complete reviewed boxes: 835 in
six training clips, 275 in subject-02 development, and 1,836 in external-match
development, with 20 unlocalizable records excluded. All 16 original videos
exist and their hashes match the source metadata. The 149 exported images
are only the sparse pose-reference subset. A specialized active-keeper
detector therefore needs a separate full-box exporter and experiment, not
another delivery of the same 835 boxes. Its class must mean the defending
goalkeeper in the current attempt; the sparse pose export must not be reused
as exhaustive generic-person labels. No such detector was trained in this
audit, and the existing development groups are not fresh independent tests.
