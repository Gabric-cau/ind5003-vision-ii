# Ambiguous Goal Context Hotfix

## Reported Problem

A teammate reported that two goal detections caused a video to be skipped.
The exact failing video and logs have not yet been supplied, so this report
does not claim to reproduce that particular run.

The code contains a matching failure path. `detect_goal_frame` returns
`selected=None` when two distinct proposals have similar scores. The pose
shot selector previously required a unique supported goal in at least one of
three sampled frames. If all three were ambiguous, it skipped that entire
shot before running pose inference. If every shot was rejected, it returned
an empty failed pose result. The main analyzer already continues other stages;
it does not immediately terminate because a detector found two proposals.

## Change

Scene eligibility and unique goal geometry now have separate fields:

- `supported` retains its old meaning: unique supported goal and field evidence.
- `analysis_allowed` additionally permits an ambiguous-goal scene when at least
  one sampled frame has grass coverage of at least 0.18 and at least two
  distinct detector proposals with scores of at least 0.72.
- `ambiguous_goal_frames` retains the sampled frame IDs. The original proposals,
  `selected=None`, and detector abstention are preserved.

Such a shot may now run pose and contact-candidate search. It still needs a
qualified on-field actor and the existing duration/close-up checks. The pose
result records `ambiguous_goal_geometry_analysis_only`, requires review, and
cannot be automatically accepted. Missing field context or weak/no goal
proposals still cause rejection.

Goal-plane calibration, goalward departure, keeper role selection and ball
association retain their unique-geometry checks. If no later usable geometry
exists, these outputs remain unknown. The change neither selects the largest
goal arbitrarily nor treats failed detection as successful measurement.

## Portable v5 Update

Use `apply_goal_context_hotfix.py` from the small hotfix ZIP. With the hotfix
directory as the working directory on Windows:

```powershell
py -3.11 apply_goal_context_hotfix.py --bundle "C:\path\portable_bundle_v5"
py -3.11 "C:\path\portable_bundle_v5\verify_bundle.py"
```

Download: [v5 goal-context hotfix](https://github.com/Gabric-cau/ind5003-vision-ii/releases/tag/v2026.10.30-goal-context-hotfix).

Then run the same video command in `portable_bundle_v5`, using a new output
directory. No model download or retraining is required.

The updater checks the exact original v5 files and patch payload before any
changes. It preserves the original files and manifests. The contact model
binds the whole `step_pipeline.py` file, although its extractor uses unaffected
functions. The updater verifies that all other top-level Python syntax is
unchanged, records the original/new binding, and updates the checksums. It
does not disable model or source checks or modify weights. A checkout with
different source files is rejected instead of overwritten.

## Validation and Limits

Regression cases cover ambiguous goals on a field, rejected non-field and
weak-candidate scenes, unchanged unique-goal handling, preserved unknown
goalward evidence, and review flags. A synthetic video containing two drawn
goals exercises the actual video decoder and classical goal detector; only
the subsequent pose stage is stubbed in that test.

The focused suite passed 165 tests and 12 subtests. The updater was applied to
an isolated copy of the released v5 bundle: its own verifier passed all 366
listed files, and all model/source bindings passed the packaging verifier.
The original v5 archive and weights remain unchanged. This is not a fresh
installation test or a complete neural inference rerun on the reported clip.

Existing accuracy figures and frozen results have not been recomputed. This
is a processing-coverage fix, not demonstrated recognition-accuracy improvement.
The teammate's original failing video and logs are still needed to verify the
specific report and measure its downstream benefit.
