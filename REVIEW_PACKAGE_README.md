# IND5003 Vision II saved-result review package

Prepared 22 October 2026. This is a local review package, not an official submission or a complete raw-video reproduction. The report is 12 pages; the current Canvas guidance is an upper limit of 20-25 pages, excluding cover, references and appendices.

## Open first

- `outputs/fresh_course_experiment/submission_review_2026-10-22/IND5003_Vision_II_Review_2026-10-22.pdf`
- `docs/HUMAN_REFERENCE_INTEGRATION_2026-10-22.md`
- `outputs/fresh_course_experiment/human_evaluation_2026-10-22/per_event_comparison.csv`
- `outputs/fresh_course_experiment/review_v3_shoe_fusion_ui3/index.html` for the frozen automatic image review; its historical human-review flags predate the separate human-reference integration.

The package includes current source, configuration, tests, dependency lists, the frozen 39-event feature bundle, repaired landing predictions, foot supplement, human-reference source bytes and new comparison results, report PDF/Word, existing silent review video, and the two locally available raw context clips. It does not contain all four original recordings, every intermediate neural artifact or model weights. The model acquisition and extraction requirements remain documented in the source repository; only the saved-result routes below are verified by this package.

## Reproduce saved-result analysis

From the extracted project directory, create and activate a Python environment, then install `requirements-experiments.txt` and `requirements-report.txt`. The source checkout was verified with the existing Python 3.13 environment; the archive verification does not claim a fresh dependency installation or model inference.

```sh
python -m src.fresh_eda --bundle-dir outputs/fresh_course_experiment/final_bundle_v3_shoe_fusion --output-dir outputs/reproduced_eda
python -m scripts.integrate_human_reference_2026_10_22 --source outputs/fresh_course_experiment/human_evaluation_2026-10-22/reference_source --output outputs/reproduced_human_comparison
python -m scripts.run_pivot_support_development_2026_10_22
python -m scripts.audit_feature_gaps_2026_10_22
python -m pytest -q tests/test_single_human_reference_integration.py tests/test_shoe_landing_evaluation.py tests/test_fresh_eda.py tests/test_fresh_results.py tests/test_replay_shoe_landings.py tests/test_shoe_landing_events.py tests/test_shoe_pivot_support.py
```

The second command preserves the imported human annotations and separately computes single-reviewer diagnostics. It does not create a second reviewer, change the frozen predictions or claim held-out accuracy. The original evaluator's consensus aggregates remain null.

## Remaining project work

The reference contains 39 reviewed events, 228 landing observations including five outside the frozen windows, and 156 qualitative shot/save fields from one declared reviewer. The old pending-reference file is retained as historical evidence and is no longer the current annotation status. On 27 comparable windows, the repaired rule matches only 13 of 156 human-listed landings, so the implementation repair does not complete the measurement task.

The new fixed heel/toe support supplement matches 41 of 156 listed landings, with 24 unmatched predictions and count MAE 3.370 instead of 4.963. It uses saved poses and unchanged thresholds; all original predictions remain preserved. This remains a single-reviewer development comparison. `docs/DEVELOPMENT_COMPLETION_2026-10-22.md` documents exact remaining inputs for continuous ground trajectories, strict height, endpoints and playback classification.

Complete the feature gaps and their failure analysis, obtain the full permitted raw footage for raw-to-feature reproduction, record a student's natural narration, confirm real members and contributions, and review the final document. The existing 4:30 video is silent review material; it is not the final presentation. No files have been uploaded by this packaging step.

`REVIEW_PACKAGE_MANIFEST.json` records the byte hashes of packaged files. The adjacent `package_verification.json` records archive extraction, saved-result reruns and focused tests. The ZIP is not evidence of Canvas or SharePoint submission.
