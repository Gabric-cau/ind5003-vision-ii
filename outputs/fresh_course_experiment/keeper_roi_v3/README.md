# Keeper ROI pilot

Three source-bound course clips were processed with new normal and rotated pose inference in a fixed goalkeeper/goal ROI: Kane, Rashford and Cuadrado. Baseline and treatment use the same contact and the same camera estimate produced for this pilot. The protocol's phrase "original contact and camera evidence unchanged" is imprecise: contact is frozen, while camera evidence was rebuilt and shared across both arms.

Only same-frame original person detections can be refined. Their original IDs and exclusions are retained, missing-frame detections cannot add new identities, and crop-edge boxes are rejected. The executed source snapshot matches the hash in `protocol.json`.

Kane gains a preparation body reference. None of the three gains an observed-height or complete-dive category. Body observations remain 80, 87 and 21 respectively; intermediate improvements are not counted as new requested-feature coverage. Numeric image diagnostics are not physical heights.

See `summary.json` and the paired `results/` records. This is a development pilot without human labels or held-out validation. The earlier local ROI variants were superseded because newly assigned IDs could evade exclusions.
