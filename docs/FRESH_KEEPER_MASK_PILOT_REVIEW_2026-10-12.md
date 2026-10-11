# Fresh Keeper Mask Pilot Review

This review concerns the actual 4090 SAM2 pilot in `outputs/fresh_course_experiment/keeper_mask_pilot_v1_n`, using fresh pose V7 and fresh ball V3 normal-model contacts. It is an AI-assisted source inspection, not human annotation or independent accuracy validation. The three events produced 211 actual masks: 70 for Eriksen, 62 for Krohn-Dehli and 79 for Brozovic. The mask runner records the exact model, source video, source-frame crop, fixed ROI, keeper role, event cache and per-mask hashes. No legacy mask was reused.

The diagnostic sheets under that directory's `source_diagnostics` were inspected at source-frame level. They use the actual GPU input pixels from the downloaded lossless ROI video, whose decoded frame hashes match the producer receipt. Local Mac decoding of the original compressed MP4 differs slightly from the GPU FFmpeg decoding, typically about 0.66 intensity levels per BGR channel and at most three levels in the inspected frames. The sheets therefore insert the hash-verified GPU ROI into the full-scene context and record the decoder difference separately. Video-byte identity checks were not relaxed.

## Eriksen

The keeper body remains inside the ROI at the frame-283 stop. The failure comes from a detached extra object: source inspection shows SAM2 has also followed the white ball towards the left ROI edge. At frame 275, the detached component has 171 pixels, 12.49% of the mask, with bbox `[613,377,18,13]`; the 1,198-pixel keeper component has bbox `[664,421,62,34]`. At frame 282, the extra component has 156 pixels, 9.68%, at bbox `[502,351,16,12]`, while the keeper component has 1,455 pixels at bbox `[632,416,56,40]`. At frame 283, one stray pixel reaches x=500, the ROI boundary, and stops the connected identity prefix. Increasing the crop alone would hide this particular edge trigger while retaining the ball leak. It would not repair the identity problem. The raw masks and the stop remain available for diagnosis.

## Krohn-Dehli

All 62 masks passed the existing connected prefix guard, and the body is visibly on the ground in the later source frames. The incomplete strict result is therefore not a mask-edge failure. The fixed pre-contact ankle reference is invalidated by forward/depth motion: the locked silhouette's lower edge reaches approximately -0.51 preparation body heights relative to the original line. Critical torso/ankle measurements also vary during the transition. A visible body silhouette alone does not establish a calibrated ground point or an anatomical torso measurement where confident joints are missing. The current strict result remains unknown.

## Brozovic

The frame-264 stop is actual keeper-body truncation at the right ROI boundary, after the initial dive and while the keeper is recovering upright. The ROI is `[160,132,678,502]`. Frame 262 has a single 1,857-pixel body component with bbox `[623,259,52,79]`; frame 264 has a single 1,888-pixel component with bbox `[628,251,50,87]`, with six mask pixels on the right edge. Source inspection does not show drift onto the net. Enlarging the fixed ROI to the right is justified for a new inference version if a complete recovery prefix is needed. It does not by itself recover the confident anatomical torso observations missing during the earlier horizontal extension.

## Full Development Replay

The CPU height replay in `outputs/fresh_course_experiment/height_selected_v1` used all 39 source-bound events and the selected normal/medium ball package. Krohn-Dehli and Brozovic masks were reused because their event-cache hashes match exactly. Eriksen's selected ball producer changed, so its normal-model mask was not mixed with the incompatible event cache. The replay contains ten measured image-plane extension maxima, six observed ordinal signal candidates and four values too close to an engineering category boundary. All 39 strict and complete pose ordinal fields remain unknown. Five of the six signal candidates also flag a failed fixed-depth assumption; these values must not be reported as six established low physical dives.

A full 39-event SAM2 run is not automatically warranted. The pilot demonstrates a useful body-identity signal but also a specific ball leak and true crop truncation. A new run should first address these mechanisms with an explicit versioned crop/identity plan, and may need actual cropped/rotated pose inference to recover anatomical torso observations. Reporting the already measured continuous image signal does not require manufacturing additional mask coverage.
