# Final Keeper Training Review

Selected standalone development candidate: `control`.

Exposed development positive frames; no independent test or whole-project accuracy.

Highest-confidence active-keeper box IoU >= 0.5; missing prediction is a miss.

| Development group | Model | Top1 hits/frames | Clip-macro top1 | Any-candidate hits/frames |
| --- | --- | ---: | ---: | ---: |
| development_external_match | pretrained_person_candidate_control | - | - | 229/294 |
| development_subject_02 | pretrained_person_candidate_control | - | - | 98/98 |
| development_external_match | control | 60/294 | 20.47% | 63/294 |
| development_subject_02 | control | 92/98 | 93.02% | 92/98 |
| development_external_match | photometric | 44/294 | 15.09% | 77/294 |
| development_subject_02 | photometric | 92/98 | 93.02% | 92/98 |

Photometric only if both development clip-macro top1 recalls do not decrease and their equal-weight mean strictly increases; otherwise control.

Both arms completed 12 epochs and 838 verified optimizer updates with identical initialization and sampling.

Training-reference diagnostics and the aggregate containing them are excluded from selection.

Any-candidate overlap only, not goalkeeper-role top1 accuracy. Inference over all 455 frozen evaluation frames, excluding model load; not full-video latency.

This selects an experimental detector only. No default pipeline change, temporal identity, pose, direction, height or whole-project improvement is established.
