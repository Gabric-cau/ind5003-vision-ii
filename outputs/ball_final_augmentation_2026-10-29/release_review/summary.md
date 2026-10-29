# Final Ball Augmentation Review

Selected development candidate: `initializer`.

Top-ranked ball box IoU >=0.5 on selected positive frames; missing prediction is a miss.

Exposed development retests only; no independent test or whole-project accuracy is established.

| Group | Model | Mode | Dedup hits/N | Recall | Candidate oracle/N | Event macro recall | Run seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| core_development | initializer | full | 53/91 | 58.24% | 59/91 | 58.57% | 5.19 |
| core_development | initializer | tiles | 65/91 | 71.43% | 69/91 | 71.47% | 19.81 |
| core_development | control_full | full | 50/91 | 54.95% | 59/91 | 55.23% | 5.14 |
| core_development | control_full | tiles | 62/91 | 68.13% | 69/91 | 68.14% | 19.06 |
| core_development | augmented_mixed | full | 55/91 | 60.44% | 55/91 | 60.93% | 5.15 |
| core_development | augmented_mixed | tiles | 62/91 | 68.13% | 66/91 | 69.19% | 18.85 |
| arsenal_development | initializer | full | 40/88 | 45.45% | 44/88 | 52.34% | 6.42 |
| arsenal_development | initializer | tiles | 61/88 | 69.32% | 65/88 | 74.97% | 44.31 |
| arsenal_development | control_full | full | 38/88 | 43.18% | 41/88 | 47.10% | 4.86 |
| arsenal_development | control_full | tiles | 58/88 | 65.91% | 62/88 | 70.92% | 43.64 |
| arsenal_development | augmented_mixed | full | 37/88 | 42.05% | 37/88 | 48.12% | 4.85 |
| arsenal_development | augmented_mixed | tiles | 58/88 | 65.91% | 60/88 | 70.21% | 43.45 |

## Selection

Tiles: historical dual gate; no group below current initializer; strictly higher equal-group mean; new-arm ties prefer control; otherwise keep initializer.

- control_full: Does not reach both historical gates: core 65/91 and ARS 62/88; Regresses on core_development versus this run's initializer; Regresses on arsenal_development versus this run's initializer; Equal-group mean does not strictly improve over this run's initializer.
- augmented_mixed: Does not reach both historical gates: core 65/91 and ARS 62/88; Regresses on core_development versus this run's initializer; Regresses on arsenal_development versus this run's initializer; Equal-group mean does not strictly improve over this run's initializer.

Both arms completed 10 epochs and 388 actual optimizer updates using identical source sequences.

The mixed arm tests a combined crop, contrast, brightness and rotation intervention. It does not create new independent events or true 3D viewpoints.

Any-candidate recall is a reference-selected diagnostic upper bound, not deployed accuracy. Group inference seconds include model loading; not full-video or full-pipeline latency.

Selection does not change deployment. A source-bound full-video smoke comparison remains required.
