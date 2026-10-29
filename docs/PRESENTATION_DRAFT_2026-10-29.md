# IND5003 Vision II Presentation Draft

This is a 12-slide source draft for a ten-minute, normal-speed presentation. Replace the group-member placeholders and rehearse the spoken wording. The speaker should explain the actual evidence, keep unknowns visible, and use a student's natural voice.

## Slide 1 — Penalty video analysis

**On screen:** Title, four World Cup shoot-outs, 39 penalty events, group members.
**Message:** We build an evidence-first video pipeline for the nine requested outputs and four camera-view intervals.

## Slide 2 — What the assignment asks for

**On screen:** The nine outputs: run-up/top-down trajectory, kicking foot, steps, goalkeeper pre-kick movement, dive direction, dive height, ball trajectory, final ball position and goal/no-goal. Add the four camera views.
**Message:** These are separate temporal and visual questions; a single frame classifier cannot answer all of them.

## Slide 3 — Data and event definition

**On screen:** 39 events, four matches, event-level slicing, source-frame clock and view examples.
**Message:** A penalty starts at the approach and ends only when the play is observable. Different views of one penalty remain one event.

## Slide 4 — From video to evidence table

**On screen:** Source video → event slice → person/ball/pose → temporal identity → geometry and phase rules → feature table + evidence frames.
**Message:** Every output carries a source frame, time basis and provenance; unsupported outputs remain unknown.

## Slide 5 — Labels, splits and validation

**On screen:** Human event labels, core reviews, 835 goalkeeper boxes, development cohort and the eight-event frozen held-out primary check.
**Message:** The new packages mostly add labels to existing events; they do not create independent new penalties.

## Slide 6 — Exploratory coverage

**On screen:** `01_fresh_coverage.png` and a short table.
**Message:** Contact is 33/39, pre-kick movement 29/39, ball-track candidates 33/39, outcome fusion 35/39. Coverage is uneven.

## Slide 7 — RQ1: What can broadcast views show?

**On screen:** `02_fresh_steps_timing.png` plus a supported approach example.
**Message:** Temporal and pose evidence supports some contact and approach measurements, but a sparse ground projection is not a complete top-down trajectory.

## Slide 8 — RQ2: Foot, steps and augmentation

**On screen:** Foot/step result table and final augmentation comparison.
**Message:** Foot evidence is only about 13–14/39 and step validation is development-only. The ball augmentation arm fell from initializer 65/91 and 61/88 to 62/91 and 58/88, so it was not promoted.

## Slide 9 — RQ3: Ball path and outcome

**On screen:** `03_fresh_ball_references.png` and `05_fresh_scoreboard_outcomes.png`.
**Message:** Ball trajectory candidates cover 33/39 and outcome fusion covers 35/39 in development. This is not independent accuracy because the cohort was used during development.

## Slide 10 — Failure boundaries

**On screen:** One occluded keeper frame, one camera-cut/end-boundary frame, strict/provisional split.
**Message:** Strict dive height is 0/39 and strict event-final ball position is 0/39. A separate provisional view has 17/39 height candidates and 27/39 last-visible ball observations, but neither is accuracy.

## Slide 11 — Generalisation and reproducibility

**On screen:** held-out 8-event result, Dell RTX 2060 portable run, repository structure and runtime.
**Message:** The primary pipeline completed 8/8 held-out runs, but strict kicking-foot accuracy was 3/8 and goal outcome/keeper height had zero valid coverage. The portable package proves execution, not high accuracy.

## Slide 12 — Conclusion and limitations

**On screen:** Three findings, three limitations, next steps.
**Message:** The contribution is a reproducible evidence pipeline with honest abstention. The remaining work is a narrated presentation, final report review and additional independent labels/footage for height, endpoints and camera playback.

### Timing

| Slides | Time | Focus |
| --- | ---: | --- |
| 1–2 | 0:00–0:50 | Task and contribution |
| 3–5 | 0:50–2:30 | Data, definition and pipeline |
| 6–9 | 2:30–6:50 | EDA and research questions |
| 10 | 6:50–8:00 | Failure boundaries |
| 11 | 8:00–9:10 | Held-out and reproducibility |
| 12 | 9:10–10:00 | Conclusion and limitations |
