# IND5003 Vision II Final Video Outline

**Target:** ten minutes, normal playback speed, student's natural narration, no AI voice and no artificial speed-up. This is a recording outline, not the final video.

| Time | Visuals | Spoken content |
| --- | --- | --- |
| 0:00–0:40 | Coverage chart and one representative penalty | State the problem and the three main findings first: several features have useful candidates, strict height and final ball position remain unknown, and the pipeline keeps abstentions. |
| 0:40–1:30 | Match list, event ID, source frame and FPS | Explain the 39 events, four matches, event boundaries and why multiple views of one penalty are not independent samples. |
| 1:30–2:30 | Pipeline diagram and review page | Show source slicing, person/ball/pose inference, temporal identity, geometry, feature extraction and evidence export. |
| 2:30–3:45 | Foot/approach example and step timing chart | Explain foot and step definitions, the difference between a candidate and a verified count, and the landing replay repair. |
| 3:45–5:00 | Keeper movement/direction example | Show how pre-kick movement and screen-direction are measured, then explain why dive height needs a stable body and ground reference. |
| 5:00–6:15 | Ball track, goal-line view and scoreboard chart | Explain ball continuity and outcome fusion. Distinguish development agreement from independent accuracy. |
| 6:15–7:00 | Four camera-view candidates and one unknown interval | Explain the camera task. Show that candidate intervals exist but playback and slow-motion classification remain incomplete. |
| 7:00–7:50 | Held-out table and runtime screenshot | Report the 8-event primary held-out check and the Dell 2060 portable run. Do not claim that successful execution means accurate prediction. |
| 7:50–8:50 | Failure example: occlusion/camera cut/source boundary | Explain why strict height and final ball position are `0/39`. Show the source evidence that causes abstention. |
| 8:50–9:30 | Training comparison slide | State that ball augmentation regressed and keeper photometric augmentation regressed on the external group, so neither replaced the selected baseline. |
| 9:30–10:00 | Final result table and limitations | Summarise the contribution, the remaining data needs and the reproducibility package. End with what the team would collect next: endpoint boxes/timestamps, full dive views and camera playback labels. |

Before recording, replace the group-member placeholders, verify every number against the committed result tables and add one sentence describing each member's actual contribution. Keep the distinction between strict outputs, provisional observations and independent tests visible on screen.
