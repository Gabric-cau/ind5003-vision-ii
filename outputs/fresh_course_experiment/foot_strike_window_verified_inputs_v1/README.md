# Strike-window development replay

The 39 existing paired pose caches were replayed with a vote ending at the first observed ball departure. The previous rule included a further 0.20 seconds, during which a planted leg can recoil and dominate the motion vote. The experiment retains confidence, mirror, anatomy and ball-identity guards; it also requires the motion channel to approach the stationary-ball region. It does not select the nearest ankle.

This replay returns three right-channel candidates:

| Event | Previous V2 | Strike-window diagnostic | Source-frame evidence |
| --- | --- | --- | --- |
| Trippier, Colombia–England | unknown | right | [normal and inverse-mirror frames](evidence/col_eng_p08_trippier_eng12.jpg) |
| Dier, Colombia–England | unknown | right | [normal and inverse-mirror frames](evidence/col_eng_p10_dier_eng4.jpg) |
| Smolov, Spain–Russia | unknown | right | [normal and inverse-mirror frames](evidence/esp_rus_p02_smolov_rus10.jpg) |

The evidence sheets show a swing on the blue COCO right channel while the red COCO left channel stays around the planted-foot position through the relevant part of the bracket. These are enlarged source pixels with pose overlays, not enhanced detail or independent anatomical annotations. They make the new hypotheses reviewable; they do not establish verified shoe-ball contact.

The change does not improve overall coverage by itself: the narrow rule abstains on the eight previous V2 candidates. The two methods must not be combined into an accuracy claim. Kane and Rashford have right-channel motion in the normal and paired-average views, but their original mirrored view fails the agreement rule. Keeping them unknown is a useful diagnosis of pose instability, rather than a reason to lower the threshold after seeing the outcome.

The other unknowns comprise five anatomical-channel ambiguities, 22 unsupported paired-motion cases, six unavailable upstream neural events, and three ambiguous ball identities. There is no new neural inference, no reviewer label read, and no held-out accuracy evaluation. The main feature bundle is unchanged.

The runner checks actual source-video bytes, producer pose/contact/manifest hashes, qualified upstream scene and model task, and exact replay of the original neural evidence before applying the new rule. `protocol.json` records the method and source snapshots. `summary.json` records all 39 outcomes. `evidence/manifest.json` records the decoded source frame IDs, source hashes and sheet hashes.

```bash
python scripts/replay_fresh_foot_strike_window.py \
  --clips-dir data/course_context_clips_2026-10-12 \
  --pose-dir outputs/fresh_course_experiment/pose_v7 \
  --ball-dir outputs/fresh_course_experiment/ball_selected_v1 \
  --foot-dir outputs/fresh_course_experiment/foot_v1_selected \
  --output-dir outputs/my_strike_window_replay

python scripts/render_strike_window_evidence.py \
  --experiment-dir outputs/my_strike_window_replay \
  --clips-dir data/course_context_clips_2026-10-12 \
  --foot-dir outputs/fresh_course_experiment/foot_v1_selected
```
