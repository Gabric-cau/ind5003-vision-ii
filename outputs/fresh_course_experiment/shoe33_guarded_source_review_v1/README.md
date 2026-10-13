# Five shoe-region candidates

These boards put the decoded source pixels beside the normal prediction and the reflected prediction mapped back into the original image. Red is the model's left leg and shoe channel; blue is its right channel. The yellow circle marks the last detected stationary ball position. Each board also includes the full source frame so the person and scene can be checked.

| Event | Supporting frame | Frames shown | Board |
| --- | ---: | --- | --- |
| Rashford | 310 | 309-311 | [Source and predictions](col_eng_p04_rashford_eng19.png) |
| Dier | 297 | 296-298 | [Source and predictions](col_eng_p10_dier_eng4.png) |
| Pivaric | 327 | 326-328 | [Source and predictions](cro_den_p08_pivaric_cro22.png) |
| Smolov | 294 | 293-295 | [Source and predictions](esp_rus_p02_smolov_rus10.png) |
| Vida | 272 | 271-273 | [Source and predictions](rus_cro_p08_vida_cro21.png) |

The five crops contain the expected shooting player, and the displayed model channels do not show an obvious person substitution or reversal. That is a check of the predictions against visible pixels, not a set of human foot labels. In particular, Dier's ball has already moved in frame 297: the rule accepts proximity to the last stationary region, not observed shoe-ball contact in that frame. Heel and foot-index predictions are not verified shoe outlines.

[receipt.json](receipt.json) binds each board to the source video, result, pose records and protocol with SHA-256 hashes. It also records the decoded full-frame and original-crop pixel hashes, the exact crop bounds, and the rendered frame IDs. The renderer checks the decoder's frame clock and checks the saved call, detection and accepted-ROI counts. Crops are enlarged for display; missing predictions remain missing.

`human_verified`, `anatomical_side_verified`, `shoe_contact_verified` and `accuracy_validated` remain `false`. These are development candidates from the same course events, not an independent accuracy test. The renderer is [render_shoe33_source_review.py](../../../scripts/render_shoe33_source_review.py).
