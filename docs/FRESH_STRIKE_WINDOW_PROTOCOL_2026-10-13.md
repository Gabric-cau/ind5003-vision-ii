# Strike-window experiment

This is a development experiment on the same 39 course events, using existing paired cropped pose observations. No new inference, reviewer labels or player-name priors enter predictions. Previous development outputs have been inspected, including Kane's support-leg recoil after departure. It is not a held-out test.

The experiment ends its motion vote at the first observed ball departure, rather than including a further 0.20 seconds. Its start is the last stationary-ball frame minus 0.13 source seconds. The source departure bracket must be no longer than 0.15 seconds, end at the first actual departure detection, and include two distinct actual departure detections. A fixed preparation-box scale, full bilateral leg confidence, source-coordinate normal/mirror agreement and local anatomical channel ambiguity guards are retained.

Normal, inverse-mapped mirrored and paired-average observations must all independently satisfy the existing motion rule: four connected pairs, median speed ratio at least 1.25, at least 65% pair support, and motion at least 0.3 body heights per source second. The same anatomical channel must win all three views.

That channel must additionally approach within 0.35 body heights of the final actual stationary-ball point and reduce its distance by at least 0.03 body heights relative to its preparation observations. This is only a relevance gate. The nearest ankle cannot select the label, because the supporting foot can be nearer the ball. Departing ball coordinates and post-departure poses cannot affect the prediction. Ball-identity ambiguities retain unknown.

These thresholds are engineering choices frozen before the cohort replay. A larger number of candidates is not an accuracy improvement. Newly covered events need source-frame review, especially when a pose channel tracks the wrong leg or a narrow departure bracket follows the wrong ball. Keep this diagnostic separate from the main result table until that review is complete.
