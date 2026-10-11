# Final Analysis Methods

## Study Scope and Research Questions

The unit of analysis is a distinct penalty attempt, identified by its canonical clip ID. The study contains 39 attempts from four 2018 FIFA World Cup shootouts. Camera segments, sampled frames, and replays are evidence associated with an attempt; they are not additional independent attempts. The final analysis consolidates saved experiments. Building the consolidated table does not constitute fresh model inference, model training, a new human review, or an independent accuracy experiment.

The analysis addresses three engineering research questions. First, which requested features have available candidate evidence, and how does availability vary across the four recordings? Second, how are step-count candidates distributed, and what do parameter sensitivity and known incomplete run-ups imply for their interpretation? Third, how often do automatic ball rules issue a decision, and how do the decided cases compare with the recorded external outcome references?

These questions concern observability, uncertainty, and the behavior of the extraction pipeline. They do not estimate a causal effect of run-up style on scoring, a globally optimal penalty strategy, or the generalization performance of a trained classifier. Those conclusions would require validated action labels, an appropriate study design, and independent observations.

## Data Preparation and Provenance

The canonical manifest fixes the membership of the 39 attempts. Manually selecting penalty intervals is allowed by the project brief; the subsequent feature extraction and consolidation are scripted. Each row is assembled by clip ID from the saved step, goalkeeper, camera, and ball-event records. A row retains its source hash and separates candidates, reference outcomes, verification status, and review requirements. The old 49-row demonstration table is excluded from the final analysis.

The analysis input is `outputs/final_analysis/features_39.csv`. Its adjacent `manifest.json` records the canonical IDs and hashes of the CSV and `evaluation.json`; `provenance.json` records the source files and their hashes. The EDA reader checks the schema, exact sample count, unique IDs, sidecar agreement, and artifact hashes before computing statistics. These checks establish consistency of the saved analysis artifacts. The bundle explicitly records that raw source-video verification was not performed during this consolidation; recorded hashes do not prove that fresh detections were computed or that a candidate is visually correct.

Numeric missing values remain missing. An empty step count is not converted to zero, and an unknown category is excluded from the numerator of candidate availability while remaining in the event denominator. Boolean verification flags are interpreted separately from candidate existence. The recovered camera supplement's CSV and JSON match their recorded hashes and provide 197 candidate intervals across the 39 attempts. This establishes saved interval availability, not the correctness of interval boundaries or labels. Course-class candidates remain unknown for 187 intervals; three are behind-the-kicker candidates and seven are 45-degree-right candidates. Playback-speed status is unknown for all 197, and no interval is recorded as human verified. Four-view extraction and slow-motion identification therefore remain unvalidated.

## Question 1: Feature Availability Across Recordings

For each feature, candidate availability is the number of attempts with a recognized category or nonmissing numeric candidate divided by the number of attempts in the relevant cohort. Counts are reported for all 39 attempts and separately for each match. The figure includes an event-level availability chart and a match-level table of observed numerator and denominator pairs. It distinguishes the external outcome reference from a video-derived rule decision.

Validated ground-plane calibration, verified event-final ball position, and human-verified step counts are summarized independently. A nonempty candidate, a rule-QA pass, a stable parameter result, and an accurate measurement are different properties. None is used as a substitute for another. Silhouette height proxies and strict phase-based goalkeeper height candidates remain separate fields; a proxy is not promoted to physical height.

## Question 2: Step Candidates, Parameter Sensitivity, and Censoring

The step analysis reports the distribution of visible step-count candidates without treating ankle-signal extrema as verified foot landings. It identifies attempts with a recorded partial-runup flag. An absent partial flag means that incompleteness has not been recorded; it does not establish full coverage of the approach. The known partial attempt remains visible in the distribution and is not interpreted as a complete total.

Where saved fixed-parameter variants are available, the analysis plots their minimum and maximum candidate counts for each attempt. These ranges describe sensitivity to the saved thresholds and contact-window choices. They are not statistical confidence intervals, and their endpoints are not asserted to contain the true number of steps. Missing sensitivity information is reported as insufficient data.

The analysis does not estimate an effect of partial recording on true step count. Independent landing references are unavailable for the full dataset, and the known partial subgroup is too small to support an effect comparison. MAE, RMSE, exact step-count accuracy, and an inferred relationship between step count and scoring are therefore not reported by this EDA.

## Question 3: Ball Rules, Abstention, and External References

The automatic ball rule can issue `goal_likely`, `no_goal_likely`, or `unknown`. The external reference is stored separately as `goal` or `no_goal`. The comparison maps the two decided rule categories to their corresponding reference categories and forms a confusion table only where both a decided rule and a recorded reference exist. It reports the number of comparable events, the number of rule abstentions, and the recorded reasons for abstention. An unknown rule result is never counted as a correctly detected miss.

Covered reference agreement is the number of matching decided cases divided by the number of comparable decided cases. Coverage uses the full event denominator. Both quantities must be reported together because selecting only easy, decided cases can inflate agreement. In the present bundle, the references are recorded as external FIFA-source labels, but their individual source evidence has not been independently reverified during consolidation. The visual AI judgments are neither rule inputs nor independent human truth.

All 39 attempts have been involved in development or diagnostic inspection. They form a descriptive development cohort, not an untouched test set. The separate ten-clip kicking-foot comparison uses prior user-review references recorded in upstream annotations; the original review messages were not independently inspected during this consolidation. Methods were explored after feedback on those clips. Their reported matches must not be transferred to the other attempts or presented as held-out accuracy. The final EDA accordingly labels its rule/reference comparison as descriptive agreement and leaves held-out accuracy undefined.

## Interpretation and Reproducibility

The saved experiments provide evidence of iterative analysis through failed associations, candidate availability, parameter changes, and stricter observation requirements. The final three questions organize those records into an inspectable account of what the pipeline currently supports. Graphs and conclusions should be regenerated from the same frozen feature bundle used by the written report.

The read-only notebook `notebooks/final_analysis.ipynb` validates and loads the bundle, displays the recorded evaluation scope, recomputes descriptive summaries, and renders figures in memory. It does not run detection, write annotations, or overwrite experimental outputs. The command below exports the corresponding figures and summary to a separate EDA directory.

```sh
python src/eda_analysis.py --features-csv outputs/final_analysis/features_39.csv --output-dir outputs/final_analysis/eda
```

Independent evaluation is the next evidence requirement. A reviewer should define event boundaries and coordinate conventions, record actual review identities and dates, and verify the relevant identities, kicking feet, landing events, camera angles, and ball decisions. Before future method tuning, new and uninspected events or a new match should be reserved for a separately reported test, with every associated shot and replay grouped with its event. The existing 39 attempts have already been inspected and cannot retroactively become an untouched test set. The small sample should remain an explicit limitation; no full-mark or universal accuracy claim follows from a successful script run.

AI assistance was used to audit the existing project, implement the consolidation and EDA safeguards, and draft this methods text. The students remain responsible for checking the code, source evidence, analysis, references, and statements presented in the final report and video.
