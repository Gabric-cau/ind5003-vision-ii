# Original 36-video height augmentation, 2026-10-26

The fixed experiment trained a real classifier on German Cup 8 plus 36 original
2018 events. It did not improve the previously inspected Italian 7 development
events, so the candidate is not promoted. The World Cup 8 was not opened or
evaluated again; it had already been released as a test before this development.

| Training cohort | Italian correct | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| German 8 + original 36, new candidate | 2/7 | 28.57% | 0.1481 |
| German 8 + resolved core 8, prior candidate | 4/7 | 57.14% | 0.4444 |
| German 8, original first-frame difference head | 4/7 | 57.14% | 0.3833 |
| Training-majority `low` control | 5/7 | 71.43% | 0.2778 |

These are development comparisons on one seven-event match, not independent
accuracy estimates. None of the learned height heads exceeds the majority
control by accuracy here. The results do not establish that more data is harmful;
they show that this fixed descriptor and mixed-source cohort did not transfer.

## Data actually used

The original human reference contains 39 events: 17 low, 13 middle, 7 high and
2 unknown. Koke is also excluded because old middle and newer low descriptions
conflict. The admitted original 36 therefore contain 17 low, 12 middle and 7 high.
Adding German 8 gives 44 training events: 20 low, 14 middle and 10 high. This adds
28 previously unused original events and replaces the eight duplicate core cuts
with their originals. No event is counted twice through its core and old cuts.

All 36 original MP4 files exist on `lab4090`; extraction independently recomputed
each video SHA-256 and matched its original human-reference source hash. The
existing `pose_v7` automatic first-viable shooting windows are qualified and state
`reference_labels_used_for_prediction=false`. Their durations are 2.33-13.07
seconds. Extraction reads only these source-only windows, never human contact
times, manual replay boundaries, target labels, or human-selected frames.

The old-window producer differs from the external automatic first-shot producer.
Both are automatic, but this is not a pure sample-count-only experiment. The
cached pose JSON SHA is recorded for every window. Historical pose caches do not
embed the exact producer-code hash from their creation, so that historical
binding cannot be reconstructed merely by hashing today's source file.

## Fixed Training And Artifacts

Each event uses 64 evenly spaced frames, capped at 20 seconds, resized to
512x288 and passed through the same JPEG-95 normalization and frozen ImageNet
ResNet18 encoder. Spatial 2x2 features are averaged in four temporal bins and
the first-frame feature is subtracted, producing 8192 inputs. StandardScaler,
PCA-7 and balanced logistic regression C=0.1 are fitted only on the 44 training
rows. There was one fit and no subsequent hyperparameter search.

The 36-video encoding completed in 18.64 seconds on the laboratory RTX 4090,
with other GPU work allowed concurrently. This is feature-extraction time only,
not an isolated end-to-end inference benchmark. All source embeddings, including
one NPZ per event and the combined cache, are retained locally (about 2.77 MB).

Artifacts are under `outputs/external_ball_training_2026-10-26/height_old36_v1/`:

- `protocol.json`: fixed cohort, exclusions, feature settings and evaluation scope.
- `source_only_inputs.json`: video hashes and automatic windows, without targets.
- `source_embeddings/`: all 36 embeddings, frame IDs, video/window/code hashes.
- `model/report.json`: per-event Italian predictions, all controls and provenance.
- `model/height_weights.npz`: portable numeric parameters of the newly fitted head.
- `model/training_source.py`: exact new training source snapshot.

Protocol SHA-256: `839cfeadac2f9d05a2ecc7f738ed4b17507270eae1b4ad12ccfd9daa8e414260`.
Weights SHA-256: `4b0f844589b35b433e374402584530836e3f38d1d6ce1e34a8c8bce4612c771f`.

Four focused tests passed: source-only/duplicate guards, fixed automatic-window
sampling, human label/source binding, and training-only preprocessing with
numeric export parity. The existing deployment was not replaced. A future
changed height model needs a new independent match; the previously released
World Cup test cannot be reported as unseen again.
