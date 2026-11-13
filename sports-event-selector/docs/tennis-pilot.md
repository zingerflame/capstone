# Tennis broadcast training pilot — 2026-11-04

The download and training attempt completed. The resulting model failed this small held-out test and should not be used as a tennis event detector.

## What was actually processed

Ten complete professional match broadcasts from the official Australian Open YouTube channel were downloaded sequentially, totaling **14.9468 hours / 2.148 GB**. These are full match uploads with broadcast scoreboards, not highlights or isolated practice strokes. Files were checked for a decodable first frame and agreement between container duration and upload duration; this does not prove every frame is intact. Source hashes and metadata are in `data/tennis_broadcasts/inventory.json`. Files contain video; audio was not required.

Only **128 seconds** of these videos were exhaustively reviewed as a bounded pilot. The remaining nearly 15 hours are **unlabeled**, not negative examples. This run does not constitute full-match teacher annotation or full-corpus training. Sparse scouting and hand-picked early-match regions bias the sample. All ten matches are represented across train/validation/test, but only six contribute optimizer updates.

| Match upload | Duration | Split |
|---|---:|---|
| [Ashleigh Barty v Danielle Collins - Australian Open 2022 Final](https://www.youtube.com/watch?v=xqYKxmGc1Ys) | 131.8 min | train |
| [Qinwen Zheng v Aryna Sabalenka - Australian Open 2024 Final](https://www.youtube.com/watch?v=8EShbWpBm_0) | 103.4 min | train |
| [Agnieszka Radwanska v Victoria Azarenka - Australian Open 2014 Quarterfinal](https://www.youtube.com/watch?v=2EPvYcN9WHY) | 98.7 min | train |
| [Naomi Osaka v Caroline Garcia - Australian Open 2025 First Round](https://www.youtube.com/watch?v=qNsz4jh02AY) | 108.2 min | train |
| [Aryna Sabalenka v Paula Badosa - Australian Open 2025 Semifinal](https://www.youtube.com/watch?v=FMYm0jfUNd0) | 71.2 min | train |
| [Emma Raducanu v Coco Gauff - Australian Open 2023 Second Round](https://www.youtube.com/watch?v=PC1cCXFOG4w) | 87.6 min | train |
| [Emma Raducanu v Iga Swiatek - Australian Open 2025 Third Round](https://www.youtube.com/watch?v=4YZwF-d36PM) | 66.4 min | val |
| [Aryna Sabalenka v Elina Svitolina - Australian Open 2026 Semifinal](https://www.youtube.com/watch?v=ohD0yDeqYAU) | 75.5 min | val |
| [Elena Rybakina v Elise Mertens - Australian Open 2026 Fourth Round](https://www.youtube.com/watch?v=iA_Ff1-AqWM) | 74.5 min | test |
| [Coco Gauff v Olga Danilovic - Australian Open 2026 Second Round](https://www.youtube.com/watch?v=F8AxEv0FKyQ) | 79.5 min | test |

## Teacher and dataset

ChatGPT in this conversation inspected every retained frame of the 21 review clips using contact sheets. No ChatGPT API, Gemini request, or independent human reviewer was used. The committed teacher registry records 45 approximate event sequences of three frames each, text descriptions, confidence, and notes. These are pseudo-labels, not verified ground truth. Contact sheets are under `data/tennis_pilot/review/`; original frame mappings and broadcast timestamps are in each `clip.json`.

| Split | Matches | Clips | Seconds | Retained frames | Teacher sequences |
|---|---:|---:|---:|---:|---:|
| Training | 6 | 12 | 72 | 691 | 25 |
| Validation | 2 | 4 | 24 | 224 | 15 |
| Test | 2 | 5 | 32 | 287 | 5 |

Labels describe visible serves, return swings and groundstrokes using near/far court positions. They do not infer successful points, fault calls, or player identity from faces. Text is stored for auditing and never supplied to the student. Empty annotations are explicit reviewed negatives: close-ups, spectators, waiting players. The Sabalenka–Badosa replay segment is excluded as repeated action. A scout-selected Rybakina–Mertens region contained only preparation; it was kept as a hard negative and an additional later serve clip was reviewed before training.

Full frames and channel-specific scoreboard crops both feed the model. This run uses visual crops, not score OCR or demonstrated score-change reasoning. At 25 source FPS, the current timeout sampler often produces 8.33 FPS during low motion because of frame-time quantization; effective clip rates here are 8.5–12.75 FPS including forced endpoints. Thus this is approximately, not strictly, the requested 10–20 FPS. Changing sampling requires fresh frame indices and reannotation.

## Training and evaluation

The existing small CNN/Transformer trained from scratch on CPU for 20 epochs, seed 17, AdamW learning rate 0.0003, four CPU threads. Mean training loss fell from 5.1943 to 0.3907, but this did not translate to reliable event selection. Ten-frame stride-one windows, six-positive cap, three-window support, forced endpoints, distance-soft targets, and the excess-frame penalty remained enabled. Inference budget density came only from training annotations.

Checkpoint selection during training used validation F1 at the default threshold. After training, the best and last checkpoints were compared on validation only, using thresholds 0.1–0.7 and consensus values 0.5/0.65/0.8. The chosen checkpoint was epoch 8 with threshold 0.4 and consensus 0.65. Validation micro F1 was 0.1695 at ±3 filtered frames. No settings were changed after inspecting test performance.

| Held-out test measure | Trained selector | Periodic baseline |
|---|---:|---:|
| Predicted event frames | 10 | 26 |
| Teacher event frames | 15 | 15 |
| Matched within ±3 frames | 0 | 3 |
| Precision | 0.000 | 0.115 |
| Recall | 0.000 | 0.200 |
| F1 | **0.000** | **0.146** |

Metrics use one-to-one matching, so multiple predictions cannot earn repeated credit for one teacher frame. Boundary-context frames are retained but receive no event credit. The baseline uses periodic triplets, the same training-derived budget cap, and the same decoder; actual output counts differ because the cap is a maximum, not a quota. The student retained 30/287 test frames (10.45%), including 20 mandatory boundary frames. Exact-frame and ±1/±2 F1 were also zero.

The student proposed early waiting frames in Gauff–Danilovic and preparation frames in Rybakina–Mertens, while returning no event sequence for the reviewed Rybakina serve clip. It returned no proposals on the two off-court test clips. This suggests some coarse scene distinction but no demonstrated stroke timing. Small sample size, approximate teacher timing, low-resolution visual input, loss/threshold interactions, and training from scratch are plausible contributors; this experiment does not isolate the cause. The tiny test has only five teacher events and cannot support a broad accuracy estimate.

## Saved artifacts

- `runs/tennis_chatgpt/calibrated.pt`: selected pilot checkpoint; `best.pt` and `last.pt` also retained.
- `runs/tennis_chatgpt/history.json`: per-epoch/per-clip loss and validation measurements.
- `runs/tennis_chatgpt/calibration.json`: complete validation-only search.
- `runs/tennis_chatgpt/test_metrics.json`: test metrics and baseline comparison.
- `runs/tennis_chatgpt/predictions/`: label-free predicted event arrays with broadcast timestamps.
- `runs/tennis_chatgpt/dataset_snapshot.json`: split/group membership and clip fingerprints.
- `examples/tennis_broadcasts.json`: ten-source manifest.
- `examples/tennis_review_selections.json`: exact reviewed time intervals and scoreboard regions.
- `examples/tennis_teacher_annotations.json`: teacher labels and provenance.

Large video files, prepared images and weights remain local and are ignored by Git. Small metric snapshots are committed under `docs/tennis-results/`.

## Reproduce

On this machine, with the prepared data already present:

```powershell
./run-tennis.ps1 -Epochs 20 -Threads 4 -Output runs/tennis_chatgpt_repeat
```

For a portable installation, from the repository root:

```powershell
python -m pip install -e ".[dev,broadcasts]"
python -m sports_events.broadcasts acquire --manifest examples/tennis_broadcasts.json --output data/tennis_broadcasts
python -m sports_events.broadcasts review --inventory data/tennis_broadcasts/inventory.json --selections examples/tennis_review_selections.json --output data/tennis_pilot
python -m sports_events.pilot --epochs 20 --threads 4 --output runs/tennis_chatgpt_repeat
```

Downloads depend on continued public availability. Regenerated files using different encoding/sampling versions need the labels checked against newly generated contact sheets before reuse. Teacher import rejects missing clip reviews and invalid indices; it does not automatically assess the truth of a label. For new regions, review all exported frames and add explicit annotations, including empty lists for genuine negatives. Do not simply label the unreviewed remainder as empty.

## Verification and next experiment

All 36 tests passed, including overlap/endpoint rules, loss backpropagation, dataset split integrity, mocked teacher transport, manual import validation and negative-clip metric aggregation. The first local test attempts hit temporary-directory and local-network restrictions; rerunning with a workspace temporary directory and local-server access passed.

The next useful experiment is a substantially larger teacher-labeled sample distributed across complete matches, including long waiting periods, replays, and different broadcast layouts, followed by independent spot review. A pretrained video/image backbone and validation of the loss terms should then be compared against this saved baseline. Repeated experiments need a new untouched test set because these test results have now been inspected. The repository’s existing Gemini integration can automate larger-scale annotation later; the present ChatGPT review is deliberately small.
