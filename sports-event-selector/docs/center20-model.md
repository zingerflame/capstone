# Twenty-frame event windows and center-distance evaluation

**Current default:** the recall-focused profile uses the entire predicted 20-frame
window and decays credit to 30% at its edges. See [recall profile and comparison](center20-recall.md).
The ±3 loss and pilot described below are the historical `--profile center3` run.

The current TenniSet training entry point now defaults to `center20`:

```powershell
python -m sports_events.tenniset_train train --output runs/tenniset_center20_new
```

The existing frozen ResNet18 encoder, motion features and ten-frame Transformer
remain. The supervised quantity changes from relevant frame triplets to event
center scores. Every accepted proposal expands into **exactly 20 consecutive
indices from the original delta-filtered stream**. These are not necessarily
consecutive raw video frames. Output windows can overlap for nearby events.

Implementation: `src/sports_events/center20.py`. Old checkpoints retain their old
decoder. For historical retraining use `--target-mode legacy_frames` and explicit
historical dataset/cache/output arguments. Saved earlier results remain unchanged.

## Centers and supervision

- A window `[s, ..., s+19]` has center `s+9.5`. Its central two frames are `s+9`
  and `s+10`. Half-frame centers are intentional, not rounded.
- Ground truth is the midpoint of the original human annotation interval,
  projected continuously between retained original frame indices. It is not
  the center of the augmented 2–3-frame target used in the earlier pilot.
- The loss aggregates overlapping ten-frame logits, interpolates each candidate
  center's logit between its central two frames, then uses distance-soft targets:
  `1-distance/4` within ±3 filtered positions, zero beyond that range.
- Balanced positive/negative BCE trains the center scores. Additional terms
  penalize excess predicted center mass and a differentiable estimate of the
  unique frames covered by output windows. This is a surrogate for the hard
  decoder's output cost, not differentiation through discrete grouping.
- Decoder voting retains the six-positive cap per ten-frame window and requires
  at least three-window support for both central frames. Agreement ranks first,
  then confidence. Suppression removes proposals closer than three center positions.
- The event-count budget comes from training event density. Validation chooses
  threshold, consensus and budget multiplier. No test labels set these values.

Mandatory first/last two input frames stay in `selected_indices` as context.
They earn no event credit merely by being retained. Context shorter than 20 frames
produces no event window; frames are never padded or repeated to fake a full event.

## Evaluation

Each predicted center matches at most one truth center within the tolerance.
Matching maximizes the number of matches, then minimizes their total distance.
Report precision, recall and F1 at 0, 0.5, 1, 2 and 3 filtered positions, alongside
mean center error for matched pairs. No matched pairs means a null error, not zero.
False positives and missed events remain visible through precision and recall.
Full 20-frame overlap with an event is insufficient if the window's center is wrong.

The metric counts events rather than matching 2–3 individual frame anchors. It
must not be compared directly with the old reported frame-anchor F1.
Predictions also include center timestamps in original-video seconds and original
frame metadata for all 20 frames. The frame-position metric is distinct from
elapsed seconds because filtering can make spacing irregular.

## Completed pilot

`runs/tenniset_center20/` contains the 20-epoch run, seed 17, with the same corrected
70/24/24 training/validation/test clips as the preceding pilot. Validation selected
epoch 12, threshold 0.1, consensus 0.5 and event-budget multiplier 3.

| Evaluation | Matched events | Predictions | Truth events | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation, ±3 | 2 | 4 | 4 | 50% | 50% | 50% |
| Reused test, ±3 | 0 | 4 | 24 | 0% | 0% | 0% |

Test retention is 8.25%, including mandatory context. All four predicted event
windows have exactly 20 consecutive filtered frames. This run establishes the new
output contract but does not demonstrate useful event detection.

Pilot clips are only eight seconds long. Six of 24 test centers, 16 training
centers, and one validation center lie outside the range where a 20-frame window
can center exactly inside its clip. The evaluator retains their real positions;
it does not move them to match predictions. Some edge centers can still match
within tolerance. This limitation does not by itself explain all test failures.
The next dataset revision should give each supervised event adequate surrounding
context from the full filtered feed. It should also expand validation beyond four
events and test new, independently reviewed matches.

## Prediction

The existing pretrained prediction entry point dispatches by checkpoint output mode:

```powershell
python -m sports_events.experiment3 predict --checkpoint runs/tenniset_center20/model.pt --clip data/tenniset_delta/clips/tenniset_V010_001 --output runs/tenniset_center20_example.json
```

New result keys include `output_mode`, `event_sequences`, `event_centers`,
`center_timestamps_seconds`, and `event_sequence_frames`. The legacy frame-metric
evaluator rejects center20 checkpoints to prevent misleading scores.

The commentary exporter accepts center20 predictions, sends all twenty images
with a center-time instruction, and compares centers rather than any-frame overlap.
Prepare its inputs with `--predictions runs/tenniset_center20/predictions` and a
fresh commentary output directory. The separate Vertex
20-frame training dataset remains unchanged: its anchor-centered context exports
are supervised examples, not predictions from this newly trained student.
