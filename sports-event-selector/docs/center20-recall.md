# Recall-focused 20-frame selector

The user's requested range is the **predicted twenty-frame output window**, not
the original annotation interval. For a predicted window `[s, ..., s+19]`, the
center is `c=s+9.5`. For truth center `t`, define distance `d=abs(c-t)`:

```text
credit = 1 - 0.7 * d / 9.5    when d <= 9.5
credit = 0                     otherwise
```

Thus credit is 1 at the center, 0.65 halfway to an edge, and 0.30 at either
endpoint. Outside the window it is zero. The endpoint discontinuity is intentional.
This is a training target and evaluation credit, not a calibrated probability
or a claim of 30% real-world accuracy.

Truth remains the original human interval midpoint, projected continuously into
delta-filtered frame coordinates. Predictions remain exactly twenty unique,
consecutive filtered frames, including when output windows overlap.

## Training and selection changes

- Replace the old ±3 distance target with the full-window credit rule.
- Increase the positive-region loss weight from 0.50 to 0.75. All-negative clips
  still contribute negative supervision.
- Reduce each excess-frame and excess-center-mass penalty weight from 0.20 to 0.05.
- Validate thresholds 0.05/0.1/0.2/0.3/0.5, consensus 0.25/0.5/0.8, and event-budget
  multipliers 1/2/3/6. The base event rate still comes from training labels only.
- Rank validation candidates by ordinary recall first, then weighted recall,
  precision, fewer predictions and smaller budgets. Test data never chooses these.

The encoder, ten-frame Transformer, six-positive cap, three-window support rule,
and minimum center separation remain unchanged. Several changes occur together,
so this comparison does not isolate the new decay rule alone.

## Metrics

One-to-one matching maximizes the number of covered truth events, then summed
credit. A single output cannot receive credit for several truth events, and
several outputs cannot repeatedly claim the same truth.

- Ordinary recall: matched truth events / all truth events.
- Weighted recall: summed decayed matching credit / all truth events.
- Precision: matched predictions / all predictions.
- Retention: unique selected filtered frames, including mandatory boundary context,
  divided by all input frames. This differs from image count sent to a VLM when
  overlapping windows repeat frames in separate requests.

The old exact/±0.5/±1/±2/±3 center metrics are also saved as localization diagnostics.
The commentary adapter recognizes `target_profile=window_recall` and uses window
coverage for its event comparison. Its comparison does not replace the student's
weighted-credit report.

## Completed 20-epoch pilot

Seed 17; same 70 training, 24 validation and 24 test clips. Validation selected
epoch 8, probability threshold 0.3, consensus 0.5 and budget multiplier 6.
It covered 4/4 validation events with 29 predictions, demonstrating the precision
cost of prioritizing recall. The validation set contains only four events.

Both frozen checkpoints below use identical test clips and the **same new scoring
rule**. The old checkpoint was not retuned.

| Reused test metric | Previous center20 | Recall-focused center20 |
| --- | ---: | ---: |
| Matched events | 4 / 24 | 21 / 24 |
| Recall | 16.7% | 87.5% |
| Weighted recall | 11.6% | 65.1% |
| Precision | 100.0% | 61.8% |
| F1 | 28.6% | 72.4% |
| Predicted windows | 4 | 34 |
| Unique frame retention | 8.25% | 21.19% |

Under the unchanged stricter ±3 metric, new recall is 62.5% and precision 44.1%,
versus zero for the previous checkpoint. The gain is therefore not solely a
consequence of expanding the evaluation tolerance, though broad changes and the
small reused test prevent a causal or generalization claim.

VLM frame inputs before deduplication increase from 80 to 680 (4 vs 34 windows,
20 images each), excluding player references. More events are found, but this
operating point costs more frames and produces more false positives.

## Run and artifacts

```powershell
python -m sports_events.tenniset_train train --profile window_recall --output runs/tenniset_center20_recall_new
# Reproduce the earlier narrow target profile in a fresh folder:
python -m sports_events.tenniset_train train --profile center3 --output runs/tenniset_center20_legacy_new
```

Completed artifacts: `runs/tenniset_center20_recall/model.pt`, `protocol.json`,
`frozen_selection.json`, `results.json`, `comparison.json`, `test-details.json`,
and `predictions/`. Raw source labels and earlier checkpoints remain unchanged.
Implementation: `src/sports_events/center20.py`. Tests check endpoint credit,
outside-window cutoff, duplicate prediction handling and finite training gradients.
