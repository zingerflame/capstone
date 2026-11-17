# TenniSet correction: filter first, then map ground truth

The first TenniSet pilot mapped source triplets onto a **fixed stride-2** stream.
That was not the deployed frame-delta preprocessing. Its mapping could therefore
not validate the actual pipeline. A second issue is that all three adjacent
source anchors can map to a single retained frame after filtering.

## Corrected order

1. Decode each complete original video from frame zero.
2. Apply the shared `video.filtered_frames()` filter without consulting labels:
   grayscale 64×36 mean delta, threshold 0.035, nominal 10–20fps timing constraints.
3. Record **every retained frame's original index, global filtered index and PTS**
   in `data/tenniset_delta/feeds/Vxxx.json`.
4. Project each original annotation triplet to the closest retained frames inside
   the human-annotated interval, with at most 0.16 seconds distance to a source
   anchor. Ties choose the earlier original frame.
5. Deduplicate projections. If fewer than three unique frames remain, add the
   nearest unused eligible retained frames to reach three (or two if only two
   candidates exist). Never add a dropped frame back into the input stream.
6. If fewer than two eligible retained frames exist, mark the sequence unmappable.
   Exclude any pilot input clip containing an unresolved action rather than
   falsely labeling it negative.
7. Slice the already-filtered stream into the same original eight-second windows
   as the earlier pilot. Convert global filtered indices into local model indices.
8. Train with the existing ±3-**filtered-index** distance loss and overlap decoder.

The filter is run over the entire original video, not reset at every training
clip. Annotated targets only determine which retained images are saved for the
export; they never influence filter decisions. The full retained index manifest
includes background and all non-exported frames.

## Frame identities and files

Original export remains unchanged: `data/sequence-frames/sequences.json`.

Corrected export: `data/tenniset_delta/sequence-frames/sequences.json`.

Each corrected record preserves its original annotation and adds:

- `original_sequence_frame_numbers`: original, prefilter triplet.
- `original_frame_numbers`: original-video indices of the chosen **retained** frames.
- `filtered_frame_numbers`: indices in the whole filtered video stream.
- `image_paths`: absolute local paths to PNGs of those retained frames.
- `mapping`: nearest-frame distances, collapse count, augmentation count and status.
- `sampling`: filter parameters used to create the stream.

Training `clip.json` records additionally carry local `index`,
`filtered_index_global` and `source_frame`. Each `annotations.json` target uses
the **local filtered index**, which is what the model and distance loss consume.
The metadata preserves the other two coordinates for audit and video playback.

PNG images are saved from the decoded retained frames and pixel-verified after
writing. All candidate images within the bounded mapping neighborhood may remain
on disk; `sequences.json` identifies the actual chosen pair/triplet.

## Reproduce

From the repository root, using the completed original export and earlier pilot
preparation file to keep the original clip locations fixed:

```powershell
python -m sports_events.tenniset_filtered
python -m sports_events.tenniset_train train --dataset data/tenniset_delta/clips --output runs/tenniset_delta --cache data/tenniset_delta_features --epochs 20
```

The builder accepts `--min-fps`, `--max-fps`, `--delta`,
`--max-gap-seconds`, `--baseline-preparation`, `--output` and `--workers`.
Use a new output directory for changed parameters. The old
`sports_events.tenniset_train prepare` command remains available only for
reproducing the historical fixed-stride baseline.

Model architecture, optimizer, seed (17), 20 epochs, validation selection grid and
match groups remain unchanged. This is a **regression comparison** on the same
previously evaluated test intervals, not a fresh independent test. Both the input
sampling and the target projection change, so it cannot isolate either one alone.

## Endpoint fix and verification

The shared filter delays emission by two original frames. This fixes the old
case where an unselected penultimate frame was skipped if the last frame had
already been saved. First/last two originals are retained exactly once in order.

`tests/test_tenniset_filtered.py` checks equality with the production sampler,
label-independent filter decisions, collapsed-anchor augmentation, bounded
unmappable cases, endpoint retention, exported images and all three index spaces.
The rest of the pipeline, including existing stream/batch tests, also passes.

Filtering remains quantized by source timestamps: nominal 10fps can become
8.33fps on 25fps input. This change deliberately preserves those timing decisions;
it addresses label alignment rather than retuning the sampler.

## Completed corrected pilot

All five videos were filtered: 858,507 original frames became 343,897 retained
frames (40.06%). All 4,468 annotation records mapped successfully, including
3,568 action records. Of those actions, 1,044 collapsed to one unique nearest
frame before augmentation; all 3,568 required at least one additional bounded
retained frame to construct their final sequences.

Two training clips were excluded because their mapped actions crossed a clip
boundary. The final pilot uses 70 training clips / 5,584 frames / 60 events,
24 validation clips / 2,061 frames / 4 events, and 24 test clips / 1,940 frames /
24 events. Conversion covers the full dataset; training still uses this small
pilot subset. The anchors remain derived representatives within human intervals,
not human-verified ball-contact timestamps.

The 20-epoch run selected epoch 11 using validation only (threshold 0.1,
consensus 0.8, budget factor 3). Both checkpoints below were evaluated on the
same corrected test inputs and targets, without retuning the old checkpoint.

| Model / matching tolerance | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Previous checkpoint, exact or ±3 filtered frames | 0.0% | 0.0% | 0.0% |
| Corrected retraining, exact frame | 33.3% | 6.9% | 11.5% |
| Corrected retraining, ±3 filtered frames | 66.7% | 13.9% | 23.0% |

At ±3, 10 of 15 predicted event-frame anchors matched 72 target anchors using
one-to-one matching. These are frame-anchor precision/recall, not event accuracy.
The model retained 111/1,940 input frames (5.72%), including 96 mandatory boundary
frames. Mandatory boundary frames are not included as event predictions in the
metrics above. Low retention is not a successful compression claim when recall
is this low.

The alignment correction improves this regression result, but recall remains
poor. Four validation events are insufficient for robust threshold selection.
Next experiments should expand event-rich training/validation coverage, examine
the selection budget and false negatives on validation, and reserve new matches
for a fresh test. Sampling, targets, and two excluded training clips changed
together, so this comparison does not identify a single causal improvement.

Verification: 51 automated tests passed. The completed data audit preserved all
4,468 original annotations, verified 12,727 unique target PNG hashes and bounded
index mappings, and checked pixel correspondence for all 264 target frames across
the train/validation/test pilot. Evidence is saved in
`runs/tenniset_delta/{results,comparison,verification,conversion-report}.json`.
The full corrected export remains in
`data/tenniset_delta/sequence-frames/sequences.json`; data and images are excluded
from the capstone Git copy.
