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
