# TenniSet: exact frames and converted events

Source: [HaydenFaulkner/Tennis](https://github.com/HaydenFaulkner/Tennis),
Faulkner and Dick, *TenniSet: A Dataset for Dense Fine-Grained Event Recognition,
Localisation and Description*, DICTA 2017. Download locations are the publisher's
[data README](https://github.com/HaydenFaulkner/Tennis/blob/master/data/README.md).

## Requested local dataset

`data/sequence-frames/sequences.json` is an array. Each record contains:

- `original_annotation`: the original category and unmodified annotation object.
- `event`: normalized type, readable label, original attributes/description and role.
- `image_paths`: an ordered array of absolute local PNG paths inside
  `data/sequence-frames/<video_id>/`.
- `original_frame_numbers`: the corresponding ordered integers in the original
  **unsampled, entire video**, not clip-relative or filtered indices.
- `video_id` and `source_video`: disambiguate frames across matches.
- `frame_index_base`: **0**, consistent with the publisher's frame extraction code.
- `original_interval`, annotation/video SHA-256, image checksums, selection policy
  and `training_eligible`: provenance and safe consumption of the labels.

JSON has arrays rather than tuples. Position i in `image_paths` corresponds to
position i in `original_frame_numbers` and `image_sha256`. Files are shared when
multiple annotation records reference the same original frame.

All 4,468 annotation entries are preserved: 2,551 Hit, 1,017 Serve, 746 Point,
118 Game, 11 Set, 5 Match, 5 USE and 15 SPLITS. Only the 3,568 Hit/Serve records
are action-selection targets. Point/Game/Set/Match are context. USE/SPLITS are
dataset metadata; neither is silently reclassified as a sports action.
The upstream derived OTH (background) class is represented by non-anchor positions
in surrounding training footage; it is not fabricated as another positive event.

## What the ground truth does and does not mean

Human annotations provide event **intervals**, not a uniquely labeled contact
frame. The converter chooses the midpoint and its adjacent original frames,
constrained to the interval (two frames for a two-frame interval). These are
deterministic representative targets derived from human intervals. They must not
be described as exact human contact annotations or independent caption validation.
Point descriptions are preserved as context, not substituted for shot labels.

PNG extraction decodes sequentially from frame zero without timestamp rounding
or seeking. Every written PNG is decoded again and compared pixel-for-pixel with
the corresponding decoded source frame. The image checksum and source-video hash
are recorded in `<video_id>/extraction.json`. PNG is lossless relative to decoded
video pixels; it cannot undo compression already present in the source video.

`sequences.pending.json` and `conversion-report.json` are written first. The
canonical `sequences.json` is published only once every referenced image exists
and has been verified across all five videos. Training checks completion and
image hashes. Downloads and PNGs remain under ignored `data/` and are not committed.

## Reproduce conversion

Put the publisher's original `V006.json` through `V010.json` in
`data/tenniset/annotations/` and corresponding unmodified MP4s in
`data/tenniset/videos/`. From the repository root:

```powershell
python -m sports_events.tenniset convert
python -m sports_events.tenniset extract
```

`python tools/download_tenniset_videos.py` downloads the five publisher-linked
videos sequentially (about 11 GB); it verifies expected byte counts and records
SHA-256 receipts. Annotation JSONs are separate downloads from the publisher's
annotations folder. This tool uses the public Drive download form because the
connected Drive tool limits individual file downloads to 256 MB.

Both commands accept `--annotations`, `--videos` and `--output`. The extractor
can resume completed videos after verifying their source and image hashes. If
moving the project to another computer, regenerate the absolute paths by running
the extractor with the new output location; do not keep old machine paths.

## Training integration

```powershell
python -m sports_events.tenniset_train prepare --clips-per-video 24
python -m sports_events.tenniset_train train --epochs 20
```

The first run is a bounded integration pilot, not a claim to train on every
annotation. Preparation uses evenly spaced eight-second intervals within each
video's USE range, independent of event labels. It includes negative intervals.
Each original match stays in one split: V006/V007/V008 train, V009 validation,
V010 test. Publisher temporal SPLITS are retained but not used, because they
mix intervals from the same match across training and evaluation.

For unambiguous mapping, the pilot retains every second original frame (12.5fps
for these 25fps videos), instead of using adaptive frame delta. It maps the exact
stored source-frame targets to nearest retained indices, deduplicates and records
both original and mapped indices. This is a documented preprocessing difference
from previous delta-sampled pilots, so do not attribute differences to labels
alone. An event with anchors crossing an input clip boundary is recorded as
excluded. Mapping must leave 2–3 unique indices; it never silently duplicates a
frame to satisfy the schema.

The model remains the frozen ImageNet ResNet-18 plus motion-aware ten-frame
Transformer head and v2 loss. No teacher API is called. Board ROI is absent in
this first integration, so its crop features are explicitly masked. The full
scene still includes the broadcast graphics. Validation chooses epoch, threshold,
consensus and budget; the chosen settings are written before test prediction.
The test evaluates derived anchor targets, not event captions or human contact
timing. Existing ±3-index metrics represent ±0.24 seconds at this fixed rate.

`reviewed=true` on imported training annotations means they come from the
publisher's human annotations with explicit derived-target provenance; it does
not claim a second human audit. Increase `--clips-per-video` to broaden coverage,
or design a full USE-range partition with explicit boundary handling before a
full-dataset benchmark. Do not train using only the isolated positive PNGs:
the selector also needs the surrounding frame stream and negative examples.

## Code and checks

- `src/sports_events/tenniset.py`: conversion, label mapping, exact PNG extraction.
- `src/sports_events/tenniset_train.py`: label-independent pilot preparation and
  reusable pretrained-model training entry point.
- `tests/test_tenniset.py`: interval validation, category roles, original-index
  mapping, preservation of annotations and pixel-verified extraction/resume.
- `data/sequence-frames/conversion-report.json`: actual conversion completion.
- `data/tenniset_pilot/clips/preparation.json`: actual selected intervals/counts.
- `runs/tenniset_pilot/protocol.json`, `frozen_selection.json`, `results.json`:
  actual experiment evidence when the run completes.

## Completed local export and pilot (15 November 2026)

The completed index contains 4,468 records referencing 13,197 distinct verified
PNGs (about 16.16 GB). There are 3,568 action records. All source annotations were
compared against the converted records and preserved exactly. The implementation
test suite passes 48 tests, including real video decoding on synthetic fixtures.

The 24-clips-per-video pilot completed 20 epochs, seed 17:

| Split | Matches | Clips | Seconds | Sampled frames | Action events | Negative clips |
|---|---|---:|---:|---:|---:|---:|
| Train | V006–V008 | 72 | 576 | 7,200 | 68 | 46 |
| Validation | V009 | 24 | 192 | 2,400 | 4 | 22 |
| Test | V010 | 24 | 192 | 2,400 | 24 | 17 |

Validation selected the expanded-budget checkpoint from epoch 19, with factor 2,
probability threshold 0.1 and consensus 0.65. Validation F1 was 0.4: two matched
anchor frames out of eight ground-truth anchors, from only four events. That very
small validation set is a material limitation, not strong evidence of quality.

On the untouched test match's selected intervals, the model predicted three event
anchors and matched **zero of 48** ground-truth anchors at every evaluated tolerance
(0–3 sampled frames). Precision, recall and F1 were all zero. Retention was 4.125%,
including 96 mandatory boundary frames and three predicted event frames.
Checkpoint: `runs/tenniset_pilot/model/expanded.pt`.

This is a completed but unsuccessful model integration pilot. It does not invalidate
the verified source-frame mapping, and it is not a full TenniSet benchmark. The
pilot trained on 68 action events, not all 3,568 exported actions. A next experiment
should expand training and validation coverage, audit representative interval
targets against contact timing, and compare budget/recall operating points on
validation. Do not tune against this test result and then call the same test fresh.
