# Vertex Gemini dataset: 20 filtered frames and player references

## What each training line contains

Vertex tuning uses `contents`, not arbitrary top-level `input` and `output` keys.
The user's input is a sequence of text and image parts. The desired output is a
JSON-encoded string inside the model message. Every example contains two named
player reference JPEGs, then exactly twenty chronological full-frame JPEGs.
There are 22 images per example. Metadata and audit details stay in sidecar files.

Schematic only (the generated JSONL includes all 20 frames, without ellipses):

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {"text": "Describe the action nearest the specified anchor time..."},
        {"text": "player_information: {\"name\":\"Williams\"}"},
        {"fileData": {"mimeType": "image/jpeg", "fileUri": "gs://YOUR_BUCKET/tenniset20/assets/players/V010/player_01.jpg"}},
        {"text": "frame01: original_frame=12921; timestamp_seconds=516.840"},
        {"fileData": {"mimeType": "image/jpeg", "fileUri": "gs://YOUR_BUCKET/tenniset20/assets/frames/V010/0000012921.jpg"}}
      ]
    },
    {
      "role": "model",
      "parts": [{"text": "{\"name\":\"Williams\",\"action_base\":\"serve\",\"action_details\":null,\"consequence\":\"in\"}"}]
    }
  ]
}
```

The dataset references actual JPEG bytes through Cloud Storage `fileData` URIs.
The documented tuning schema lists text/fileData parts, so we do not assume the
inference API's inline base64 image format is accepted for supervised tuning.
Do not upload the `.template.jsonl` files unchanged: their bucket is deliberately
unresolved. Render them with your actual destination before uploading.

## Frame selection and extraction

Source: `data/tenniset_events_v2/sequences.json`, using the preserved human interval.
Anchor: floor of `(original_start_frame + original_end_frame) / 2`.
For another specific original frame, the reusable `nearest_frames()` function
accepts that frame index as `anchor`.

The selector reads each complete video's already-computed frame-delta manifest
from `data/tenniset_delta/feeds`. It chooses the 20 retained frames closest to the
anchor in original-video time (25 FPS sources), breaks ties in favor of the
earlier frame, then orders the chosen frames chronologically. It does not filter
again, reset the sampler at each event, duplicate edge frames, or use the student's
2–3 selected frames. This is offline context, potentially including future frames.
At video boundaries it takes the available nearest frames from the other side.

Twenty frames can contain more than one stroke. The prompt specifies the target
anchor time so the output remains a single event. Original annotation bounds stay
in the sidecar, not the model prompt. The input carries frame timestamps and the
anchor but no target player, shot class, spin or result.

Extraction verifies original video hashes and frame counters. Every seek/decode
block must reproduce an existing lossless PNG that the earlier sequential pass
verified. Mismatches stop the build. JPEGs use quality 85, maximum edge 768 pixels,
preserved aspect ratio. Receipts store each JPEG's SHA-256. Existing full exports
remain unchanged. The new images contain exact source-frame content with JPEG
compression and resizing, not lossless source pixels.

## Player reference module

`src/sports_events/player_references.py` provides two commands:

```powershell
# Automated proposal using a configured API key and an explicit vision model ID
python -m sports_events.player_references detect --image path/to/frame.jpg --provider gemini --model YOUR_MODEL_ID --output work/player-detection.json

# Build JPEG crops from a reviewed specification
python -m sports_events.player_references build --specification data/tenniset_vertex20/player-reference-spec.json --output data/tenniset_vertex20
```

`detect` supports GPT or Gemini through the existing provider adapter. It reads
visible scoreboard names, detects anonymous near/far player boxes, and proposes
name associations only when a scoreboard server marker and serving-side evidence
are available. It never identifies a person by their face. Proposals require
review. The reviewed specification supplies video ID, name, source image/hash,
source frame/time, normalized crop box, and identity evidence.

The completed local reference build uses ten reviewed crops from five videos.
This conversation inspected source images and scoreboard text. Set.Near and roster
metadata supplement the scoreboards; V008 has clipped names and explicitly uses
source-assisted attribution. **No live API detector calls were made.** Both names
appear in fixed alphabetical order in every example from that match, regardless
of the target actor. References do not cross match boundaries. Image crops retain
native resolution; small far-court players may be weak identity references.

The first V009 serve annotation conflicts with visible server evidence at source
frame 5739. `label-review.json` excludes `V009_serve_00000` from JSONL until reviewed,
while keeping its source record and extracted context. No silent relabeling occurs.
Other source labels may also need review; this is not a full annotation audit.

## Build, validate and upload yourself

From the repository root:

```powershell
python -m sports_events.vertex_dataset build
python -m sports_events.vertex_dataset render
python -m sports_events.vertex_dataset validate
```

Without a bucket, `render` creates `.template.jsonl` previews. To produce the
final files using your own bucket:

```powershell
python -m sports_events.vertex_dataset render --gcs-prefix gs://YOUR_ACTUAL_BUCKET/tenniset20
python -m sports_events.vertex_dataset validate
gcloud storage rsync --recursive data/tenniset_vertex20/assets gs://YOUR_ACTUAL_BUCKET/tenniset20/assets
gcloud storage cp data/tenniset_vertex20/train.vertex.jsonl data/tenniset_vertex20/validation.vertex.jsonl gs://YOUR_ACTUAL_BUCKET/tenniset20/
```

Use `train.vertex.jsonl` as the training dataset and `validation.vertex.jsonl`
as validation in your Vertex tuning job. Keep test data out of tuning. All model
and storage permissions, region support and token limits must be checked against
your selected model. No upload, model call, token-count request or tuning job is
performed by these dataset commands.

### Split files

- Train: all 1,205 action examples from V006–V008.
- Validation: a deterministic, time-spread subset of 256 V009 examples.
- `validation_full`: all 1,948 non-quarantined V009 examples for offline evaluation.
- Test: 414 V010 examples. This match has already been inspected during development.

The smaller validation file avoids submitting a validation set larger than the
training set and stays conservatively below the documented service limits.
`validation_full` includes the 256-example subset and must not be treated as an
additional independent holdout. The dataset has no negative-action examples.
Changing the target input from sparse stills to wider context does not itself
guarantee that names, spin or serve outcomes are visually supported.

### Output folder

`data/tenniset_vertex20/` contains `assets/`, JSONL files, `examples.json`,
`players.json`, `player-reference-spec.json`, `label-review.json`, `plan.json`,
`receipts/`, `line-index.json`, `verification.json`, and `export-report.json`.
`line-index.json` maps each JSONL's 1-based line number to its source event ID
(the arrays themselves use standard zero-based indexing).
`examples.json` maps each example back to
original/filtered indices and timestamps. The four-field target schema is unchanged.

## Documentation checked

- [Prepare supervised tuning data](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/supervised-tuning/prepare-data): contents/user/model schema and Cloud Storage upload.
- [Image tuning](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/image): image input format, 30-image limit per example, 20 MB per image.
- [Tuning limits](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/supervised-tuning): dataset, token and validation limits vary by model.

Local tests: `python -m pytest tests/test_vertex_dataset.py tests/test_structured_events.py -q`.
Mocked API tests verify payload handling, not live provider performance.

## Completed local build

The build extracted 50,784 unique context JPEGs and 10 player reference JPEGs.
All 50,794 asset hashes passed verification. The audit checked 3,823 JSONL lines
across the four files (including the validation subset repeated in validation_full),
and 1,018 seek/decode blocks reproduced previously verified source pixels.
There are 3,567 unique non-quarantined event examples. Nine targeted tests passed.
`example.pretty.json` contains one complete readable example with all 22 images.
Bucket substitution, upload, cloud token counting and actual Vertex acceptance
remain for the user's chosen project/model.
