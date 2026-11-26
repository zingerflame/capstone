# VLM datasets from the recall-focused selector

`src/sports_events/selector_vlm.py` runs the frozen
`runs/tenniset_center20_recall/model.pt` on the existing delta-filtered pilot
clips, then builds Gemini Vertex tuning JSONL. It does not retrain the selector,
upload files, or launch VLM training. The output is `data/tenniset_selector_vlm/`.

## Completed local build

| Split | Match IDs | Predicted windows | Labeled tuning/evaluation rows | Distinct target events |
|---|---|---:|---:|---:|
| Train | V006–V008 | 169 | 78 | 48 |
| Validation | V009 | 29 | 6 | 3 |
| Test | V010 | 34 | 31 | 17 |

There are 60/4/24 original eligible events in the train/validation/test clip
contexts. This export covers existing short clips, not all frames of the full
matches. Training predictions are in-sample. The validation match also selected
the DL checkpoint, and the test match has been inspected during development.
This is a pilot export, not an independent end-to-end benchmark. Six validation
rows are particularly weak evidence, and cloud tuning eligibility must be checked
for the selected Gemini model. Expand to more match footage before a serious run.

## How labels attach to predictions

Each example uses exactly the selector's 20 consecutive delta-filtered frames.
Frames are not recentered on the human annotation. The input anchor is the
predicted center's original-video time, not the ground-truth time. The target is
the eligible human event whose original interval midpoint is closest to that
center and lies within the window; ties choose earlier interval, then ID.

Targets retain `name`, `action_base`, `action_details`, and `consequence`.
`examples.json` preserves original annotations, exact original timestamps,
source frame numbers, global filtered indices, and the mapped target ID.
Ground-truth annotations and timestamps never enter the model input. Player
references include both names in alphabetical order, irrespective of target actor.
Every model example has 22 images: two reviewed player references plus 20 frames.

Unmatched windows are excluded from tuning rather than fabricated as negative
examples. They remain in `examples.json`, and all 34 test windows are evaluated.
The three unmatched test windows are unscored because absence of a matching
annotation is not proof of no visible action. The existing quarantined label
`V009_serve_00000` remains excluded. The prompt allows JSON null for an unclear
window, but this positive-only dataset does not teach abstention explicitly.

Several windows can map to the same event. Therefore 31 labeled test windows
represent 17 distinct target events. This is different from the selector's
21/24 one-to-one temporal coverage: here a window is assigned its single nearest
event, whereas temporal matching can assign a different covered event. Even
perfect single-target captions cannot exceed 17/24 exact-event recall under
this export's target assignment. Multi-event outputs are a possible later change.

## Build and prepare cloud files

From the repository root, using Python with this package installed:

```powershell
python -m pip install -e ".[vertex]"
python -m sports_events.selector_vlm build
```

The build refuses to overwrite an existing output directory. For a new build:

```powershell
python -m sports_events.selector_vlm build --output data/tenniset_selector_vlm_v2
```

Change `--checkpoint`, `--dataset`, `--labels`, or `--references` to use a new
compatible run. The checkpoint must be `center20` / `window_recall`. Match splits
are checked explicitly. Assets are copied and SHA-256 verified; the checkpoint
and label hashes are recorded in `build-report.json`.

Without a bucket the files end in `.template.jsonl` and deliberately contain an
unresolved bucket. Generate uploadable references with your actual bucket:

```powershell
python -m sports_events.selector_vlm render --gcs-prefix gs://YOUR_BUCKET/selector-vlm
gcloud storage rsync --recursive data/tenniset_selector_vlm/assets gs://YOUR_BUCKET/selector-vlm/assets
gcloud storage cp data/tenniset_selector_vlm/train.vertex.jsonl data/tenniset_selector_vlm/validation.vertex.jsonl gs://YOUR_BUCKET/selector-vlm/
```

Use only `train.vertex.jsonl` and `validation.vertex.jsonl` in the tuning job.
Keep all test files out of tuning. Each tuning row is `contents` with user image
and text parts followed by a model message containing the target JSON string.
The image references use `fileData` / `gs://` paths, following Google's
[image-tuning format](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/image).
`test-inputs.*.jsonl` is an inference manifest with IDs and user-only contents,
not a tuning file or a Vertex batch request file.

## Test a completed Vertex model

The evaluator uses the [Google Gen AI SDK](https://googleapis.github.io/python-genai/index.html)
and accepts the tuned endpoint resource returned by your tuning job as `--model`.
It reads local images directly, so evaluation does not require uploading them.
Authenticate with your own application default credentials:

```powershell
gcloud auth application-default login
python -m sports_events.selector_vlm evaluate --model projects/PROJECT/locations/REGION/endpoints/ENDPOINT --project PROJECT --location REGION --output runs/vlm-tuned-test
```

For a credential-free input check (already run locally):

```powershell
python -m sports_events.selector_vlm evaluate --model YOUR_TUNED_ENDPOINT --output runs/vlm-test-dry-run --dry-run
```

`--limit 2` can smoke-test a small subset before the complete paid run. Partial
runs are marked; their event-recall denominator remains all test events. The
actual cloud evaluator is not run until a trained model and credentials exist.

Requests contain the same ordered instructions, player references and frames as
tuning, but no model answer. Replies, token usage and call durations are cached
by model/project/region/prompt/image hashes under `calls/`. Rerun the same command
to resume after a service error. Use a new output directory for each model or
comparison. API exceptions stop the run rather than silently counting outages
as model failures. Invalid JSON or schema responses count as incorrect outputs.

Outputs are `predictions.json` and `metrics.json`, reporting normalized exact
event accuracy, per-field accuracy, invalid-output count, and deduplicated exact
event recall over all 24 reference events. Nulls must match nulls; comparison
ignores letter case and surrounding whitespace, but does not judge semantic
equivalence. Unmatched-window hallucination rates require manual review and are
not inferred from missing annotations.

## Files and audit trail

- `examples.json`: all 232 proposals, labels and provenance for auditing.
- `truths.json`: reference event denominator, including missed events.
- `train.template.jsonl`, `validation.template.jsonl`: local tuning previews.
- `test.template.jsonl`: 31 labeled test rows, never use for tuning.
- `test-inputs.template.jsonl`: all 34 test requests without answers.
- `line-index.json`: maps labeled JSONL rows back to example IDs.
- `assets/`, `assets.json`: staged images and integrity hashes.
- `predictions/`: exact frozen-model outputs for every pilot clip.
- `build-report.json`, `export-report.json`: counts, exclusions and provenance.

No VLM training or cloud evaluation has been performed by these commands.
