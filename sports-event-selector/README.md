# Sports Event Selector

A local research repository for learning to select 2–3-frame sports-event proposals from broadcast video. It includes sequential video acquisition, frame-delta sampling, scoreboard crops, VLM taxonomy generation and annotation, PyTorch training, overlap voting, inference, evaluation, and an incremental frame-feed API.

**TenniSet ground-truth import:** see [exact-frame conversion and training](docs/tenniset-ground-truth.md).
For current training, use the [filter-first correction](docs/tenniset-filtering-correction.md):
`python -m sports_events.tenniset_filtered` applies the actual full-video delta
filter before projecting and augmenting targets to nearby retained frames.
The new `sports_events.tenniset` adapter preserves every publisher annotation and
exports lossless source-frame PNGs with absolute paths and zero-based original-video
indices to `data/sequence-frames/sequences.json`. Only Serve/Hit annotations become
action targets; their representative frame triplets are derived from human intervals.
`sports_events.tenniset_train` prepares surrounding footage and runs a match-disjoint
pretrained-model pilot without a VLM teacher. Check the conversion and run reports
for completion; these commands do not change the historical results below.

**Status:** a real tennis pilot has now been trained, but it failed the small held-out evaluation. Ten full Australian Open broadcasts (14.95 hours) are downloaded locally. ChatGPT visually annotated 128 seconds across those matches; the student trained for 20 epochs on 72 seconds from six matches. Test frame F1 was 0, versus 0.146 for a periodic baseline at ±3 frames. This is not a working sports detector. See [the tennis experiment report, sources and reproduction steps](docs/tennis-pilot.md).

**Second experiment:** a controlled loss revision improved F1 from 0.146 to 0.256 on 28 seconds of fresh test intervals; recall is still only 16.7%. See the [detailed diagnosis and results](docs/second-experiment.md) and [four-diagram visual guide](docs/second-experiment.html). The original run remains preserved.

Gemini remains the default automated VLM integration; these pilots used ChatGPT visual inspection in the conversation, with no paid VLM requests. The earlier [GolfDB sample](docs/golfdb-baseline.md) and synthetic demo remain available. Gemini annotation requires `GEMINI_API_KEY`.

## Quick start

Use Python 3.10+ (tested with Python 3.12). From this repository:

On this machine, the supplied `run.ps1` can use the already-installed workspace dependencies directly: `./run.ps1 demo --output runs/demo --epochs 2`. For a portable standalone installation:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
python -m sports_events demo --output runs/demo --epochs 2
```

The demo creates three synthetic videos in separate match groups, samples them, trains a small model, saves checkpoints, predicts on the test clip, and writes evaluation JSON. Its annotations are clearly marked synthetic. Re-running the demo preserves existing prepared data.

For CUDA training, install the appropriate PyTorch build for your system and pass `--device cuda` to training/inference. A CPU is sufficient for the demo. The real training time and memory budget depend on video length; benchmark a short clip first.

## Real-data workflow

1. Create a video manifest using [the example](examples/manifest.json). Sources may be direct HTTP(S) video URLs or local video paths relative to the manifest. Match IDs must keep every clip/replay from the same game in the same train/val/test split. Database-specific access/download scripts can export this manifest; this repository does not bypass a dataset's access requirements. Web watch pages are not direct video URLs.
2. Configure Gemini with a local environment variable. The model ID is configurable; the default follows the model used in Google's documentation when this repository was created. Credentials and quota must support the model you select.
3. Generate the taxonomy with the teacher, then process one video at a time:

```powershell
$env:VLM_PROVIDER = 'gemini'
$env:GEMINI_API_KEY = 'YOUR_KEY'
$env:GEMINI_MODEL = 'gemini-3.8-flash'
python -m sports_events taxonomy --output data/taxonomy.json
python -m sports_events prepare --manifest examples/manifest.json --output data/clips --annotate --taxonomy data/taxonomy.json
```

The generic example contains placeholder paths: replace them before running, or use the ready GolfDB sample manifest in the baseline guide. `prepare` acquires, samples, annotates, and then removes the temporary downloaded copy before the next video. Original source files are untouched. Use `--keep-downloads` to retain copies. If annotation fails, the prepared clip and response cache remain available for resumption. A changed manifest or annotation configuration requires a fresh output location to prevent mixing data versions. For an alternative local endpoint, set `VLM_PROVIDER=compatible`, `VLM_ENDPOINT`, `VLM_MODEL` and optionally `VLM_API_KEY`.

4. Inspect each `annotations.json`, fix missing/incorrect events and mark `"reviewed": true`. Empty `events` is a valid negative example. Training defaults to reviewed data; `--allow-unreviewed` explicitly enables experiments with teacher-only pseudo-labels.
5. Train and evaluate:

```powershell
python -m sports_events train --dataset data/clips --output runs/baseline --epochs 20 --device cpu
python -m sports_events evaluate --dataset data/clips --checkpoint runs/baseline/best.pt --output runs/baseline/test_metrics.json
python -m sports_events predict --clip data/clips/example_test --checkpoint runs/baseline/best.pt --output runs/baseline/prediction.json
```

To process unseen videos, prepare a new manifest **without** `--annotate`, then call `predict`. No annotations are read at inference. Text labels are retained for teacher supervision/review only. Final predicted event sequences contain frame indices, paths and timestamps, with no event class or caption. A downstream VLM can caption these proposals.

## What is implemented

| Requirement | Implementation |
|---|---|
| Nine sports | Tennis, baseball, golf, badminton, basketball, soccer, American football, rugby and table tennis; aliases accept `Batminton` and `PingPong` |
| Frame reduction | Grayscale absolute delta versus last selected image; movement can trigger up to 20 FPS, a timeout maintains approximately 10 FPS; source frame/time mapping is preserved |
| Scoreboards | Per-channel normalized rectangle in the manifest; full-resolution crop supplied to teacher, separate learned visual branch for student; a missing-board mask prevents false context |
| Teacher annotation | Overlapping 40-frame chunks, 10-frame context overlap, validated 2/3-frame events, JSON cache and provenance, replay exclusion prompt |
| Selector | Full-frame CNN + scoreboard CNN + time gaps + two-layer temporal Transformer; ten independent binary relevance outputs |
| Sliding windows | Stride 1; 100 frames produce 91 ten-frame windows |
| Window cap | At most six positive predictions per window |
| Overlap | At least three distinct windows; rank global frames by positive-vote fraction, then mean probability |
| Endpoints | Always retain filtered indices 0, 1, N−2, N−1 (user's 1, 2, N−1, N); deduplicate for tiny clips |
| Distance loss | Targets 1, .75, .5, .25 at distances 0, 1, 2, 3; zero beyond 3 **filtered indices** |
| Excess penalty | Actual uncapped, post-voting/grouping count in the forward loss; straight-through differentiable surrogate for backward |
| Budget | Training target is unique teacher frames union mandatory endpoints; inference budget rate estimated only on training labels |
| Sequence output | Deterministic nearby-frame grouping into 2/3-frame proposals; isolated optional selections discarded |
| Verification | Unit/integration tests, mock HTTP teacher, whole-pipeline synthetic run, held-out matching metrics |

## Frame feed

`FrameStream` accepts already delta-filtered BGR image arrays and monotonically increasing timestamps. It keeps ten visual embeddings and numeric predictions; it does not retain the image feed. `push` yields window scores. `finish` applies global overlap, budget and endpoint rules.

```python
from sports_events.learning import load_model
from sports_events.stream import FrameStream

model, state = load_model("runs/baseline/best.pt")
stream = FrameStream(model, state["budget_rate"], scoreboard_roi=[0.02, 0.02, 0.40, 0.15])
for bgr_frame, timestamp_seconds in filtered_frame_feed:
    window = stream.push(bgr_frame, timestamp_seconds)
result = stream.finish()
event_frame_arrays = result["event_sequences"]
```

Exact global ranking and knowing which frames are N−1/N require end-of-stream. This implementation does not claim irrevocable live event decisions. For an endless broadcast, close logical segments and accept that endpoint/budget rules then apply per segment. See [design decisions](docs/design.md), [dataset schema](docs/data.md), [research and 13 primary sources](docs/research.md), and [verification results](docs/verification.md).

## Limitations that matter

- A ten-frame window at 10–20 FPS spans roughly 0.5–1 second. Long plays, player identity and causality may require more context. The teacher sees larger chunks, but the student intentionally follows the requested short-window design.
- The default visual encoders train from scratch at 96×96. The separate third-experiment runner now supports frozen ImageNet ResNet-18 features at 224×224 with full-frame letterboxing. Both resolutions can obscure the ball and small gestures.
- The student scoreboard branch learns from pixels; it is not an OCR parser. Score changes are delayed and do not identify most routine actions. Configure each channel's crop and include missing/occluded scoreboards in training.
- Sequence grouping is a heuristic, not a learned event-boundary model. Close actions may merge and long actions may split. Names such as Curry/LeBron are not inferred by this selector.
- VLM output is pseudo-ground truth and may miss fast motion, hallucinate or repeat events. Cached responses and human review make these errors inspectable; the code cannot guarantee teacher accuracy.
- GolfDB supplies the initial golf adapter. No database covering all nine requested sports is assumed; the common manifest interface can receive the forthcoming reference dataset.

## Third experiment and presentation

Six additional training runs compare pretrained appearance features, adjacent-feature motion cues, and fixed versus expanded selection budgets. On 28 seconds of newly reviewed test intervals, the validation-selected variant averages F1 **0.419 ± 0.053** across three seeds versus **0.200** for the prior checkpoint on the same footage. Retention increases from **12.1% to 21.3%**. These are approximate frame-anchor metrics on a tiny pilot, not full-match event accuracy.

See [experiment design, results and reproduction](docs/third-experiment.md) and the [seven-slide speaking script](docs/seven-slide-script.md). Training still uses only 72 seconds. All 43 tests pass, and the pretrained inference command has been exercised without reading teacher labels.
