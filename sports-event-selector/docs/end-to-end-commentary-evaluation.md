# DL/VLM/LLM and full-video commentary comparison

Entry point: `python -m sports_events.end_to_end_eval --stage all`.
Stages can also be called separately: `vlm`, `prepare`, `pipeline`, `baseline`, `evaluate`. The optional `vlm` stage replays/resumes vision inference on saved DL proposals using the original per-request cache.
Use `--run` for an isolated run directory. The notebook
`notebooks/end_to_end_commentary_evaluation.ipynb` imports these same stage functions.
Its default is cache-only review; enable `RUN_REMOTE` deliberately for paid calls.

## Models and data

- DL: frozen recall-focused center20 checkpoint and its existing test proposals.
- Tuned VLM model: `5554697559152263168@1`, endpoint `4910225415601127424` in `us`.
- Tuned commentary LLM: `2514767810677178368@1`, endpoint `8459061921969078272` in `us`.
- Full-video baseline: `gemini-3.1-flash-lite` in `global`.
- Independent judge: `gemini-3.5-flash` in `global`.
- Semantic embeddings: `gemini-embedding-001`, `us-central1`, 768 dimensions,
  `SEMANTIC_SIMILARITY` task.

The prior VLM test's complete cached results are reused, not regenerated. The
new text LLM receives only its own generated history, the previous two valid
predicted events, and the current predicted event. Invalid/null VLM events are
logged and skipped without label-based repair. No ground-truth text is inserted
into history. The append-only reference is read only for scoring.

The pilot test consists of 24 nonoverlapping eight-second excerpts of V010,
192 seconds total, with 24 labeled reference events. The baseline receives
**only those exact 24 complete clips**, never other match footage. All source
frames inside each interval are retained, resized to width 640 with audio removed.
Native Gemini video sampling still applies. The same player-reference images
are supplied. The baseline sees no event labels or reference commentary; its
own previous generated commentary supplies history across clips, with gaps
explicitly acknowledged. The source video is read locally to extract the clips.

The baseline sees each full eight-second clip; the pipeline sees DL-selected
windows. This is an offline comparison, not an equal streaming-latency test.
Generated timestamps describe estimated action completion; pipeline stamps use
window end frames and references use original annotation midpoints for temporal matching (end-frame commit stamps remain in the reference metadata). Matching allows
±3 seconds within the same clip. The earlier whole-match attempt is superseded
and its baseline produced no successful outputs.

## Scoring

`commentary_metrics.py` implements:

- Soft semantic recall/precision: mean best nonnegative embedding cosine within
  the temporal gate. Missed references and unmatched captions score zero.
  These can reuse matches and are **not factual event accuracy**.
- ROUGE-L F1: wording overlap averaged across clips containing reference events.
- Exact duplicate fraction: repeated normalized sentences within a test clip.

The judge receives source events, reference prose, times, and anonymized A/B
commentaries. Source-specific candidate IDs and extra event metadata are removed.
Each nonempty case runs twice, swapping A/B. Its matching must be one-to-one,
within three seconds, with correct player and compatible action. Credit is 1
for full annotated details, 0.5 for correct player/action missing details, and
0 for contradiction. Reports include weighted/full recall, weighted precision,
contradictions, unsupported claims, fluency/coherence, wins and order agreement.
Returned IDs, uniqueness, timing and numeric ranges are validated. Empty cases
are not sent to the judge. Unsupported means absent from sparse references,
not established false. Human review remains necessary.

## Reproduction and outputs

Install the package with Vertex dependencies, configure Google Cloud Application
Default Credentials and select project `gen-lang-client-0084130013`. Video
encoding also requires `imageio-ffmpeg` (`broadcasts` extra).

```powershell
python -m pip install -e ".[vertex,broadcasts,dev]"
python -m sports_events.end_to_end_eval --stage prepare
python -m sports_events.end_to_end_eval --stage pipeline
python -m sports_events.end_to_end_eval --stage baseline
python -m sports_events.end_to_end_eval --stage evaluate
python tools/build_evaluation_notebook.py
python -m pytest tests/test_end_to_end_eval.py -q
```

The protocol records file hashes, endpoint IDs, test scope and assumptions.
`calls/` caches completed requests by model, input, media hashes and generation
settings. Service errors retry with bounded backoff and then stop; rerunning
resumes from completed calls. Nothing edits cloud model settings or training.

Outputs: `reference-commentary.txt`, `comparison.csv`, `pipeline-commentary.txt`, `baseline-full-commentary.txt`,
`baseline-test-commentary.txt`, `judge-results.json`, `results.json`, `usage.json`,
and `report.md`. The notebook loads the same artifacts. Request durations are
not end-to-end streaming latency. Prices are estimates excluding storage,
embedding/preflight calls, local compute and interrupted/failed requests; reused
VLM costs are included once in pipeline accounting but not charged again.

## Source and model-selection references

- Pricing: https://cloud.google.com/model-garden/pricing
- Older Flash-Lite retirement: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/2-5-flash-lite
- Video sampling: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/video-understanding
- Embedding task: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/embeddings/task-types

At execution time, Gemini 3.1 Flash-Lite had the lowest current standard input
and output unit prices among the available Gemini VLMs checked. Historical
2.0/2.5 Flash-Lite list prices do not mean those retired models are callable.
Judge scores do not establish reliability beyond this reused development match.
