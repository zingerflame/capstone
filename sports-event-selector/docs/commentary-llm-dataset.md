# Text-only event-to-commentary tuning

Run `python -m sports_events.commentary_dataset` from the repository root to
create `data/tenniset_commentary_llm/`. Existing nonempty output directories are
protected; use `--output` for another build.

Upload `train.vertex.jsonl` (1,205 rows) and `validation.vertex.jsonl` (256 rows)
for Vertex supervised tuning. These use Google's contents/user/model format,
with only text parts: no image files, bytes or Cloud Storage asset references.
The JSONL files themselves must still be uploaded. Keep `test.vertex.jsonl`
(414 rows) out of tuning. `validation_full.vertex.jsonl` contains all 1,948
validation events for optional offline evaluation. The smaller validation file
is a deterministic, time-spread subset; its histories still include all earlier
eligible validation events.

Example logical input (serialized as JSON text in the user message):

```json
{
  "current_commentary": "Williams serves into court.",
  "prev_event": [
    {"name":"Williams","action_base":"serve","action_details":null,"consequence":"in"}
  ],
  "curr_event": {"name":"Azarenka","action_base":"hit","action_details":"backhand topspin","consequence":null}
}
```

Example output: `A topspin backhand from Azarenka.`

The output is only the next sentence, never a regenerated full transcript.
Append it to `current_commentary` for the following call. Pass the last two
events as `prev_event`, oldest first. At match start use an empty string and
empty array; after one event the array has one item. Full commentary history is
preserved without truncation, as requested. In live inference it is the model's
actual generated history; training histories contain target sentences.

## Provenance and limitations

The source is `data/tenniset_events_v2/sequences.json`. Each eligible Hit/Serve
annotation is used once, sorted by original interval midpoint within its match.
Splits remain V006–V008 train, V009 validation, V010 test. The known quarantined
annotation V009_serve_00000 remains excluded. Original annotation timestamps
and IDs are preserved in `audit.json`; `line-index.json` maps each output line
to its source. Neither timestamps nor future-event labels enter the input.

These are deterministic, label-derived commentary targets, **not human-written
broadcast transcripts or independently reviewed natural commentary**. Neutral
templates preserve the name, shot side/type and known serve consequence; nulls
are omitted. No point winners, scores, emotional language or causal transitions
are invented. Prior VLM errors are not used as training truths. This trains
faithful verbalization, not rich broadcast style or correction of wrong events.
The full ground-truth label set is used, not just the 78 selector-based VLM
training windows. Noisy VLM event streams should be evaluated separately.

Histories follow eligible annotations and may omit unlabeled actions. They are
not complete match transcripts. Match-level separation prevents history from
crossing splits, but these matches were already used during development.
Large full histories raise token costs; exact cloud tokenization/model limits
have not been validated. Long-context accumulation can be replaced by a recent
history policy in a separate dataset version if desired.

The build uses no cloud calls, submits no tuning job, and uploads nothing.
Tests: `python -m pytest tests/test_commentary_dataset.py -q`.

Schema reference: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/supervised-tuning/prepare-data
