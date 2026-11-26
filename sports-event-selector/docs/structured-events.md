# Four-field TenniSet event targets

Run from the repository root:

```powershell
python -m sports_events.structured_events
```

This creates `data/tenniset_events_v2` from the corrected, filter-first
`data/tenniset_delta/sequence-frames/sequences.json`. Original datasets and past
experiment results stay intact. The new `sequences.json` replaces each record's
`event` with the requested four-field object and retains the old value as
`legacy_event`. Original annotations, image paths and original/filtered frame
indices remain available for auditing. It does not change student frame targets.

```json
{
  "name": "Williams",
  "action_base": "hit",
  "action_details": "forehand topspin",
  "consequence": null
}
```

## Mapping

| Source category | action_base | action_details | consequence |
| --- | --- | --- | --- |
| Hit | hit | Lowercase Side + Type, such as forehand topspin | null |
| Serve | serve | null | in, fault, let, or null |
| Point/Game/Set with valid winner | win_point / win_game / win_set | null | score: followed by the literal source score, or null |
| Point/Game/Set without valid winner | complete_point / complete_game / complete_set | null | Same score handling |
| Match | match_interval | null | null |
| USE | usable_region | null | null |
| SPLITS | dataset_split | Source split name | null |

`name` is the source Player for actions, or the valid source Winner for context
labels. Otherwise it is the explicit string `unknown`, since the requested
schema requires a string. No player identity is guessed. In particular, the
malformed Winner value `40-30` is flagged and never treated as a person.

`Unsure` is an unknown shot type, not a shot technique. For example,
Backhand + Unsure becomes `backhand`, with an audit issue. A completely absent
detail becomes JSON `null`, never the string `"null"`.

The requested object has no score_result property. Scores belong in consequence
on their own point/game/set records. No point winner or score is propagated into
a stroke record. A serve marked In does not imply an ace or won point. Source
free-text descriptions remain in original_annotation and are not heuristically
expanded into extra facts or events.

## Outputs and fine-tuning boundary

- `events.json`: all 4,468 event objects, in video/source-interval order.
- `timestamped-events.json`: compact events with video ID and original timestamps.
- `sequences.json`: the same events with complete source and image provenance.
- `event.schema.json`: exact four-key JSON Schema, with nullable details/outcome.
- `fine_tuning/train.jsonl`: 1,205 action examples from V006–V008.
- `fine_tuning/validation.jsonl`: 1,949 action examples from V009.
- `fine_tuning/test.jsonl`: 414 action examples from V010.
- `conversion-report.json`: counts and source-quality issues.

Each JSONL example has an id, an input instruction plus ordered local image_paths,
and an output object containing exactly the four requested fields. These are
provider-neutral local manifests, **not Gemini upload-ready JSONL**. A later
provider adapter must upload/encode images and wrap the output as the assistant
target using the chosen tuning interface. The output object itself is the target
the model should learn to produce. No tuning job has been submitted.

Only the 3,568 mapped serve/hit records enter these manifests. All context and
administrative records remain in the complete export, but are not action targets.
The split is by entire source match, preserving the previous match separation,
and covers full converted action labels rather than only the earlier pilot clips.
V010 has already been inspected during development and is not a fresh benchmark.

The annotations may contain player identities or serve outcomes that the sparse
images cannot establish. Inspect training examples for visual learnability before
tuning; correct JSON structure alone does not make every target visually supported.
There are no negative-action examples in this export.

## Original dataset timestamps

`timestamped-events.json`, `sequences.json` and each fine-tuning example include
`original_timestamps`. Start/end come directly from the original annotation's
frame bounds, divided by the original video's FPS. They are relative to the
beginning of the full source video, not the cropped test excerpt or filtered
stream. Both seconds and HH:MM:SS.mmm are provided, along with original frame
bounds and FPS. The end value identifies the annotated end frame, not an
invented exclusive boundary or exact ball-contact instant. For these 25 FPS
sources the temporal resolution is 40 milliseconds.

Selected-frame timestamps are listed in image-path order. Fine-tuning inputs
include these as `image_timestamps_seconds`. Full annotated interval timestamps
remain provenance outside the four-field output, so the target schema is unchanged.
Timestamp conversion assumes the source constant frame rate used by the dataset;
it is not a wall-clock timestamp or a new measurement of action timing.

Tests: `python -m pytest tests/test_structured_events.py -q`.
