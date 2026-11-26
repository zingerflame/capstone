# Append-only commentary ground truth

The frozen reference is `data/tenniset_commentary_append_only/ground-truth.jsonl`.
It contains 3,567 eligible human-derived Hit/Serve events across five matches.
The known quarantined V009 serve remains excluded. Commentary is synthetic,
factual verbalization of those labels, not a human broadcast transcript.

Each event appends exactly one sentence. Prior text, spelling and punctuation
are never revised. The update is `new_state = old_state + append_text`.
`stamped_append_text` includes the original-video time and zero-based frame:

```text
[HH:MM:SS.mmm | frame N] New commentary sentence.
```

Events are committed at their original annotation **end frame**, ordered by end
frame, then start frame and ID. This differs deliberately from the older
midpoint ordering: a recorded serve consequence is not announced before its
annotated interval has finished. These stamps are label-availability times, not
measured inference or spoken-word timings. Original start/end intervals and
timestamps remain in every record. Annotations can be sparse; no connective
claims about unobserved play are invented.

Read `V006.commentary.txt` through `V010.commentary.txt` for timestamped timelines.
The corresponding `.plain.txt` files contain the accumulated prose. The JSONL
stores new text, source event, annotation, stamps, and SHA-256 chains for both
the text state and event record. The builder refuses to overwrite an existing
version. The verifier detects changes against `manifest.json`; this is not an
operating-system write lock or protection against rewriting both log and manifest.

The text-only tuning files are extracted **from the saved timeline**, without
regenerating or editing its sentences:

| File | Rows |
|---|---:|
| train.vertex.jsonl | 1,205 |
| validation.vertex.jsonl | 256 |
| validation_full.vertex.jsonl | 1,948 |
| test.vertex.jsonl | 414 |

The user payload retains `current_commentary`, `prev_event` and `curr_event`.
`current_commentary` is the exact plain-text prefix before that entry. The output
is its new sentence only. The application appends the sentence and attaches the
frame/time metadata; the language model is not asked to invent timestamps that
are absent from its event input. Full history is retained, resets at each match,
and never includes future events. `prev_event` contains the preceding two events
in this commitment order, or fewer at match start.

Train matches remain V006–V008, validation V009, and test V010. These are reused
development matches. Training uses correct reference histories; free-running
generated histories and imperfect VLM events need separate evaluation. Targets
teach factual verbalization and are stylistically simple.

```powershell
python -m sports_events.commentary_timeline
python -m sports_events.commentary_timeline --verify
python -m pytest tests/test_commentary_timeline.py tests/test_commentary_dataset.py -q
```

Use `--output` to create a new version. Nothing was uploaded and no tuning job
was started. Previous datasets remain unchanged.
