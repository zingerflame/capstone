# Dataset files

`prepare` creates one directory per video:

```
data/clips/video_id/
  clip.json
  prepared.json
  frames/00000000.jpg
  scoreboards/00000000.jpg
  annotations.json       # only after teacher/manual annotation
  vlm_cache/            # validated response cache
```

`clip.json` has `schema_version`, `id`, `sport`, `group_id`, `split`, `scoreboard_roi`, source FPS/count, sampling parameters and `frames`. Every frame record contains:

```json
{"index": 0, "source_frame": 0, "timestamp": 0.0,
 "image": "frames/00000000.jpg", "scoreboard": "scoreboards/00000000.jpg"}
```

Image paths are relative to the clip directory. Full-frame JPEGs preserve the source dimensions. Student loading resizes the whole image, without cropping out the playing field. Scoreboard JPEGs are 96×32; normalized ROI coordinates are `[left, top, right, bottom]`. A null ROI writes a blank crop and activates the missing-board mask. Coordinates in the example manifest are illustrative, not verified channel layouts.

`annotations.json` example:

```json
{
  "schema_version": 1,
  "reviewed": true,
  "provenance": {"model": "actual-teacher-id", "prompt_version": "1"},
  "events": [
    {"frame_indices": [20, 22, 25], "event_type": "pass",
     "label": "The ball handler passes to a teammate on the wing",
     "confidence": 0.93, "is_replay": false}
  ]
}
```

Each event requires a nonempty label and 2/3 distinct increasing in-range integer indices. The VLM validator additionally rejects indices outside the supplied chunk. Training unions frame indices and discards semantic labels; it does not weight them by teacher confidence. Retain descriptions for audit and later captioning.

`prepared.json` records a SHA-256 digest of the downloaded video and a manifest hash. Optional manifest `sha256` verifies a known source hash. Caches resume matching configurations. A local file changed in place while retaining the same manifest path is not automatically detected on resume; use a SHA-256 in the manifest and a fresh destination to prepare a new source version.

`prediction.json` returns `event_sequences` as arrays of zero-based filtered indices, `event_sequence_frames` as arrays of full frame records, `selected_frames` as the final retained set including required context, and `boundary_context`/`forced_indices` separately. Diagnostic agreement/support arrays explain each decision. No caption/class label is predicted. Joining each record's `image` to its input clip directory retrieves the selected image, while `source_frame` and `timestamp` map it to the original broadcast.

Inputs are your own local dataset/configuration files. The repository is not designed as a public untrusted-manifest ingestion service. Secrets stay in environment variables; `.env`, downloaded media, checkpoints and training runs are excluded from Git.
