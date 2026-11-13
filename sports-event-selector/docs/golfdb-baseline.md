# GolfDB + Gemini baseline

The initial dataset is [GolfDB](https://github.com/wmcnally/golfdb), introduced by [McNally et al., CVPR Workshops 2019](https://arxiv.org/abs/1903.06528). It contains 1,400 golf-swing samples. Golf is a useful controlled starting sport, although the impact itself is fast. The repository provides a small demonstration clip, metadata and a separately hosted preprocessed video archive.

## Ready on this machine

- `data/golfdb/golfDB.mat`: official metadata, successfully parsed as 1,400 records.
- `data/golfdb/test_video.mp4`: official repository's small demonstration video.
- `data/golfdb/sample_manifest.json`: local sample source and SHA-256.
- `data/golfdb/source.json`: source URLs, checksums and provenance.
- `data/golfdb_sample/golfdb_demo/`: full sampled frames and source timestamp mapping.

The single upstream example is a **pipeline sample**, not a train/validation/test dataset. It has no invented labels and is not split into artificial independent examples. A genuine real-data training run is still pending Gemini credentials and the training clips.

## Annotate the sample with Gemini

Set the key in your local shell; do not send it in chat or commit it. The native integration uses Google's documented [multimodal generateContent request](https://ai.google.dev/gemini-api/docs/generate-content/image-understanding) and [structured JSON output](https://ai.google.dev/gemini-api/docs/generate-content/structured-output). It validates all returned event indices against the frames supplied.

```powershell
$env:GEMINI_API_KEY = 'YOUR_KEY'
$env:GEMINI_MODEL = 'gemini-3.8-flash'
./run.ps1 taxonomy --output data/golfdb/taxonomy.json
./run.ps1 annotate --clip data/golfdb_sample/golfdb_demo --taxonomy data/golfdb/taxonomy.json
```

`GEMINI_MODEL` can be changed to another available Gemini vision model. The default is based on Google's examples checked on 24 October 2026; account availability is not verified. The code uses the documented Generate Content endpoint, which Google's current documentation labels legacy, rather than depending on a provider SDK. A mock validates its request/response contract; a live Gemini response has not been tested without a key.

To recreate the sample on another computer:

```powershell
python -m sports_events golfdb-bootstrap --output data/golfdb
python -m sports_events prepare --manifest data/golfdb/sample_manifest.json --output data/golfdb_sample
```

## Build the full baseline dataset

Obtain and extract the author's [videos_160.zip archive](https://drive.google.com/file/d/1uBwRxFxW04EqG87VCoX3l6vXeV5T5JYJ/view) linked from the official repository. This setup downloaded only the small sample and metadata, not that archive. The archive supplies cropped, trimmed videos named `<id>.mp4`. Use the source repository's attribution/access terms for the downloaded material; its README identifies a noncommercial license for its code. No upstream model code is copied into this implementation.

```powershell
python -m pip install -e ".[dev,golfdb]"
python -m sports_events golfdb-manifest --metadata data/golfdb/golfDB.mat --videos data/golfdb/videos_160 --output data/golfdb/manifest.json
python -m sports_events prepare --manifest data/golfdb/manifest.json --output data/golfdb_training --annotate --taxonomy data/golfdb/taxonomy.json
# Review annotations, then:
python -m sports_events train --dataset data/golfdb_training --output runs/golfdb --epochs 20
python -m sports_events evaluate --dataset data/golfdb_training --checkpoint runs/golfdb/best.pt --output runs/golfdb/test_metrics.json
```

The adapter groups clips by original YouTube source and assigns a stable hash-based 80/10/10 train/validation/test allocation. It retains the official fold number as metadata but **does not claim the published GolfDB four-fold protocol**. This split prevents a single source video's multiple swings leaking across our training and validation sets. A tiny `--limit` subset may omit a split; use enough distinct source groups.

Official GolfDB event timestamps describe golf swing phases, not 2/3-frame commentary sequences. They are preserved as source metadata and are **not silently converted into Gemini ground truth**. The native annotation pipeline still asks Gemini to generate the requested sequences and descriptions. Its output can later be audited against the official phases after accounting for trim offsets.

## Transfer to sports broadcasts

GolfDB's preprocessed clips are 160×160 crops. They cannot evaluate the original full-broadcast framing requirement or scoreboard branch. The baseline preserves each input image's whole available view and uses `scoreboard_roi: null`. Configure actual channel scoreboards when the forthcoming broadcast reference is added, then test that branch separately. Neither this sample nor its cropped training set establishes cross-sport accuracy.
