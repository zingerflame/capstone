# Gemini Live Commentary

Real-time AI sports commentary generated from video, powered by Google's Gemini API and served through a Gradio UI.

## What's in this repo

| File | Description |
|---|---|
| `LiveMain.py` | Main app. Streams frames from an uploaded video (or webcam/stream source) and generates continuous, context-aware play-by-play commentary at a fixed interval. Suppresses near-duplicate commentary via similarity checking. |
| `LiveMainESPN.py` | `LiveMain.py` plus an **ESPN match data** mode: commentary from ESPN's live scores, stats and play-by-play instead of video frames, for nearly 70 leagues across many sports. See [README_ESPN.md](README_ESPN.md). |
| `Demomain.py` | Demo app. Uploads a full video clip to Gemini and streams back three commentary variants in parallel (225 words, 150 words, and a 75-word French version). |
| `test_live_api.py` | Standalone script to list available Gemini models and sanity-check basic streaming. |
| `espn_multisport/` | ESPN data for every sport (event lists, scores, stats, play-by-play). Used by `LiveMainESPN.py`. |
| `espn_matchinfo_api/` | Older soccer-only ESPN tools. `LiveMainESPN.py` uses them for soccer's extra stats. |

`LiveMain.py`, `LiveMainESPN.py` and `Demomain.py` each launch their own Gradio interface.

## Prerequisites

- Python 3.10+
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/apikey))

## Setup

1. Clone the repo and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set your Gemini API key. Get one from [Google AI Studio](https://aistudio.google.com/apikey).

   Copy `.env.example` to `.env` (next to the scripts) and fill in your key:

   ```
   GEMINI_API_KEY=your-key-here
   ```

   The scripts load this file at startup and refuse to start without the key. No
   quotes needed around the value. `.env` is gitignored — never commit a real key.

   A real environment variable still works and takes precedence if you prefer that:

   **macOS/Linux:**
   ```bash
   export GEMINI_API_KEY="your-key-here"
   ```

   **Windows (PowerShell):**
   ```powershell
   $env:GEMINI_API_KEY="your-key-here"
   ```

## Running

### Live commentary app

```bash
python LiveMain.py
```

Opens a Gradio UI. Upload a video clip, click **Start Commentary**, and watch commentary stream in alongside the video preview. Click **Stop Commentary** to end the session.

- Runs with `share=True`, so a public `gradio.live` link is also printed to the console (expires after 72 hours). Anyone with that link can use the app and consume your API quota — don't share it publicly unless you mean to.
- Logs are written per-run to `logs/run_<timestamp>.log`.

### Multi-sport ESPN commentary app

```bash
python LiveMainESPN.py
```

Choose **ESPN match data**, pick a sport, click **List events**, paste an event id (or leave it blank for the first live game), then click **Start Commentary**. Key features:

- **Works across sports:** soccer, NBA, NFL, MLB, NHL, rugby, golf, tennis, MMA, F1 and more, via a sport dropdown.
- **Accurate:** names, scores and stats come from ESPN, not guessed from video.
- **Efficient:** Gemini is only called when something new happens (a play, score, period or stat change).
- **Replays:** finished games replay play by play, with adjustable speed and batch size, and an option for key/scoring plays only.
- **Soccer extras:** passes, tackles and interceptions live, plus running stats during replays.
- **Original mode kept:** the frame-sampling mode from `LiveMain.py` is still available.

Full details, settings and limitations: [README_ESPN.md](README_ESPN.md).

### Multi-variant demo app

```bash
python Demomain.py
```

Upload a video clip and click **Start Analysis** to get three streamed commentary variants side by side.

### Model/API sanity check

```bash
python test_live_api.py
```

Lists available Gemini models and runs a basic streaming test against the configured model.

## Configuration

Key constants you may want to tune in `LiveMain.py`:

- `MODEL` — Gemini model used for commentary (default: `gemini-2.5-flash`)
- `COMMENTARY_INTERVAL` — seconds between commentary requests
- `SIMILARITY_THRESHOLD` — suppresses new commentary that's too similar to the last displayed line
- `DISPLAY_WIDTH` / `DISPLAY_HEIGHT` — resolution shown in the UI
- `GEMINI_WIDTH` / `GEMINI_HEIGHT` — resolution sent to Gemini (kept low for latency)

## Known issues

- `Demomain.py`'s `VideoDuration` function references `VideoFileClip`, which isn't imported anywhere (would need `moviepy`). It's currently unused by the app, so it doesn't block normal operation.
