# Gemini Live Commentary

Real-time AI sports commentary generated from video, powered by Google's Gemini API and served through a Gradio UI.

## What's in this repo

| File | Description |
|---|---|
| `LiveMain.py` | Main app. Streams frames from an uploaded video (or webcam/stream source) and generates continuous, context-aware play-by-play commentary at a fixed interval. Suppresses near-duplicate commentary via similarity checking. |
| `Demomain.py` | Demo app. Uploads a full video clip to Gemini and streams back three commentary variants in parallel (225 words, 150 words, and a 75-word French version). |
| `test_live_api.py` | Standalone script to list available Gemini models and sanity-check basic streaming. |

Both `LiveMain.py` and `Demomain.py` launch their own Gradio interface.

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
