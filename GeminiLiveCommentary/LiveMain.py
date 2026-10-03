import gradio as gr
import threading
import queue
import difflib
import cv2
import base64
import time
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from collections import deque
from google import genai
from google.genai import types

# Load GEMINI_API_KEY from the .env file sitting next to this script.
# Explicit path so it works no matter which directory you run from.
load_dotenv(Path(__file__).with_name(".env"))

if not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Add it to the .env file next to this script "
        "(GEMINI_API_KEY=your-key-here), or set it as an environment variable."
    )

client = genai.Client()

MODEL = "gemini-3.5-flash-lite"
COMMENTARY_INTERVAL = 5      # seconds between each commentary request
SIMILARITY_THRESHOLD = 0.85  # suppress output if new commentary is >= this similar to the last

DISPLAY_WIDTH, DISPLAY_HEIGHT = 854, 480   # 480p for smooth UI playback
GEMINI_WIDTH, GEMINI_HEIGHT = 640, 360     # 360p sent to Gemini for lower latency

# --- Logging setup ---
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/run_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()         # also print to console
    ]
)
log = logging.getLogger(__name__)
log.info(f"Log file created: {log_filename}")
log.info(f"Model: {MODEL} | Commentary interval: {COMMENTARY_INTERVAL}s")

# Shared state
is_running = False
is_thinking = False
latest_frame = None
frame_lock = threading.Lock()
commentary_buffer = []
commentary_lock = threading.Lock()
capture_thread = None
commentary_thread = None


def capture_frames(source):
    """Continuously capture frames from a video source and store the latest one.
    For video files, loops back to the start when the clip ends.
    For webcams/streams, runs until stopped.
    """
    global is_running, latest_frame
    log.info(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Could not open video source '{source}'")
        with commentary_lock:
            commentary_buffer.append(f"[Error: Could not open video source '{source}']")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_video_file = total_frames > 0

    if is_video_file:
        sleep_interval = 1 / fps if fps > 0 else 1 / 30
        log.info(f"Video file detected. FPS: {fps:.1f}, Total frames: {total_frames}, looping enabled")
    else:
        sleep_interval = 1 / 15
        log.info("Live source (webcam/stream). Capture rate: 15fps")

    frame_count = 0
    while is_running:
        ret, frame = cap.read()
        if not ret:
            if is_video_file:
                log.info("End of video clip reached — stopping all threads")
                is_running = False
            else:
                log.warning("Failed to read frame from video source")
            break
        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)
        with frame_lock:
            latest_frame = frame.copy()
        frame_count += 1
        if frame_count % max(1, int((fps or 15) * 10)) == 0:
            log.info(f"Frames captured so far: {frame_count}")
        time.sleep(sleep_interval)

    cap.release()
    log.info(f"Video capture stopped. Total frames captured: {frame_count}")


def get_frame_as_blob():
    """Encode the latest frame as a Gemini-compatible inline image blob."""
    with frame_lock:
        if latest_frame is None:
            log.debug("No frame available yet")
            return None
        frame = latest_frame.copy()

    frame = cv2.resize(frame, (GEMINI_WIDTH, GEMINI_HEIGHT), interpolation=cv2.INTER_LINEAR)
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    log.debug(f"Frame encoded, size: {len(buffer)} bytes")
    return types.Part(
        inline_data=types.Blob(
            data=base64.b64encode(buffer.tobytes()).decode(),
            mime_type="image/jpeg"
        )
    )


def run_commentary_loop():
    """
    Timer thread  : fires a Gemini request every COMMENTARY_INTERVAL seconds.
                    Each request runs in its own thread so slow responses never
                    block the next round.
    Writer thread : drains rounds strictly in order, streaming each chunk to
                    commentary_buffer only after the previous round is fully
                    written — guaranteeing no interleaving.
    """
    global is_running

    CONTEXT_WINDOW_SECONDS = 30
    max_context_entries = max(1, CONTEXT_WINDOW_SECONDS // COMMENTARY_INTERVAL)
    recent_commentaries = deque(maxlen=max_context_entries)
    recent_commentaries_lock = threading.Lock()

    # Each round puts its chunk queue here in fire order.
    # The writer reads this FIFO so output is always sequential.
    ordered_rounds = queue.Queue()

    log.info(f"Commentary loop started. Context window: {CONTEXT_WINDOW_SECONDS}s ({max_context_entries} entries max)")

    def call_gemini(round_num, frame_part, prompt, timestamp, chunk_queue):
        """Streams Gemini chunks into chunk_queue. Sentinel None signals completion."""
        fired_at = time.time()
        first_chunk_time = None
        try:
            response = client.models.generate_content_stream(
                model=MODEL,
                contents=[frame_part, prompt],
                config=types.GenerateContentConfig(
                    # Gemini 3.x replaced thinking_budget with thinking_level.
                    # MINIMAL is the floor — thinking cannot be fully disabled.
                    thinking_config=types.ThinkingConfig(thinking_level="MINIMAL")
                )
            )
            for chunk in response:
                if not is_running:
                    log.info(f"Round {round_num}: interrupted")
                    break
                if chunk.text:
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - fired_at
                        log.info(f"Round {round_num}: first token in {first_chunk_time:.2f}s")
                    chunk_queue.put(chunk.text)

            total_time = time.time() - fired_at
            log.info(f"Round {round_num}: streaming done in {total_time:.2f}s")

        except Exception as e:
            log.error(f"Round {round_num}: Gemini error: {e}", exc_info=True)
            chunk_queue.put(f"[Error: {e}]")
        finally:
            chunk_queue.put(None)  # sentinel — always sent so writer never hangs

    def writer_loop():
        """Reads ordered_rounds FIFO and writes each round's chunks in sequence.
        Collects the full commentary first, checks similarity against the last
        displayed round, then either streams it to the buffer or suppresses it.
        """
        global is_thinking
        last_displayed = ""

        while is_running or not ordered_rounds.empty():
            try:
                round_num, chunk_queue, timestamp = ordered_rounds.get(timeout=1)
            except queue.Empty:
                continue

            # Collect all chunks for this round first
            chunks = []
            while True:
                try:
                    chunk = chunk_queue.get(timeout=10)
                except queue.Empty:
                    log.warning(f"Round {round_num}: writer timed out waiting for chunks")
                    break
                if chunk is None:
                    break
                chunks.append(chunk)

            current_commentary = "".join(chunks).strip()
            if not current_commentary:
                is_thinking = False
                continue

            # Similarity check against the last displayed commentary
            similarity = difflib.SequenceMatcher(
                None, last_displayed.lower(), current_commentary.lower()
            ).ratio()

            if similarity >= SIMILARITY_THRESHOLD:
                log.info(f"Round {round_num}: suppressed (similarity {similarity:.2f} >= {SIMILARITY_THRESHOLD})")
                is_thinking = False
                # Still add to context so future rounds have accurate history
                with recent_commentaries_lock:
                    recent_commentaries.append(current_commentary)
                continue

            # New enough — stream it to the buffer word by word
            is_thinking = False
            with commentary_lock:
                commentary_buffer.append(f"\n\n[{timestamp}]\n")
            for chunk in chunks:
                with commentary_lock:
                    commentary_buffer.append(chunk)

            last_displayed = current_commentary
            with recent_commentaries_lock:
                recent_commentaries.append(current_commentary)
            log.info(f"Round {round_num}: displayed ({len(current_commentary)} chars, similarity {similarity:.2f})")

        log.info("Writer loop exited")

    # Start the writer thread once — it lives for the session
    writer_thread = threading.Thread(target=writer_loop, daemon=True)
    writer_thread.start()

    round_number = 0
    while is_running:
        time.sleep(COMMENTARY_INTERVAL)
        if not is_running:
            break

        round_number += 1
        log.info(f"--- Commentary round {round_number} ---")

        frame_part = get_frame_as_blob()
        if frame_part is None:
            log.warning(f"Round {round_number}: no frame yet, skipping")
            continue

        with recent_commentaries_lock:
            context_snapshot = list(recent_commentaries)

        context = (
            f"Previous commentary (for context only — do NOT repeat or quote this): "
            f"\"{' '.join(context_snapshot)}\"\n\n"
        ) if context_snapshot else ""

        prompt = (
            "You are a live sports play-by-play commentator on ESPN. "
            "Energetic, precise, immersive.\n\n"
            f"{context}"
            "Output ONLY your new commentary — 1-2 short sentences continuing from where the previous commentary left off. "
            "Do not repeat, quote, or paraphrase anything from the previous commentary. "
            "Never say 'in this frame' or 'the image shows'."
        )

        timestamp = time.strftime("%H:%M:%S")
        chunk_queue = queue.Queue()
        is_thinking = True

        # Register with writer BEFORE firing, so order is guaranteed
        ordered_rounds.put((round_number, chunk_queue, timestamp))

        threading.Thread(
            target=call_gemini,
            args=(round_number, frame_part, prompt, timestamp, chunk_queue),
            daemon=True
        ).start()

    log.info(f"Commentary loop ended after {round_number} rounds")


def start_live(video_file):
    global is_running, capture_thread, commentary_thread, commentary_buffer, latest_frame

    if is_running:
        log.warning("start_live called but already running")
        return "Already running."

    if video_file is None:
        return "Please upload a video file first."

    source = video_file
    log.info(f"Starting live commentary. Source: {source}")
    commentary_buffer = []
    latest_frame = None
    is_running = True

    capture_thread = threading.Thread(target=capture_frames, args=(source,), daemon=True)
    capture_thread.start()
    log.info("Capture thread started")

    commentary_thread = threading.Thread(target=run_commentary_loop, daemon=True)
    commentary_thread.start()
    log.info("Commentary thread started")

    return f"Live commentary started. First commentary in {COMMENTARY_INTERVAL}s..."


def stop_live():
    global is_running
    log.info("Stop requested by user")
    is_running = False
    log.info("is_running set to False — threads will exit on next cycle")
    return "Stopped."


def get_commentary():
    with commentary_lock:
        return "".join(commentary_buffer)


def get_thinking_status():
    if is_running and is_thinking:
        return "_thinking..._"
    return ""


def get_latest_frame():
    with frame_lock:
        if latest_frame is None:
            return None
        return cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)


log.info("Launching Gradio demo")

with gr.Blocks() as demo:
    gr.Markdown("## Sports Live Commentary")

    video_file = gr.File(
        label="Upload Video Clip",
        file_types=["video"],
        file_count="single",
    )

    with gr.Row():
        start_btn = gr.Button("Start Commentary", variant="primary")
        stop_btn = gr.Button("Stop Commentary", variant="stop")

    status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        webcam_feed = gr.Image(label="Live Feed", interactive=False)
        with gr.Column():
            commentary_out = gr.Textbox(
                label="Live Commentary", interactive=False, lines=15, autoscroll=True
            )
            thinking_status = gr.Markdown(value="")

    start_btn.click(fn=start_live, inputs=[video_file], outputs=status)
    stop_btn.click(fn=stop_live, inputs=None, outputs=status)

    frame_timer = gr.Timer(value=0.1)
    frame_timer.tick(fn=get_latest_frame, outputs=webcam_feed)

    commentary_timer = gr.Timer(value=0.1)
    commentary_timer.tick(fn=get_commentary, outputs=commentary_out)
    commentary_timer.tick(fn=get_thinking_status, outputs=thinking_status)

demo.launch(share=True)
log.info("Gradio demo exited")
