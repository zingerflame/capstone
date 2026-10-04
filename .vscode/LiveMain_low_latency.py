import gradio as gr
import threading
import cv2
import base64
import time
import logging
import os
from collections import deque
from google import genai
from google.genai import types

if not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Set it as an environment variable before running, "
        "e.g. `export GEMINI_API_KEY=your-key-here` (macOS/Linux) or "
        "`$env:GEMINI_API_KEY=\"your-key-here\"` (PowerShell)."
    )

client = genai.Client()

MODEL = "gemini-2.5-flash"
COMMENTARY_INTERVAL = 2      # lower-latency cadence (was 5s)
MAX_FRAME_AGE = 3.0          # drop commentary if its source frame is too old
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
latest_frame_timestamp = None
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
    global is_running, latest_frame, latest_frame_timestamp
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
            latest_frame_timestamp = time.monotonic()
        frame_count += 1
        if frame_count % max(1, int((fps or 15) * 10)) == 0:
            log.info(f"Frames captured so far: {frame_count}")
        time.sleep(sleep_interval)

    cap.release()
    log.info(f"Video capture stopped. Total frames captured: {frame_count}")


def get_frame_as_blob():
    """Encode the latest frame and return it together with its capture timestamp."""
    with frame_lock:
        if latest_frame is None:
            log.debug("No frame available yet")
            return None, None
        frame = latest_frame.copy()
        frame_timestamp = latest_frame_timestamp

    frame = cv2.resize(frame, (GEMINI_WIDTH, GEMINI_HEIGHT), interpolation=cv2.INTER_LINEAR)
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    log.debug(f"Frame encoded, size: {len(buffer)} bytes")
    return types.Part(
        inline_data=types.Blob(
            data=base64.b64encode(buffer.tobytes()).decode(),
            mime_type="image/jpeg"
        )
    ), frame_timestamp


def run_commentary_loop():
    """Low-latency commentary loop.

    Differences from the original implementation:
      * fires the first request as soon as a frame exists (no initial 5s wait)
      * uses a 2s request cadence
      * publishes complete sentences/phrases while Gemini is still streaming
      * timestamps source frames and drops stale output
      * prevents a late older request from being displayed after a newer one
    """
    global is_running, is_thinking

    CONTEXT_WINDOW_SECONDS = 30
    max_context_entries = max(1, CONTEXT_WINDOW_SECONDS // COMMENTARY_INTERVAL)
    recent_commentaries = deque(maxlen=max_context_entries)
    recent_commentaries_lock = threading.Lock()

    state_lock = threading.Lock()
    latest_accepted_round = 0
    active_requests = 0

    def set_request_active(delta):
        nonlocal active_requests
        global is_thinking
        with state_lock:
            active_requests += delta
            is_thinking = active_requests > 0

    def publish_phrase(round_num, phrase, display_timestamp, frame_timestamp):
        """Publish only fresh output that has not been superseded by a newer round."""
        nonlocal latest_accepted_round
        phrase = phrase.strip()
        if not phrase:
            return False

        frame_age = time.monotonic() - frame_timestamp if frame_timestamp else float("inf")
        if frame_age > MAX_FRAME_AGE:
            log.info(
                f"Round {round_num}: dropped stale phrase "
                f"(source frame age {frame_age:.2f}s > {MAX_FRAME_AGE:.2f}s)"
            )
            return False

        with state_lock:
            if round_num < latest_accepted_round:
                log.info(
                    f"Round {round_num}: dropped because newer round "
                    f"{latest_accepted_round} has already been displayed"
                )
                return False
            first_output_for_round = round_num > latest_accepted_round
            if first_output_for_round:
                latest_accepted_round = round_num

        with commentary_lock:
            if first_output_for_round:
                commentary_buffer.append(f"\n\n[{display_timestamp}]\n")
            commentary_buffer.append(phrase + " ")
        return True

    def call_gemini(round_num, frame_part, frame_timestamp, prompt, display_timestamp):
        """Stream Gemini output and publish complete phrases as soon as possible."""
        fired_at = time.monotonic()
        first_chunk_time = None
        full_response = ""
        phrase_buffer = ""
        displayed_anything = False
        set_request_active(+1)

        try:
            response = client.models.generate_content_stream(
                model=MODEL,
                contents=[frame_part, prompt],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )

            for chunk in response:
                if not is_running:
                    break
                if not chunk.text:
                    continue

                if first_chunk_time is None:
                    first_chunk_time = time.monotonic() - fired_at
                    log.info(f"Round {round_num}: first token in {first_chunk_time:.2f}s")

                text = chunk.text
                full_response += text
                phrase_buffer += text

                # Flush on sentence punctuation. This is safer for eventual TTS than
                # speaking/displaying individual model tokens.
                while True:
                    punctuation_positions = [
                        phrase_buffer.find(mark) for mark in (".", "!", "?")
                        if phrase_buffer.find(mark) != -1
                    ]
                    if not punctuation_positions:
                        break
                    cut = min(punctuation_positions) + 1
                    phrase = phrase_buffer[:cut]
                    phrase_buffer = phrase_buffer[cut:]
                    if publish_phrase(round_num, phrase, display_timestamp, frame_timestamp):
                        displayed_anything = True

            # Flush a short trailing phrase if Gemini ended without punctuation.
            if phrase_buffer.strip():
                if publish_phrase(round_num, phrase_buffer, display_timestamp, frame_timestamp):
                    displayed_anything = True

            total_time = time.monotonic() - fired_at
            log.info(f"Round {round_num}: streaming done in {total_time:.2f}s")

            if full_response.strip():
                with recent_commentaries_lock:
                    recent_commentaries.append(full_response.strip())

            if displayed_anything:
                log.info(f"Round {round_num}: streamed to UI while generation was in progress")

        except Exception as e:
            log.error(f"Round {round_num}: Gemini error: {e}", exc_info=True)
        finally:
            set_request_active(-1)

    log.info(
        f"Low-latency commentary loop started. Interval={COMMENTARY_INTERVAL}s, "
        f"max frame age={MAX_FRAME_AGE}s"
    )

    round_number = 0
    next_fire = time.monotonic()

    while is_running:
        now = time.monotonic()
        if now < next_fire:
            time.sleep(min(0.05, next_fire - now))
            continue

        frame_part, frame_timestamp = get_frame_as_blob()
        if frame_part is None:
            # Capture thread may need a moment to produce its first frame.
            time.sleep(0.05)
            continue

        round_number += 1
        log.info(f"--- Commentary round {round_number} ---")

        with recent_commentaries_lock:
            context_snapshot = list(recent_commentaries)

        context = (
            "Previous commentary (context only; do not repeat it): "
            f"\"{' '.join(context_snapshot)}\"\n\n"
        ) if context_snapshot else ""

        prompt = (
            "You provide low-latency live sports commentary for a blind or low-vision viewer. "
            "Prioritize the most important NEW action over background description. "
            "Be precise and concise.\n\n"
            f"{context}"
            "Output ONLY one short sentence of new play-by-play commentary. "
            "Do not repeat or paraphrase previous commentary. "
            "Never say 'in this frame' or 'the image shows'."
        )

        display_timestamp = time.strftime("%H:%M:%S")
        threading.Thread(
            target=call_gemini,
            args=(round_number, frame_part, frame_timestamp, prompt, display_timestamp),
            daemon=True
        ).start()

        # Schedule relative to this fire time so model response time does not delay
        # when the next observation is made.
        next_fire = time.monotonic() + COMMENTARY_INTERVAL

    log.info(f"Commentary loop ended after {round_number} rounds")


def start_live(video_file):
    global is_running, capture_thread, commentary_thread, commentary_buffer, latest_frame, latest_frame_timestamp

    if is_running:
        log.warning("start_live called but already running")
        return "Already running."

    if video_file is None:
        return "Please upload a video file first."

    source = video_file
    log.info(f"Starting live commentary. Source: {source}")
    commentary_buffer = []
    latest_frame = None
    latest_frame_timestamp = None
    is_running = True

    capture_thread = threading.Thread(target=capture_frames, args=(source,), daemon=True)
    capture_thread.start()
    log.info("Capture thread started")

    commentary_thread = threading.Thread(target=run_commentary_loop, daemon=True)
    commentary_thread.start()
    log.info("Commentary thread started")

    return "Live commentary started. First commentary will be requested as soon as the first frame is available."


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
