"""Live commentary with two modes:

  Frame sampling (original) - identical to LiveMain.py: every COMMENTARY_INTERVAL
                              seconds, the latest video frame is sent to Gemini.
  ESPN match data           - no frames needed. Polls ESPN's multi-sport feed
                              (espn_multisport) and calls Gemini with a text prompt only
                              when something new happens. Works for every sport in
                              espn_multisport.SPORTS:
                                team sports (soccer, NBA, NFL, MLB, NHL, ...): new plays,
                                  score and stat changes; finished games are replayed
                                  play by play in small batches.
                                soccer also uses the older soccer-only tracker
                                  (espn_matchinfo_api/minute_tracker) for passes, tackles
                                  and interceptions live, and running stats in replays.
                                golf, tennis, MMA, racing: no play-by-play exists, so
                                  Gemini comments on what changed on the leaderboard /
                                  scoreboard / results since the last poll.
"""
import gradio as gr
import threading
import queue
import difflib
import cv2
import base64
import re
import sys
import time
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from dateutil import parser as dateparser
from dotenv import load_dotenv
from collections import deque
from google import genai
from google.genai import types

# Both ESPN helpers live in subfolders next to this script; the folders go on
# sys.path and the modules are imported by name (minute_tracker itself does
# `from espn_feed import ...`).
#   espn_multisport              - every sport: event lists, snapshots, play-by-play
#   espn_matchinfo_api/minute_tracker - soccer extras: the fuller live stat set
#                                  (passes, tackles, ...) and running stats rebuilt
#                                  from commentary for replays
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here / "espn_matchinfo_api"))
sys.path.insert(0, str(_here / "espn_multisport"))
import espn_multisport as espn  # noqa: E402
import minute_tracker  # noqa: E402

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

# --- ESPN mode settings ---
MODE_FRAMES = "Frame sampling (original)"
MODE_ESPN = "ESPN match data"
ESPN_POLL_INTERVAL = 15      # seconds between ESPN polls for a live game
REPLAY_SECONDS_PER_UPDATE = 4  # finished game: real seconds per replay update
REPLAY_PLAYS_PER_UPDATE = 4  # finished game: max plays per Gemini call (a score always ends an update)
ESPN_CONTEXT_ENTRIES = 4     # previous commentaries kept for continuity
ESPN_FIRST_POLL_PLAYS = 5    # joining a live game: comment on just the latest few plays
DEFAULT_SPORT = "soccer"     # key from espn_multisport.SPORTS

# --- Logging setup ---
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/run_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),  # player names like "Tomić"
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


def capture_frames(source, loop_video=False):
    """Continuously capture frames from a video source and store the latest one.
    For video files, stops everything when the clip ends (or rewinds if loop_video,
    used in ESPN mode where the video is display-only).
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
            if is_video_file and loop_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
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


# --- ESPN mode ---

# How an ESPN sport path reads in a prompt ("a live ___ commentator")
SPORT_NOUNS = {
    "football": "American football", "hockey": "ice hockey", "rugby-league": "rugby league",
    "australian-football": "Australian rules football", "mma": "MMA", "racing": "motor racing",
}
# Sports where the away team is conventionally written first ("BOS 98 @ LAL 101")
AWAY_FIRST = {"basketball", "football", "baseball", "hockey", "lacrosse"}
# Sports that score so often that ending a replay update at every score would
# make nearly every update a single basket (an NBA game has ~100 scoring plays)
HIGH_SCORING = {"basketball", "australian-football"}

# Soccer: minute_tracker's stat names -> labels shown to Gemini
SOCCER_STAT_LABELS = {
    "possession_pct": "Possession %", "shots": "Shots", "shots_on_target": "On target",
    "corners": "Corners", "fouls": "Fouls", "offsides": "Offsides", "yellow_cards": "Yellow cards",
    "red_cards": "Red cards", "saves": "Saves", "passes": "Passes",
    "accurate_passes": "Accurate passes", "tackles": "Tackles", "interceptions": "Interceptions",
}
# Stats that move on almost every poll; shown in TEAM STATS, but not listed as "changes"
CONSTANTLY_CHANGING = ("%", "pass", "possession")

FRAME_NOTE = (
    "The attached image is the current broadcast frame. The facts above are the truth; "
    "the image may show a later moment, so use it only for atmosphere (crowd, weather, "
    "celebrations, players' reactions) and never to contradict or add to the facts. "
    "If the image doesn't fit, ignore it. Never say 'the image shows'.\n"
)


def espn_call(fn, *args, **kwargs):
    """espn_multisport doubles as a CLI and reports bad input (unknown sport,
    event not found) with SystemExit, which `except Exception` would miss."""
    try:
        return fn(*args, **kwargs)
    except SystemExit as e:
        raise RuntimeError(str(e)) from None


def _int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def sport_noun(snap):
    return SPORT_NOUNS.get(snap["sport"], snap["sport"].replace("-", " "))


def play_key(p):
    """Identity of a play across polls (a few soccer lines have no id)."""
    return p["id"] or (p["period"], p["clock"], p["text"])


def team_sides(snap):
    """{team display name: 0 for home, 1 for away}, to credit goals in play_score."""
    return {snap["teams"]["home"]["name"]: 0, snap["teams"]["away"]["name"]: 1}


def play_score(p, score, sides=None):
    """(home, away) after this play; unchanged if the play doesn't say.
    sides (team_sides()) lets a bare "Goal" play by a team count, for soccer
    leagues where ESPN only has key events ("Mourad Batna (Al Fateh) Goal at 70'")."""
    h, a = _int(p["home_score"]), _int(p["away_score"])
    if h is not None and a is not None:
        return h, a
    # Soccer plays carry no running score; it's in the commentary text (home first):
    # "Goal! Liverpool 0, Arsenal 1." / "VAR Decision: No Goal Liverpool 0-1 Arsenal."
    m = minute_tracker.GOAL_SCORE_RE.search(p["text"].strip())
    if m:
        return int(m.group(2)), int(m.group(4))
    m = minute_tracker.VAR_SCORE_RE.search(p["text"].strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    # Own goals are skipped: which team ESPN lists for them isn't consistent
    kind = (p.get("type") or "").lower()
    side = (sides or {}).get(p.get("team"))
    if side is not None and ((kind.startswith("goal") and not any(w in kind for w in ("kick", "disallowed")))
                             or kind == "penalty - scored"):
        new = list(score)
        new[side] += 1
        return tuple(new)
    return score


def when_text(sport, p):
    """Where in the game a play happened: "35'", "3rd Quarter 5:12", "7th Inning", "Q2 4:10"."""
    if sport == "soccer" and p["clock"]:
        return p["clock"]
    period = p["period_label"]
    n = p["period"]
    if not period and n:
        if sport == "soccer":
            period = {1: "1st half", 2: "2nd half"}.get(n, "extra time")
        elif sport == "football":
            period = f"Q{n}" if n <= 4 else "OT"
        else:
            period = f"Period {n}"
    return f"{period or ''} {p['clock']}".strip()


def score_text(snap, score):
    h, a = snap["teams"]["home"]["abbr"], snap["teams"]["away"]["abbr"]
    hs, as_ = score
    if snap["sport"] in AWAY_FIRST:
        return f"{a} {as_} @ {h} {hs}"
    return f"{h} {hs}-{as_} {a}"


def live_score(snap):
    t = snap["teams"]
    return _int(t["home"]["score"]) or 0, _int(t["away"]["score"]) or 0


def headline_stats(snap):
    """{label: {team abbr: value}} for the sport's headline team stats.
    Soccer uses minute_tracker's fuller set (adds passes, tackles, interceptions, offsides)."""
    if snap["sport"] == "soccer":
        wanted = {espn_name: SOCCER_STAT_LABELS.get(ours, ours)
                  for ours, espn_name in minute_tracker.BOXSCORE_STATS.items() if espn_name}
    else:
        wanted = espn.CONTEXT_STATS.get(snap["sport"], {})
    stats = snap["team_stats"]
    out = {label: {team: vals.get(name) for team, vals in stats.items()}
           for name, label in wanted.items() if any(name in v for v in stats.values())}
    h, a = snap["teams"]["home"], snap["teams"]["away"]
    if "hits" in h:  # baseball R-H-E line
        out["Hits"] = {a["abbr"]: a.get("hits"), h["abbr"]: h.get("hits")}
        out["Errors"] = {a["abbr"]: a.get("errors"), h["abbr"]: h.get("errors")}
    return out


def stat_changes(old, new):
    return [f"{label} {team}: {old.get(label, {}).get(team)} -> {v}"
            for label, vals in new.items()
            if not any(word in label.lower() for word in CONSTANTLY_CHANGING)
            for team, v in vals.items()
            if v is not None and old.get(label, {}).get(team) not in (None, v)]


def soccer_key_event_stats(snap):
    """Running card and substitution counts from ESPN's key events, for leagues
    without full commentary. Key events only hold goals, cards and subs, so
    shots, corners and fouls can't be rebuilt from them. Same shape as
    soccer_replay_stats(); {} if there are no key events."""
    abbr = {snap["teams"][s]["name"]: snap["teams"][s]["abbr"] for s in ("home", "away")}
    totals = {a: {"Yellow cards": 0, "Red cards": 0, "Substitutions": 0} for a in abbr.values()}
    out = {}
    for p in snap["key_events"]:
        kind, team = (p["type"] or "").lower(), abbr.get(p["team"])
        stat = ("Red cards" if "red card" in kind or "second yellow" in kind
                else "Yellow cards" if "yellow card" in kind
                else "Substitutions" if "substitution" in kind else None)
        if stat and team:
            totals[team][stat] += 1
        label = minute_tracker._minute_label(p["period"], p["clock"]) if p["period"] else None
        if label:
            out[(p["period"], label)] = {s: {t: totals[t][s] for t in totals} for s in totals[next(iter(totals))]}
    return out


def soccer_replay_stats(raw, snap=None):
    """Finished soccer game: running stats after every minute, rebuilt from ESPN's
    commentary by minute_tracker (shots, on target, corners, fouls, offsides, cards).
    Leagues with key events only fall back to soccer_key_event_stats(snap).
    Returns {(period, minute label): {label: {team abbr: value}}}, or {} if it can't."""
    try:
        minutes = minute_tracker.rebuild_minutes(raw, minute_tracker.match_info(raw))
    except Exception as e:
        log.warning(f"Soccer running stats unavailable: {e}")
        return soccer_key_event_stats(snap) if snap else {}
    out = {}
    for m in minutes:
        teams = m["stats"]
        names = [n for n in next(iter(teams.values()), {}) if n != "goals"]
        out[(m["period"], m["minute"])] = {
            SOCCER_STAT_LABELS.get(n, n): {t: s.get(n) for t, s in teams.items()} for n in names
        }
    # Leagues where ESPN only has key events ("X Yellow Card at 12'") don't use the
    # commentary phrasing minute_tracker counts, so every total stays 0: count the
    # key events instead (cards and subs only)
    if not any(v for stats in out.values() for vals in stats.values() for v in vals.values()):
        log.info("Soccer running stats: no full commentary for this match, counting key events instead")
        return soccer_key_event_stats(snap) if snap else {}
    return out


def soccer_stats_at(running, plays, current):
    """Running stats as of the last play in `plays` that has a match minute."""
    for p in reversed(plays):
        label = minute_tracker._minute_label(p["period"], p["clock"]) if p["period"] else None
        if (p["period"], label) in running:
            return running[(p["period"], label)]
    return current


def replay_plays(snap, key_only):
    """Plays to replay for a finished team game (empty if ESPN has no play-by-play)."""
    plays = snap["key_events"] if key_only else snap["plays"]
    return [p for p in plays if p["text"].strip()]


def group_plays(plays, per_update, split_on_score=True, sides=None):
    """Split a finished game's plays into replay updates of up to per_update plays.
    An update never spans two periods and, with split_on_score, ends early after
    a score (off for key-plays-only replays, where most plays are scores).
    Returns [(plays, (home, away) score after them)]."""
    updates, batch, score = [], [], (0, 0)
    for p in plays:
        if batch and p["period"] and batch[-1]["period"] and p["period"] != batch[-1]["period"]:
            updates.append((batch, score))
            batch = []
        new_score = play_score(p, score, sides)
        batch.append(p)
        scored, score = new_score != score, new_score
        if (scored and split_on_score) or len(batch) >= per_update:
            updates.append((batch, score))
            batch = []
    if batch:
        updates.append((batch, score))
    return updates


def finish_prompt(snap, body, task, recent_commentaries, with_frame, role="play-by-play commentator"):
    context = (
        f"Your previous commentary (for continuity only — do NOT repeat it): "
        f"\"{' '.join(recent_commentaries)}\"\n\n"
    ) if recent_commentaries else ""
    return (
        f"You are a live {sport_noun(snap)} {role} on ESPN. Energetic, precise, immersive.\n\n"
        + "\n".join(body) + "\n\n"
        + context
        + task + " Use only the facts above; never invent players, events or scores.\n"
        + (FRAME_NOTE if with_frame else "")
    )


def build_team_prompt(snap, plays, score, recent_commentaries, with_frame, live, changes=(),
                      stats=None):
    """Team sports: score, headline stats, and the new plays.
    Live: stats come from the snapshot's box score. Replay: the snapshot's stats
    are the final ones, so only `stats` passed in (soccer's rebuilt running
    stats) are shown."""
    sport = snap["sport"]
    h, a = snap["teams"]["home"], snap["teams"]["away"]
    st, g = snap["status"], snap["game"]
    where = (st["detail"] or st["description"]) if live else (when_text(sport, plays[-1]) if plays else "")
    body = [
        f"GAME: {a['name']} ({a['abbr']}) at {h['name']} ({h['abbr']})",
        f"SCORE: {score_text(snap, score)}" + "".join(f" | {x}" for x in (where, g.get("league")) if x),
    ]
    if live:
        stats = headline_stats(snap)
    if stats:
        body.append("TEAM STATS:" if live else "TEAM STATS (so far):")
        body += [f"  {label}: " + " | ".join(f"{t} {'-' if v is None else v}" for t, v in vals.items())
                 for label, vals in stats.items()]
    if plays:
        body.append("NEW PLAYS:")
        for p in plays:
            when = when_text(sport, p)
            body.append(f"  - {'[' + when + '] ' if when else ''}{p['text'].strip()}")
    if changes:
        body.append("ALSO CHANGED SINCE YOUR LAST UPDATE:")
        body += [f"  - {c}" for c in changes]

    if plays:
        task = ("Output ONLY your commentary on the NEW PLAYS — 1-3 short sentences. "
                "If anyone scored, call it and give the new score.")
    elif changes:
        task = "Output ONLY your commentary on what CHANGED — 1-2 short sentences."
    else:
        task = "You're joining this game now: set the scene in 1-2 short sentences."
    return finish_prompt(snap, body, task, recent_commentaries, with_frame)


def build_summary_prompt(snap, changes, recent_commentaries, with_frame):
    """Golf/tennis/MMA/racing (and team games without play-by-play): ESPN's
    text summary of the event, plus what changed since the last poll.
    changes=None means this is the first look at the event."""
    body = espn.build_context(snap, recent=8).splitlines()
    if changes:
        body += ["", "CHANGED SINCE YOUR LAST UPDATE:"] + [f"  - {c}" for c in changes]
        task = "Output ONLY your commentary on what CHANGED — 1-3 short sentences."
    elif snap["status"]["state"] == "post":
        task = "This event is over: sum up how it finished in 2-3 short sentences."
    else:
        task = "Give a quick overview of where things stand right now — 2-3 short sentences."
    return finish_prompt(snap, body, task, recent_commentaries, with_frame, role="commentator")


def status_header(snap):
    st = snap["status"]
    return f"{time.strftime('%H:%M:%S')}  {st['detail'] or st['description'] or ''}".rstrip()


def stream_espn_commentary(prompt, header, frame_part=None, stop_on_stop=True):
    """Gemini call (text, plus an optional frame), streamed straight into the commentary buffer.
    stop_on_stop=False for one-off calls (commentary at a timestamp), which run
    while no session is active."""
    global is_thinking
    is_thinking = True
    fired_at = time.time()
    first_chunk_time = None
    parts = []
    try:
        response = client.models.generate_content_stream(
            model=MODEL,
            contents=[frame_part, prompt] if frame_part else prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL")
            )
        )
        for chunk in response:
            if stop_on_stop and not is_running:
                log.info("ESPN round interrupted")
                break
            if chunk.text:
                with commentary_lock:
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - fired_at
                        log.info(f"ESPN round: first token in {first_chunk_time:.2f}s")
                        commentary_buffer.append(f"\n\n[{header}]\n")
                    commentary_buffer.append(chunk.text)
                parts.append(chunk.text)
        log.info(f"ESPN round: streaming done in {time.time() - fired_at:.2f}s")
    except Exception as e:
        log.error(f"ESPN round: Gemini error: {e}", exc_info=True)
        with commentary_lock:
            commentary_buffer.append(f"\n\n[Error: {e}]")
    finally:
        is_thinking = False
    return "".join(parts).strip()


def team_update(snap, state):
    """What's new in a live team game since the last poll.
    Returns (prompt builder, header), or None if nothing worth a Gemini call.
    state is a dict this function keeps between polls (empty on the first one)."""
    st = snap["status"]
    plays = [p for p in snap["plays"] if p["text"].strip()]
    keys = [play_key(p) for p in plays]
    first = "seen" not in state
    if first:
        new = plays[-ESPN_FIRST_POLL_PLAYS:]  # joining mid-game: just the latest plays
    else:
        new = [p for p, k in zip(plays, keys) if k not in state["seen"]]
    score, stats = live_score(snap), headline_stats(snap)

    changes = []
    if not first:
        if score != state["score"]:
            changes.append(f"Score: {score_text(snap, state['score'])} -> {score_text(snap, score)}")
        if st["period"] != state["period"]:
            changes.append(f"Game status: {st['detail'] or st['description']}")
    # A stat change (a shot, corner, card, ...) also triggers a call, as in the
    # old soccer-only version; stats that move on every poll (possession %,
    # passes) are left out by stat_changes so they don't.
    stat_moves = [] if first else stat_changes(state["stats"], stats)
    state.update(seen=set(keys), score=score, period=st["period"], stats=stats)

    if not (first or new or changes or stat_moves):
        return None
    header = f"{st['detail'] or st['description'] or ''}  {score_text(snap, score)}".strip()
    return (lambda recent, with_frame: build_team_prompt(
        snap, new, score, recent, with_frame, live=True, changes=changes + stat_moves)), header


def event_update(snap, state):
    """Golf/tennis/MMA/racing: compare ESPN's text summary with the last poll's.
    Returns (prompt builder, header), or None if nothing changed."""
    lines = [ln for ln in espn.build_context(snap, recent=8).splitlines() if ln.strip()]
    changes = None if "lines" not in state else [ln.strip() for ln in lines if ln not in state["lines"]]
    state["lines"] = set(lines)
    if changes == []:
        return None
    return (lambda recent, with_frame: build_summary_prompt(snap, changes, recent, with_frame)), status_header(snap)


def run_espn_loop(sport_key, snap, raw, poll_interval, replay_seconds, plays_per_update, key_only,
                  attach_frame=False):
    """Live game/event: poll ESPN, comment only when something new happened.
    Finished team game: replay its plays, plays_per_update at a time, one
    update every replay_seconds. Finished event without plays: one recap.
    raw: ESPN's untouched response (soccer replays rebuild running stats from it).
    attach_frame: also send the latest video frame with each call."""
    global is_running
    recent_commentaries = deque(maxlen=ESPN_CONTEXT_ENTRIES)

    def comment(make_prompt, header):
        """make_prompt(recent_commentaries, with_frame) -> prompt text."""
        frame_part = get_frame_as_blob() if attach_frame else None
        if attach_frame and frame_part is None:
            log.warning("ESPN round: no frame available yet, sending text only")
        prompt = make_prompt(list(recent_commentaries), frame_part is not None)
        text = stream_espn_commentary(prompt, header, frame_part)
        if text:
            recent_commentaries.append(text)

    try:
        if snap["status"]["state"] == "post":
            plays = replay_plays(snap, key_only) if snap["kind"] == "team" else []
            if plays:
                # Key-plays replays are mostly scores already, so don't split on them there
                split = not key_only and snap["sport"] not in HIGH_SCORING
                running = soccer_replay_stats(raw, snap) if snap["sport"] == "soccer" else {}
                replay_team_game(snap, plays, plays_per_update, replay_seconds, comment,
                                 split_on_score=split, running_stats=running)
            else:
                log.info("ESPN: finished event without play-by-play — one recap")
                comment(lambda r, f: build_summary_prompt(snap, None, r, f), status_header(snap))
        else:
            follow_live(sport_key, snap, poll_interval, comment)
    except Exception as e:
        log.error(f"ESPN loop failed: {e}", exc_info=True)
        with commentary_lock:
            commentary_buffer.append(f"\n\n[Error: {e}]")
    finally:
        is_running = False


def replay_team_game(snap, plays, plays_per_update, replay_seconds, comment, split_on_score=True,
                     running_stats=None):
    """running_stats: soccer_replay_stats() output, or empty/None for no stats."""
    updates = group_plays(plays, plays_per_update, split_on_score, team_sides(snap))
    log.info(f"ESPN replay: {len(plays)} plays in {len(updates)} updates at {replay_seconds}s each"
             f"{' with running stats' if running_stats else ''}")
    stats = None
    for i, (batch, score) in enumerate(updates):
        if not is_running:
            break
        started = time.time()
        if i == len(updates) - 1:
            score = live_score(snap)  # end of the game: ESPN's final score is authoritative
        if running_stats:
            stats = soccer_stats_at(running_stats, batch, stats)
        header = f"{when_text(snap['sport'], batch[-1])}  {score_text(snap, score)}".strip()
        comment(lambda r, f: build_team_prompt(snap, batch, score, r, f, live=False, stats=stats), header)
        time.sleep(max(0, replay_seconds - (time.time() - started)))
    log.info("ESPN replay finished")


def follow_live(sport_key, snap, poll_interval, comment):
    event_id = snap["event_id"]
    update = team_update if snap["kind"] == "team" else event_update
    state = {}
    polls = calls = 0
    while is_running:
        status = snap["status"]
        if status["state"] == "pre":
            log.info(f"ESPN: not started yet ({status['detail'] or status['description']}), waiting")
        else:
            found = update(snap, state)
            if found:
                calls += 1
                comment(*found)
            else:
                log.info(f"ESPN poll {polls}: nothing new ({status['detail']}) — skipping Gemini")
            if status["state"] == "post":
                with commentary_lock:
                    commentary_buffer.append("\n\n[Final]")
                break

        time.sleep(poll_interval)
        if not is_running:
            break
        try:
            fetched_at = time.time()
            snap = espn_call(espn.get_snapshot, sport_key, event_id)
            polls += 1
            log.info(f"ESPN poll {polls}: fetched in {time.time() - fetched_at:.2f}s "
                     f"({snap['status']['detail']})")
        except Exception as e:
            log.warning(f"ESPN fetch failed: {e}")

    log.info(f"ESPN loop ended after {polls} polls, {calls} Gemini calls")


def list_matches(sport_key, date):
    """Games/events for a sport on a date (default today), live first."""
    sport_key = (sport_key or "").strip()
    try:
        rows = espn_call(espn.list_events, sport_key, date_arg(date))
    except Exception as e:
        return f"Could not get ESPN events: {e}"
    if not rows:
        return "No events found."
    rows.sort(key=lambda r: {"in": 0, "pre": 1, "post": 2}.get(r["state"], 3))
    lines = []
    for r in rows:
        if "home" in r:
            what = f"{r['away']} {r['away_score']} at {r['home']} {r['home_score']}"
        else:
            what = r["name"] + (f"  (leader: {r['leader']})" if r.get("leader") else "")
        lines.append(f"{r['event_id']}  [{r['status']}]  {what}")
    return "\n".join(lines)


# --- Find a game by name and date ---

# "Warriors vs Rockets", "Lakers @ Warriors", "Arsenal v Liverpool", "Celtics, Heat"
TEAM_SPLIT_RE = re.compile(r"\s+(?:vs\.?|v\.?|versus|at|@|-)\s+|\s*,\s*", re.I)


def date_arg(date):
    """Date picker value ('2025-01-25', a timestamp or None) -> 'YYYYMMDD', or None for today."""
    if date in (None, ""):
        return None
    if isinstance(date, (int, float)):
        return datetime.fromtimestamp(date).strftime("%Y%m%d")
    text = str(date).strip()
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 8 and re.match(r"\d{4}\D?\d{2}\D?\d{2}", text):
        return digits[:8]
    return dateparser.parse(text).strftime("%Y%m%d")


def _local_start(iso):
    """ESPN start time (UTC) -> 'Sat Jan 25, 08:30 PM' in this computer's time zone."""
    try:
        return dateparser.isoparse(iso).astimezone().strftime("%a %b %d, %I:%M %p")
    except (TypeError, ValueError):
        return iso or "?"


def _side_names(ev, kind):
    """Every name each side goes by, lowercased: {'golden state warriors', 'warriors', 'gs', ...}.
    Golf/tennis/MMA/racing: the event's own name is the only 'side'."""
    if kind != "team":
        return [{(ev.get("name") or "").lower(), (ev.get("shortName") or "").lower()} - {""}]
    sides = []
    for c in (ev.get("competitions") or [{}])[0].get("competitors", []):
        t = c.get("team") or {}
        sides.append({str(t[k]).lower() for k in ("displayName", "shortDisplayName", "name",
                                                  "location", "abbreviation", "nickname") if t.get(k)})
    return sides


def _part_matches(part, names):
    """Does one query part ('warriors', 'gsw', 'warriers') name this side?"""
    for n in names:
        # Whole words or word starts: "warr" -> Warriors, but "nets" must not hit "Hornets"
        if part == n or (len(part) >= 3 and re.search(r"\b" + re.escape(part), n)):
            return True
        # Abbreviations vary: ESPN says "GS" / "NY" where people type "GSW" / "NYK"
        if 2 <= min(len(part), len(n)) and max(len(part), len(n)) <= 4 and abs(len(part) - len(n)) <= 1 \
                and " " not in part + n and (part.startswith(n) or n.startswith(part)):
            return True
        if len(part) >= 4 and any(difflib.SequenceMatcher(None, part, w).ratio() >= 0.85
                                  for w in [n, *n.split()]):
            return True  # small typos
    return False


def _event_matches(ev, kind, parts):
    sides = _side_names(ev, kind)
    if kind != "team":
        return all(any(_part_matches(w, s) for w in p.split()) for p in parts for s in sides)
    if len(parts) == 2 and len(sides) == 2:  # "A vs B": one team each side, either order
        a, b = parts
        return ((_part_matches(a, sides[0]) and _part_matches(b, sides[1]))
                or (_part_matches(a, sides[1]) and _part_matches(b, sides[0])))
    return all(any(_part_matches(p, s) for s in sides) for p in parts)


def _event_row(ev, kind):
    status = espn._status(ev.get("status") or {})
    if kind == "team":
        sides = {c.get("homeAway"): c for c in (ev.get("competitions") or [{}])[0].get("competitors", [])}
        h, a = sides.get("home", {}), sides.get("away", {})
        name = (f"{(a.get('team') or {}).get('displayName')} {a.get('score', '')} at "
                f"{(h.get('team') or {}).get('displayName')} {h.get('score', '')}")
    else:
        name = ev.get("name")
    return {"event_id": ev["id"], "name": name, "start_utc": ev.get("date"),
            "status": status["detail"] or status["description"]}


def search_events(sport_key, query, date=None, start_time=None):
    """Games on `date` (default today) whose teams match `query`, best first.
    ESPN files games under the US Eastern date, so if nothing matches on the day
    itself the day before and after are tried too. start_time ('7:30 PM') puts
    the game starting closest to it first (doubleheaders, series).
    Returns (matches, every row on the searched day, YYYYMMDD searched)."""
    kind = espn_call(espn.resolve, sport_key)[3]
    parts = [p.strip().lower() for p in TEAM_SPLIT_RE.split(query or "") if p and p.strip()]
    day = date_arg(date) or datetime.now().strftime("%Y%m%d")
    base = datetime.strptime(day, "%Y%m%d")
    matches, day_rows = [], []
    for offset in (0, -1, 1):
        d = (base + timedelta(days=offset)).strftime("%Y%m%d")
        events = espn_call(espn.get_scoreboard_raw, sport_key, d).get("events", [])
        if offset == 0:
            day_rows = [_event_row(ev, kind) for ev in events]
        matches = [_event_row(ev, kind) for ev in events if not parts or _event_matches(ev, kind, parts)]
        if matches:
            day = d
            break
    if start_time and str(start_time).strip() and len(matches) > 1:
        target = dateparser.parse(str(start_time), default=base, tzinfos=TZINFOS)
        target = target.astimezone() if target.tzinfo is None else target
        matches.sort(key=lambda r: abs((dateparser.isoparse(r["start_utc"]) - target).total_seconds()))
    return matches, day_rows, day


def find_game(sport_key, query, date, start_time):
    """'Find game' button: fill the event id with the best match and list the candidates."""
    sport_key = (sport_key or "").strip()
    if not (query or "").strip():
        return gr.update(), "Type a game to find, e.g. 'Warriors vs Rockets'."
    try:
        matches, day_rows, day = search_events(sport_key, query, date, start_time)
    except Exception as e:
        return gr.update(), f"Could not search ESPN: {e}"
    pretty_day = datetime.strptime(day, "%Y%m%d").strftime("%a %b %d, %Y")
    if not matches:
        others = "\n".join(f"{r['event_id']}  [{r['status']}]  {r['name']}  ({_local_start(r['start_utc'])})"
                           for r in day_rows) or "(no games that day)"
        return gr.update(), (f"No '{sport_key}' game matching '{query}' on {pretty_day} "
                             f"(or the day either side).\nGames that day:\n{others}")
    lines = [f"{'->' if i == 0 else '  '} {r['event_id']}  [{r['status']}]  {r['name']}  "
             f"({_local_start(r['start_utc'])})" for i, r in enumerate(matches)]
    note = "" if len(matches) == 1 else "\nMore than one match; the first was picked. Paste another id to switch."
    return matches[0]["event_id"], f"Found on {pretty_day}:\n" + "\n".join(lines) + note


def resolve_event_id(sport_key, event_id, query, date, start_time):
    """Event id to load: the id box if filled, else the name search, else the first
    live game (team sports) / the current event (golf, tennis, MMA, racing; None)."""
    event_id = (event_id or "").strip()
    if event_id:
        return event_id
    if (query or "").strip():
        matches, _, day = search_events(sport_key, query, date, start_time)
        if not matches:
            raise RuntimeError(f"no game matching '{query}' around {day}. Use 'Find game' to see that day's games.")
        return matches[0]["event_id"]
    if espn_call(espn.resolve, sport_key)[3] == "team":
        live = espn_call(espn.list_events, sport_key, live_only=True)
        if not live:
            raise RuntimeError(f"no live '{sport_key}' games right now. Enter an event id, "
                               "or find a game by name and date.")
        return live[0]["event_id"]
    return None


# --- Commentary at a timestamp ---

_TZ = {"America/New_York": ("ET", "EST", "EDT"), "America/Chicago": ("CT", "CST", "CDT"),
       "America/Denver": ("MT", "MST", "MDT"), "America/Los_Angeles": ("PT", "PST", "PDT"),
       "Europe/London": ("BST",), "UTC": ("UTC", "GMT", "Z")}
TZINFOS = {abbr: ZoneInfo(zone) for zone, abbrs in _TZ.items() for abbr in abbrs}

# A timestamp is real-world time if it has AM/PM, a date, a time zone, or is a
# bare "h:mm" (no quarter/period); anything else is read as game clock.
WALL_TIME_RE = re.compile(
    r"\b(?:[ap]\.?m\.?)\b|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d|"
    r"\b(?:" + "|".join(a.lower() for a in TZINFOS) + r")\b|^\s*\d{1,2}:\d{2}(?::\d{2})?\s*$",
    re.I)
HAS_DATE_RE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}|"
                         r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d", re.I)
INF = float("inf")
# Soccer commentary lines that start a period but carry no period number
SOCCER_PERIOD_STARTS = [("second half begins", 2), ("first half extra time begins", 3),
                        ("second half extra time begins", 4)]


def _clock_secs(clock):
    """'5:12' -> 312, '38.9' -> 38.9, '' -> None."""
    m = re.fullmatch(r"(\d+):(\d{2}(?:\.\d+)?)", clock or "")
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    return float(clock) if re.fullmatch(r"\d+(?:\.\d+)?", clock or "") else None


def _clock_counts_down(plays):
    """Basketball/football clocks count down, ESPN's hockey clock counts up; read it from the data."""
    by_period = {}
    for p in plays:
        s = _clock_secs(p["clock"])
        if p["period"] and s is not None:
            by_period.setdefault(p["period"], []).append(s)
    votes = sum((s[0] > s[-1]) - (s[0] < s[-1]) for s in by_period.values() if len(s) > 1)
    return votes >= 0


def _game_positions(sport, plays, counts_down):
    """A comparable (period, progress, extra) for every play; plays with no time
    (lineups, 'Match ends') take the previous play's."""
    out, last, half, period = [], (0, -INF, -INF), 0, None
    for p in plays:
        pos = None
        if sport == "soccer":
            # Some lines ("Second Half begins") have no period: carry the current one
            text = p["text"].strip().lower()
            period = p["period"] or next((n for phrase, n in SOCCER_PERIOD_STARTS if text.startswith(phrase)),
                                         period)
            label = minute_tracker._minute_label(period, p["clock"]) if period else None
            if label:
                m = re.match(r"(\d+)'(?:\+(\d+)')?", label)
                pos = (period, int(m.group(1)), int(m.group(2) or 0))
        elif sport == "baseball":  # no clock: (inning, top 0 / bottom 1)
            text = p["text"].strip().lower()
            if text.startswith("top of the"):
                half = 0
            elif text.startswith("bottom of the"):
                half = 1
            if p["period"]:
                pos = (p["period"], half, 0)
        else:
            s = _clock_secs(p["clock"])
            if p["period"] and s is not None:
                pos = (p["period"], -s if counts_down else s, 0)
        last = pos or last
        out.append(last)
    return out


def _parse_game_clock(text, sport, sport_key, counts_down):
    """Game-clock timestamp -> position comparable with _game_positions().
    soccer: 67 / 67' / 45+2 / 67:30   baseball: top 7 / bottom 7th / 7th inning / T7
    others: Q3 5:12 / 3rd 5:12 / P2 14:05 / 2OT 1:30 / Q4 (end of Q4)   all: HT, FT"""
    t = text.strip().lower()
    if t in ("ft", "full time", "fulltime", "final", "end", "end of game"):
        return (INF, INF, INF)
    if t in ("ht", "half time", "halftime", "half"):
        if sport == "soccer" or sport_key == "ncaam":
            return (1, INF, INF)
        if sport in ("basketball", "football"):
            return (2, INF, INF)
        raise ValueError(f"'{text}' isn't a time in {sport}.")
    if sport == "soccer":
        m = re.fullmatch(r"(\d{1,3})\s*['’]?\s*(?:\+\s*(\d{1,2})\s*['’]?)?", t)
        m2 = re.fullmatch(r"(\d{1,3}):(\d{2})", t)
        if m:
            base, added = int(m.group(1)), int(m.group(2) or 0)
        elif m2:  # 66:30 on a match clock is the 67th minute
            base, added = int(m2.group(1)) + 1, 0
        else:
            raise ValueError(f"Couldn't read '{text}' as a soccer minute. Try 67, 67' or 45+2.")
        period = next((p for p, end in ((1, 45), (2, 90), (3, 105)) if base <= end), 4)
        return (period, base, added)
    if sport == "baseball":
        m = re.search(r"(top|bottom|bot|middle|mid|end|t|b)?\.?\s*(?:of\s+(?:the\s+)?)?(\d{1,2})(?:st|nd|rd|th)?", t)
        if not m:
            raise ValueError(f"Couldn't read '{text}' as an inning. Try 'top 7', 'bottom 7th' or '7th inning'.")
        half = 0 if m.group(1) in ("top", "t", "middle", "mid") else 1
        return (int(m.group(2)), half, INF)

    clock = None
    m = re.search(r"(\d{1,2}:\d{2}(?:\.\d+)?|\b\d{1,2}\.\d\b)", t)
    if m:
        clock = _clock_secs(m.group(1))
        t = t[:m.start()] + " " + t[m.end():]
    regulation = 2 if sport_key == "ncaam" else {"hockey": 3}.get(sport, 4)
    m_ot = re.search(r"(\d*)\s*(?:ot|overtime)\b", t)
    m_p = re.search(r"(?:\b|^)(?:q|p|h|qtr|quarter|period|half)\s*(\d)\b|\b(\d)\s*(?:st|nd|rd|th)?\s*"
                    r"(?:q|qtr|quarter|period|per|half|h|p)?\b", t)
    if m_ot:
        period = regulation + int(m_ot.group(1) or 1)
    elif m_p:
        period = int(m_p.group(1) or m_p.group(2))
    else:
        raise ValueError(f"Couldn't read '{text}' as a game time. Include the period, e.g. "
                         "'Q3 5:12', '2nd 14:05' or 'OT 1:30'.")
    if clock is None:
        return (period, INF, INF)  # just "Q3": the end of that period
    return (period, -clock if counts_down else clock, INF)


def _parse_wall_time(text, plays):
    """Real-world timestamp -> UTC epoch seconds. No time zone = this computer's.
    No date = the game's date (the next day if that falls inside a game that ran past midnight)."""
    walls = [dateparser.isoparse(p["wallclock"]) for p in plays if p["wallclock"]]
    if not walls:
        raise ValueError("ESPN has no real-world times for this game's plays; use the game clock instead.")
    day = walls[0].astimezone().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    dt = dateparser.parse(text, default=day, tzinfos=TZINFOS)
    dt = dt.astimezone() if dt.tzinfo is None else dt
    if not HAS_DATE_RE.search(text) and dt < walls[0] and dt + timedelta(days=1) <= walls[-1] + timedelta(hours=1):
        dt += timedelta(days=1)
    return dt.timestamp()


def plays_up_to(snap, plays, sport_key, text):
    """The plays that had happened by timestamp `text`, and how it was read."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Enter a timestamp, e.g. 67', Q3 5:12, top 7, or 9:42 PM.")
    if WALL_TIME_RE.search(text):
        target = (_parse_wall_time(text, plays),)
        positions, last = [], (-INF,)
        for p in plays:
            if p["wallclock"]:
                last = (dateparser.isoparse(p["wallclock"]).timestamp(),)
            positions.append(last)
        read_as = "real time " + datetime.fromtimestamp(target[0]).strftime("%a %b %d, %I:%M %p")
    else:
        counts_down = _clock_counts_down(plays)
        target = _parse_game_clock(text, snap["sport"], sport_key, counts_down)
        positions = _game_positions(snap["sport"], plays, counts_down)
        read_as = "game clock"
    upto = max((i for i, pos in enumerate(positions) if pos <= target), default=None)
    return ([] if upto is None else plays[:upto + 1]), read_as


def commentary_at_time(sport_key, event_id, query, date, start_time, timestamp, plays_per_update, key_only):
    """'Commentary at timestamp' button: one Gemini comment on the replay update
    that the most recent plays as of `timestamp` would have produced."""
    if is_running:
        return "Stop the running commentary first, then try again."
    sport_key = (sport_key or "").strip()
    try:
        event_id = resolve_event_id(sport_key, event_id, query, date, start_time)
        snap, raw = espn_call(espn.get_snapshot_and_raw, sport_key, event_id)
    except Exception as e:
        log.warning(f"Timestamp lookup failed: {e}")
        return f"Could not load the game: {e}"
    if snap["kind"] != "team":
        return "Timestamps need play-by-play, which ESPN doesn't have for golf, tennis, MMA or racing."
    plays = replay_plays(snap, bool(key_only))
    if not plays:
        return "ESPN has no play-by-play for this game, so there's nothing to look up by time."
    try:
        upto, read_as = plays_up_to(snap, plays, sport_key, timestamp)
    except (ValueError, OverflowError) as e:
        return str(e)
    if not upto:
        return f"Nothing had happened yet at '{timestamp}' ({read_as})."

    split = not key_only and snap["sport"] not in HIGH_SCORING
    batch, score = group_plays(upto, int(plays_per_update), split, team_sides(snap))[-1]
    if len(upto) == len(plays):
        score = live_score(snap)  # up to the latest play: ESPN's own current score is authoritative
    stats = soccer_stats_at(soccer_replay_stats(raw, snap), batch, None) if snap["sport"] == "soccer" else None
    when = when_text(snap["sport"], batch[-1])
    header = f"As of {timestamp.strip()}: {when}  {score_text(snap, score)}"
    prompt = build_team_prompt(snap, batch, score, [], False, live=False, stats=stats)
    log.info(f"Timestamp '{timestamp}' ({read_as}): {len(batch)} plays up to {when}")
    threading.Thread(target=stream_espn_commentary, args=(prompt, header),
                     kwargs={"stop_on_stop": False}, daemon=True).start()
    title = f"{snap['teams']['away']['name']} at {snap['teams']['home']['name']}"
    # Live game, time not reached yet: we can only go up to ESPN's latest play
    ahead = (" That's later than ESPN's latest play, so this is the latest there is."
             if snap["status"]["state"] == "in" and len(upto) == len(plays) else "")
    return (f"{title}, as of '{timestamp.strip()}' ({read_as}): commenting on the latest "
            f"{len(batch)} play(s), last one at {when or 'the end'}, score {score_text(snap, score)}.{ahead}")


def start_live(mode, video_file, sport_key, event_id, query, game_date, start_time, poll_interval,
               replay_seconds, plays_per_update, key_only, attach_frame):
    global is_running, capture_thread, commentary_thread, commentary_buffer, latest_frame

    if is_running:
        log.warning("start_live called but already running")
        return "Already running."

    if mode == MODE_FRAMES:
        if video_file is None:
            return "Please upload a video file first."
        snap = None
    else:
        if attach_frame and video_file is None:
            return "Upload a video to attach frames, or untick 'Attach current video frame'."
        sport_key = (sport_key or "").strip()
        event_id = (event_id or "").strip()
        try:
            # Id box, else name + date search, else first live game / current event
            event_id = resolve_event_id(sport_key, event_id, query, game_date, start_time)
            snap, raw = espn_call(espn.get_snapshot_and_raw, sport_key, event_id or None)
        except Exception as e:
            log.error(f"ESPN lookup failed: {e}", exc_info=True)
            return f"Could not load ESPN event '{event_id}' for '{sport_key}': {e}"

    log.info(f"Starting live commentary. Mode: {mode} | Video: {video_file} | "
             f"Sport: {sport_key} | Event: {event_id}")
    commentary_buffer = []
    latest_frame = None
    is_running = True

    # In ESPN mode the video (if any) is display-only
    if video_file is not None:
        capture_thread = threading.Thread(
            target=capture_frames, args=(video_file, mode == MODE_ESPN), daemon=True
        )
        capture_thread.start()
        log.info("Capture thread started")

    if mode == MODE_FRAMES:
        commentary_thread = threading.Thread(target=run_commentary_loop, daemon=True)
        commentary_thread.start()
        log.info("Commentary thread started")
        return f"Live commentary started. First commentary in {COMMENTARY_INTERVAL}s..."

    commentary_thread = threading.Thread(
        target=run_espn_loop,
        args=(sport_key, snap, raw, int(poll_interval), float(replay_seconds), int(plays_per_update),
              bool(key_only), bool(attach_frame)),
        daemon=True
    )
    commentary_thread.start()
    log.info(f"ESPN commentary thread started (attach frame: {bool(attach_frame)})")

    if snap["kind"] == "team":
        title = f"{snap['teams']['away']['name']} at {snap['teams']['home']['name']}"
    else:
        title = snap["event"]["name"]
    if snap["status"]["state"] == "post":
        n = len(replay_plays(snap, bool(key_only))) if snap["kind"] == "team" else 0
        how = f"replaying {n} plays" if n else "finished, no play-by-play, so one recap"
    else:
        how = f"polling every {int(poll_interval)}s"
    frames = ", with video frames" if attach_frame else ""
    status = snap["status"]["detail"] or snap["status"]["description"]
    return f"{title} ({status}) — {how}{frames}."


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
    if is_thinking:  # also covers one-off timestamp calls, which run with no session
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

    mode = gr.Radio([MODE_FRAMES, MODE_ESPN], value=MODE_FRAMES, label="Commentary source")

    video_file = gr.File(
        label="Upload Video Clip (required for frame sampling, optional display for ESPN)",
        file_types=["video"],
        file_count="single",
    )

    with gr.Group(visible=False) as espn_options:
        with gr.Row():
            sport = gr.Dropdown(
                choices=[(f"{name} ({key})", key) for key, (_, _, name, _) in espn.SPORTS.items()],
                value=DEFAULT_SPORT, label="Sport / league", allow_custom_value=True, filterable=True,
                info="Not listed? Type sport/league, e.g. soccer/tur.1", scale=2,
            )
            event_id = gr.Textbox(
                label="ESPN event id",
                info="Used first if filled. Blank = the game found by name below, else the first "
                     "live game (golf, tennis, MMA, racing: the current event)", scale=2,
            )
        with gr.Row():
            search = gr.Textbox(label="Find a game by name", placeholder="Warriors vs Rockets", scale=2,
                                info="Team names, nicknames or abbreviations; tournament name for golf etc.")
            game_date = gr.DateTime(label="Date (blank = today)", include_time=False, type="string", scale=1)
            start_time = gr.Textbox(label="Start time (optional)", placeholder="7:30 PM", scale=1,
                                    info="Picks between several matches, e.g. a doubleheader")
        with gr.Row():
            find_btn = gr.Button("Find game", scale=1)
            list_btn = gr.Button("List events on date", scale=1)
        with gr.Row():
            timestamp = gr.Textbox(
                label="Timestamp", scale=3,
                info="Game clock: 67', 45+2, Q3 5:12, 2nd 14:05, OT 1:30, top 7, HT, FT. "
                     "Or real time: 9:42 PM, 9:42 PM ET, 2025-01-25 21:42 (no time zone = this computer's).",
            )
            at_btn = gr.Button("Commentary at timestamp", scale=1)
        with gr.Row():
            poll_interval = gr.Slider(5, 60, value=ESPN_POLL_INTERVAL, step=1,
                                      label="Live: seconds between ESPN polls")
            replay_seconds = gr.Slider(1, 15, value=REPLAY_SECONDS_PER_UPDATE, step=1,
                                       label="Finished game replay: seconds per update")
            plays_per_update = gr.Slider(1, 15, value=REPLAY_PLAYS_PER_UPDATE, step=1,
                                         label="Finished game replay: plays per update")
        key_only = gr.Checkbox(
            value=False,
            label="Replay key/scoring plays only (goals, cards, touchdowns, runs... — much shorter replays)",
        )
        attach_frame = gr.Checkbox(
            value=False,
            label="Attach current video frame (atmosphere only; best with the live broadcast of the same game)",
        )
        match_list = gr.Textbox(label="Events", interactive=False, lines=8)

    mode.change(fn=lambda m: gr.update(visible=(m == MODE_ESPN)), inputs=mode, outputs=espn_options)
    list_btn.click(fn=list_matches, inputs=[sport, game_date], outputs=match_list)
    find_btn.click(fn=find_game, inputs=[sport, search, game_date, start_time], outputs=[event_id, match_list])

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

    start_btn.click(fn=start_live,
                    inputs=[mode, video_file, sport, event_id, search, game_date, start_time,
                            poll_interval, replay_seconds, plays_per_update, key_only, attach_frame],
                    outputs=status)
    at_btn.click(fn=commentary_at_time,
                 inputs=[sport, event_id, search, game_date, start_time, timestamp, plays_per_update, key_only],
                 outputs=status)
    stop_btn.click(fn=stop_live, inputs=None, outputs=status)

    frame_timer = gr.Timer(value=0.1)
    frame_timer.tick(fn=get_latest_frame, outputs=webcam_feed)

    commentary_timer = gr.Timer(value=0.1)
    commentary_timer.tick(fn=get_commentary, outputs=commentary_out)
    commentary_timer.tick(fn=get_thinking_status, outputs=thinking_status)

demo.launch(share=True)
log.info("Gradio demo exited")
