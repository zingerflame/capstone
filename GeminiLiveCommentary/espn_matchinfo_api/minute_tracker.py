"""Minute-by-minute status for a single soccer match (ESPN data).

    python minute_tracker.py --event 401861129      # finished match: rebuilt from play-by-play
    python minute_tracker.py --event 401900582      # live match: polled every 60s until full time
    python minute_tracker.py                        # first match live right now

Writes data/minutes_<event_id>.json (rewritten after every minute when live):

    {"match": {...}, "mode": "live" | "rebuilt", "minutes": [
        {"minute": "35'", "period": 1,
         "score": {"FRA": 0, "BEL": 1},
         "stats": {"FRA": {"shots": 9, "shots_on_target": 2, ...}, "BEL": {...}},
         "this_minute": {"deltas": {"BEL": {"shots": 1, "goals": 1}},
                         "events": ["Goal! France 0, Belgium 1. ..."]}},
        ...]}

Finished matches: every minute (incl. stoppage time) is rebuilt from ESPN's
commentary. Goals, shots, shots on target, corners, fouls, offsides and cards
are counted from it - these match ESPN's final box score. Possession and
passing aren't in the commentary, so they're only available live.

Live matches: one entry per poll, with ESPN's box score at that moment, which
adds possession, passes, saves, tackles and interceptions.
"""

import argparse
import json
import os
import re
import sys
import time

from espn_feed import get_live_matches, get_summary_raw, save_json

DATA_DIR = "data"
DEFAULT_INTERVAL = 60  # seconds

COUNTED_STATS = ["goals", "shots", "shots_on_target", "corners", "fouls",
                 "offsides", "yellow_cards", "red_cards"]

# Live-only: our stat name -> ESPN box score name
BOXSCORE_STATS = {
    "goals": None,  # taken from the score
    "shots": "totalShots",
    "shots_on_target": "shotsOnTarget",
    "corners": "wonCorners",
    "fouls": "foulsCommitted",
    "offsides": "offsides",
    "yellow_cards": "yellowCards",
    "red_cards": "redCards",
    "possession_pct": "possessionPct",
    "passes": "totalPasses",
    "accurate_passes": "accuratePasses",
    "saves": "saves",
    "tackles": "totalTackles",
    "interceptions": "interceptions",
}

# Commentary text prefix -> stats it adds to the acting team. ESPN's
# commentary uses fixed phrasing ("Attempt saved.", "Foul by ...").
# Goals aren't counted here - they're taken from the running score, so goals
# overturned by VAR drop back out.
COMMENTARY_RULES = [
    (r"^Goal!", ["shots", "shots_on_target"]),
    (r"^(Attempt|Penalty) saved", ["shots", "shots_on_target"]),
    (r"^Attempt (missed|blocked)|hits the (left |right )?(post|bar|crossbar)|^Penalty missed", ["shots"]),
    (r"^Corner,", ["corners"]),
    (r"^(Foul|Handball|Dangerous play) by|^Penalty conceded by", ["fouls"]),
    (r"^Offside,", ["offsides"]),
    (r"is shown the yellow card", ["yellow_cards"]),
    (r"is shown the red card|^Second yellow card to", ["red_cards"]),
]

# Lines that carry the running score: "Goal! Home 1, Away 0." / "Own Goal by
# X, Team. Home 1, Away 0." and VAR reversals "VAR Decision: No Goal Home 0-1 Away."
GOAL_SCORE_RE = re.compile(r"^(?:Goal!|Own Goal)[^.]*?(?:\.\s*)?([^.,]+?) (\d+), ([^.,]+?) (\d+)\.")
VAR_SCORE_RE = re.compile(r"^VAR Decision: (?:No )?Goal\b.*?(\d+)-(\d+)")


def _num(value):
    try:
        f = float(str(value).rstrip("%"))
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return None


def match_info(data):
    comp = data["header"]["competitions"][0]
    sides = {c["homeAway"]: c for c in comp["competitors"]}
    teams = {side: {"name": c["team"]["displayName"], "abbr": c["team"].get("abbreviation")}
             for side, c in sides.items()}
    return {
        "event_id": data["header"].get("id"),
        "league": (data["header"].get("league") or {}).get("name"),
        "kickoff_utc": comp.get("date"),
        "home": teams["home"],
        "away": teams["away"],
        "status": comp["status"]["type"]["description"],
        "state": comp["status"]["type"]["state"],
        "clock": comp["status"].get("displayClock"),
        "final_score": {teams[s]["abbr"]: int(sides[s].get("score") or 0) for s in sides},
    }


# --- Finished match: rebuild every minute from commentary ---

def _minute_label(period, display):
    """Normalise ESPN's clock text to "N'" / "N'+K'"; second-half kickoff
    ("45'" in period 2) belongs to minute 46."""
    m = re.match(r"(\d+)'(?:\+(\d+)')?", display or "")
    if not m:
        return None
    base, added = int(m.group(1)), int(m.group(2) or 0)
    period_start = {1: 1, 2: 46, 3: 91, 4: 106}.get(period, 1)
    if not added and base < period_start:
        base = period_start
    return f"{base}'" + (f"+{added}'" if added else "")


def _minute_order(period):
    """All regular minutes of a period, e.g. 1'..45' for period 1."""
    start, end = {1: (1, 45), 2: (46, 90), 3: (91, 105), 4: (106, 120)}.get(period, (1, 45))
    return [f"{m}'" for m in range(start, end + 1)]


def rebuild_minutes(data, info):
    home, away = info["home"], info["away"]
    abbr_by_name = {home["name"]: home["abbr"], away["name"]: away["abbr"]}

    # Bucket commentary lines by (period, minute label)
    buckets = {}
    periods_seen = set()
    period = 1
    for c in data.get("commentary", []):
        play = c.get("play") or {}
        # Lines like "First Half ends" have no play; they belong to the current period
        period = (play.get("period") or {}).get("number") or period
        label = _minute_label(period, (c.get("time") or {}).get("displayValue"))
        if label is None:
            continue
        periods_seen.add(period)
        buckets.setdefault((period, label), []).append(c)

    # Minute list: every regular minute plus any stoppage minutes that occurred
    timeline = []
    for period in sorted(periods_seen):
        labels = _minute_order(period)
        extra = sorted({lbl for (p, lbl) in buckets if p == period and "+" in lbl},
                       key=lambda s: int(s.split("+")[1].rstrip("'")))
        timeline += [(period, lbl) for lbl in labels + extra]

    totals = {t: {s: 0 for s in COUNTED_STATS} for t in (home["abbr"], away["abbr"])}
    score = {home["abbr"]: 0, away["abbr"]: 0}
    minutes = []
    for period, label in timeline:
        deltas, events = {}, []
        for c in buckets.get((period, label), []):
            text = c.get("text") or ""
            team = abbr_by_name.get(((c.get("play") or {}).get("team") or {}).get("displayName"))
            # Scores are read by position (home first): commentary team names can
            # differ from the header ("Bosnia and Herzegovina" vs "Bosnia-Herzegovina")
            m = GOAL_SCORE_RE.search(text)
            if m:
                score = {home["abbr"]: int(m.group(2)), away["abbr"]: int(m.group(4))}
            m = VAR_SCORE_RE.search(text)
            if m:
                score = {home["abbr"]: int(m.group(1)), away["abbr"]: int(m.group(2))}
            if team:
                for pattern, stats in COMMENTARY_RULES:
                    if re.search(pattern, text):
                        for s in stats:
                            totals[team][s] += 1
                            deltas.setdefault(team, {}).setdefault(s, 0)
                            deltas[team][s] += 1
                        break
            if text:
                events.append(text)
        for team in totals:
            if score[team] != totals[team]["goals"]:
                deltas.setdefault(team, {})["goals"] = score[team] - totals[team]["goals"]
                totals[team]["goals"] = score[team]
        minutes.append({
            "minute": label,
            "period": period,
            "score": dict(score),
            "stats": {t: dict(s) for t, s in totals.items()},
            "this_minute": {"deltas": deltas, "events": events},
        })
    return minutes


# --- Live match: poll the box score every minute ---

def live_entry(data, info, prev):
    comp = data["header"]["competitions"][0]
    sides = {c["homeAway"]: c for c in comp["competitors"]}
    score = {info[s]["abbr"]: int(sides[s].get("score") or 0) for s in ("home", "away")}

    stats = {}
    for t in data.get("boxscore", {}).get("teams", []):
        abbr = t["team"].get("abbreviation")
        raw = {s["name"]: _num(s.get("displayValue")) for s in t.get("statistics", [])}
        stats[abbr] = {ours: (score.get(abbr) if espn is None else raw.get(espn))
                       for ours, espn in BOXSCORE_STATS.items()}

    deltas, events = {}, []
    if prev:
        for team, s in stats.items():
            for name, value in s.items():
                old = prev["stats"].get(team, {}).get(name)
                if isinstance(value, (int, float)) and isinstance(old, (int, float)) and value != old:
                    deltas.setdefault(team, {})[name] = round(value - old, 1)
        seen = prev["_seen_seq"]
    else:
        seen = set()
    commentary = data.get("commentary", [])
    events = [c.get("text") for c in commentary if _commentary_key(c) not in seen and c.get("text")]
    if not prev:
        events = events[-5:]  # first poll: just the latest lines, not the whole match

    status = comp["status"]
    return {
        "minute": status.get("displayClock") or status["type"].get("shortDetail"),
        "period": status.get("period"),
        "status": status["type"]["description"],
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "score": score,
        "stats": stats,
        "this_minute": {"deltas": deltas, "events": events},
        "_seen_seq": {_commentary_key(c) for c in commentary},
    }


def _commentary_key(c):
    """Identity of a commentary line. Some competitions send no sequence
    numbers (every line has sequence None), so fall back to the play id,
    then to clock + text."""
    if c.get("sequence") is not None:
        return c["sequence"]
    play_id = (c.get("play") or {}).get("id")
    if play_id:
        return play_id
    return ((c.get("time") or {}).get("value"), c.get("text"))


# --- Output ---

def print_minute(entry, info):
    h, a = info["home"]["abbr"], info["away"]["abbr"]
    s = entry["stats"]

    def pair(name):
        return f"{s.get(h, {}).get(name, '-')}-{s.get(a, {}).get(name, '-')}"

    line = (f"{entry['minute']:>7} | {h} {entry['score'][h]}-{entry['score'][a]} {a} | "
            f"shots {pair('shots')}  on target {pair('shots_on_target')}  "
            f"corners {pair('corners')}  fouls {pair('fouls')}")
    if s.get(h, {}).get("possession_pct") is not None:
        line += f"  poss {pair('possession_pct')}"
    print(line)
    for e in entry["this_minute"]["events"]:
        print(f"{'':>9}- {e}")


def write_output(path, info, mode, minutes):
    clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in minutes]
    save_json({"match": info, "mode": mode, "minutes": clean}, path)


def main():
    parser = argparse.ArgumentParser(description="Minute-by-minute status for one soccer match")
    parser.add_argument("--event", help="ESPN event id (default: first live match)")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help="seconds between polls for a live match")
    parser.add_argument("--out", help=f"output file (default: {DATA_DIR}/minutes_<id>.json)")
    parser.add_argument("--quiet", action="store_true", help="don't print each minute")
    args = parser.parse_args()

    event_id = args.event
    if not event_id:
        live = get_live_matches()
        if not live:
            print("No live matches right now. Pass --event <id> (find ids with: python espn_feed.py list)")
            return
        event_id = live[0]["event_id"]

    out = args.out or os.path.join(DATA_DIR, f"minutes_{event_id}.json")
    data = get_summary_raw(event_id)
    info = match_info(data)
    print(f"{info['home']['name']} vs {info['away']['name']} ({info['league']}) - {info['status']}")

    if info["state"] == "post":
        if not data.get("commentary"):
            print(f"ESPN has no play-by-play for this match (final score only: "
                  f"{info['final_score']}), so minutes can't be rebuilt. Smaller "
                  f"competitions often lack it - try a major league match.")
            return
        minutes = rebuild_minutes(data, info)
        if not args.quiet:
            for m in minutes:
                print_minute(m, info)
        write_output(out, info, "rebuilt", minutes)
        print(f"\nSaved {len(minutes)} minutes -> {out}")
        return

    # Pre-match: wait for kickoff. Live: one entry per poll until full time.
    minutes, prev = [], None
    try:
        while True:
            if info["state"] == "pre":
                print(f"[{time.strftime('%H:%M:%S')}] not started (kickoff {info['kickoff_utc']}), waiting...")
            else:
                entry = live_entry(data, info, prev)
                minutes.append(entry)
                prev = entry
                if not args.quiet:
                    print_minute(entry, info)
                write_output(out, info, "live", minutes)
                if info["state"] == "post":
                    print(f"\nFull time. Saved {len(minutes)} entries -> {out}")
                    return
            time.sleep(args.interval)
            try:
                data = get_summary_raw(event_id)
                info = match_info(data)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] fetch failed: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print(f"\nStopped. Saved {len(minutes)} entries -> {out}")


if __name__ == "__main__":
    main()
