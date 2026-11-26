"""ESPN soccer feed: full match snapshot, saved as structured JSON.

Uses ESPN's public (unofficial, undocumented) site API - no key required.
There is no SLA: endpoints can change or rate-limit without notice.

Usage:
    python espn_feed.py list                        # today's matches, all leagues
    python espn_feed.py list --live                 # only matches in progress
    python espn_feed.py list --date 20261005 --league eng.1 --out data/matches.json

    python espn_feed.py match --event 401861129             # -> data/match_401861129.json
    python espn_feed.py match --event 401861129 --raw       # also save ESPN's untouched JSON

Every command also prints a short text summary; add --quiet to skip it.
For minute-by-minute data, use minute_tracker.py.

Common league slugs: all, eng.1 (Premier League), esp.1 (LaLiga), ger.1,
ita.1, fra.1, usa.1 (MLS), uefa.champions, uefa.europa, fifa.world.

In code:
    from espn_feed import get_snapshot, build_context
    prompt_context = build_context(get_snapshot(event_id))
"""

import argparse
import html
import json
import os
import re
import time
import urllib.error
import urllib.request

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"
REQUEST_TIMEOUT = 15  # seconds
DATA_DIR = "data"

# Team stats shown in the short text summary (the JSON keeps all of them)
SUMMARY_STATS = {
    "possessionPct": "Possession %",
    "totalShots": "Shots",
    "shotsOnTarget": "Shots on target",
    "wonCorners": "Corners",
    "foulsCommitted": "Fouls",
    "yellowCards": "Yellow cards",
    "redCards": "Red cards",
    "saves": "Saves",
}


# --- HTTP ---

def fetch_json(url, retries=2):
    """GET a URL and parse JSON, retrying briefly on network errors."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
            time.sleep(2 * (attempt + 1))


def get_scoreboard(league="all", date=None):
    """List matches for a league on a date (YYYYMMDD; default = today per ESPN)."""
    url = f"{BASE_URL}/{league}/scoreboard"
    if date:
        url += f"?dates={date}"
    data = fetch_json(url)
    matches = []
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        status = ev["status"]
        matches.append({
            "event_id": ev["id"],
            "name": ev["name"],
            "kickoff_utc": ev["date"],
            "state": status["type"]["state"],  # pre / in / post
            "status": status["type"]["description"],
            "clock": status.get("displayClock"),
            "home": teams["home"]["team"]["displayName"],
            "away": teams["away"]["team"]["displayName"],
            "home_score": teams["home"].get("score"),
            "away_score": teams["away"].get("score"),
            "venue": (comp.get("venue") or {}).get("fullName"),
        })
    return matches


def get_live_matches(league="all"):
    return [m for m in get_scoreboard(league) if m["state"] == "in"]


def get_summary_raw(event_id, league="all"):
    """Full match summary JSON. league='all' works for any event id."""
    return fetch_json(f"{BASE_URL}/{league}/summary?event={event_id}")


# --- Parsing ---

def _num(value):
    """'65.6' -> 65.6, '4' -> 4, anything else unchanged."""
    if isinstance(value, (int, float)) or value is None:
        return value
    try:
        f = float(str(value).rstrip("%"))
        return int(f) if f.is_integer() else f
    except ValueError:
        return value


def _players(item):
    return [p["athlete"]["displayName"] for p in item.get("participants", []) if p.get("athlete")]


def _play(item, extra=None):
    """Common fields for key events and commentary plays."""
    out = {
        "id": item.get("id"),
        "period": (item.get("period") or {}).get("number"),
        "minute": (item.get("clock") or {}).get("displayValue", ""),
        "seconds": (item.get("clock") or {}).get("value"),
        "type": (item.get("type") or {}).get("text", ""),
        "team": (item.get("team") or {}).get("displayName"),
        "players": _players(item),
        "text": item.get("text") or "",
        "wallclock": item.get("wallclock"),
        "field_position": (
            {"x": item["fieldPositionX"], "y": item["fieldPositionY"]}
            if "fieldPositionX" in item else None
        ),
    }
    out.update(extra or {})
    return out


def _team(competitor, roster):
    t = competitor["team"]
    return {
        "id": t["id"],
        "name": t["displayName"],
        "abbr": t.get("abbreviation"),
        "score": int(competitor.get("score") or 0),
        "score_by_half": [_num(ls.get("displayValue")) for ls in competitor.get("linescores", [])],
        "winner": competitor.get("winner"),
        "formation": (roster or {}).get("formation"),
        "record": {r["type"]: r.get("summary") for r in competitor.get("record", [])},
    }


def _lineup(roster):
    players = []
    for p in roster.get("roster", []):
        players.append({
            "name": p["athlete"]["displayName"],
            "jersey": p.get("jersey"),
            "position": (p.get("position") or {}).get("displayName"),
            "starter": p.get("starter"),
            "subbed_in": p.get("subbedIn"),
            "subbed_out": p.get("subbedOut"),
            "formation_place": p.get("formationPlace"),
            "stats": {s["name"]: _num(s.get("displayValue")) for s in p.get("stats", [])},
        })
    return players


def _leaders(data):
    out = {}
    for team in data.get("leaders", []):
        abbr = team["team"].get("abbreviation") or team["team"]["displayName"]
        out[abbr] = {
            cat.get("displayName") or cat["name"]: [
                {"player": l["athlete"]["displayName"], "value": _num(l.get("displayValue"))}
                for l in cat.get("leaders", [])
            ]
            for cat in team.get("leaders", [])
        }
    return out


def _standings(data):
    rows = []
    for group in (data.get("standings") or {}).get("groups", []):
        for entry in (group.get("standings") or {}).get("entries", []):
            row = {"group": group.get("header"), "team": entry.get("team")}
            row.update({s.get("abbreviation") or s["name"]: _num(s.get("displayValue"))
                        for s in entry.get("stats", [])})
            rows.append(row)
    return rows


def _last_five(data):
    out = {}
    for team in data.get("lastFiveGames", []):
        out[team["team"]["displayName"]] = [{
            "date": g.get("gameDate"),
            "opponent": (g.get("opponent") or {}).get("displayName"),
            "home_or_away": "home" if g.get("atVs") == "vs" else "away",
            "score": g.get("score"),
            "result": g.get("gameResult"),
            "competition": g.get("competitionName"),
            "round": g.get("roundName"),
            "note": g.get("matchNote") or None,
        } for g in team.get("events", [])]
    return out


def _head_to_head(data):
    series = (data.get("seasonseries") or [None])[0]
    if not series:
        return None
    games = []
    for ev in series.get("events", []):
        sides = {c["homeAway"]: c for c in ev.get("competitors", [])}
        games.append({
            "date": ev.get("date"),
            "competition": ev.get("competitionName"),
            "home": sides.get("home", {}).get("team", {}).get("displayName"),
            "away": sides.get("away", {}).get("team", {}).get("displayName"),
            "score": f"{sides.get('home', {}).get('score')}-{sides.get('away', {}).get('score')}",
        })
    return {"summary": series.get("summary"), "games": games}


def _odds(data):
    odds = (data.get("odds") or [None])[0]
    if not odds:
        return None
    return {
        "provider": (odds.get("provider") or {}).get("name"),
        "details": odds.get("details"),
        "over_under": odds.get("overUnder"),
        "home_moneyline": (odds.get("homeTeamOdds") or {}).get("moneyLine"),
        "away_moneyline": (odds.get("awayTeamOdds") or {}).get("moneyLine"),
        "draw_moneyline": (odds.get("drawOdds") or {}).get("moneyLine"),
    }


def _recap(data):
    article = data.get("article") or {}
    if not article.get("headline"):
        return None
    story = re.sub(r"<[^>]+>", "", article.get("story", ""))
    story = re.sub(r"\n\s*\n+", "\n\n", html.unescape(story)).strip()
    return {
        "headline": article.get("headline"),
        "description": article.get("description"),
        "published": article.get("published"),
        "story": story,
    }


def parse_summary(data):
    """Turn ESPN's summary JSON into a structured snapshot dict."""
    header = data["header"]
    comp = header["competitions"][0]
    status = comp["status"]
    sides = {c["homeAway"]: c for c in comp["competitors"]}
    rosters = {r["homeAway"]: r for r in data.get("rosters", [])}
    game_info = data.get("gameInfo") or {}
    venue = game_info.get("venue") or {}
    meta = data.get("meta") or {}

    team_stats = {}
    for t in data.get("boxscore", {}).get("teams", []):
        abbr = t["team"].get("abbreviation") or t["team"]["displayName"]
        team_stats[abbr] = {s["name"]: _num(s.get("displayValue")) for s in t.get("statistics", [])}

    lineups = {}
    for roster in rosters.values():
        abbr = roster["team"].get("abbreviation") or roster["team"]["displayName"]
        lineups[abbr] = _lineup(roster)

    return {
        "event_id": header.get("id"),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "ESPN site API (unofficial)",
        "match": {
            "league": (header.get("league") or {}).get("name"),
            "league_slug": (header.get("league") or {}).get("slug"),
            "season": (header.get("season") or {}).get("year"),
            "kickoff_utc": comp.get("date"),
            "venue": venue.get("fullName"),
            "city": (venue.get("address") or {}).get("city"),
            "country": (venue.get("address") or {}).get("country"),
            "attendance": game_info.get("attendance"),
            "first_play_wallclock": meta.get("firstPlayWallClock"),
            "last_play_wallclock": meta.get("lastPlayWallClock"),
        },
        "status": {
            "state": status["type"]["state"],  # pre / in / post
            "description": status["type"]["description"],
            "clock": status.get("displayClock") or status["type"].get("shortDetail"),
            "period": status.get("period"),
        },
        "teams": {
            "home": _team(sides["home"], rosters.get("home")),
            "away": _team(sides["away"], rosters.get("away")),
        },
        "team_stats": team_stats,
        "lineups": lineups,
        "key_events": [_play(e, {"scoring": bool(e.get("scoringPlay"))})
                       for e in data.get("keyEvents", [])],
        "commentary": [
            _play({**(c.get("play") or {}), "text": c.get("text"), "clock": c.get("time")},
                  {"seq": c.get("sequence")})
            for c in data.get("commentary", [])
        ],
        "leaders": _leaders(data),
        "standings": _standings(data),
        "last_five": _last_five(data),
        "head_to_head": _head_to_head(data),
        "odds": _odds(data),
        "recap": _recap(data),
    }


def get_snapshot(event_id, league="all"):
    return parse_summary(get_summary_raw(event_id, league))


# --- Output ---

def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # atomic, so readers never see a half-written file


def build_context(snapshot, recent=6):
    """Compact plain-text match context to prepend to an LLM prompt."""
    h, a = snapshot["teams"]["home"], snapshot["teams"]["away"]
    m, st = snapshot["match"], snapshot["status"]
    lines = [
        f"MATCH: {h['name']} {h['score']} - {a['score']} {a['name']}",
        " | ".join(x for x in (
            f"STATUS: {st['description']}", f"Clock: {st['clock']}", m.get("league"), m.get("venue"),
        ) if x),
    ]
    if h.get("formation") or a.get("formation"):
        lines.append(f"FORMATIONS: {h['abbr']} {h.get('formation')} | {a['abbr']} {a.get('formation')}")

    stats = snapshot["team_stats"]
    if stats:
        lines.append("STATS:")
        for name, label in SUMMARY_STATS.items():
            if any(name in s for s in stats.values()):
                lines.append(f"  {label}: " + " | ".join(f"{t} {s.get(name, '-')}" for t, s in stats.items()))

    events = [e for e in snapshot["key_events"] if e["text"]]
    if events:
        lines.append("KEY EVENTS:")
        lines += [f"  {e['minute']} {e['type']}: {e['text']}" for e in events]

    recent_commentary = [c for c in snapshot["commentary"] if c["text"]][-recent:]
    if recent_commentary:
        lines.append("LATEST PLAY-BY-PLAY:")
        lines += [f"  {c['minute'] or '--'} {c['text']}" for c in recent_commentary]

    return "\n".join(lines)


# --- CLI ---

def cmd_list(args):
    matches = get_scoreboard(args.league, args.date)
    if args.live:
        matches = [m for m in matches if m["state"] == "in"]
    if args.out:
        save_json(matches, args.out)
        print(f"Saved {len(matches)} matches -> {args.out}")
    if not matches:
        print("No matches found.")
        return
    for m in matches:
        clock = f" {m['clock']}" if m["state"] == "in" else ""
        print(f"{m['event_id']}  {m['kickoff_utc']}  [{m['status']}{clock}]  "
              f"{m['home']} {m['home_score']} - {m['away_score']} {m['away']}")


def cmd_match(args):
    raw = get_summary_raw(args.event, args.league)
    snapshot = parse_summary(raw)

    out = args.out or os.path.join(DATA_DIR, f"match_{args.event}.json")
    save_json(snapshot, out)
    if args.raw:
        raw_path = os.path.join(os.path.dirname(out) or ".", f"match_{args.event}_raw.json")
        save_json(raw, raw_path)
        print(f"Saved ESPN raw response -> {raw_path}")

    if not args.quiet:
        print(build_context(snapshot, recent=args.recent))
    print(f"\nSaved -> {out}  ({len(snapshot['commentary'])} commentary lines, "
          f"{sum(len(p) for p in snapshot['lineups'].values())} players, "
          f"{len(next(iter(snapshot['team_stats'].values()), {}))} team stats each)")


def main():
    parser = argparse.ArgumentParser(description="ESPN soccer match snapshot -> JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list matches for a date")
    p.add_argument("--league", default="all")
    p.add_argument("--date", help="YYYYMMDD (default: today)")
    p.add_argument("--live", action="store_true", help="only in-progress matches")
    p.add_argument("--out", help="also save the list to this JSON file")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("match", help="save a past (or current) match to JSON")
    p.add_argument("--event", required=True)
    p.add_argument("--league", default="all")
    p.add_argument("--out", help=f"output file (default: {DATA_DIR}/match_<id>.json)")
    p.add_argument("--raw", action="store_true", help="also save ESPN's untouched response")
    p.add_argument("--recent", type=int, default=10, help="commentary lines in the text summary")
    p.add_argument("--quiet", action="store_true", help="don't print the text summary")
    p.set_defaults(func=cmd_match)

    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
