"""ESPN multi-sport feed: live scores, box scores and play-by-play as JSON.

Works for soccer, basketball, American football, baseball, hockey, rugby,
Aussie rules, lacrosse, volleyball, golf, tennis, MMA and motor racing.

Uses ESPN's public site API. It is UNOFFICIAL and UNDOCUMENTED: no API key is
needed, but there is no SLA, endpoints can change or rate-limit without
notice, and some leagues return less detail than others. Be polite: don't
poll faster than every ~15-30 seconds.

Requires only the Python standard library (Python 3.8+).

-----------------------------------------------------------------------------
QUICK START
-----------------------------------------------------------------------------
    python espn_multisport.py sports                  # every supported sport/league
    python espn_multisport.py sports --check          # ...and test each one live

    python espn_multisport.py list --sport nba        # today's NBA games
    python espn_multisport.py list --sport mlb --date 20261028
    python espn_multisport.py list --sport epl --live # only games in progress

    python espn_multisport.py snapshot --sport nba --event 401909907
    python espn_multisport.py snapshot --sport mlb --event 401908046 --raw
    python espn_multisport.py snapshot --sport pga    # current golf tournament
    python espn_multisport.py snapshot --sport f1     # latest race weekend

    # Every play / minute of a finished game, start to finish
    python espn_multisport.py plays --sport nba --event 401909907
    python espn_multisport.py plays --sport mls --event 761927 --scoring   # goals/cards/subs only
    python espn_multisport.py plays --sport mlb --event 401908046 --csv data/mlb_plays.csv

Snapshots are saved to data/<sport>_<event id>.json. --raw also saves ESPN's
untouched response, which is the best way to explore what else is in there.

Any league not in the list can still be tried as "sport/league", e.g.
    python espn_multisport.py list --sport soccer/tur.1

In code:
    from espn_multisport import get_snapshot, build_context
    snap = get_snapshot("nba", "401909907")
    print(build_context(snap))      # compact text, handy for an LLM prompt

-----------------------------------------------------------------------------
HOW THE API IS SHAPED
-----------------------------------------------------------------------------
    https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard[?dates=YYYYMMDD]
    https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/summary?event={id}

  * TEAM sports (two sides, home/away): `scoreboard` lists games and
    `summary` gives the full detail: box score, player stats, play-by-play,
    leaders, standings, odds, recap.
  * INDIVIDUAL sports (golf, tennis, MMA, racing): `summary` does NOT work
    (ESPN returns 404/400). Everything comes from `scoreboard`:
      golf   -> one tournament with a leaderboard and round-by-round scores
      tennis -> one tournament with many matches, grouped by draw
      mma    -> one fight card with many bouts
      racing -> one race weekend with sessions (practice, qualifying, race)

What each team sport's summary includes (checked November 2026):
    soccer      team stats, lineups, commentary, key events (goals/cards/subs)
    basketball  team + player box score, full play-by-play, win probability
    football    team + player box score, drives, scoring plays, win probability
    baseball    batting/pitching box score, pitch-level play-by-play, R-H-E
    hockey      team + player box score (skaters/goalies), play-by-play
    afl         box score, play-by-play
    rugby, lacrosse, volleyball, field hockey, water polo, college hockey:
                score by period and box score only (no play-by-play)

Not included: cricket (league ids are per-series and the data is often stale)
and boxing (the endpoint returns 400).
"""

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"
REQUEST_TIMEOUT = 15  # seconds
DATA_DIR = "data"

# --- Supported sports -------------------------------------------------------
# key: (sport path, league slug, display name, kind)
# kind decides how the data is parsed: "team", "golf", "tennis", "mma", "racing"
# Every entry below was checked against the live API in November 2026.
SPORTS = {
    # Soccer
    "soccer":          ("soccer", "all", "All soccer leagues", "team"),
    "epl":             ("soccer", "eng.1", "English Premier League", "team"),
    "championship":    ("soccer", "eng.2", "English Championship", "team"),
    "fa-cup":          ("soccer", "eng.fa", "English FA Cup", "team"),
    "laliga":          ("soccer", "esp.1", "Spanish LaLiga", "team"),
    "bundesliga":      ("soccer", "ger.1", "German Bundesliga", "team"),
    "serie-a":         ("soccer", "ita.1", "Italian Serie A", "team"),
    "ligue-1":         ("soccer", "fra.1", "French Ligue 1", "team"),
    "eredivisie":      ("soccer", "ned.1", "Dutch Eredivisie", "team"),
    "primeira-liga":   ("soccer", "por.1", "Portuguese Primeira Liga", "team"),
    "liga-mx":         ("soccer", "mex.1", "Mexican Liga MX", "team"),
    "mls":             ("soccer", "usa.1", "MLS", "team"),
    "nwsl":            ("soccer", "usa.nwsl", "NWSL", "team"),
    "isl":             ("soccer", "ind.1", "Indian Super League", "team"),
    "ucl":             ("soccer", "uefa.champions", "UEFA Champions League", "team"),
    "europa":          ("soccer", "uefa.europa", "UEFA Europa League", "team"),
    "libertadores":    ("soccer", "conmebol.libertadores", "CONMEBOL Libertadores", "team"),
    "world-cup":       ("soccer", "fifa.world", "FIFA World Cup", "team"),
    # Basketball
    "nba":             ("basketball", "nba", "NBA", "team"),
    "wnba":            ("basketball", "wnba", "WNBA", "team"),
    "g-league":        ("basketball", "nba-development", "NBA G League", "team"),
    "ncaam":           ("basketball", "mens-college-basketball", "NCAA Men's Basketball", "team"),
    "ncaaw":           ("basketball", "womens-college-basketball", "NCAA Women's Basketball", "team"),
    "nbl":             ("basketball", "nbl", "Australian NBL", "team"),
    "fiba":            ("basketball", "fiba", "FIBA World Cup", "team"),
    # American / Canadian football
    "nfl":             ("football", "nfl", "NFL", "team"),
    "ncaaf":           ("football", "college-football", "NCAA Football", "team"),
    "cfl":             ("football", "cfl", "Canadian Football League", "team"),
    "ufl":             ("football", "ufl", "United Football League", "team"),
    # Baseball
    "mlb":             ("baseball", "mlb", "MLB", "team"),
    "college-baseball": ("baseball", "college-baseball", "NCAA Baseball", "team"),
    "wbc":             ("baseball", "world-baseball-classic", "World Baseball Classic", "team"),
    "caribbean-series": ("baseball", "caribbean-series", "Caribbean Series", "team"),
    "little-league":   ("baseball", "llb", "Little League Baseball", "team"),
    # Ice hockey
    "nhl":             ("hockey", "nhl", "NHL", "team"),
    "ncaa-hockey-m":   ("hockey", "mens-college-hockey", "NCAA Men's Ice Hockey", "team"),
    "ncaa-hockey-w":   ("hockey", "womens-college-hockey", "NCAA Women's Ice Hockey", "team"),
    "olympic-hockey-m": ("hockey", "olympics-mens-ice-hockey", "Olympic Men's Ice Hockey", "team"),
    # Rugby (rugby union leagues use numeric ids)
    "rugby-world-cup": ("rugby", "164205", "Rugby World Cup", "team"),
    "six-nations":     ("rugby", "180659", "Six Nations", "team"),
    "super-rugby":     ("rugby", "242041", "Super Rugby Pacific", "team"),
    "urc":             ("rugby", "270557", "United Rugby Championship", "team"),
    "premiership":     ("rugby", "267979", "Gallagher Premiership", "team"),
    "rugby-tests":     ("rugby", "289234", "Rugby international test matches", "team"),
    "nrl":             ("rugby-league", "3", "NRL (rugby league)", "team"),
    # Other team sports
    "afl":             ("australian-football", "afl", "AFL (Aussie rules)", "team"),
    "pll":             ("lacrosse", "pll", "Premier Lacrosse League", "team"),
    "nll":             ("lacrosse", "nll", "National Lacrosse League", "team"),
    "ncaa-lacrosse-m": ("lacrosse", "mens-college-lacrosse", "NCAA Men's Lacrosse", "team"),
    "ncaa-volleyball-w": ("volleyball", "womens-college-volleyball", "NCAA Women's Volleyball", "team"),
    "ncaa-volleyball-m": ("volleyball", "mens-college-volleyball", "NCAA Men's Volleyball", "team"),
    "ncaa-field-hockey-w": ("field-hockey", "womens-college-field-hockey", "NCAA Women's Field Hockey", "team"),
    "ncaa-water-polo-m": ("water-polo", "mens-college-water-polo", "NCAA Men's Water Polo", "team"),
    # Golf
    "pga":             ("golf", "pga", "PGA Tour", "golf"),
    "lpga":            ("golf", "lpga", "LPGA Tour", "golf"),
    "dp-world":        ("golf", "eur", "DP World Tour", "golf"),
    "champions-tour":  ("golf", "champions-tour", "PGA Tour Champions", "golf"),
    "korn-ferry":      ("golf", "ntw", "Korn Ferry Tour", "golf"),
    "liv":             ("golf", "liv", "LIV Golf", "golf"),
    "olympic-golf-m":  ("golf", "mens-olympics-golf", "Olympic Golf (men)", "golf"),
    # Tennis
    "atp":             ("tennis", "atp", "ATP Tour (men)", "tennis"),
    "wta":             ("tennis", "wta", "WTA Tour (women)", "tennis"),
    # Combat sports
    "ufc":             ("mma", "ufc", "UFC", "mma"),
    "pfl":             ("mma", "pfl", "Professional Fighters League", "mma"),
    # Motor racing
    "f1":              ("racing", "f1", "Formula 1", "racing"),
    "indycar":         ("racing", "irl", "IndyCar Series", "racing"),
    "nascar-cup":      ("racing", "nascar-premier", "NASCAR Cup Series", "racing"),
    "nascar-xfinity":  ("racing", "nascar-secondary", "NASCAR second-tier series", "racing"),
    "nascar-truck":    ("racing", "nascar-truck", "NASCAR Truck Series", "racing"),
}

# Used when someone passes a raw "sport/league" that is not in SPORTS
KIND_BY_SPORT = {"golf": "golf", "tennis": "tennis", "mma": "mma", "racing": "racing"}

# A few headline team stats per sport for the text summary (JSON keeps all)
CONTEXT_STATS = {
    "soccer": {
        "possessionPct": "Possession %", "totalShots": "Shots", "shotsOnTarget": "On target",
        "wonCorners": "Corners", "foulsCommitted": "Fouls", "yellowCards": "Yellow cards",
        "redCards": "Red cards", "saves": "Saves",
    },
    "basketball": {
        "fieldGoalsMade-fieldGoalsAttempted": "FG", "fieldGoalPct": "FG %",
        "threePointFieldGoalsMade-threePointFieldGoalsAttempted": "3PT",
        "freeThrowsMade-freeThrowsAttempted": "FT", "totalRebounds": "Rebounds",
        "assists": "Assists", "turnovers": "Turnovers", "steals": "Steals", "blocks": "Blocks",
    },
    "football": {
        "totalYards": "Total yards", "netPassingYards": "Passing yards", "rushingYards": "Rushing yards",
        "firstDowns": "First downs", "thirdDownEff": "3rd down", "turnovers": "Turnovers",
        "totalPenaltiesYards": "Penalties", "possessionTime": "Possession",
    },
    "hockey": {
        "shotsTotal": "Shots", "powerPlayGoals": "PP goals", "powerPlayOpportunities": "PP chances",
        "faceoffPercent": "Faceoff %", "hits": "Hits", "blockedShots": "Blocked shots",
        "penaltyMinutes": "PIM",
    },
}


# --- HTTP -------------------------------------------------------------------

def fetch_json(url, retries=2):
    """GET a URL and parse JSON, retrying briefly on network errors."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.load(resp)
        except urllib.error.HTTPError:
            raise  # 4xx/5xx: retrying rarely helps
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
            time.sleep(2 * (attempt + 1))


def resolve(sport_key):
    """'nba' or 'basketball/nba' -> (sport, league, display name, kind)."""
    if sport_key in SPORTS:
        return SPORTS[sport_key]
    if "/" in sport_key:
        sport, league = sport_key.split("/", 1)
        return sport, league, sport_key, KIND_BY_SPORT.get(sport, "team")
    raise SystemExit(f"Unknown sport '{sport_key}'. Run 'python espn_multisport.py sports' "
                     f"for the list, or pass sport/league like 'soccer/tur.1'.")


def get_scoreboard_raw(sport_key, date=None, event_id=None):
    sport, league, _, _ = resolve(sport_key)
    params = {}
    if date:
        params["dates"] = date
    if event_id:
        params["event"] = event_id
    query = ("?" + urllib.parse.urlencode(params)) if params else ""
    return fetch_json(f"{BASE_URL}/{sport}/{league}/scoreboard{query}")


def get_summary_raw(sport_key, event_id):
    """Full game summary (team sports only)."""
    sport, league, _, _ = resolve(sport_key)
    return fetch_json(f"{BASE_URL}/{sport}/{league}/summary?event={event_id}")


# --- Small helpers ----------------------------------------------------------

def _num(value):
    """'65.6' -> 65.6, '4' -> 4, 67.0 -> 67, '-25' -> -25, anything else unchanged."""
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    try:
        f = float(str(value).rstrip("%"))
        return int(f) if f.is_integer() else f
    except ValueError:
        return value


def _abbr(team):
    return team.get("abbreviation") or team.get("displayName") or team.get("id")


def _status(status):
    st = status.get("type") or {}
    return {
        "state": st.get("state"),               # pre / in / post
        "description": st.get("description"),   # e.g. "In Progress", "Final"
        "detail": st.get("detail") or st.get("shortDetail"),  # e.g. "Top 5th", "Q3 5:12"
        "clock": status.get("displayClock"),
        "period": status.get("period"),
    }


def _country(athlete):
    return ((athlete or {}).get("flag") or {}).get("alt")


# --- Lists (scoreboard) -----------------------------------------------------

def list_events(sport_key, date=None, live_only=False):
    """One row per game (team sports) or per tournament/card/race (others)."""
    _, _, _, kind = resolve(sport_key)
    data = get_scoreboard_raw(sport_key, date)
    rows = []
    for ev in data.get("events", []):
        status = _status(ev.get("status") or {})
        row = {
            "event_id": ev["id"],
            "name": ev.get("name"),
            "start_utc": ev.get("date"),
            "end_utc": ev.get("endDate"),
            "state": status["state"],
            "status": status["detail"] or status["description"],
        }
        comps = ev.get("competitions") or []
        if kind == "team" and comps:
            sides = {c.get("homeAway"): c for c in comps[0].get("competitors", [])}
            home, away = sides.get("home", {}), sides.get("away", {})
            row.update({
                "home": (home.get("team") or {}).get("displayName"),
                "away": (away.get("team") or {}).get("displayName"),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "venue": (comps[0].get("venue") or {}).get("fullName"),
            })
        elif kind == "golf" and comps:
            leader = next(iter(sorted(comps[0].get("competitors", []),
                                      key=lambda c: c.get("order", 999))), None)
            if leader:
                row["leader"] = f"{leader['athlete']['displayName']} ({leader.get('score')})"
        elif kind == "tennis":
            row["matches"] = sum(len(g.get("competitions", [])) for g in ev.get("groupings", []))
        elif kind == "mma":
            row["bouts"] = len(comps)
        elif kind == "racing":
            row["sessions"] = [((c.get("type") or {}).get("abbreviation")) for c in comps]
        rows.append(row)
    if live_only:
        rows = [r for r in rows if r["state"] == "in"]
    return rows


# --- Team sports (summary endpoint) -----------------------------------------

def _team(competitor):
    t = competitor.get("team") or {}
    out = {
        "id": t.get("id"),
        "name": t.get("displayName"),
        "abbr": t.get("abbreviation"),
        "score": _num(competitor.get("score")),
        # Score per period: halves (soccer), quarters, periods, innings, ...
        "score_by_period": [_num(ls.get("displayValue", ls.get("value")))
                            for ls in competitor.get("linescores", [])],
        "winner": competitor.get("winner"),
        "record": {r.get("type") or r.get("name"): r.get("summary")
                   for r in competitor.get("record", []) if isinstance(r, dict)},
    }
    if "hits" in competitor:  # baseball: R-H-E line
        out["hits"] = competitor.get("hits")
        out["errors"] = competitor.get("errors")
    return out


def _team_stats(boxscore):
    """{team abbr: {stat name: value}}. Baseball stats are grouped, e.g. 'batting.hits'."""
    out = {}
    for t in boxscore.get("teams", []):
        stats = {}
        for s in t.get("statistics", []):
            if "stats" in s:  # grouped (baseball: batting / pitching / fielding)
                for x in s["stats"]:
                    stats[f"{s.get('name')}.{x.get('name')}"] = _num(x.get("displayValue"))
            else:
                stats[s.get("name") or s.get("label")] = _num(s.get("displayValue"))
        out[_abbr(t.get("team") or {})] = stats
    return out


def _player_stats(boxscore):
    """{team abbr: {stat group: [ {name, starter, <label>: value, ...} ]}}.

    Groups are e.g. passing/rushing (NFL), batting/pitching (MLB),
    forwards/defenses/goalies (NHL), or just 'stats' (NBA).
    """
    out = {}
    for t in boxscore.get("players", []):
        groups = {}
        for g in t.get("statistics", []):
            name = g.get("name") or g.get("type") or g.get("text") or "stats"
            labels = g.get("labels") or g.get("keys") or []
            groups[name] = [{
                "name": a["athlete"].get("displayName"),
                "starter": a.get("starter"),
                "did_not_play": a.get("didNotPlay"),
                **{lbl: _num(v) for lbl, v in zip(labels, a.get("stats", []))},
            } for a in g.get("athletes", []) if a.get("athlete")]
        out[_abbr(t.get("team") or {})] = groups
    return out


def _lineups(data):
    """Soccer/rugby/baseball rosters: who started, positions, per-player stats."""
    out = {}
    for roster in data.get("rosters", []):
        players = []
        for p in roster.get("roster", []):
            players.append({
                "name": (p.get("athlete") or {}).get("displayName"),
                "jersey": p.get("jersey"),
                "position": (p.get("position") or {}).get("displayName"),
                "starter": p.get("starter"),
                "subbed_in": p.get("subbedIn"),
                "subbed_out": p.get("subbedOut"),
                "stats": {s.get("name") or s.get("abbreviation"): _num(s.get("displayValue"))
                          for s in p.get("stats", []) if isinstance(s, dict)},
            })
        team = roster.get("team") or {}
        out[_abbr(team)] = {"formation": roster.get("formation"), "players": players}
    return out


def _play(item, team_names):
    team = item.get("team") or {}
    period = item.get("period") or {}
    return {
        "id": item.get("id"),
        "period": period.get("number"),
        "period_label": period.get("displayValue"),
        "clock": (item.get("clock") or {}).get("displayValue", ""),
        "type": (item.get("type") or {}).get("text", ""),
        "team": team.get("displayName") or team_names.get(team.get("id")),
        "players": [p["athlete"]["displayName"] for p in item.get("participants", [])
                    if (p.get("athlete") or {}).get("displayName")],
        "text": item.get("text") or "",
        "scoring": bool(item.get("scoringPlay")),
        "home_score": item.get("homeScore"),
        "away_score": item.get("awayScore"),
        "wallclock": item.get("wallclock"),
    }


def _plays(data, team_names):
    """Play-by-play, whatever the sport calls it."""
    if data.get("commentary"):  # soccer
        items = [{**(c.get("play") or {}), "text": c.get("text"), "clock": c.get("time")}
                 for c in data["commentary"]]
    elif data.get("plays"):     # basketball, baseball, hockey, AFL
        items = data["plays"]
    else:                       # American football: plays live inside drives
        drives = data.get("drives") or {}
        all_drives = list(drives.get("previous", []))
        if drives.get("current"):
            all_drives.append(drives["current"])
        items, seen = [], set()
        for d in all_drives:
            for p in d.get("plays", []):
                if p.get("id") not in seen:
                    seen.add(p.get("id"))
                    items.append(p)
    return [_play(p, team_names) for p in items]


def _key_events(data, plays, team_names):
    """Soccer: goals/cards/subs. Others: scoring plays."""
    if data.get("keyEvents"):
        return [_play(e, team_names) for e in data["keyEvents"]]
    if data.get("scoringPlays"):
        return [_play(e, team_names) for e in data["scoringPlays"]]
    return [p for p in plays if p["scoring"]]


def _leaders(data):
    out = {}
    for team in data.get("leaders", []):
        out[_abbr(team.get("team") or {})] = {
            cat.get("displayName") or cat.get("name"): [
                {"player": l["athlete"]["displayName"], "value": l.get("displayValue")}
                for l in cat.get("leaders", []) if l.get("athlete")
            ]
            for cat in team.get("leaders", [])
        }
    return out


def _standings(data):
    rows = []
    for group in (data.get("standings") or {}).get("groups", []):
        for entry in (group.get("standings") or {}).get("entries", []):
            row = {"group": group.get("header"), "team": entry.get("team")}
            row.update({s.get("abbreviation") or s.get("name"): _num(s.get("displayValue"))
                        for s in entry.get("stats", [])})
            rows.append(row)
    return rows


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
    }


def _win_probability(data):
    wp = data.get("winprobability") or []
    if not wp:
        return None
    last = wp[-1]
    return {"home_win_pct": last.get("homeWinPercentage"), "tie_pct": last.get("tiePercentage")}


def _injuries(data):
    out = {}
    for team in data.get("injuries", []):
        out[_abbr(team.get("team") or {})] = [{
            "player": (i.get("athlete") or {}).get("displayName"),
            "status": i.get("status"),
            "detail": (i.get("details") or {}).get("type"),
        } for i in team.get("injuries", [])]
    return out


def _recap(data):
    article = data.get("article") or {}
    if not article.get("headline"):
        return None
    story = re.sub(r"<[^>]+>", "", article.get("story", ""))
    story = re.sub(r"\n\s*\n+", "\n\n", html.unescape(story)).strip()
    return {"headline": article.get("headline"), "description": article.get("description"),
            "published": article.get("published"), "story": story}


def parse_team_summary(data, sport_key):
    sport, _, display, _ = resolve(sport_key)
    header = data["header"]
    comp = header["competitions"][0]
    sides = {c.get("homeAway"): c for c in comp["competitors"]}
    team_names = {c["team"]["id"]: c["team"]["displayName"] for c in comp["competitors"]}
    game_info = data.get("gameInfo") or {}
    venue = game_info.get("venue") or {}
    box = data.get("boxscore") or {}
    plays = _plays(data, team_names)

    return {
        "sport": sport,
        "sport_key": sport_key,
        "kind": "team",
        "event_id": header.get("id"),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "ESPN site API (unofficial)",
        "game": {
            "league": (header.get("league") or {}).get("name") or display,
            "season": (header.get("season") or {}).get("year"),
            "start_utc": comp.get("date"),
            "venue": venue.get("fullName"),
            "city": (venue.get("address") or {}).get("city"),
            "attendance": game_info.get("attendance"),
        },
        "status": _status(comp.get("status") or {}),
        "teams": {"home": _team(sides.get("home", {})), "away": _team(sides.get("away", {}))},
        "team_stats": _team_stats(box),
        "player_stats": _player_stats(box),
        "lineups": _lineups(data),
        "key_events": _key_events(data, plays, team_names),
        "plays": plays,
        "leaders": _leaders(data),
        "win_probability": _win_probability(data),
        "injuries": _injuries(data),
        "standings": _standings(data),
        "odds": _odds(data),
        "recap": _recap(data),
    }


# --- Individual sports (scoreboard only) ------------------------------------

def parse_golf(ev):
    comp = (ev.get("competitions") or [{}])[0]
    players = []
    for c in sorted(comp.get("competitors", []), key=lambda c: c.get("order", 999)):
        rounds = c.get("linescores", [])
        players.append({
            "rank": c.get("order"),
            "name": (c.get("athlete") or {}).get("displayName"),
            "country": _country(c.get("athlete")),
            "to_par": c.get("score"),
            "rounds": [{"round": r.get("period"), "strokes": _num(r.get("value")),
                        "to_par": r.get("displayValue")} for r in rounds],
            # holes with a score in the latest round (18 = round finished)
            "holes_played_this_round": len((rounds[-1] if rounds else {}).get("linescores", [])),
        })
    return {"leaderboard": players}


def _set_scores(competitor):
    out = []
    for s in competitor.get("linescores", []):
        game = str(_num(s.get("value")))
        if s.get("tiebreak") is not None:
            game += f"({s['tiebreak']})"
        out.append(game)
    return out


def parse_tennis(ev):
    matches = []
    for g in ev.get("groupings", []):
        draw = (g.get("grouping") or {}).get("displayName")
        for c in g.get("competitions", []):
            players = sorted(c.get("competitors", []), key=lambda p: p.get("order", 9))
            matches.append({
                "match_id": c.get("id"),
                "draw": draw,
                "round": (c.get("round") or {}).get("displayName"),
                "start_utc": c.get("date"),
                "status": _status(c.get("status") or {}),
                "players": [{
                    # singles: one athlete; doubles: a "roster" of two
                    "name": ((p.get("athlete") or {}).get("displayName")
                             or (p.get("roster") or {}).get("displayName")),
                    "country": _country(p.get("athlete")) or " / ".join(
                        filter(None, (_country(a) for a in (p.get("roster") or {}).get("athletes", [])))) or None,
                    "winner": p.get("winner"),
                    "sets": _set_scores(p),
                } for p in players],
            })
    return {"matches": matches}


def parse_mma(ev):
    bouts = []
    for c in ev.get("competitions", []):
        status = c.get("status") or {}
        bouts.append({
            "bout_id": c.get("id"),
            "weight_class": (c.get("type") or {}).get("abbreviation"),
            "start_utc": c.get("date"),
            "status": _status(status),
            "result": (status.get("result") or {}).get("displayName"),
            "fighters": [{
                "name": (f.get("athlete") or {}).get("displayName"),
                "country": _country(f.get("athlete")),
                "record": next((r.get("summary") for r in f.get("records", [])), None),
                "winner": f.get("winner"),
            } for f in sorted(c.get("competitors", []), key=lambda f: f.get("order", 9))],
        })
    return {"bouts": bouts}  # ESPN lists the main event last


def parse_racing(ev):
    circuit = ev.get("circuit") or {}
    sessions = []
    for c in ev.get("competitions", []):
        sessions.append({
            "session": (c.get("type") or {}).get("abbreviation"),  # FP1, Qual, Race, ...
            "start_utc": c.get("date"),
            "status": _status(c.get("status") or {}),
            "results": [{
                "position": d.get("order"),
                "name": (d.get("athlete") or {}).get("displayName"),
                "country": _country(d.get("athlete")),
                "winner": d.get("winner"),
            } for d in sorted(c.get("competitors", []), key=lambda d: d.get("order", 999))],
        })
    return {
        "circuit": circuit.get("fullName"),
        "city": (circuit.get("address") or {}).get("city"),
        "country": (circuit.get("address") or {}).get("country"),
        "sessions": sessions,
    }


INDIVIDUAL_PARSERS = {"golf": parse_golf, "tennis": parse_tennis, "mma": parse_mma, "racing": parse_racing}


def find_event(sport_key, event_id=None, date=None):
    """Find one event in the scoreboard. No event_id -> the first one listed."""
    data = get_scoreboard_raw(sport_key, date, event_id)
    events = data.get("events", [])
    if event_id:
        events = [e for e in events if str(e.get("id")) == str(event_id)]
        if not events and not date:  # some sports ignore ?event=, try today's board
            events = [e for e in get_scoreboard_raw(sport_key).get("events", [])
                      if str(e.get("id")) == str(event_id)]
    if not events:
        raise SystemExit(f"Event {event_id or ''} not found for '{sport_key}'. "
                         f"Try 'list --sport {sport_key} --date YYYYMMDD' to find it.")
    return events[0], data


def parse_individual(ev, sport_key):
    sport, _, display, kind = resolve(sport_key)
    snap = {
        "sport": sport,
        "sport_key": sport_key,
        "kind": kind,
        "event_id": ev.get("id"),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "ESPN site API (unofficial)",
        "event": {
            "league": display,
            "name": ev.get("name"),
            "start_utc": ev.get("date"),
            "end_utc": ev.get("endDate"),
        },
        "status": _status(ev.get("status") or {}),
    }
    snap.update(INDIVIDUAL_PARSERS[kind](ev))
    return snap


# --- One entry point for every sport ----------------------------------------

def get_snapshot_and_raw(sport_key, event_id=None, date=None):
    _, _, _, kind = resolve(sport_key)
    if kind == "team":
        if not event_id:
            raise SystemExit("Team sports need --event. Use 'list' to find event ids.")
        raw = get_summary_raw(sport_key, event_id)
        return parse_team_summary(raw, sport_key), raw
    ev, raw = find_event(sport_key, event_id, date)
    return parse_individual(ev, sport_key), raw


def get_snapshot(sport_key, event_id=None, date=None):
    """Structured dict for any sport. event_id is optional for individual sports."""
    return get_snapshot_and_raw(sport_key, event_id, date)[0]


# --- Output -----------------------------------------------------------------

def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # atomic, so readers never see a half-written file


def _context_team(s, recent):
    h, a = s["teams"]["home"], s["teams"]["away"]
    st, g = s["status"], s["game"]
    started = st["state"] != "pre"
    lines = [
        f"GAME: {a['name']} {a['score']} at {h['name']} {h['score']}" if started
        else f"GAME: {a['name']} at {h['name']} (not started)",
        " | ".join(str(x) for x in (st["detail"] or st["description"], g.get("league"), g.get("venue")) if x),
    ]
    if h["score_by_period"] or a["score_by_period"]:
        lines.append("BY PERIOD:")
        for t in (a, h):
            line = f"  {t['abbr']}: " + " ".join(str(x) for x in t["score_by_period"])
            if "hits" in t:
                line += f"   (R {t['score']}  H {t['hits']}  E {t['errors']})"
            lines.append(line)

    wanted = CONTEXT_STATS.get(s["sport"], {})
    stats = s["team_stats"]
    shown = [(n, lbl) for n, lbl in wanted.items() if any(n in v for v in stats.values())]
    if shown:
        lines.append("TEAM STATS:")
        for name, label in shown:
            lines.append(f"  {label}: " + " | ".join(f"{t} {v.get(name, '-')}" for t, v in stats.items()))

    for team, cats in s["leaders"].items():
        tops = [f"{cat}: {ls[0]['player']} {ls[0]['value']}" for cat, ls in cats.items() if ls]
        if tops:
            lines.append(f"LEADERS {team}: " + "; ".join(tops[:4]))

    if s["win_probability"] and s["win_probability"]["home_win_pct"] is not None:
        lines.append(f"WIN PROBABILITY: {h['abbr']} {s['win_probability']['home_win_pct'] * 100:.0f}%")

    key = [e for e in s["key_events"] if e["text"]][-recent:]
    if key:
        lines.append("KEY / SCORING EVENTS:")
        lines += [f"  {e['period_label'] or e['period'] or ''} {e['clock']} {e['text']}".rstrip() for e in key]

    plays = [p for p in s["plays"] if p["text"]][-recent:]
    if plays:
        lines.append("LATEST PLAY-BY-PLAY:")
        lines += [f"  {p['period_label'] or p['period'] or ''} {p['clock']} {p['text']}".rstrip() for p in plays]
    return lines


def _context_golf(s, recent):
    lines = ["LEADERBOARD:"]
    for p in s["leaderboard"][:max(recent, 10)]:
        rounds = " ".join(str(r["strokes"]) for r in p["rounds"] if r["strokes"] is not None)
        lines.append(f"  {p['rank']:>3}. {p['name']} ({p['country']})  {p['to_par']}   rounds: {rounds}")
    return lines


def _context_tennis(s, recent):
    matches = s["matches"]
    live = [m for m in matches if m["status"]["state"] == "in"]
    done = [m for m in matches if m["status"]["state"] == "post"][-recent:]
    lines = []
    for title, group in (("LIVE MATCHES:", live), ("LATEST RESULTS:", done)):
        if group:
            lines.append(title)
            for m in group:
                p = m["players"]
                score = "  ".join(f"{x}-{y}" for x, y in zip(p[0]["sets"], p[1]["sets"])) if len(p) == 2 else ""
                names = " vs ".join(f"{x['name']}{' (W)' if x['winner'] else ''}" for x in p)
                lines.append(f"  [{m['draw']}, {m['round']}] {names}  {score}")
    return lines


def _context_mma(s, recent):
    del recent  # the whole card is always shown; arg kept for a uniform signature
    lines = ["FIGHT CARD (main event last):"]
    for b in s["bouts"]:
        names = " vs ".join(f"{f['name']} ({f['record']}){' (W)' if f['winner'] else ''}" for f in b["fighters"])
        extra = f" - {b['result']}" if b["result"] else ""
        lines.append(f"  [{b['weight_class']}] {names}  {b['status']['detail']}{extra}")
    return lines


def _context_racing(s, recent):
    lines = [f"CIRCUIT: {s['circuit']} ({s['city']}, {s['country']})"]
    for sess in s["sessions"]:
        top = ", ".join(f"{r['position']}. {r['name']}" for r in sess["results"][:max(recent, 3)])
        lines.append(f"  {sess['session']} [{sess['status']['description']}]: {top}")
    return lines


def build_context(snapshot, recent=6):
    """Compact plain-text summary of a snapshot, e.g. to prepend to an LLM prompt."""
    kind = snapshot["kind"]
    if kind == "team":
        return "\n".join(_context_team(snapshot, recent))
    ev, st = snapshot["event"], snapshot["status"]
    head = [f"EVENT: {ev['name']} ({ev['league']})",
            f"STATUS: {st['detail'] or st['description']}"]
    body = {"golf": _context_golf, "tennis": _context_tennis,
            "mma": _context_mma, "racing": _context_racing}[kind](snapshot, recent)
    return "\n".join(head + body)


# --- CLI --------------------------------------------------------------------

def cmd_sports(args):
    rows = list(SPORTS.items())
    status = {}
    if args.check:
        def check(item):
            key = item[0]
            try:
                n = len(get_scoreboard_raw(key).get("events", []))
                return key, f"OK ({n} events today)"
            except Exception as exc:  # noqa: BLE001 - report any failure
                return key, f"FAIL ({exc})"
        with ThreadPoolExecutor(8) as pool:
            status = dict(pool.map(check, rows))

    current = None
    for key, (sport, league, name, kind) in rows:
        if sport != current:
            current = sport
            print(f"\n{sport.upper()}")
        extra = f"  {status[key]}" if status else ""
        print(f"  {key:<20} {name:<36} {sport}/{league:<28} [{kind}]{extra}")
    print("\nAny other league can be tried as sport/league, e.g. --sport soccer/tur.1")


def cmd_list(args):
    rows = list_events(args.sport, args.date, args.live)
    if args.out:
        save_json(rows, args.out)
        print(f"Saved {len(rows)} events -> {args.out}")
    if not rows:
        print("No events found.")
        return
    for r in rows:
        if "home" in r:
            print(f"{r['event_id']}  {r['start_utc']}  [{r['status']}]  "
                  f"{r['away']} {r['away_score']} at {r['home']} {r['home_score']}")
        else:
            extra = {k: v for k, v in r.items()
                     if k in ("leader", "matches", "bouts", "sessions")}
            print(f"{r['event_id']}  {r['start_utc']}  [{r['status']}]  {r['name']}  {extra or ''}")


def cmd_snapshot(args):
    snap, raw = get_snapshot_and_raw(args.sport, args.event, args.date)
    key = args.sport.replace("/", "_")
    out = args.out or os.path.join(DATA_DIR, f"{key}_{snap['event_id']}.json")
    save_json(snap, out)
    if args.raw:
        raw_path = os.path.join(os.path.dirname(out) or ".", f"{key}_{snap['event_id']}_raw.json")
        save_json(raw, raw_path)
        print(f"Saved ESPN raw response -> {raw_path}")
    if not args.quiet:
        print(build_context(snap, recent=args.recent))
    print(f"\nSaved -> {out}")


def cmd_plays(args):
    """Every play of one game, start to finish (works best on finished games)."""
    if resolve(args.sport)[3] != "team":
        raise SystemExit("Play-by-play is only available for team sports. "
                         "Use 'snapshot' for golf, tennis, MMA and racing.")
    snap = get_snapshot(args.sport, args.event)
    plays = [p for p in snap["plays"] if p["text"]]
    if args.scoring:
        plays = [p for p in snap["key_events"] if p["text"]]
    if not plays:
        print("ESPN has no play-by-play for this game (common for rugby, lacrosse, "
              "volleyball and games that haven't started).")
        return
    h, a = snap["teams"]["home"], snap["teams"]["away"]
    print(f"{a['name']} at {h['name']} - {snap['status']['detail']} - {len(plays)} plays\n")
    for p in plays:
        when = f"{p['period_label'] or p['period'] or ''} {p['clock']}".strip()
        score = (f"[{a['abbr']} {p['away_score']}-{p['home_score']} {h['abbr']}]"
                 if p["home_score"] is not None else "")
        print(f"{when:<18} {score:<16} {p['text']}")
    if args.csv:
        import csv
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        cols = ["period", "period_label", "clock", "team", "type", "text", "scoring",
                "away_score", "home_score", "wallclock"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(plays)
        print(f"\nSaved {len(plays)} plays -> {args.csv}")


def main():
    parser = argparse.ArgumentParser(description="ESPN multi-sport snapshot -> JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sports", help="list supported sports and leagues")
    p.add_argument("--check", action="store_true", help="test each endpoint live")
    p.set_defaults(func=cmd_sports)

    p = sub.add_parser("list", help="list games / events for a date")
    p.add_argument("--sport", required=True, help="key from 'sports', or sport/league")
    p.add_argument("--date", help="YYYYMMDD (default: today)")
    p.add_argument("--live", action="store_true", help="only events in progress")
    p.add_argument("--out", help="also save the list to this JSON file")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("snapshot", aliases=["game", "match"], help="save one game/event to JSON")
    p.add_argument("--sport", required=True, help="key from 'sports', or sport/league")
    p.add_argument("--event", help="event id (required for team sports)")
    p.add_argument("--date", help="YYYYMMDD, helps find past golf/tennis/MMA/racing events")
    p.add_argument("--out", help=f"output file (default: {DATA_DIR}/<sport>_<id>.json)")
    p.add_argument("--raw", action="store_true", help="also save ESPN's untouched response")
    p.add_argument("--recent", type=int, default=8, help="lines per section in the text summary")
    p.add_argument("--quiet", action="store_true", help="don't print the text summary")
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("plays", help="print every play of a game (full timeline)")
    p.add_argument("--sport", required=True, help="key from 'sports', or sport/league")
    p.add_argument("--event", required=True, help="event id (from 'list')")
    p.add_argument("--scoring", action="store_true",
                   help="only scoring/key plays (soccer: goals, cards, subs)")
    p.add_argument("--csv", help="also save the plays to this CSV file")
    p.set_defaults(func=cmd_plays)

    args = parser.parse_args()
    try:  # Windows consoles default to cp1252, which can't print names like "Tomić"
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    try:
        args.func(args)
    except KeyboardInterrupt:
        pass
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"ESPN returned HTTP {exc.code} for {exc.url}") from None


if __name__ == "__main__":
    main()
