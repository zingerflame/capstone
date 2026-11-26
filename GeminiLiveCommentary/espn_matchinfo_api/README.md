# ESPN Match Data

Saves soccer match data from ESPN as JSON: a full snapshot of a game, or the game minute by minute.

- Python 3.8+, no extra packages, no API key.
- Uses ESPN's unofficial public API, which can change or stop working without notice.
- Keep both scripts in the same folder (`minute_tracker.py` uses `espn_feed.py`).

## Run

```bash
# 1. Find a match id
python espn_feed.py list                     # today's matches
python espn_feed.py list --date 20261005     # a given day (YYYYMMDD)
python espn_feed.py list --live              # games in progress

# 2a. Full snapshot of a game -> data/match_<id>.json
python espn_feed.py match --event 401861129

# 2b. Minute by minute -> data/minutes_<id>.json
python minute_tracker.py --event 401861129   # finished game: built at once
python minute_tracker.py --event <live id>   # live game: updates every 60s until full time
```

Add `--quiet` to skip console output. For a single league, add `--league eng.1` to `list` (`esp.1`, `ger.1`, `ita.1`, `fra.1`, `usa.1` and `uefa.champions` also work).

## What you get

**Full snapshot:** teams, score, 28 team stats, lineups with per-player stats, key events, all commentary, standings, last 5 games, head-to-head, odds and the match recap. Stats are full-game totals.

**Minute by minute:** one entry per minute with the score, running stats per team, and what changed that minute (stat changes and commentary lines).

**Sample:** `sample_live_minutes.json` was recorded live from Real Sociedad vs Deportivo (LaLiga, 11 Oct 2026), minutes 29' to 32', and includes a goal.

## Live vs recorded (minute by minute)

| | Recorded (finished game) | Live game |
|---|---|---|
| Score, goals, shots, shots on target, corners, fouls, offsides, cards | ✅ | ✅ |
| Possession, passes, saves, tackles, interceptions | ❌ | ✅ |
| Commentary lines per minute | ✅ | ✅ |
| Minutes covered | Every minute, incl. stoppage time | One entry per poll; minutes can be skipped or repeated |
| How | Rebuilt from ESPN's play-by-play text | ESPN's live box score, recorded each minute |

A finished game can only be rebuilt if ESPN has play-by-play for it. Major leagues do, but smaller competitions often don't. To get the full stats minute by minute, record the game while it's live.
