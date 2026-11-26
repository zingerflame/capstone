# ESPN Match Data Mode (`LiveMainESPN.py`)

`LiveMainESPN.py` is a copy of `LiveMain.py` with a second way to produce commentary: instead of sending video frames to Gemini, it reads live game data from ESPN and only asks Gemini to comment when something actually happens. It works for soccer, basketball, American football, baseball, hockey, rugby, golf, tennis, MMA, motor racing and more. `LiveMain.py` itself is unchanged.

## Key features

- **Many sports, one app.** Pick from nearly 70 leagues in a dropdown (Premier League, NBA, NFL, MLB, NHL, PGA, ATP, UFC, F1, ...), or type any ESPN `sport/league` such as `soccer/tur.1`.
- **Facts from ESPN, not guessed from video.** Gemini gets the real score, team stats and play-by-play as text, and is told never to invent players, events or scores.
- **Only calls Gemini when something happens.** A live game is polled every *N* seconds. A comment is triggered by new plays, a score change, a new period, or a stat change such as a shot, corner or card. Quiet polls make no Gemini call.
- **Live team stats in every prompt.** Each sport gets its own headline stats, e.g. NBA field goals, 3-pointers, rebounds and assists, or NFL yards, first downs and turnovers.
- **Soccer gets extra detail.** Live matches include passes, accurate passes, tackles, interceptions, offsides and saves. Replays show running totals (shots, on target, corners, fouls, offsides, cards) rebuilt minute by minute from the commentary.
- **Replays of finished games.** The full play-by-play is replayed in small updates, which is handy for testing without a live game. You set the speed (seconds per update) and the size (plays per update).
- **Every score gets its own comment** in low-scoring sports (soccer, hockey, football, baseball). Basketball scores too often for that, so its updates are cut by size only.
- **Key plays only option.** Replays just goals, cards, touchdowns and runs. An NBA replay drops from about 120 updates to about 30.
- **Golf, tennis, MMA and racing** have no play-by-play, so Gemini comments on what changed since the last poll: a new leader, a finished match, a fight result. Finished events get a one-off recap.
- **Join a game in progress.** The first comment covers only the latest 5 plays, not the whole game so far.
- **Find a game by name and date.** Type `Warriors vs Rockets` and pick a date instead of looking up an ESPN event id (ids still work).
- **Commentary at a timestamp.** Type a moment, either game clock (`Q3 5:12`, `67'`, `top 7`) or real-world time (`9:42 PM ET`), and get one comment on what had just happened at that point. Works on finished games and on live games for any time that has already passed.
- **Optional video frame.** You can attach the current broadcast frame, which Gemini uses for atmosphere only (crowd, celebrations), never for facts.
- **Original frame-sampling mode kept.** It's unchanged and selectable at the top of the page.

## The two modes

Pick one with **Commentary source** at the top of the page.

| | Frame sampling (original) | ESPN match data |
|---|---|---|
| Input | Uploaded video (required) | Sport + ESPN event id; video optional |
| When Gemini is called | Every 5s, regardless of what's happening | Only when ESPN reports something new |
| What Gemini receives | One 360p frame | Score, stats and new plays as text (plus a frame, if enabled) |
| Facts (names, score) | Guessed from the picture | Taken from ESPN |
| Sports | Any | Every sport in `espn_multisport.SPORTS` (see below) |

Frame sampling is the original code copied over unchanged, so it behaves exactly like `LiveMain.py`.

## How the ESPN mode works

ESPN publishes a free, unofficial JSON feed per game. `espn_multisport/espn_multisport.py` fetches and parses it for every sport into one common shape (teams, score, team stats, a list of plays). For soccer, the older `espn_matchinfo_api/minute_tracker.py` adds two things on top (see [Soccer extras](#soccer-extras)). `LiveMainESPN.py` uses the data in one of these ways:

- **Live team game** (soccer, NBA, NFL, MLB, NHL, ...): checks ESPN every *N* seconds. If there are new plays, a score change, a new period or a stat change (a shot, corner, card, ...), Gemini writes 1–3 sentences about them and the reply streams into the page. Stats that move on every poll (possession %, passes) are shown to Gemini but don't trigger a call by themselves. If nothing happened, no Gemini call is made. When you join mid-game, the first comment covers only the latest 5 plays.
- **Finished team game:** replays the whole play-by-play in small updates of a few plays each. An update never spans two periods, and in low-scoring sports (soccer, hockey, football, baseball) it ends early after a score so every goal gets its own comment. Basketball and Aussie rules score too often for that, so their updates are cut by size only. Handy for testing without a live game.
- **Golf, tennis, MMA, racing:** ESPN has no play-by-play for these, only a leaderboard / draw / fight card / results. The app compares ESPN's summary with the previous poll and asks Gemini about the lines that changed (a new leader, a finished match, a fight result). A finished event, or a finished team game with no play-by-play (common for rugby, lacrosse, volleyball), gets one recap.
- **Not started yet:** waits, checking every *N* seconds.

The prompt tells Gemini to use only the facts given and to call scores with the new score.

### Soccer extras

Soccer is slow and stat-driven, so it gets the extra detail from the old soccer-only tracker (`espn_matchinfo_api/minute_tracker.py`):

- **Live:** the team stats in the prompt are the tracker's fuller set: shots, on target, corners, fouls, offsides, yellow/red cards, possession %, passes, accurate passes, saves, tackles and interceptions.
- **Replays:** ESPN only gives the final box score for a finished match, so the tracker rebuilds running totals (shots, on target, corners, fouls, offsides, cards) minute by minute from the commentary. Each replay update shows the totals as of that moment, under "TEAM STATS (so far)". These match ESPN's final box score. Possession and passing aren't in the commentary, so replays don't have them.

Other sports' replays show the running score but no stats, since only the final box score exists.

## Running it

From this folder:

```bash
python LiveMainESPN.py
```

Then open http://127.0.0.1:7860. Logs go to `logs/run_<timestamp>.log` in the folder you ran it from.

1. Choose **ESPN match data**.
2. Pick a **Sport / league** (e.g. `NBA (nba)`, `English Premier League (epl)`, `PGA Tour (pga)`). For a league that isn't listed, type `sport/league`, e.g. `soccer/tur.1`.
3. Choose the game, in one of three ways:
   - **By name:** type it in **Find a game by name** (e.g. `Warriors vs Rockets`), pick the **Date**, and click **Find game**. This fills in the event id.
   - **By id:** click **List events on date** and paste an id from the list into **ESPN event id**.
   - **Live:** leave everything blank to get the first live game (golf, tennis, MMA and racing: the current event).
4. Click **Start Commentary**, or type a **Timestamp** and click **Commentary at timestamp** (see below).

If both are filled in, the event id wins over the name search.

Known-good finished games for replay: `epl` 401879246 (Liverpool 0–1 Arsenal), `nba` 401909938, `nfl` 401873037, `mlb` 401908046, `nhl` 401892662, `nba` "Lakers vs Warriors" on 2025-01-25.

### Finding a game by name and date

- **Names:** use full names, nicknames, cities or abbreviations, in any order: `Warriors vs Rockets`, `Lakers @ Warriors`, `GSW at LAL`, `Arsenal v Liverpool`, or just `Celtics`. Small typos are fine (`warriers`). For golf, tennis, MMA and racing, type the tournament or card name instead.
- **Date:** blank means today. ESPN files games under the US Eastern date, so if nothing matches on the day picked, the day before and after are searched too.
- **Start time** (optional, e.g. `7:30 PM`): when several games match, such as a doubleheader or a common word like `United`, the one starting closest to this time is picked.
- If nothing matches, the app says so and lists every game on that day. For example, the Warriors and Rockets didn't play each other on Jan 25, 2025: that night was Lakers at Warriors and Rockets at Cavaliers.

### Commentary at a timestamp

Type a moment in the game and click **Commentary at timestamp**. The app cuts the play-by-play off at that moment and makes **one** Gemini call on the update the replay would have shown then: up to **plays per update** of the latest plays, with the score at that point (and running stats for soccer). It uses the same **plays per update** and **key/scoring plays only** settings as a replay. It works for finished games and for live games up to now, but not for golf, tennis, MMA or racing, which have no play-by-play. Stop any running session first.

The timestamp can be either kind; the app works out which:

| Kind | Examples |
|---|---|
| **Game clock**: soccer | `67`, `67'`, `45+2`, `HT`, `FT` |
| **Game clock**: basketball, football, hockey | `Q3 5:12`, `3rd 5:12`, `P2 14:05`, `OT 1:30`, `Q2` (= end of Q2), `HT`, `FT` |
| **Game clock**: baseball | `top 7`, `bottom 7th`, `7th inning` (= end of the 7th), `T7` |
| **Real-world time** | `9:42 PM`, `9:42 PM ET`, `21:42`, `2025-01-25 21:42`, `2026-11-01 17:05Z` |

- A real-world time without a time zone is read in this computer's time zone. Without a date, it's taken as the game's date (or the next day, for a game that ran past midnight).
- A bare `h:mm` like `5:12` is read as real-world time, because it has no quarter or period. Add the period for the game clock: `Q3 5:12`.
- If the time is before the first play, the app says nothing had happened yet.

**Live games.** Timestamps work on a game that's still in progress, as long as the time has passed. Each click fetches the game fresh from ESPN, so earlier moments (`30'` during a match now in its 77th minute, or a real-world time from earlier, like `10:50 AM`) give the plays up to that point, with the score at that moment. A time that hasn't happened yet, or that's beyond ESPN's latest play, gives the latest play there is, and the Status box says so. Plays ESPN hasn't published yet (its feed runs tens of seconds behind) aren't included. Tested on a live Saudi Pro League match on 26 Nov 2026.

**Smaller soccer leagues.** Some leagues (e.g. the Saudi Pro League) only get key events on ESPN, meaning goals, cards and substitutions ("Mourad Batna (Al Fateh) Goal at 70'"), not full commentary. For these:
- **Score:** it's counted from the goal plays. Own goals aren't counted, because ESPN isn't consistent about which team they're listed under, but at the latest play the app uses ESPN's own score, so the current or final score is always right.
- **Stats in replays and timestamps:** the old soccer tracker rebuilds running stats from full commentary wording ("is shown the yellow card", "Attempt saved"), which these leagues don't have. So the app counts the key events instead: yellow cards, red cards and substitutions. Shots, corners, fouls and so on aren't in key events, so they can't be rebuilt for an earlier moment.
- **Live stats are unaffected:** a live game's stats come from ESPN's box score, which has the full set (shots, corners, fouls, possession, ...) even for these leagues.

## Settings

- **Live: seconds between ESPN polls** (default 15): how often a live game is checked. Lower means commentary appears sooner but sends ESPN more requests, and ESPN only updates so often, so going much below 10s gains little. The real gap is a little longer, because the next countdown starts after Gemini's reply.
- **Finished game replay: seconds per update** (default 4): replay speed. The time includes Gemini's reply, so values below Gemini's response time (about 5s when tested) make no difference.
- **Finished game replay: plays per update** (default 4): how many plays Gemini sees per call. Higher means fewer, longer comments and a shorter replay.
- **Replay key/scoring plays only** (off by default): replays just ESPN's key events (soccer: goals, cards, subs; other sports: scoring plays). An NBA replay drops from about 120 updates to about 30.
- **Attach current video frame** (off by default): also sends the latest video frame with each call. The prompt treats ESPN's data as the facts and the picture as atmosphere only (crowd, weather, celebrations), because ESPN's data lags the broadcast and the frame usually shows a later moment. It only makes sense with the live broadcast of the same game. In ESPN mode an uploaded video loops instead of ending the session.

How many updates (Gemini calls) a full replay takes. Each update takes at least ~5s, so 100 updates is roughly 8+ minutes:

| Game | 4 plays per update (default) | 15 plays per update | Key plays only | Note |
|---|---|---|---|---|
| Soccer | ~30 | | ~7 | |
| NFL | ~50 | | ~4 | |
| NHL | ~90 | | ~3 | |
| NBA | ~120 | ~33 | ~30 | Every shot, rebound and foul |
| MLB | ~120 | ~44 | ~4 | Pitch by pitch, so updates are often "Strike 1, Ball 1" |

In soccer, NFL and NHL, updates end at each score, so raising plays per update shortens them less. For NBA and MLB, raise **plays per update** or tick **key/scoring plays only**.

## Latency: what was measured

Measured on 24 Oct 2026 with `gemini-3.5-flash-lite` and minimal thinking:

- **Each Gemini call took about 5s** to the first word, ranging from 0.8s to 7s, with occasional 503 errors. Speed depended on Gemini's load at the time.
- **Text-only calls were no faster than calls with a frame.** Interleaved runs of each both averaged about 5s. Removing the picture doesn't speed up a call.
- **Frame sampling runs about 5–10s behind the video:** up to 5s waiting for the next frame plus about 5s for Gemini. Replies are also held until complete, and a slow round holds up later ones.
- **The ESPN mode is behind the real action by ESPN's publishing delay, plus up to one poll interval, plus about 5s for Gemini.** ESPN's delay hasn't been measured, but is likely tens of seconds.

**So the ESPN mode reduces how often Gemini is called and makes the commentary accurate, but it doesn't lower latency.** Options that could:

1. **Gemini's Live API:** one open connection with a continuous stream of frames, built for real-time video. Not yet tried in this project.
2. **Shorter waits in frame mode:** a frame every 2–3s and replies shown as they stream, at the cost of more calls.
3. **A faster or less loaded model.**

## Limitations

- **ESPN's API is unofficial.** It has no key, no guarantees and no terms covering products built on it, and it can change or be blocked at any time. It's fine for experiments; a product would need a licensed provider (e.g. Sportradar, Stats Perform, Genius Sports, or the leagues directly).
- **Only soccer replays have running stats.** ESPN only gives the final box score for a finished game. Soccer's are rebuilt from the commentary (see [Soccer extras](#soccer-extras)); other sports' replays show the running score only. Live games get stats in every sport that has them.
- **Some sports have no play-by-play.** Rugby, lacrosse, volleyball, field hockey and water polo only have scores by period, so live commentary reacts to score and period changes only. Golf, tennis, MMA and racing react to changes in the leaderboard / draw / card / results.
- **Video must be uploaded.** ESPN's feed contains no broadcast video. Live broadcasts are copy-protected and covered by ESPN's terms, so they shouldn't be captured. The capture code (OpenCV) can also read a webcam, capture card or stream URL, but the page has no field for that yet.
- **The "thinking..." indicator never shows in frame mode.** This bug is inherited unchanged from `LiveMain.py`, where `is_thinking` is set without `global` in `run_commentary_loop`.

## File layout

```
GeminiLiveCommentary/
├── LiveMain.py               original frame-sampling app (unchanged)
├── LiveMainESPN.py           both modes, chosen in the UI
├── README_ESPN.md            this file
├── espn_multisport/
│   └── espn_multisport.py    every sport: event lists, game snapshots, play-by-play
└── espn_matchinfo_api/       older soccer-only feed
    ├── espn_feed.py          soccer match list and summary (used by minute_tracker.py)
    ├── minute_tracker.py     soccer extras: fuller live stat set, running stats for replays
    └── README.md             details of the soccer data
```

`LiveMainESPN.py` adds both folders to Python's import path and imports `espn_multisport` and `minute_tracker` by name (`minute_tracker.py` imports `espn_feed` by bare name). That module also reports bad input (unknown sport, event not found) by raising `SystemExit`, because it doubles as a command-line tool, so `LiveMainESPN.py` turns those into ordinary errors shown in the Status box.
