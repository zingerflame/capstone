"""Seed vocabulary; the taxonomy command asks the configured VLM to refine it."""
SPORTS = {
    "tennis": ["serve", "return", "forehand", "backhand", "volley", "smash", "ace", "double fault", "winner", "point won"],
    "baseball": ["pitch", "swing", "hit", "strikeout", "walk", "catch", "throw", "stolen base", "tag out", "home run"],
    "golf": ["tee shot", "approach shot", "chip", "bunker shot", "putt", "ball holed"],
    "badminton": ["serve", "return", "clear", "drop shot", "smash", "drive", "net shot", "rally won"],
    "basketball": ["pass", "dribble", "layup", "dunk", "jump shot", "three-point shot", "rebound", "steal", "block", "free throw", "turnover"],
    "soccer": ["pass", "cross", "dribble", "tackle", "interception", "shot", "save", "goal", "corner", "free kick", "penalty"],
    "american_football": ["snap", "handoff", "pass", "reception", "rush", "tackle", "sack", "interception", "fumble", "touchdown", "field goal", "punt"],
    "rugby": ["pass", "carry", "tackle", "ruck", "maul", "scrum", "lineout", "kick", "try", "conversion", "penalty kick"],
    "table_tennis": ["serve", "return", "drive", "loop", "push", "chop", "block", "smash", "rally won"],
}


def sport_name(value):
    value = value.lower().replace(" ", "_").replace("-", "_")
    value = {"batminton": "badminton", "pingpong": "table_tennis", "ping_pong": "table_tennis"}.get(value, value)
    if value not in SPORTS:
        raise ValueError(f"Unknown sport: {value}; choose from {list(SPORTS)}")
    return value
