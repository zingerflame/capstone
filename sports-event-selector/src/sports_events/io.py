import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_events(events, n, allowed=None):
    if not isinstance(events, list):
        raise ValueError("events must be an array")
    clean = []
    for event in events:
        ids = event.get("frame_indices")
        if not isinstance(ids, list) or len(ids) not in (2, 3):
            raise ValueError("Each event needs exactly 2 or 3 frame indices")
        if any(type(i) is not int or i < 0 or i >= n for i in ids) or ids != sorted(set(ids)):
            raise ValueError("Event indices must be unique, sorted, in-range integers")
        if allowed is not None and not set(ids).issubset(allowed):
            raise ValueError("VLM referenced a frame it was not shown")
        if not isinstance(event.get("label"), str) or not event["label"].strip():
            raise ValueError("Each event needs a text label")
        confidence = event.get("confidence", 1.0)
        if not isinstance(confidence, (float, int)) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        clean.append({**event, "confidence": float(confidence)})
    return clean
