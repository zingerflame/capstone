"""Gemini teacher (default), with optional compatible-endpoint transport."""
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2

from .io import fingerprint, read_json, write_json, validate_events
from .taxonomy import SPORTS

PROMPT_VERSION = "1"


class VLM:
    def __init__(self):
        self.provider = os.environ.get("VLM_PROVIDER", "gemini")
        if self.provider == "gemini":
            self.model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
            self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models/" + self.model + ":generateContent"
            self.key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not self.key:
                raise ValueError("Set GEMINI_API_KEY locally before Gemini annotation; do not put keys in files or chat")
        elif self.provider == "compatible":
            self.endpoint = os.environ.get("VLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
            self.model = os.environ.get("VLM_MODEL")
            if not self.model:
                raise ValueError("Set VLM_MODEL to your deployed vision-language model")
            self.key = os.environ.get("VLM_API_KEY", "")
        else:
            raise ValueError("VLM_PROVIDER must be gemini or compatible")

    def ask(self, prompt, images=(), schema=None):
        content = [{"type": "text", "text": prompt}]
        for caption, path in images:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Unreadable image: {path}")
            h, w = image.shape[:2]
            if max(h, w) > 768:
                image = cv2.resize(image, (round(w*768/max(h,w)), round(h*768/max(h,w))))
            ok, encoded = cv2.imencode(".jpg", image)
            if not ok:
                raise ValueError("JPEG encoding failed")
            content.extend([{"type": "text", "text": caption}, {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode()}}])
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "temperature": 0, "response_format": {"type": "json_object"}}
        headers = {"Content-Type": "application/json"}
        if self.provider == "gemini":
            parts = []
            for item in content:
                if item["type"] == "text":
                    parts.append({"text":item["text"]})
                else:
                    parts.append({"inlineData":{"mimeType":"image/jpeg",
                        "data":item["image_url"]["url"].split(",",1)[1]}})
            config = {"temperature":0, "responseMimeType":"application/json"}
            if schema is not None:
                config["responseJsonSchema"] = schema
            body = {"contents":[{"role":"user", "parts":parts}], "generationConfig":config}
            headers["x-goog-api-key"] = self.key
        elif self.key:
            headers["Authorization"] = "Bearer " + self.key
        encoded_body = json.dumps(body).encode()
        if self.provider == "gemini" and len(encoded_body) >= 19_000_000:
            raise ValueError("Gemini inline request too large; reduce annotation chunk_size")
        request = urllib.request.Request(self.endpoint, encoded_body, headers)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    result = json.load(response)
                if self.provider == "gemini":
                    candidates = result.get("candidates", [])
                    if not candidates or candidates[0].get("finishReason") != "STOP":
                        raise ValueError("Gemini returned no complete answer; inspect quota, safety or output limit settings")
                    answer = "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", [])
                                     if not p.get("thought"))
                else:
                    answer = result["choices"][0]["message"]["content"]
                return json.loads(answer)
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("VLM request failed")


def generate_taxonomy(client, path):
    result = client.ask("Return JSON mapping each of these nine sport keys to a nonempty list of "
        "commentate-able event names. Refine the seed list; include routine passes/hits, not only "
        "scoring highlights. Do not include studio commentary, graphics or crowd shots. " + json.dumps(SPORTS),
        schema={"type":"object", "properties":{s:{"type":"array","items":{"type":"string"},"minItems":1}
                for s in SPORTS}, "required":list(SPORTS), "additionalProperties":False})
    if set(result) != set(SPORTS) or any(not isinstance(v, list) or not v or
        any(not isinstance(x, str) or not x.strip() for x in v) for v in result.values()):
        raise ValueError("VLM taxonomy must contain all nine sports and nonempty string lists")
    write_json(path, result)
    return result


def annotate_clip(directory, client, taxonomy, chunk_size=40, overlap=10):
    if not 0 <= overlap < chunk_size:
        raise ValueError("Require 0 <= overlap < chunk_size")
    directory = Path(directory)
    clip = read_json(directory / "clip.json")
    frames, events = clip["frames"], []
    provenance = {"model": client.model, "endpoint": client.endpoint, "prompt_version": PROMPT_VERSION,
                  "taxonomy": taxonomy[clip["sport"]], "clip": fingerprint(clip),
                  "chunk_size": chunk_size, "overlap": overlap}
    cache_key = fingerprint(provenance)
    existing = directory / "annotations.json"
    if existing.exists():
        saved = read_json(existing)
        if saved.get("provenance") == provenance:
            validate_events(saved.get("events"), len(frames))
            return saved  # Preserve manual corrections and reviewed status on resume.
        raise ValueError("Annotation provenance changed; move existing annotations before regenerating")
    for start in range(0, len(frames), chunk_size - overlap):
        chunk = frames[start:start + chunk_size]
        if len(chunk) < 2:
            continue
        allowed = {f["index"] for f in chunk}
        cache = directory / "vlm_cache" / f"{cache_key}_{start}.json"
        if cache.exists():
            response = read_json(cache)
        else:
            prompt = (f"Analyze these {clip['sport']} broadcast frames in order. Frame indices are "
                "global ZERO-BASED filtered-video indices; timestamps are seconds. Identify discrete "
                "commentate-able sports actions, including routine actions. Return JSON "
                '{"events":[{"frame_indices":[1,2,3],"label":"player passes the ball",'
                '"event_type":"pass","confidence":0.9,"is_replay":false}]}.'
                " Select exactly 2 or 3 distinct shown indices per action (setup/contact/outcome); "
                "never invent indices. Use [] for no events. Do not infer player names without "
                "clear visual evidence. Use scoreboard crops to corroborate score/clock/period "
                "context, never as sole proof of an action. Exclude replays, ads, crowd and studio "
                "shots. Avoid duplicate detections. Taxonomy: " + json.dumps(taxonomy[clip["sport"]]))
            images = []
            for f in chunk:
                caption = f"Frame {f['index']} at {f['timestamp']:.3f}s"
                images.append((caption, directory / f["image"]))
                if clip.get("scoreboard_roi"):
                    images.append((caption + " scoreboard crop", directory / f["scoreboard"]))
            response = client.ask(prompt, images, schema={"type":"object", "properties":{"events":{
                "type":"array", "items":{"type":"object", "properties":{
                    "frame_indices":{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":3},
                    "label":{"type":"string"}, "event_type":{"type":"string"},
                    "confidence":{"type":"number","minimum":0,"maximum":1}, "is_replay":{"type":"boolean"}},
                    "required":["frame_indices","label","event_type","confidence","is_replay"]}}},
                "required":["events"]})
            validate_events(response.get("events"), len(frames), allowed)
            write_json(cache, response)
        validated = validate_events(response.get("events"), len(frames), allowed)
        # Each event belongs to the chunk whose nonoverlap core contains its midpoint.
        # Boundary context is still shown to the teacher.
        end = len(frames) if start + chunk_size >= len(frames) else start + chunk_size - overlap
        events.extend(e for e in validated if start <= e["frame_indices"][len(e["frame_indices"])//2] < end
                      and not e.get("is_replay", False))
        if start + chunk_size >= len(frames):
            break
    unique = {}
    for event in events:
        key = tuple(event["frame_indices"])
        if key not in unique or event["confidence"] > unique[key]["confidence"]:
            unique[key] = event
    result = {"schema_version": 1, "reviewed": False, "provenance": provenance,
              "events": sorted(unique.values(), key=lambda e: e["frame_indices"][0])}
    write_json(directory / "annotations.json", result)
    return result
