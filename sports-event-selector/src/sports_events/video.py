"""Sequential acquisition and streaming frame-delta sampling."""
import hashlib
import math
import re
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .io import read_json, write_json, fingerprint
from .taxonomy import sport_name


def acquire(row, directory, manifest_dir):
    identifier = row["id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
        raise ValueError("Video IDs may contain only letters, digits, hyphens and underscores")
    destination = Path(directory) / (identifier + ".video")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial")
    source = row["source"]
    if source.startswith(("https://", "http://")):
        with urllib.request.urlopen(source, timeout=120) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
    else:
        path = Path(source)
        shutil.copyfile(path if path.is_absolute() else Path(manifest_dir) / path, partial)
    digest = hashlib.sha256()
    with partial.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if row.get("sha256") and digest.hexdigest() != row["sha256"]:
        raise ValueError(f"Checksum mismatch for {identifier}")
    partial.replace(destination)
    return destination, digest.hexdigest()


def crop_scoreboard(frame, roi):
    if roi is None:
        return np.zeros((32, 96, 3), dtype=np.uint8)
    if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise ValueError("scoreboard_roi must be normalized [left,top,right,bottom]")
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = int(roi[0]*w), int(roi[1]*h), math.ceil(roi[2]*w), math.ceil(roi[3]*h)
    return cv2.resize(frame[y0:y1, x0:x1], (96, 32))


def filtered_frames(capture, min_fps=10.0, max_fps=20.0, delta=0.035):
    """Shared, label-independent full-feed filter. Yield image, source index, PTS.

    Caller owns capture. Endpoint exceptions retain the first/last two originals.
    Read failure before the decoder's advertised frame count is an error.
    """
    if not (0 < min_fps <= max_fps) or not 0 <= delta <= 1:
        raise ValueError("Invalid sampling rates or delta")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Cannot decode video / invalid FPS')
    previous, last_time, tail = None, -1e9, []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if not math.isfinite(timestamp) or timestamp < 0:
            timestamp = tail[-1][2] + 1/fps if tail else 0.0
        if index and timestamp <= tail[-1][2]:
            timestamp = tail[-1][2] + 1/fps
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64,36)).astype(float)/255
        elapsed = timestamp-last_time
        changed = previous is None or np.abs(gray-previous).mean() >= delta
        chosen = index < 2 or elapsed+1e-9 >= 1/min_fps or (elapsed+1e-9 >= 1/max_fps and changed)
        if chosen:
            previous, last_time = gray, timestamp
        # Delay emission by two originals so a previously skipped penultimate
        # frame can be emitted BEFORE an already-selected final frame.
        tail.append((frame,index,timestamp,chosen))
        if len(tail)>2:
            old_frame,old_index,old_time,old_chosen=tail.pop(0)
            if old_chosen:yield old_frame,old_index,old_time
        index += 1
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if expected > 0 and index < expected:
        raise ValueError(f'Decode ended early: {index}/{expected} frames')
    if not tail: raise ValueError('Empty video')
    for frame, source_index, timestamp, _ in tail:
        yield frame, source_index, timestamp


def sample_video(source, output, row, min_fps=10.0, max_fps=20.0, delta=0.035):
    if not (0 < min_fps <= max_fps) or not 0 <= delta <= 1:
        raise ValueError("Invalid sampling rates or delta")
    output = Path(output)
    (output / "frames").mkdir(parents=True, exist_ok=True)
    (output / "scoreboards").mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"Cannot decode video / invalid FPS: {source}")
    records = []
    frame_index = 0
    roi = row.get("scoreboard_roi")

    def save(frame, index, timestamp):
        k = len(records)
        image, board = f"frames/{k:08d}.jpg", f"scoreboards/{k:08d}.jpg"
        # Keep original full-resolution images; resize only when loading for training.
        if not cv2.imwrite(str(output / image), frame) or not cv2.imwrite(str(output / board), crop_scoreboard(frame, roi)):
            raise IOError("Could not write sampled frame")
        records.append({"index": k, "source_frame": index, "timestamp": timestamp,
                        "image": image, "scoreboard": board})

    try:
        for frame, index, timestamp in filtered_frames(capture, min_fps, max_fps, delta):
            save(frame, index, timestamp)
            frame_index = index + 1
    finally:
        capture.release()
    if not records:
        raise ValueError("Empty video")
    clip = {"schema_version": 1, "id": row["id"], "sport": sport_name(row["sport"]),
            "group_id": row.get("group_id", row["id"]), "split": row.get("split", "train"),
            "scoreboard_roi": roi, "source_fps": fps, "source_frame_count": frame_index,
            "sampling": {"min_fps": min_fps, "max_fps": max_fps, "delta": delta},
            "frames": records}
    write_json(output / "clip.json", clip)
    return clip


def prepare(manifest, output, annotate=None, keep_downloads=False):
    """Finish each video's sampling and optional annotation before acquiring the next."""
    manifest = Path(manifest)
    rows = read_json(manifest)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate video IDs")
    output = Path(output)
    completed = []
    for row in rows:
        sport_name(row["sport"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row["id"]):
            raise ValueError("Unsafe video ID")
        clip_dir = output / row["id"]
        marker = clip_dir / "prepared.json"
        key = fingerprint(row)
        if marker.exists() and read_json(marker)["manifest_hash"] == key:
            if annotate:
                annotate(clip_dir)
            completed.append(str(clip_dir))
            continue
        if clip_dir.exists() and any(clip_dir.iterdir()):
            raise ValueError(f"Incomplete or changed clip {clip_dir}; use a fresh output directory")
        source, digest = acquire(row, output / "downloads", manifest.parent)
        sample_video(source, clip_dir, row, **row.get("sampling", {}))
        write_json(marker, {"manifest_hash": key, "source_sha256": digest})
        if annotate:
            annotate(clip_dir)
        if not keep_downloads:
            source.unlink()
        completed.append(str(clip_dir))
    return completed
