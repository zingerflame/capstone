"""Credential-free synthetic integration fixture, not sports training data."""
from pathlib import Path
import cv2
import numpy as np
from .io import write_json
from .video import prepare


def create_demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, split in enumerate(("train", "val", "test")):
        video = output / f"synthetic_{split}.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 60, (160, 96))
        if not writer.isOpened():
            raise RuntimeError("MJPG video writer unavailable")
        try:
            for i in range(120):
                image = np.zeros((96, 160, 3), np.uint8)
                image[:] = (25, 75, 25)
                x = 10 + (i * (k+1)) % 140
                cv2.circle(image, (x, 60), 6, (20, 160, 240), -1)
                cv2.putText(image, str(int(i >= 60)), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
                writer.write(image)
        finally:
            writer.release()
        rows.append({"id": f"synthetic_{split}", "source": video.name, "sport": "basketball",
                     "group_id": f"synthetic_match_{k}", "split": split, "scoreboard_roi": [0,0,.4,.3]})
    write_json(output / "manifest.json", rows)
    paths = prepare(output / "manifest.json", output / "dataset")
    from .io import read_json
    for path in paths:
        n = len(read_json(Path(path) / "clip.json")["frames"])
        center = n//2
        write_json(Path(path) / "annotations.json", {"schema_version": 1, "reviewed": True,
            "provenance": {"provider": "synthetic fixture; not VLM or human sports labels"},
            "events": [{"frame_indices": [center-1, center, center+1],
                        "label": "Synthetic moving circle near score change", "confidence": 1.0}]})
    return output / "dataset"
