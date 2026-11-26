"""Original decoded frame identity, independent of delta filtering and clip numbering."""
import hashlib
import math
import cv2


def video_frames(path, start=0, end=None):
    """Zero-based [start,end). Decode from the beginning; never trust approximate seeks.

    POS_MSEC is the decoder presentation timestamp. FPS-based timing is explicitly
    marked as a fallback. Frame indices always count the original decoded feed.
    """
    if type(start) is not int or start < 0 or (end is not None and (type(end) is not int or end <= start)):
        raise ValueError('Expected nonnegative start and end > start')
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise ValueError('Cannot open video: ' + str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        cap.release(); raise ValueError('Video has no valid FPS')
    index = 0; last_pts = -1.
    try:
        while end is None or index < end:
            if not cap.grab(): break
            if index >= start:
                ok, image = cap.retrieve()
                if not ok: raise ValueError(f'Cannot decode original frame {index}')
                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                valid = math.isfinite(pts) and pts >= 0 and (index == 0 or pts > last_pts) and (index == 0 or pts > 0)
                seconds = pts if valid else max(index / fps,last_pts+1/fps)
                yield {'source_frame': index, 'timestamp': seconds, 'source_fps': fps,
                       'timestamp_basis': 'decoder_pts' if valid else 'frame_index_over_fps',
                       'decoded_sha256': hashlib.sha256(image.tobytes()).hexdigest()}, image
                last_pts = seconds
            index += 1
        expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target=min(expected,end) if end is not None else expected
        if expected>0 and index<target:raise ValueError(f'Decode ended early at original frame {index}, expected {target}')
    finally:
        cap.release()


def frame_key(source_sha256, source_frame):
    if len(source_sha256) != 64 or type(source_frame) is not int or source_frame < 0:
        raise ValueError('Invalid source identity')
    return f'{source_sha256}:{source_frame:010d}'
