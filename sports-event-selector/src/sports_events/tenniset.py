"""Lossless TenniSet annotation conversion and exact original-frame extraction.

Publisher: https://github.com/HaydenFaulkner/Tennis
The midpoint triplet is a deterministic derived target, not a human contact label.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path

import cv2
from .io import read_json, write_json

CATEGORIES = {'Serve', 'Hit', 'Point', 'Game', 'Set', 'Match', 'USE', 'SPLITS'}
SOURCE = 'https://github.com/HaydenFaulkner/Tennis'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def frame_triplet(start, end):
    """Three consecutive original frames centered inside the annotated interval."""
    if isinstance(start, bool) or isinstance(end, bool):
        raise ValueError('Frame boundaries must be integers')
    if int(start) != start or int(end) != end or start < 0 or end < start:
        raise ValueError('Invalid original-frame interval')
    start, end = int(start), int(end)
    if end - start < 1:
        raise ValueError('Annotation has fewer than two original frames')
    count = min(3, end - start + 1)
    first = max(start, min((start + end) // 2 - 1, end - count + 1))
    return list(range(first, first + count))


def convert_event(category, annotation):
    if category not in CATEGORIES:
        raise ValueError(f'Unmapped annotation category: {category}')
    custom = annotation.get('custom', {})
    player = custom.get('Player') or 'Player'
    if category == 'Serve':
        label = f'{player} serves'
        if custom.get('Result'): label += f" ({custom['Result'].lower()})"
    elif category == 'Hit':
        detail = ' '.join(str(custom[k]).lower() for k in ('Side', 'Type') if custom.get(k))
        label = f'{player} hits' + (f' a {detail} shot' if detail else ' the ball')
    elif category in {'Point', 'Game', 'Set'}:
        label = f'{category} interval'
        if custom.get('Winner'): label += f"; winner: {custom['Winner']}"
        if custom.get('Score'): label += f"; score: {custom['Score']}"
    elif category == 'Match': label = 'Match interval'
    elif category == 'USE': label = 'Annotated usable video region'
    else: label = f"Publisher split region: {custom.get('Type', 'unspecified')}"
    return {'type': category.lower(), 'label': label,
            'attributes': custom, 'description': annotation.get('desc', ''),
            'role': 'action' if category in {'Serve', 'Hit'} else
                    'metadata' if category in {'USE', 'SPLITS'} else 'context'}


def convert_annotations(annotations, output):
    """Convert every annotation; paths are planned until extract() verifies them."""
    output = Path(output).resolve()
    records = []
    for annotation_file in sorted(Path(annotations).glob('V*.json')):
        video_id = annotation_file.stem
        document = read_json(annotation_file)
        digest = file_hash(annotation_file)
        for category, entries in document['classes'].items():
            for index, annotation in enumerate(entries):
                frames = frame_triplet(annotation['start'], annotation['end'])
                event = convert_event(category, annotation)
                records.append({
                    'id': f'{video_id}_{category.lower()}_{index:05d}',
                    'video_id': video_id,
                    'original_annotation': {'category': category, 'annotation': annotation},
                    'event': event,
                    'image_paths': [str(output / video_id / f'{frame:010d}.png') for frame in frames],
                    'original_frame_numbers': frames,
                    'frame_index_base': 0,
                    'original_interval': [int(annotation['start']), int(annotation['end'])],
                    'training_eligible': event['role'] == 'action',
                    'selection_policy': 'consecutive_original_frames_at_interval_midpoint_v1',
                    'target_provenance': 'human_interval_with_derived_representative_frames',
                    'annotation_file_sha256': digest,
                    'source_repository': SOURCE,
                })
    if not records: raise ValueError('No V*.json annotation files found')
    # A plan must not masquerade as a completed image dataset.
    write_json(output / 'sequences.pending.json', records)
    write_json(output / 'conversion-report.json', {
        'records': len(records), 'by_category': dict(Counter(r['event']['type'] for r in records)),
        'by_video': dict(Counter(r['video_id'] for r in records)),
        'action_records': sum(r['training_eligible'] for r in records),
        'complete': False,
        'note': 'sequences.json is published only after all referenced PNGs are extracted and verified.'})
    return records


def extract(annotations, videos, output, video_ids=None):
    """Decode from frame zero, with no seeking or timestamp rounding.

    grab() advances one decoded source frame; retrieve() materializes wanted images.
    PNG preserves decoded pixels. Per-video receipts permit validated resumptions.
    """
    output = Path(output).resolve()
    records = convert_annotations(annotations, output)
    receipts = {}
    available = {r['video_id'] for r in records}
    if video_ids is not None and not set(video_ids) <= available:
        raise ValueError('Unknown video requested')
    for video_id in sorted(available if video_ids is None else set(video_ids)):
        source = Path(videos).resolve() / f'{video_id}.mp4'
        if not source.is_file(): raise FileNotFoundError(source)
        rows = [r for r in records if r['video_id'] == video_id]
        wanted = {f for row in rows for f in row['original_frame_numbers']}
        directory = output / video_id
        directory.mkdir(exist_ok=True)
        receipt_path = directory / 'extraction.json'
        digest = file_hash(source)
        if receipt_path.exists():
            previous = read_json(receipt_path)
            if previous['source_sha256'] != digest: raise ValueError(f'Source changed: {video_id}')
            if set(map(int, previous['images'])) == wanted and all(
                (directory / f'{f:010d}.png').exists() and
                file_hash(directory / f'{f:010d}.png') == previous['images'][str(f)] for f in wanted):
                receipts[video_id] = previous
                print(f'{video_id}: verified existing {len(wanted)} PNGs', flush=True)
                continue
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened(): raise ValueError(f'Cannot open {source}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if max(wanted) >= count: raise ValueError(f'Annotation exceeds {video_id} frame count')
        images = {}
        try:
            for frame in range(max(wanted) + 1):
                if not capture.grab(): raise ValueError(f'Decode stopped at {video_id}:{frame}')
                if frame in wanted:
                    ok, image = capture.retrieve()
                    if not ok: raise ValueError(f'Cannot retrieve {video_id}:{frame}')
                    path = directory / f'{frame:010d}.png'
                    if not cv2.imwrite(str(path), image): raise IOError(path)
                    # Verify lossless pixel round trip, not just file existence.
                    saved = cv2.imread(str(path))
                    if saved is None or not (saved == image).all(): raise ValueError(f'PNG mismatch {path}')
                    images[str(frame)] = file_hash(path)
                if frame and frame % 25000 == 0: print(f'{video_id}: decoded {frame}/{max(wanted)}', flush=True)
        finally:
            capture.release()
        receipts[video_id] = {'source_video': str(source), 'source_sha256': digest,
            'fps': fps, 'source_frame_count': count, 'frame_index_base': 0,
            'decode_policy': 'sequential_from_zero', 'images': images}
        write_json(receipt_path, receipts[video_id])
        print(f'{video_id}: extracted {len(images)} verified PNGs', flush=True)
    if video_ids is not None:
        print('Requested videos extracted; run without --video to verify and publish the complete dataset.', flush=True)
        return records
    for row in records:
        receipt = receipts[row['video_id']]
        row['source_video'] = receipt['source_video']
        row['source_video_sha256'] = receipt['source_sha256']
        row['source_fps'] = receipt['fps']
        row['image_sha256'] = [receipt['images'][str(f)] for f in row['original_frame_numbers']]
    write_json(output / 'sequences.json', records)
    report = read_json(output / 'conversion-report.json')
    report.update(complete=True, unique_images=sum(len(r['images']) for r in receipts.values()),
                  image_format='lossless PNG', frame_index_base=0)
    write_json(output / 'conversion-report.json', report)
    print(report, flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('convert', 'extract'))
    parser.add_argument('--annotations', default='data/tenniset/annotations')
    parser.add_argument('--videos', default='data/tenniset/videos')
    parser.add_argument('--output', default='data/sequence-frames')
    parser.add_argument('--video', action='append', help='Extract a particular video without publishing a partial dataset')
    args = parser.parse_args()
    if args.command == 'convert': convert_annotations(args.annotations, args.output)
    else: extract(args.annotations, args.videos, args.output, args.video)


if __name__ == '__main__': main()
