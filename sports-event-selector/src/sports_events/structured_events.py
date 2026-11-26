"""Export source TenniSet labels as four-field events without inferred outcomes."""
import argparse
from collections import Counter
from pathlib import Path
import json
import math

from .io import read_json, write_json

SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['name', 'action_base', 'action_details', 'consequence'],
    'properties': {'name': {'type': 'string'}, 'action_base': {'type': 'string'},
                   'action_details': {'type': ['string', 'null']},
                   'consequence': {'type': ['string', 'null']}}
}
PROMPT = ('Describe the tennis action in the ordered images. Return only a JSON object '
          'with name, action_base, action_details and consequence. Use action_base hit '
          'or serve. Use "unknown" for an unidentified player. Use null for unsupported '
          'details or consequences. Never infer a point winner or score from a stroke.')


def convert(category, annotation, players):
    custom = annotation.get('custom', {})
    issues = []
    person = custom.get('Player') if category in ('Hit', 'Serve') else custom.get('Winner')
    name = person if person in players else 'unknown'
    if category in ('Hit', 'Serve', 'Point', 'Game', 'Set') and name == 'unknown':
        issues.append('missing_or_invalid_player' if not person else 'invalid_player:' + str(person))
    details = consequence = None
    if category == 'Hit':
        action = 'hit'
        details = ' '.join(str(custom[k]).strip().lower() for k in ('Side', 'Type')
                           if custom.get(k) and custom[k].strip().lower() != 'unsure') or None
        if custom.get('Type') == 'Unsure':
            issues.append('source_shot_type_unsure')
    elif category == 'Serve':
        action = 'serve'
        consequence = {'In': 'in', 'Fault': 'fault', 'Let': 'let'}.get(custom.get('Result'))
        if custom.get('Result') and consequence is None:
            issues.append('unrecognized_serve_result')
    elif category in ('Point', 'Game', 'Set'):
        action = ('win_' if name != 'unknown' else 'complete_') + category.lower()
        score = str(custom.get('Score') or '').strip()
        consequence = 'score: ' + score if score else None
    elif category == 'Match':
        action = 'match_interval'
    elif category == 'USE':
        action = 'usable_region'
    elif category == 'SPLITS':
        action = 'dataset_split'
        details = str(custom.get('Type') or '').lower() or None
    else:
        raise ValueError('Unsupported source category: ' + category)
    return {'name': name, 'action_base': action, 'action_details': details,
            'consequence': consequence}, issues


def source_timestamps(row):
    fps = float(row['source_fps'])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid original video FPS')
    annotation = row['original_annotation']['annotation']
    start, end = annotation['start'], annotation['end']
    if start < 0 or end < start or int(start) != start or int(end) != end:
        raise ValueError('Invalid source interval')
    def timecode(frame):
        ms = round(frame / fps * 1000)
        hours, remainder = divmod(ms, 3600000)
        minutes, remainder = divmod(remainder, 60000)
        seconds, millis = divmod(remainder, 1000)
        return f'{hours:02}:{minutes:02}:{seconds:02}.{millis:03}'
    return {'reference': 'original_video_start', 'method': 'original_frame_index / source_fps',
            'source_fps': fps, 'frame_index_base': 0,
            'start_frame': int(start), 'end_frame': int(end),
            'start_seconds': start / fps, 'end_seconds': end / fps,
            'start_timecode': timecode(start), 'end_timecode': timecode(end),
            'selected_frame_seconds': [f / fps for f in row['original_frame_numbers']],
            'selected_frame_timecodes': [timecode(f) for f in row['original_frame_numbers']]}


def export(source, output):
    source, output = Path(source), Path(output)
    rows = read_json(source)
    players = {r['original_annotation']['annotation'].get('custom', {}).get('Player')
               for r in rows if r['original_annotation']['category'] in ('Hit', 'Serve')}
    players.discard(None); players.discard('')
    converted, splits = [], {'train': [], 'validation': [], 'test': []}
    split_map = {'V006': 'train', 'V007': 'train', 'V008': 'train', 'V009': 'validation', 'V010': 'test'}
    for row in sorted(rows, key=lambda r: (r['video_id'], r['original_interval'][0], r['id'])):
        original = row['original_annotation']
        event, issues = convert(original['category'], original['annotation'], players)
        action = original['category'] in ('Hit', 'Serve')
        eligible = bool(action and row.get('training_eligible') and event['name'] != 'unknown')
        if eligible and (not 2 <= len(row['image_paths']) <= 3 or not all(Path(p).is_file() for p in row['image_paths'])):
            raise ValueError('Missing sequence images: ' + row['id'])
        timestamps = source_timestamps(row)
        record = {**row, 'legacy_event': row['event'], 'event': event, 'original_timestamps': timestamps,
                  'structured_training_eligible': eligible, 'conversion_issues': issues}
        converted.append(record)
        if eligible:
            splits[split_map[row['video_id']]].append({
                'id': row['id'], 'video_id': row['video_id'], 'original_timestamps': timestamps,
                'input': {'instruction': PROMPT, 'image_paths': row['image_paths'],
                          'image_timestamps_seconds': timestamps['selected_frame_seconds']},
                'output': event})
    write_json(output / 'sequences.json', converted)
    write_json(output / 'events.json', [r['event'] for r in converted])
    write_json(output / 'timestamped-events.json', [
        {'id': r['id'], 'video_id': r['video_id'], 'original_timestamps': r['original_timestamps'],
         'event': r['event']} for r in converted])
    write_json(output / 'event.schema.json', SCHEMA)
    for split, examples in splits.items():
        path = output / 'fine_tuning' / (split + '.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in examples), encoding='utf-8')
    report = {'source': str(source.resolve()), 'records': len(converted),
              'by_action': dict(Counter(r['event']['action_base'] for r in converted)),
              'fine_tuning_examples': {k: len(v) for k, v in splits.items()},
              'issues': dict(Counter(i for r in converted for i in r['conversion_issues'])),
              'notes': ['All labels preserved, including administrative labels.',
                        'Fine-tuning manifests contain only eligible serve/hit labels.',
                        'Local provider-neutral manifests, not Gemini upload-ready requests.',
                        'Source identity/outcome labels may not be inferable from sparse frames.',
                        'Scores stay with their own point/game/set label; no inferred stroke outcome.']}
    write_json(output / 'conversion-report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='data/tenniset_delta/sequence-frames/sequences.json')
    p.add_argument('--output', default='data/tenniset_events_v2')
    args = p.parse_args()
    print(json.dumps(export(args.source, args.output), indent=2))


if __name__ == '__main__':
    main()
