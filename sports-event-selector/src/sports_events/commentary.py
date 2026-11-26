"""Prediction-only vision -> chronological commentary, with a separate label baseline.

Run `python -m sports_events.commentary --help`. No training or checkpoint changes.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .io import read_json, write_json, fingerprint


def clock(seconds):
    milliseconds = round(seconds * 1000)
    h, remainder = divmod(milliseconds, 3600000)
    m, remainder = divmod(remainder, 60000)
    s, ms = divmod(remainder, 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


class Provider:
    """Explicit model IDs; credentials only from environment; responses cached on disk."""
    def __init__(self, provider, model, output):
        self.provider, self.model, self.output = provider, model, Path(output)
        self.key = os.environ.get('OPENAI_API_KEY') if provider == 'gpt' else (
            os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'))
        if not self.key:
            raise ValueError(f'No API key configured for {provider}; ChatGPT sign-in is not an API key.')

    def ask(self, prompt, images=()):
        blobs = [base64.b64encode(Path(p).read_bytes()).decode() for p in images]
        signature = fingerprint({'provider': self.provider, 'model': self.model,
                                 'prompt': prompt, 'images': blobs})
        cache = self.output / 'calls' / (signature + '.json')
        if cache.exists():
            return read_json(cache)['answer']
        if self.provider == 'gpt':
            endpoint = 'https://api.openai.com/v1/responses'
            content = [{'type': 'input_text', 'text': prompt}]
            content += [{'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' + b} for b in blobs]
            body = {'model': self.model, 'store': False,
                    'input': [{'role': 'user', 'content': content}],
                    'text': {'format': {'type': 'json_object'}}, 'max_output_tokens': 6000}
            headers = {'Authorization': 'Bearer ' + self.key}
        else:
            endpoint = 'https://generativelanguage.googleapis.com/v1beta/models/' + urllib.parse.quote(self.model, safe='') + ':generateContent'
            parts = [{'text': prompt}] + [{'inlineData': {'mimeType': 'image/jpeg', 'data': b}} for b in blobs]
            body = {'contents': [{'role': 'user', 'parts': parts}],
                    'generationConfig': {'responseMimeType': 'application/json', 'maxOutputTokens': 6000}}
            headers = {'x-goog-api-key': self.key}
        headers['Content-Type'] = 'application/json'
        started = time.monotonic()
        request = urllib.request.Request(endpoint, json.dumps(body).encode(), headers)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(f'{self.provider} API returned HTTP {error.code}') from None
                time.sleep(2 ** attempt)
        if self.provider == 'gpt':
            if result.get('status') != 'completed':
                raise ValueError('GPT response incomplete or blocked')
            answer = ''.join(part.get('text', '') for item in result.get('output', [])
                             for part in item.get('content', []) if part.get('type') == 'output_text')
        else:
            candidates = result.get('candidates', [])
            if not candidates or candidates[0].get('finishReason') != 'STOP':
                raise ValueError('Gemini response incomplete or blocked')
            answer = ''.join(p.get('text', '') for p in candidates[0]['content']['parts'] if not p.get('thought'))
        answer = json.loads(answer)
        write_json(cache, {'provider': self.provider, 'requested_model': self.model,
                          'returned_model': result.get('model', result.get('modelVersion')),
                          'prompt': prompt, 'image_sha256': [hashlib.sha256(base64.b64decode(b)).hexdigest() for b in blobs],
                          'seconds': time.monotonic() - started,
                          'usage': result.get('usage', result.get('usageMetadata')), 'answer': answer})
        return answer


def prepare(dataset, predictions, video, output):
    groups, reference, clips = [], [], []
    modes=set(); profiles=set(); source_cache={}
    for directory in sorted(Path(dataset).iterdir()):
        if not (directory / 'clip.json').exists():
            continue
        clip = read_json(directory / 'clip.json')
        if clip['split'] != 'test' or clip['group_id'] != 'tenniset_' + video:
            continue
        prediction = read_json(Path(predictions) / (clip['id'] + '.json'))
        mode=prediction.get('output_mode','legacy_frames');modes.add(mode)
        profiles.add(prediction.get('target_profile','center3'))
        frames = clip['frames']
        clips.append({'id': clip['id'], 'start': frames[0]['broadcast_timestamp'],
                      'end': frames[-1]['broadcast_timestamp']})
        def item(ids, identifier):
            if len(ids) not in (2, 3, 20) or ids != sorted(set(ids)) or any(type(i) is not int or i < 0 or i >= len(frames) for i in ids):
                raise ValueError('Invalid selected sequence indices')
            if len(ids)==20 and (mode!='center20' or ids!=list(range(ids[0],ids[0]+20))):
                raise ValueError('Expected consecutive center20 sequence')
            return {'id': identifier, 'clip_id': clip['id'], 'frame_indices': ids,
                    'source_frames': [frames[i]['source_frame'] for i in ids],
                    'start': frames[ids[0]]['broadcast_timestamp'], 'end': frames[ids[-1]]['broadcast_timestamp'],
                    'images': [str((directory / frames[i]['image']).resolve()) for i in ids]}
        for n, ids in enumerate(prediction['event_sequences']):
            row=item(ids, clip['id'] + f'_prediction_{n:03}')
            if mode=='center20':
                row['center_index']=(ids[0]+ids[-1])/2
                row['center_timestamp']=(frames[ids[9]]['broadcast_timestamp']+frames[ids[10]]['broadcast_timestamp'])/2
            groups.append(row)
        # Reference labels never enter the VLM payload or commentary generation.
        annotation=read_json(directory / 'annotations.json')
        for event in annotation['events']:
            row = item(event['frame_indices'], event['sequence_id'])
            row['text'] = event['label']
            if mode=='center20':
                import numpy as np
                source_path=annotation['source_sequences']
                if source_path not in source_cache:
                    source_cache[source_path]={r['id']:r for r in read_json(source_path)}
                original=source_cache[source_path][event['sequence_id']]
                row['center_index']=float(np.interp(sum(original['original_interval'])/2,
                    [f['source_frame'] for f in frames],range(len(frames))))
            reference.append(row)
    if not clips:
        raise ValueError('No matching test video')
    if len(modes)!=1:raise ValueError('Cannot mix legacy and center20 outputs')
    if len(profiles)!=1:raise ValueError('Cannot mix target profiles')
    groups.sort(key=lambda e: (e['start'], e['id']))
    reference.sort(key=lambda e: (e['start'], e['id']))
    for group in groups:
        group['image_sha256'] = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in group['images']]
    manifest = {'video': video, 'output_mode':next(iter(modes)), 'target_profile':next(iter(profiles)), 'scope': 'Existing test intervals only; not the entire match.',
                'clips': clips, 'sequences': groups}
    manifest['fingerprint'] = fingerprint(manifest)
    write_json(Path(output) / 'input.json', manifest)
    write_json(Path(output) / 'reference.json', reference)
    Path(output, 'reference-commentary.txt').write_text('\n'.join(
        f"[{clock(e['start'])}] {e['text']}" for e in reference), encoding='utf-8')
    return manifest, reference


VISION_PROMPT = '''Inspect these chronologically ordered tennis frames from ONE model-selected sequence.
Do not assume selection implies an event. Describe only visible evidence; use near/far player if identity
is uncertain. Do not invent ball contact, spin, score changes, outcomes or action between frames.
Return JSON {"kind":"serve|hit|uncertain|no_event", "text":"short factual description",
"confidence":0.0}. Confidence is subjective, not calibrated. Images may be insufficient to confirm a stroke.'''


def validate_observations(observations, manifest):
    expected = [e['id'] for e in manifest['sequences']]
    if [e.get('id') for e in observations] != expected:
        raise ValueError('Observations must cover every selected sequence exactly once, in order')
    for row in observations:
        if row.get('kind') not in ('serve', 'hit', 'uncertain', 'no_event'):
            raise ValueError('Invalid event kind')
        if not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError('Missing event description')
        if type(row.get('confidence')) not in (int, float) or not 0 <= row['confidence'] <= 1:
            raise ValueError('Invalid confidence')


def sequence_prompt(observations, manifest):
    timeline = [{**o, 'start': s['start'], 'end': s['end'], 'clip_id': s['clip_id']}
                for o, s in zip(observations, manifest['sequences'])]
    return '''Convert ALL these ordered observations into concise timestamped tennis commentary.
Return JSON {"segments":[{"id":"unchanged sequence id","text":"commentary"}]}.
Keep exactly one segment per observation, in the same order, including uncertain/no_event observations.
Preserve uncertainty. Never invent names, shot types, contact, winners, scores or connecting rallies.
Different clip_id values are disconnected excerpts, not a continuous rally. Timestamps are assigned by
the program, so do not invent timestamps. JSON observations: ''' + json.dumps(timeline)


def finish(output, observations, segments, provenance):
    output = Path(output)
    manifest, reference = read_json(output / 'input.json'), read_json(output / 'reference.json')
    for group in manifest['sequences']:
        if group['image_sha256'] != [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in group['images']]:
            raise ValueError('Selected images changed since preparation')
    validate_observations(observations, manifest)
    expected = [e['id'] for e in observations]
    if [e.get('id') for e in segments] != expected or any(not isinstance(e.get('text'), str) or not e['text'].strip() for e in segments):
        raise ValueError('Commentary must preserve every observation exactly once in order')
    timeline = [{**segment, 'start': source['start'], 'end': source['end'], 'clip_id': source['clip_id']}
                for segment, source in zip(segments, manifest['sequences'])]
    write_json(output / 'observations.json', observations)
    write_json(output / 'commentary.json', {'provenance': provenance, 'input_fingerprint': manifest['fingerprint'], 'segments': timeline})
    (output / 'commentary.txt').write_text('\n'.join(f"[{clock(e['start'])}] {e['text']}" for e in timeline), encoding='utf-8')
    # Event-level matching diagnostic, separate from the earlier frame-anchor metric.
    # Maximum-cardinality bipartite matching within each clip, ±3 filtered indices.
    def eligible(pred, truth):
        if manifest.get('output_mode')=='center20':
            radius=9.5 if manifest.get('target_profile')=='window_recall' else 3
            return pred['clip_id']==truth['clip_id'] and abs(pred['center_index']-truth['center_index'])<=radius
        return pred['clip_id'] == truth['clip_id'] and min(abs(a-b) for a in pred['frame_indices'] for b in truth['frame_indices']) <= 3
    def matching(indices, typed=False):
        assigned = {}
        def visit(i, seen):
            for j, truth in enumerate(reference):
                kind = 'serve' if re.search(r'\bserves?\b', truth['text'], re.I) else 'hit'
                if j in seen or not eligible(manifest['sequences'][i], truth) or (typed and observations[i]['kind'] != kind):
                    continue
                seen.add(j)
                if j not in assigned or visit(assigned[j], seen):
                    assigned[j] = i
                    return True
            return False
        for i in indices:
            visit(i, set())
        return [{'prediction_id': manifest['sequences'][i]['id'], 'reference_id': reference[j]['id']} for j, i in sorted(assigned.items())]
    confirmed = [i for i, o in enumerate(observations) if o['kind'] in ('hit', 'serve')]
    matches = matching(confirmed, typed=True)
    comparison = {'selected_sequences': len(expected), 'reference_events': len(reference),
                  'confirmed_vlm_events': len(confirmed), 'uncertain_or_no_event': len(expected)-len(confirmed),
                  'selector_temporal_matches': matching(range(len(expected))),
                  'typed_vlm_matches': matches,
                  'typed_precision': len(matches)/len(confirmed) if confirmed else 0,
                  'typed_recall': len(matches)/len(reference) if reference else 0,
                  'definition': 'One-to-one event matches: same clip, minimum anchor distance <=3 filtered frames, serve/hit class agrees. Not caption correctness or original frame-level scores.',
                  'limitations': 'Derived interval targets, tiny reused test; names/spin and prose factuality require separate review.'}
    write_json(output / 'comparison.json', comparison)
    if manifest.get('output_mode')=='center20':
        radius=9.5 if manifest.get('target_profile')=='window_recall' else 3
        comparison['definition']=f'One-to-one event matches: same clip, center distance <={radius} filtered positions, serve/hit class agrees. Coverage only; decayed localization credit is in the student model results.'
        write_json(output / 'comparison.json', comparison)
    lines = ['# Commentary comparison', '', manifest['scope'], '',
             f"Selected sequences: {len(expected)}. Original-label events: {len(reference)}.", '',
             '## Generated commentary and nearby reference labels', '']
    for source, segment, observation in zip(manifest['sequences'], timeline, observations):
        nearby = [r['text'] for r in reference if eligible(source, r)]
        lines += [f"### {clock(source['start'])} — {source['id']}",
                  f"Generated: {segment['text']}", f"VLM classification: {observation['kind']}",
                  'Nearby labels: ' + ('; '.join(nearby) if nearby else 'None within the evaluation tolerance.'), '']
    lines += ['## Complete original-label timeline', '', (output / 'reference-commentary.txt').read_text(encoding='utf-8')]
    (output / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    return comparison


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run', 'import-review'])
    p.add_argument('--dataset', default='data/tenniset_delta/clips')
    p.add_argument('--predictions', default='runs/tenniset_delta/test/predictions')
    p.add_argument('--video', default='V010')
    p.add_argument('--output', default='runs/tenniset_commentary/V010')
    p.add_argument('--vlm-provider', choices=['gpt', 'gemini'], default='gpt')
    p.add_argument('--llm-provider', choices=['gpt', 'gemini'], default='gpt')
    p.add_argument('--vlm-model')
    p.add_argument('--llm-model')
    p.add_argument('--review', help='ChatGPT-assisted review JSON; not represented as an API run')
    args = p.parse_args()
    if args.command == 'import-review':
        if not args.review:
            p.error('import-review requires --review')
        review = read_json(args.review)
        manifest = read_json(Path(args.output) / 'input.json')
        if review['input_fingerprint'] != manifest['fingerprint']:
            raise ValueError('Review belongs to another input')
        result = finish(args.output, review['observations'], review['segments'],
                        {'mode': 'chatgpt-assisted-review', 'api_calls': 0, 'details': review['provenance']})
        print(json.dumps(result, indent=2))
        return
    if args.command == 'run' and (not args.vlm_model or not args.llm_model):
        p.error('Specify both --vlm-model and --llm-model; no assumed model version')
    manifest, reference = prepare(args.dataset, args.predictions, args.video, args.output)
    if args.command == 'prepare':
        print(json.dumps({'clips': len(manifest['clips']), 'sequences': len(manifest['sequences']), 'reference_events': len(reference)}))
        return
    vision = Provider(args.vlm_provider, args.vlm_model, args.output)
    writer = Provider(args.llm_provider, args.llm_model, args.output)
    observations = []
    for sequence in manifest['sequences']:
        center_instruction=(f" Describe only the action nearest the window center at {sequence['center_timestamp']:.3f} source-video seconds. Other actions may be context."
                            if 'center_timestamp' in sequence else '')
        observation = vision.ask(VISION_PROMPT+center_instruction, sequence['images'])
        observations.append({**observation, 'id': sequence['id']})
    validate_observations(observations, manifest)
    segments = writer.ask(sequence_prompt(observations, manifest))['segments'] if observations else []
    result = finish(args.output, observations, segments, {'mode': 'api', 'vlm': [args.vlm_provider, args.vlm_model], 'llm': [args.llm_provider, args.llm_model]})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
