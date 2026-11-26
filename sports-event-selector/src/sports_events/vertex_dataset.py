"""Stage 20-frame examples and render documented Vertex image-tuning JSONL."""
import argparse
from bisect import bisect_left
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from .io import read_json, write_json, fingerprint

SPLITS = {'V006': 'train', 'V007': 'train', 'V008': 'train', 'V009': 'validation', 'V010': 'test'}
INSTRUCTION = ('Describe the tennis action nearest the specified anchor time, using the 20 ordered '
    'frames and the player reference images. Other actions may appear in the context. Return exactly '
    'one JSON object with name, action_base, action_details, consequence. Use hit or serve as action_base. '
    'Use null for unsupported details/consequences and unknown for an unidentified player. '
    'Do not infer a point winner or score. Player references are appearance context, not additional event frames.')


def nearest_frames(frames, anchor, count=20):
    """Nearest in original-video time for CFR, tie earlier, then chronological."""
    sources = [f['source_frame'] for f in frames]
    if len(sources) < count or sources != sorted(set(sources)):
        raise ValueError('Need a sorted unique filtered feed with at least 20 frames')
    at = bisect_left(sources, anchor)
    candidates = frames[max(0, at-count):min(len(frames), at+count)]
    return sorted(sorted(candidates, key=lambda f: (abs(f['source_frame']-anchor), f['source_frame']))[:count],
                  key=lambda f: f['source_frame'])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def plan(source, feeds, output):
    rows = read_json(source)
    manifests = {v: read_json(Path(feeds)/f'{v}.json') for v in SPLITS}
    result = []
    for r in rows:
        if not r['structured_training_eligible']:
            continue
        feed = manifests[r['video_id']]
        if feed['source_sha256'] != r['source_video_sha256'] or feed['fps'] != r['source_fps']:
            raise ValueError('Source/feed provenance mismatch')
        anchor = sum(r['original_interval']) // 2
        chosen = nearest_frames(feed['frames'], anchor)
        result.append({'id': r['id'], 'video_id': r['video_id'], 'split': SPLITS[r['video_id']],
            'anchor_source_frame': anchor, 'anchor_seconds': anchor/feed['fps'],
            'original_timestamps': r['original_timestamps'], 'output': r['event'],
            'frames': [{**f, 'path': f"assets/frames/{r['video_id']}/{f['source_frame']:010d}.jpg"} for f in chosen]})
    result.sort(key=lambda r: (r['video_id'], r['anchor_source_frame'], r['id']))
    write_json(Path(output)/'examples.json', result)
    write_json(Path(output)/'plan.json', {'examples': len(result), 'frames_per_example': 20,
        'sampling': manifests['V006']['sampling'], 'anchor': 'floor of original annotation midpoint',
        'selection': '20 closest retained original frame indices; ties earlier; chronological output',
        'split_counts': dict(Counter(r['split'] for r in result)),
        'feed_signatures': {v: m['signature'] for v, m in manifests.items()}})
    return rows, manifests, result


def extract_video(video, feed, examples, rows, output):
    output = Path(output)
    wanted = sorted({f['source_frame'] for e in examples if e['video_id'] == video for f in e['frames']})
    signature = fingerprint({'feed': feed['signature'], 'wanted': wanted, 'jpeg_quality': 85, 'max_edge': 768})
    receipt = output/'receipts'/f'{video}.json'
    if receipt.exists():
        prior = read_json(receipt)
        if prior['signature'] != signature:
            raise ValueError('Changed extraction plan; use another output directory')
        if all((output/p).is_file() and sha(output/p) == h for p, h in prior['images'].items()):
            print(video + ': verified cached frames', flush=True)
            return prior
    if sha(feed['source_video']) != feed['source_sha256']:
        raise ValueError('Original video hash mismatch')
    checks = {f: p for r in rows if r['video_id'] == video for f, p in zip(r['original_frame_numbers'], r['image_paths'])}
    blocks = []
    for n in wanted:
        if blocks and n - blocks[-1][1] <= 100:
            blocks[-1][1] = n
        else:
            blocks.append([n, n])
    cap = cv2.VideoCapture(feed['source_video'])
    if not cap.isOpened():
        raise ValueError('Cannot open source video')
    wanted_set = set(wanted); hashes = {}; verified = 0
    try:
        for block_no, (start, end) in enumerate(blocks):
            # Every random-access block must reproduce an already sequentially verified PNG.
            check_frames = [n for n in checks if start <= n <= end]
            if not check_frames:
                raise ValueError('No original-frame verification anchor in extraction block')
            check_at = min(check_frames)
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
                raise ValueError('Seek failed')
            for n in range(start, end+1):
                if round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != n or not cap.grab():
                    raise ValueError('Source frame position mismatch')
                if n not in wanted_set and n != check_at:
                    continue
                ok, im = cap.retrieve()
                if not ok:
                    raise ValueError('Decode failed')
                if n == check_at:
                    original = cv2.imread(checks[n])
                    if original is None or original.shape != im.shape or not np.array_equal(original, im):
                        raise ValueError(f'Seek pixel verification failed: {video}:{n}')
                    verified += 1
                if n in wanted_set:
                    h,w = im.shape[:2]
                    if max(h,w)>768:
                        im=cv2.resize(im,(round(w*768/max(h,w)),round(h*768/max(h,w))),interpolation=cv2.INTER_AREA)
                    rel=f'assets/frames/{video}/{n:010d}.jpg'; destination=output/rel
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    if not cv2.imwrite(str(destination), im, [cv2.IMWRITE_JPEG_QUALITY,85]):
                        raise ValueError('Image write failed')
                    hashes[rel]=sha(destination)
            if block_no % 50 == 0:
                print(f'{video}: {block_no+1}/{len(blocks)} blocks, {len(hashes)}/{len(wanted)} images',flush=True)
    finally:
        cap.release()
    if len(hashes) != len(wanted):
        raise ValueError('Incomplete extraction')
    result={'signature':signature,'video':video,'images':hashes,'verified_seek_blocks':verified,
            'source_sha256':feed['source_sha256']}
    write_json(receipt,result)
    print(f'{video}: complete, {len(hashes)} images',flush=True)
    return result


def image_part(relative, prefix):
    return {'fileData': {'mimeType': 'image/jpeg', 'fileUri': prefix.rstrip('/')+'/'+relative}}


def render(output, references, prefix=None):
    output = Path(output)
    examples = read_json(output/'examples.json')
    refs = read_json(references)
    issues_path=output/'label-review.json'
    issues=read_json(issues_path) if issues_path.exists() else []
    excluded={r['id'] for r in issues if r.get('exclude_from_tuning')}
    validation=[e for e in examples if e['split']=='validation' and e['id'] not in excluded]
    # Conservative 256-example validation file; keep all remaining validation examples separately.
    validation_ids={validation[round(i*(len(validation)-1)/255)]['id'] for i in range(256)} if len(validation)>256 else {e['id'] for e in validation}
    if prefix is not None and not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]{1,220}[a-z0-9](?:/[^\s]*)?', prefix):
        raise ValueError('Supply your actual gs://bucket/prefix')
    prefix = prefix or 'gs://REPLACE_WITH_YOUR_BUCKET/tenniset20'
    preview = 'REPLACE_WITH_YOUR_BUCKET' in prefix
    # Player images must come from the same match; all players included in a fixed order.
    handles = {s: (output / (s+('.template.jsonl' if preview else '.vertex.jsonl'))).open('w',encoding='utf-8')
               for s in ('train','validation','validation_full','test')}
    counts=Counter(); max_images=0; line_index={s:[] for s in handles}
    try:
        for e in examples:
            if e['id'] in excluded:continue
            players = sorted(refs[e['video_id']], key=lambda r:r['name'])
            if not players or len(players)+20 > 30:
                raise ValueError('Need player references and at most 30 total images')
            parts=[{'text': INSTRUCTION + f" Anchor: {e['anchor_seconds']:.3f} seconds in the original video."}]
            for r in players:
                if not r.get('reviewed'):
                    raise ValueError('Unreviewed player reference')
                if sha(output/r['path']) != r['sha256']:
                    raise ValueError('Player reference changed')
                parts += [{'text':'player_information: '+json.dumps({'name':r['name']})}, image_part(r['path'],prefix)]
            for i, f in enumerate(e['frames'],1):
                if not (output/f['path']).is_file():
                    raise ValueError('Missing frame asset')
                parts += [{'text': f"frame{i:02d}: original_frame={f['source_frame']}; timestamp_seconds={f['timestamp']:.3f}"},
                          image_part(f['path'],prefix)]
            record={'contents':[{'role':'user','parts':parts},
                                {'role':'model','parts':[{'text':json.dumps(e['output'],ensure_ascii=False)}]}]}
            validate_record(record)
            serialized=json.dumps(record,ensure_ascii=False)+'\n'
            if e['split']=='validation':
                handles['validation_full'].write(serialized);counts['validation_full']+=1
                line_index['validation_full'].append(e['id'])
            if e['split']!='validation' or e['id'] in validation_ids:
                handles[e['split']].write(serialized);counts[e['split']]+=1
                line_index[e['split']].append(e['id'])
            max_images=max(max_images,len(players)+20)
    finally:
        for h in handles.values(): h.close()
    report={'examples':dict(counts),'images_per_example':max_images,'frame_images':20,
            'excluded_label_ids': sorted(excluded),
            'template_requires_bucket_replacement':preview,'gcs_prefix':prefix,
            'cloud_upload_performed':False,'cloud_validation_performed':False}
    write_json(output/'export-report.json',report)
    write_json(output/'line-index.json',line_index)
    print(json.dumps(report,indent=2))


def validate_record(record):
    if set(record) != {'contents'} or [r['role'] for r in record['contents']] != ['user','model']:
        raise ValueError('Invalid tuning roles')
    parts=record['contents'][0]['parts']
    images=[p for p in parts if 'fileData' in p]
    if not 20 <= len(images) <= 30:
        raise ValueError('Wrong image count')
    for p in parts:
        if set(p) not in ({'text'},{'fileData'}): raise ValueError('Invalid part')
    for p in images:
        if p['fileData']['mimeType']!='image/jpeg' or not p['fileData']['fileUri'].startswith('gs://'):
            raise ValueError('Invalid image URI')
    event=json.loads(record['contents'][1]['parts'][0]['text'])
    if set(event)!={'name','action_base','action_details','consequence'}:
        raise ValueError('Invalid output schema')
    if not isinstance(event['name'],str) or event['action_base'] not in ('hit','serve') or any(
        event[k] is not None and not isinstance(event[k],str) for k in ('action_details','consequence')):
        raise ValueError('Invalid output values')
    numbered=[p['text'].split(':')[0] for p in parts if p.get('text','').startswith('frame')]
    if numbered != [f'frame{i:02d}' for i in range(1,21)]:raise ValueError('Expected exactly 20 ordered frames')
    return event


def validate_export(output):
    output=Path(output)
    report=read_json(output/'export-report.json')
    index=read_json(output/'line-index.json')
    examples={e['id']:e for e in read_json(output/'examples.json')}
    receipts=[read_json(output/'receipts'/f'{v}.json') for v in SPLITS]
    images={p:h for r in receipts for p,h in r['images'].items()}
    for refs in read_json(output/'players.json').values():
        images.update({r['path']:r['sha256'] for r in refs})
    for relative,digest in images.items():
        if sha(output/relative)!=digest:raise ValueError('Asset mismatch: '+relative)
    suffix='.template.jsonl' if report['template_requires_bucket_replacement'] else '.vertex.jsonl'
    verified=0
    for split,ids in index.items():
        count=0
        with (output/(split+suffix)).open(encoding='utf-8') as stream:
            for count,line in enumerate(stream,1):
                record=json.loads(line);event=validate_record(record)
                e=examples[ids[count-1]]
                if event!=e['output']:raise ValueError('Output label mismatch')
                paths=[p['fileData']['fileUri'].removeprefix(report['gcs_prefix'].rstrip('/')+'/')
                       for p in record['contents'][0]['parts'] if 'fileData' in p]
                if len(paths)!=22 or any(p not in images for p in paths):raise ValueError('Invalid asset references')
                if paths[-20:]!=[f['path'] for f in e['frames']]:raise ValueError('Frame ordering mismatch')
                verified+=1
        if count!=len(ids):raise ValueError('Line count mismatch')
    if set(index['train']) & (set(index['validation_full'])|set(index['test'])):
        raise ValueError('Split overlap')
    result={'status':'passed','unique_images_verified':len(images),'jsonl_lines_verified':verified,
            'verified_seek_blocks':sum(r['verified_seek_blocks'] for r in receipts),
            'scope':'Local schema, labels, line order, image hashes and referenced files; no cloud acceptance test.'}
    write_json(output/'verification.json',result);print(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['build','render','validate'])
    p.add_argument('--source',default='data/tenniset_events_v2/sequences.json')
    p.add_argument('--feeds',default='data/tenniset_delta/feeds')
    p.add_argument('--output',default='data/tenniset_vertex20')
    p.add_argument('--references')
    p.add_argument('--gcs-prefix')
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    cv2.setNumThreads(2)
    if args.command=='build':
        rows, feeds, examples=plan(args.source,args.feeds,args.output)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda v:extract_video(v,feeds[v],examples,rows,args.output),feeds))
        print('All 20-frame contexts extracted and checked.',flush=True)
    elif args.command=='render':
        render(args.output,args.references or str(Path(args.output)/'players.json'),args.gcs_prefix)
    else:validate_export(args.output)


if __name__=='__main__':main()
