"""Build Vertex tuning data from frozen selector outputs and evaluate a tuned endpoint."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import time

from .io import read_json, write_json, fingerprint
from .vertex_dataset import sha, validate_record, SPLITS

PROMPT = ('Describe the tennis action nearest the predicted center time in the 20 ordered frames. '
          'Return one JSON object with name, action_base (hit or serve), action_details, consequence. '
          'Use null for unsupported details or consequences and unknown for an unidentified player. '
          'If no action is identifiable return JSON null. Do not invent a winner or score. '
          'Player reference images are appearance context, not event frames.')
FIELDS = ('name', 'action_base', 'action_details', 'consequence')


def choose_target(frames, labels):
    """Nearest human midpoint INSIDE the predicted window; ties choose earlier/id."""
    lo, hi = frames[0]['source_frame'], frames[-1]['source_frame']
    center = (frames[9]['source_frame'] + frames[10]['source_frame']) / 2
    eligible = [r for r in labels if lo <= sum(r['original_interval']) / 2 <= hi]
    return min(eligible, key=lambda r: (abs(sum(r['original_interval'])/2-center),
                                       sum(r['original_interval']), r['id'])) if eligible else None


def build(args):
    import torch
    from .center20 import predict
    from .learning import discover
    from .pretrained import FeatureSelector, load_encoder, extract_features
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory; existing exports are never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if state.get('output_mode') != 'center20' or state.get('target_profile') != 'window_recall':
        raise ValueError('Expected the recall-focused center20 checkpoint')
    model = FeatureSelector(motion=state['motion'], width=state['width'])
    model.load_state_dict(state['model']); model.eval()
    encoder = load_encoder(args.weights)
    source = read_json(args.labels)
    ref_root = Path(args.references)
    refs = read_json(ref_root/'players.json')
    excluded = {r['id'] for r in read_json(ref_root/'label-review.json') if r.get('exclude_from_tuning')}
    labels = [r for r in source if r['structured_training_eligible'] and r['id'] not in excluded]
    assets = {}; examples = []; truths = {}; counts = Counter()

    def stage(src, rel):
        digest = sha(src)
        if rel not in assets:
            dest = output/rel; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            if sha(dest) != digest: raise ValueError('Copy verification failed')
            assets[rel] = digest
        elif assets[rel] != digest: raise ValueError('Asset collision')
        return rel

    for directory, clip, _ in discover(args.dataset):
        video = clip['group_id'].removeprefix('tenniset_')
        split = 'validation' if clip['split'] == 'val' else clip['split']
        if SPLITS.get(video) != split: raise ValueError('Unexpected match split')
        targets = [r for r in labels if r['video_id'] == video]
        frames = clip['frames']
        for r in targets:
            midpoint = sum(r['original_interval'])/2
            if frames[0]['source_frame'] <= midpoint <= frames[-1]['source_frame']:
                truths[r['id']] = {'id': r['id'], 'split': split, 'event': r['event'],
                                   'original_timestamps': r['original_timestamps']}
        result = predict(model, state, extract_features(directory, clip, encoder, args.cache), clip)
        write_json(output/'predictions'/(clip['id']+'.json'), result)
        players = []
        for ref in sorted(refs[video], key=lambda r:r['name']):
            if not ref.get('reviewed') or sha(ref_root/ref['path']) != ref['sha256']:
                raise ValueError('Unreviewed or changed reference')
            players.append({'name': ref['name'], 'path': stage(ref_root/ref['path'], ref['path'])})
        for i, indices in enumerate(result['event_sequences']):
            if len(indices) != 20 or indices != list(range(indices[0], indices[0]+20)):
                raise ValueError('Expected exactly 20 consecutive frames')
            chosen = [frames[j] for j in indices]
            if any(b['filtered_index_global'] != a['filtered_index_global']+1 for a,b in zip(chosen,chosen[1:])):
                raise ValueError('Window skips delta-filtered frames')
            target = choose_target(chosen, targets)
            row = {'id': f"{clip['id']}_{i:03d}", 'clip_id': clip['id'], 'video_id': video,
                   'split': split, 'center_seconds': result['center_timestamps_seconds'][i],
                   'players': players, 'frames': [], 'target_id': target['id'] if target else None,
                   'output': target['event'] if target else None,
                   'original_timestamps': target['original_timestamps'] if target else None,
                   'original_annotation': target['original_annotation'] if target else None}
            for f in chosen:
                rel = f"assets/frames/{video}/{f['source_frame']:010d}.jpg"
                row['frames'].append({'path': stage(directory/f['image'],rel),
                    'source_frame':f['source_frame'],'filtered_index_global':f['filtered_index_global'],
                    'timestamp':f['broadcast_timestamp']})
            examples.append(row); counts[split+'_windows'] += 1
            counts[split+('_matched' if target else '_unmatched')] += 1
        print(f"{clip['id']}: {len(result['event_sequences'])} windows", flush=True)
    write_json(output/'examples.json', examples)
    write_json(output/'truths.json', list(truths.values()))
    write_json(output/'assets.json', assets)
    report = {'counts':dict(counts), 'checkpoint_sha256':sha(args.checkpoint),
              'labels_sha256':sha(args.labels),'excluded_labels':sorted(excluded),
              'scope':'Existing pilot clips, not full matches. Training predictions are in-sample.',
              'matching':'Nearest original midpoint inside predicted window; target never enters input.',
              'unmatched_policy':'Excluded from tuning; retained as unscored test/audit windows.',
              'limitations':'Validation match also selected DL checkpoint; V010 is reused development test.',
              'distinct_target_counts':{s:len({e['target_id'] for e in examples if e['split']==s and e['target_id']}) for s in SPLITS.values()},
              'truth_counts':dict(Counter(r['split'] for r in truths.values())),
              'cloud_calls_performed':False}
    write_json(output/'build-report.json',report)
    render(output,args.gcs_prefix)
    print(json.dumps(report,indent=2))


def input_parts(e, prefix):
    parts = [{'text':PROMPT+f" Predicted center: {e['center_seconds']:.3f} seconds."}]
    for r in e['players']:
        parts += [{'text':'player_information: '+json.dumps({'name':r['name']})},
                  {'fileData':{'mimeType':'image/jpeg','fileUri':prefix+'/'+r['path']}}]
    for i,f in enumerate(e['frames'],1):
        parts += [{'text':f"frame{i:02d}: original_frame={f['source_frame']}; timestamp_seconds={f['timestamp']:.3f}"},
                  {'fileData':{'mimeType':'image/jpeg','fileUri':prefix+'/'+f['path']}}]
    return parts


def render(output, prefix=None):
    output = Path(output)
    if prefix and not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+(?:/[^\s]*)?',prefix):
        raise ValueError('Provide a valid gs://bucket/prefix')
    prefix = (prefix or 'gs://REPLACE_WITH_YOUR_BUCKET/selector-vlm').rstrip('/')
    suffix = 'template' if 'REPLACE_WITH_YOUR_BUCKET' in prefix else 'vertex'
    examples = read_json(output/'examples.json'); counts = Counter(); index = {}
    for rel,digest in read_json(output/'assets.json').items():
        if sha(output/rel)!=digest: raise ValueError('Asset hash mismatch: '+rel)
    for split in ('train','validation','test'):
        index[split] = []
        with (output/f'{split}.{suffix}.jsonl').open('w',encoding='utf-8') as f:
            for e in examples:
                if e['split']!=split or not e['target_id']:continue
                record={'contents':[{'role':'user','parts':input_parts(e,prefix)},
                                    {'role':'model','parts':[{'text':json.dumps(e['output'])}]}]}
                validate_record(record)
                f.write(json.dumps(record)+'\n'); counts[split]+=1; index[split].append(e['id'])
    # All proposed test windows, including unmatched ones; no answer or target timestamp.
    with (output/f'test-inputs.{suffix}.jsonl').open('w',encoding='utf-8') as f:
        for e in examples:
            if e['split']=='test':
                f.write(json.dumps({'id':e['id'],'contents':[{'role':'user','parts':input_parts(e,prefix)}]})+'\n')
    write_json(output/'line-index.json',index)
    write_json(output/'export-report.json',{'tuning_rows':dict(counts),'gcs_prefix':prefix,
               'requires_bucket':suffix=='template','assets_verified':len(read_json(output/'assets.json'))})


def valid_answer(answer):
    return answer is None or (isinstance(answer,dict) and set(answer)==set(FIELDS)
        and isinstance(answer['name'],str) and answer['action_base'] in ('hit','serve')
        and all(answer[k] is None or isinstance(answer[k],str) for k in FIELDS[2:]))


def score(examples, predictions, truths):
    labeled=[e for e in examples if e['target_id']]
    correct=Counter(); exact=0; recovered=set(); invalid=0
    def norm(v): return v.strip().casefold() if isinstance(v,str) else v
    for e in examples:
        a=predictions.get(e['id'],{}).get('answer',{})
        valid=valid_answer(a); invalid+=not valid
        if not e['target_id']:continue
        fields={k:valid and isinstance(a,dict) and norm(a[k])==norm(e['output'][k]) for k in FIELDS}
        correct.update(k for k,v in fields.items() if v)
        if all(fields.values()): exact+=1; recovered.add(e['target_id'])
    n=len(labeled)
    return {'windows':len(examples),'labeled_windows':n,'unscored_windows':len(examples)-n,
            'invalid_or_missing_outputs':invalid,'exact_event_accuracy':exact/n if n else None,
            'field_accuracy':{k:correct[k]/n if n else None for k in FIELDS},
            'distinct_exact_events':len(recovered),'reference_events':len(truths),
            'end_to_end_exact_event_recall':len(recovered)/len(truths) if truths else None,
            'note':'Exact normalized string matching, not semantic judging. Unmatched windows are unscored, not proven negatives.'}


def evaluate(args):
    output=Path(args.output); dataset=Path(args.dataset)
    examples=[e for e in read_json(dataset/'examples.json') if e['split']=='test']
    if args.limit is not None:
        if args.limit<1:raise ValueError('limit must be positive')
        examples=examples[:args.limit]
    assets=read_json(dataset/'assets.json')
    for e in examples:
        for item in e['players']+e['frames']:
            if sha(dataset/item['path'])!=assets[item['path']]:raise ValueError('Changed image asset')
    if args.dry_run:
        report={'dry_run':True,'model':args.model,'windows':len(examples),
                'images':sum(len(e['players'])+20 for e in examples),'api_calls':0}
        write_json(output/'dry-run.json',report);print(json.dumps(report,indent=2));return
    if not args.project:raise ValueError('--project is required for Vertex inference')
    from google import genai
    from google.genai import types
    client=genai.Client(vertexai=True,project=args.project,location=args.location)
    predictions={}
    for e in examples:
        raw=input_parts(e,'local:')
        signature=fingerprint({'model':args.model,'project':args.project,'location':args.location,
                               'parts':raw,'hashes':[assets[x['path']] for x in e['players']+e['frames']]})
        cache=output/'calls'/(signature+'.json')
        if cache.exists():result=read_json(cache)
        else:
            parts=[]
            for part in raw:
                if 'text' in part:parts.append(types.Part.from_text(text=part['text']))
                else:
                    rel=part['fileData']['fileUri'].removeprefix('local:/')
                    parts.append(types.Part.from_bytes(data=(dataset/rel).read_bytes(),mime_type='image/jpeg'))
            started=time.monotonic()
            response=client.models.generate_content(model=args.model,
                contents=[types.Content(role='user',parts=parts)],
                config=types.GenerateContentConfig(temperature=0,response_mime_type='application/json',max_output_tokens=2048))
            text=response.text or ''
            try:answer=json.loads(text)
            except json.JSONDecodeError:answer={}
            result={'answer':answer,'raw_text':text,'model':args.model,'seconds':time.monotonic()-started,
                    'usage':response.usage_metadata.model_dump(mode='json') if response.usage_metadata else None}
            write_json(cache,result)
        predictions[e['id']]=result
        write_json(output/'predictions.json',predictions)
        print(e['id'],flush=True)
    truths=[r for r in read_json(dataset/'truths.json') if r['split']=='test']
    report=score(examples,predictions,truths)
    report.update({'model':args.model,'partial_run':args.limit is not None,
                   'dataset_report_sha256':sha(dataset/'build-report.json')})
    write_json(output/'metrics.json',report);print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build')
    b.add_argument('--dataset',default='data/tenniset_delta/clips')
    b.add_argument('--checkpoint',default='runs/tenniset_center20_recall/model.pt')
    b.add_argument('--labels',default='data/tenniset_events_v2/sequences.json')
    b.add_argument('--references',default='data/tenniset_vertex20')
    b.add_argument('--weights',default='data/pretrained/resnet18-f37072fd.pth')
    b.add_argument('--cache',default='data/tenniset_delta_features')
    b.add_argument('--output',default='data/tenniset_selector_vlm');b.add_argument('--gcs-prefix')
    r=sub.add_parser('render');r.add_argument('--dataset',default='data/tenniset_selector_vlm');r.add_argument('--gcs-prefix',required=True)
    e=sub.add_parser('evaluate');e.add_argument('--dataset',default='data/tenniset_selector_vlm')
    e.add_argument('--model',required=True);e.add_argument('--project');e.add_argument('--location',default='us-central1')
    e.add_argument('--output',required=True);e.add_argument('--limit',type=int);e.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.command=='build':build(args)
    elif args.command=='render':render(args.dataset,args.gcs_prefix)
    else:evaluate(args)


if __name__=='__main__':main()
