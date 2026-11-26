"""Text-only Vertex event-to-commentary tuning from human-derived event labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from .io import read_json, write_json

SPLITS={'V006':'train','V007':'train','V008':'train','V009':'validation','V010':'test'}
INSTRUCTION=('Write only the next short tennis commentary sentence to append to current_commentary. '
    'The input JSON contains current_commentary, prev_event (up to two preceding events, oldest first), '
    'and curr_event. Describe curr_event using only its supported facts. Use the history for continuity; '
    'do not repeat earlier commentary. Preserve the player, shot side, shot type and stated serve result. '
    'Null means unknown: omit it. Do not invent a score, point winner, court placement, emotion or causality. '
    'Return plain text, not JSON. Event history may contain gaps between annotated actions.')


def sentence(event, index=0):
    name=event['name'] if event['name']!='unknown' else 'The player'
    action=event['action_base'];details=event['action_details'];result=event['consequence']
    if action=='serve':
        if details is not None:raise ValueError('Unexpected serve details need a reviewed verbalizer')
        if result=='fault':return f'{name} serves a fault.'
        if result=='let':return f"A let on {name}'s serve."
        if result=='in':return f'{name} serves into court.'
        if result is None:return f'{name} serves.'
        raise ValueError('Unknown serve consequence')
    if action!='hit' or result is not None:raise ValueError('Unsupported action/consequence')
    if details is None:return f'{name} hits the ball.'
    words=details.split()
    if words[0] not in ('forehand','backhand') or len(words)>2:raise ValueError('Unsupported shot details')
    side=words[0];kind=words[1] if len(words)==2 else None
    if kind not in (None,'flat','slice','topspin','volley','smash'):raise ValueError('Unsupported shot type')
    shot=f'{kind} {side}' if kind in ('flat','slice','topspin') else details
    return (f'{name} hits a {shot}.',f'A {shot} from {name}.',f'{name} plays a {shot}.')[index%3]


def records(rows):
    ordered=sorted(rows,key=lambda r:(r['video_id'],sum(r['original_interval'])/2,r['id']))
    video=None;history=[];previous=[];ids=[]
    for r in ordered:
        if r['video_id']!=video:
            video=r['video_id'];history=[];previous=[];ids=[]
        payload={'current_commentary':' '.join(history),'prev_event':previous[-2:], 'curr_event':r['event']}
        target=sentence(r['event'],len(history))
        record={'systemInstruction':{'parts':[{'text':INSTRUCTION}]},
                'contents':[{'role':'user','parts':[{'text':json.dumps(payload,ensure_ascii=False)}]},
                            {'role':'model','parts':[{'text':target}]}]}
        audit={'id':r['id'],'video_id':video,'split':SPLITS[video],
               'previous_event_ids':ids[-2:],'history_event_count':len(history),
               'history_characters':len(payload['current_commentary']),
               'original_timestamps':r['original_timestamps'],'original_annotation':r['original_annotation'],
               'event':r['event'],'target_commentary':target,
               'target_provenance':'deterministic label-derived sentence, not human commentary transcript'}
        yield record,audit
        history.append(target);previous.append(r['event']);ids.append(r['id'])


def build(source, exclusions, output, validation_limit=256):
    if validation_limit<1:raise ValueError('validation_limit must be positive')
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new output directory')
    rows=read_json(source)
    excluded={r['id'] for r in read_json(exclusions) if r.get('exclude_from_tuning')}
    rows=[r for r in rows if r['structured_training_eligible'] and r['id'] not in excluded]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate source IDs')
    if any(r['video_id'] not in SPLITS for r in rows):raise ValueError('Unexpected match')
    val=sorted((r for r in rows if SPLITS[r['video_id']]=='validation'),key=lambda r:(sum(r['original_interval']),r['id']))
    chosen={r['id'] for r in val} if len(val)<=validation_limit else {
        val[round(i*(len(val)-1)/(validation_limit-1))]['id'] for i in range(validation_limit)} if validation_limit>1 else {val[0]['id']}
    output.mkdir(parents=True)
    handles={s:(output/f'{s}.vertex.jsonl').open('w',encoding='utf-8') for s in ('train','validation','validation_full','test')}
    audit=[];counts=Counter();max_chars=Counter();line_ids={s:[] for s in handles}
    try:
        for record,meta in records(rows):
            split=meta['split'];destinations=[split] if split!='validation' else ['validation_full']+(['validation'] if meta['id'] in chosen else [])
            encoded=json.dumps(record,ensure_ascii=False)
            assert [m['role'] for m in record['contents']]==['user','model']
            assert set(json.loads(record['contents'][0]['parts'][0]['text']))=={'current_commentary','prev_event','curr_event'}
            for dest in destinations:
                handles[dest].write(encoded+'\n');counts[dest]+=1;line_ids[dest].append(meta['id'])
                max_chars[dest]=max(max_chars[dest],len(encoded))
            audit.append(meta)
    finally:
        for h in handles.values():h.close()
    write_json(output/'audit.json',audit);write_json(output/'line-index.json',line_ids)
    report={'examples':dict(counts),'split_by_match':SPLITS,'excluded_ids':sorted(excluded),
            'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'maximum_record_characters':dict(max_chars),'history':'All preceding eligible events within the same match, no truncation.',
            'previous_events':'0 at match start, then 1, then 2; oldest first.',
            'target':'Next sentence only, deterministically verbalized from ground truth events.',
            'limitations':['No human commentary transcripts; targets are synthetic and stylistically simple.',
                          'Teacher-forced history; deployment uses generated history and can accumulate errors.',
                          'Sparse annotations are not complete match narration; no invented transitions.',
                          'No noisy VLM predictions used as training ground truth.',
                          'Validation/test matches were previously used in development.',
                          'Full history increases token use; cloud token limits have not been checked.'],
            'images':0,'cloud_calls':0}
    write_json(output/'build-report.json',report);print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='data/tenniset_events_v2/sequences.json')
    p.add_argument('--exclusions',default='data/tenniset_vertex20/label-review.json')
    p.add_argument('--output',default='data/tenniset_commentary_llm')
    p.add_argument('--validation-limit',type=int,default=256)
    a=p.parse_args();build(a.source,a.exclusions,a.output,a.validation_limit)


if __name__=='__main__':main()
