"""Freeze timestamped, append-only label-derived commentary before exporting tuning rows."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from .io import read_json, write_json, fingerprint
from .commentary_dataset import sentence, SPLITS, INSTRUCTION


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def timecode(seconds):
    ms=round(seconds*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
    return f'{h:02}:{m:02}:{s:02}.{ms:03}'


def timeline(rows):
    # Commit only at the annotation end: serve outcomes cannot be known at its start.
    ordered=sorted(rows,key=lambda r:(r['video_id'],r['original_interval'][1],r['original_interval'][0],r['id']))
    video=None;plain='';stamped='';chain='';index=0
    for r in ordered:
        if r['video_id']!=video:
            video=r['video_id'];plain='';stamped='';chain='';index=0
        frame=r['original_interval'][1];seconds=frame/r['source_fps']
        new_sentence=sentence(r['event'],index)
        addition=(' ' if plain else '')+new_sentence
        stamped_addition=f'[{timecode(seconds)} | frame {frame}] {new_sentence}\n'
        entry={'id':r['id'],'video_id':video,'split':SPLITS[video],'sequence_index':index,
               'event':r['event'],'original_annotation':r['original_annotation'],
               'original_timestamps':r['original_timestamps'],'original_frame_interval':r['original_interval'],
               'commit_frame':frame,'commit_seconds':seconds,'commit_timecode':timecode(seconds),
               'frame_index_base':0,'new_sentence':new_sentence,'append_text':addition,
               'stamped_append_text':stamped_addition,'previous_text_sha256':digest(plain),
               'text_sha256':digest(plain+addition),'stamped_text_sha256':digest(stamped+stamped_addition),
               'previous_entry_sha256':chain}
        entry['entry_sha256']=fingerprint(entry)
        yield entry
        plain+=addition;stamped+=stamped_addition;chain=entry['entry_sha256'];index+=1


def verify(entries):
    states={};seen=set()
    for e in entries:
        if e['id'] in seen:raise ValueError('Duplicate event')
        seen.add(e['id'])
        plain,stamped,chain,index,last=states.get(e['video_id'],('','','',0,-1))
        unsigned={k:v for k,v in e.items() if k!='entry_sha256'}
        if fingerprint(unsigned)!=e['entry_sha256'] or e['previous_entry_sha256']!=chain:
            raise ValueError('Changed or reordered timeline entry')
        if e['sequence_index']!=index or e['commit_frame']<last:
            raise ValueError('Out-of-order timeline')
        if e['append_text']!=(' ' if plain else '')+e['new_sentence']:
            raise ValueError('Invalid append operation')
        expected=f"[{timecode(e['commit_seconds'])} | frame {e['commit_frame']}] {e['new_sentence']}\n"
        if e['stamped_append_text']!=expected:raise ValueError('Invalid timestamped append')
        if e['previous_text_sha256']!=digest(plain) or e['text_sha256']!=digest(plain+e['append_text']):
            raise ValueError('Earlier commentary was modified')
        plain+=e['append_text'];stamped+=e['stamped_append_text']
        if e['stamped_text_sha256']!=digest(stamped):raise ValueError('Stamped transcript changed')
        states[e['video_id']]=(plain,stamped,e['entry_sha256'],index+1,e['commit_frame'])
    return states


def export_tuning(entries,output,limit=256):
    # Read the frozen log, rather than regenerate its prose.
    verify(entries)
    validation=[e for e in entries if e['split']=='validation']
    selected={e['id'] for e in validation} if len(validation)<=limit else {
        validation[round(i*(len(validation)-1)/(limit-1))]['id'] for i in range(limit)}
    handles={s:(output/f'{s}.vertex.jsonl').open('w',encoding='utf-8') for s in ('train','validation','validation_full','test')}
    histories={};counts=Counter();indices={s:[] for s in handles}
    try:
        for e in entries:
            text,previous=histories.get(e['video_id'],('',[]))
            payload={'current_commentary':text,'prev_event':previous[-2:],'curr_event':e['event']}
            row={'systemInstruction':{'parts':[{'text':INSTRUCTION}]},
                 'contents':[{'role':'user','parts':[{'text':json.dumps(payload,ensure_ascii=False)}]},
                             {'role':'model','parts':[{'text':e['new_sentence']}]}]}
            destinations=[e['split']] if e['split']!='validation' else ['validation_full']+(['validation'] if e['id'] in selected else [])
            for split in destinations:
                handles[split].write(json.dumps(row,ensure_ascii=False)+'\n');counts[split]+=1;indices[split].append(e['id'])
            histories[e['video_id']]=(text+e['append_text'],previous+[e['event']])
    finally:
        for h in handles.values():h.close()
    write_json(output/'line-index.json',indices)
    return dict(counts)


def build(source,exclusions,output):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh version directory; frozen output is never overwritten')
    excluded={r['id'] for r in read_json(exclusions) if r.get('exclude_from_tuning')}
    rows=[r for r in read_json(source) if r['structured_training_eligible'] and r['id'] not in excluded]
    entries=list(timeline(rows));states=verify(entries)
    output.mkdir(parents=True)
    log=output/'ground-truth.jsonl'
    with log.open('w',encoding='utf-8') as f:
        for e in entries:f.write(json.dumps(e,ensure_ascii=False)+'\n')
    # Round-trip the durable timeline before constructing training data.
    frozen=[json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
    for video,(plain,stamped,chain,n,_) in states.items():
        (output/f'{video}.commentary.txt').write_text(stamped,encoding='utf-8')
        (output/f'{video}.plain.txt').write_text(plain,encoding='utf-8')
    counts=export_tuning(frozen,output)
    report={'entries':len(entries),'tuning_rows':counts,'ground_truth_sha256':digest(log.read_text(encoding='utf-8')),
            'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'videos':{v:{'sentences':s[3],'final_entry_sha256':s[2],'plain_sha256':digest(s[0]),'stamped_sha256':digest(s[1])} for v,s in states.items()},
            'provenance':'Synthetic commentary verbalized from human event labels, not a human speech transcript.',
            'ordering':'Original annotation end frame; ties by start frame then source ID.',
            'timestamp_meaning':'Earliest label-complete availability, not measured model or speech latency.',
            'append_only':'Every state is prior text + new text; hash-chain detects changes against this manifest. Files are not write-protected.',
            'excluded_ids':sorted(excluded),'full_history':True,'cloud_calls':0}
    write_json(output/'manifest.json',report);print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='data/tenniset_events_v2/sequences.json')
    p.add_argument('--exclusions',default='data/tenniset_vertex20/label-review.json')
    p.add_argument('--output',default='data/tenniset_commentary_append_only')
    p.add_argument('--verify',action='store_true')
    a=p.parse_args()
    if a.verify:
        out=Path(a.output);log=out/'ground-truth.jsonl';manifest=read_json(out/'manifest.json')
        if digest(log.read_text(encoding='utf-8'))!=manifest['ground_truth_sha256']:raise ValueError('Frozen log changed')
        states=verify([json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()])
        for video,state in states.items():
            if (out/f'{video}.commentary.txt').read_text(encoding='utf-8')!=state[1]:raise ValueError('Transcript differs from log')
            if (out/f'{video}.plain.txt').read_text(encoding='utf-8')!=state[0]:raise ValueError('Plain transcript differs from log')
        print('Verified append-only chains and transcripts for',len(states),'matches')
    else:build(a.source,a.exclusions,a.output)


if __name__=='__main__':main()
