"""Budgeted scoreboard reading and frame-aligned context for both generation stages."""
import argparse
import json
from pathlib import Path
import time
from .io import read_json, write_json, fingerprint
from .vertex_dataset import sha

SCORE_PROMPT = '''Transcribe only this cropped sports scoreboard. Return JSON:
{"visible":true,"rows":[{"label":"literal displayed name or country code",
"sets":null,"games":null,"points":null}],"server_row":null,"raw_text":"literal text"}.
Score values are strings or null. server_row is a zero-based row index only if an
unambiguous serving marker is visible; otherwise null. Never expand country codes
into player names, recognize faces, infer unseen scores, or use remembered match
results. If the crop is not legible use visible:false, rows:[], server_row:null.
Do not infer column meanings without a clear tennis scoreboard layout.'''

CONTEXT_INSTRUCTION = (' Broadcast context is observation-only. Use only confirmed player_names '
    'attached to the visible track boxes; appearance associations are unverified. If the acting '
    'track cannot be identified, return name="unknown". Do not identify players by court end, '
    'country code, remembered match knowledge, or older reference images. current_score is '
    'the observed display at its source frame, not the outcome of this action. A score may '
    'lag play. If status is unknown/stale/unreadable, do not state a score. Never infer a winner '
    'from a display update alone. Bounding boxes marked held are approximate.')

COMMENTARY_CONTEXT_INSTRUCTION = (' Use current_score only when status is observed; it describes '
    'the display before/around the event and may lag the action. Never infer a point winner from it. '
    'An unknown event name must remain unnamed: say "the player". Do not recover a name from '
    'earlier commentary, countries, court positions, or match knowledge. Use only the current '
    'event name supported by confirmed visible track bindings. Return only the next commentary sentence.')


def validate_score(value):
    if not isinstance(value,dict) or type(value.get('visible')) is not bool:raise ValueError('Invalid scoreboard response')
    rows=value.get('rows');server=value.get('server_row')
    if not isinstance(rows,list) or len(rows)>4:raise ValueError('Invalid scoreboard rows')
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get('label'),str) or not row['label'].strip():raise ValueError('Invalid scoreboard label')
        for field in ('sets','games','points'):
            if row.get(field) is not None and not isinstance(row[field],str):raise ValueError('Scores must be literal strings or null')
    if server is not None and (type(server) is not int or not 0<=server<len(rows)):raise ValueError('Invalid server row')
    if not value['visible'] and (rows or server is not None):raise ValueError('Unreadable scoreboard must not contain scores')
    if not isinstance(value.get('raw_text'),str):raise ValueError('Missing literal scoreboard text')
    return {k:value[k] for k in ('visible','rows','server_row','raw_text')}


def read_scores(directory, model, project, max_calls=6, location='global'):
    """One crop per change, hard total cache/request budget, no automatic retry spend."""
    from google import genai
    from google.genai import types
    if max_calls<0:raise ValueError('max_calls must be nonnegative')
    directory=Path(directory);candidates=read_json(directory/'score-candidates.json')
    client=None;rows=[];used=0;new_calls=0
    usage_totals={'prompt_token_count':0,'candidates_token_count':0,'thoughts_token_count':0}
    request_seconds=0.
    for candidate in candidates:
        path=directory/candidate['path']
        if sha(path)!=candidate['sha256']:raise ValueError('Scoreboard crop changed')
        identity={'model':model,'project':project,'location':location,'prompt':SCORE_PROMPT,'crop':candidate['sha256']}
        cache=directory/'score-calls'/(fingerprint(identity)+'.json')
        entry={**candidate,'status':'budget_exhausted','score':None}
        if candidate.get('action')=='invalidate':entry['status']='display_changed'
        elif used<max_calls:
            used+=1
            if cache.exists():response=read_json(cache)
            else:
                if client is None:client=genai.Client(vertexai=True,project=project,location=location)
                started=time.monotonic()
                result=client.models.generate_content(model=model,contents=[types.Content(role='user',parts=[
                    types.Part.from_text(text=SCORE_PROMPT),types.Part.from_bytes(data=path.read_bytes(),mime_type='image/jpeg')])],
                    config=types.GenerateContentConfig(temperature=0,max_output_tokens=1024,response_mime_type='application/json'))
                new_calls+=1
                if not result.candidates or 'STOP' not in str(result.candidates[0].finish_reason):raise ValueError('Incomplete scoreboard response')
                response={'answer':validate_score(json.loads(result.text)),
                    'usage':result.usage_metadata.model_dump(mode='json') if result.usage_metadata else {},
                    'seconds':time.monotonic()-started,'identity':identity}
                write_json(cache,response)
            score=validate_score(response['answer']);entry.update(status='observed' if score['visible'] else 'unreadable',score=score)
            for key in usage_totals:usage_totals[key]+=(response.get('usage') or {}).get(key) or 0
            request_seconds+=response['seconds']
        rows.append(entry)
        write_json(directory/'scores.json',rows)
    write_json(directory/'score-usage.json',{'candidates':len(candidates),'max_calls':max_calls,'new_calls':new_calls,
        'selected_crops':used,'model':model,'selected_call_tokens':usage_totals,'selected_call_seconds':request_seconds,
        'note':'Token/time totals count selected cached calls once, not additional charges.'})
    return rows


def score_at(scores,frame,fps,max_age_seconds=10):
    # Latest changed crop invalidates an older observation even when the new crop was not read.
    eligible=[s for s in scores if s['source_frame']<=frame]
    if not eligible:return {'status':'unknown','score':None}
    s=max(eligible,key=lambda r:r['source_frame']);age=(frame-s['source_frame'])/fps
    visually_current=s.get('last_verified_frame',s['source_frame'])>=frame
    status=s['status'] if age<=max_age_seconds or visually_current else 'stale'
    return {'status':status,'source_frame':s['source_frame'],'age_seconds':age,
            'unchanged_display_verified':visually_current,
            'score':s['score'] if status=='observed' else None}


def build_context(directory,video_id,bindings=None,max_score_age=10):
    directory=Path(directory);report=read_json(directory/'report.json');track_list=read_json(directory/'tracks.json')
    tracks={t['id']:t for t in track_list};confirmed={}
    if bindings:
        spec=read_json(bindings)
        if spec.get('source_sha256')!=report['source_sha256']:raise ValueError('Identity binding refers to different source video')
        if spec.get('tracking_report_sha256')!=sha(directory/'report.json'):raise ValueError('Identity binding refers to different tracking run')
        for b in spec.get('bindings',[]):
            if b.get('track_id') not in tracks or not b.get('reviewed') or not b.get('evidence') or not isinstance(b.get('name'),str) or not b['name'].strip():
                raise ValueError('Identity requires a known track, reviewed name, and explicit evidence')
            if b['track_id'] in confirmed:raise ValueError('Duplicate identity binding')
            confirmed[b['track_id']]={'name':b['name'],'evidence':b['evidence'],'status':'confirmed'}
    # No implicit propagation across appearance links or court-end changes.
    scores=read_json(directory/'scores.json') if (directory/'scores.json').exists() else [
        {**c,'status':'not_read','score':None} for c in read_json(directory/'score-candidates.json')]
    frames=[]
    for line in (directory/'frames.jsonl').open(encoding='utf-8'):
        f=json.loads(line)
        objects=[]
        for d in f['objects']:
            if d['kind']!='person':continue
            binding=confirmed.get(d['track_id'])
            objects.append({k:d[k] for k in ('track_id','bbox','moving','observation','box_source_frame','box_age_frames')})
            objects[-1].update(player_name=binding['name'] if binding else None,
                               identity_status='confirmed' if binding else 'unknown')
        frames.append({'source_frame':f['source_frame'],'timestamp':f['timestamp'],'frame_id':f['frame_id'],
            'segment_id':f['segment_id'],'camera_status':f['camera']['status'],
            'width':f.get('width'),'height':f.get('height'),'people':objects})
    result={'schema':'broadcast_context_v1','video_id':video_id,'source_sha256':report['source_sha256'],
        'source_fps':report['source_fps'],'tracking_report_sha256':sha(directory/'report.json'),
        'max_score_age_seconds':max_score_age,'scores':scores,'frames':frames,
        'confirmed_bindings':confirmed,'coordinate_system':'bbox x1,y1,x2,y2 in original video pixels',
        'identity_policy':'Anonymous unless explicitly reviewed; no automatic cross-shot name transfer.'}
    write_json(directory/'context.json',result)
    write_json(directory/'identity-bindings.template.json',{'source_sha256':report['source_sha256'],
        'tracking_report_sha256':sha(directory/'report.json'),'bindings':[
            {'track_id':t['id'],'name':'','reviewed':False,'evidence':'','crop':t.get('crop')}
            for t in track_list if t['kind']=='person' and t['moving_observations']>0]})
    return result


class ContextIndex:
    def __init__(self,path):
        self.path=Path(path);self.hash=sha(path);self.data=read_json(path)
        if self.data.get('schema')!='broadcast_context_v1':raise ValueError('Unknown broadcast context schema')
        self.frames={r['source_frame']:r for r in self.data['frames']}
        if len(self.frames)!=len(self.data['frames']):raise ValueError('Duplicate source frames in context')
        for f,r in self.frames.items():
            if r['frame_id']!=f"{self.data['source_sha256']}:{f:010d}":raise ValueError('Frame identity mismatch')

    def for_example(self,example):
        if example['video_id']!=self.data['video_id']:return None
        if example.get('source_video_sha256') and example['source_video_sha256']!=self.data['source_sha256']:
            raise ValueError('Example belongs to a different original video')
        selected=[]
        for f in example['frames']:
            r=self.frames.get(f['source_frame'])
            if r is None:return None
            if abs(r['timestamp']-f['timestamp'])>1/self.data['source_fps']+.001:raise ValueError('Original frame/time mismatch')
            selected.append(r)
        center=min(selected,key=lambda r:abs(r['timestamp']-example['center_seconds']))
        moving={d['track_id'] for r in selected for d in r['people'] if d['moving'] or d['identity_status']=='confirmed'}
        selected=[{**r,'people':[d for d in r['people'] if d['track_id'] in moving]} for r in selected]
        return {'context_sha256':self.hash,'video_id':example['video_id'],'source_sha256':self.data['source_sha256'],
            'coordinate_system':self.data['coordinate_system'],'frames':selected,
            'current_score':score_at(self.data['scores'],center['source_frame'],self.data['source_fps'],self.data['max_score_age_seconds']),
            'player_names':sorted({d['player_name'] for r in selected for d in r['people'] if d['identity_status']=='confirmed'})}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('read-score');r.add_argument('--tracking',required=True);r.add_argument('--model',required=True)
    r.add_argument('--project',required=True);r.add_argument('--max-calls',type=int,default=6);r.add_argument('--location',default='global')
    b=sub.add_parser('build');b.add_argument('--tracking',required=True);b.add_argument('--video-id',required=True);b.add_argument('--bindings')
    a=p.parse_args()
    if a.command=='read-score':read_scores(a.tracking,a.model,a.project,a.max_calls,a.location)
    else:build_context(a.tracking,a.video_id,a.bindings)


if __name__=='__main__':main()
