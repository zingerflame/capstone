import json
import cv2
import numpy as np
import pytest
from sports_events.broadcast_tracking import camera_step,stable_segments,Tracker,associate,ScoreGate
from sports_events.broadcast_context import score_at,validate_score,ContextIndex,build_context
from sports_events.frame_identity import video_frames,frame_key
from sports_events.io import write_json
from sports_events.selector_vlm import input_parts


def test_camera_distinguishes_translation_from_local_motion():
    rng=np.random.default_rng(42);a=rng.integers(0,255,(240,320),dtype=np.uint8)
    static=a.copy();static[100:115,150:165]=255
    still=camera_step(a,static)
    moved=cv2.warpAffine(a,np.float32([[1,0,5],[0,1,3]]),(320,240))
    motion=camera_step(a,moved)
    assert still['status']=='estimated' and still['step_pixels']<.2
    assert motion['status']=='estimated' and motion['step_pixels']>5
    assert camera_step(np.zeros_like(a),np.zeros_like(a))['status']=='unknown'


def test_stable_spans_never_bridge_unknown_frames():
    rows=[{'source_frame':10+i,'segment_id':None,'camera':{'status':s}} for i,s in enumerate(
        ['stable']*4+['unknown']+['stable']*2+['moving']+['stable']*4)]
    spans=stable_segments(rows,10,.3)
    assert [(s['start_frame'],s['end_frame_exclusive']) for s in spans]==[(10,14),(18,22)]
    assert rows[5]['segment_id'] is None


def test_tracking_is_one_to_one_and_does_not_cross_camera_segments():
    tracker=Tracker(10);features=[np.array([1.,0.]),np.array([0.,1.])]
    ds=[{'bbox':[0,0,10,20],'kind':'person','moving':True}, {'bbox':[30,0,40,20],'kind':'person','moving':False}]
    a=tracker.update(ds,features,0,'shot0');b=tracker.update(ds,features,1,'shot0')
    assert [x['track_id'] for x in a]==[x['track_id'] for x in b]
    c=tracker.update(ds,features,2,'shot1')
    assert set(x['track_id'] for x in a).isdisjoint(x['track_id'] for x in c)
    links=associate(list(tracker.tracks.values()),.8,.1)
    assert not links  # new tracks have only one observation
    tracker.update(ds,features,3,'shot1');links=associate(list(tracker.tracks.values()),.8,.1)
    assert len(links)==2 and all(not l['name_transfer_allowed'] for l in links)


def test_appearance_ambiguity_is_rejected():
    tracks=[{'id':str(i),'segment_id':'a' if i==0 else 'b','samples':3,'kind':'person','feature':np.array([1.,0.])} for i in range(3)]
    assert associate(tracks,.8,.1)==[]


def test_exact_original_frame_indices_and_pts(tmp_path):
    path=tmp_path/'test.avi';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48))
    assert writer.isOpened()
    for i in range(12):writer.write(np.full((48,64,3),i*15,np.uint8))
    writer.release();frames=list(video_frames(path,5,9))
    assert [f['source_frame'] for f,_ in frames]==[5,6,7,8]
    assert all(abs(f['timestamp']-f['source_frame']/10)<.001 for f,_ in frames)
    assert abs(frames[0][1].mean()-75)<3
    assert frame_key('a'*64,5).endswith(':0000000005')


def test_score_gate_invalidates_before_debounced_read():
    gate=ScoreGate(5,settle_frames=3);a=np.zeros((10,10),np.uint8);b=a.copy();b[:2]=255
    assert gate.update(a,0)=='read'
    assert gate.update(a,1)=='unchanged'
    assert gate.update(b,2)=='invalidate'
    assert gate.update(b,3)=='pending'
    assert gate.update(b,5)=='read'


def test_scores_never_look_ahead_or_reuse_invalidated_display():
    rows=[{'source_frame':10,'status':'observed','score':{'x':1},'last_verified_frame':100},
          {'source_frame':101,'status':'display_changed','score':None}]
    assert score_at(rows,9,10)['score'] is None
    assert score_at(rows,99,10,1)['score']=={'x':1}  # continuous local visual verification
    assert score_at(rows,101,10)['score'] is None
    assert score_at(rows[:1],200,10,1)['status']=='stale'
    with pytest.raises(ValueError):validate_score({'visible':True,'rows':[],'server_row':True,'raw_text':''})


def test_context_prompt_omits_old_named_references_and_labels():
    e={'center_seconds':1.,'players':[{'name':'WRONG_OLD_PLAYER','path':'ref.jpg'}],
        'frames':[{'source_frame':10,'timestamp':1.,'path':'frame.jpg'}],
        'output':{'name':'SECRET_TRUTH'}}
    context={'player_names':[],'current_score':{'status':'unknown'}}
    parts=input_parts(e,'local:',context);text=json.dumps(parts)
    assert 'WRONG_OLD_PLAYER' not in text and 'SECRET_TRUTH' not in text
    assert 'current_score' in text and len([p for p in parts if 'fileData' in p])==1


def test_context_requires_exact_source_time_and_frame_id(tmp_path):
    p=tmp_path/'context.json';source='a'*64
    data={'schema':'broadcast_context_v1','source_sha256':source,'video_id':'V010','source_fps':25.,
        'coordinate_system':'pixels','max_score_age_seconds':10,'scores':[],
        'frames':[{'source_frame':100,'timestamp':4.,'frame_id':frame_key(source,100),'people':[]}]}
    write_json(p,data);index=ContextIndex(p)
    e={'video_id':'V010','center_seconds':4.,'frames':[{'source_frame':100,'timestamp':4.}]}
    assert index.for_example(e)['current_score']['score'] is None
    assert index.for_example({**e,'video_id':'OTHER'}) is None
    with pytest.raises(ValueError):index.for_example({**e,'frames':[{'source_frame':100,'timestamp':5.}]})
    data['frames'][0]['frame_id']='different';write_json(p,data)
    with pytest.raises(ValueError):ContextIndex(p)


def test_named_binding_requires_review_and_run_provenance(tmp_path):
    from sports_events.vertex_dataset import sha
    source='b'*64
    write_json(tmp_path/'report.json',{'source_sha256':source,'source_fps':25.})
    write_json(tmp_path/'tracks.json',[{'id':'t','kind':'person','moving_observations':1}])
    write_json(tmp_path/'score-candidates.json',[])
    (tmp_path/'frames.jsonl').write_text('')
    spec={'source_sha256':source,'tracking_report_sha256':sha(tmp_path/'report.json'),
          'bindings':[{'track_id':'t','name':'Player','reviewed':False,'evidence':'source lineup'}]}
    p=tmp_path/'bindings.json';write_json(p,spec)
    with pytest.raises(ValueError):build_context(tmp_path,'V010',p)
    spec['bindings'][0]['reviewed']=True;write_json(p,spec)
    assert build_context(tmp_path,'V010',p)['confirmed_bindings']['t']['name']=='Player'
    spec['source_sha256']='c'*64;write_json(p,spec)
    with pytest.raises(ValueError):build_context(tmp_path,'V010',p)


def test_llm_receives_score_and_rejects_old_vlm_cache(tmp_path,monkeypatch):
    from sports_events import end_to_end_eval as module
    from sports_events.vertex_dataset import sha
    source='a'*64;p=tmp_path/'context.json'
    write_json(p,{'schema':'broadcast_context_v1','source_sha256':source,'video_id':'V010','source_fps':25.,
        'coordinate_system':'pixels','max_score_age_seconds':10,
        'scores':[{'source_frame':100,'status':'observed','score':{'points':'15'}}],
        'frames':[{'source_frame':100,'timestamp':4.,'frame_id':frame_key(source,100),'people':[]}]})
    dataset=tmp_path/'dataset';vlm=tmp_path/'vlm';out=tmp_path/'out';out.mkdir()
    write_json(dataset/'examples.json',[{'id':'e','clip_id':'c','split':'test','video_id':'V010','center_seconds':4.,
        'frames':[{'source_frame':100,'timestamp':4.}]}])
    event={'name':'unknown','action_base':'hit','action_details':None,'consequence':None}
    write_json(vlm/'predictions.json',{'e':{'answer':event,'context_sha256':sha(p)}})
    calls=[]
    class Fake:
        def __init__(self,c):pass
        def generate(self,model,prompt,**kw):calls.append(json.loads(prompt));return {'text':'The player hits the ball.'}
    monkeypatch.setattr(module,'Calls',Fake)
    c=module.Config(dataset=str(dataset),vlm_run=str(vlm),run=str(out),context=str(p))
    module.run_pipeline(c)
    assert calls[0]['current_score']['score']=={'points':'15'}
    assert calls[0]['curr_event']['name']=='unknown'
    write_json(vlm/'predictions.json',{'e':{'answer':event}})
    with pytest.raises(ValueError,match='Rerun VLM'):module.run_pipeline(c)


def test_score_reader_hard_budget_and_cache(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace,ModuleType
    from sports_events.broadcast_context import read_scores
    from sports_events.vertex_dataset import sha
    candidates=[]
    for frame,action in [(0,'read'),(10,'invalidate'),(20,'read')]:
        p=tmp_path/f'{frame}.jpg';p.write_bytes(str(frame).encode())
        candidates.append({'source_frame':frame,'timestamp':frame/25,'path':p.name,'sha256':sha(p),'action':action})
    write_json(tmp_path/'score-candidates.json',candidates);calls=[]
    def generate(**kw):
        calls.append(kw)
        return SimpleNamespace(text=json.dumps({'visible':True,'rows':[{'label':'AAA','sets':'0','games':'1','points':'15'}],
            'server_row':0,'raw_text':'AAA 0 1 15'}),candidates=[SimpleNamespace(finish_reason='STOP')],usage_metadata=None)
    google=ModuleType('google');genai=ModuleType('google.genai');types=ModuleType('google.genai.types')
    types.Part=SimpleNamespace(from_text=lambda **k:k,from_bytes=lambda **k:k)
    types.Content=lambda **k:k;types.GenerateContentConfig=lambda **k:k
    genai.Client=lambda **k:SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    genai.types=types;google.genai=genai
    monkeypatch.setitem(sys.modules,'google',google);monkeypatch.setitem(sys.modules,'google.genai',genai)
    monkeypatch.setitem(sys.modules,'google.genai.types',types)
    rows=read_scores(tmp_path,'test-model','project',max_calls=1)
    assert [r['status'] for r in rows]==['observed','display_changed','budget_exhausted']
    read_scores(tmp_path,'test-model','project',max_calls=1)
    assert len(calls)==1
