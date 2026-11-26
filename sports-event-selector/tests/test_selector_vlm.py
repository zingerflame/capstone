import json
from sports_events.selector_vlm import choose_target, input_parts, score, valid_answer


def test_target_is_inside_window_with_earlier_tie_break():
    frames=[{'source_frame':i*3} for i in range(20)]
    rows=[{'id':'b','original_interval':[30,30]}, {'id':'a','original_interval':[27,27]},
          {'id':'outside','original_interval':[58,60]}]
    assert choose_target(frames,rows)['id']=='a'
    assert choose_target(frames,rows[2:]) is None
    assert choose_target(frames,[{'id':'edge','original_interval':[57,57]}])['id']=='edge'


def test_prompt_contains_prediction_but_no_label_or_original_annotation():
    e={'center_seconds':1.23,'players':[{'name':'Roster','path':'p.jpg'}],
       'frames':[{'path':f'{i}.jpg','source_frame':i,'timestamp':i/10} for i in range(20)],
       'output':{'name':'SECRET_TARGET'},'original_annotation':{'secret':'DO_NOT_SEND'}}
    parts=input_parts(e,'gs://bucket/run')
    text=json.dumps(parts)
    assert 'SECRET_TARGET' not in text and 'DO_NOT_SEND' not in text
    assert '1.230' in text and len([p for p in parts if 'fileData' in p])==21


def test_duplicate_windows_do_not_inflate_event_recall():
    event={'name':'Williams','action_base':'hit','action_details':None,'consequence':None}
    examples=[{'id':'a','target_id':'truth1','output':event},
              {'id':'b','target_id':'truth1','output':event},
              {'id':'c','target_id':None,'output':None}]
    result=score(examples,{'a':{'answer':event},'b':{'answer':event},'c':{'answer':None}},[{},{}])
    assert result['exact_event_accuracy']==1
    assert result['end_to_end_exact_event_recall']==.5
    assert result['unscored_windows']==1
    assert result['invalid_or_missing_outputs']==0
    result=score(examples,{},[{},{}])
    assert result['invalid_or_missing_outputs']==3
    assert result['exact_event_accuracy']==0


def test_schema_rejects_wrong_types_and_unknown_fields():
    assert valid_answer(None)
    assert not valid_answer({'name':'A'})
    assert not valid_answer({'name':'A','action_base':'dance','action_details':None,'consequence':None})


def test_endpoint_runner_uses_input_only_and_resumes_cache(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace, ModuleType
    from sports_events.selector_vlm import evaluate
    from sports_events.vertex_dataset import sha
    from sports_events.io import write_json, read_json
    dataset=tmp_path/'data';dataset.mkdir()
    (dataset/'image.jpg').write_bytes(b'test-image')
    event={'name':'SECRET_TARGET','action_base':'hit','action_details':None,'consequence':None}
    row={'id':'a','split':'test','target_id':'truth','output':event,'center_seconds':10,
         'players':[{'name':'Roster','path':'image.jpg'}],
         'frames':[{'path':'image.jpg','source_frame':i,'timestamp':i/10} for i in range(20)]}
    write_json(dataset/'examples.json',[row]);write_json(dataset/'truths.json',[{'split':'test'}])
    write_json(dataset/'assets.json',{'image.jpg':sha(dataset/'image.jpg')})
    write_json(dataset/'build-report.json',{})
    calls=[]
    def generate(**kwargs):
        calls.append(kwargs)
        assert 'SECRET_TARGET' not in repr(kwargs)
        assert len(kwargs['contents'][0]['parts'])==43
        return SimpleNamespace(text=json.dumps(event),usage_metadata=None)
    google=ModuleType('google');genai=ModuleType('google.genai');types=ModuleType('google.genai.types')
    types.Part=SimpleNamespace(from_text=lambda **k:k,from_bytes=lambda **k:k)
    types.Content=lambda **k:k
    types.GenerateContentConfig=lambda **k:k
    genai.Client=lambda **k:SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    genai.types=types;google.genai=genai
    monkeypatch.setitem(sys.modules,'google',google)
    monkeypatch.setitem(sys.modules,'google.genai',genai)
    monkeypatch.setitem(sys.modules,'google.genai.types',types)
    args=SimpleNamespace(dataset=str(dataset),output=str(tmp_path/'run'),model='endpoint',project='project',
                         location='region',limit=None,dry_run=False)
    evaluate(args);evaluate(args)
    assert len(calls)==1
    assert read_json(tmp_path/'run'/'metrics.json')['exact_event_accuracy']==1
