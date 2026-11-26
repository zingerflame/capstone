import pytest
from sports_events.commentary_metrics import rouge_l,semantic_metrics
from sports_events.end_to_end_eval import validate_judge,pilot_bounds


def test_video_source_length_is_never_used_as_clip_length():
    clip={'source_start_frame':1000,'source_fps':25,'source_frame_count':109890,
          'frames':[{'broadcast_timestamp':40.04},{'broadcast_timestamp':47.96}]}
    assert pilot_bounds(clip,{'start':1000,'seconds':8})==(40,48)
    with pytest.raises(ValueError):pilot_bounds(clip,{'start':1000,'seconds':4395})


def test_semantic_temporal_gate_and_missing_events():
    refs=[{'id':'r','clip_id':'c','seconds':2,'text':'truth'}]
    preds=[{'id':'p','clip_id':'c','seconds':2,'text':'caption'}]
    vectors={'truth':[1,0],'caption':[1,0]}
    assert semantic_metrics(preds,refs,vectors)['semantic_soft_recall']==1
    preds[0]['seconds']=6
    assert semantic_metrics(preds,refs,vectors)['semantic_soft_recall']==0
    assert semantic_metrics([],refs,vectors)['semantic_soft_recall']==0
    assert rouge_l('A forehand from Williams.','A forehand from Williams.')==1
    assert rouge_l('','reference')==0


def test_judge_rejects_duplicates_and_out_of_time_matches():
    ref=[{'id':'r','seconds':2}];arms={'A':[{'id':'p','seconds':3}],'B':[]}
    s={'matches':[{'reference_id':'r','candidate_id':'p','credit':.5}],
       'contradictory_ids':[],'unsupported_ids':[],'fluency':4,'coherence':4,'reason':'partial'}
    result={'A':s,'B':{**s,'matches':[]},'winner':'A'}
    assert validate_judge(result,arms,ref)==result
    s['matches']*=2
    with pytest.raises(ValueError):validate_judge(result,arms,ref)
    s['matches']=s['matches'][:1];arms['A'][0]['seconds']=6
    with pytest.raises(ValueError):validate_judge(result,arms,ref)
