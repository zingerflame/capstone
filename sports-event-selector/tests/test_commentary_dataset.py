import json
import pytest
from sports_events.commentary_dataset import records, sentence


def row(id,video,time,event):
    return {'id':id,'video_id':video,'original_interval':[time,time], 'event':event,
            'original_timestamps':{'start_seconds':time},'original_annotation':{}}


def test_history_is_past_only_and_resets_at_match_boundary():
    event={'name':'A','action_base':'hit','action_details':'forehand topspin','consequence':None}
    rs=[row(str(i),'V006',i,event) for i in range(4)]+[row('new','V009',0,event)]
    generated=list(records(list(reversed(rs))))
    texts=[]
    for i,(record,audit) in enumerate(generated[:4]):
        payload=json.loads(record['contents'][0]['parts'][0]['text'])
        assert payload['current_commentary']==' '.join(texts)
        assert len(payload['prev_event'])==min(i,2)
        assert audit['previous_event_ids']==[str(j) for j in range(max(0,i-2),i)]
        texts.append(record['contents'][1]['parts'][0]['text'])
    payload=json.loads(generated[-1][0]['contents'][0]['parts'][0]['text'])
    assert payload['current_commentary']=='' and payload['prev_event']==[]


def test_verbalizer_preserves_details_without_inventing_winners():
    for kind in ('topspin','slice','flat','volley','smash'):
        out=sentence({'name':'Williams','action_base':'hit','action_details':'forehand '+kind,'consequence':None})
        assert 'Williams' in out and 'forehand' in out and kind in out
    assert sentence({'name':'A','action_base':'serve','action_details':None,'consequence':'fault'})=='A serves a fault.'
    assert sentence({'name':'A','action_base':'serve','action_details':None,'consequence':None})=='A serves.'
    with pytest.raises(ValueError):sentence({'name':'A','action_base':'hit','action_details':'unknown shot','consequence':None})
