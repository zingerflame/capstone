from copy import deepcopy
import json
import pytest
from sports_events.commentary_timeline import timeline,verify,export_tuning


def rows():
    event={'name':'A','action_base':'serve','action_details':None,'consequence':'fault'}
    return [{'id':str(i),'video_id':video,'original_interval':[start,end], 'source_fps':25,
             'original_timestamps':{},'original_annotation':{},'event':event}
            for i,(video,start,end) in enumerate([('V006',0,25),('V006',10,20),('V009',0,10)])]


def test_no_retroactive_outcome_and_exact_prefixes():
    entries=list(timeline(rows()))
    assert [e['id'] for e in entries]==['1','0','2']
    assert entries[0]['commit_seconds']==.8
    states=verify(entries)
    assert states['V006'][0]=='A serves a fault. A serves a fault.'
    assert states['V009'][0]=='A serves a fault.'
    assert '[00:00:00.800 | frame 20]' in states['V006'][1]


def test_changed_earlier_words_and_reordering_are_detected():
    entries=list(timeline(rows()));changed=deepcopy(entries)
    changed[0]['new_sentence']='A wins the point.'
    with pytest.raises(ValueError):verify(changed)
    with pytest.raises(ValueError):verify([entries[1],entries[0],entries[2]])


def test_training_is_extracted_from_frozen_history(tmp_path):
    entries=list(timeline(rows()));export_tuning(entries,tmp_path)
    lines=[json.loads(x) for x in (tmp_path/'train.vertex.jsonl').read_text().splitlines()]
    assert json.loads(lines[0]['contents'][0]['parts'][0]['text'])['current_commentary']==''
    assert json.loads(lines[1]['contents'][0]['parts'][0]['text'])['current_commentary']==entries[0]['new_sentence']
    assert lines[1]['contents'][1]['parts'][0]['text']==entries[1]['new_sentence']
