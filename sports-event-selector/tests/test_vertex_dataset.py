import json
import pytest
from sports_events.vertex_dataset import nearest_frames, validate_record, image_part
from sports_events.player_references import box_pixels, detect


def test_nearest_uses_original_distance_and_tie_breaks_earlier():
    frames=[{'source_frame':i,'filtered_index':n} for n,i in enumerate([0,1,5,8,10,12,30,31])]
    assert [f['source_frame'] for f in nearest_frames(frames,9,4)]==[5,8,10,12]
    assert [f['source_frame'] for f in nearest_frames(frames,9,1)]==[8]
    assert [f['source_frame'] for f in nearest_frames(frames,0,4)]==[0,1,5,8]
    assert [f['source_frame'] for f in nearest_frames(frames,99,4)]==[10,12,30,31]
    with pytest.raises(ValueError):nearest_frames(frames,0,20)


def test_vertex_message_layout_and_frame_count():
    parts=[{'text':'player_information'},image_part('player.jpg','gs://bucket/assets')]
    for i in range(1,21):parts += [{'text':f'frame{i:02d}: timestamp_seconds={i}'},image_part(f'{i}.jpg','gs://bucket/assets')]
    event={'name':'Player','action_base':'hit','action_details':None,'consequence':None}
    record={'contents':[{'role':'user','parts':parts},{'role':'model','parts':[{'text':json.dumps(event)}]}]}
    assert validate_record(record)==event
    parts[-2]['text']='frame19: duplicate'
    with pytest.raises(ValueError):validate_record(record)


def test_player_crop_bounds():
    assert box_pixels([.1,.2,.3,.8],100,100)==(10,20,30,80)
    for box in ([.4,0,.2,1],[0,0,2,1],[0,0,.01,.01]):
        with pytest.raises(ValueError):box_pixels(box,100,100)


def test_scoreboard_detector_proposes_review_not_face_identity(tmp_path,monkeypatch):
    import cv2
    import numpy as np
    image=tmp_path/'frame.jpg';cv2.imwrite(str(image),np.zeros((100,100,3),dtype=np.uint8))
    response={'scoreboard_names':['A','B'],'server_name_text':'B','near_bbox':[.1,.5,.4,.9],
              'far_bbox':[.3,.1,.5,.4],'serving_side':'far','evidence':'Arrow beside B, far player serving.'}
    class Fake:
        def __init__(self,*args):pass
        def ask(self,prompt,images):return dict(response)
    monkeypatch.setattr('sports_events.player_references.Provider',Fake)
    result=detect(image,'gemini','test',tmp_path/'detection.json')
    assert [p['name'] for p in result['proposals']]==['B','A']
    assert all(not p['reviewed'] for p in result['proposals'])
    response['server_name_text']='invented'
    with pytest.raises(ValueError):detect(image,'gemini','test',tmp_path/'invalid.json')
