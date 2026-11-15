import cv2
import numpy as np
import pytest
from sports_events.io import write_json, read_json
from sports_events.video import sample_video
from sports_events.tenniset import file_hash
from sports_events.tenniset_filtered import map_targets, scan_video, build
from sports_events.tenniset_train import SPLITS
from sports_events.learning import discover


def test_collapsed_targets_augmented_only_inside_interval_and_time_bound():
    mapping=map_targets([100,101,102],[95,110],[96,101,106],25)
    assert mapping['nearest_unique_count']==1
    assert mapping['augmented_count']==2
    assert mapping['original_frame_numbers']==[96,101,106]
    assert max(mapping['distance_to_original_target_seconds'])<=.16
    assert map_targets([100,101,102],[100,102],[96,101,106],25)['status']=='unmappable'
    assert map_targets([100,101,102],[90,120],[90,115],25)['status']=='unmappable'
    with pytest.raises(ValueError):map_targets([1,2,3],[0,5],[3,2],25)


def make_video(path):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),25,(32,24))
    for i in range(40):writer.write(np.full((24,32,3),i*4,np.uint8))
    writer.release()


def record(video_id,source,kind,anchors,interval):
    return {'id':video_id+'_'+kind,'video_id':video_id,'original_frame_numbers':anchors,
        'original_interval':interval,'source_video':str(source),'source_fps':25,
        'source_video_sha256':file_hash(source),'image_paths':[],
        'event':{'type':kind,'label':kind},'training_eligible':kind=='hit',
        'original_annotation':{'category':kind,'annotation':{'start':interval[0],'end':interval[1]}}}


def test_feed_is_identical_to_production_sampler_and_independent_of_labels(tmp_path):
    source=tmp_path/'video.mp4';make_video(source)
    spec={'id':'tenniset_V006_000','start':4,'seconds':1.,'split':'train'}
    rows=[record('V006',source,'hit',[19,20,21],[10,30])]
    params={'min_fps':10.,'max_fps':20.,'delta':.035}
    state=scan_video('V006',rows,[spec],tmp_path/'scan',params,.16)
    clip=sample_video(source,tmp_path/'production',{'id':'test','sport':'tennis'},**params)
    assert [r['source_frame'] for r in state['frames']]==[r['source_frame'] for r in clip['frames']]
    indices=[r['source_frame'] for r in state['frames']]
    assert indices[:2]==[0,1] and indices[-2:]==[38,39]
    other=scan_video('V006',[record('V006',source,'hit',[9,10,11],[4,20])],[spec],tmp_path/'other',params,.16)
    assert other['frames']==state['frames']
    # Original indices, global filtered indices and local clip indices differ.
    selected=state['clip_frames'][spec['id']]
    assert selected[0]['index']==0 and selected[0]['filtered_index_global']>0
    assert selected[0]['source_frame']==4


def test_full_build_exports_retained_pngs_and_uses_filtered_local_targets(tmp_path):
    records=[];specs=[]
    for video_id,split in SPLITS.items():
        source=tmp_path/f'{video_id}.mp4';make_video(source)
        records.extend([record(video_id,source,'hit',[19,20,21],[10,30]),
                        record(video_id,source,'use',[19,20,21],[0,39])])
        specs.append({'id':f'tenniset_{video_id}_000','start':4,'seconds':1.,'split':split})
    write_json(tmp_path/'original/sequences.json',records)
    write_json(tmp_path/'original/conversion-report.json',{'complete':True})
    write_json(tmp_path/'baseline.json',{'clips':specs})
    output=tmp_path/'filtered'
    result=build(tmp_path/'original/sequences.json',tmp_path/'baseline.json',output,workers=1)
    assert result['mapped_actions']==5 and not result['excluded_clips']
    sequences=read_json(output/'sequence-frames/sequences.json')
    for row in sequences:
        feed=read_json(output/'feeds'/f"{row['video_id']}.json")
        assert row['original_sequence_frame_numbers']==[19,20,21]
        assert row['original_frame_numbers']==[feed['frames'][i]['source_frame'] for i in row['filtered_frame_numbers']]
        assert all(cv2.imread(p) is not None for p in row['image_paths'])
    for directory,clip,truth in discover(output/'clips'):
        annotation=read_json(directory/'annotations.json')['events'][0]
        assert [clip['frames'][i]['source_frame'] for i in truth]==annotation['original_frame_numbers']
        assert [clip['frames'][i]['filtered_index_global'] for i in truth]==annotation['filtered_frame_numbers']
