import cv2
import numpy as np
import pytest
from sports_events.io import write_json, read_json
from sports_events.tenniset import frame_triplet, convert_event, convert_annotations, extract


def test_triplet_original_indices_and_validation():
    assert frame_triplet(100, 120) == [109, 110, 111]
    assert frame_triplet(0, 1) == [0, 1]
    for a, b in [(1, 1), (-1, 4), (3, 2), (1.5, 4), (True, 4)]:
        with pytest.raises(ValueError): frame_triplet(a, b)


def test_roles_and_unmapped_labels():
    annotation = {'custom': {'Player': 'Williams', 'Result': 'Fault'}}
    assert convert_event('Serve', annotation)['label'] == 'Williams serves (fault)'
    assert convert_event('Point', annotation)['role'] == 'context'
    assert convert_event('USE', annotation)['role'] == 'metadata'
    with pytest.raises(ValueError): convert_event('Unrecognised', annotation)


def test_projection_deduplicates_without_changing_source_numbering():
    from sports_events.tenniset_train import project_anchors, SPLITS
    assert project_anchors([109, 110, 111], list(range(100, 120, 2))) == [4, 5]
    assert project_anchors([110, 111, 112], list(range(100, 120, 2))) == [5, 6]
    assert SPLITS == {'V006':'train','V007':'train','V008':'train','V009':'val','V010':'test'}


def test_exact_frames_preserve_all_annotations(tmp_path):
    annotations, videos, output = (tmp_path / name for name in ('annotations', 'videos', 'sequence-frames'))
    videos.mkdir()
    original = {'name': 'a', 'start': 4, 'end': 8, 'custom': {'Player': 'P'}, 'desc': 'source text'}
    write_json(annotations / 'V001.json', {'classes': {'Hit': [original], 'USE': [dict(original)]}})
    convert_annotations(annotations, output)
    assert not (output / 'sequences.json').exists()
    writer = cv2.VideoWriter(str(videos / 'V001.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (32, 24))
    for i in range(12): writer.write(np.full((24, 32, 3), i * 15, np.uint8))
    writer.release()
    records = extract(annotations, videos, output)
    assert len(records) == 2
    assert records[0]['original_annotation']['annotation'] == original
    assert records[0]['original_frame_numbers'] == [5, 6, 7]
    assert records[0]['training_eligible'] and not records[1]['training_eligible']
    capture = cv2.VideoCapture(str(videos / 'V001.mp4'))
    for i in range(8):
        ok, frame = capture.read()
        assert ok
        if i in (5, 6, 7):
            stored = cv2.imread(records[0]['image_paths'][i-5])
            assert np.array_equal(frame, stored)
    capture.release()
    assert read_json(output / 'conversion-report.json')['complete']
    assert extract(annotations, videos, output) == records


def test_prepared_context_uses_completed_sequences_and_match_splits(tmp_path):
    from sports_events.tenniset_train import prepare, SPLITS
    from sports_events.learning import discover
    annotations, videos, output = (tmp_path / name for name in ('annotations', 'videos', 'sequence-frames'))
    videos.mkdir()
    for video_id in SPLITS:
        write_json(annotations / f'{video_id}.json', {'classes': {
            'USE': [{'start':0,'end':11,'custom':{}}],
            'Hit': [{'start':4,'end':8,'custom':{'Player':'P'}}]}})
        writer=cv2.VideoWriter(str(videos/f'{video_id}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(32,24))
        for i in range(12):writer.write(np.full((24,32,3),i*15,np.uint8))
        writer.release()
    extract(annotations,videos,output)
    prepared=tmp_path/'clips'
    prepare(output/'sequences.json',prepared,clips_per_video=1,seconds=.32,stride=2)
    records=discover(prepared)
    assert len(records)==5
    assert {clip['split'] for _,clip,_ in records} == {'train','val','test'}
    for _,clip,truth in records:
        assert [f['source_frame'] for f in clip['frames']]==[0,2,4,6]
        assert truth==[2,3]
    with pytest.raises(ValueError):prepare(output/'sequences.json',prepared)
