import pytest

from sports_events.io import read_json, write_json
from sports_events.pilot import import_teacher, aggregate


def test_manual_teacher_requires_every_clip_even_negatives(tmp_path):
    dataset = tmp_path/'clips'
    write_json(dataset/'a'/'clip.json', {'frames': [{}]*4})
    registry = tmp_path/'registry.json'
    write_json(registry, {'clips': {}})
    with pytest.raises(ValueError, match='explicit teacher review'):
        import_teacher(dataset, registry)
    assert not (dataset/'a'/'annotations.json').exists()
    write_json(registry, {'teacher': 'ChatGPT', 'independent_human_review': False, 'clips': {'a': []}})
    assert import_teacher(dataset, registry) == 1
    annotation = read_json(dataset/'a'/'annotations.json')
    assert annotation['events'] == []
    assert annotation['reviewed_frame_indices'] == [0,1,2,3]
    assert not annotation['provenance']['independent_human_review']


def test_invalid_teacher_indices_do_not_write_partial_dataset(tmp_path):
    dataset = tmp_path/'clips'
    for name in ('a','b'):
        write_json(dataset/name/'clip.json', {'frames': [{}]*4})
    registry = tmp_path/'registry.json'
    write_json(registry, {'clips': {'a': [], 'b': [{'frame_indices':[2,4], 'label':'serve'}]}})
    with pytest.raises(ValueError, match='in-range'):
        import_teacher(dataset, registry)
    assert not list(dataset.glob('*/annotations.json'))


def test_micro_metrics_count_false_positives_on_empty_clips():
    result = aggregate([{'matched':2,'predicted':2,'truth':4}, {'matched':0,'predicted':8,'truth':0}])
    assert result['precision'] == .2
    assert result['recall'] == .5
    assert result['f1'] == pytest.approx(4/14)
