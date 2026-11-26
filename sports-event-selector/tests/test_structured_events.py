from sports_events.structured_events import convert
from sports_events.structured_events import source_timestamps


def test_timestamps_use_original_video_frames_not_filtered_indices():
    row = {'source_fps': 25, 'original_annotation': {'annotation': {'start': 25857, 'end': 25878}},
           'original_frame_numbers': [25865, 25868, 25871], 'filtered_frame_numbers': [10251, 10252, 10253]}
    result = source_timestamps(row)
    assert result['start_seconds'] == 1034.28
    assert result['start_timecode'] == '00:17:14.280'
    assert result['end_timecode'] == '00:17:15.120'
    assert result['selected_frame_seconds'] == [1034.6, 1034.72, 1034.84]


def test_hit_preserves_details_without_inferred_consequence():
    event, _ = convert('Hit', {'custom': {'Player': 'Williams', 'Side': 'Forehand', 'Type': 'Topspin'}}, {'Williams'})
    assert event == {'name': 'Williams', 'action_base': 'hit', 'action_details': 'forehand topspin', 'consequence': None}


def test_serve_and_unknown_details():
    event, _ = convert('Serve', {'custom': {'Player': 'Williams', 'Result': 'Fault'}}, {'Williams'})
    assert event['consequence'] == 'fault' and event['action_details'] is None
    event, issues = convert('Hit', {'custom': {'Player': 'Williams', 'Side': 'Backhand', 'Type': 'Unsure'}}, {'Williams'})
    assert event['action_details'] == 'backhand' and issues == ['source_shot_type_unsure']


def test_bad_winner_is_not_a_player_and_score_stays_literal():
    event, issues = convert('Point', {'custom': {'Winner': '40-30', 'Score': 'Game'}}, {'Williams'})
    assert event['name'] == 'unknown' and event['action_base'] == 'complete_point'
    assert event['consequence'] == 'score: Game' and issues


def test_metadata_has_exact_schema():
    event, _ = convert('USE', {'custom': {}}, {'Williams'})
    assert set(event) == {'name', 'action_base', 'action_details', 'consequence'}
    assert event['name'] == 'unknown' and event['action_base'] == 'usable_region'
