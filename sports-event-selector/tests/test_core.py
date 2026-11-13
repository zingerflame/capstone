import numpy as np
import pytest
from sports_events.core import boundaries, decode, distance_targets, match_metrics, valid_windows, vote, split_sequences
from sports_events.io import validate_events


def test_100_frames_have_91_windows():
    ids, valid = valid_windows(100)
    assert ids.shape == (91, 10)
    assert ids[0].tolist() == list(range(10))
    assert ids[-1].tolist() == list(range(90,100))
    assert valid.all()


def test_three_window_boundary_and_cap():
    p = np.ones((91,10))
    chosen, _, _, support = vote(p, 100)
    assert (chosen.sum(1) <= 6).all()
    assert support[:4].tolist() == [1,2,3,4]
    assert support[-4:].tolist() == [4,3,2,1]


@pytest.mark.parametrize("n", [1,2,3,4,8,10,11,12,100])
def test_endpoints_always_kept_no_fabricated_support(n):
    p = np.zeros(valid_windows(n)[0].shape)
    result = decode(p, n, budget=0)
    assert result["selected_indices"] == boundaries(n)
    if n <= 10:
        assert max(result["support"]) == 1
    assert all(len(s) in (2,3) for s in result["event_sequences"])


def test_distance_taper_in_filtered_coordinates():
    assert distance_targets(11,[5]).tolist() == [0,0,.25,.5,.75,1,.75,.5,.25,0,0]
    assert not distance_targets(10,[]).any()


def test_budget_and_sequences():
    ids, _ = valid_windows(100)
    p = np.where(np.isin(ids, [20,21,22,60,61,62]), .99, .01)
    result = decode(p,100,budget=7)
    assert len(result["selected_indices"]) <= 7
    assert result["event_sequences"] == [[20,21,22]]
    assert set(boundaries(100)).issubset(result["selected_indices"])
    assert all(result["support"][i] >= 3 for seq in result["event_sequences"] for i in seq)


def test_runs_do_not_create_singletons_or_invent_frames():
    for n in range(2,20):
        groups = split_sequences(list(range(n)))
        assert [x for g in groups for x in g] == list(range(n))
        assert all(len(g) in (2,3) for g in groups)


def test_one_to_one_metric_does_not_credit_duplicates():
    result = match_metrics([4,5,6], [5])
    assert result["matched"] == 1
    assert result["precision"] == pytest.approx(1/3)


@pytest.mark.parametrize("indices", [[1], [1,1], [-1,2], [2,1], [1,20], [True,2]])
def test_reject_bad_teacher_indices(indices):
    with pytest.raises(ValueError):
        validate_events([{"frame_indices":indices, "label":"pass"}],10)


def test_teacher_cannot_reference_unseen_frame():
    with pytest.raises(ValueError):
        validate_events([{"frame_indices":[1,2],"label":"pass"}],10,{3,4})
