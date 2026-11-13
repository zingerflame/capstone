import numpy as np
import torch

from sports_events.pretrained import FeatureSelector, feature_logits, full_frame_tensor
from sports_events.experiment3 import budget_for


def test_padding_and_absent_scoreboard_do_not_change_valid_predictions():
    torch.manual_seed(5)
    model = FeatureSelector(motion=True).eval()
    x = torch.randn(1, 10, 642)
    x[:, :, 640] = 0
    valid = torch.tensor([[True] * 6 + [False] * 4])
    changed = x.clone()
    changed[:, :, 512:640] = 100
    changed[:, 6:] = -30
    with torch.no_grad():
        expected = model(x, valid)[:, :6]
        actual = model(changed, valid)[:, :6]
    torch.testing.assert_close(actual, expected)


def test_motion_is_limited_to_each_ten_frame_window():
    torch.manual_seed(7)
    model = FeatureSelector(motion=True).eval()
    features = torch.randn(15, 642)
    changed = features.clone()
    changed[10:] += 30
    with torch.no_grad():
        original, ids, valid = feature_logits(model, features)
        altered, _, _ = feature_logits(model, changed)
    torch.testing.assert_close(original[0], altered[0])
    assert ids[0].tolist() == list(range(10))
    assert valid.all()


def test_letterbox_retains_wide_frame_edges():
    image = np.zeros((50, 200, 3), dtype=np.uint8)
    image[:, :10] = 255
    image[:, -10:] = 255
    value = full_frame_tensor(image)
    assert value.shape == (3, 224, 224)
    assert (value[:, 112, 0] > 1).all()
    assert (value[:, 112, -1] > 1).all()
    assert (value[:, 112, 112] < -1).all()


def test_budget_preserves_endpoints_and_caps_at_length():
    for n in range(1, 20):
        assert min(4, n) <= budget_for(n, .12, 3) <= n
    assert budget_for(100, .12, 3) > budget_for(100, .12, 1)
