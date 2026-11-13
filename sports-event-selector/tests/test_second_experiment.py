import numpy as np
import pytest
import torch

from sports_events.core import distance_targets, valid_windows
from sports_events.learning import selection_loss
from sports_events.experiment2 import validate
from sports_events.model import Selector


def test_revised_duplicate_term_accepts_the_requested_soft_target():
    ids, valid = map(torch.tensor, valid_windows(30))
    target = torch.tensor(distance_targets(30,[14,15,16]))
    logits = torch.logit(target.clamp(1e-6,1-1e-6))[ids]
    _, old = selection_loss(logits,ids,valid,30,[14,15,16])
    _, new = selection_loss(logits,ids,valid,30,[14,15,16],version='v2')
    assert old['duplicate'] > .1
    assert new['duplicate'] < 1e-9


def test_revised_negative_clip_still_penalizes_excess_and_backpropagates():
    ids, valid = map(torch.tensor,valid_windows(30))
    logits = torch.full(ids.shape,3.,requires_grad=True)
    loss, metrics = selection_loss(logits,ids,valid,30,[],version='v2')
    assert metrics['excess'] > 0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.sum() > 0  # Gradient descent lowers false-positive logits.


def test_epoch_calibration_never_loads_test_clips():
    # This nonexistent directory would fail if a test record were loaded.
    with pytest.raises(ValueError,match='validation'):
        validate(Selector(), [('nonexistent',{'split':'test'},[1,2])], .1)
