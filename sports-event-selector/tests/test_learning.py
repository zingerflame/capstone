import numpy as np
import torch
from sports_events.core import valid_windows
from sports_events.learning import selection_loss
from sports_events.model import Selector


def test_global_loss_backpropagates_and_excess_is_penalized():
    ids, valid = valid_windows(30)
    ids, valid = torch.tensor(ids), torch.tensor(valid)
    logits = torch.full(ids.shape, 5.0, requires_grad=True)
    loss, stats = selection_loss(logits,ids,valid,30,[14,15,16])
    assert stats["excess"] > 0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_negative_clip_loss_and_no_labels_in_forward():
    torch.set_num_threads(2)
    model = Selector()
    feature = model.encode(torch.rand(10,3,96,96), torch.rand(10,3,32,96),
                           torch.zeros(10), torch.ones(10)*.1)
    ids, valid = valid_windows(10)
    logits = model.classify(feature[torch.tensor(ids)], torch.tensor(valid))
    assert logits.shape == (1,10)
    loss, _ = selection_loss(logits,torch.tensor(ids),torch.tensor(valid),10,[])
    loss.backward()
    assert model.scene.layers[0].weight.grad.abs().sum() > 0


def test_missing_scoreboard_pixels_do_not_change_embedding():
    model = Selector().eval()
    scene, zero, dt = torch.rand(2,3,96,96), torch.zeros(2), torch.zeros(2)
    with torch.no_grad():
        a = model.encode(scene, torch.zeros(2,3,32,96),zero,dt)
        b = model.encode(scene, torch.ones(2,3,32,96),zero,dt)
    assert torch.equal(a,b)
