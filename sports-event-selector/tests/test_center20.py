import numpy as np
import torch
import pytest
from sports_events.center20 import decode_centers,center_metrics,center_loss,center_targets,truth_centers
from sports_events.core import valid_windows
from sports_events.center20 import window_credit,window_targets,window_metrics,aggregate_windows
from sports_events.io import write_json


def test_full_window_credit_center_edges_and_outside():
    assert window_credit(19.5,19.5)==1
    assert window_credit(19.5,10)==pytest.approx(.3)
    assert window_credit(19.5,29)==pytest.approx(.3)
    assert window_credit(19.5,9.99)==0
    assert window_credit(19.5,29.01)==0
    assert abs(window_credit(19.5,24.25)-.65)<1e-8
    assert window_targets(40,[19.])[0]==np.float32(.3)


def test_window_matching_keeps_recall_and_localization_separate():
    result=aggregate_windows([window_metrics([19.5,19.5],[10])])
    assert result['recall']==1 and result['precision']==.5
    assert result['weighted_recall']==pytest.approx(.3)
    assert aggregate_windows([window_metrics([],[])])['mean_matched_credit'] is None


def test_recall_loss_finite_on_positive_and_negative_clips():
    ids,valid=map(torch.tensor,valid_windows(40))
    for truth in ([],[20.]):
        logits=torch.zeros(ids.shape,requires_grad=True)
        loss=center_loss(logits,ids,valid,40,truth,profile='window_recall')
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_windows_exact_consecutive_and_short_inputs_not_padded():
    n=40;ids,valid=valid_windows(n)
    p=np.zeros_like(ids,dtype=float)
    p[(ids==20)|(ids==21)]=.99
    result=decode_centers(p,n,event_budget=1)
    assert len(result['event_sequences'])==1
    seq=result['event_sequences'][0]
    assert len(seq)==20 and seq==list(range(seq[0],seq[0]+20))
    assert result['event_centers']==[(seq[0]+seq[-1])/2]
    assert {0,1,38,39}<=set(result['selected_indices'])
    ids,_=valid_windows(19)
    assert decode_centers(np.ones_like(ids)*.9,19)['event_sequences']==[]


def test_one_to_one_center_matching_with_fractional_error():
    m=center_metrics([10.5,11.5],[11.],1)
    assert m['matched']==1 and m['precision']==.5 and m['matched_center_mae']==.5
    assert center_metrics([10.5],[11.],0)['matched']==0
    assert center_metrics([10.5],[11.],3)['matched']==1
    assert center_metrics([], [11.],3)['matched_center_mae'] is None


def test_soft_center_targets_and_gradient():
    target=center_targets(40,[20.5])
    assert target[11]==1 and target[8]==.25 and target[7]==0
    ids,valid=map(torch.tensor,valid_windows(40))
    logits=torch.zeros(ids.shape,requires_grad=True)
    loss=center_loss(logits,ids,valid,40,[20.5]);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0


def test_truth_uses_human_interval_not_augmented_triplet(tmp_path):
    clip={'frames':[{'source_frame':100+3*i,'filtered_index_global':50+i} for i in range(30)]}
    write_json(tmp_path/'annotations.json',{'events':[{'sequence_id':'a','frame_indices':[10,11,12]}]})
    source={'a':{'original_interval':[130,133]}}
    assert truth_centers(tmp_path,clip,source)==[10.5]
