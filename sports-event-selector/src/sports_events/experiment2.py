"""Controlled loss revision with epoch-wise validation calibration, then locked tests."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch

from .core import boundaries, decode, distance_targets, match_metrics, valid_windows
from .io import read_json, write_json, fingerprint
from .learning import discover, clip_logits, selection_loss
from .model import Selector
from .pilot import aggregate, benchmark, import_teacher

THRESHOLDS = (.1, .2, .3, .4, .5, .6, .7)
CONSENSUS = (.5, .65, .8)


def key(row):
    return (row['f1'], -row['predicted'], row['threshold'], row['consensus'])


def validate(model, records, rate):
    cached = []
    model.eval()
    with torch.no_grad():
        for directory, clip, truth in records:
            if clip['split'] != 'val':
                continue
            logits, _, _ = clip_logits(model, directory, clip, 'cpu')
            n = len(clip['frames'])
            budget = len(boundaries(n)) + round(rate*(n-len(boundaries(n))))
            cached.append((logits.sigmoid().numpy(), n, budget, truth))
    if not cached:
        raise ValueError('Independent validation groups required')
    candidates = []
    for threshold in THRESHOLDS:
        for consensus in CONSENSUS:
            rows = []
            for p, n, budget, truth in cached:
                decoded = decode(p, n, budget, threshold, consensus)
                rows.append(match_metrics([i for s in decoded['event_sequences'] for i in s], truth))
            candidates.append({'threshold':threshold, 'consensus':consensus, **aggregate(rows)})
    return max(candidates, key=key), candidates


def run_training(records, output, version, epochs=20, seed=17):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    training = [r for r in records if r[1]['split'] == 'train']
    positive = sum(len(set(t)-set(boundaries(len(c['frames'])))) for _, c, t in training)
    possible = sum(len(c['frames'])-len(boundaries(len(c['frames']))) for _, c, _ in training)
    rate = positive/max(1,possible)
    model = Selector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    history, best = [], None
    for epoch in range(1, epochs+1):
        model.train()
        order = list(training); random.shuffle(order)
        details = []
        count_weight = .5 if version == 'v1' else .2 * min(1, max(0, (epoch-1)/5))
        for directory, clip, truth in order:
            optimizer.zero_grad(set_to_none=True)
            logits, ids, valid = clip_logits(model, directory, clip, 'cpu')
            loss, metrics = selection_loss(logits, ids, valid, len(clip['frames']), truth,
                                           version=version, count_weight=count_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            details.append({'clip':clip['id'], 'gradient_norm_before_clip':float(grad), **metrics})
        winner, candidates = validate(model, records, rate)
        state = {'model':model.state_dict(), 'width':model.width, 'budget_rate':rate,
                 'threshold':winner['threshold'], 'consensus':winner['consensus'],
                 'epoch':epoch, 'seed':seed, 'loss_version':version}
        torch.save(state, output/'last.pt')
        if best is None or key(winner) > key(best):
            best = {**winner, 'epoch':epoch}
            torch.save(state, output/'best.pt')
        history.append({'epoch':epoch, 'count_weight':count_weight, 'training':details,
                        'validation_winner':winner, 'validation_candidates':candidates})
        write_json(output/'history.json', history)
        write_json(output/'selection.json', best)
        print(f'{version} epoch {epoch}/{epochs}: loss={np.mean([r["loss"] for r in details]):.4f}, calibrated val F1={winner["f1"]:.4f}',flush=True)
    return best


def diagnose(records, output):
    ideal = []
    for _, clip, truth in records:
        if clip['split'] != 'train' or not truth:
            continue
        n = len(clip['frames'])
        target = torch.tensor(distance_targets(n,truth))
        ids, valid = map(torch.tensor, valid_windows(n))
        logits = torch.logit(target.clamp(.000001,.999999))[ids]
        values = {}
        for version in ('v1','v2'):
            _, stats = selection_loss(logits,ids,valid,n,truth,version=version)
            values[version] = stats
        ideal.append({'clip':clip['id'], 'positive_frames':len(truth),
                      'soft_target_mass':float(target.sum()), **values})
    old = read_json('runs/tennis_chatgpt/history.json')
    loss_parts = []
    for epoch in old:
        parts = {k:float(np.mean([r[k] for r in epoch['training']])) for k in
                 ('loss','relevance','excess','coverage','duplicate')}
        parts['weighted_excess'] = .5*parts['excess']
        loss_parts.append({'epoch':epoch['epoch'],**parts})
    write_json(output, {'ideal_target_diagnostic':ideal, 'original_loss_history':loss_parts})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='runs/tennis_second')
    parser.add_argument('--epochs',type=int,default=20)
    args=parser.parse_args()
    torch.set_num_threads(4)
    output=Path(args.output)
    if (output/'comparison.json').exists():
        raise ValueError('Completed experiment exists; use a fresh output path')
    records=discover('data/tennis_pilot/clips')
    import_teacher('data/tennis_second/clips','examples/tennis_second_teacher.json')
    fresh=discover('data/tennis_second/clips')
    trainval=[r for r in records if r[1]['split']!='test']
    if {c['group_id'] for _,c,_ in trainval} & {c['group_id'] for _,c,_ in fresh}:
        raise ValueError('Fresh test match leaks into train/validation')
    protocol={'epochs':args.epochs,'seed':17,'threads':4,'thresholds':THRESHOLDS,'consensus':CONSENSUS,
        'unchanged':'architecture, training data, sampling, budget, optimizer, window rules',
        'control':'v1 loss, calibrate all epochs on validation micro F1',
        'revised':'v2 loss, same epoch calibration; balanced regions, target-consistent duplicate mass, count warmup and gentler normalization',
        'fresh_test':'New intervals in the same two test matches; not new independent matches',
        'teacher_registry_fingerprint':fingerprint(read_json('examples/tennis_second_teacher.json')),
        'trainval_clips':[c['id'] for _,c,_ in trainval], 'fresh_clips':[c['id'] for _,c,_ in fresh]}
    write_json(output/'protocol.json',protocol)
    diagnose(records,output/'diagnostics.json')
    winners={}
    for version in ('v1','v2'):
        winners[version]=run_training(trainval,output/version,version,args.epochs)
    # Both training and model-selection procedures finish before opening test predictions.
    results={'validation_selections':winners,'fresh_test':{},'old_test_regression':{}}
    checkpoints={'original':'runs/tennis_chatgpt/calibrated.pt',
                 'control':output/'v1/best.pt','revised':output/'v2/best.pt'}
    for name,path in checkpoints.items():
        results['fresh_test'][name]=benchmark(fresh,path,output/'fresh_test'/name)
        results['old_test_regression'][name]=benchmark(records,path,output/'old_test'/name)
    write_json(output/'comparison.json',results)
    print(results,flush=True)


if __name__=='__main__':main()
