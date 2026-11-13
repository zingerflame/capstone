"""Auditable manual teacher import and a small real-broadcast experiment."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from .core import boundaries, decode, match_metrics, valid_windows
from .io import read_json, write_json, validate_events, fingerprint
from .learning import discover, train, load_model, clip_logits, infer_clip


def import_teacher(dataset, registry_path):
    registry = read_json(registry_path)
    pending = []
    paths = {p.parent.name: p for p in Path(dataset).glob('*/clip.json')}
    if set(paths) != set(registry['clips']):
        raise ValueError('Every prepared clip must have an explicit teacher review; no implicit negatives')
    for name, path in paths.items():
        clip = read_json(path)
        events = validate_events(registry['clips'][name], len(clip['frames']))
        pending.append((path.parent / 'annotations.json', {
            'events': events, 'reviewed': True,
            'review_kind': 'ChatGPT visual inspection; not independent human review',
            'provenance': {k: v for k, v in registry.items() if k != 'clips'},
            'registry_sha256': hashlib.sha256(Path(registry_path).read_bytes()).hexdigest(),
            'clip_fingerprint': fingerprint(clip),
            'reviewed_frame_indices': list(range(len(clip['frames'])))}))
    for path, annotation in pending:
        write_json(path, annotation)
    return len(pending)


def aggregate(rows):
    tp = sum(r['matched'] for r in rows)
    predicted = sum(r['predicted'] for r in rows)
    truth = sum(r['truth'] for r in rows)
    precision = tp / max(1, predicted)
    recall = tp / max(1, truth)
    return {'matched': tp, 'predicted': predicted, 'truth': truth,
            'precision': precision, 'recall': recall,
            'f1': 2 * tp / max(1, predicted + truth)}


def calibrate(records, checkpoints, output):
    """Only validation images/labels choose checkpoint and decision thresholds."""
    validation = [r for r in records if r[1]['split'] == 'val']
    if not validation:
        raise ValueError('Validation clips required')
    candidates = []
    for checkpoint in checkpoints:
        model, state = load_model(checkpoint)
        cached = []
        with torch.no_grad():
            for directory, clip, truth in validation:
                logits, _, _ = clip_logits(model, directory, clip, 'cpu')
                n = len(clip['frames'])
                budget = min(n, len(boundaries(n)) + round(state['budget_rate'] * (n-len(boundaries(n)))))
                cached.append((logits.sigmoid().numpy(), n, budget, truth))
        for threshold in (.1, .2, .3, .4, .5, .6, .7):
            for consensus in (.5, .65, .8):
                scores = []
                for p, n, budget, truth in cached:
                    result = decode(p, n, budget, threshold, consensus)
                    scores.append(match_metrics([i for s in result['event_sequences'] for i in s], truth))
                candidates.append({'checkpoint': str(checkpoint), 'threshold': threshold,
                                   'consensus': consensus, **aggregate(scores)})
    winner = max(candidates, key=lambda r: (r['f1'], -r['predicted'], r['threshold'], r['consensus']))
    state = torch.load(winner['checkpoint'], weights_only=True)
    state.update(threshold=winner['threshold'], consensus=winner['consensus'])
    torch.save(state, Path(output) / 'calibrated.pt')
    write_json(Path(output) / 'calibration.json', {'selection_split': 'val',
        'clip_ids': [c['id'] for _, c, _ in validation], 'winner': winner, 'candidates': candidates})
    return winner


def benchmark(records, checkpoint, output):
    model, state = load_model(checkpoint)
    rows = []
    for directory, clip, truth in records:
        if clip['split'] != 'test':
            continue
        result = infer_clip(model, directory, clip, 'cpu', state['budget_rate'], state['threshold'], state['consensus'])
        write_json(Path(output) / 'predictions' / (clip['id']+'.json'), result)
        prediction = [i for s in result['event_sequences'] for i in s]
        n = len(clip['frames'])
        # A deterministic image-blind baseline with the SAME budget and decoder.
        # Periodic triplets distribute scores across time without seeing labels.
        budget = len(boundaries(n)) + round(state['budget_rate'] * (n-len(boundaries(n))))
        spacing = max(4, round((n-4) / max(1, (budget-4)/3)))
        scores = np.array([.9 if (i-2) % spacing < 3 else .01 for i in range(n)])
        ids, _ = valid_windows(n)
        baseline = decode(scores[ids], n, budget, .5, .5)
        baseline_indices = [i for s in baseline['event_sequences'] for i in s]
        rows.append({'clip': clip['id'], 'frames': n, 'selected': len(result['selected_indices']),
            'model': {str(d): match_metrics(prediction, truth, d) for d in range(4)},
            'periodic_baseline': {str(d): match_metrics(baseline_indices, truth, d) for d in range(4)}})
    summary = {method: {str(d): aggregate([r[method][str(d)] for r in rows]) for d in range(4)}
               for method in ('model', 'periodic_baseline')}
    summary['retained_fraction'] = sum(r['selected'] for r in rows)/sum(r['frames'] for r in rows)
    write_json(Path(output) / 'test_metrics.json', {'summary': summary, 'clips': rows})
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/tennis_pilot/clips')
    parser.add_argument('--registry', default='examples/tennis_teacher_annotations.json')
    parser.add_argument('--output', default='runs/tennis_chatgpt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    import_teacher(args.dataset, args.registry)
    records = discover(args.dataset)
    write_json(Path(args.output)/'dataset_snapshot.json', [
        {'clip': c['id'], 'split': c['split'], 'group': c['group_id'], 'frames': len(c['frames']),
         'truth_frames': len(t), 'clip_fingerprint': fingerprint(c)} for _, c, t in records])
    train(args.dataset, args.output, epochs=args.epochs)
    checkpoints = [Path(args.output)/name for name in ('best.pt', 'last.pt')]
    print('Validation calibration:', calibrate(records, checkpoints, args.output), flush=True)
    print('Held-out test:', benchmark(records, Path(args.output)/'calibrated.pt', args.output), flush=True)


if __name__ == '__main__':
    main()
