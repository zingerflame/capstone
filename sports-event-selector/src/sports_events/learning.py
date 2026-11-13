"""Whole-clip overlap loss with checkpointed image/window batches."""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .core import boundaries, decode, distance_targets, match_metrics, valid_windows
from .io import read_json, validate_events, write_json
from .model import Selector


def load_images(directory, clip, indices, device):
    images, boards = [], []
    for i in indices.tolist():
        row = clip["frames"][i]
        pair = []
        for key, size in (("image", (96, 96)), ("scoreboard", (96, 32))):
            image = cv2.imread(str(Path(directory) / row[key]))
            if image is None:
                raise ValueError(f"Missing or unreadable image: {row[key]}")
            image = cv2.cvtColor(cv2.resize(image, size), cv2.COLOR_BGR2RGB)
            pair.append(torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255)
        images.append(pair[0])
        boards.append(pair[1])
    present = torch.full((len(images),), float(clip.get("scoreboard_roi") is not None), device=device)
    times = [clip["frames"][i]["timestamp"] for i in range(len(clip["frames"]))]
    dt = torch.tensor([0 if i == 0 else min(2, times[i] - times[i-1]) for i in indices.tolist()],
                      dtype=torch.float32, device=device)
    return torch.stack(images).to(device), torch.stack(boards).to(device), present, dt


def clip_logits(model, directory, clip, device, batch_size=64):
    n = len(clip["frames"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    def encode(indices):
        return model.encode(*load_images(directory, clip, indices, device))

    encoded = []
    for start in range(0, n, batch_size):
        indices = torch.arange(start, min(n, start+batch_size))
        encoded.append(checkpoint(encode, indices, use_reentrant=False) if model.training else encode(indices))
    features = torch.cat(encoded)
    ids, valid = valid_windows(n)
    ids = torch.as_tensor(ids, device=device)
    valid = torch.as_tensor(valid, device=device)
    results = []
    for start in range(0, len(ids), batch_size):
        x, mask = features[ids[start:start+batch_size]], valid[start:start+batch_size]
        results.append(checkpoint(model.classify, x, mask, use_reentrant=False)
                       if model.training else model.classify(x, mask))
    return torch.cat(results), ids, valid


def selection_loss(logits, ids, valid, n, truth, threshold=0.5, consensus=0.5,
                   version="v1", count_weight=None):
    """Exact hard decoded count in forward; soft surrogate in backward.

    Loss includes distance-soft BCE, coverage, exclusive GT assignment, and global
    excess count. Top-six, majority and grouping are discrete; we do NOT claim an
    exact derivative through them. Straight-through estimates train the selector.
    """
    device = logits.device
    if version not in ("v1", "v2"):
        raise ValueError("Unknown loss version")
    truth = sorted(set(truth))
    p = logits.sigmoid()
    target = torch.tensor(distance_targets(n, truth), device=device)
    per_window = F.binary_cross_entropy_with_logits(logits, target[ids], reduction="none")
    support = torch.zeros(n, device=device).scatter_add(0, ids[valid], torch.ones_like(p[valid]))
    # Balance repeated views: each global frame contributes equal BCE mass.
    frame_weight = torch.ones(n, device=device)
    if version == "v2":
        # Balance temporal neighborhoods against background without changing the
        # requested distance-soft targets. All-negative clips still train normally.
        region = target > 0
        if region.any() and (~region).any():
            frame_weight[region] = n / (2 * region.sum())
            frame_weight[~region] = n / (2 * (~region).sum())
    relevance = (per_window[valid] * frame_weight[ids[valid]] / support[ids[valid]]).sum() / n
    masked = p.masked_fill(~valid, -1)
    order = torch.argsort(masked, dim=1, descending=True, stable=True)[:, :6]
    hard = torch.zeros_like(p).scatter(1, order, 1) * (p >= threshold) * valid
    straight = hard.detach() + (p - p.detach()) * valid
    vote = torch.zeros(n, device=device).scatter_add(0, ids[valid], straight[valid]) / support.clamp_min(1)
    soft_final = torch.sigmoid((vote - consensus) / 0.15) * (support >= 3)
    force = torch.zeros(n, dtype=torch.bool, device=device)
    force[boundaries(n)] = True
    soft_final = torch.where(force, torch.ones_like(soft_final), soft_final)
    budget = len(set(truth) | set(boundaries(n)))
    actual = decode(p.detach().cpu().numpy(), n, budget=None, threshold=threshold, consensus=consensus)
    hard_count = logits.new_tensor(float(len(actual["selected_indices"])))
    count = hard_count + soft_final.sum() - soft_final.sum().detach()
    excess = (F.relu(count - budget) / max(10 if version == "v2" else 1, budget)).square()
    # Mean probability is useful for distance-aware coverage and exclusive matching.
    mean = torch.zeros(n, device=device).scatter_add(0, ids[valid], p[valid]) / support.clamp_min(1)
    coverage, duplicate = logits.new_zeros(()), logits.new_zeros(())
    if truth:
        # Only seven positions per GT: avoid an N x number_of_events allocation.
        gt = torch.tensor(truth, device=device)
        neighbors = gt[:, None] + torch.arange(-3, 4, device=device)[None, :]
        in_range = (neighbors >= 0) & (neighbors < n)
        weights = (1 - torch.arange(-3, 4, device=device).abs()/4)[None, :]
        value = mean[neighbors.clamp(0, n-1)] * weights * in_range
        coverage = (1 - value.max(dim=1).values).mean()
        # Assign each predicted position to its single nearest GT; cap mass per GT.
        positions = torch.arange(n, device=device)
        right = torch.searchsorted(gt, positions).clamp(max=len(gt)-1)
        left = (right-1).clamp(min=0)
        nearest = torch.where((positions-gt[left]).abs() <= (positions-gt[right]).abs(), left, right)
        nearby = (positions - gt[nearest]).abs() <= 3
        mass = torch.zeros(len(gt), device=device).scatter_add(0, nearest[nearby], mean[nearby])
        capacity = torch.ones_like(mass)
        if version == "v2":
            # The old cap of one contradicted soft positive neighborhoods, even
            # when the model exactly reproduced its target probabilities.
            capacity = torch.zeros_like(mass).scatter_add(0, nearest[nearby], target[nearby])
        duplicate = F.relu(mass-capacity).square().mean()
    weight = (0.5 if version == "v1" else 0.2) if count_weight is None else count_weight
    total = relevance + weight*excess + 0.2*coverage + 0.1*duplicate
    return total, {"loss": float(total.detach()), "relevance": float(relevance.detach()),
                   "excess": float(excess.detach()), "coverage": float(coverage.detach()),
                   "duplicate": float(duplicate.detach()), "decoded_count": int(hard_count), "budget": budget}


def discover(dataset, allow_unreviewed=False):
    records, group_splits = [], {}
    for path in sorted(Path(dataset).glob("*/clip.json")):
        clip = read_json(path)
        annotation = read_json(path.parent / "annotations.json")
        if not annotation.get("reviewed") and not allow_unreviewed:
            raise ValueError(f"Review {path.parent / 'annotations.json'} or explicitly use --allow-unreviewed")
        events = validate_events(annotation["events"], len(clip["frames"]))
        split = clip["split"]
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        group = clip["group_id"]
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Match/group {group} leaks across splits")
        group_splits[group] = split
        truth = sorted({i for e in events for i in e["frame_indices"]})
        records.append((path.parent, clip, truth))
    if not records:
        raise ValueError("No annotated clips found")
    return records


def infer_clip(model, directory, clip, device, rate, threshold=0.5, consensus=0.5):
    model.eval()
    with torch.no_grad():
        logits, _, _ = clip_logits(model, directory, clip, device)
    n = len(clip["frames"])
    budget = min(n, len(boundaries(n)) + round(rate * max(0, n-len(boundaries(n)))))
    result = decode(logits.sigmoid().cpu().numpy(), n, budget, threshold, consensus)
    result["event_sequence_frames"] = [[clip["frames"][i] for i in seq] for seq in result["event_sequences"]]
    result["selected_frames"] = [clip["frames"][i] for i in result["selected_indices"]]
    result["clip_id"] = clip["id"]
    return result


def train(dataset, output, epochs=5, device="cpu", allow_unreviewed=False, seed=17):
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    records = discover(dataset, allow_unreviewed)
    training = [r for r in records if r[1]["split"] == "train"]
    validation = [r for r in records if r[1]["split"] == "val"]
    if not training or not validation:
        raise ValueError("Separate match-level train and val clips are required")
    # Learn output budget from training labels only. No ground-truth count at inference.
    positive = sum(len(set(t) - set(boundaries(len(c["frames"])))) for _, c, t in training)
    possible = sum(max(0, len(c["frames"])-len(boundaries(len(c["frames"])))) for _, c, _ in training)
    rate = positive / max(1, possible)
    model = Selector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history, best = [], -1
    for epoch in range(epochs):
        model.train()
        order = list(training)
        random.shuffle(order)
        details = []
        for directory, clip, truth in order:
            optimizer.zero_grad(set_to_none=True)
            logits, ids, valid = clip_logits(model, directory, clip, device)
            loss, metrics = selection_loss(logits, ids, valid, len(clip["frames"]), truth)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            details.append({"clip": clip["id"], **metrics})
        scores = []
        for directory, clip, truth in validation:
            result = infer_clip(model, directory, clip, device, rate)
            # Boundary context does not earn event-detection credit.
            prediction = [i for s in result["event_sequences"] for i in s]
            scores.append(match_metrics(prediction, truth))
        score = float(np.mean([x["f1"] for x in scores]))
        entry = {"epoch": epoch+1, "training": details, "validation": scores, "val_f1": score}
        history.append(entry)
        print(f"Epoch {epoch+1}/{epochs}: loss={np.mean([d['loss'] for d in details]):.4f}, val F1={score:.4f}", flush=True)
        state = {"model": model.state_dict(), "width": model.width, "budget_rate": rate,
                 "threshold": 0.5, "consensus": 0.5, "seed": seed, "epoch": epoch+1}
        torch.save(state, output / "last.pt")
        if score > best:
            best = score
            torch.save(state, output / "best.pt")
        write_json(output / "history.json", history)
    return history


def load_model(path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    model = Selector(state["width"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def evaluate(dataset, checkpoint_path, output, split="test", device="cpu"):
    model, state = load_model(checkpoint_path, device)
    results = []
    for directory, clip, truth in discover(dataset):
        if clip["split"] != split:
            continue
        result = infer_clip(model, directory, clip, device, state["budget_rate"], state["threshold"], state["consensus"])
        prediction = [i for seq in result["event_sequences"] for i in seq]
        results.append({"clip": clip["id"], "retained_fraction": len(result["selected_indices"])/len(clip["frames"]),
                        "tolerance_metrics": {str(d): match_metrics(prediction, truth, d) for d in range(4)}})
    if not results:
        raise ValueError(f"No reviewed {split} clips")
    write_json(output, results)
    return results
