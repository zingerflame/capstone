# Design and mathematical contract

All file and code indices are zero-based. Distances are measured after delta filtering; original frame indices and timestamps remain available for video retrieval and temporal evaluation.

## Sampling

For each decoded image, compute normalized grayscale mean absolute difference at 64×36 against the last selected image. Select when at least 1/max_fps has elapsed and delta exceeds the threshold, or when 1/min_fps has elapsed. Preserve the first and last two original frames as exceptions. Sampling is streaming in pixel memory; only two tail frames are held. Original-resolution selected JPEGs are written incrementally. FPS bounds are approximate because input frames arrive at discrete times; low-FPS sources cannot supply missing frames. Decode timestamps are preferred, with nominal-FPS fallback when decoder timestamps do not advance.

## Teacher and student

The teacher first proposes vocabulary for all nine sports. It then sees selected full views and optional scoreboard crops in overlapping chunks with explicit global frame IDs and timestamps. It writes 2–3-frame event descriptions. Chunk-core midpoint ownership reduces repeated detections from overlap; exact frame-set deduplication removes identical proposals. This cannot resolve all near-duplicates. Manual review is the quality gate.

The student receives images, optional scoreboard imagery, presence masks and time deltas. It receives **no text or event-type input**. Scene and scoreboard CNNs produce 64-dimensional embeddings; their concatenation with two metadata values is projected to 64 dimensions. Two Transformer layers operate on ten embeddings plus learned position embeddings. A shared linear head gives ten Bernoulli logits. This is multi-label selection, rather than a softmax over ten mutually exclusive choices.

Image and temporal batches use non-reentrant gradient checkpointing during training. All window logits are aggregated for the complete clip before calculating the count loss; no artificial training-chunk boundaries are added. This trades image rereads and recomputation for activation memory. Saved embeddings, logits and metadata still grow linearly with video length; full broadcasts need substantial storage and a measured hardware budget. `batch_size` in `clip_logits` controls recomputation batches, not independent label boundaries.

## Voting and hard selection

For N ≥ 10, windows start at s = 0…N−10. Each window predicts probabilities p[s,j] for j = 0…9. Keep at most the six highest probabilities at or above 0.5; ties prefer lower local indices. For global frame i:

```
support[i] = number of distinct windows containing i
agreement[i] = positive window votes for i / support[i]
confidence[i] = mean window probability for i
```

An optional frame is eligible only if support ≥ 3 and agreement ≥ 0.5. Sort eligible frames by descending agreement, then confidence, then ascending index. Keep within the available budget after reserving mandatory endpoints. This interprets “highest overlap” as **highest normalized agreement** rather than largest raw number of votes, which would favor interior positions merely because they have more windows.

Group retained optional indices into runs with successive gaps ≤ 3. Split runs into groups of 2 or 3, using 2+2 instead of 3+1 for a run of four. Discard singleton optional groups. No grouping step invents or adds frames. Forced endpoints are returned separately as context, never presented as verified sports events. The final retained-frame list is their union with event proposals. Output budgets can therefore be underfilled.

For N < 10, pad one window with the last frame and mask the padding. It still counts as **one** window. For N = 10 or 11, no optional frame can satisfy three-window support. N = 12 has three windows but some nonmandatory positions near either end still have insufficient support. Return those positions in `unsupported_interior`; do not fabricate windows. Mandatory endpoints override support and budget constraints, and are added **after** the per-window six-positive cap. A one-frame clip has context but no valid 2/3-frame event.

## Distance-aware supervision and backpropagation

Let G be the union of unique teacher-selected frame indices. Define:

```
d[i] = min(abs(i-g) for g in G)
y[i] = 1 - d[i]/4 if d[i] <= 3, else 0
```

If G is empty, every target is zero. Use maximum proximity to any true frame; do not sum neighborhoods. Window BCE is weighted by reciprocal frame support so central frames are not counted ten times more heavily than boundary frames.

The loss combines:

```
L = distance_soft_BCE + 0.5 * excess_count + 0.2 * coverage + 0.1 * duplicate_mass
B = |G union mandatory_endpoints|
excess_count = (relu(decoded_count_without_budget - B) / max(1,B))**2
```

The decoded count uses the same top-six, overlap and sequence-grouping code as inference, but with no cap. Otherwise a hard budget would make the excess penalty identically zero. For backward computation, a straight-through estimate replaces hard window votes with sigmoid gradients; a smooth majority function estimates final selection mass. Its count is substituted only in the derivative. This is an **optimization surrogate**, not exact differentiation through sorting/grouping.

Coverage penalizes missed teacher frames using the best probability weighted by distance inside ±3. Duplicate mass assigns each predicted position to its single nearest teacher frame and penalizes assigned probability mass above one. It discourages selecting several nearby substitutes for one target. It is a soft regularizer, not a proof of one-to-one correspondence; evaluation uses actual one-to-one matching to prevent repeated credit.

Mandatory context is included in the count budget but does not become a positive event label. When a true event is at an endpoint, it remains in supervision; the current grouping decoder returns that endpoint as context and will therefore lose event-metric credit there. This is an explicit conservative limitation of separating required context from detected events.

## Budget without inference-label leakage

Estimate r from training clips only:

```
r = total unique teacher positives outside forced endpoints
    / total available nonendpoint frames
inference_budget = forced_count + round(r * nonendpoint_frame_count)
```

The number of teacher events enters through the union of their 2/3-frame selections; overlapping events do not inflate the budget by counting a frame twice. This is a fixed learned density prior, not an oracle for the unseen video's event count. The final count penalty discourages unnecessary frames during learning; ranking applies the density budget at inference. High-action videos may be underselected. Future work should compare per-sport rates and a learned count estimator, calibrated only on validation data.

## Evaluation

Hold out entire matches, not overlapping windows. The loader rejects group IDs spanning splits. Report one-to-one frame precision/recall/F1 at tolerances 0, 1, 2 and 3, plus retained fraction. Boundary context earns no event credit. These are selector diagnostics, not the published class-aware mAP metrics from sports spotting papers. For real experiments add event-level recall, temporal-distance-in-seconds metrics, per-sport/channel breakdowns and human-verified test annotations. Compare against uniform sampling and delta-only selection at equal output budgets before claiming improvement.
