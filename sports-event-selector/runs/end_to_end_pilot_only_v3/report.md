# End-to-end commentary comparison

Both systems receive only the same 24 DL-pilot test clips (192 seconds). No other match footage is sent.

| Metric | DL + tuned VLM + tuned LLM | Full-video baseline |
|---|---:|---:|
| semantic_soft_recall | 0.8947 | 0.9255 |
| semantic_soft_precision | 0.8611 | 0.3501 |
| semantic_soft_f1 | 0.8776 | 0.5080 |
| exact_duplicate_fraction | 0.3333 | 0.6865 |
| temporal_tolerance_seconds | 3.0000 | 3.0000 |
| reference_events | 24.0000 | 24.0000 |
| predicted_segments | 33.0000 | 185.0000 |
| mean_clip_rouge_l | 0.3784 | 0.3151 |
| judge_weighted_event_recall | 0.2188 | 0.2292 |
| judge_full_event_recall | 0.0000 | 0.0000 |
| judge_weighted_precision | 0.1591 | 0.0297 |
| judge_contradiction_fraction | 0.2879 | 0.0838 |
| judge_unsupported_fraction | 0.3939 | 0.8568 |
| judge_mean_fluency | 3.8571 | 4.2083 |
| judge_mean_coherence | 2.9286 | 3.5208 |
| judge_wins_labeled_clips | 8.0000 | 6.0000 |
| judge_labeled_passes | 14.0000 | 14.0000 |

Judge: gemini-3.5-flash; two passes with candidate order swapped. Winner agreement: 58.3%.
Embeddings: gemini-embedding-001; cosine scores use same-clip candidates within ±3 seconds.
Cosine scores are soft best-match similarity, not factual accuracy; judge matching is one-to-one.
ROUGE-L measures wording overlap. Duplicate fraction is exact text repetition within each clip.

Reused development match; synthetic label-derived reference.
Baseline sees each complete eight-second clip; pipeline sees selected windows. Both are offline, not a streaming-latency test.
Single invalid VLM output is skipped, not repaired using labels.
No ground-truth events or commentary enter either generation path.
Judge sees sparse labels, not video; unsupported is not necessarily false.

See usage.json for tokens, timings, and explicitly estimated USD costs.
