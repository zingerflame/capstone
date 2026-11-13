# Third experiment: pretrained features, motion and frame budgets

Six CPU training runs completed: two architectures, seeds 17/29/43, 20 epochs each. The primary variant was frozen from mean validation F1 before evaluating either test collection. It is the motion variant with an expanded frame budget. Its fresh-test F1 averaged **0.419**, compared with **0.200** for the previous v2 checkpoint on the same clips. This remains a small pseudo-labeled pilot.

## Data and teacher

Ten official Australian Open full-match broadcasts were downloaded (14.95 hours). Training still uses only 72 seconds from six matches; validation uses 24 seconds from two other matches. This experiment adds five test clips totaling 28 seconds, 264 retained frames, six labeled events and 18 anchor frames. They come from later intervals of the same two previously held-out matches, not new independent matches. The earlier test collections total 60 seconds.

Across all three collections, 31 clips / 184 seconds / 1,720 retained frames have teacher annotations. ChatGPT visually reviewed contact sheets and wrote explicit event and negative annotations. This is direct visual annotation in the conversation, not an automated teacher API run or independent human ground truth. `examples/tennis_third_teacher.json` records frame indices, descriptions, confidence and limitations. The clip named `third_gauff_waiting` actually contains replay/preparation context; its negative annotation documents that distinction. Timing is approximate and replay identification can require context beyond a student's ten frames.

## Changes and controls

1. Frozen ImageNet ResNet-18 features replace the scratch visual encoder. Full broadcasts use a 224-square letterbox, preserving edges. Features are cached with hashes of image bytes, metadata, the preprocessing recipe and verified official weights. Only the temporal head trains.
2. The motion arm adds differences between adjacent normalized scene features inside each ten-frame window. This is a simple motion cue, not optical flow or ball tracking. The appearance arm provides the controlled comparison.
3. Both arms compare the original frame budget with validation selection among 1x, 2x and 3x budgets. Every chosen expanded checkpoint used 3x. The cap changes globally; the per-window top-six and minimum-three-window overlap rules remain unchanged. Mandatory first two and last two frames remain included.
4. Training uses the prior v2 loss, including distance targets within three filtered frames, balanced positive neighborhoods, corrected duplicate penalties and gradual count penalties.

Shared temporal settings: width 64, two transformer layers, four heads, AdamW learning rate 0.0003, 20 epochs. Validation checks thresholds 0.1–0.7 and consensus 0.5/0.65/0.8 every epoch. The original annotation-derived budget rate is about 0.1166 for nonmandatory frames. Expanded budget selection is an explicit relaxation of compression, not a free quality gain. Selection uses validation F1 with fewer predictions as a tie-breaker. No test labels determine a budget.

Compared with the previous scratch model, pretraining, scene resolution and scoreboard representation all change. Therefore that comparison cannot isolate pretraining alone. The scoreboard is a grayscale visual crop, not OCR or verified score-change extraction. The motion arm adds parameters. This is inspired by temporal event-spotting work, not a reproduction of E2E-Spot or T-DEED.

## Fresh test results

One-to-one anchor matching within ±3 retained-frame indices. F1 is frame-anchor F1, not event-caption accuracy. New models show mean ± population standard deviation over three seeds. Previous model is a single existing checkpoint. Retention includes mandatory endpoints; event metrics exclude endpoints unless they belong to predicted event sequences.

| Model | F1 | Precision | Recall | Retained frames |
|---|---:|---:|---:|---:|
| Previous v2 | 0.200 | 0.250 | 0.167 | 12.1% |
| Pretrained appearance, fixed budget | 0.428 ± 0.239 | 0.457 | 0.407 | 13.1% |
| Pretrained motion, fixed budget | 0.208 ± 0.069 | 0.257 | 0.185 | 13.0% |
| Pretrained appearance, expanded budget | 0.459 ± 0.080 | 0.318 | 0.833 | 25.6% |
| **Pretrained motion, expanded budget (primary)** | **0.419 ± 0.053** | **0.319** | **0.648** | **21.3%** |

The appearance-expanded variant scored higher on this test, but it does not replace the validation-selected primary after looking at test scores. The primary's mean validation F1 was 0.776. Exact-index fresh-test F1 is only 0.166, illustrating uncertain timing. Its ±3 F1 values are recorded per seed in `runs/tennis_third/comparison.json`; variation over three seeds is not a confidence interval over unseen matches. The periodic baseline scores 0.222 on this fresh collection.

On the pooled earlier test clips, primary mean F1 is 0.471 ± 0.032 versus previous v2 0.295, with primary retention 27.5%. These are regression checks on previously inspected footage. The motion cue did not improve the fixed-budget fresh result, so it is not a generally proven improvement.

## Workflow

```mermaid
flowchart LR
 A[Official full-match broadcasts] --> B[Short review intervals]
 B --> C[Frame delta and timestamps]
 C --> D[ChatGPT visual teacher]
 D --> E[2–3 frame events and explicit negatives]
 C --> F[Frozen scene features and scoreboard crop]
 E --> G[Train ten-frame temporal head]
 F --> G
 G --> H[Validation selects checkpoint and budget]
 H --> I[Freeze choice before test]
 I --> J[Top six votes and overlap consensus]
 J --> K[Budget and 2–3 frame sequences]
```

## Next experiments

- Expand labeled training across full matches and independently review ambiguous contact frames and replays. Preserve new matches for a final untouched test.
- Select the operating point with an explicit retention ceiling or recall-versus-cost curve, rather than F1 alone. Fix the 25-fps sampling quantization that can reduce nominal 10-fps minimum to about 8.33 fps, then regenerate annotations before retraining.
- Compare the same input resolution and scoreboard branch with random versus pretrained weights to isolate pretraining. Try multiscale temporal context while preserving the requested ten-frame decision interface.
- Evaluate scoreboard OCR/change features and replay detection separately. Add one-to-one event-sequence accuracy, temporal error in seconds, and final caption quality.

## Reproduction

From the repository with dependencies installed and downloaded videos/annotations present:

```powershell
python -m sports_events.experiment3 run --output runs/tennis_third_reproduction
python -m sports_events.experiment3 predict --checkpoint runs/tennis_third/motion/17/expanded.pt --clip data/tennis_third/clips/third_gauff_rally --output runs/tennis_third/inference_demo.json
```

The inference command reads frame metadata and images without reading teacher annotations. Seed 17 is the predetermined representative checkpoint, not the best seed chosen on test. Official ResNet weights must be available at `data/pretrained/resnet18-f37072fd.pth`; `pretrained.py` verifies their SHA-256. Downloaded media, feature caches, checkpoints and full metrics remain local and ignored by git.

## Primary research sources

- [He et al., Deep Residual Learning for Image Recognition, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html): visual feature backbone.
- [Hong et al., Precise Event Spotting in Sports Videos, ECCV 2022](https://arxiv.org/abs/2207.10213): event spotting with local motion and temporal context.
- [Xarles et al., T-DEED, CVPR Workshops 2024](https://arxiv.org/abs/2404.05392): temporal discriminability and multiscale context motivate future controlled experiments.
- [PyTorch ResNet-18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) and [official implementation/weight registry](https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html): pretrained parameters. Our letterbox preprocessing differs from the official center-crop recipe.

The repository's original research document contains the broader literature review. These sources explain the new experiments, not evidence that this pilot generalizes.
