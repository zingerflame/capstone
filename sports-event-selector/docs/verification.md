# Verification record

Executed locally on Windows, 24 October 2026.

## Automated tests

**33 tests passed** in the final run, including:

- 91 stride-one windows for 100 frames; maximum six positives per window.
- Three-window support, mandatory endpoints, short/empty event cases and 2/3-frame grouping.
- ±3 filtered-frame taper and one-to-one evaluation matching.
- Rejection of invalid, duplicate, unsorted and unseen teacher indices.
- Whole-clip count penalty and finite backward gradients.
- Gradient propagation into both scene and scoreboard encoders through checkpointed batches.
- Missing-scoreboard masking and match/source split-leakage rejection.
- Real video creation/decoding/sampling and incremental-versus-batch score agreement.
- Local mock HTTP VLM requests, all-nine-sport taxonomy, caching and preservation of reviewed corrections.
- Native Gemini multimodal/JSON request serialization, response parsing, absent-key and truncated-response handling (mock transport).
- GolfDB source grouping in the manifest adapter.

The restricted environment initially prevented the test runner's default temporary directory and loopback socket access. Tests were rerun with a workspace temporary directory and local network access. No external VLM was contacted by the tests.

Tested runtime: Python 3.12, PyTorch 2.14.1, NumPy 2.5.4, OpenCV headless 5.0.0.93, pytest 9.1.1 and SciPy 1.18.1. Dependencies were installed under the workspace's `work/deps`; `run.ps1` discovers them on this machine. Portable installation instructions are in the README.

## Executed synthetic training

`demo --epochs 2` completed acquisition from local fixtures, filtering, training, checkpoint save/load, prediction and test evaluation. Artifacts remain in `runs/demo` and are ignored by Git.

| Epoch | Training loss | Validation frame F1 |
|---|---:|---:|
| 1 | 0.8219 | 0.0000 |
| 2 | 0.7447 | 0.0000 |

This short run selected mandatory boundary context but no test event sequences. It is an execution smoke test, **not evidence that the selector has learned sports events**. Loss decreased, gradients were independently checked, and the full checkpoint/inference path executed.

## Executed GolfDB preparation

Downloaded the official `golfDB.mat` metadata (117,644 bytes) and repository demonstration `test_video.mp4` (440,224 bytes). Parsed 1,400 metadata records using SciPy, verified their expected fields, and successfully decoded/sampled the golf demonstration. Source SHA-256 hashes are saved in `data/golfdb/source.json` and the sample manifest.

The convenience launcher and CLI help were also executed successfully.

## Not executed

- Live Gemini annotation: no local Gemini/Google API key is configured.
- Download of the full preprocessed GolfDB archive or real-data student training.
- Real sports accuracy, event-level recall, cross-channel generalization or GPU throughput measurements.
- Production broadcast scoreboard extraction/OCR, pretrained-backbone integration or automatic player identification.

These are not implied by passing unit tests. The baseline guide gives the next commands once the Gemini key and training clips are available.
