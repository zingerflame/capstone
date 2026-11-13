# Seven-slide speaking script

MIE 429 / PROGRESS UPDATE. Approximately 20 seconds per slide. Cost, duration and latency values on slide 1 are modeled estimates, not measured full-corpus outcomes.

## Slide 1 — Why add a deep learning model?

The selector sends fewer pictures to Gemini. In a matched ten-frame-per-second scenario, estimated processing charges fall from about forty-two dollars to nine dollars for fifteen hours of footage. The retained pictures equal about three hours and twelve minutes at that same rate. An illustrative latency model suggests fifty-five percent less VLM processing time.

## Slide 2 — How the model works

The model has three main parts. ResNet-eighteen supplies existing visual knowledge, frame differences highlight movement, and a small Transformer connects nearby action. We keep the visual encoder frozen and train the selector. Overlapping ten-frame windows vote on which pictures to retain, producing short groups for later commentary.

## Slide 3 — Final pilot: exact-frame results

Using exact frame matching, the final pilot scores eighty-three percent accuracy, thirteen percent precision, and twenty-six percent recall. The accuracy is misleading because most frames are not labeled events. Rejecting every frame scores even higher. We need more reliable labels and better timing before this model is ready.

## Slide 4 — Three model iterations

We began with a visual model trained from scratch, but it missed the action. Next, we balanced the training penalties so useful frames were less strongly suppressed. Finally, we reused pretrained visual features, added motion cues, and tested a larger frame allowance. Validation selected the final pilot before testing.

## Slide 5 — Training pipeline

We download matches and filter repeated frames. ChatGPT labels short event sequences. A frozen visual encoder converts the pictures into features, and the selector learns from nearby matches and missed events. We choose settings on separate validation matches, freeze that choice, then evaluate the final model on test clips.

## Slide 6 — Tennis example: building key events

This filmstrip uses the Zheng versus Sabalenka training footage. ChatGPT labeled a serve, a return, and the next rally shot. Each highlighted picture represents a short group of two or three frames. These examples teach the selector which moments are useful; they are teacher annotations, not model predictions.

## Slide 7 — Next steps

The next priority is stronger evidence. We need more independently reviewed examples and tests on matches the system has never seen. We should improve timing, recognize replays, and use the scoreboard more reliably. Finally, a measured Gemini comparison must check actual cost, latency, and commentary quality at equal sampling settings.

