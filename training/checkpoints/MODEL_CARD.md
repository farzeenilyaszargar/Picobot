# Picoboi 100K v1

Trained locally from random initialization on 21 October 2026. This is a controlled-language research baseline, not a general-purpose chatbot. The website is not connected to this checkpoint.

## Model and training

| Property | Value |
| --- | --- |
| Trainable parameters | 99,120, including embeddings and normalization |
| Architecture | Width 48, two unique blocks executed A, B, A, B |
| Attention | Two heads, causal full attention with RoPE |
| Feed-forward | SwiGLU, width 144 |
| Embeddings | Shared input/output weights |
| Vocabulary | 800 lowercase word forms plus 12 structural/punctuation tokens |
| Context | 128 tokens including history and generated answer |
| Precision | FP32 |
| Hardware | Local Apple silicon laptop, 8 GB memory; CPU, four threads |
| Framework | PyTorch 2.14.1, Python 3.13 |
| Batch size | 64 complete dialogues, dynamically padded |
| Optimizer | AdamW, betas 0.9/0.95, matrix weight decay 0.01 |
| Peak learning rate | 0.003, 5% warmup, cosine decay to 10% |
| Gradient clipping | Norm 1.0 |
| Training | Three shuffled epochs, 6,708 optimizer updates |
| Training input positions | 28,971,042, excluding trailing padding |
| Supervised assistant targets | 12,299,670 |
| Initialization/data-order seed | 20261021 |
| Export size | 405,367 bytes, including checkpoint metadata |

The corpus was not changed for this training run. Its generator produces variations of authored dialogue templates. Of the 800 words, 310 occur only in copying exercises in the training split. The brand name is not part of this frozen vocabulary. Short CPU/MPS, batch-size, thread-count, and learning-rate pilots selected practical settings. This is not an exhaustive architecture or hyperparameter search, and the alternative architectures have not been trained for comparison.

The run was checkpointed and resumed once to correct best-checkpoint tracking to retain every strict validation improvement. Optimizer state and the next epoch/batch position were retained; no training examples were skipped or repeated by the resume. The test suite independently checks exact CPU continuation against uninterrupted training on a small run. Cross-device bitwise reproducibility is not claimed.

## Evaluation

The best checkpoint was the final update, selected using validation loss. The held-out test split was evaluated after the training run finished.

| Measurement | Result |
| --- | ---: |
| Validation assistant-token cross entropy | 0.026291 |
| Test assistant-token cross entropy | 0.023298 |
| Test assistant-token perplexity | 1.02357 |
| First-answer exact match, first 100 test dialogues | 85 / 100 |

Loss is measured only on assistant content, assistant turn endings, and final EOS. It uses teacher-forced full dialogue context. The generation test instead supplies only the first user message and decodes the first answer greedily, forbidding pad/unknown/input-role tokens. The 100-dialogue sample shares generator families with training; it is not a natural-chat benchmark. Fifteen examples request stories with arbitrary generated details, so their single expected answer is not a unique valid answer. None of those story outputs exactly matched the reference. Full examples and family counts are included in the evaluation JSON.

Selected natural-prompt probes, including failures:

| User prompt | Greedy model answer |
| --- | --- |
| the red ball is on the table. where is it? | the ball is on the table. |
| i have two apples. i get three more. how many do i have now? | you have five apples. |
| please fix this sentence: he are happy. | he is happy. |
| hello. | say. |
| who are you? | the park. |
| the blue book is in the box. what color is it? | the book is is blue. |
| where is my book? | my friend is is is is is in the phone now. |

The model learned the controlled patterns well but does not reliably handle unfamiliar phrasing, greetings, identity, or missing information. Low held-out token loss substantially overstates its usefulness as an unrestricted chatbot. Further work should emphasize grounded conversational data and independently authored evaluations before spending on more epochs or making capability claims. Those data changes have not been made.

## Files and reproduction

- `picoboi-100k-v1.pt`: FP32 model state and training/configuration metadata, without optimizer state.
- `../results/picoboi-100k-v1/evaluation.json`: full evaluation, generated probes, sample references, and export checksum.
- `../results/picoboi-100k-v1/benchmark.json`: local speed and learning-rate pilots.
- `../results/picoboi-100k-v1/settings.json`: exact settings and corpus fingerprint.
- `../results/picoboi-100k-v1/metrics.jsonl`: training and validation history.
- `../results/picoboi-100k-v1/sha256.json`: checksums for the exported model and results.

The local `training/runs/picoboi-100k-local/` folder retains resumable optimizer checkpoints and logs, ignored by Git. See `training/README.md` for installation and training commands. To inspect the exported checkpoint from the repository root:

```sh
.venv/bin/python training/chat.py 'the red ball is on the table. where is it?' --checkpoint training/checkpoints/picoboi-100k-v1.pt
```

This debugging CLI recomputes the full prefix, rather than implementing browser KV caching. Unknown vocabulary is rejected. This artifact is not a browser-ready model export, quantized model, or deployed service.
