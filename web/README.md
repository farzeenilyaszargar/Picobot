# PicoBoi frontend

A buildless chat page with the trained Picoboi-100K model running in a browser worker. No inference server, third-party runtime, analytics, or persistent conversation storage is used. Serve over HTTPS or localhost for worker loading and checksum verification.

## Preview

From the repository root:

```sh
python3 -m http.server 4173 --bind 127.0.0.1 --directory web/dist
```

Open http://localhost:4173. Once the status says On device, try Find the ball, Count apples, or Fix a sentence. Replies stream from the worker. New chat clears history and cancels outstanding generation.

## Inference

- `dist/models/picoboi-100k-v1/`: actual FP32 checkpoint weights, tensor manifest, tokenizer.
- `dist/inference.js`: causal transformer, per-stage KV caches, RMSNorm, RoPE, SwiGLU, tied embeddings, greedy decoding.
- `dist/model-worker.js`: same-origin loading, SHA-256 verification, generation and cancellation.
- `dist/normalize.js`: browser-local greeting aliases, contraction expansion, curated spelling fixes, and conservative edit-distance suggestions.
- `dist/app.js`: chat state, streamed replies, errors, model status, and navigation.

Weights contain 99,120 learned parameters in 396,480 raw bytes. `training/export_browser.py` exports them from the trained checkpoint. Answers are not hard-coded or corrected after generation. Application notices are separate from model output.

Before tokenization, user input goes through a deterministic normalizer. Whole greetings such as `yo`, `hi`, and `whats up` become `hello.`. Contractions expand, common typos are corrected, and explicit aliases map to supported words. The user sees the original input plus an `I read that as:` notice for rewrites. Successful normalized input is retained in model history. Model output is never normalized. Other recognized words retain their tense and form; negation is preserved. Broad one-edit spelling matches are suggestions, not automatic guesses. Unlisted names, synonyms, and ambiguous abbreviations such as `ur` can still be rejected. Bare ambiguous forms such as `its`, `were`, and `well` are not treated as contractions.

The tokenizer still supports the same 800 word forms plus 12 punctuation/structural tokens. The checkpoint, tokenizer IDs, and corpus have not changed. Unsupported words, digits, and characters receive a notice. Context includes history and generated output. At least 32 tokens are reserved for an answer; older complete conversation pairs are omitted as needed, with a notice. Oversized messages are rejected. Replies stop on a learned turn/EOS token or length/context limits. The normalizer improves input tolerance; it does not improve the model's underlying greeting skills, eliminate ambiguity, or reduce its weight size. Contraction expansion can increase token count.

## Verification

```sh
.venv/bin/python training/export_browser.py
node training/tests/browser-inference.mjs
node training/tests/normalize-input.mjs
```

Fixtures contain PyTorch logits and greedy answers. Tests check numerical agreement, the binary checksum, vocabulary rejection, context trimming, and cancellation/reuse. Browser checks cover real generation, follow-up questions, unsupported words, and New chat. See the model card for capability limits. This implementation uses JavaScript/CPU and does not promise offline page loading after the tab closes. Only static model assets are downloaded; conversation messages are not sent to an inference service.
