# Picoboi 100K: controlled chat corpus

This is a reproducible synthetic baseline for a model with about 100,000 parameters. It teaches a narrow selection of conversational patterns. It is generated from authored patterns and grammar-aware slot combinations, not 10 million independently written LLM tokens. No external text dataset or paid generation API is used.

The proposed model architecture, exact parameter budgets, and experiment choices are in [MODEL_DESIGN_100K.md](MODEL_DESIGN_100K.md). Inspect the configuration specifications with `python3 training/inspect_architecture.py`. These specifications do not contain trained weights or an inference implementation.

## Data contract

- Exactly 800 permitted lowercase English word forms, frozen in `vocabulary_800.txt`. Inflections count separately. This is a curated conversational list, not a measured frequency ranking of the top 800 English words.
- Five punctuation tokens: `. , ? ! :`.
- Seven structural tokens: `<pad> <unk> <bos> <eos> <user> <assistant> <turn>`.
- Total model vocabulary: **812 token IDs**. Unknown content is rejected during preparation; it is not silently mapped to `<unk>`.
- Lowercase text only, with no contractions. Every dialogue contains alternating user and assistant messages and ends with a complete answer.
- The target of 10 million tokens counts words, punctuation, and chat structure. It does not mean 10 million English word occurrences.

Content includes location tracking, possession, short plans, descriptions, short stories inside chat, tense changes, negation, sentence repair, preferences, basic supportive conversation, clarification, small-number counting, and explicit word copying. Intentionally incorrect grammar occurs in user messages requesting sentence repair.

## Included artifacts

The checked-in v1 corpus has **10,000,088 model tokens**, **6,672,057 English word occurrences**, and **145,953 unique dialogues**:

| Split | Model tokens | Dialogues |
| --- | ---: | ---: |
| Train | 9,800,070 | 143,056 |
| Validation | 100,015 | 1,452 |
| Test | 100,003 | 1,445 |

Of the 800 allowed words, 490 occur outside copying exercises and 310 occur only in copying exercises in the training split. The longest dialogue has 107 model tokens.

`data/picoboi-800-v1/` contains compressed chat JSONL shards, a tokenizer specification, the frozen vocabulary, a 200-dialogue inspection sample, statistics, checksums, and an independent audit. Each JSONL record includes messages, a content hash, family, pattern, semantic group hash, and token count. The whole chat is the training sequence; no private reasoning or hidden chain of thought is included.

Packed binary files are available locally and ignored by Git because they can be reconstructed exactly from the included shards:

```sh
python3 training/validate_corpus.py --prepare
```

This restores `train.bin`, `validation.bin`, `test.bin`, their assistant masks, and their dialogue offsets, then validates all artifacts. Python 3.10 or newer is sufficient; there are no third-party dependencies. Run without `--prepare` to validate existing packed files. Assertions perform the audit, so do not invoke Python with `-O`.

Tokens are little-endian uint16; masks are uint8. Offsets are token offsets, not byte offsets. Each unshifted sequence is:

```text
<bos><user>content<turn><assistant>content<turn>...<eos>
```

Each mask entry identifies whether its token is an assistant target. To train next-token prediction, use `tokens[:-1]` as inputs, `tokens[1:]` as labels, and **`mask[1:]`** as label weights. Role markers and user tokens have zero weight; assistant content, assistant turn endings, and the final EOS have weight one. Keep dialogue boundaries when batching, and mask padding and cross-dialogue attention. All current examples fit within 128 tokens. With 812 embeddings, a tied output head will help preserve the small parameter budget. A word-level model cannot directly represent arbitrary user vocabulary.

## Regenerate

```sh
python3 training/build_corpus.py --output-dir /tmp/picoboi-new-corpus
python3 training/validate_corpus.py /tmp/picoboi-new-corpus
```

The default targets are 9.8 million training tokens, 100,000 validation tokens, and 100,000 test tokens. Complete dialogues cause a small overshoot. Generation uses a fixed seed. Elapsed runtime in the report can differ between runs; data payloads are deterministic with the same generator and vocabulary. Initial vocabulary creation uses `--init-vocab`, which refuses to replace an existing list. The frozen list is the source of truth for subsequent builds.

## Evaluation limits

Exact duplicate dialogue texts are removed globally. Semantic groups are assigned to splits by a stable hash, so the same group cannot cross splits. Template families are shared, individual turns can repeat, and closely related combinations can appear across splits. This measures performance on held-out combinations within this generator, not general conversational ability. Keep an independently authored human test set before making quality claims.

The generator uses 498 word forms in its initial grammar inventory. Some words occur only in explicitly labeled copying exercises; the report lists these. Copying a word does not teach its meaning. This baseline must not be presented as semantically grounding all 800 words. The audit verifies vocabulary, formatting, deduplication, split assignment, packed files, selected language invariants, and counting arithmetic. It does not prove that every generated conversation is natural or that a trained model will chat well.

Compare assistant-only loss against full-sequence loss, and report both held-out loss and human checks of grammatical answers, context recall, and repetition. The website is not connected to the local training pipeline.

## Local model training

The implementation in `model.py` has exactly 99,120 trainable parameters, including embeddings and norms. Its two distinct blocks execute A, B, A, B. It uses the existing frozen corpus without adding identity dialogues or changing vocabulary. `train.py` uses complete individual dialogues, causal attention, trailing padding, correctly shifted assistant masks, clipped gradients, AdamW, learning-rate warmup and cosine decay, validation, and early stopping. Every epoch shuffles all training dialogues without replacement. Validation uses all validation dialogues; throughput pilots and learning-rate pilots are short diagnostics, not claims of global optimality.

Install into a project-only environment:

```sh
uv venv --python python3.13 .venv
uv pip install --python .venv/bin/python -r training/requirements-local.txt
.venv/bin/python -m unittest discover -s training -p test_training.py -v
.venv/bin/python training/train.py --benchmark
.venv/bin/python training/train.py --auto --epochs 3
```

The benchmark compares CPU thread counts, CPU versus available MPS, batch sizes, and two initial learning rates using shuffled corpus examples. `--auto` applies its selected settings. On the initial laptop benchmark, CPU with four threads, batch size 64, and peak learning rate 0.003 was selected. Sustained speed and quality can differ from the pilot.

Local outputs are under `training/runs/picoboi-100k-local/`, ignored by Git. `last.pt` contains model, optimizer, deterministic epoch/batch cursor, and settings for resume; `best.pt` tracks the lowest observed validation loss. Checkpoints are written through a temporary file and atomic rename. `metrics.jsonl`, `status.json`, `settings.json`, and `benchmark.json` record the run. The agent's detached launch additionally writes `train.log` and `process.json` and uses `caffeinate -i` only while training runs.

Resume with the same settings:

```sh
.venv/bin/python training/train.py --auto --epochs 3 --resume
```

SIGINT/SIGTERM request a checkpoint after the current update. Do not launch two trainers into the same run directory. The test suite checks causality, prefix equivalence, loss-mask behavior, actual parameter count, and exact CPU continuation after interruption. Runs use FP32 and fixed seeds, but cross-device bitwise reproducibility is not promised.

After completion, evaluate the best checkpoint on the held-out test split and optionally export a compact model checkpoint:

```sh
.venv/bin/python training/evaluate_checkpoint.py --export training/checkpoints/picoboi-100k-v1.pt
.venv/bin/python training/chat.py 'the red ball is on the table. where is it?'
```

`chat.py` is a CPU debugging tool with greedy decoding and a restricted set of output tokens. It recomputes the prefix and is not the optimized browser inference engine. It rejects unknown words and unsupported characters. Only load checkpoints you trust: these local tools use PyTorch checkpoint deserialization. Deployment and quantization are separate steps; a trained checkpoint alone does not connect the website.
