# Picoboi 100K: controlled chat corpus

This is a reproducible synthetic baseline for a model with about 100,000 parameters. It teaches a narrow selection of conversational patterns. It is generated from authored patterns and grammar-aware slot combinations, not 10 million independently written LLM tokens. No external text dataset or paid generation API is used.

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

Start by training and evaluating a small run before spending on a longer run. Compare assistant-only loss against full-sequence loss, and report both held-out loss and human checks of grammatical answers, context recall, and repetition. No model has been trained by these scripts, and the website has not been connected to a model.
