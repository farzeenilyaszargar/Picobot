# Parallel tokenizer experiments

Both variants preserve the same 1,290 canonical conversations containing 103,749 English word occurrences. Source files and the active word-level contract remain unchanged. These are tokenizer and prepared-data artifacts, not newly trained models.

| Variant | Model vocabulary | Packed tokens | Longest conversation |
| --- | ---: | ---: | ---: |
| word-level | 988 IDs, including 968 words | 143,505 | 586 |
| bpe-512 | 512 IDs, including 492 pieces | 171,414 | 839 |

Both include 20 reserved names: seven controls, five punctuation marks, and eight parser references. The first 12 IDs retain the same order; reference and word IDs differ between variants. Weights cannot be shared across the two tokenizers.

BPE uses lowercase English characters and a suffix word-end marker `▁`. It is not byte-level BPE. The vocabulary includes all 26 letters, the marker, and learned merges. Its pieces reflect corpus frequency, not guaranteed roots or suffixes. It encodes unseen ASCII words, but this does not make the concept parser understand unsupported words. The parser uses the preserved 968-word semantic lexicon via `word_vocabulary`, independently of the model's piece IDs.

At embedding width 48, tied input/output embeddings contain 47,424 parameters for word-level versus 24,576 for BPE. BPE saves 22,848 embedding parameters while producing 19.45% more tokens here. This is an in-sample compression result; no trained quality comparison has been performed.

Recreate both tokenizers and packed files from the repository root:

```sh
python3 training/prepare_tokenizer_variants.py
python3 -m unittest discover -s training/tests -p test_tokenizer_variants.py -v
```

Packed streams are local, ignored by Git, under `training/data/conversation-variants/<variant>/`. Each contains little-endian uint16 IDs, a uint8 assistant-target mask, and conversation offsets. Checked-in manifests record source, tokenizer, stream, and mask hashes. No conversation is truncated, and every complete conversation fits 1,024 tokens. These streams have no new validation split; create a conversation-level split before training and fit the final BPE on training data only for a fair comparison. Existing held-out cases are excluded from these streams.

The mask is aligned with token IDs. For next-token training, use input `ids[:-1]`, target `ids[1:]`, and loss mask `mask[1:]`. Assistant content, assistant end-of-turn, and final EOS are supervised. User content and role markers are not.

Python codec:

```python
from training.tokenization import load_variant
codec = load_variant('bpe-512')  # or 'word-level'
ids = codec.encode('i know the facts.')
assert codec.decode(ids) == 'i know the facts.'
```

Browser codec: import `createTokenizer` from `../tokenization.mjs` and pass one of these JSON specifications. Canonical text must be lowercase ASCII words, the five punctuation marks, and valid parser references. Run input normalization and the concept parser before encoding, then decode model output before restoring bound terms.

The live website still uses its existing trained model. A future run must explicitly choose its tokenizer, initialize matching embeddings, and save that exact tokenizer alongside its weights.
