# Picoboi 10M training plan

Train a small English conversational model from scratch for laptop browsers. Aim for grammatical short replies, useful follow-up questions, and correct references to entities mentioned earlier in a chat. Use imported Hugging Face datasets and the existing Picoboi conversations. The initial experiment tests whether a local dictionary helps a 10M decoder handle names it never saw in training.

[Local preparation and training instructions](../training/PICOBOI_10M.md)

## Decoder

| Component | Starting configuration |
| --- | --- |
| Decoder | 12 distinct blocks, width 256, pre-RMSNorm, RoPE |
| Attention | Four query heads, two key/value heads, head width 64 |
| Feed-forward | Bias-free SwiGLU, width 704 |
| Tokenizer | 4,096-entry byte-level BPE, including controls and references, with tied embeddings |
| Context | 512 tokens, with most pilot examples at 128 or 256 |
| Size | 9,902,336 parameters, about 19.8 MB FP16 or 9.9 MB INT8 for weights alone |

The tied table uses 1,048,576 parameters. Each block has 196,608 attention weights and 540,672 feed-forward weights. Norms contribute 6,400 parameters in total. The batch-one FP16 key/value cache is 3 MiB at 512 tokens. Measure the tokenizer, dictionary, runtime, and any quantization scales separately in the browser download and memory budget.

## Dictionary and parser

A dictionary match turns each recognized city into a reusable model token with a conversation-local reference. A single generic place token would erase which city was the origin and which was the destination.

| Stage | Example |
| --- | --- |
| User | I'm going from Delhi to Mumbai via Paris. |
| Model input | I'm going from `<city_1>` to `<city_2>` via `<city_3>`. |
| Model output | How long will you stay in `<city_2>`? |
| Display | How long will you stay in Mumbai? |

Start with eight atomic city reference tokens. The parser stores each slot's canonical ID, original spelling, and source span. Prefer longest whole-name matches. Keep IDs stable across turns and decode only slots currently bound. A token such as `<city_2>` is a decoder symbol. The city name, aliases, identity, and any lookup data belong to the parser. The dictionary can grow without expanding the decoder vocabulary. A one-token slot can also shorten an input that otherwise takes several BPE tokens.

Preserve raw text when an alias is ambiguous, unrecognized, in a protected quote, or beyond the slot limit. Fall back to byte-level BPE for that text. Reserve controls internally so typing a literal slot cannot create a binding. An assistant target can reuse a slot introduced by a visible user turn or tool result. Assistant-only names stay in plain text. When a reply depends on city facts, give the model the relevant evidence through a dictionary lookup or retrieval. The binding by itself carries identity and type.

Reuse the existing [concept parser](../training/conversations/concept-parser.mjs) for the slot design. Run shared fixtures against the Python data pipeline and browser parser before training. Start with city names. Consider categories such as fruit once the tests cover cases where replacing a specific name loses meaning, such as comparing an apple with a banana.

## Training data and schedule

| Source | Use |
| --- | --- |
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | Simple English grammar and coherence during pretraining. Its [paper](https://arxiv.org/abs/2305.07759) demonstrates fluent story generation below 10M parameters. |
| [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) | Import short English everyday conversations, clarifications, preferences, and a range of reply lengths. Filter coding, roleplay, and long instruction answers from the initial mix. |
| [Existing Picoboi conversations](../training/conversations/README.md) | Preserve the authored files and use a deduplicated subset for the desired chat style and follow-ups. |
| Deterministic city exchanges | Build a small supplement for repeated mentions, corrections, multiple cities, and lookup results. Keep city identities and template families separate across train and test. |

1. Stream pinned Hugging Face revisions and count usable rows and tokens after filtering. Split and deduplicate before normalization, then train the tokenizer only on training text. Keep raw and parsed views of the same rows. Record source and license metadata.
2. Run a 10M-token pilot per variant using a provisional 70/30 split between simple prose and dialogue by model tokens. Train next-token prediction on both formats, with explicit role boundaries for dialogue. Use short length buckets and benchmark throughput on the Mac before estimating the larger run.
3. Continue the best runs toward 50–100M processed tokens if held-out chat improves and enough distinct data is available. Finish on filtered chat with assistant-only loss. Track unique data, repeated exposures, processed tokens, and assistant loss targets separately.
4. Export FP16 as the reference, then test INT8 in a browser worker. Measure download bytes, first-token latency, tokens per second, and peak memory on the same laptop. Compare replies against the FP16 checkpoint before testing lower precision.

## Comparison

Train raw and city-slot variants with the same 4,096-entry BPE tokenizer, reserved IDs, model shape, and underlying examples. Compare at equal example exposure and report processed token counts and wall time. If the difference is meaningful, repeat at matched compute. Separately test 2,048 entries. At width 256 this saves 524,288 parameters, and the shorter vocabulary may increase sequence length. Report any reallocation of the saved parameters in a separate comparison.

Evaluate normal chat, new city names, a city absent from the dictionary, ambiguous names, corrections across turns, and questions whose answer changes when the city changes. Track chat quality, exact entity restoration, wrong or invalid slot rates, tokens per prompt, and download bytes including the dictionary. Keep compression if it improves entity handling or efficiency while preserving everyday chat quality.
