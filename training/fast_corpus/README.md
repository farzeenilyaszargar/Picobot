# Fast vocabulary-controlled chat generation

This is a new teacher-generated track. It does not alter, relabel, or count the directly authored corpus as machine-generated. Existing data and website weights stay intact. The active semantic word list is still the preserved 968 words. The new stream deliberately uses no parser reference markers, so this task does not migrate the older reference protocol.

## Recommended route

Use a batched teacher model on one cloud GPU, with constrained decoding, then independently validate every completed conversation. Start with a 100-request pilot, review its language and relevance, and measure accepted words per second before a full run. A 7B instruction teacher is a starting candidate, not a validated best model. Increase teacher capacity if the pilot is grammatically weak or repetitive.

An initial compatible candidate is [Qwen2.5-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ), an Apache-2.0, 4-bit instruction model with documented vLLM serving support. Quantization reduces its teacher memory requirement; it is independent of the eventual Picoboi model's precision. The attached [Colab notebook](colab.ipynb) installs the server and runner together. You do not need an existing API account or GPU. You must select a Colab GPU runtime. Free GPU availability, hardware, runtime duration, and the ability to finish a production run are not guaranteed. See [Colab's FAQ](https://research.google.com/colaboratory/faq.html).

For predictable sustained generation, rent a 24GB or larger GPU and use the same commands below. As checked on 2026-10-21, [Runpod's pricing page](https://www.runpod.io/pricing) lists a 24GB RTX 4090 at $0.89/hour and a 48GB L40S at $1.09/hour. These are listed compute rates, not quotes or availability guarantees. Storage, startup, idle time, and review add cost. No purchase or cloud job has been started by this repository change.

## Why this should be faster

- Concurrency keeps the teacher busy serving several independent conversations. Start at 8 or 16 requests, then benchmark larger batches if memory allows. More workers do not create more GPU capacity.
- [vLLM structured outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/) can constrain generation to a regular expression. This runner builds one from the whole permitted words, punctuation, and alternating role delimiters. Unlike a per-teacher-token whitelist, it does not accidentally allow arbitrary concatenations of word fragments.
- The fixed vocabulary and system instructions are the common prompt prefix. [Prefix caching](https://docs.vllm.ai/en/v0.13.0/features/automatic_prefix_caching/) saves repeated prefill computation; it does not eliminate decoding work.
- Each response produces one complete dialogue. Only malformed, incomplete, duplicate, or overlong dialogues are rejected. There is no long serial authoring/review tool loop between responses.
- Exact and approximate near-duplicate screening keeps repeated text from inflating the count. This is not an exhaustive semantic similarity check.

Grammar-constrained decoding enforces the permitted language surface, not correct grammar, facts, relevance, or diversity. Constrained decoding may slow decoding; compare its accepted-word throughput against prompt-only generation in a separate output directory. Do not replace unsupported words with generic terms blindly. Retain the existing explicit parser policy, and review whether each simplified claim remains true.

## Setup and pilot

On a Linux NVIDIA GPU machine, install a compatible vLLM build and serve the teacher. The Colab notebook performs these steps interactively and keeps the endpoint private to its runtime.

```sh
python3 -m pip install vllm
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --host 127.0.0.1 --port 8000 --dtype half --max-model-len 6144 --max-num-seqs 16 --enable-prefix-caching
```

In another terminal, from the repository root:

```sh
python3 training/fast_corpus/generate.py --model Qwen/Qwen2.5-7B-Instruct-AWQ --base-url http://127.0.0.1:8000/v1 --workers 8 --max-jobs 100 --target-words 10000
```

`--preview` prints the first planned request and its vocabulary constraint without network calls or output files. `--no-constraints` supports compatible servers that lack vLLM regex constraints, but every output must still pass the same local vocabulary check. The optional `PICOBOI_TEACHER_API_KEY` environment variable supplies authentication to an endpoint you select. Never commit keys.

Inspect at least 100 pilot dialogues for natural English, direct answers, correct negation, factual consistency, and varied situations. Check word coverage and length distributions. Review failures by scenario rather than merely accepting a low rejection rate. Include an independent human-written evaluation set; generated validation chats are only a supplementary check. No actual GPU pilot has yet been run or reviewed.

After reviewing a successful pilot, resume the same directory with:

```sh
python3 training/fast_corpus/generate.py --model Qwen/Qwen2.5-7B-Instruct-AWQ --base-url http://127.0.0.1:8000/v1 --workers 16 --max-jobs 100000 --target-words 10000000
```

Run only one runner process per output directory. It refuses changed model, endpoint, tokenizer, seed, or generator code in an existing directory. Changing request concurrency or word target is allowed. A request budget is not a token or dollar budget. Requests already in flight finish and are recorded after the target is reached, so the final word count can exceed the target. Network retries can consume additional provider requests. Interrupting a request may cause an unrecorded completion; completed JSONL records are flushed, but a truncated last line must be inspected before resume.

The seed controls the prompt schedule, not deterministic teacher sampling. Non-retryable endpoint errors stop further submissions after outstanding requests are recorded.

## What it produces

`training/data/fast-chat-v1/`, ignored by Git:

- `config.json`: teacher, endpoint, tokenizer hashes, generator hash, seed, and constraint mode.
- `accepted.jsonl`: messages, teacher provenance, scenario group, split, English word count, and token counts under both existing tokenizer variants. New data does not enter the directly authored corpus or its audit automatically.
- `attempts.jsonl`: completed job IDs, status, rejection reason, and provider usage when reported.
- `report.json`: actual accepted count, elapsed time, new accepted words per second, and remaining-time projection. It distinguishes reaching the mechanical word target from passing quality review.

All messages alternate user/assistant and end with the assistant. Outputs with incomplete finish reasons are rejected. English words exclude role labels and metadata. Both tokenizers must encode each whole conversation within 1,024 tokens. No text is truncated. Structural tokens, punctuation, and BPE segmentation increase model token counts, so 10M English words and 10M model tokens are different targets.

The 24 task types and 12 settings create 288 prompt groups. Three groups are held out for validation and three for test, assigned before generation; length/style variants of a group stay in the same split. These labels are stored in each row, not packed into separate training streams yet. Generated groups are not proof of independent real-user generalization. The runner screens new text against existing authored and evaluation conversations. Before model training, review and export by split, fit any final BPE only on the training split, and keep any manually created test conversations fully separate.

The length mix is a starting policy: short, medium, and long exchanges. It includes greetings, preferences, clarification, corrections, plans, simple explanations, context questions, and story discussions. These are teacher instructions, not templates that fill the output dialogue. Monitor diversity at scale and extend reviewed scenario seeds if repetition rises.

## Timing and alternatives

Use accepted English words per second, not advertised teacher tokens per second:

| Measured accepted words/sec | Time for 10M words, excluding startup and semantic review |
| --- | ---: |
| 100 | 27.8 hours |
| 500 | 5.6 hours |
| 1,000 | 2.8 hours |
| 2,000 | 1.4 hours |

These are arithmetic scenarios, not measured hardware performance. Pilot quality rejection can reduce the useful rate further.

Downloading a public dataset is faster than generating fresh text, but it does not satisfy an arbitrary strict vocabulary. [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) selects more concise chats for small models; it still includes vocabulary and tasks outside Picoboi's scope. Use licensed sources as optional scenario inputs or rewrite candidates, retaining source provenance and original license obligations. Do not claim they are already strict-vocabulary dialogues. Serially rewriting every downloaded word with a teacher can cost as much as generating fresh text.

[TinyStories](https://arxiv.org/abs/2305.07759) supports the broader idea of teacher-generated simple-language datasets. Its results do not demonstrate that this particular 100K-parameter chatbot will be fluent. Evaluate Picoboi at an early checkpoint before paying for the entire corpus.

Offline checks:

```sh
python3 -m unittest discover -s training/tests -p test_fast_corpus.py -v
```

These exercise validation, duplicates, group splits, and concurrent HTTP generation/resume using an offline mock teacher. They do not test teacher quality or GPU throughput.
