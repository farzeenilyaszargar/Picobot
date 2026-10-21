# Picoboi 100K: architecture and training experiments

Research reviewed on 21 October 2026. These are proposed experiments, not measured Picoboi results. The goal is useful short conversations and reliable tracking of simple facts within a closed 800-word vocabulary. General reasoning or broad language understanding at this scale has not been established by the cited work.

## Priorities

1. Improve grounded dialogue data and build an independent evaluation set.
2. Tie input and output embeddings; avoid spending the parameter budget twice on vocabulary.
3. Compare a conventional transformer, a deeper and narrower transformer, and a shared-block transformer.
4. Establish a floating-point baseline before studying quantization.
5. Keep memory, calculations, and tool execution explicit in the browser application.

The current training split has 9,800,070 model tokens, but only 4,099,890 are supervised targets with assistant-only loss. Its 310 words used only in copying exercises need meaningful contexts. Ten million generated tokens are not ten million independent linguistic situations. Scaling-law token ratios derived from much larger models should not be treated as optimal schedules for this experiment.

## Three initial architectures

All configurations below use 812 token IDs, causal multi-head attention, bias-free projections, a SwiGLU feed-forward network, two learned RMSNorm scales per block, one final RMSNorm, rotary positional encoding without learned parameters, and tied token/output embeddings. There are no learned position embeddings or output biases. Count every trainable embedding and normalization scale in the headline parameter count.

| Candidate | Width | FFN width | Unique blocks | Attention heads | Unique parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: conventional baseline | 48 | 144 | 2 | 2 | 99,120 |
| B: deeper and narrower | 32 | 144 | 4 | 2 | 97,952 |
| C: one shared block | 64 | 128 | 1 | 2 | 93,120 |

For vocabulary size V, model width d, SwiGLU width f, and L unique blocks:

```text
parameters = V*d + L*(4*d*d + 3*d*f) + (2*L + 1)*d
```

Baseline A allocates 38,976 parameters to embeddings, 18,432 to attention, 41,472 to feed-forward projections, and 240 to normalization. An untied output matrix would add 38,976 parameters. Learned absolute positions for a 256-token context would add another 12,288 parameters at width 48. RoPE avoids that learned position table, although it still requires computation and stored or computed frequencies.

Candidate B tests depth against representational width. More layers cannot be assumed better: its narrower states may lose useful information. MobileLLM provides motivation for this comparison at 125M and 350M parameters, not proof at 100K.

Candidate C applies the same block repeatedly, initially comparing two and four passes, with six as a later experiment. Its state changes on each pass, but weights are reused. Any learned pass identifiers or extra normalization scales must be included in the count. Train the recurrence explicitly; repeating a normally trained block at inference is not an equivalent method. Shared weights do not imply a shared KV cache: distinct pass states still need distinct caches or recomputation. Recurrence buys computation at a fixed weight count, not extra stored knowledge or guaranteed reasoning. Fixed passes are the first experiment; learned halting can wait.

Start with a context of 128 tokens, sufficient for the current corpus's maximum of 107. Test 256 only after adding examples with longer dependencies. A larger advertised context does not itself teach memory.

## Data changes most likely to matter

- Give all 800 word forms meaningful uses in questions, answers, and contrasting contexts. Copy tasks alone do not ground words.
- Teach controlled contrasts: before/after, more/less, same/different, who owns what, negation, and corrections to earlier facts.
- Vary surface wording and sentence lengths while preserving the underlying answer. Hold out entire wording patterns and combinations during evaluation.
- Include uncertainty and clarification, with examples where required facts are absent. Teach a short admission of missing information instead of a guess.
- Add distractor facts and delayed questions gradually. Teach two-fact dependencies before trying longer chains.
- Compare full-sequence next-token training, assistant-only loss, and a mixed objective that gives user tokens a smaller weight. User text also contains grammar worth learning; there is no established best weighting here.
- Start with short grammatical exchanges, then introduce multi-turn state updates and harder combinations. Compare this curriculum against a shuffled mixture at equal token budgets.

Synthetic data from a stronger teacher is compatible with training the student's weights from random initialization. It is different from adapting a pretrained student. Distillation using soft token probabilities is more involved: teacher and student vocabularies must be aligned. Do not directly compare logits from a teacher's subword tokenizer with this 812-token word-level output. A larger teacher trained with the same tokenizer is an optional clean experiment.

## Precision and browser efficiency

For 100,000 scalar weights, ideal raw payload sizes are 400 KB at FP32, 200 KB at FP16, 100 KB at INT8, and 25 KB at packed 2-bit precision. These are decimal sizes excluding tokenizer, tensor metadata, scales, code, activations, and KV cache. Ternary weights have an information-theoretic fixed-length block-coding limit approaching 1.585 bits per weight; straightforward storage uses two bits per weight. Leaving embeddings in floating point further increases the overall average bit width.

At this model size, weight download size is already small. Establish quality in floating point, then measure INT8, then consider lower precision or native ternary training. The BitNet results cited below are at billions of parameters, and cannot establish that a 100K model tolerates ternary weights. Quantization saves storage; it does not add trainable parameters or reliably increase understanding. Browser speedups require kernels that exploit the format; unpacking low-bit values can erase the benefit.

Benchmark CPU/WASM and WebGPU rather than assuming GPU execution wins for such small matrices. Run inference off the main thread, use incremental decoding with appropriate caches, and measure actual first-token latency and decode latency. Weight size, resident memory, and total application download are separate measurements.

## Second-round alternatives

- Factorized tied embeddings: use a smaller word embedding followed by a learned projection into the transformer width. For V=812, embedding width 24, and model width 48, this costs 20,640 parameters instead of 38,976. This imposes a representation/output rank bottleneck; use the savings for a matched-budget experiment and evaluate whether the bottleneck harms word distinctions. Tie both the word matrix and projection in the output path.
- One GRU baseline around the same parameter count: sequential computation can be a practical browser option, but its compressed hidden state may lose exact context. Compare with the same tokenizer, data, and evaluation.
- Grouped-query attention: can reduce projection parameters and KV storage, but at 128 tokens and 100K parameters the saving may not justify reduced attention capacity. It is not a first priority.
- Avoid mixture-of-experts as the initial architecture: routing and extra expert weights compete for an already tiny total budget. Low active parameters do not equal low total parameters.

## Application capabilities

Keep arithmetic, time, file conversion, and tool execution in ordinary browser code. Use explicit UI controls or a small, separately evaluated routing mechanism. Do not depend on the 100K generator reliably producing arbitrary tool arguments. A structured memory of selected facts can reduce the language model's memory burden, but it is an application feature and must be reported separately from the standalone model's performance.

For example, the application can retain `ball color = red` and `ball location = table`, update the location to `box`, and provide only relevant facts for the answer. The facts supplied to the model must still respect its vocabulary. Raw web pages, personal Markdown files, and unknown names are not directly representable in the current word-level tokenizer. Broader inputs need a deliberate tokenizer change, a separately evaluated conversion step, or a deterministic output path. Scraping arbitrary websites from a browser also depends on their cross-origin access rules.

## Experiment and reporting plan

Train A and B on the same data and token schedule, then C with two and four passes. Use at least three random seeds for the final comparison. Report parameter count, weight precision, training tokens, supervised targets, training compute, and browser latency. Match training tokens for a practical deployment comparison, and also compare training compute where recurrence or depth differs. Weight sharing is not a free-compute result.

Evaluate assistant-token loss plus exact-answer accuracy for location updates, ownership, ordering, negation, arithmetic, missing-information cases, and unfamiliar combinations. Use independently authored natural prompts and hold out whole template families. Inspect grammar, repetition, and answer relevance with a fixed rubric. Quantize the strongest floating-point candidate and measure its accuracy and browser behavior again. Report both standalone model performance and performance with deterministic browser tools.

## Primary sources

- [TinyStories](https://arxiv.org/abs/2305.07759): motivation for restricted language and carefully designed synthetic data. Its results do not establish general chat capability at 100K parameters.
- [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859): input/output embedding tying.
- [MobileLLM](https://arxiv.org/abs/2402.14905): depth, width, embedding sharing, and block sharing at substantially larger model sizes.
- [Universal Transformers](https://arxiv.org/abs/1807.03819): attention with recurrence over depth and shared transformations.
- [RoFormer](https://arxiv.org/abs/2104.09864): rotary positional encoding.
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531): soft-target knowledge distillation.
- [BitNet b1.58 2B4T](https://arxiv.org/abs/2504.12285): native ternary training at billion-parameter scale.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556): scaling evidence from much larger models; its token ratios are not validated prescriptions for this experiment.
