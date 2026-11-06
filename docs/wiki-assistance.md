# Wikipedia assistance

Wikipedia access is gated by the browser worker's `UNKNOWN_WORDS` error or a structured lookup request produced from the model's answer. The main thread does not scan user messages or source excerpts for tool instructions. Ordinary answers, generic uncertainty, malformed input, context errors and load errors do not initiate lookup. The existing lookup checkbox disables both triggers.

The current checkpoint can ask to look something up using a complete phrase such as `i need to look that up.` or `let me look it up.`. A complete model-output `[[WIKI: topic]]` request is also recognized for compatible future checkpoints. The deployed tokenizer cannot generate that marker's arbitrary text; no new tool-call training is claimed.

After an authorized search, the first Wikipedia extract is supplied to a second worker generation. This pass has a bounded prompt and uses the existing `<unk>` token ID for source words outside the fixed vocabulary. It does not add words, alter token IDs, change weights or migrate the separate training corpus. Unknown-token encoding loses word identity, so this small model cannot reliably synthesize arbitrary Wikipedia facts.

The reference pass is buffered. A generated answer is shown only when its word sequence occurs in the source extract (at least four words); otherwise the attributed source excerpt is shown. This conservative check establishes textual support, not that the excerpt answers every question. Source links remain visible. There is at most one lookup operation and one reference generation per submitted turn; a request generated during the reference pass cannot cause another lookup. The lookup service retains its existing one bounded query retry for an empty search result.

Reference-only turns are not put into ordinary model history, because the fixed-vocabulary tokenizer cannot safely represent those raw questions and excerpts. Follow-up questions requiring the same outside topic may need to name it again. New Chat cancels active generation and fetches, and stale responses are ignored.

Validation:

- `node training/tests/wiki-assistance.mjs`: gate behavior, disabled lookup, one-lookup budget, source-supported output, fallback, cancellation, context limits and real-checkpoint reference inference.
- `node training/tests/browser-inference.mjs`: unchanged inference parity, vocabulary rejection, context trimming and cancellation.
- `node training/tests/web-lookup.mjs`: query extraction, source ranking, service failures, bounded retry and cancellation.
- `node training/tests/normalize-input.mjs`: unchanged input normalization behavior.
