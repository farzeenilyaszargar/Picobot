# Concept parser prototype

This is an inspectable browser-compatible JavaScript adapter for the next model. It is not connected to the deployed 800-word model and does not contain another language model.

The model vocabulary has 1,000 lowercase word forms, 12 existing punctuation/structure tokens, and eight new reference tokens: `<term_0>` through `<term_7>`. The total is 1,020 token IDs. All previous 818 IDs retain their meanings.

## Protocol

Keep one parser instance and its bindings for one chat:

```js
import { createConceptParser } from './concept-parser.mjs';
const parser = createConceptParser(tokenizer);
const parsed = parser.parse('my cat is under the table. my dog is by the door.');
if (!parsed.ready) {
  // Ask for clarification using parsed.unsupported and referenceCapacityExceeded.
} else {
  // Give parsed.modelText to a model trained with this protocol.
}
```

The model input is:

```text
my animal <term_0> is under the table. my animal <term_1> is by the door.
```

The side table is ordinary application data, not model vocabulary:

```json
[
  {"token":"<term_0>","surface":"cat","concept":"animal"},
  {"token":"<term_1>","surface":"dog","concept":"animal"}
]
```

For example, an authored protocol response `the animal <term_1> is by the door.` renders as `the dog is by the door.` This illustrates rendering, not a response already produced by trained new weights.

A generic answer such as `an animal needs food.` stays generic. The renderer only replaces a category followed by its explicit bound reference. It rejects unbound slots and category/slot mismatches. It also adjusts a/an when replacing a reference, so `an animal <term_0>` becomes `a cat`. Plural words receive distinct bindings, preserving `cats` as `animals <term_n>`.

Slots identify surface words, not physical entities. Two mentions of `cat` share one word binding; this does not assert that they describe the same cat. The model still needs contextual evidence such as `my`, `your`, positions, and counts. There is no implemented general entity resolver.

## Deliberate limits

- The current dictionary covers a small set of animal words and selected tool, vehicle, flower, and tree words. Recognized model words are preserved rather than unnecessarily generalized.
- Unknown words outside that dictionary remain unsupported. They are not silently changed into a similar spelling or into `object`.
- Cat and dog share a category but do not share all facts. The binding table preserves the word; it does not teach the model species-specific sounds, food, behavior, or other knowledge. Such tasks need grounded facts supplied in supported language, a separate tool, or an admission of uncertainty.
- There are eight distinct surface-word bindings per chat. When full, the parser reports failure without reusing a slot. Failed parses do not alter committed state.
- User-supplied reference tokens are rejected. Do not let users overwrite bindings.
- `reset()` must coincide with discarding the associated model history. Never reinterpret an existing history using a new table.
- General paraphrasing and robust coreference resolution are not implemented. The existing contraction, greeting, and spelling adapter is reused before concept mapping.

## Training and evaluation

Training uses canonical category/reference sequences, not the original out-of-vocabulary nouns. Four directly authored conversations in `authored/batch-000047.jsonl` include raw first-user text and surface-binding metadata for inspection. Metadata does not count toward the 10M-word target. These are training examples, not held-out performance evidence.

The next model must learn to preserve references, generic wording, negation, number, and relations over longer contexts. It has not been trained yet. Measure the standalone model and parser-assisted system separately. A correct renderer cannot repair a model selecting the wrong valid slot.

Run:

```sh
node training/tests/concept-parser.mjs
python3 -m unittest discover -s training/tests -p test_conversation_contract.py -v
python3 training/audit_conversations.py
```

Tests cover distinct words, reuse across turns, plurals, generic mentions, unknown words, wrong or missing references, reserved-token input, slot exhaustion, reset, article agreement, corpus/parser compatibility, vocabulary size, preserved IDs, and the 1,024-token corpus limit. They test the adapter and data contract, not model understanding.
