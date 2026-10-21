# Directly authored conversational corpus

Target: **10,000,000 English word occurrences**, now using exactly **1,000 permitted word forms**. Punctuation, role names, reference tokens, metadata, and structural tokens do not count toward the English-word target. See `audit.json` for the actual verified count; the target is not a claim of completion.

All dialogue content is written directly by the assistant in this conversation. No template expansion, slot substitution to inflate volume, repetition for volume, imported generated corpus, or programmatic dialogue generator is permitted. Scripts may validate, count, deduplicate, and serialize content. The old template baseline is preserved separately and is not counted here. The parser's deterministic reference rendering is an application feature, not a corpus generator.

Each JSONL record has an ID, a scenario, topic tags, provenance, and alternating user/assistant messages. Different scenarios should require different responses or contextual dependencies, not merely replace a color or noun. Short replies and clarification remain useful. Longer stories, explanations, and multi-turn planning are appropriate when requested; do not pad answers to increase counts.

The author checks relevance, grammar, speaker perspective, consistency, and basic factual/logical correctness. The auditor checks vocabulary, format, exact duplicates, reference declarations, and lengths. Passing it does not prove semantic quality or future model capability.

## Active contract

`contract.json` specifies 1,000 lowercase word forms and a maximum of 1,024 tokens per complete conversation. This raises the previous 512-token data allowance and is a requirement for future training, not an upgrade to the live model's 128-token context. Four new longer conversations exceed 512 tokens; the audit records actual lengths.

The active `vocabulary.txt` preserves the previous 806 forms and adds 194 forms listed in `vocabulary-additions-1000.txt`. Additions include broad concepts such as animal, plant, object, vehicle, tool, and container, plus common grammar and conversational language. Inflections count separately. `audit.json` reports which words have actually appeared; adding a word to the list alone does not teach its meaning.

`tokenizer.json` preserves the previous 818 IDs, appends 194 new word IDs, then appends eight reference tokens `<term_0>` through `<term_7>`. Total: **1,020 IDs**. English word counting excludes the reference tokens. An assistant reference must previously have appeared in a user turn, preceded by the same category word. The category cannot change for that slot during the conversation.

[The concept parser prototype](PARSER.md) retains original surface words in a side table and uses explicit references to restore them. It does not replace generic category nouns or recover facts lost by generalization. The next model must be trained on this protocol; the deployed model cannot emit these new token IDs.

## Preservation and evaluation

All earlier authored batches, the original `../vocabulary_800.txt`, older training data, the deployed tokenizer, and trained 100K weights are preserved. `archive/vocabulary-806.txt` and `archive/tokenizer-818.json` snapshot the previous active contract. Earlier 807-word and 820-token drafts are also retained. Do not mix these tokenizer versions when packing data or loading weights.

Creator attribution is a built-in webpage reply with a clickable link. Five earlier creator-handle conversations remain unchanged but are excluded using `excluded-conversations.json`. Every data packer must apply those exclusions; blindly concatenating source JSONL is incorrect. `farilyaz` and `@` are not active model tokens.

The intended next model remains about 100K parameters and is untrained on this corpus. A larger vocabulary consumes more embedding/output capacity; the final architecture must respect the total parameter budget. The earlier 1M proposal in `../configs/` is historical and does not specify this next model.

The 24 separately authored held-out relevance cases in `evaluation/` are excluded from training word counts. Parser fixtures and training examples test mechanics, not held-out new-model accuracy. Future evaluation must cover longer context and reference accuracy independently of the training examples.

Run from the repository root:

```sh
python3 training/audit_conversations.py
node training/tests/concept-parser.mjs
python3 -m unittest discover -s training/tests -p test_conversation_contract.py -v
```

The full 10M-word objective is incomplete unless the verified count reaches that target. No new model training is implied by a passing corpus audit.
