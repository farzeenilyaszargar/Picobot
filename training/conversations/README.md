# Directly authored conversational corpus

Target: **10,000,000 English word occurrences**, now using exactly **997 permitted word forms**. Punctuation, role names, reference tokens, metadata, and structural tokens do not count toward the English-word target. See `audit.json` for the actual verified count; the target is not a claim of completion.

All dialogue content is written directly by the assistant in this conversation. No template expansion, slot substitution to inflate volume, repetition for volume, imported generated corpus, or programmatic dialogue generator is permitted. Scripts may validate, count, deduplicate, and serialize content. The old template baseline is preserved separately and is not counted here. The parser's deterministic reference rendering is an application feature, not a corpus generator.

Each JSONL record has an ID, a scenario, topic tags, provenance, and alternating user/assistant messages. Different scenarios should require different responses or contextual dependencies, not merely replace a color or noun. Short replies and clarification remain useful. Longer stories, explanations, and multi-turn planning are appropriate when requested; do not pad answers to increase counts.

The author checks relevance, grammar, speaker perspective, consistency, and basic factual/logical correctness. The auditor checks vocabulary, format, exact duplicates, reference declarations, and lengths. Passing it does not prove semantic quality or future model capability.

## Active contract

`contract.json` specifies 997 lowercase word forms and a maximum of 1,024 tokens per complete conversation. This raises the previous 512-token data allowance and is a requirement for future training, not an upgrade to the live model's 128-token context. Four new longer conversations exceed 512 tokens; the audit records actual lengths.

The active `vocabulary.txt` starts from the previous 1,000-word list, retires `apple`, `apples`, `banana`, and `lemon`, and adds `fruits`, for a net reduction of three forms. The parser maps those names to `fruit` or `fruits` while retaining the surface name in a reference binding. Plural `bananas` and `lemons` are also recognized by the parser without entering the model vocabulary. `orange` remains because it also names a color. No other categories are retired in this change. `vocabulary-additions-1000.txt` records the earlier expansion and remains historical. Inflections count separately. `audit.json` reports which words have actually appeared; adding a word to the list alone does not teach its meaning.

`tokenizer.json` now has 997 word IDs, 12 existing structure/punctuation IDs, and eight reference tokens `<term_0>` through `<term_7>`. Total: **1,017 IDs**. This v4 tokenizer deliberately renumbers word IDs after the removed words. It is incompatible with older weights; use it only for the untrained next model. English word counting excludes the reference tokens. An assistant reference must previously have appeared in a user turn, preceded by the same category word. The category cannot change for that slot during the conversation.

[The concept parser prototype](PARSER.md) retains original surface words in a side table and uses explicit references to restore them. It does not replace generic category nouns or recover facts lost by generalization. The next model must be trained on this protocol; the deployed model cannot emit these new token IDs.

## Model view of preserved source

Earlier source messages still contain fruit names. `canonical.py` derives their canonical model text before validation, counting, or training. User fruit names receive explicit bindings; assistant mentions reuse those bindings. An assistant introducing a fruit not named by the user uses a generic category instead of inventing an unbound reference. Generated categories are not blanket statements about every fruit; the current migrated cases were inspected for this issue.

A future packer must use `canonicalize(row, tokenizer)` and apply the exclusion manifest. Training directly on raw authored files would reintroduce retired words. The audit counts only the canonical view once, not both versions, and records raw source checksums.

## Preservation and evaluation

All earlier authored batches, the original `../vocabulary_800.txt`, older training data, the deployed tokenizer, and trained 100K weights are preserved. `archive/vocabulary-1000.txt` and `archive/tokenizer-1020.json` snapshot v3. `archive/vocabulary-806.txt` and `archive/tokenizer-818.json` retain the earlier contract. Earlier 807-word and 820-token drafts are also retained. Do not mix these tokenizer versions when packing data or loading weights.

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
