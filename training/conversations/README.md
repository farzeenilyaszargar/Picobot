# Directly authored conversational corpus

Target: **10,000,000 English word occurrences**, using the original 800 words plus six explicitly added project terms. Punctuation, role names, metadata, and structural tokens do not count as English words toward this target. See `audit.json` for the current verified count; the target is not a claim of completion.

All dialogue content here is written directly by the assistant in this conversation. No template expansion, slot substitution, repetition for volume, imported generated corpus, or programmatic dialogue generator is permitted. Scripts may validate, count, deduplicate, and serialize content; they must not invent dialogue content. The previous 10M-token template corpus is not mixed into this corpus or counted toward its target.

Each JSONL record has an ID, a scenario, topic tags, provenance, and alternating user/assistant messages. Only message content is constrained by the vocabulary. Different scenarios should require different responses, decisions, or contextual dependencies, rather than merely replacing a color or noun. Ordinary short replies, clarification questions, and admissions of missing information are part of useful conversation. Do not pad answers to raise word counts.

The author checks each exchange for relevance to the latest user request, consistency with earlier turns, correct speaker perspective, and simple factual/logical correctness. The validator checks vocabulary, structural integrity, duplicates, lengths, and provenance labels. Passing that validator does not prove relevance or general conversational quality.

Coverage priorities include greetings, requests for help, user preferences, choosing between options, following constraints, remembering and correcting facts, resolving references, asking for missing details, acknowledging uncertainty, explaining simple relations, counting, everyday planning, writing assistance, and emotional support in ordinary situations. Show a short reason when it helps the user. Do not produce hidden reasoning traces or long explanations when a short answer is appropriate. Do not pretend to have a body, access to the user's room, or live knowledge of facts absent from the conversation.

The original `../vocabulary_800.txt`, all older training data, the deployed tokenizer, and the trained 100K model are preserved. The active `vocabulary.txt` contains the original 800 words plus `picoboi`, `llm`, `webpage`, `pico`, `model`, and `browser`, for 806 word forms. The active `tokenizer.json` preserves all 812 baseline token IDs and appends those six words, for 818 token IDs. The previous 807-word vocabulary and 820-token specification are retained under `archive/` for provenance; they are not the active training contract.

Creator attribution is handled in the webpage by a built-in reply with a clickable link. It is not a model-generated answer. `farilyaz` and `@` are therefore absent from the active tokenizer. `excluded-conversations.json` identifies five earlier creator-handle conversations that remain unchanged in their original source file but must not be used for training or counted toward the target. The auditor applies this manifest before validation and counting. Any future data packer must apply the same exclusions; blindly concatenating all source JSONL files is incorrect.

Other identity exchanges teach the model its name, browser location, and limits. They do not change the already trained model. New training must use this corpus and its own tokenizer together; preserving baseline token IDs does not make the old 812-row embedding matrix compatible with the new model. No old packed data is automatically mixed into this corpus.

The intended next model is about 100K parameters and has not been trained on this corpus. The earlier `../configs/picoboi-1m-conversation.json` proposal is retained for reference, not the current model target. The 24 separately authored held-out relevance cases in `evaluation/` are excluded from training word counts. The corpus includes conversations up to 428 model tokens; the future training context and batching must account for their full lengths. The existing website/model remain the baseline while this data is built.

Run from the repository root:

```sh
python3 training/audit_conversations.py
```

The resulting audit records the exact completed word count and remaining words. Work proceeds in directly authored batches. Creating a first batch does not satisfy the full 10M-word objective.
