# Directly authored conversational corpus

Target: **100,000,000 English word occurrences**, within the existing frozen 800-word vocabulary. Punctuation, role names, metadata, and structural tokens do not count as English words toward this target. See `audit.json` for the current verified count; the target is not a claim of completion.

All dialogue content here is written directly by the assistant in this conversation. No template expansion, slot substitution, repetition for volume, imported generated corpus, or programmatic dialogue generator is permitted. Scripts may validate, count, deduplicate, and serialize content; they must not invent dialogue content. The previous 10M-token template corpus is not mixed into this corpus or counted toward its target.

Each JSONL record has an ID, a scenario, topic tags, provenance, and alternating user/assistant messages. Only message content is constrained by the vocabulary. Different scenarios should require different responses, decisions, or contextual dependencies, rather than merely replacing a color or noun. Ordinary short replies, clarification questions, and admissions of missing information are part of useful conversation. Do not pad answers to raise word counts.

The author checks each exchange for relevance to the latest user request, consistency with earlier turns, correct speaker perspective, and simple factual/logical correctness. The validator checks vocabulary, structural integrity, duplicates, lengths, and provenance labels. Passing that validator does not prove relevance or general conversational quality.

Coverage priorities include greetings, requests for help, user preferences, choosing between options, following constraints, remembering and correcting facts, resolving references, asking for missing details, acknowledging uncertainty, explaining simple relations, counting, everyday planning, writing assistance, and emotional support in ordinary situations. Show a short reason when it helps the user. Do not produce hidden reasoning traces or long explanations when a short answer is appropriate. Do not pretend to have a body, access to the user's room, or live knowledge of facts absent from the conversation.

The 800-word list is unchanged. This means the brand name and many general-purpose terms cannot be represented. No new word is silently added. A larger tokenizer would be a separate, explicit dataset/model revision.

The proposed next model is specified in `../configs/picoboi-1m-conversation.json`: 957,056 parameters, four distinct width-128 blocks, context 512. It is untrained. Context 512 is an initial design target, not proof of reliable recall. We must prepare separately authored held-out conversations, audit the full corpus, and benchmark local training before starting the new training run. The existing website/model remain the current baseline while this data is built.

Run from the repository root:

```sh
python3 training/audit_conversations.py
```

The resulting audit records the exact completed word count and remaining words. Work proceeds in directly authored batches. Creating a first batch does not satisfy the full 100M-word objective.
