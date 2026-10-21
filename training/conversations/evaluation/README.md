# Held-out conversational relevance cases

These cases are directly authored evaluation conversations, separate from the training data. Each ends with a user turn and supplies a reference answer and a grading rubric. References show an acceptable answer; wording need not match exactly unless the rubric asks for a particular short format.

`relevance.jsonl` contains the first 12 cases. `relevance-002.jsonl` adds 12 cases covering hypothetical versus actual state, recipient preferences, missing input, writing versus acting, counting after a return, directions, listening without giving a plan, speaker identity, brevity after a distractor, conditional reasoning, preference versus possession, and unknown personal details.

Evaluate the final answer using its rubric and the full preceding conversation. Record the actual model answer and the judgment for each case. A fluent but irrelevant answer fails. Do not count these prompts or references toward the training word target, copy them into training, or tune the model repeatedly against these cases and still describe them as an untouched test set.

The corpus audit checks vocabulary and role order, and rejects a held-out final prompt that appears verbatim as a training user message. That check does not detect every semantic overlap or prove generalization. These 24 cases are an initial evaluation set, not a comprehensive benchmark. No score for the proposed 1M model exists because it has not been trained.
