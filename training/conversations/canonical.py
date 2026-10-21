"""Derive the model view of preserved authored text; never create new dialogue."""
import copy
import re


def canonicalize(row, tokenizer):
    result = copy.deepcopy(row)
    mappings = tokenizer.get('generalized_words', {})
    slots = tokenizer['reference_tokens']
    occupied = {slot for m in row['messages'] for slot in slots if slot in m['content']}
    bindings = {}
    for message in result['messages']:
        def replace(match):
            surface = match.group()
            if surface not in mappings:
                return surface
            if surface not in bindings:
                # The model can invent a generic fruit in a story, not an unbound surface word.
                if message['role'] == 'assistant':
                    return mappings[surface]
                available = next((slot for slot in slots if slot not in occupied), None)
                if available is None:
                    raise ValueError(f"{row['id']}: no reference slots left for {surface}")
                occupied.add(available)
                bindings[surface] = dict(token=available, surface=surface, concept=mappings[surface])
            binding = bindings[surface]
            return f"{binding['concept']} {binding['token']}"
        message['content'] = re.sub(r"[a-z]+(?:'[a-z]+)?", replace, message['content'])
        message['content'] = re.sub(
            r'\b(a|an)\s+([a-z]+\s+<term_\d+>|fruits?\b)',
            lambda m: ('an' if m[2][0] in 'aeiou' else 'a')+' '+m[2], message['content'])
    if bindings:
        result['generalized_bindings'] = list(bindings.values())
    return result
