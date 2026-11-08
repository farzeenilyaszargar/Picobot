"""Encode named-city references within a dialogue, using input-visible bindings."""
import re


CITY_SLOTS = [f'<city_{i}>' for i in range(1, 9)]


class CityReferences:
    def __init__(self, aliases, atomic=False):
        if not isinstance(aliases, dict) or any(
                not isinstance(name, str) or not name or '<' in name or '>' in name
                or not isinstance(identity, str) or not identity for name, identity in aliases.items()):
            raise ValueError('Cities must map nonempty case-sensitive aliases to identities')
        self.aliases = aliases
        self.atomic = atomic
        names = sorted(aliases, key=lambda name: (-len(name), name))
        names_pattern = r'(?<!\w)(?:' + '|'.join(map(re.escape, names)) + r')(?!\w)'
        protected = r'```[\s\S]*?(?:```|$)|`[^`\n]*`|"[^"\n]*"|“[^”\n]*”|'
        self.pattern = re.compile((protected if atomic else '') + names_pattern) if names else None

    def normalize(self, messages):
        bindings = {}
        if self.pattern is None:
            return messages

        def replace(match, introduce):
            name = match.group()
            if name not in self.aliases:
                return name
            # Bare Paris can also be a person's name. The explicit alias still works.
            if self.atomic and name == 'Paris' and not re.search(
                    r'\b(?:in|from|to|via|visit|leave|reach)\s+$', match.string[:match.start()], re.I):
                return name
            identity = self.aliases[name]
            if identity not in bindings and introduce and len(bindings) < (len(CITY_SLOTS) if self.atomic else 32):
                bindings[identity] = len(bindings) + 1
            if identity not in bindings:
                return name
            index = bindings[identity]
            return CITY_SLOTS[index - 1] if self.atomic else f'<city> <ref_{index}>'

        result = []
        for message in messages:
            # A target may reference an entity from prior input. It cannot allocate one.
            introduce = message['role'] in {'user', 'tool'}
            content = self.pattern.sub(lambda match: replace(match, introduce), message['content'])
            result.append({**message, 'content': content})
        return result
