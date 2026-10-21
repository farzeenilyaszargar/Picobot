"""Word-level and English-only character BPE codecs for the next model."""
from collections import Counter
import json
from pathlib import Path
import re
import string

END = '▁'


def merge_pair(parts, pair):
    result, i = [], 0
    while i < len(parts):
        if i + 1 < len(parts) and (parts[i], parts[i + 1]) == pair:
            result.append(''.join(pair))
            i += 2
        else:
            result.append(parts[i])
            i += 1
    return tuple(result)


def train_bpe(frequency, word_spec, size=512):
    reserved = [t for t in word_spec['tokens'] if not re.fullmatch('[a-z]+', t)]
    tokens = reserved + list(string.ascii_lowercase) + [END]
    if size < len(tokens):
        raise ValueError('BPE size is smaller than its reserved tokens and alphabet')
    vocabulary = set(tokens)
    lexicon = {word: tuple(word)+(END,) for word in sorted(frequency)}
    merges = []
    while len(tokens) < size:
        pairs = Counter()
        for word, parts in lexicon.items():
            for pair in zip(parts, parts[1:]):
                pairs[pair] += frequency[word]
        if not pairs:
            raise ValueError('Not enough distinct merges to reach requested BPE size')
        best = max(pairs, key=lambda p: (pairs[p], p))
        combined = ''.join(best)
        if combined not in vocabulary:
            tokens.append(combined)
            vocabulary.add(combined)
        merges.append(list(best))
        lexicon = {word: merge_pair(parts, best) for word, parts in lexicon.items()}
    return dict(
        format_version=1, type='bpe', name='picoboi-bpe-512-v1', tokens=tokens,
        merges=merges, end_of_word=END, lowercase=True,
        word_vocabulary=[t for t in word_spec['tokens'] if re.fullmatch('[a-z]+', t)],
        reference_tokens=word_spec['reference_tokens'],
        generalized_words=word_spec['generalized_words'],
        reserved_token_names=reserved, scope='lowercase ASCII English with five punctuation marks',
        status='tokenizer_fitted_model_untrained',
    )


class Tokenizer:
    def __init__(self, spec):
        self.spec = spec
        self.tokens = spec['tokens']
        self.ids = {token: i for i, token in enumerate(self.tokens)}
        if len(self.ids) != len(self.tokens):
            raise ValueError('Duplicate token strings')
        self.bpe = spec.get('type') == 'bpe'
        self.references = set(spec['reference_tokens'])
        self.role_tokens = {'<pad>', '<bos>', '<eos>', '<user>', '<assistant>', '<turn>'}
        refs = '|'.join(re.escape(t) for t in sorted(self.references, key=len, reverse=True))
        self.pattern = re.compile(rf'{refs}|[a-z]+|[.,?!:]')
        self.cache = {}

    def pieces(self, word):
        if not re.fullmatch('[a-z]+', word):
            raise ValueError('Expected a lowercase ASCII word')
        if not self.bpe:
            if word not in self.ids:
                raise ValueError(f'Word outside word-level vocabulary: {word}')
            return [word]
        if word not in self.cache:
            parts = tuple(word)+(self.spec['end_of_word'],)
            for pair in self.spec['merges']:
                parts = merge_pair(parts, tuple(pair))
            if any(part not in self.ids for part in parts):
                raise ValueError(f'Word uses characters outside the tokenizer alphabet: {word}')
            self.cache[word] = list(parts)
        return list(self.cache[word])

    def encode(self, text):
        parts = self.pattern.findall(text)
        if text != text.lower() or ''.join(parts) != ''.join(text.split()):
            raise ValueError('Expected lowercase canonical text, punctuation, and reference tokens only')
        result = []
        for part in parts:
            if re.fullmatch('[a-z]+', part):
                result.extend(self.ids[p] for p in self.pieces(part))
            else:
                result.append(self.ids[part])
        return result

    def decode(self, ids, skip_role_tokens=False):
        chunks, pending = [], ''
        reserved = set(self.spec.get('reserved_token_names', []))
        for index in ids:
            if type(index) is not int or not 0 <= index < len(self.tokens):
                raise ValueError(f'Invalid token ID: {index}')
            token = self.tokens[index]
            if self.bpe and token not in reserved:
                pending += token
                if pending.endswith(self.spec['end_of_word']):
                    chunks.append(pending[:-len(self.spec['end_of_word'])])
                    pending = ''
            else:
                if pending:
                    chunks.append(pending)
                    pending = ''
                if not (skip_role_tokens and token in self.role_tokens):
                    chunks.append(token)
        if pending:
            chunks.append(pending)
        return re.sub(r'\s+([.,?!:])', r'\1', ' '.join(chunks))

    def encode_conversation(self, messages):
        if not messages or len(messages) % 2:
            raise ValueError('A complete conversation must end with an assistant turn')
        ids, mask = [self.ids['<bos>']], [0]
        for i, message in enumerate(messages):
            expected = 'user' if i % 2 == 0 else 'assistant'
            if message['role'] != expected:
                raise ValueError('Conversation roles must alternate')
            content = self.encode(message['content'])
            ids.extend([self.ids[f'<{expected}>'], *content, self.ids['<turn>']])
            mask.extend([0, *([int(expected == 'assistant')] * len(content)), int(expected == 'assistant')])
        ids.append(self.ids['<eos>'])
        mask.append(1)
        return ids, mask


def load_variant(name='word-level'):
    if name not in {'word-level', 'bpe-512'}:
        raise ValueError('Choose word-level or bpe-512')
    path = Path(__file__).resolve().parent/'conversations/tokenizers'/name/'tokenizer.json'
    return Tokenizer(json.loads(path.read_text()))
