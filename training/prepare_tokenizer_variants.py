#!/usr/bin/env python3
"""Fit and prepare both tokenizers over identical canonical conversations. No dialogue generation."""
from collections import Counter
import array
import hashlib
import json
from pathlib import Path
import re
import sys

from conversations.canonical import canonicalize
from tokenization import Tokenizer, train_bpe

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'conversations'
OUTPUT = DATA/'tokenizers'
PACKED = ROOT/'data/conversation-variants'


def source_rows(spec):
    excluded = json.loads((DATA/'excluded-conversations.json').read_text())
    for path in sorted((DATA/'authored').glob('*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['id'] not in excluded:
                yield canonicalize(row, spec)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    source = json.loads((DATA/'tokenizer.json').read_text())
    audit = json.loads((DATA/'audit.json').read_text())
    frequency = Counter()
    source_files = {}
    for path in sorted((DATA/'authored').glob('*.jsonl')):
        sha = digest(path.read_bytes())
        if audit['files'][path.name]['sha256'] != sha:
            raise ValueError('Run the corpus audit before preparing tokenizer variants')
        source_files[path.name] = sha
    for row in source_rows(source):
        for message in row['messages']:
            text = re.sub(r'<term_\d+>', '', message['content'])
            frequency.update(re.findall('[a-z]+', text))
    if sum(frequency.values()) != audit['english_words']:
        raise ValueError('Canonical source count differs from the audit')
    word = {**source, 'type': 'word_level', 'format_version': 1,
            'word_vocabulary': (DATA/'vocabulary.txt').read_text().splitlines()}
    specs = {'word-level': word, 'bpe-512': train_bpe(frequency, source)}
    summaries = {}
    for name, spec in specs.items():
        codec = Tokenizer(spec)
        stream, masks, offsets = array.array('H'), bytearray(), []
        max_length = 0
        assistant_targets = 0
        for row in source_rows(source):
            ids, mask = codec.encode_conversation(row['messages'])
            if len(ids) > 1024:
                raise ValueError(f'{name}: {row["id"]} needs {len(ids)} tokens; no truncation applied')
            # Exact word/marker recovery apart from canonical whitespace formatting.
            for message in row['messages']:
                expected = re.sub(r'\s+([.,?!:])', r'\1', ' '.join(message['content'].split()))
                if codec.decode(codec.encode(message['content'])) != expected:
                    raise ValueError(f'{name}: round trip failed for {row["id"]}')
            start = len(stream)
            stream.extend(ids)
            masks.extend(mask)
            offsets.append({'id': row['id'], 'start': start, 'end': len(stream)})
            max_length = max(max_length, len(ids))
            assistant_targets += sum(mask)
        if sys.byteorder != 'little':
            stream.byteswap()
        directory = OUTPUT/name
        packed = PACKED/name
        directory.mkdir(parents=True, exist_ok=True)
        packed.mkdir(parents=True, exist_ok=True)
        spec_bytes = (json.dumps(spec, indent=2)+'\n').encode()
        (directory/'tokenizer.json').write_bytes(spec_bytes)
        (directory/'vocabulary.txt').write_text('\n'.join(spec['tokens'])+'\n')
        if name == 'bpe-512':
            (directory/'merges.txt').write_text('# English character BPE; ▁ marks word end\n'+
                '\n'.join(' '.join(pair) for pair in spec['merges'])+'\n')
        token_bytes = stream.tobytes()
        (packed/'train.bin').write_bytes(token_bytes)
        (packed/'train.assistant_mask.bin').write_bytes(masks)
        (packed/'train.offsets.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in offsets))
        summary = dict(
            name=name, status='prepared_data_model_untrained', tokenizer_size=len(spec['tokens']),
            tokenizer_sha256=digest(spec_bytes), conversations=len(offsets),
            source_english_words=sum(frequency.values()), model_tokens=len(stream),
            assistant_targets=assistant_targets, longest_conversation_tokens=max_length,
            max_context=1024, split='training_only_no_validation_split_created',
            tokens_sha256=digest(token_bytes), assistant_mask_sha256=digest(masks),
            token_dtype='little-endian uint16', mask_dtype='uint8',
            local_data=str(packed.relative_to(ROOT.parent)),
            source_tokenizer_sha256=digest((DATA/'tokenizer.json').read_bytes()),
            source_files=source_files,
        )
        (directory/'manifest.json').write_text(json.dumps(summary, indent=2)+'\n')
        summaries[name] = {key: summary[key] for key in (
            'tokenizer_size', 'conversations', 'source_english_words', 'model_tokens',
            'longest_conversation_tokens', 'assistant_targets')}
    if summaries['word-level']['model_tokens'] != audit['model_tokens']:
        raise ValueError('Word-level packed data differs from the corpus audit')
    word_codec, bpe_codec = Tokenizer(word), Tokenizer(specs['bpe-512'])
    samples = ['fact', 'facts', 'know', 'knows', 'knowing', 'known', 'knew',
               'i know the facts.', 'the animal <term_0> is not hungry.']
    examples = []
    for text in samples:
        examples.append(dict(text=text, word_level_tokens=[word_codec.tokens[i] for i in word_codec.encode(text)],
                             bpe_tokens=[bpe_codec.tokens[i] for i in bpe_codec.encode(text)]))
    comparison = dict(status='both_prepared_model_untrained', default='word-level', variants=summaries,
                      bpe_sequence_ratio=summaries['bpe-512']['model_tokens']/summaries['word-level']['model_tokens'],
                      examples=examples,
                      notes=['Both variants use exactly the same canonical source conversations.',
                             'BPE was fitted on this same corpus. This is not held-out quality evidence.',
                             'The concept parser lexicon is separate from BPE model token IDs.',
                             'Packed files are ignored by Git and recreated by this script.',
                             'Word and reference IDs differ between variants. Weights are not interchangeable.'])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    print(json.dumps({**{k:v for k,v in comparison.items() if k not in ['examples','notes']}},indent=2))


if __name__ == '__main__':
    main()
