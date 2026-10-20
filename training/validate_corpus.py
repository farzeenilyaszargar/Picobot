#!/usr/bin/env python3
"""Audit the shipped JSONL independently and optionally restore packed files."""
import argparse
from array import array
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys


def audit(root, prepare=False):
    tokenizer = json.loads((root / 'tokenizer.json').read_text())
    report = json.loads((root / 'report.json').read_text())
    vocab = (root / 'vocabulary_800.txt').read_text().splitlines()
    assert len(vocab) == len(set(vocab)) == 800
    assert all(re.fullmatch('[a-z]+', word) for word in vocab)
    ids = tokenizer['token_to_id']
    assert len(ids) == 812 and set(ids.values()) == set(range(812))
    assert tokenizer['tokens'] == list(ids)
    assert set(vocab) == set(ids) - set(tokenizer['special_tokens']) - set(tokenizer['punctuation'])
    seen, group_splits, all_words = set(), {}, Counter()
    result = {}
    number = {w: n for n, w in enumerate('zero one two three four five six seven eight nine ten'.split())}
    for split, expected in report['splits'].items():
        packed, masks, offsets = array('H'), bytearray(), []
        word_counts, families = Counter(), Counter()
        count = 0
        for shard in expected['jsonl_shards']:
            with gzip.open(root / shard, 'rt') as source:
                for line in source:
                    row = json.loads(line)
                    messages = row['messages']
                    assert len(messages) >= 2 and len(messages) % 2 == 0
                    canonical = json.dumps(messages, separators=(',', ':')).encode()
                    digest = hashlib.sha256(canonical).hexdigest()
                    assert row['id'] == digest and digest not in seen, 'Duplicate dialogue'
                    seen.add(digest)
                    group = row['group_id']
                    assert group_splits.setdefault(group, split) == split, 'Group crosses splits'
                    if row['pattern'] == 'coverage':
                        assert row['family'] == 'vocabulary_copy' and split == 'train'
                    else:
                        bucket = int(group[:8], 16) % 10000
                        assigned = 'validation' if bucket < 100 else 'test' if bucket < 200 else 'train'
                        assert assigned == split
                    encoded, mask = [ids['<bos>']], [0]
                    for i, message in enumerate(messages):
                        role, content = message['role'], message['content']
                        assert role == ('assistant' if i % 2 else 'user')
                        assert content and content == content.lower()
                        tokens = re.findall(r'[a-z]+|[.,?!:]', content)
                        assert ''.join(tokens) == ''.join(content.split())
                        assert all(t in ids for t in tokens), 'Out-of-vocabulary content'
                        assert 'a orange' not in content
                        if role == 'assistant' and row['family'] in {'plans_and_order', 'preferences', 'feelings_and_small_talk'}:
                            assert not re.search(r'\bmy\b', content), 'Wrong speaker perspective'
                        words = [t for t in tokens if t.isalpha()]
                        word_counts.update(words)
                        encoded.extend([ids[f'<{role}>'], *[ids[t] for t in tokens], ids['<turn>']])
                        mask.extend([0, *[int(role == 'assistant')] * len(tokens), int(role == 'assistant')])
                    encoded.append(ids['<eos>'])
                    mask.append(1)
                    assert len(encoded) == row['token_count']
                    if row['family'] == 'small_counts':
                        a, b = re.search(r'i have (\w+) \w+\. i get (\w+) more', messages[0]['content']).groups()
                        total = number[messages[1]['content'].split()[2]]
                        given = number[messages[2]['content'].split()[2]]
                        left = number[messages[3]['content'].split()[2]]
                        assert number[a] + number[b] == total and total - given == left
                    offsets.append(dict(id=digest, offset=len(packed), length=len(encoded)))
                    packed.extend(encoded)
                    masks.extend(mask)
                    families[row['family']] += 1
                    count += 1
        assert count == expected['dialogues']
        assert len(packed) == expected['tokens'] == len(masks)
        assert dict(word_counts) == expected['word_counts']
        assert families == expected['family_examples']
        all_words.update(word_counts)
        if sys.byteorder != 'little':
            packed.byteswap()
        binary = packed.tobytes()
        offset_text = ''.join(json.dumps(x) + '\n' for x in offsets).encode()
        for filename, payload in [(f'{split}.bin', binary), (f'{split}.assistant_mask.bin', bytes(masks)),
                                  (f'{split}.offsets.jsonl', offset_text)]:
            path = root / filename
            if prepare:
                path.write_bytes(payload)
            assert path.read_bytes() == payload, f'Packed file mismatch: {filename}'
        result[split] = dict(tokens=len(packed), dialogues=count, supervised_tokens=sum(masks))
    assert set(all_words) == set(vocab)
    assert sum(x['tokens'] for x in result.values()) == report['total_tokens']
    assert sum(x['dialogues'] for x in result.values()) == report['total_dialogues']
    for filename, expected_hash in json.loads((root / 'sha256.json').read_text()).items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == expected_hash, filename
    summary = dict(status='passed', english_word_types=len(all_words), english_word_occurrences=sum(all_words.values()),
                   total_tokens=report['total_tokens'], total_dialogues=len(seen),
                   duplicate_dialogues=0, cross_split_groups=0, out_of_vocabulary_words=0,
                   checks=['JSONL hashes and exact deduplication', 'semantic group split assignment',
                           'closed vocabulary and formatting', 'roles and loss masks',
                           'packed tokens and offsets', 'small-count arithmetic',
                           'selected speaker and article invariants', 'artifact checksums'], splits=result)
    (root / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parent / 'data/picoboi-800-v1')
    parser.add_argument('--prepare', action='store_true', help='Restore packed files from JSONL before auditing')
    args = parser.parse_args()
    audit(args.root, args.prepare)
