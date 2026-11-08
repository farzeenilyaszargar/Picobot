#!/usr/bin/env python3
"""Prepare a reproducible byte-BPE pilot and indexed, role-masked sequences."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from city_references import CITY_SLOTS, CityReferences

ROOT = Path(__file__).resolve().parent
CONTROLS = ['<pad>', '<bos>', '<eos>', '<user>', '<assistant>', '<tool>', '<system>',
            '<end_turn>', '<call>', '<city>'] + [f'<ref_{i}>' for i in range(1, 33)]
ATOMIC_CONTROLS = CONTROLS[:9] + CITY_SLOTS
ROLES = {'user', 'assistant', 'tool', 'system'}


def examples(args, provenance):
    if not args.no_local:
        folder = ROOT / 'conversations'
        excluded = json.loads((folder / 'excluded-conversations.json').read_text())
        files = sorted((folder / 'authored').glob('batch-*.jsonl'))
        fingerprint = hashlib.sha256()
        for path in files:
            blob = path.read_bytes()
            fingerprint.update(blob)
            for line in blob.splitlines():
                row = json.loads(line)
                if row['id'] not in excluded:
                    yield 'local', row['messages']
        fingerprint.update((folder / 'excluded-conversations.json').read_bytes())
        provenance.append(dict(dataset='local', sha256=fingerprint.hexdigest()))
    if args.calculator:
        from calculator_traces import calculator_traces
        provenance.append(dict(dataset='calculator', sha256=hashlib.sha256(
            (ROOT / 'calculator_traces.py').read_bytes()).hexdigest()))
        for messages in calculator_traces():
            yield 'calculator', messages
    if args.city_dialogues:
        from city_dialogues import city_dialogues
        provenance.append(dict(dataset='city-dialogues', sha256=hashlib.sha256(
            (ROOT / 'city_dialogues.py').read_bytes()).hexdigest()))
        for messages in city_dialogues():
            yield 'city-dialogues', messages
    for path in args.jsonl:
        fingerprint = hashlib.sha256()
        with path.open('rb') as stream:
            for line in stream:
                fingerprint.update(line)
                if line.strip():
                    row = json.loads(line)
                    yield str(path), row.get('messages', row.get('text'))
        provenance.append(dict(dataset=str(path), sha256=fingerprint.hexdigest()))
    for source in args.sources:
        from datasets import load_dataset
        from huggingface_hub import HfApi, hf_hub_download

        name = source['dataset']
        revision = HfApi().dataset_info(name, revision=source.get('revision', 'main')).sha
        config, split, limit = source.get('config'), source.get('split', 'train'), source['rows']
        if type(limit) is not int or limit < 1:
            raise ValueError('Each HF source needs a positive row limit')
        print(f'Reading up to {limit:,} rows from {name} at {revision[:12]}', file=sys.stderr, flush=True)
        shard = source.get('parquet_shard') if args.download_shards else None
        provenance.append({**source, 'config': config, 'split': split, 'revision': revision,
                           'downloaded_shard': shard})
        if shard:
            local = hf_hub_download(name, shard, repo_type='dataset', revision=revision)
            data = load_dataset('parquet', data_files=local, split='train', streaming=True)
        else:
            data = load_dataset(name, config, split=split, streaming=True, revision=revision)
        if source.get('shuffle_buffer'):
            data = data.shuffle(seed=20261108, buffer_size=source['shuffle_buffer'])
        column = source.get('column', 'text')
        for row in data.take(limit):
            value = row.get(column)
            if source.get('short_chat') and not short_chat(value):
                value = None
            yield name, value


def short_chat(value):
    if not isinstance(value, list) or not 2 <= len(value) <= 8:
        return False
    for item in value:
        if not isinstance(item, dict) or item.get('role') not in {'user', 'assistant'}:
            return False
        content = item.get('content')
        if not isinstance(content, str) or len(content.split()) > 90 or re.search(
                r'```|^\s*(?:def |function |import )|\b(?:roleplay|role-play|pretend (?:to be|you are)|'
                r'write (?:a |an )?(?:code|python|script|function))\b', content, re.I | re.M):
            return False
    return all(item['role'] == ('user' if i % 2 == 0 else 'assistant') for i, item in enumerate(value))


def clean(value):
    if isinstance(value, str):
        value = value.strip()
        return value if value and not any(tag in value for tag in CONTROLS + CITY_SLOTS) and not re.search(r'<unused_\d+>', value) else None
    if not isinstance(value, list) or not value:
        return None
    messages = []
    for item in value:
        if not isinstance(item, dict) or item.get('role') not in ROLES:
            return None
        content = clean(item.get('content')) if isinstance(item.get('content'), str) else None
        if not content:
            return None
        message = dict(role=item['role'], content=content)
        if item.get('call'):
            if item['role'] != 'assistant':
                return None
            message['call'] = True
        messages.append(message)
    if messages[-1]['role'] != 'assistant' or not any(m['role'] == 'user' for m in messages):
        return None
    for i, message in enumerate(messages):
        if message.get('call') and (i + 1 == len(messages) or messages[i + 1]['role'] != 'tool'):
            return None
        if message['role'] == 'tool' and (not i or not messages[i - 1].get('call')):
            return None
    return messages


def read_spool(path, train_only=False):
    with path.open() as stream:
        for line in stream:
            split, value = json.loads(line)
            if not train_only or split == 'train':
                yield split, value


def train_tokenizer(spool, size, limit, controls):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    def texts():
        for i, (_, value) in enumerate(read_spool(spool, train_only=True)):
            if i == limit:
                break
            if isinstance(value, str):
                yield value
            else:
                yield from (message['content'] for message in value)
    trainer = trainers.BpeTrainer(vocab_size=size, initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
                                  special_tokens=controls, min_frequency=2, show_progress=False)
    tokenizer.train_from_iterator(texts(), trainer)
    actual = tokenizer.get_vocab_size()
    tokenizer.add_special_tokens([f'<unused_{i}>' for i in range(actual, size)])
    return tokenizer, actual


def encode(tokenizer, value):
    ids = [tokenizer.token_to_id('<bos>')]
    mask = [0]
    eos = tokenizer.token_to_id('<eos>')
    if isinstance(value, str):
        body = tokenizer.encode(value, add_special_tokens=False).ids + [eos]
        return ids + body, mask + [0] * len(body)
    for message in value:
        role = message['role']
        body = [tokenizer.token_to_id(f'<{role}>')]
        if message.get('call'):
            body.append(tokenizer.token_to_id('<call>'))
        body += tokenizer.encode(message['content'], add_special_tokens=False).ids
        body.append(tokenizer.token_to_id('<end_turn>'))
        ids.extend(body)
        mask.extend([0] + [int(role == 'assistant')] * (len(body) - 1))
    return ids + [eos], mask + [1]


def prepare(args, output):
    provenance, counts, seen = [], Counter(), set()
    city_hash = None
    normalizer = None
    if args.cities:
        aliases = args.cities.read_bytes()
        city_hash = hashlib.sha256(aliases).hexdigest()
        normalizer = CityReferences(json.loads(aliases), atomic=args.paired)
    controls = ATOMIC_CONTROLS if args.paired else CONTROLS
    holdout = None
    if args.holdout_cities:
        holdout = re.compile(r'(?<!\w)(?:' + '|'.join(map(re.escape, args.holdout_cities)) + r')(?!\w)', re.I)
    spool = output / 'spool.jsonl'
    with spool.open('w') as stream:
        for source, raw in examples(args, provenance):
            value = clean(raw)
            if value is None:
                counts[f'{source}/rejected'] += 1
                continue
            if holdout and any(holdout.search(text) for text in (
                    [value] if isinstance(value, str) else [m['content'] for m in value])):
                counts[f'{source}/heldout_city'] += 1
                continue
            line = json.dumps(value, ensure_ascii=False, sort_keys=True)
            digest = hashlib.sha256(line.encode()).digest()
            if digest in seen:
                counts[f'{source}/duplicate'] += 1
                continue
            seen.add(digest)
            split = 'validation' if int.from_bytes(digest[:4], 'big') % 100 < 5 else 'train'
            stream.write(json.dumps([split, value], ensure_ascii=False) + '\n')
            counts[f'{source}/{split}'] += 1
    if args.tokenizer:
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        actual = sum(not token.startswith('<unused_') for token in tokenizer.get_vocab())
    else:
        tokenizer, actual = train_tokenizer(spool, args.vocab_size, args.tokenizer_rows, controls)
    if tokenizer.get_vocab_size() != args.vocab_size or any(
            tokenizer.token_to_id(tag) != index for index, tag in enumerate(controls)):
        raise ValueError('Tokenizer does not match the model vocabulary and control IDs')
    variants = {'raw': None, 'cities': normalizer} if args.paired else {'': normalizer}
    manifests = {}
    for name, transform in variants.items():
        folder = output / name
        folder.mkdir(exist_ok=True)
        manifests[name] = pack(args, folder, spool, tokenizer, actual, counts.copy(), provenance,
                               transform, normalizer if args.paired and not transform else None, city_hash)
    spool.unlink()
    return manifests if args.paired else manifests['']


def pack(args, output, spool, tokenizer, actual, counts, provenance, transform, paired_transform, city_hash):
    tokenizer.save(str(output / 'tokenizer.json'))

    packed = {}
    for split in ('train', 'validation'):
        index, tokens, supervised = [], 0, 0
        with (output / f'{split}.bin').open('wb') as target, (output / f'{split}.assistant_mask.bin').open('wb') as masks:
            for row_split, value in read_spool(spool):
                if row_split != split:
                    continue
                dialogue = isinstance(value, list)
                if dialogue and (transform or paired_transform):
                    parsed = (transform or paired_transform).normalize(value)
                    raw_ids, raw_mask = encode(tokenizer, value)
                    parsed_ids, parsed_mask = encode(tokenizer, parsed)
                    if max(len(raw_ids), len(parsed_ids)) > args.context + 1:
                        counts[f'{split}/overlong_dialogue'] += 1
                        continue
                    ids, mask = (parsed_ids, parsed_mask) if transform else (raw_ids, raw_mask)
                else:
                    ids, mask = encode(tokenizer, value)
                    if dialogue and len(ids) > args.context + 1:
                        counts[f'{split}/overlong_dialogue'] += 1
                        continue
                counts[f'{split}/{"dialogue" if dialogue else "prose"}_targets'] += len(ids) - 1
                for start in range(0, len(ids) - 1, args.context):
                    chunk = ids[start:start + args.context + 1]
                    labels = mask[start:start + args.context + 1]
                    active = sum(labels[1:])
                    index.append((tokens, len(chunk), active, len(raw_ids) if dialogue and (
                        transform or paired_transform) else len(chunk)))
                    tokens += len(chunk)
                    supervised += active
                    target.write(np.asarray(chunk, dtype='<u2').tobytes())
                    masks.write(bytes(labels))
        if not index:
            raise ValueError(f'No usable examples in {split}; use a larger sample')
        np.save(output / f'{split}.index.npy', np.asarray(index, dtype='<i8'))
        packed[split] = dict(records=len(index), stored_tokens=tokens,
                             next_token_targets=tokens - len(index), assistant_targets=supervised)
    manifest = dict(format=3, context=args.context, vocab_size=args.vocab_size, active_vocab=actual,
                    tokenizer_sha256=hashlib.sha256((output / 'tokenizer.json').read_bytes()).hexdigest(),
                    city_dictionary_sha256=city_hash if transform else None,
                    sources=provenance, source_counts=dict(counts), packed=packed,
                    holdout_cities=args.holdout_cities)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, help='JSON list of dataset, optional config/split/revision/column, and rows')
    parser.add_argument('--download-shards', action='store_true', help='Cache specified Parquet shards for local reading')
    parser.add_argument('--no-local', action='store_true')
    parser.add_argument('--jsonl', type=Path, action='append', default=[], help='Add JSONL text or messages')
    parser.add_argument('--cities', type=Path, help='Map unambiguous city names to canonical identities')
    parser.add_argument('--paired', action='store_true', help='Write matched raw and atomic-city variants')
    parser.add_argument('--city-dialogues', action='store_true', help='Add deterministic city reference exchanges')
    parser.add_argument('--holdout-cities', nargs='+', default=[], help='Exclude rows mentioning these city aliases')
    parser.add_argument('--calculator', action='store_true', help='Add exact calculator pilot traces')
    parser.add_argument('--tokenizer', type=Path, help='Reuse the same tokenizer across stages')
    parser.add_argument('--tokenizer-rows', type=int, default=50000)
    parser.add_argument('--context', type=int, default=512)
    parser.add_argument('--vocab-size', type=int, default=8192)
    args = parser.parse_args()
    args.sources = json.loads(args.source_manifest.read_text()) if args.source_manifest else []
    if args.paired and not args.cities:
        parser.error('--paired requires --cities')
    if not 1 <= args.context <= 2048 or not 300 <= args.vocab_size <= 65536 or args.tokenizer_rows < 1:
        parser.error('Invalid context or vocabulary size')
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.prepare-', dir=args.output.parent))
    try:
        manifest = prepare(args, temporary)
        temporary.rename(args.output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
