#!/usr/bin/env python3
"""Validate and count authored conversations. This script creates no dialogue."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'conversations'
TARGET = 10_000_000


def audit():
    base_words = (ROOT/'vocabulary_800.txt').read_text().splitlines()
    words = (DATA/'vocabulary.txt').read_text().splitlines()
    tokenizer = json.loads((DATA/'tokenizer.json').read_text())
    if len(base_words) != len(set(base_words)) or len(base_words) != 800:
        raise ValueError('The frozen baseline vocabulary must contain 800 distinct word forms')
    if len(words) != len(set(words)) or not set(base_words).issubset(words):
        raise ValueError('The new vocabulary must preserve every baseline word')
    expected_tokens = ['<pad>', '<unk>', '<bos>', '<eos>', '<user>', '<assistant>', '<turn>', '.', ',', '?', '!', ':'] + base_words
    expected_tokens += [word for word in words if word not in set(base_words)]
    if tokenizer['tokens'] != expected_tokens:
        raise ValueError('Tokenizer must preserve baseline IDs and append new tokens')
    excluded = json.loads((DATA/'excluded-conversations.json').read_text())
    seen_ids = set()
    vocabulary = set(words)
    ids, scenarios, digests = set(), set(), set()
    word_counts, topics, lengths, files = Counter(), Counter(), [], {}
    total_tokens = total_messages = assistant_words = 0
    errors = []
    for path in sorted((DATA/'authored').glob('*.jsonl')):
        file_words = file_dialogues = 0
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            row = json.loads(line)
            location = f'{path.name}:{line_number}'
            if row['id'] in seen_ids:
                errors.append(f'{location}: duplicate source ID')
            seen_ids.add(row['id'])
            if row['id'] in excluded:
                continue
            if row['id'] in ids or row['scenario'] in scenarios:
                errors.append(f'{location}: duplicate ID or scenario')
            ids.add(row['id']); scenarios.add(row['scenario'])
            if row.get('provenance') != 'assistant_direct_authorship':
                errors.append(f'{location}: missing direct-authorship provenance')
            messages = row['messages']
            if len(messages) < 2 or len(messages) % 2:
                errors.append(f'{location}: incomplete conversation')
            count = 2  # BOS/EOS
            for i, message in enumerate(messages):
                role, content = message['role'], message['content']
                expected_role = 'assistant' if i % 2 else 'user'
                if role != expected_role or not content or content != content.lower():
                    errors.append(f'{location}: invalid role or lowercase content at message {i}')
                tokens = re.findall(r'[a-z]+|[.,?!:]', content)
                if ''.join(tokens) != ''.join(content.split()):
                    errors.append(f'{location}: unsupported characters in message {i}')
                terms = re.findall(r'[a-z]+', content)
                unknown = sorted(set(terms)-vocabulary)
                if unknown:
                    errors.append(f'{location} message {i}: unknown words {unknown}')
                word_counts.update(terms)
                file_words += len(terms)
                if role == 'assistant': assistant_words += len(terms)
                count += len(tokens)+2  # role / turn boundary
            digest = hashlib.sha256(json.dumps(messages,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if digest in digests: errors.append(f'{location}: duplicate complete conversation')
            digests.add(digest)
            topics.update(row['topics'])
            lengths.append(count)
            total_tokens += count; total_messages += len(messages); file_dialogues += 1
        files[path.name] = dict(dialogues=file_dialogues, english_words=file_words,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if set(excluded)-seen_ids:
        errors.append('Exclusion manifest references missing source conversations')
    if errors:
        raise ValueError('\n'.join(errors))
    evaluation_cases = 0
    training_user_messages = set()
    for path in (DATA/'authored').glob('*.jsonl'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['id'] in excluded:
                continue
            training_user_messages.update(m['content'] for m in row['messages'] if m['role']=='user')
    for path in sorted((DATA/'evaluation').glob('*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if not row['messages'] or row['messages'][-1]['role'] != 'user':
                raise ValueError(f"Evaluation case must end with user: {row['id']}")
            if row['messages'][-1]['content'] in training_user_messages:
                raise ValueError(f"Held-out final prompt appears in training: {row['id']}")
            for i, message in enumerate(row['messages']):
                if message['role'] != ('assistant' if i%2 else 'user'):
                    raise ValueError(f"Wrong evaluation role order: {row['id']}")
            for text in [*(m['content'] for m in row['messages']), row['reference']]:
                tokens=re.findall(r'[a-z]+|[.,?!:]',text)
                if text != text.lower() or ''.join(tokens) != ''.join(text.split()) or set(re.findall('[a-z]+',text))-vocabulary:
                    raise ValueError(f"Evaluation content violates vocabulary: {row['id']}: {text}")
            evaluation_cases += 1
    completed = sum(word_counts.values())
    report = dict(status='passed', source='directly_authored_conversations_only', target_english_words=TARGET,
                  english_words=completed, remaining_english_words=max(0,TARGET-completed),
                  target_complete=completed>=TARGET, conversations=len(ids),
                  retained_source_conversations=len(seen_ids), excluded_conversations=excluded, messages=total_messages,
                  assistant_english_words=assistant_words, model_tokens=total_tokens,
                  held_out_relevance_cases=evaluation_cases, evaluation_words_counted_toward_target=False,
                  vocabulary_word_types_used=len(word_counts), vocabulary_limit=len(words),
                  added_vocabulary=sorted(vocabulary-set(base_words)), tokenizer_size=len(tokenizer['tokens']),
                  unused_vocabulary=sorted(vocabulary-set(word_counts)),
                  shortest_conversation_tokens=min(lengths,default=0), longest_conversation_tokens=max(lengths,default=0),
                  conversations_over_512_tokens=sum(n>512 for n in lengths), topics=dict(sorted(topics.items())),
                  files=files, limitations=['Structural validation does not verify semantic relevance.',
                                            'The full target is incomplete unless target_complete is true.',
                                            'Evaluation prompts are separately authored, but thematic overlap is still possible.'])
    (DATA/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','target_english_words','english_words','remaining_english_words',
                                         'conversations','vocabulary_word_types_used','longest_conversation_tokens')},indent=2))


if __name__=='__main__':
    audit()
