#!/usr/bin/env python3
"""Parallel teacher-generated chats, separate from directly authored source data."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokenization import load_variant

TASKS = [
    'a greeting develops into a natural conversation about the day',
    'someone explains a preference and the other person asks a relevant follow-up',
    'someone changes a plan and the answer respects the new plan',
    'someone asks for help finding an object using stated locations',
    'a short story is followed by questions about its events',
    'someone asks for a longer story and then asks for a shorter account',
    'someone shares a feeling and receives a kind, specific answer',
    'someone asks an unclear question and receives a useful clarification question',
    'someone gives a correction and the answer uses the corrected fact',
    'someone asks for a simple explanation and then gives a concrete example',
    'someone asks for a small plan with a clear order of actions',
    'someone compares two choices using only the supplied facts',
    'someone asks a question with not and the answer keeps the negation correct',
    'someone asks about yesterday and tomorrow with correct verb forms',
    'someone wants a sentence explained or corrected',
    'someone asks a small counting question using numbers from zero to ten',
    'someone asks what another person said or wanted earlier in the chat',
    'someone asks for information that was never supplied and gets an honest answer',
    'someone asks how to do an ordinary activity and gets clear steps',
    'someone disagrees politely and explains the reason',
    'someone asks to change the tone or length of an earlier answer',
    'someone describes an object and the other person asks a useful question',
    'someone asks a question about cause and effect in a supplied event',
    'someone starts with small talk and then asks a practical question',
]
SETTINGS = ['home', 'school', 'garden', 'kitchen', 'a visit to a friend', 'a journey',
            'a shared meal', 'a quiet evening', 'a family day', 'a walk outside',
            'reading and writing', 'work']


def sha(value):
    return hashlib.sha256(value).hexdigest()


GROUP_ORDER = sorted(sha((task+'\n'+setting).encode()) for task in TASKS for setting in SETTINGS)
GROUP_SPLITS = {group: 'test' if i < 3 else 'validation' if i < 6 else 'train'
                for i,group in enumerate(GROUP_ORDER)}


def grammar(words):
    word = '(?:' + '|'.join(sorted(words, key=lambda w: (-len(w), w))) + ')'
    text = word + '(?:(?: |[.,?!:] )' + word + ')*[.,?!:]?'
    pair = 'user: ' + text + '\nassistant: ' + text
    return pair + '(?:\n' + pair + '){0,5}'


def profile(index, seed):
    rng = random.Random(f'{seed}:{index}')
    task, setting = rng.choice(TASKS), rng.choice(SETTINGS)
    # Group assignment is independent of length/style or the number of retries.
    group = sha((task + '\n' + setting).encode())
    return dict(task=task, setting=setting, group=group,
                split=GROUP_SPLITS[group],
                length=rng.choice(['short', 'medium', 'long']),
                style=rng.choice(['friendly', 'calm', 'direct', 'curious']))


def make_request(config, index, words):
    plan = profile(index, config['seed'])
    lengths = {'short': '40 to 100', 'medium': '120 to 240', 'long': '260 to 420'}
    system = '''Write one original, coherent user and assistant conversation for a tiny English chatbot.
Use only the permitted lowercase word forms below in message content. Inflections must also be listed.
Use only spaces and these punctuation marks: . , ? ! :
Format each turn on its own line, starting with user: or assistant: . Alternate roles and end with an assistant answer.
Write one to six user/assistant pairs. Do not add titles, metadata, JSON, or commentary.
Keep answers relevant and grammar natural. Do not pad, repeat a sentence for volume, or make vocabulary lists.
Use broad concepts such as animal and fruit. Do not invent facts about their specific kinds.
Do not include reference markers, handles, URLs, tool outputs, or claims of real web access.
Avoid confusing two specific things that have the same category name. Use clear descriptions when needed.
The assistant should ask when information is missing and track facts supplied earlier.
Permitted content words:
''' + ' '.join(words)
    prompt = (f"Situation: {plan['task']}. Setting: {plan['setting']}. "
              f"Style: {plan['style']}. Aim for {lengths[plan['length']]} English words total. "
              f"Create new details and a distinct exchange. Variation label: {config['seed']}-{index}.")
    payload = dict(model=config['model'], messages=[dict(role='system', content=system),
                   dict(role='user', content=prompt)], temperature=0.9, top_p=0.95,
                   max_tokens=config['max_output_tokens'])
    if config['constrained']:
        payload['structured_outputs'] = {'regex': grammar(words)}
    return plan, payload


def validate(text, codecs, min_words=20, max_context=1024):
    messages = []
    for i, line in enumerate(text.strip().splitlines()):
        role = 'user' if i % 2 == 0 else 'assistant'
        prefix = role + ': '
        if not line.startswith(prefix):
            raise ValueError('format')
        content = line[len(prefix):]
        if not content or content != content.strip():
            raise ValueError('format')
        messages.append(dict(role=role, content=content))
    if not messages or len(messages) % 2 or len(messages) > 12:
        raise ValueError('format')
    # References and non-ASCII characters are intentionally excluded in this new stream.
    words = []
    allowed = set(codecs['word-level'].spec['word_vocabulary'])
    for message in messages:
        content = message['content']
        if re.search(r'[^a-z .,?!:]', content):
            raise ValueError('characters')
        current = re.findall('[a-z]+', content)
        if not set(current) <= allowed:
            raise ValueError('vocabulary')
        words.extend(current)
    if len(words) < min_words:
        raise ValueError('too_short')
    sentences = [s.strip() for m in messages if m['role'] == 'assistant'
                 for s in re.split('[.!?]', m['content']) if len(s.split()) >= 8]
    if len(sentences) != len(set(sentences)):
        raise ValueError('repeated_assistant_sentence')
    counts = {}
    for name, codec in codecs.items():
        ids, _ = codec.encode_conversation(messages)
        if len(ids) > max_context:
            raise ValueError('context')
        counts[name] = len(ids)
    return messages, len(words), counts


class Dedupe:
    """Exact plus approximate 5-word-shingle screening, not exhaustive semantic dedupe."""
    def __init__(self):
        self.exact = set()
        self.bands = defaultdict(list)
        self.documents = []

    @staticmethod
    def shingles(text):
        words = text.split()
        return {' '.join(words[i:i+5]) for i in range(max(1, len(words)-4))}

    @staticmethod
    def sketch(shingles):
        values = [hashlib.blake2b(s.encode(), digest_size=32).digest() for s in shingles]
        return tuple(min(v[i:i+8] for v in values) for i in range(0,32,8))

    def add(self, messages):
        # Keep word order and turn boundaries; ignore punctuation for comparison.
        text = ' | '.join(' '.join(re.findall('[a-z]+', m['content'])) for m in messages)
        digest = sha(text.encode())
        if digest in self.exact:
            return 'exact_duplicate'
        shingles = self.shingles(text)
        sketch = self.sketch(shingles)
        candidates = {i for band,key in enumerate(sketch) for i in self.bands[(band,key)]}
        for i in candidates:
            other = self.shingles(self.documents[i])
            if len(shingles & other) / len(shingles | other) >= 0.85:
                return 'near_duplicate'
        self.exact.add(digest)
        index = len(self.documents)
        self.documents.append(text)
        for band,key in enumerate(sketch): self.bands[(band,key)].append(index)
        return None


class EndpointError(RuntimeError):
    pass


def request(config, payload):
    headers = {'Content-Type': 'application/json'}
    key = os.environ.get('PICOBOI_TEACHER_API_KEY')
    if key: headers['Authorization'] = 'Bearer ' + key
    req = urllib.request.Request(config['base_url'].rstrip('/')+'/chat/completions',
                                 data=json.dumps(payload).encode(), headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            error.close()
            if error.code not in {429,500,502,503,504}: raise EndpointError(f'HTTP {error.code}') from None
        except (urllib.error.URLError, TimeoutError):
            pass
        if attempt < 2: time.sleep(2 ** attempt)
    raise RuntimeError('request_failed_after_retries')


def run(args):
    output = Path(args.output)
    config = dict(version=1, model=args.model, base_url=args.base_url,
                  seed=args.seed, constrained=not args.no_constraints,
                  max_output_tokens=args.max_output_tokens)
    specs = {name: load_variant(name) for name in ['word-level','bpe-512']}
    words = specs['word-level'].spec['word_vocabulary']
    config['tokenizers'] = {name: sha(json.dumps(codec.spec, sort_keys=True).encode())
                            for name,codec in specs.items()}
    config['generator_sha256'] = sha(Path(__file__).read_bytes())
    run_id = sha(json.dumps(config, sort_keys=True).encode())[:12]
    if args.preview:
        plan, payload = make_request(config, 0, words)
        print(json.dumps(dict(profile=plan, request=payload), indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    config_file = output/'config.json'
    if config_file.exists():
        if json.loads(config_file.read_text()) != config:
            raise ValueError('Configuration changed. Use a new output directory.')
    else:
        if (output/'accepted.jsonl').exists() or (output/'attempts.jsonl').exists():
            raise ValueError('Existing records have no configuration. Use a new directory.')
        config_file.write_text(json.dumps(config,indent=2)+'\n')
    dedupe = Dedupe()
    # Exclude existing authored/evaluation text from exact and approximate matches.
    for folder in ['authored','evaluation']:
        for path in sorted((ROOT/'conversations'/folder).glob('*.jsonl')):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if 'messages' in row: dedupe.add(row['messages'])
    total_words, total_tokens, accepted, done = 0, Counter(), 0, set()
    accepted_path, attempts_path = output/'accepted.jsonl', output/'attempts.jsonl'
    if accepted_path.exists():
        for line in accepted_path.read_text().splitlines():
            row = json.loads(line)
            _, count, tokens = validate(row['messages_text'], specs)
            dedupe.add(row['messages'])
            total_words += count
            total_tokens.update(tokens)
            accepted += 1
            done.add(row['job'])
    if attempts_path.exists():
        done.update(json.loads(line)['job'] for line in attempts_path.read_text().splitlines())
    next_job = max(done, default=-1) + 1
    started, start_words = time.monotonic(), total_words
    reasons, attempted, halted = Counter(), 0, False
    with accepted_path.open('a') as good, attempts_path.open('a') as log, \
         ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {}
        def submit():
            nonlocal next_job
            job = next_job
            next_job += 1
            plan, payload = make_request(config, job, words)
            pending[executor.submit(request, config, payload)] = (job,plan)
        while pending or (not halted and total_words < args.target_words and attempted < args.max_jobs):
            while (not halted and total_words < args.target_words and len(pending) < args.workers
                   and attempted + len(pending) < args.max_jobs):
                submit()
            if not pending: break
            completed,_ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                job,plan = pending.pop(future)
                attempted += 1
                record = dict(job=job, profile=plan)
                try:
                    response = future.result()
                    record['usage'] = response.get('usage',{})
                    choice = response['choices'][0]
                    if choice.get('finish_reason') != 'stop': raise ValueError('incomplete_output')
                    text = choice['message']['content']
                    messages,count,tokens = validate(text,specs)
                    duplicate = dedupe.add(messages)
                    if duplicate: raise ValueError(duplicate)
                    row = dict(id=f"fast-{run_id}-{job:08d}", job=job,
                               provenance='teacher_synthetic_generated', teacher=args.model,
                               profile=plan, messages=messages, messages_text=text,
                               english_words=count, model_tokens=tokens,
                               quality_status='mechanical_checks_only_not_semantically_reviewed')
                    good.write(json.dumps(row)+'\n')
                    good.flush()
                    total_words += count
                    total_tokens.update(tokens)
                    accepted += 1
                    record['status'] = 'accepted'
                except (ValueError, KeyError, TypeError, IndexError, RuntimeError) as error:
                    if isinstance(error, EndpointError): halted = True
                    record['status'] = 'rejected'
                    record['reason'] = str(error)
                    reasons[str(error)] += 1
                log.write(json.dumps(record)+'\n')
                log.flush()
            elapsed = time.monotonic()-started
            print(json.dumps(dict(attempted=attempted, accepted_total=accepted,
                                  english_words_total=total_words,
                                  new_words_per_second=round((total_words-start_words)/elapsed,2))),flush=True)
    elapsed = time.monotonic()-started
    rate = (total_words-start_words)/elapsed
    report = dict(target_english_words=args.target_words, english_words=total_words,
                  model_tokens=dict(total_tokens), accepted_total=accepted,
                  attempted_this_run=attempted, rejected_this_run=dict(reasons),
                  halted_for_endpoint_error=halted,
                  elapsed_seconds=elapsed, accepted_words_per_second=rate,
                  estimated_remaining_hours=(max(0,args.target_words-total_words)/rate/3600 if rate else None),
                  status='mechanical_target_reached_pending_quality_review' if total_words >= args.target_words
                         else 'partial_pending_more_generation_and_quality_review',
                  note='Rates include local mechanical screening, not semantic review. In-flight jobs may overshoot target.')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--base-url',required=True,help='vLLM-style endpoint including /v1')
    parser.add_argument('--output',default=str(ROOT/'data/fast-chat-v1'))
    parser.add_argument('--target-words',type=int,default=100000)
    parser.add_argument('--max-jobs',type=int,default=1000,help='New requests allowed in this invocation')
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max-output-tokens',type=int,default=1400,help='Teacher tokens, not Picoboi tokens')
    parser.add_argument('--no-constraints',action='store_true',help='Prompt-only generation for other compatible servers')
    parser.add_argument('--preview',action='store_true',help='Print first request without network access or output files')
    args = parser.parse_args()
    if min(args.target_words,args.max_jobs,args.workers,args.max_output_tokens) < 1:
        parser.error('Counts must be positive')
    run(args)


if __name__ == '__main__': main()
