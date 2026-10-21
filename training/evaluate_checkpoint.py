#!/usr/bin/env python3
"""Evaluate a trusted local checkpoint and export portable FP32 model weights."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import torch
from model import Picoboi
from train import Dialogues, evaluate
from chat import answer


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=root/'runs/picoboi-100k-local')
    parser.add_argument('--data',type=Path,default=root/'data/picoboi-800-v1')
    parser.add_argument('--export',type=Path)
    args=parser.parse_args()
    status=json.loads((args.run_dir/'status.json').read_text())
    if status['status'] not in ('complete','early_stopped'):
        raise ValueError('Wait until training finishes before opening the held-out test set')
    torch.set_num_threads(4)
    state=torch.load(args.run_dir/'best.pt',map_location='cpu',weights_only=False)
    settings=state['settings']
    if hashlib.sha256((args.data/'sha256.json').read_bytes()).hexdigest()!=settings['data_fingerprint']:
        raise ValueError('Dataset fingerprint differs from training')
    model=Picoboi(settings['config']);model.load_state_dict(state['model']);model.eval()
    tokenizer=json.loads((args.data/'tokenizer.json').read_text())
    report=dict(parameters=sum(p.numel() for p in model.parameters()),checkpoint_step=state['step'],
                training_status=status,training_settings=settings,
                limitation='Generator families are shared across splits. Low token loss is not proof of natural chat ability.')
    for split in ('validation','test'):
        data=Dialogues(args.data,split,settings['config']['train_context'])
        loss=evaluate(model,data,64,'cpu')
        report[split]=dict(assistant_token_loss=loss,assistant_token_perplexity=math.exp(loss))
    prompts=['hello.', 'who are you?', 'can you help me?',
             'the red ball is on the table. where is it?',
             'the blue book is in the box. what color is it?',
             'i have two apples. i get three more. how many do i have now?',
             'please fix this sentence: he are happy.',
             'where is my book?']
    report['greedy_samples']=[dict(prompt=p,answer=answer(model,tokenizer,p)) for p in prompts]
    # Decode without feeding the expected assistant answer back to the model.
    corpus_report=json.loads((args.data/'report.json').read_text())
    attempts=[]
    for shard in corpus_report['splits']['test']['jsonl_shards']:
        with gzip.open(args.data/shard,'rt') as source:
            for line in source:
                row=json.loads(line)
                prompt,expected=row['messages'][0]['content'],row['messages'][1]['content']
                actual=answer(model,tokenizer,prompt,max_tokens=64)
                attempts.append(dict(id=row['id'],family=row['family'],prompt=prompt,
                                     expected=expected,actual=actual,exact_match=actual==expected))
                if len(attempts)>=100: break
        if len(attempts)>=100: break
    report['heldout_first_answers']=dict(count=len(attempts),exact_matches=sum(r['exact_match'] for r in attempts),
                                       samples=attempts,
                                       limitation='First 100 test dialogues, shared template families; stories may have multiple valid answers.')
    families={r['family'] for r in attempts}
    report['heldout_first_answers']['by_family']={family:dict(
        count=sum(r['family']==family for r in attempts),
        exact_matches=sum(r['family']==family and r['exact_match'] for r in attempts))
        for family in sorted(families)}
    if args.export:
        args.export.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(status='trained',model=state['model'],settings=settings,step=state['step'],
                     best_validation_loss=state['best_validation_loss'],
                     tokenizer_sha256=hashlib.sha256((args.data/'tokenizer.json').read_bytes()).hexdigest())
        torch.save(payload,args.export)
        report['export']=dict(path=args.export.name,bytes=args.export.stat().st_size,
                             sha256=hashlib.sha256(args.export.read_bytes()).hexdigest())
    (args.run_dir/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
