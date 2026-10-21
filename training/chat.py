#!/usr/bin/env python3
"""Inspect a local trusted Picoboi checkpoint, without connecting the website."""
import argparse
import json
from pathlib import Path
import re
import torch
from model import Picoboi


def answer(model,tokenizer,prompt,max_tokens=40):
    text=prompt.lower().strip()
    pieces=re.findall(r'[a-z]+|[.,?!:]',text)
    if ''.join(pieces)!=''.join(text.split()): raise ValueError('Unsupported characters in prompt')
    ids=tokenizer['token_to_id']
    unknown=sorted(set(pieces)-set(ids))
    if unknown: raise ValueError(f'Words outside the frozen vocabulary: {unknown}')
    sequence=[ids['<bos>'],ids['<user>'],*[ids[p] for p in pieces],ids['<turn>'],ids['<assistant>']]
    if len(sequence)>=model.config['train_context']: raise ValueError('Prompt leaves no room for an answer')
    generated=[]
    forbidden=[ids[t] for t in ('<pad>','<unk>','<bos>','<user>','<assistant>')]
    with torch.no_grad():
        for _ in range(min(max_tokens,model.config['train_context']-len(sequence))):
            logits=model(torch.tensor([sequence],dtype=torch.long))[:,-1,:]
            logits[:,forbidden]=-float('inf')
            token=int(logits.argmax(-1).item())
            if token in (ids['<turn>'],ids['<eos>']): break
            sequence.append(token);generated.append(tokenizer['tokens'][token])
    return re.sub(r'\s+([.,?!:])',r'\1',' '.join(generated))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt')
    parser.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parent/'runs/picoboi-100k-local/best.pt')
    parser.add_argument('--max-tokens',type=int,default=40)
    args=parser.parse_args()
    torch.set_num_threads(2)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=Picoboi(state['settings']['config']);model.load_state_dict(state['model']);model.eval()
    tokenizer=json.loads((Path(__file__).resolve().parent/'data/picoboi-800-v1/tokenizer.json').read_text())
    print(answer(model,tokenizer,args.prompt,args.max_tokens))
