#!/usr/bin/env python3
"""Export actual checkpoint tensors to a dependency-free browser asset."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
checkpoint = ROOT / 'checkpoints/picoboi-100k-v1.pt'
destination = ROOT.parent / 'web/dist/models/picoboi-100k-v1'
state = torch.load(checkpoint, map_location='cpu', weights_only=False)
destination.mkdir(parents=True, exist_ok=True)
tensors, payload = {}, bytearray()
for name, tensor in state['model'].items():
    blob = tensor.detach().float().contiguous().numpy().astype('<f4').tobytes()
    tensors[name] = dict(shape=list(tensor.shape), offset=len(payload)//4, length=tensor.numel())
    payload.extend(blob)
count = sum(t['length'] for t in tensors.values())
if count != 99120:
    raise ValueError(f'Unexpected parameter count: {count}')
(destination/'weights.bin').write_bytes(payload)
manifest = dict(format='picoboi-f32-le-v1', name='Picoboi-100K', parameters=count,
                config=state['settings']['config'], tensors=tensors,
                weights='weights.bin', weights_bytes=len(payload),
                weights_sha256=hashlib.sha256(payload).hexdigest(),
                source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
(destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
tokenizer = json.loads((ROOT/'data/picoboi-800-v1/tokenizer.json').read_text())
(destination/'tokenizer.json').write_text(json.dumps(dict(tokens=tokenizer['tokens'],token_to_id=tokenizer['token_to_id']),separators=(',',':'))+'\n')
print(json.dumps(dict(parameters=count, weights_bytes=len(payload), sha256=manifest['weights_sha256'])))
