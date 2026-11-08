#!/usr/bin/env python3
"""Export the completed city-slot assistant checkpoint for browser inference."""
import hashlib
import json
from pathlib import Path
import shutil

from safetensors import safe_open

from inspect_architecture import inspect

root = Path(__file__).resolve().parent
checkpoint = root / 'runs/picoboi-10m-cities-assistant/last.safetensors'
tokenizer = root / 'runs/picoboi-10m-paired/cities/tokenizer.json'
destination = root.parent / 'web/dist/models/picoboi-10m-v1'
destination.mkdir(parents=True, exist_ok=True)
with safe_open(checkpoint, framework='numpy') as source:
    state = json.loads(source.metadata()['run'])
    settings = state['settings']
    if state['step'] != settings['steps'] or settings['loss'] != 'assistant':
        raise ValueError('Expected a completed assistant checkpoint')
    if hashlib.sha256(tokenizer.read_bytes()).hexdigest() != settings['tokenizer_sha256']:
        raise ValueError('Tokenizer and checkpoint disagree')
    config = settings['config']
    architecture = inspect(config)
    tensors, payload = {}, bytearray()
    for name, shape in architecture['tensor_shapes'].items():
        tensor = source.get_tensor(f'model.{name}')
        if list(tensor.shape) != shape:
            raise ValueError(f'Unexpected shape for {name}')
        tensors[name] = dict(shape=shape, offset=len(payload) // 4, length=tensor.size)
        payload.extend(tensor.astype('<f4', copy=False).tobytes())
manifest = dict(format='picoboi-f32-le-v1', name='PicoBoi-10M', parameters=architecture['unique_parameters'],
                config=config, tensors=tensors, weights='weights.bin', weights_bytes=len(payload),
                weights_sha256=hashlib.sha256(payload).hexdigest(),
                tokenizer_sha256=settings['tokenizer_sha256'])
(destination / 'weights.bin').write_bytes(payload)
shutil.copyfile(tokenizer, destination / 'tokenizer.json')
aliases = json.loads((root / 'picoboi_10m_cities.json').read_text())
aliases.update(Lisbon='lisbon', Auckland='auckland')
(destination / 'cities.json').write_text(json.dumps(aliases, indent=2) + '\n')
(destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f"Exported {manifest['parameters']:,} parameters in {len(payload):,} bytes")
