#!/usr/bin/env python3
"""Validate proposed Picoboi configurations and account for their tensor shapes.

No model is instantiated or trained. Runtime estimates are not benchmarks.
"""
import argparse
import json
from math import prod
from pathlib import Path


def inspect(config):
    v, d, f = (config[k] for k in ('vocab_size', 'd_model', 'd_ff'))
    n, heads = config['unique_blocks'], config['n_heads']
    order, context = config['block_order'], config['train_context']
    if any(type(x) is not int or x <= 0 for x in (v, d, f, n, heads, context)):
        raise ValueError('Dimensions must be positive integers')
    if d % heads or (d // heads) % 2:
        raise ValueError('Attention head width must be an even integer for RoPE')
    if not order or any(type(i) is not int for i in order) or set(order) != set(range(n)):
        raise ValueError('Every unique block must be used, with valid integer indices')
    supported = dict(normalization='pre_rmsnorm', feed_forward='swiglu', position_encoding='rope',
                     tie_embeddings=True, linear_bias=False, output_bias=False, learned_step_embeddings=False)
    if any(config[k] != value for k, value in supported.items()):
        raise ValueError('This accounting supports only tied, bias-free RoPE/SwiGLU/RMSNorm configurations')
    tensors = {'embedding.weight': [v, d], 'final_norm.weight': [d]}
    for i in range(n):
        prefix = f'blocks.{i}.'
        for projection in ('q', 'k', 'v', 'o'):
            tensors[prefix + projection + '.weight'] = [d, d]
        for projection in ('gate', 'up'):
            tensors[prefix + projection + '.weight'] = [f, d]
        tensors[prefix + 'down.weight'] = [d, f]
        for norm in ('attention_norm', 'ffn_norm'):
            tensors[prefix + norm + '.weight'] = [d]
    # Output projection aliases embedding.weight; reused blocks alias their original tensors.
    total = sum(prod(shape) for shape in tensors.values())
    formula = v*d + n*(4*d*d + 3*d*f) + (2*n+1)*d
    if total != formula or total != config['expected_parameters'] or total > config['parameter_budget']:
        raise ValueError(f'Parameter count/budget mismatch: {total}')
    stages = len(order)
    return dict(name=config['name'], status=config['status'], unique_parameters=total,
                unique_blocks=n, executed_blocks=stages, block_order=order,
                head_dimension=d//heads, train_context=context,
                parameter_breakdown=dict(embeddings=v*d, attention=n*4*d*d,
                                         feed_forward=n*3*d*f, normalization=(2*n+1)*d),
                raw_weight_bytes=dict(fp32=4*total, fp16=2*total, int8=total),
                fp16_kv_cache_bytes_at_full_context=2*context*d*stages*2,
                approximate_cached_decode_macs_at_full_context=stages*(4*d*d+3*d*f+2*context*d)+v*d,
                estimate_limits='Batch size one; MHA; separate KV cache per execution stage; excludes '
                                'RoPE, norms, activations, softmax, runtime buffers, metadata, and quantization scales.',
                tensor_shapes=tensors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configs', nargs='*', type=Path)
    args = parser.parse_args()
    paths = args.configs or sorted((Path(__file__).resolve().parent / 'configs').glob('*.json'))
    reports = [inspect(json.loads(path.read_text())) for path in paths]
    print(json.dumps(reports, indent=2))
