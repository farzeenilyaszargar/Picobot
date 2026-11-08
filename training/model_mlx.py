"""Apple Silicon implementation of the same Picoboi weight layout."""
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from inspect_architecture import inspect


class Linear(nn.Linear):
    def __call__(self, x):
        return x @ self.weight.astype(x.dtype).T


class RMSNorm(nn.RMSNorm):
    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight.astype(x.dtype), self.eps)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, f = config['d_model'], config['d_ff']
        self.heads = config['n_heads']
        self.kv_heads = config.get('n_kv_heads', self.heads)
        self.head_dim = d // self.heads
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=config['rope_theta'])
        self.attention_norm = RMSNorm(d, eps=config['norm_epsilon'])
        self.ffn_norm = RMSNorm(d, eps=config['norm_epsilon'])
        for name in ('q', 'k', 'v', 'o'):
            width = self.kv_heads * self.head_dim if name in ('k', 'v') else d
            setattr(self, name, Linear(d, width, bias=False))
        self.gate = Linear(d, f, bias=False)
        self.up = Linear(d, f, bias=False)
        self.down = Linear(f, d, bias=False)

    def __call__(self, x):
        b, t, d = x.shape
        z = self.attention_norm(x)
        q = self.q(z).reshape(b, t, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k, v = [projection(z).reshape(b, t, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
                for projection in (self.k, self.v)]
        attention = mx.fast.scaled_dot_product_attention(
            self.rope(q), self.rope(k), v, scale=self.head_dim ** -0.5, mask='causal')
        x = x + self.o(attention.transpose(0, 2, 1, 3).reshape(b, t, d))
        z = self.ffn_norm(x)
        return x + self.down(nn.silu(self.gate(z)) * self.up(z))


class Picoboi(nn.Module):
    def __init__(self, config, compute_dtype=mx.float32):
        super().__init__()
        report = inspect(config)
        self.config = config
        self.compute_dtype = compute_dtype
        d = config['d_model']
        self.embedding = nn.Embedding(config['vocab_size'], d)
        self.blocks = [Block(config) for _ in range(config['unique_blocks'])]
        self.final_norm = RMSNorm(d, eps=config['norm_epsilon'])
        params = tree_flatten(self.parameters())
        if sum(value.size for _, value in params) != report['unique_parameters']:
            raise ValueError('Parameter count disagrees with the architecture')
        initialized = []
        for name, value in params:
            if value.ndim == 2:
                scale = 0.02
                if name.endswith(('.o.weight', '.down.weight')):
                    scale /= math.sqrt(2 * len(config['block_order']))
                value = mx.random.normal(value.shape) * scale
            initialized.append((name, value))
        self.update(tree_unflatten(initialized))
        mx.eval(self.parameters())

    def __call__(self, tokens):
        if tokens.shape[1] > self.config['train_context']:
            raise ValueError('Input exceeds configured context')
        x = self.embedding(tokens).astype(self.compute_dtype)
        for i in self.config['block_order']:
            x = self.blocks[i](x)
        return self.final_norm(x) @ self.embedding.weight.astype(x.dtype).T
