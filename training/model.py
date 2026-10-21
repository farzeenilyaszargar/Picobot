"""The bias-free, tied-embedding Picoboi transformer."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from inspect_architecture import inspect


class RMSNorm(nn.Module):
    def __init__(self, width, epsilon):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.epsilon) * self.weight


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, f = config['d_model'], config['d_ff']
        self.heads = config['n_heads']
        self.head_dim = d // self.heads
        self.attention_norm = RMSNorm(d, config['norm_epsilon'])
        self.ffn_norm = RMSNorm(d, config['norm_epsilon'])
        for name in ('q', 'k', 'v', 'o'):
            setattr(self, name, nn.Linear(d, d, bias=False))
        self.gate = nn.Linear(d, f, bias=False)
        self.up = nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        z = self.attention_norm(x)
        q, k, v = [layer(z).view(b, t, self.heads, self.head_dim).transpose(1, 2)
                   for layer in (self.q, self.k, self.v)]
        def rotate(z):
            even, odd = z[..., 0::2], z[..., 1::2]
            return torch.stack((even*cos - odd*sin, even*sin + odd*cos), -1).flatten(-2)
        attended = F.scaled_dot_product_attention(rotate(q), rotate(k), v, is_causal=True)
        x = x + self.o(attended.transpose(1, 2).reshape(b, t, d))
        z = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(z)) * self.up(z))


class Picoboi(nn.Module):
    def __init__(self, config):
        super().__init__()
        inspect(config)
        self.config = config
        d = config['d_model']
        self.embedding = nn.Embedding(config['vocab_size'], d)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config['unique_blocks'])])
        self.final_norm = RMSNorm(d, config['norm_epsilon'])
        head_dim = d // config['n_heads']
        angles = torch.outer(torch.arange(config['train_context'], dtype=torch.float32),
                             config['rope_theta'] ** (-torch.arange(0, head_dim, 2, dtype=torch.float32)/head_dim))
        self.register_buffer('rope_cos', angles.cos()[None, None], persistent=False)
        self.register_buffer('rope_sin', angles.sin()[None, None], persistent=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
        for block in self.blocks:
            for layer in (block.o, block.down):
                nn.init.normal_(layer.weight, std=0.02/math.sqrt(2*len(config['block_order'])))
        count = sum(p.numel() for p in self.parameters())
        if count != config['expected_parameters']:
            raise ValueError(f'Actual model has {count} parameters')

    def forward(self, tokens):
        t = tokens.shape[1]
        if t > self.config['train_context']:
            raise ValueError('Input exceeds configured context')
        x = self.embedding(tokens)
        for block_id in self.config['block_order']:
            x = self.blocks[block_id](x, self.rope_cos[..., :t, :], self.rope_sin[..., :t, :])
        return F.linear(self.final_norm(x), self.embedding.weight)


def masked_loss(logits, labels, mask):
    losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none').view_as(mask)
    numerator, denominator = (losses * mask).sum(), mask.sum()
    return numerator / denominator.clamp_min(1), numerator.detach(), denominator.detach()
