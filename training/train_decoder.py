#!/usr/bin/env python3
"""Bounded Picoboi decoder training, resume and local throughput benchmarks."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
from inspect_architecture import inspect

ROOT = Path(__file__).resolve().parent


class Sequences:
    def __init__(self, root, split, context, vocab, mode, dialogue_share=None):
        self.ids = np.memmap(root / f'{split}.bin', mode='r', dtype='<u2')
        self.mask = np.memmap(root / f'{split}.assistant_mask.bin', mode='r', dtype='u1')
        self.index = np.load(root / f'{split}.index.npy', mmap_mode='r', allow_pickle=False)
        if self.index.ndim != 2 or self.index.shape[1] not in (3, 4) or not len(self.index):
            raise ValueError(f'Invalid {split} index')
        starts, sizes, targets = self.index[:, :3].T
        sampling_sizes = self.index[:, 3] if self.index.shape[1] == 4 else sizes
        if (len(self.ids) != len(self.mask) or starts[0] != 0
                or (starts[1:] != starts[:-1] + sizes[:-1]).any()
                or starts[-1] + sizes[-1] != len(self.ids)
                or sizes.min() < 2 or sizes.max() > context + 1
                or sampling_sizes.min() < 2 or sampling_sizes.max() > context + 1
                or (targets < 0).any() or (targets >= sizes).any()):
            raise ValueError(f'Invalid {split} sequence boundaries')
        self.rows = np.flatnonzero(targets) if mode == 'assistant' else np.arange(len(self.index))
        if not len(self.rows):
            raise ValueError(f'No {mode} training examples in {split}')
        self.context, self.vocab = context, vocab
        self.bucket_ids = (sampling_sizes[self.rows] - 2) // 64
        self.anchors = None
        if dialogue_share is not None:
            groups = targets[self.rows] > 0
            self.anchors = [np.flatnonzero(groups == kind) for kind in (False, True)]
            if any(not len(group) for group in self.anchors):
                raise ValueError('Dialogue mix requires both prose and chat')
            prose, dialogue = [(sampling_sizes[self.rows[group]] - 1).mean() for group in self.anchors]
            self.dialogue_probability = dialogue_share * prose / (
                dialogue_share * prose + (1 - dialogue_share) * dialogue)
            self.bucket_ids = self.bucket_ids * 2 + groups
        self.buckets = {int(key): np.flatnonzero(self.bucket_ids == key)
                        for key in np.unique(self.bucket_ids)}

    def sample(self, rng, count):
        if self.anchors is None:
            anchor = rng.integers(len(self.rows))
        else:
            anchor = rng.choice(self.anchors[int(rng.random() < self.dialogue_probability)])
        return rng.choice(self.buckets[int(self.bucket_ids[anchor])], size=count)

    def steps_for_tokens(self, seed, batch_size, token_budget):
        rng = np.random.default_rng(seed)
        sizes = self.index[:, -1] if self.index.shape[1] == 4 else self.index[:, 1]
        tokens = steps = 0
        while tokens < token_budget:
            rows = self.sample(rng, batch_size)
            tokens += int((sizes[self.rows[rows]] - 1).sum())
            steps += 1
        return steps, tokens

    def batch(self, rows, mode):
        entries = self.index[self.rows[rows]]
        length = int(entries[:, 1].max()) - 1
        width = min(self.context, ((length + 63) // 64) * 64)
        tokens = np.zeros((len(entries), width + 1), dtype=np.int32)
        weights = np.zeros((len(entries), width), dtype=np.float32)
        processed = 0
        for i, (offset, size, _) in enumerate(entries[:, :3]):
            values = self.ids[offset:offset + size]
            if values.max() >= self.vocab:
                raise ValueError('Token ID outside vocabulary')
            tokens[i, :size] = values
            weights[i, :size - 1] = 1 if mode == 'full' else self.mask[offset + 1:offset + size]
            processed += int(size) - 1
        return tokens[:, :-1].copy(), tokens[:, 1:].copy(), weights, processed


class TorchTrainer:
    def __init__(self, config, args):
        import torch
        from model import Picoboi
        from train import optimizer_for

        self.torch = torch
        self.device = args.device
        self.precision = args.precision
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        self.model = Picoboi(config).to(self.device)
        self.optimizer = optimizer_for(self.model, args.lr)
        if self.device == 'cuda':
            # Fused FP32 AdamW updates with BF16 compute when selected.
            self.optimizer = torch.optim.AdamW(self.optimizer.param_groups, lr=args.lr,
                                               betas=(0.9, 0.95), fused=True)

    def loss_sum(self, x, y, mask):
        torch = self.torch
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.precision == 'bf16'):
            logits = self.model(x.long())
            losses = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1).float(), y.long().flatten(), reduction='none')
            return (losses * mask.flatten()).sum()

    def step(self, batches, lr):
        torch = self.torch
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.zero_grad(set_to_none=True)
        count = sum(float(mask.sum()) for _, _, mask in batches)
        total = 0.
        for batch in batches:
            tensors = [torch.from_numpy(x).to(self.device) for x in batch]
            loss = self.loss_sum(*tensors) / count
            loss.backward()
            total = total + loss.detach()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return float(total)

    def evaluate(self, batch):
        with self.torch.no_grad():
            return float(self.loss_sum(*[self.torch.from_numpy(x).to(self.device) for x in batch]))

    def save(self, path, metadata):
        self.torch.save(dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                             metadata=metadata), path)

    def load(self, path, weights_only=False):
        if path.suffix == '.safetensors':
            if not weights_only:
                raise ValueError('Cross-backend loading supports weights-only initialization')
            from safetensors import safe_open

            with safe_open(path, framework='pt', device='cpu') as source:
                weights = {key.removeprefix('model.'): source.get_tensor(key)
                           for key in source.keys() if key.startswith('model.')}
                self.model.load_state_dict(weights)
                return json.loads(source.metadata()['run'])
        payload = self.torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        self.model.load_state_dict(payload['model'])
        if not weights_only:
            self.optimizer.load_state_dict(payload['optimizer'])
        return payload['metadata']

    def memory(self):
        if self.device == 'mps':
            return self.torch.mps.driver_allocated_memory()
        if self.device == 'cuda':
            return self.torch.cuda.max_memory_allocated()
        return None


class MLXTrainer:
    def __init__(self, config, args):
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_map
        from model_mlx import Picoboi

        self.mx, self.nn = mx, nn
        mx.random.seed(args.seed)
        self.model = Picoboi(config, compute_dtype=mx.bfloat16 if args.precision == 'bf16' else mx.float32)
        self.optimizer = optim.MultiOptimizer([
            optim.AdamW(args.lr, betas=[0.9, 0.95], bias_correction=True, weight_decay=0.01),
            optim.AdamW(args.lr, betas=[0.9, 0.95], bias_correction=True, weight_decay=0.0),
        ], filters=[lambda path, param: param.ndim >= 2])
        self.optimizer.init(self.model.parameters())
        mx.eval(self.optimizer.state)

        def objective(model, x, y, mask, count):
            losses = nn.losses.cross_entropy(model(x).astype(mx.float32), y, reduction='none')
            return (losses * mask).sum() / count
        value_grad = nn.value_and_grad(self.model, objective)

        def update(batches, count):
            total, gradients = value_grad(self.model, *batches[0], count)
            for batch in batches[1:]:
                loss, increment = value_grad(self.model, *batch, count)
                total = total + loss
                gradients = tree_map(lambda a, b: a + b, gradients, increment)
            gradients, _ = optim.clip_grad_norm(gradients, 1.0)
            self.optimizer.update(self.model, gradients)
            return total

        state = [self.model.state, self.optimizer.state]
        self.update = update if args.eager else mx.compile(update, inputs=state, outputs=state)

    def step(self, batches, lr):
        mx = self.mx
        self.optimizer.learning_rate = mx.array(lr)
        count = mx.array(sum(float(mask.sum()) for _, _, mask in batches))
        loss = self.update(tuple(tuple(mx.array(x) for x in batch) for batch in batches), count)
        mx.eval(loss, self.model.state, self.optimizer.state)
        return float(loss)

    def evaluate(self, batch):
        x, y, mask = (self.mx.array(item) for item in batch)
        losses = self.nn.losses.cross_entropy(self.model(x).astype(self.mx.float32), y, reduction='none')
        return float((losses * mask).sum())

    def save(self, path, metadata):
        from mlx.utils import tree_flatten

        trees = dict(model=self.model.parameters(), optimizer=self.optimizer.state)
        arrays = dict(tree_flatten(trees))
        self.mx.save_safetensors(path, arrays, metadata={'run': json.dumps(metadata)})

    def load(self, path, weights_only=False):
        from mlx.utils import tree_unflatten

        if path.suffix == '.pt':
            if not weights_only:
                raise ValueError('Cross-backend loading supports weights-only initialization')
            import torch

            payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            self.model.load_weights([(key, self.mx.array(value.float().numpy()))
                                     for key, value in payload['model'].items()], strict=True)
            self.mx.eval(self.model.parameters())
            return payload['metadata']
        arrays, tags = self.mx.load(path, return_metadata=True)
        trees = tree_unflatten(list(arrays.items()))
        self.model.load_weights([(key.removeprefix('model.'), value)
                                 for key, value in arrays.items() if key.startswith('model.')], strict=True)
        if not weights_only:
            # Preserve the dictionaries captured by mx.compile.
            for optimizer, state in zip(self.optimizer.optimizers, trees['optimizer']['states'], strict=True):
                optimizer.state.clear()
                optimizer.state.update(state)
        self.mx.eval(self.model.state, self.optimizer.state)
        return json.loads(tags['run'])

    def memory(self):
        return self.mx.get_peak_memory()


@contextmanager
def run_lock(root):
    lock = root / 'run.lock'
    with lock.open('x') as stream:
        stream.write(f'{os.getpid()}\n')
    try:
        yield
    finally:
        lock.unlink()


def checkpoint(engine, path, state, count):
    required = 12 * count + 512 * 2**20
    if shutil.disk_usage(path.parent).free < required:
        raise OSError(f'Need {required / 2**30:.2f} GiB free for an atomic checkpoint')
    pending = path.with_name(f'pending{path.suffix}')
    try:
        engine.save(pending, state)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def schedule(step, total, lr):
    warmup = max(1, min(100, total // 20))
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def benchmark(engine, args, config):
    rng = np.random.default_rng(args.seed)
    tokens = rng.integers(config['vocab_size'], size=(args.batch, args.context + 1), dtype=np.int32)
    weights = np.ones((args.batch, args.context), dtype=np.float32)
    batches = [(tokens[:, :-1].copy(), tokens[:, 1:].copy(), weights)] * args.accumulate
    for _ in range(args.warmup):
        engine.step(batches, args.lr)
    start = time.perf_counter()
    for _ in range(args.steps):
        loss = engine.step(batches, args.lr)
        if not math.isfinite(loss):
            raise RuntimeError('Non-finite benchmark loss')
    elapsed = time.perf_counter() - start
    count = args.steps * args.batch * args.context * args.accumulate
    print(json.dumps(dict(backend=args.backend, device=args.device, precision=args.precision,
                          compiled=args.backend == 'mlx' and not args.eager,
                          batch=args.batch, context=args.context, accumulate=args.accumulate,
                          warmup=args.warmup, steps=args.steps, tokens_per_second=round(count / elapsed, 1),
                          memory_bytes=engine.memory(), loss=loss)))


def validate(engine, dataset, mode, batch_size, limit):
    total, count = 0., 0.
    rows = np.linspace(0, len(dataset.rows) - 1, min(len(dataset.rows), limit), dtype=int)
    for start in range(0, len(rows), batch_size):
        indices = rows[start:start + batch_size]
        x, y, weights, _ = dataset.batch(indices, mode)
        total += engine.evaluate((x, y, weights))
        count += float(weights.sum())
    if not count or not math.isfinite(total):
        raise ValueError('Validation has no targets or a non-finite loss')
    return total / count


def train(engine, args, config):
    manifest_bytes = (args.data / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    tokenizer_hash = hashlib.sha256((args.data / 'tokenizer.json').read_bytes()).hexdigest()
    if (manifest.get('format') not in (2, 3) or manifest['vocab_size'] != config['vocab_size']
            or manifest['context'] > config['train_context']
            or tokenizer_hash != manifest['tokenizer_sha256']):
        raise ValueError('Dataset, tokenizer and model configuration do not match')
    train_data, validation = [Sequences(args.data, split, manifest['context'], config['vocab_size'], args.loss,
                                        args.dialogue_share if split == 'train' else None)
                              for split in ('train', 'validation')]
    if args.tokens is not None:
        args.steps, planned = train_data.steps_for_tokens(args.seed, args.batch * args.accumulate, args.tokens)
        print(json.dumps(dict(planned_steps=args.steps, planned_raw_targets=planned)), flush=True)
    settings = {key: getattr(args, key) for key in
                ('backend', 'device', 'precision', 'batch', 'accumulate', 'steps', 'lr', 'seed', 'eager',
                 'dialogue_share', 'tokens')}
    settings.update(config=config, loss=args.loss, tokenizer_sha256=tokenizer_hash,
                    manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
    rng = np.random.default_rng(args.seed)
    begin, processed, supervised = 0, 0, 0
    path = args.run_dir / ('last.safetensors' if args.backend == 'mlx' else 'last.pt')
    if args.resume:
        saved = engine.load(path)
        if saved['settings'] != settings:
            raise ValueError('Resume requires the original settings, tokenizer and data')
        begin, processed, supervised = saved['step'], saved['processed'], saved['supervised']
        rng.bit_generator.state = saved['rng']
    elif path.exists():
        raise ValueError('Checkpoint exists; use --resume or a new run directory')
    elif args.init:
        saved = engine.load(args.init, weights_only=True)
        if saved['settings']['config'] != config or saved['settings']['tokenizer_sha256'] != tokenizer_hash:
            raise ValueError('Initialization requires the original model and tokenizer')
    stop = False
    def interrupt(*_):
        nonlocal stop
        stop = True
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupt)

    end = min(args.steps, begin + args.stop_after) if args.stop_after else args.steps
    with (args.run_dir / 'metrics.jsonl').open('a') as metrics:
        for step in range(begin, end):
            start = time.perf_counter()
            indices = train_data.sample(rng, args.batch * args.accumulate)
            batch = train_data.batch(indices, args.loss)
            x, y, mask, count = batch
            active = int(mask.sum())
            if not active:
                raise ValueError('Batch contains no supervised targets')
            groups = [(x[i:i + args.batch], y[i:i + args.batch], mask[i:i + args.batch])
                      for i in range(0, len(x), args.batch)]
            lr = schedule(step, args.steps, args.lr)
            loss = engine.step(groups, lr)
            if not math.isfinite(loss):
                raise RuntimeError('Non-finite training loss; previous checkpoint retained')
            processed += count
            supervised += active
            row = dict(step=step + 1, loss=loss, lr=lr, processed=processed, supervised=supervised,
                       tokens_per_second=round(count / (time.perf_counter() - start), 1))
            save = stop or step + 1 == end or (step + 1) % args.checkpoint_every == 0
            if save:
                row['validation_loss'] = validate(engine, validation, args.loss, args.batch,
                                                  args.validation_rows)
                if not args.no_checkpoint:
                    checkpoint(engine, path, dict(settings=settings, step=step + 1, processed=processed,
                                                  supervised=supervised, rng=rng.bit_generator.state),
                               config['expected_parameters'])
            print(json.dumps(row), flush=True)
            metrics.write(json.dumps(row) + '\n')
            metrics.flush()
            if stop:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/picoboi-100m.json')
    parser.add_argument('--backend', choices=['torch', 'mlx'], default='torch')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--eager', action='store_true', help='Disable MLX compile')
    parser.add_argument('--threads', type=int, default=4)
    duration = parser.add_mutually_exclusive_group(required=True)
    duration.add_argument('--steps', type=int, help='Total optimizer updates')
    duration.add_argument('--tokens', type=int, help='Plan updates from a raw target-token budget')
    parser.add_argument('--stop-after', type=int, help='Maximum updates in this invocation')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=20261108)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--context', type=int, default=128, help='Synthetic benchmark sequence length')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--loss', choices=['full', 'assistant'], default='full')
    parser.add_argument('--dialogue-share', type=float, help='Approximate dialogue share of raw target tokens')
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--validation-rows', type=int, default=32)
    parser.add_argument('--no-checkpoint', action='store_true', help='Bounded smoke run, at most 20 updates')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--init', type=Path, help='Load weights and start a new optimizer, also across backends')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    parameters = inspect(config)['unique_parameters']
    if min(args.batch, args.accumulate, args.steps or args.tokens or 0, args.threads, args.checkpoint_every,
           args.validation_rows) < 1 or args.lr <= 0:
        parser.error('Training settings must be positive')
    if args.stop_after is not None and args.stop_after < 1 or args.warmup < 0:
        parser.error('Invalid stop or warmup setting')
    if args.resume and args.init or not args.benchmark and (not args.data or not args.run_dir):
        parser.error('Provide --data and --run-dir for training; --init and --resume are exclusive')
    if args.no_checkpoint and (args.resume or args.steps is None or args.steps > 20):
        parser.error('A smoke run cannot resume or exceed 20 updates')
    if args.benchmark and args.steps is None:
        parser.error('Synthetic benchmarks require --steps')
    if args.dialogue_share is not None and (args.loss != 'full' or not 0 < args.dialogue_share < 1):
        parser.error('--dialogue-share requires full loss and a value strictly between zero and one')
    if args.backend == 'mlx':
        if args.device:
            parser.error('MLX uses the Apple GPU without a --device argument')
        args.device = 'metal'
    else:
        import torch
        args.device = args.device or ('cuda' if torch.cuda.is_available() else
                                      'mps' if torch.backends.mps.is_available() else 'cpu')
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            parser.error('CUDA was requested but no CUDA device is available')
        capacity = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    else:
        try:
            capacity = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        except (AttributeError, OSError, ValueError):
            capacity = None
    minimum = 16 * parameters  # FP32 weights, gradients, and two AdamW moment arrays.
    if capacity and 0 < capacity < minimum:
        parser.error(f'{args.device} has {capacity / 2**30:.2f} GiB total memory. '
                     f'This trainer needs at least {minimum / 2**30:.2f} GiB before activations.')
    if not (args.benchmark or args.no_checkpoint):
        args.run_dir.mkdir(parents=True, exist_ok=True)
        required = 12 * parameters + 512 * 2**20
        if shutil.disk_usage(args.run_dir).free < required:
            parser.error(f'Need {required / 2**30:.2f} GiB free for an atomic checkpoint in the run directory')
    if args.backend == 'mlx':
        engine = MLXTrainer(config, args)
    else:
        engine = TorchTrainer(config, args)
    if args.benchmark:
        if not 1 <= args.context <= config['train_context']:
            parser.error('Benchmark context outside model limit')
        benchmark(engine, args, config)
    else:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        with run_lock(args.run_dir):
            train(engine, args, config)


if __name__ == '__main__':
    main()
