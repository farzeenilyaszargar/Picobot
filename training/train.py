#!/usr/bin/env python3
"""Local training with complete dialogues, assistant-only targets, and resume."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import signal
import time

import numpy as np
import torch
from model import Picoboi, masked_loss

ROOT = Path(__file__).resolve().parent


class Dialogues:
    def __init__(self, root, split, context):
        self.tokens = np.memmap(root / f'{split}.bin', dtype='<u2', mode='r')
        self.masks = np.memmap(root / f'{split}.assistant_mask.bin', dtype='u1', mode='r')
        with (root / f'{split}.offsets.jsonl').open() as source:
            entries = [json.loads(line) for line in source]
        self.offsets = np.array([r['offset'] for r in entries], dtype=np.int64)
        self.lengths = np.array([r['length'] for r in entries], dtype=np.int64)
        if self.lengths.max() > context:
            raise ValueError('A dialogue exceeds context; do not silently truncate it')
        if len(self.tokens) != len(self.masks) or self.offsets[-1] + self.lengths[-1] != len(self.tokens):
            raise ValueError('Packed data dimensions disagree')

    def __len__(self):
        return len(self.offsets)

    def batch(self, indices, device):
        lengths, offsets = self.lengths[indices], self.offsets[indices]
        pos = np.arange(int(lengths.max()))[None, :]
        valid = pos < lengths[:, None]
        addresses = np.minimum(offsets[:, None]+pos, len(self.tokens)-1)
        tokens = np.where(valid, self.tokens[addresses], 0).astype(np.int64)
        masks = np.where(valid, self.masks[addresses], 0).astype(np.float32)
        # Shift BOTH labels and the assistant-target mask by one token.
        x = torch.from_numpy(tokens[:, :-1].copy()).to(device)
        y = torch.from_numpy(tokens[:, 1:].copy()).to(device)
        weights = torch.from_numpy(masks[:, 1:].copy()).to(device)
        return x, y, weights, int((lengths-1).sum())


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()


@torch.no_grad()
def evaluate(model, data, batch_size, device, max_batches=0):
    model.eval()
    num, den = 0., 0.
    for j, start in enumerate(range(0, len(data), batch_size)):
        if max_batches and j >= max_batches:
            break
        x, y, weights, _ = data.batch(np.arange(start, min(start+batch_size, len(data))), device)
        _, n, d = masked_loss(model(x), y, weights)
        num += n.item(); den += d.item()
    model.train()
    return num / den


def optimizer_for(model, lr):
    matrices = [p for p in model.parameters() if p.ndim >= 2]
    scales = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW([{'params': matrices, 'weight_decay': 0.01},
                              {'params': scales, 'weight_decay': 0.0}], lr=lr, betas=(0.9, 0.95))


def save(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def benchmark(config, train, validation, args):
    results = []
    devices = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])
    trials = [(device, threads, batch_size) for device in devices
              for threads in ((1, 2, 4) if device == 'cpu' else (args.threads,))
              for batch_size in (32, 64, 128)]
    for device, threads, batch_size in trials:
        torch.set_num_threads(threads)
        torch.manual_seed(args.seed)
        model = Picoboi(config).to(device)
        optimizer = optimizer_for(model, 0.003)
        order = np.random.default_rng(args.seed).permutation(len(train))
        for i in range(25):
            if i == 5:
                sync(device); start = time.perf_counter(); tokens = 0
            indices = order[i*batch_size:(i+1)*batch_size]
            x, y, weights, count = train.batch(indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = masked_loss(model(x), y, weights)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            if i >= 5: tokens += count
        sync(device)
        elapsed = time.perf_counter()-start
        row = dict(device=device, threads=threads, batch_size=batch_size, seconds=elapsed,
                   tokens_per_second=round(tokens/elapsed), loss=loss.item())
        results.append(row); print(json.dumps(row), flush=True)
        del optimizer, model
        if device == 'mps': torch.mps.empty_cache()
    best = max(results, key=lambda r: r['tokens_per_second'])
    torch.set_num_threads(best['threads'])
    # Comparable short learning-rate pilots from identical initialization and minibatches.
    pilots = []
    for lr in (0.001, 0.003):
        torch.manual_seed(args.seed)
        model = Picoboi(config).to(best['device'])
        optimizer = optimizer_for(model, lr)
        order = np.random.default_rng(args.seed).permutation(len(train))
        for i in range(80):
            indices = order[i*best['batch_size']:(i+1)*best['batch_size']]
            x, y, weights, _ = train.batch(indices, best['device'])
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = masked_loss(model(x), y, weights)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        val = evaluate(model, validation, best['batch_size'], best['device'], 8)
        row = dict(learning_rate=lr, pilot_steps=80, validation_loss=val)
        pilots.append(row); print(json.dumps(row), flush=True)
        del model, optimizer
        if best['device'] == 'mps': torch.mps.empty_cache()
    result = dict(hardware=platform.platform(), torch=torch.__version__, threads=args.threads,
                  throughput_trials=results, learning_rate_trials=pilots,
                  selected={**best, 'learning_rate': min(pilots, key=lambda r:r['validation_loss'])['learning_rate']},
                  limitation='Short pilots select practical defaults, not globally optimal hyperparameters.')
    (args.run_dir / 'benchmark.json').write_text(json.dumps(result, indent=2)+'\n')


def train_model(config, train, validation, args):
    if args.auto:
        choice = json.loads((args.run_dir / 'benchmark.json').read_text())['selected']
        args.device, args.batch_size, args.lr = choice['device'], choice['batch_size'], choice['learning_rate']
        args.threads = choice['threads']
    torch.set_num_threads(args.threads)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable')
    torch.manual_seed(args.seed)
    model = Picoboi(config).to(args.device)
    optimizer = optimizer_for(model, args.lr)
    batches = math.ceil(len(train)/args.batch_size)
    total_steps = batches*args.epochs
    warmup = max(1, int(total_steps*0.05))
    step = epoch_start = batch_start = tokens_seen = supervised_seen = 0
    best = float('inf'); stale = 0; history = []
    data_fingerprint = hashlib.sha256((args.data / 'sha256.json').read_bytes()).hexdigest()
    settings = dict(config=config, device=args.device, batch_size=args.batch_size, peak_lr=args.lr,
                    epochs=args.epochs, seed=args.seed, threads=args.threads, warmup_steps=warmup,
                    total_steps=total_steps, data_fingerprint=data_fingerprint,
                    torch=torch.__version__, precision='float32', loss='assistant_only',
                    parameters=sum(p.numel() for p in model.parameters()))
    if args.resume:
        state = torch.load(args.run_dir / 'last.pt', map_location='cpu', weights_only=False)
        for key in ('config', 'batch_size', 'peak_lr', 'epochs', 'seed', 'data_fingerprint'):
            if state['settings'][key] != settings[key]: raise ValueError(f'Resume mismatch: {key}')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor) and key != 'step': values[key] = value.to(args.device)
        step, epoch_start, batch_start = state['step'], state['epoch'], state['next_batch']
        tokens_seen, supervised_seen = state['tokens_seen'], state['supervised_seen']
        best, stale, history = state['best_validation_loss'], state['stale'], state['history']
    elif (args.run_dir / 'last.pt').exists():
        raise ValueError('Checkpoint exists; use --resume or a new run directory')
    (args.run_dir / 'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    stop = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    start = time.perf_counter(); since = tokens_seen; log_num = log_den = 0.
    validation_loss = evaluate(model, validation, args.batch_size, args.device)
    print(json.dumps(dict(event='start', validation_loss=validation_loss, **settings)), flush=True)

    def checkpoint(epoch, next_batch, improved=False):
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), settings=settings,
                       step=step, epoch=epoch, next_batch=next_batch, tokens_seen=tokens_seen,
                       supervised_seen=supervised_seen, best_validation_loss=best,
                       stale=stale, history=history)
        save(args.run_dir / 'last.pt', payload)
        if improved: save(args.run_dir / 'best.pt', payload)

    if not args.resume:
        best = validation_loss; checkpoint(0, 0, True)
    for epoch in range(epoch_start, args.epochs):
        order = np.random.default_rng(args.seed+epoch).permutation(len(train))
        for batch in range(batch_start if epoch == epoch_start else 0, batches):
            indices = order[batch*args.batch_size:(batch+1)*args.batch_size]
            x, y, weights, count = train.batch(indices, args.device)
            lr = args.lr*min((step+1)/warmup, 1.0)
            if step >= warmup:
                lr = args.lr*(0.1+0.9*0.5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup))))
            for group in optimizer.param_groups: group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            loss, numerator, denominator = masked_loss(model(x), y, weights)
            if not torch.isfinite(loss).item(): raise RuntimeError('Non-finite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); step += 1; tokens_seen += count
            supervised_seen += int(denominator.item())
            log_num += numerator.item(); log_den += denominator.item()
            if step % 50 == 0:
                row = dict(event='progress', step=step, total_steps=total_steps, epoch=round(epoch+(batch+1)/batches,3),
                           train_loss=log_num/log_den, lr=lr, gradient_norm=norm.item(),
                           tokens_seen=tokens_seen, supervised_seen=supervised_seen,
                           tokens_per_second=round((tokens_seen-since)/(time.perf_counter()-start)))
                print(json.dumps(row), flush=True)
                with (args.run_dir / 'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                log_num = log_den = 0.
            if step % args.eval_every == 0 or batch == batches-1 or stop[0]:
                validation_loss = evaluate(model, validation, args.batch_size, args.device)
                improved = validation_loss < best
                if improved: best = validation_loss; stale = 0
                else: stale += 1
                row = dict(event='validation', step=step, validation_loss=validation_loss, best_validation_loss=best)
                history.append(row); print(json.dumps(row), flush=True)
                with (args.run_dir / 'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                checkpoint(epoch, batch+1, improved)
                status = dict(status='interrupted' if stop[0] else 'running', total_steps=total_steps,
                              tokens_seen=tokens_seen, supervised_seen=supervised_seen, **row)
                (args.run_dir / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
                if stop[0]: return
                if stale >= args.patience:
                    status['status']='early_stopped'
                    (args.run_dir / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
                    return
        batch_start=0
    status = dict(status='complete', step=step, total_steps=total_steps,
                  tokens_seen=tokens_seen, supervised_seen=supervised_seen,
                  validation_loss=validation_loss, best_validation_loss=best)
    (args.run_dir / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/picoboi-100k-looped.json')
    parser.add_argument('--data', type=Path, default=ROOT/'data/picoboi-800-v1')
    parser.add_argument('--run-dir', type=Path, default=ROOT/'runs/picoboi-100k-local')
    parser.add_argument('--device', choices=['cpu','mps'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.epochs,args.batch_size,args.threads,args.eval_every,args.patience) <= 0 or args.lr <= 0:
        raise ValueError('Training settings must be positive')
    torch.set_num_threads(args.threads)
    config=json.loads(args.config.read_text()); args.run_dir.mkdir(parents=True, exist_ok=True)
    train=Dialogues(args.data,'train',config['train_context'])
    validation=Dialogues(args.data,'validation',config['train_context'])
    benchmark(config,train,validation,args) if args.benchmark else train_model(config,train,validation,args)
