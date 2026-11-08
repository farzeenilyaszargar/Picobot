# Picoboi 10B parameter target

The [10B configuration](configs/picoboi-10b.json) defines a new 9,957,740,544-parameter decoder with 56 blocks, width 4,096, 32 query heads, eight key/value heads, and a tied 8,192-entry tokenizer. It has a 2,048-token context. This configuration is separate from the [100M model with a 10B-token training budget](PICOBOI_100M.md). The existing 10M checkpoints have different tensor shapes and tokenizer IDs, so a normal resume cannot initialize the new decoder.

## Capacity with the current trainer

| Allocation | Lower bound |
| --- | ---: |
| FP32 weights | 37.10 GiB |
| FP32 weights, gradients and two AdamW moment arrays | 148.38 GiB before activations and temporary buffers |
| Resumable checkpoint | 111.29 GiB before format overhead |
| Disk for the old and pending checkpoints plus the current save guard | 223.08 GiB before datasets |

The current single-device trainer keeps FP32 master weights even with BF16 compute. The memory check is a lower bound. A machine that passes still needs enough headroom for model initialization, activations, and allocator workspace. On the 8 GiB Mac, this configuration is rejected before model allocation.

```sh
.venv/bin/python training/inspect_architecture.py training/configs/picoboi-10b.json
.venv/bin/python training/train_decoder.py \
  --config training/configs/picoboi-10b.json \
  --backend torch --device cuda --precision bf16 \
  --benchmark --batch 1 --context 128 --warmup 2 --steps 5
```

Run the benchmark on an authorized machine with enough memory. Verify useful tokens per second, then run a checkpoint and resume check with a representative prepared dataset before a sustained run. The architecture, corpus, and training schedule for the smaller experiment establish starting points. A full run still needs a dedicated token budget and a measured, deduplicated data inventory. Training has yet to start.
