# Run the Picoboi 100M pilot locally

This is the 99,123,712-parameter decoder from the [100M design](../notes.md). The same trainer supports the [10M experiment](PICOBOI_10M.md). Run these commands from the repository root.

```sh
uv pip install --python .venv/bin/python -r training/requirements-decoder.txt
.venv/bin/python training/inspect_architecture.py training/configs/picoboi-100m.json
.venv/bin/python training/prepare_decoder.py \
  --source-manifest training/picoboi_100m_sources.json \
  --cities training/picoboi_100m_cities.json --calculator \
  --context 512 --vocab-size 8192 --output training/runs/picoboi-100m-ready
```

Skip preparation if the output already exists. The current local pilot includes 813,462 training and 46,517 validation targets. It contains the existing dialogues, small pinned samples from TinyStories and Smol-SmolTalk, and exact calculator traces. This input is intended for checking the training pipeline. Use an expanded, audited source inventory before the planned large run.

```sh
.venv/bin/python training/train_decoder.py \
  --config training/configs/picoboi-100m.json \
  --backend mlx --precision bf16 --batch 1 --steps 2 --validation-rows 2 \
  --data training/runs/picoboi-100m-ready \
  --run-dir training/runs/picoboi-100m-smoke --no-checkpoint
```

Both full loss and `--loss assistant` completed this smoke check on the M2. For a checkpointed pilot, remove `--no-checkpoint`, use a fresh run directory, and choose a step or token budget. Resume with the same settings and `--resume`. A 100M checkpoint is about 1.11 GiB. The trainer requires about 1.61 GiB free before every save, so reserve at least 2.73 GiB before starting to leave room for the old and pending checkpoint. This Mac had less than 1 GiB free at the end of the verification. Move the run directory to a volume with enough space or free storage before a checkpointed 100M run. Exact checkpoint/resume was verified at 10M.

In matched synthetic runs at batch two and context 256, compiled MLX BF16 reached 1,672 tokens/s and about 2.27 GiB peak MLX memory. Compiled MLX FP32 reached 1,535 tokens/s and PyTorch MPS FP32 reached 638 tokens/s. At the measured BF16 rate, 10 billion tokens would take about 69 days of uninterrupted computation before validation and checkpoints. The packed format uses at least 30 GB for 10 billion distinct packed tokens before indexes or temporary files. A small local pilot fits the machine much better than the full 10B schedule.
