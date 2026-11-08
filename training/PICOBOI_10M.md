# Train Picoboi 10M locally

The [10M design](../docs/picoboi-10m-plan.md) defines a 9,902,336-parameter decoder and the city-slot comparison. Run these commands from the repository root. On the tested Apple M2 Mac with 8 GiB of unified memory, start with compiled MLX, BF16 compute, FP32 weights and optimizer state, and batches of four. Try six if profiling with your prepared data gives better throughput. Allow about 113 MiB per resumable checkpoint plus space for its atomic replacement. The trainer checks for about 625 MiB free at each checkpoint write.

## Install and verify

Reuse the project environment or create it with `uv venv --python 3.13 .venv`.

```sh
uv pip install --python .venv/bin/python -r training/requirements-decoder.txt
.venv/bin/python training/inspect_architecture.py training/configs/picoboi-10m.json
.venv/bin/python -m unittest discover -s training -p 'test*.py' -q
node training/tests/city-slots.mjs
```

## Prepare the data

```sh
.venv/bin/python training/prepare_decoder.py \
  --source-manifest training/picoboi_10m_sources.json \
  --cities training/picoboi_10m_cities.json --paired --city-dialogues \
  --holdout-cities Lisbon Auckland --vocab-size 4096 --context 512 \
  --output training/runs/picoboi-10m-paired
```

The command streams pinned TinyStories and Smol-SmolTalk revisions. It imports the existing local dialogues and adds deterministic city reference exercises. The Smol-SmolTalk filter favors short English user/assistant exchanges and screens for common coding and roleplay patterns. These are heuristic filters. The output retains source and license metadata for review. Both arms share the tokenizer, the underlying example order, and the raw-length sampling buckets. An overlong dialogue is excluded from both arms. Prose can span multiple record-local chunks. The city arm applies only the dictionary transformation, and reserved city IDs are present in both vocabularies. Lisbon and Auckland are excluded from corpus preparation and available in the evaluation fixtures as dictionary additions.

The command writes `raw` and `cities` atomically under the chosen directory. Use another directory for reruns. Inspect `source_counts` and `packed` in each manifest for actual unique target counts. Tokenizer training uses the training partition. The sampler draws examples with replacement, so processed tokens count exposures. The split uses exact content hashes. Keep the explicit city fixture separate from training, and review near-duplicates and filtered samples before publishing training artifacts. If streaming is unreliable, `--download-shards` selects the explicit Parquet shard in each source entry and caches about 480 MB. That option samples a different subset.

The prepared pair on this Mac uses 55 MiB. Each arm has about 8.53M unique training targets, including about 1.05M dialogue targets, and 451K validation targets. The city arm contains 663 training slot occurrences. The short-chat filter retained about 2,000 of 30,000 streamed Smol-SmolTalk rows before length packing, so the planned dialogue mix will repeat examples.

## Run the pilot

Use the same seed, batch size and raw token budget for the two arms. The schedule simulates the matched sampler up front to choose the update count. `--dialogue-share 0.3` targets about 30% dialogue by raw tokens. Actual counts differ slightly across arms because the slots change sequence length.

```sh
for variant in raw cities; do
  .venv/bin/python training/train_decoder.py \
    --config training/configs/picoboi-10m.json \
    --backend mlx --precision bf16 --batch 4 \
    --data "training/runs/picoboi-10m-paired/$variant" \
    --tokens 10000000 --dialogue-share 0.3 --checkpoint-every 500 \
    --run-dir "training/runs/picoboi-10m-$variant"
done
```

For a small verification run, use `--steps 12 --no-checkpoint` instead of `--tokens 10000000` and use a separate run directory. For an interruptible training run, append `--stop-after 100` for the first invocation if desired. Resume with the same configuration, data, token budget, seed and run directory, adding `--resume`. Ctrl+C saves after the current update. For chat post-training, start a new run directory and use `--loss assistant --init path/to/last.safetensors` with the same tokenizer and model config. Omit `--dialogue-share` for that stage. The assistant stage draws from dialogue rows and masks user text, prose and padding.

## Measurements on this Mac

Synthetic training includes forward, backward and AdamW. Short runs varied with background activity. These numbers measure training, and the MLX memory column is the peak reported by its allocator.

| Backend | Batch × context | Training tokens/s | MLX peak |
| --- | ---: | ---: | ---: |
| Compiled MLX BF16 | 4 × 256 | 10,392–13,882 | 500 MiB |
| Compiled MLX BF16 | 6 × 256 | 14,695 | 657 MiB |
| Compiled MLX BF16 | 4 × 512 | 9,252 | 862 MiB |
| Compiled MLX FP32 | 4 × 256 | 9,864–13,655 | 766 MiB |
| Eager MLX BF16 | 4 × 256 | 9,125 | 550 MiB |
| PyTorch MPS FP32 | 4 × 256 | 4,624 | |
| PyTorch MPS BF16 | 4 × 256 | 4,335 | |
| PyTorch CPU FP32, four threads | 4 × 256 | 3,247 | |

Re-run a matched synthetic test with:

```sh
.venv/bin/python training/train_decoder.py \
  --config training/configs/picoboi-10m.json \
  --benchmark --backend mlx --precision bf16 --batch 4 --context 256 \
  --steps 30 --warmup 8
```

The 20-update real-data checks with batch four averaged roughly 5,800 to 8,100 non-padding tokens per second, including startup and final validation. Batch six won the fixed-shape benchmark but was slower in its short real-data check. Treat 20 to 30 minutes per 10M-token arm as an initial estimate. Compilation, checkpointing, validation and other Mac activity can change elapsed time.

Resume was tested on the exact decoder. Its parameters and optimizer arrays matched the uninterrupted run element for element after 12 updates. Assistant-only initialization from a checkpoint was tested separately. The browser city parser has shared fixtures with Python. Integrating a trained 10M checkpoint and its BPE tokenizer into the live chat remains a later step.
