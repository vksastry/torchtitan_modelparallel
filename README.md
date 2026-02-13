# Torchtitan transformer benchmark (TP/PP/CP)

This benchmark runs a local copy of the Llama3 debug transformer with synthetic tokens, while importing torchtitan components from the active venv. It avoids torchtitan's `run_train.sh` and lets you control the training loop via `benchmarks/torchtitan_transformer/train.py`.

Quick start (local tensor mode):

```bash
bash benchmarks/torchtitan_transformer/run_benchmark.sh
```

Override degrees and training shape:

```bash
TP=2 PP=2 CP=2 LOCAL_BATCH_SIZE=4 SEQ_LEN=2048 STEPS=20 \
  OUTPUT_DIR=./outputs/benchmarks/torchtitan_transformer \
  bash benchmarks/torchtitan_transformer/run_benchmark.sh
```

Notes:

- Default `COMM_MODE` is `local_tensor`. Set `COMM_MODE=fake_backend` for a dry run.
- For real multi-GPU, launch with `torchrun` and set `COMM_MODE=` to empty.
- Example: `torchrun --nproc_per_node=8 -m benchmarks.torchtitan_transformer.train --job.config_file benchmarks/torchtitan_transformer/configs/llama3_debug_tp_pp_cp.toml`
- Tokenizer assets default to `./torchtitan/tests/assets/tokenizer`.
- The loader enforces venv torchtitan imports; install torchtitan in your active venv.
- Config lives in `benchmarks/torchtitan_transformer/configs/llama3_debug_tp_pp_cp.toml`.
