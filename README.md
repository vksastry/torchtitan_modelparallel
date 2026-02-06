# Torchtitan transformer benchmark (TP/PP/CP)

This benchmark runs a torchtitan Llama3 debug transformer with synthetic tokens to exercise tensor, pipeline, and context parallelism. It uses a custom TrainSpec registered by `benchmarks/torchtitan_transformer/synthetic_train_spec.py` and does not depend on `fla`.

Quick start (8 GPUs, TP=2, PP=2, CP=2):

```bash
NGPU=8 TP=2 PP=2 CP=2 bash benchmarks/torchtitan_transformer/run_benchmark.sh
```

Common overrides:

```bash
NGPU=8 TP=2 PP=2 CP=2 LOCAL_BATCH_SIZE=4 SEQ_LEN=2048 STEPS=20 \
  OUTPUT_DIR=./outputs/benchmarks/torchtitan_transformer \
  bash benchmarks/torchtitan_transformer/run_benchmark.sh
```

Notes:

- `NGPU` must equal `TP * PP * CP * DP` (DP defaults to 1).
- Tokenizer assets are read from `./torchtitan/tests/assets/tokenizer`.
- The default config is `benchmarks/torchtitan_transformer/configs/llama3_debug_tp_pp_cp.toml`.
- For dry runs: `COMM_MODE=fake_backend` or `COMM_MODE=local_tensor`.
