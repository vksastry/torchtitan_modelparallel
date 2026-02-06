#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/benchmarks/torchtitan_transformer/configs/llama3_debug_tp_pp_cp.toml}"
NGPU="${NGPU:-8}"
TP="${TP:-2}"
PP="${PP:-2}"
CP="${CP:-2}"
DP="${DP:-1}"

export CONFIG_FILE
export NGPU

"${ROOT_DIR}/torchtitan/run_train.sh" \
  --parallelism.tensor_parallel_degree="${TP}" \
  --parallelism.pipeline_parallel_degree="${PP}" \
  --parallelism.context_parallel_degree="${CP}" \
  --parallelism.data_parallel_replicate_degree="${DP}" \
  --training.local_batch_size="${LOCAL_BATCH_SIZE:-4}" \
  --training.seq_len="${SEQ_LEN:-2048}" \
  --training.steps="${STEPS:-10}" \
  --job.dump_folder="${OUTPUT_DIR:-./outputs/benchmarks/torchtitan_transformer}" \
  "$@"
