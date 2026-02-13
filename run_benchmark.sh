#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/configs/llama3_debug_tp_pp_cp.toml}"
TP="${TP:-2}"
PP="${PP:-2}"
CP="${CP:-2}"
DP="${DP:-1}"

export LOCAL_RANK="${LOCAL_RANK:-0}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"

cd "${ROOT_DIR}"

COMM_MODE="${COMM_MODE:-local_tensor}"
COMM_ARG=()
if [ -n "${COMM_MODE}" ]; then
  COMM_ARG=("--comm.mode=${COMM_MODE}")
fi

python -m benchmarks.torchtitan_transformer.train \
  --job.config_file "${CONFIG_FILE}" \
  "${COMM_ARG[@]}" \
  --parallelism.tensor_parallel_degree="${TP}" \
  --parallelism.pipeline_parallel_degree="${PP}" \
  --parallelism.context_parallel_degree="${CP}" \
  --parallelism.data_parallel_replicate_degree="${DP}" \
  --training.local_batch_size="${LOCAL_BATCH_SIZE:-4}" \
  --training.seq_len="${SEQ_LEN:-2048}" \
  --training.steps="${STEPS:-10}" \
  --job.dump_folder="${OUTPUT_DIR:-./outputs/benchmarks/torchtitan_transformer}" \
  "$@"
