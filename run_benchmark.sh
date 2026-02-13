#!/usr/bin/env bash
#set -euo pipefail

source ~/.bashrc
export TMPDIR=/tmp/$USER
mkdir -p $TMPDIR

module load frameworks

#source /flare/datascience/vsastry/projects/ptcho_vit/venvs/pytcho_vit/bin/activate
NHOSTS=$(wc -l < "${PBS_NODEFILE}")
NGPU_PER_HOST=12
NGPUS="$((${NHOSTS}*${NGPU_PER_HOST}))"

ROOT_DIR="$(pwd)"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/configs/llama3_debug_tp_pp_cp.toml}"
TP="${TP:-2}"
PP="${PP:-1}"
CP="${CP:-1}"
DP="${DP:-1}"
cd "${ROOT_DIR}"

#mpiexec -n $NGPUS -ppn $NGPU_PER_HOST python train.py \
mpiexec -n $NGPUS -ppn $NGPU_PER_HOST python -m train \
  --job.config_file "${CONFIG_FILE}" \
  --parallelism.tensor_parallel_degree="${TP}" \
  --parallelism.pipeline_parallel_degree="${PP}" \
  --parallelism.context_parallel_degree="${CP}" \
  --parallelism.data_parallel_replicate_degree="${DP}" \
  --training.local_batch_size="${LOCAL_BATCH_SIZE:-4}" \
  --training.seq_len="${SEQ_LEN:-2048}" \
  --training.steps="${STEPS:-10}" \
  --job.dump_folder="${OUTPUT_DIR:-./outputs/benchmarks/torchtitan_transformer}" \
  "$@"
