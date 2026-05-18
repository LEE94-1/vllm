#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Start ONE backend node based on a topology config file.
#
# Usage:
#   ./launch_node.sh <config-file> <node-index>
#
# Run this on the machine whose IP matches NODE_HOSTS[<node-index>].
# `vllm serve` runs in the foreground; press Ctrl-C to stop it.
#
# Logs are tee'd to ${LOG_DIR}/node_<index>.log so they remain available
# after Ctrl-C.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <config-file> <node-index>" >&2
  exit 2
fi

CONFIG=$1
NODE_IDX=$2

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${CONFIG}"

# ---- Validation -------------------------------------------------------------

: "${TYPE:?TYPE not set in ${CONFIG}}"
: "${MODEL:?MODEL not set in ${CONFIG}}"

for var in ROLES NODE_HOSTS NODE_PORTS NODE_GPUS NODE_SIDE_CHANNEL_PORTS; do
  if ! declare -p "${var}" 2>/dev/null | grep -q 'declare -a'; then
    echo "${var} must be defined as a bash array in ${CONFIG}" >&2
    exit 1
  fi
done

NUM_NODES=${#NODE_HOSTS[@]}
# shellcheck disable=SC2153  # ROLES is set by the sourced config above.
if [[ ${NUM_NODES} -ne ${#NODE_PORTS[@]} ]] \
  || [[ ${NUM_NODES} -ne ${#NODE_GPUS[@]} ]] \
  || [[ ${NUM_NODES} -ne ${#NODE_SIDE_CHANNEL_PORTS[@]} ]] \
  || [[ ${NUM_NODES} -ne ${#ROLES[@]} ]]; then
  echo "All NODE_* arrays and ROLES must have the same length (got" \
       "hosts=${#NODE_HOSTS[@]} ports=${#NODE_PORTS[@]} gpus=${#NODE_GPUS[@]}" \
       "side=${#NODE_SIDE_CHANNEL_PORTS[@]} roles=${#ROLES[@]})" >&2
  exit 1
fi

if (( NODE_IDX < 0 || NODE_IDX >= NUM_NODES )); then
  echo "Node index ${NODE_IDX} out of range [0, ${NUM_NODES})." >&2
  exit 1
fi

HOST=${NODE_HOSTS[${NODE_IDX}]}
PORT=${NODE_PORTS[${NODE_IDX}]}
GPU=${NODE_GPUS[${NODE_IDX}]}
ROLE=${ROLES[${NODE_IDX}]}
SIDE_CHANNEL=${NODE_SIDE_CHANNEL_PORTS[${NODE_IDX}]}

# ---- Build kv-transfer-config + scheduler args by TYPE ----------------------

EXTRA_ARGS=()

case "${TYPE}" in
  pd_disagg)
    if [[ "${ROLE}" != "kv_producer" && "${ROLE}" != "kv_consumer" ]]; then
      echo "TYPE=pd_disagg requires ROLES[i] in {kv_producer,kv_consumer}, got ${ROLE}" >&2
      exit 1
    fi
    KV_CONFIG=$(_kv_config_pd_disagg "${ROLE}")
    ;;
  homogeneous)
    if [[ "${ROLE}" != "kv_both" ]]; then
      echo "TYPE=homogeneous requires ROLES[i]=kv_both, got ${ROLE}" >&2
      exit 1
    fi
    ALPHA=${ALPHA:-16}
    BUDGET_FN=${BUDGET_FN:-kv_decode_page}
    PREFILL_MIN=${PREFILL_MIN:-0}
    PREFILL_MAX=${PREFILL_MAX:-0}
    KV_EXTRA="\"kv_connector_extra_config\":{\"homogeneous_alpha\":${ALPHA},\"homogeneous_budget_fn\":\"${BUDGET_FN}\",\"homogeneous_prefill_min\":${PREFILL_MIN},\"homogeneous_prefill_max\":${PREFILL_MAX}}"
    KV_CONFIG="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"${KV_BUFFER_DEVICE}\",${KV_EXTRA}}"
    EXTRA_ARGS+=(
      --enable-chunked-prefill
      --scheduler-cls
      "homogeneous_pd.scheduler.homogeneous_scheduler.HomogeneousScheduler"
    )
    ;;
  *)
    echo "Unknown TYPE: ${TYPE} (expected pd_disagg or homogeneous)" >&2
    exit 1
    ;;
esac

# ---- Optional flags from config / defaults ----------------------------------

# Trust remote code (HF models with custom Python).
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  EXTRA_ARGS+=(--trust-remote-code)
fi

# Enforce eager (skip CUDA graphs). Default off so users get CUDA graphs.
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi

# Attention backend: set via env (vLLM reads VLLM_ATTENTION_BACKEND) so we
# do not hard-fail if the running vLLM build does not accept the CLI flag.
if [[ -n "${ATTENTION_BACKEND}" ]]; then
  export VLLM_ATTENTION_BACKEND="${ATTENTION_BACKEND}"
fi

LOG="${LOG_DIR}/node_${NODE_IDX}.log"

echo "============================================================"
echo "Launching node #${NODE_IDX} from ${CONFIG}"
echo "  TYPE     : ${TYPE}"
echo "  role     : ${ROLE}"
echo "  declared : ${HOST}:${PORT}  (this script binds 0.0.0.0:${PORT})"
echo "  CUDA dev : ${GPU}     (tensor-parallel-size=${TENSOR_PARALLEL_SIZE})"
echo "  NIXL port: ${SIDE_CHANNEL}"
echo "  model    : ${MODEL}"
echo "  attn     : ${ATTENTION_BACKEND:-<vllm default>}"
echo "  trust RC : ${TRUST_REMOTE_CODE}    enforce-eager: ${ENFORCE_EAGER}"
if [[ -n "${UCX_NET_DEVICES:-}" ]]; then
  echo "  UCX devs : ${UCX_NET_DEVICES}    UCX_TLS=${UCX_TLS:-<unset>}"
fi
if [[ "${TYPE}" == "homogeneous" ]]; then
  echo "  scheduler: HomogeneousScheduler  alpha=${ALPHA} budget_fn=${BUDGET_FN} min=${PREFILL_MIN} max=${PREFILL_MAX}"
fi
echo "  log      : ${LOG}"
echo "============================================================"

# Run vllm serve in the foreground, mirroring stdout/stderr to a log file.
# Ctrl-C kills the pipeline; the trap installed by common.sh runs as well
# but is a no-op since we did not register_pid anything.
#
# ``${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}`` is the bash-safe way of expanding a
# possibly-empty array under ``set -u``.
CUDA_VISIBLE_DEVICES="${GPU}" \
VLLM_NIXL_SIDE_CHANNEL_PORT="${SIDE_CHANNEL}" \
PYTHONPATH="${PYTHONPATH}" \
vllm serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --block-size "${BLOCK_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  --kv-transfer-config "${KV_CONFIG}" \
  2>&1 | tee "${LOG}"
