#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Launch the *adaptive* homogeneous-PD setup: NUM_NODES vLLM instances running
# kv_role=kv_both, fronted by adaptive_pd_proxy which routes prefill and decode
# stages independently based on per-backend load.
#
# Unlike launch_homogeneous.sh (which uses round_robin_proxy + always-same-node),
# this launcher uses a smarter proxy that may route P to one backend and D to
# another via NIXL.
#
# Backends still use HomogeneousScheduler so that, when the proxy decides to
# short-circuit (P and D on the same node), decode-first scheduling kicks in
# inside that node.
#
# Usage:
#   TENSOR_PARALLEL_SIZE=4 NUM_NODES=2 \
#     PREFILL_STRATEGY=min_pending_prefill DECODE_KV_WEIGHT=10.0 \
#     ./launch_adaptive.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

NUM_NODES=${NUM_NODES:-2}
NODE_BASE_PORT=${NODE_BASE_PORT:-8100}
NODE_SIDE_CHANNEL_BASE=${NODE_SIDE_CHANNEL_BASE:-5559}

# Adaptive-proxy knobs (passed through to the python module via env)
export PREFILL_STRATEGY=${PREFILL_STRATEGY:-min_pending_prefill}
export DECODE_KV_WEIGHT=${DECODE_KV_WEIGHT:-10.0}
export POLL_INTERVAL_MS=${POLL_INTERVAL_MS:-200}
export PROMPT_TOKEN_DIVISOR=${PROMPT_TOKEN_DIVISOR:-3}

NUM_GPUS=$(get_num_gpus)
TOTAL_NEEDED=$(( NUM_NODES * TENSOR_PARALLEL_SIZE ))
if (( TOTAL_NEEDED > NUM_GPUS )); then
  echo "WARNING: requesting ${TOTAL_NEEDED} GPUs (${NUM_NODES} nodes x TP=${TENSOR_PARALLEL_SIZE}) but only ${NUM_GPUS} GPU(s) detected." >&2
fi

KV_EXTRA=$(cat <<EOF
"kv_connector_extra_config":{"homogeneous_alpha":${ALPHA},"homogeneous_budget_fn":"${BUDGET_FN}","homogeneous_prefill_min":${PREFILL_MIN},"homogeneous_prefill_max":${PREFILL_MAX}}
EOF
)
KV_CONFIG=$(cat <<EOF
{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"${KV_BUFFER_DEVICE}",${KV_EXTRA}}
EOF
)

NODE_HOSTS=()
NODE_PORTS=()
SCHED_CLS="homogeneous_pd.scheduler.homogeneous_scheduler.HomogeneousScheduler"

build_gpu_group() {
  local start=$1
  local tp=$2
  local out=""
  for ((k=0; k<tp; k++)); do
    if [[ -z "${out}" ]]; then
      out="$((start+k))"
    else
      out="${out},$((start+k))"
    fi
  done
  echo "${out}"
}

for ((i = 0; i < NUM_NODES; i++)); do
  port=$((NODE_BASE_PORT + i))
  side_port=$((NODE_SIDE_CHANNEL_BASE + i))
  start_gpu=$(( i * TENSOR_PARALLEL_SIZE ))
  gpus=$(build_gpu_group "${start_gpu}" "${TENSOR_PARALLEL_SIZE}")
  log="${LOG_DIR}/adaptive_node_${i}.log"
  echo "Starting adaptive node #${i} on GPUs=${gpus} (TP=${TENSOR_PARALLEL_SIZE}) port=${port} side_channel=${side_port} -> ${log}"

  extra_args=()
  [[ "${TRUST_REMOTE_CODE}" == "1" ]] && extra_args+=(--trust-remote-code)
  [[ "${ENFORCE_EAGER}" == "1" ]] && extra_args+=(--enforce-eager)
  [[ -n "${KV_CACHE_DTYPE:-}" ]] && extra_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
  [[ -n "${COMPILATION_CONFIG:-}" ]] && extra_args+=(--compilation-config "${COMPILATION_CONFIG}")

  CUDA_VISIBLE_DEVICES="${gpus}" \
  VLLM_NIXL_SIDE_CHANNEL_PORT="${side_port}" \
  PYTHONPATH="${PYTHONPATH}" \
  vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port "${port}" \
    --block-size "${BLOCK_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --enable-chunked-prefill \
    --scheduler-cls "${SCHED_CLS}" \
    "${extra_args[@]}" \
    --kv-transfer-config "${KV_CONFIG}" \
    > "${log}" 2>&1 &

  register_pid $!
  NODE_HOSTS+=("127.0.0.1")
  NODE_PORTS+=("${port}")
done

for port in "${NODE_PORTS[@]}"; do
  wait_for_server "${port}"
done

PROXY_LOG="${LOG_DIR}/adaptive_proxy.log"
echo "Starting adaptive proxy on port ${PROXY_PORT} -> ${PROXY_LOG}"
PYTHONPATH="${PYTHONPATH}" python3 -m homogeneous_pd.proxy.adaptive_pd_proxy \
  --host 0.0.0.0 \
  --port "${PROXY_PORT}" \
  --backend-hosts "${NODE_HOSTS[@]}" \
  --backend-ports "${NODE_PORTS[@]}" \
  --prefill-strategy "${PREFILL_STRATEGY}" \
  --decode-kv-weight "${DECODE_KV_WEIGHT}" \
  --poll-interval-ms "${POLL_INTERVAL_MS}" \
  --prompt-token-divisor "${PROMPT_TOKEN_DIVISOR}" \
  > "${PROXY_LOG}" 2>&1 &
register_pid $!

sleep 2
echo ""
echo "============================================================"
echo "Adaptive PD cluster ready (${NUM_NODES} nodes, TP=${TENSOR_PARALLEL_SIZE})"
echo "  proxy:               http://127.0.0.1:${PROXY_PORT}"
echo "  backends:            ${NODE_PORTS[*]}"
echo "  prefill_strategy:    ${PREFILL_STRATEGY}"
echo "  decode_kv_weight:    ${DECODE_KV_WEIGHT}"
echo "  poll_interval_ms:    ${POLL_INTERVAL_MS}"
echo "  scheduler:           ${SCHED_CLS}"
echo "  budget_fn:           ${BUDGET_FN}  alpha=${ALPHA}"
echo "  logs:                ${LOG_DIR}"
echo "  healthcheck:         curl http://127.0.0.1:${PROXY_PORT}/proxy/healthcheck"
echo "Press Ctrl-C to stop all processes."
echo "============================================================"

wait
