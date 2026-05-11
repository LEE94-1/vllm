#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Launch the homogeneous-PD experiment: NUM_NODES vLLM instances, each acting
# as both prefill and decode (kv_role=kv_both) and using the custom
# HomogeneousScheduler so that prefill tokens per step are bounded by
# alpha * decode_tokens.
#
# Usage:
#   NUM_NODES=2 ALPHA=16 ./launch_homogeneous.sh
#   NUM_NODES=4 ALPHA=8  ./launch_homogeneous.sh
#
# Override any of the variables in common.sh by exporting them before running.
#
# After all servers and the proxy are up the script blocks; press Ctrl-C to
# tear everything down.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

NUM_NODES=${NUM_NODES:-2}
NODE_BASE_PORT=${NODE_BASE_PORT:-8100}
NODE_SIDE_CHANNEL_BASE=${NODE_SIDE_CHANNEL_BASE:-5559}

NUM_GPUS=$(get_num_gpus)
if (( NUM_NODES > NUM_GPUS )); then
  echo "WARNING: requesting ${NUM_NODES} nodes but only ${NUM_GPUS} GPU(s) detected." >&2
fi

# ALPHA / BUDGET_FN / PREFILL_MIN/MAX get baked into kv_connector_extra_config
# so HomogeneousScheduler picks them up regardless of how vllm serve forwards
# CLI args. They can also be set via env vars (HOMOGENEOUS_ALPHA etc.) 鈥?the
# scheduler resolves config in priority: extra_config > env > default.
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

for ((i = 0; i < NUM_NODES; i++)); do
  port=$((NODE_BASE_PORT + i))
  side_port=$((NODE_SIDE_CHANNEL_BASE + i))
  gpu=$((i % NUM_GPUS))
  log="${LOG_DIR}/homogeneous_${i}.log"
  echo "Starting homogeneous node #${i} on GPU=${gpu} port=${port} side_channel=${side_port} -> ${log}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
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
    --tensor-parallel-size 1 \
    --enforce-eager \
    --enable-chunked-prefill \
    --scheduler-cls "${SCHED_CLS}" \
    --kv-transfer-config "${KV_CONFIG}" \
    > "${log}" 2>&1 &

  register_pid $!
  NODE_HOSTS+=("127.0.0.1")
  NODE_PORTS+=("${port}")
done

for port in "${NODE_PORTS[@]}"; do
  wait_for_server "${port}"
done

PROXY_LOG="${LOG_DIR}/proxy.log"
echo "Starting round-robin proxy on port ${PROXY_PORT} -> ${PROXY_LOG}"
PYTHONPATH="${PYTHONPATH}" python3 -m homogeneous_pd.proxy.round_robin_proxy \
  --host 0.0.0.0 \
  --port "${PROXY_PORT}" \
  --backend-hosts "${NODE_HOSTS[@]}" \
  --backend-ports "${NODE_PORTS[@]}" \
  > "${PROXY_LOG}" 2>&1 &
register_pid $!

sleep 2
echo ""
echo "============================================================"
echo "Homogeneous PD cluster ready (${NUM_NODES} nodes)"
echo "  proxy:        http://127.0.0.1:${PROXY_PORT}"
echo "  backends:     ${NODE_PORTS[*]}"
echo "  scheduler:    ${SCHED_CLS}"
echo "  budget_fn:    ${BUDGET_FN}  alpha=${ALPHA}"
echo "  prefill cap:  min=${PREFILL_MIN} max=${PREFILL_MAX}  (0 = unbounded)"
echo "  logs:         ${LOG_DIR}"
echo "Press Ctrl-C to stop all processes."
echo "============================================================"

wait
