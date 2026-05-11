#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Launch the PD-disaggregated baseline: NUM_PREFILL prefill instances and
# NUM_DECODE decode instances, all wired together via NixlConnector and the
# stock toy_proxy_server.py from tests/v1/kv_connector/nixl_integration/.
#
# Usage:
#   NUM_PREFILL=1 NUM_DECODE=1 ./launch_pd_disagg.sh   # 1P1D
#   NUM_PREFILL=3 NUM_DECODE=1 ./launch_pd_disagg.sh   # 3P1D
#
# Override any of the variables in common.sh by exporting them before running.
#
# After all servers and the proxy are up the script blocks; press Ctrl-C to
# tear everything down (a trap in common.sh handles cleanup).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

NUM_PREFILL=${NUM_PREFILL:-1}
NUM_DECODE=${NUM_DECODE:-1}
PREFILL_BASE_PORT=${PREFILL_BASE_PORT:-8100}
DECODE_BASE_PORT=${DECODE_BASE_PORT:-8200}
PREFILL_SIDE_CHANNEL_BASE=${PREFILL_SIDE_CHANNEL_BASE:-5559}
DECODE_SIDE_CHANNEL_BASE=${DECODE_SIDE_CHANNEL_BASE:-5659}

NUM_GPUS=$(get_num_gpus)
TOTAL=$((NUM_PREFILL + NUM_DECODE))
if (( TOTAL > NUM_GPUS )); then
  echo "WARNING: requesting ${TOTAL} instances but only ${NUM_GPUS} GPU(s) detected." >&2
fi

KV_CONFIG_P=$(_kv_config_pd_disagg kv_producer)
KV_CONFIG_D=$(_kv_config_pd_disagg kv_consumer)

PREFILL_HOSTS=()
PREFILL_PORTS=()
DECODE_HOSTS=()
DECODE_PORTS=()

start_instance() {
  local role=$1     # "prefill" or "decode"
  local idx=$2
  local gpu=$3
  local port=$4
  local side_port=$5
  local kv_config=$6

  local log="${LOG_DIR}/${role}_${idx}.log"
  echo "Starting ${role} #${idx} on GPU=${gpu} port=${port} side_channel=${side_port} -> ${log}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  VLLM_NIXL_SIDE_CHANNEL_PORT="${side_port}" \
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
    --kv-transfer-config "${kv_config}" \
    > "${log}" 2>&1 &

  register_pid $!
}

# ---- Spawn prefill instances ------------------------------------------------
for ((i = 0; i < NUM_PREFILL; i++)); do
  port=$((PREFILL_BASE_PORT + i))
  side_port=$((PREFILL_SIDE_CHANNEL_BASE + i))
  gpu=$((i % NUM_GPUS))
  start_instance prefill "${i}" "${gpu}" "${port}" "${side_port}" "${KV_CONFIG_P}"
  PREFILL_HOSTS+=("127.0.0.1")
  PREFILL_PORTS+=("${port}")
done

# ---- Spawn decode instances -------------------------------------------------
for ((i = 0; i < NUM_DECODE; i++)); do
  port=$((DECODE_BASE_PORT + i))
  side_port=$((DECODE_SIDE_CHANNEL_BASE + i))
  gpu=$(((NUM_PREFILL + i) % NUM_GPUS))
  start_instance decode "${i}" "${gpu}" "${port}" "${side_port}" "${KV_CONFIG_D}"
  DECODE_HOSTS+=("127.0.0.1")
  DECODE_PORTS+=("${port}")
done

# ---- Wait for everyone to be ready -----------------------------------------
for port in "${PREFILL_PORTS[@]}" "${DECODE_PORTS[@]}"; do
  wait_for_server "${port}"
done

# ---- Spawn the PD-disagg proxy ---------------------------------------------
PROXY_SCRIPT="${REPO_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
if [[ ! -f "${PROXY_SCRIPT}" ]]; then
  echo "ERROR: PD-disagg proxy not found at ${PROXY_SCRIPT}" >&2
  exit 1
fi
PROXY_LOG="${LOG_DIR}/proxy.log"
echo "Starting PD-disagg proxy on port ${PROXY_PORT} -> ${PROXY_LOG}"
python3 "${PROXY_SCRIPT}" \
  --host 0.0.0.0 \
  --port "${PROXY_PORT}" \
  --prefiller-hosts "${PREFILL_HOSTS[@]}" \
  --prefiller-ports "${PREFILL_PORTS[@]}" \
  --decoder-hosts "${DECODE_HOSTS[@]}" \
  --decoder-ports "${DECODE_PORTS[@]}" \
  > "${PROXY_LOG}" 2>&1 &
register_pid $!

sleep 2
echo ""
echo "============================================================"
echo "PD-disaggregated cluster ready (${NUM_PREFILL}P${NUM_DECODE}D)"
echo "  proxy:    http://127.0.0.1:${PROXY_PORT}"
echo "  prefill:  ${PREFILL_PORTS[*]}"
echo "  decode:   ${DECODE_PORTS[*]}"
echo "  logs:     ${LOG_DIR}"
echo "Press Ctrl-C to stop all processes."
echo "============================================================"

# Block until interrupted; the EXIT trap in common.sh tears down everything.
wait
