#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Common helpers shared by all launchers in this directory.
#
# Source this file from another script with:
#   SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
#   source "${SCRIPT_DIR}/common.sh"
#
# After sourcing, the following are exported / available:
#   * REPO_ROOT, EXAMPLES_DIR, HOMOGENEOUS_DIR
#   * PYTHONPATH includes "${EXAMPLES_DIR}" so
#     `homogeneous_pd.scheduler.homogeneous_scheduler.HomogeneousScheduler`
#     can be loaded via `--scheduler-cls`.
#   * MODEL, BLOCK_SIZE, MAX_MODEL_LEN, MAX_NUM_BATCHED_TOKENS, MAX_NUM_SEQS,
#     GPU_MEMORY_UTILIZATION, KV_BUFFER_DEVICE, ALPHA, BUDGET_FN
#     are filled in from environment variables or defaulted.
#   * Functions: get_num_gpus, wait_for_server, register_pid,
#     stop_all_pids, _kv_config_homogeneous, _kv_config_pd_disagg.

set -Eeuo pipefail

# ---- Path setup -------------------------------------------------------------

if [[ -z "${SCRIPT_DIR:-}" ]]; then
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
fi
HOMOGENEOUS_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
EXAMPLES_DIR="$(cd -- "${HOMOGENEOUS_DIR}/.." && pwd -P)"
REPO_ROOT="$(cd -- "${EXAMPLES_DIR}/.." && pwd -P)"
# REPO_ROOT is consumed by launch_pd_disagg.sh / launch_router.sh.
export REPO_ROOT EXAMPLES_DIR HOMOGENEOUS_DIR

export PYTHONPATH="${EXAMPLES_DIR}:${PYTHONPATH:-}"

# ---- Defaults ---------------------------------------------------------------

MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
BLOCK_SIZE=${BLOCK_SIZE:-16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
KV_BUFFER_DEVICE=${KV_BUFFER_DEVICE:-cuda}

# Homogeneous-PD-specific knobs (only used by run_homogeneous_*.sh).
ALPHA=${ALPHA:-16}
BUDGET_FN=${BUDGET_FN:-linear_ratio}
PREFILL_MIN=${PREFILL_MIN:-0}
PREFILL_MAX=${PREFILL_MAX:-0}

PROXY_PORT=${PROXY_PORT:-8000}
LOG_DIR=${LOG_DIR:-/tmp/homogeneous_pd_logs}
mkdir -p "${LOG_DIR}"

# Each launcher invocation gets its own PID file (suffixed with the shell PID)
# so that two concurrent launchers on the same machine do not clobber each
# other. stop_all.sh wipes anything matching pids_*.txt under LOG_DIR.
PID_FILE=${PID_FILE:-${LOG_DIR}/pids_$$.txt}
: > "${PID_FILE}"

# ---- Helpers ----------------------------------------------------------------

SMI_BIN=$(command -v nvidia-smi || command -v rocm-smi || true)

get_num_gpus() {
  if [[ -z "${SMI_BIN}" ]]; then
    echo 1
    return
  fi
  if [[ "${SMI_BIN}" == *nvidia* ]]; then
    "${SMI_BIN}" --query-gpu=name --format=csv,noheader | wc -l
  else
    "${SMI_BIN}" -l | grep -c GPU || echo 1
  fi
}

wait_for_server() {
  local port=$1
  local timeout=${2:-1200}
  echo "Waiting for vLLM on port ${port} (timeout ${timeout}s)..."
  timeout "${timeout}" bash -c "
    until curl -fsS http://127.0.0.1:${port}/v1/models > /dev/null 2>&1; do
      sleep 2
    done
  "
}

register_pid() {
  echo "$1" >> "${PID_FILE}"
}

stop_all_pids() {
  if [[ ! -s "${PID_FILE}" ]]; then
    echo "No PIDs recorded in ${PID_FILE}."
    return
  fi
  echo "Stopping recorded processes..."
  # Reverse so the proxy is killed before the backends.
  tac "${PID_FILE}" | while read -r pid; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  sleep 2
  tac "${PID_FILE}" | while read -r pid; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done
  : > "${PID_FILE}"
}

trap_cleanup() {
  trap - INT TERM EXIT
  stop_all_pids
}
trap trap_cleanup INT TERM EXIT

# ---- KV-transfer-config builders --------------------------------------------
#
# All deployments use NixlConnector because it is the only connector that
# supports both `kv_role=kv_producer/kv_consumer` (PD-disagg) and
# `kv_role=kv_both` (homogeneous PD) with bidirectional transfer.
#
# Output is printed to stdout so it can be embedded into `vllm serve` args:
#   eval vllm serve $MODEL ... --kv-transfer-config "$(_kv_config_homogeneous)"

_kv_config_homogeneous() {
  cat <<EOF
{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"${KV_BUFFER_DEVICE}"}
EOF
}

_kv_config_pd_disagg() {
  # $1: kv_producer | kv_consumer
  cat <<EOF
{"kv_connector":"NixlConnector","kv_role":"$1","kv_buffer_device":"${KV_BUFFER_DEVICE}"}
EOF
}
