#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Start the router (a.k.a. proxy) based on a topology config file. The router
# can run on any machine that can reach all NODE_HOSTS:NODE_PORTS over HTTP.
#
# Usage:
#   ./launch_router.sh <config-file>
#
# - For TYPE=pd_disagg, the script delegates to the upstream NIXL proxy at
#   tests/v1/kv_connector/nixl_integration/toy_proxy_server.py, splitting the
#   nodes into prefill (kv_producer) and decode (kv_consumer) groups.
# - For TYPE=homogeneous, it starts homogeneous_pd.proxy.round_robin_proxy.
#
# The router runs in the foreground; press Ctrl-C to stop it. Logs are tee'd
# to ${LOG_DIR}/router.log.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <config-file>" >&2
  exit 2
fi

CONFIG=$1
if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${CONFIG}"

: "${TYPE:?TYPE not set in ${CONFIG}}"
ROUTER_HOST=${ROUTER_HOST:-0.0.0.0}
ROUTER_PORT=${ROUTER_PORT:-8000}

LOG="${LOG_DIR}/router.log"

case "${TYPE}" in
  pd_disagg)
    PREFILL_HOSTS=()
    PREFILL_PORTS=()
    DECODE_HOSTS=()
    DECODE_PORTS=()
    for i in "${!ROLES[@]}"; do
      case "${ROLES[$i]}" in
        kv_producer)
          PREFILL_HOSTS+=("${NODE_HOSTS[$i]}")
          PREFILL_PORTS+=("${NODE_PORTS[$i]}")
          ;;
        kv_consumer)
          DECODE_HOSTS+=("${NODE_HOSTS[$i]}")
          DECODE_PORTS+=("${NODE_PORTS[$i]}")
          ;;
        *)
          echo "TYPE=pd_disagg expects kv_producer/kv_consumer, got ${ROLES[$i]} at index $i" >&2
          exit 1
          ;;
      esac
    done
    if (( ${#PREFILL_HOSTS[@]} == 0 || ${#DECODE_HOSTS[@]} == 0 )); then
      echo "PD-disagg config needs at least one kv_producer and one kv_consumer." >&2
      exit 1
    fi

    PROXY_SCRIPT="${REPO_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
    if [[ ! -f "${PROXY_SCRIPT}" ]]; then
      echo "Cannot find PD-disagg proxy at ${PROXY_SCRIPT}" >&2
      exit 1
    fi

    echo "============================================================"
    echo "Launching PD-disagg router from ${CONFIG}"
    echo "  bind     : ${ROUTER_HOST}:${ROUTER_PORT}"
    echo "  prefill  : ${PREFILL_HOSTS[*]}  ports=${PREFILL_PORTS[*]}"
    echo "  decode   : ${DECODE_HOSTS[*]}  ports=${DECODE_PORTS[*]}"
    echo "  log      : ${LOG}"
    echo "============================================================"

    python3 "${PROXY_SCRIPT}" \
      --host "${ROUTER_HOST}" \
      --port "${ROUTER_PORT}" \
      --prefiller-hosts "${PREFILL_HOSTS[@]}" \
      --prefiller-ports "${PREFILL_PORTS[@]}" \
      --decoder-hosts "${DECODE_HOSTS[@]}" \
      --decoder-ports "${DECODE_PORTS[@]}" \
      2>&1 | tee "${LOG}"
    ;;

  homogeneous)
    for role in "${ROLES[@]}"; do
      if [[ "${role}" != "kv_both" ]]; then
        echo "TYPE=homogeneous expects every ROLES entry to be kv_both, got ${role}" >&2
        exit 1
      fi
    done

    echo "============================================================"
    echo "Launching homogeneous round-robin router from ${CONFIG}"
    echo "  bind     : ${ROUTER_HOST}:${ROUTER_PORT}"
    echo "  backends : ${NODE_HOSTS[*]}  ports=${NODE_PORTS[*]}"
    echo "  log      : ${LOG}"
    echo "============================================================"

    PYTHONPATH="${PYTHONPATH}" python3 -m homogeneous_pd.proxy.round_robin_proxy \
      --host "${ROUTER_HOST}" \
      --port "${ROUTER_PORT}" \
      --backend-hosts "${NODE_HOSTS[@]}" \
      --backend-ports "${NODE_PORTS[@]}" \
      2>&1 | tee "${LOG}"
    ;;

  *)
    echo "Unknown TYPE: ${TYPE} (expected pd_disagg or homogeneous)" >&2
    exit 1
    ;;
esac
