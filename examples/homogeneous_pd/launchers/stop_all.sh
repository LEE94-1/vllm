#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Best-effort cleanup of any leftover vLLM / proxy / launcher processes.
# Use this only if a launcher exited uncleanly (its EXIT trap normally handles
# cleanup automatically).

set -u

LOG_DIR=${LOG_DIR:-/tmp/homogeneous_pd_logs}

shopt -s nullglob
for pid_file in "${LOG_DIR}"/pids*.txt; do
  echo "Killing PIDs recorded in ${pid_file}..."
  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    kill "${pid}" 2>/dev/null || true
  done < "${pid_file}"
done
sleep 1
for pid_file in "${LOG_DIR}"/pids*.txt; do
  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    kill -9 "${pid}" 2>/dev/null || true
  done < "${pid_file}"
  rm -f "${pid_file}"
done
shopt -u nullglob

echo "Pkill any remaining 'vllm serve' / round_robin_proxy / toy_proxy_server..."
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "homogeneous_pd.proxy.round_robin_proxy" 2>/dev/null || true
pkill -f "toy_proxy_server.py" 2>/dev/null || true

echo "Done."
