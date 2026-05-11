#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Convenience wrapper: PD-disaggregated 1P + 1D (matches a 2-node homogeneous
# setup launched via run_homogeneous_2node.sh).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
NUM_PREFILL=1 NUM_DECODE=1 exec "${SCRIPT_DIR}/launch_pd_disagg.sh" "$@"
