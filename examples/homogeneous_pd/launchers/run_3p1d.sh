#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Convenience wrapper: PD-disaggregated 3P + 1D (matches a 4-node homogeneous
# setup launched via run_homogeneous_4node.sh).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
NUM_PREFILL=3 NUM_DECODE=1 exec "${SCRIPT_DIR}/launch_pd_disagg.sh" "$@"
