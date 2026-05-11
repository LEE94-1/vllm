#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Convenience wrapper: 2-node homogeneous PD (matches the 1P1D baseline).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
NUM_NODES=2 exec "${SCRIPT_DIR}/launch_homogeneous.sh" "$@"
