#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Convenience wrapper: 4-node homogeneous PD (matches the 3P1D baseline).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
NUM_NODES=4 exec "${SCRIPT_DIR}/launch_homogeneous.sh" "$@"
