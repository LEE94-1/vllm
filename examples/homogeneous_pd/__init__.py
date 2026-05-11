# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Homogeneous PD experiment package.

A self-contained set of scheduler / proxy / launcher / test utilities used to
compare classic PD-disaggregated serving against a "homogeneous PD" setup
where every node serves both prefill and decode in the same forward pass.

See ``README.md`` in this directory for usage.
"""
