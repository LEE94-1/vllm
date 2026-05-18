# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable per-step prefill budget functions.

A budget function receives the estimated decode tokens for the current step,
plus optional load context (KV cache occupancy) and returns how many prefill
tokens the scheduler may schedule in the same step.

Registered functions are referenced by ``homogeneous_budget_fn`` (see
:class:`~examples.homogeneous_pd.scheduler.homogeneous_scheduler.HomogeneousScheduler`).

Keyword arguments supplied by :class:`HomogeneousScheduler` on every call:

- ``decode_tokens``: conservative estimate of decode tokens this step.
- ``alpha``: configured homogeneous alpha.
- ``max_batched``: original ``max_num_scheduled_tokens`` cap.
- ``kv_cache_usage``: KV block-pool usage in ``[0.0, 1.0]`` (same family as
  Prometheus ``vllm:kv_cache_usage_perc``).
- ``kv_cache_tokens``: estimated occupied KV tokens, derived from block usage
  (``usage * (num_blocks - 1) * block_size``). This is a block-granularity
  occupancy estimate, not a per-request exact token sum.
- ``kv_budget_ratio``, ``decode_budget_ratio``, ``page_size``: used by
  ``kv_decode_page`` (defaults ``1/4096``, ``1.0``, and the scheduler block size).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

PrefillBudgetFn = Callable[..., int]


def ceil_to_page_size(tokens: float, page_size: int) -> int:
    """Round ``tokens`` up to a multiple of ``page_size`` (vLLM KV page)."""
    page = max(page_size, 1)
    if tokens <= 0:
        return 0
    return int(math.ceil(tokens / page) * page)


def linear_ratio(
    *,
    decode_tokens: int,
    alpha: float,
    max_batched: int,
    **_: Any,
) -> int:
    """``prefill_budget = decode_tokens * alpha``, clamped to remaining batch.

    This is the default policy: for every decode token in the current step,
    allow ``alpha`` prefill tokens. The clamp ensures the total batch never
    exceeds ``max_num_batched_tokens``.
    """
    raw = int(decode_tokens * alpha)
    remaining = max(0, max_batched - decode_tokens)
    return max(0, min(raw, remaining))


def kv_decode_page(
    *,
    decode_tokens: int,
    kv_cache_tokens: int = 0,
    kv_budget_ratio: float = 1.0 / 4096.0,
    decode_budget_ratio: float = 1.0,
    page_size: int = 16,
    max_batched: int,
    **_: Any,
) -> int:
    """KV occupancy + decode weighted budget, rounded up to page size.

    ``raw = kv_cache_tokens * kv_budget_ratio + decode_tokens * decode_budget_ratio``
    then ``prefill_budget = ceil_to_page_size(raw, page_size)``, clamped to the
    remaining batch capacity after decode tokens.
    """
    raw = kv_cache_tokens * kv_budget_ratio + decode_tokens * decode_budget_ratio
    budget = ceil_to_page_size(raw, page_size)
    remaining = max(0, max_batched - decode_tokens)
    return max(0, min(budget, remaining))


def fixed_cap(
    *,
    alpha: float,
    decode_tokens: int,
    max_batched: int,
    **_: Any,
) -> int:
    """``prefill_budget = alpha`` (constant), independent of decode tokens.

    Useful as a baseline / sanity check: behaves like a fixed chunked-prefill
    chunk size regardless of the number of running decodes.
    """
    raw = int(alpha)
    remaining = max(0, max_batched - decode_tokens)
    return max(0, min(raw, remaining))


_REGISTRY: dict[str, PrefillBudgetFn] = {
    "kv_decode_page": kv_decode_page,
    "linear_ratio": linear_ratio,
    "fixed_cap": fixed_cap,
}


def register_budget_fn(name: str, fn: PrefillBudgetFn) -> None:
    """Register a custom budget function under ``name``."""
    _REGISTRY[name] = fn


def get_budget_fn(name: str) -> PrefillBudgetFn:
    """Resolve a budget function by name. Raises ``ValueError`` if unknown."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown homogeneous prefill budget function: {name!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_budget_fns() -> list[str]:
    """Return the list of registered budget function names."""
    return sorted(_REGISTRY)
