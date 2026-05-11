# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable per-step prefill budget functions.

A budget function takes the number of decode tokens scheduled in the current
step (plus auxiliary keyword context such as the configured ``alpha`` and the
total ``max_batched`` token cap) and returns the number of prefill tokens that
the scheduler is allowed to schedule in the same step.

To register a new function, either call :func:`register_budget_fn` at import
time or extend the ``_REGISTRY`` dict directly. The function name is then
referenced by the ``homogeneous_budget_fn`` config key (see
:class:`~examples.homogeneous_pd.scheduler.homogeneous_scheduler.HomogeneousScheduler`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

PrefillBudgetFn = Callable[..., int]


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
