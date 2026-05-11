# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-first scheduler with prefill budget proportional to decode work.

This scheduler is the experimental piece of the *homogeneous PD* setup:
every node runs both prefill and decode in a continuous-batching forward
pass, but on every scheduling step the number of prefill tokens is bounded
by a function of the decode tokens being processed in the same step.

Activate it via the public ``--scheduler-cls`` flag:

    PYTHONPATH=path/to/vllm/examples \
    vllm serve <model> \\
        --scheduler-cls \
        homogeneous_pd.scheduler.homogeneous_scheduler.HomogeneousScheduler \\
        ...

Configuration (in priority order):

1. ``vllm_config.kv_transfer_config.kv_connector_extra_config``:
   ``homogeneous_alpha``, ``homogeneous_budget_fn``,
   ``homogeneous_prefill_min``, ``homogeneous_prefill_max``.
2. Environment variables ``HOMOGENEOUS_ALPHA``, ``HOMOGENEOUS_BUDGET_FN``,
   ``HOMOGENEOUS_PREFILL_MIN``, ``HOMOGENEOUS_PREFILL_MAX``.
3. Defaults: ``alpha=16.0`` (matches the default vLLM page size),
   ``budget_fn="linear_ratio"``, no min/max cap.
"""

from __future__ import annotations

import os
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler

from .budget_fn import PrefillBudgetFn, get_budget_fn

logger = init_logger(__name__)


_ENV_PREFIX = "HOMOGENEOUS_"


class HomogeneousScheduler(Scheduler):
    """Decode-first scheduler used by the homogeneous-PD experiment.

    On each call to :meth:`schedule`:

    1. Walk ``self.running`` and conservatively estimate the number of decode
       tokens that the base scheduler will issue this step (mirroring its own
       per-request token caps).
    2. Compute the per-step prefill budget via a pluggable function:
       ``prefill_budget = budget_fn(num_decode_tokens, alpha=..., ...)``.
       The default ``linear_ratio`` returns ``num_decode_tokens * alpha``.
    3. Temporarily clamp ``self.max_num_scheduled_tokens`` to
       ``num_decode_tokens + prefill_budget`` (capped by the original maximum)
       and delegate to the base scheduler.

    Because the base scheduler already iterates RUNNING (decode + chunked
    prefill continuations) before WAITING (new prefill), the clamp has the
    effect of guaranteeing room for all decodes first and then giving prefill
    the remainder up to the budget.

    When there are no decodes (``num_decode_tokens == 0``), no clamp is
    applied — this avoids starving the system during ramp-up or steady-state
    with no active decodes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        cfg: dict[str, Any] = {}
        kv_cfg = self.vllm_config.kv_transfer_config
        if kv_cfg is not None and kv_cfg.kv_connector_extra_config:
            cfg.update(kv_cfg.kv_connector_extra_config)

        def _resolve(key: str, default: Any) -> Any:
            if key in cfg and cfg[key] is not None:
                return cfg[key]
            env_key = _ENV_PREFIX + key.removeprefix("homogeneous_").upper()
            return os.environ.get(env_key, default)

        self.alpha: float = float(_resolve("homogeneous_alpha", 16.0))
        self.budget_fn_name: str = str(
            _resolve("homogeneous_budget_fn", "linear_ratio")
        )
        self.prefill_min: int = int(_resolve("homogeneous_prefill_min", 0))
        self.prefill_max: int = int(_resolve("homogeneous_prefill_max", 0))

        self.budget_fn: PrefillBudgetFn = get_budget_fn(self.budget_fn_name)
        self._original_max_scheduled: int = self.max_num_scheduled_tokens

        logger.info(
            "HomogeneousScheduler enabled: budget_fn=%s alpha=%.3f "
            "prefill_min=%d prefill_max=%d max_num_batched_tokens=%d",
            self.budget_fn_name,
            self.alpha,
            self.prefill_min,
            self.prefill_max,
            self._original_max_scheduled,
        )

    def _estimate_decode_tokens(self) -> int:
        """Estimate decode tokens that the base scheduler will issue this step.

        Mirrors the per-request token-cap logic at the top of
        :meth:`Scheduler.schedule` so the estimate is tight; we never
        under-budget by more than a handful of tokens. Requests still in a
        chunked-prefill stage (``is_prefill_chunk == True``) are skipped, since
        their tokens count as prefill, not decode.
        """
        threshold = self.scheduler_config.long_prefill_token_threshold
        max_pos = self.max_model_len - 1
        total = 0
        for req in self.running:
            if req.is_prefill_chunk:
                continue
            n = (
                req.num_tokens_with_spec
                + req.num_output_placeholders
                - req.num_computed_tokens
            )
            if 0 < threshold < n:
                n = threshold
            n = min(n, max_pos - req.num_computed_tokens)
            if n > 0:
                total += n
        return total

    def schedule(self) -> SchedulerOutput:
        decode_estimate = self._estimate_decode_tokens()
        if decode_estimate <= 0:
            # No decodes to protect this step: keep the original budget so
            # prefill (new or continuing) can ramp up unrestricted.
            return super().schedule()

        prefill_budget = int(
            self.budget_fn(
                decode_tokens=decode_estimate,
                alpha=self.alpha,
                max_batched=self._original_max_scheduled,
            )
        )
        if self.prefill_min > 0:
            prefill_budget = max(prefill_budget, self.prefill_min)
        if self.prefill_max > 0:
            prefill_budget = min(prefill_budget, self.prefill_max)
        prefill_budget = max(prefill_budget, 0)

        new_cap = min(
            decode_estimate + prefill_budget,
            self._original_max_scheduled,
        )
        # Always leave room for the decodes themselves.
        new_cap = max(new_cap, min(decode_estimate, self._original_max_scheduled))

        try:
            self.max_num_scheduled_tokens = new_cap
            return super().schedule()
        finally:
            self.max_num_scheduled_tokens = self._original_max_scheduled
