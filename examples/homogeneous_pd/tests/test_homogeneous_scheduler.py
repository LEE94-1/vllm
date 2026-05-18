# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the HomogeneousScheduler.

These tests reuse the lightweight scheduler scaffolding from
``tests/v1/core/utils.py`` (which the upstream tests use as well) and only
substitute in our HomogeneousScheduler subclass.

Run from the repository root::

    .venv/bin/python -m pytest examples/homogeneous_pd/tests \
        -v --no-header

The tests are CPU-only and take a few seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make `homogeneous_pd` and `tests.v1.core.utils` importable when running this
# file directly (without installing the example as a package).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
for path in (_REPO_ROOT, _EXAMPLES_DIR):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from homogeneous_pd.scheduler.budget_fn import (  # noqa: E402
    ceil_to_page_size,
    fixed_cap,
    get_budget_fn,
    kv_decode_page,
    linear_ratio,
)
from homogeneous_pd.scheduler.homogeneous_scheduler import (  # noqa: E402
    HomogeneousScheduler,
)

# tests/v1/core/utils.py provides create_requests + helpers we reuse.
from tests.v1.core.utils import create_requests  # noqa: E402
from vllm.config import (  # noqa: E402
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scheduler(
    *,
    alpha: float = 16.0,
    budget_fn: str | None = "linear_ratio",
    prefill_min: int = 0,
    prefill_max: int = 0,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    max_model_len: int | None = None,
    block_size: int = 16,
    num_blocks: int = 10000,
) -> HomogeneousScheduler:
    """Build a HomogeneousScheduler wired up with a real KVCacheConfig.

    Mirrors the structure of ``tests/v1/core/utils.create_scheduler`` but
    instantiates HomogeneousScheduler directly, with our extra config baked in.
    """
    if max_model_len is None:
        max_model_len = max_num_batched_tokens

    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=False,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="ExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "shared_storage_path": "local_storage",
            "homogeneous_alpha": alpha,
            "homogeneous_prefill_min": prefill_min,
            "homogeneous_prefill_max": prefill_max,
            **(
                {"homogeneous_budget_fn": budget_fn}
                if budget_fn is not None
                else {}
            ),
        },
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        kv_transfer_config=kv_transfer_config,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    return HomogeneousScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def _drive_to_decode(scheduler: HomogeneousScheduler, n_decodes: int, prompt_len: int):
    """Add ``n_decodes`` short requests and step them through prefill into decode.

    Returns the list of requests, all of which will be in the RUNNING state and
    have ``is_prefill_chunk == False`` after this call (i.e. they will look
    like decoding requests to the next ``schedule()`` call).
    """
    requests = create_requests(
        num_requests=n_decodes,
        num_tokens=prompt_len,
        max_tokens=128,
    )
    for req in requests:
        scheduler.add_request(req)

    # Prefill in one or more steps until every request is past prompt.
    for _ in range(8):
        out = scheduler.schedule()
        if out.total_num_scheduled_tokens == 0:
            break
        runner_out = ModelRunnerOutput(
            req_ids=[r.request_id for r in requests],
            req_id_to_index={r.request_id: i for i, r in enumerate(requests)},
            # Sample 1 token per request — but only if its prefill just finished
            # in this step (i.e. it would now be in decode).
            sampled_token_ids=[
                [0] if (not r.is_prefill_chunk) else [] for r in requests
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        scheduler.update_from_output(out, runner_out)
        if all(not r.is_prefill_chunk for r in requests):
            break

    assert all(not r.is_prefill_chunk for r in requests), (
        "failed to drive requests into decode"
    )
    return requests


# ---------------------------------------------------------------------------
# Budget-function unit tests
# ---------------------------------------------------------------------------


def test_linear_ratio_basic():
    assert linear_ratio(decode_tokens=4, alpha=16, max_batched=8192) == 64
    assert linear_ratio(decode_tokens=0, alpha=16, max_batched=8192) == 0


def test_linear_ratio_clamped_to_remaining():
    # decode + prefill must not exceed max_batched.
    assert linear_ratio(decode_tokens=100, alpha=1000, max_batched=128) == 28


def test_linear_ratio_alpha_zero_blocks_prefill():
    assert linear_ratio(decode_tokens=10, alpha=0, max_batched=8192) == 0


def test_fixed_cap_basic():
    assert fixed_cap(alpha=512, decode_tokens=8, max_batched=8192) == 512
    # Clamped to remaining batch.
    assert fixed_cap(alpha=10000, decode_tokens=8, max_batched=128) == 120


def test_ceil_to_page_size():
    assert ceil_to_page_size(0, 16) == 0
    assert ceil_to_page_size(1, 16) == 16
    assert ceil_to_page_size(16, 16) == 16
    assert ceil_to_page_size(17, 16) == 32


def test_kv_decode_page_basic():
    assert (
        kv_decode_page(
            decode_tokens=3,
            kv_cache_tokens=4096,
            kv_budget_ratio=1.0 / 4096.0,
            decode_budget_ratio=1.0,
            page_size=16,
            max_batched=8192,
        )
        == 16
    )
    assert (
        kv_decode_page(
            decode_tokens=0,
            kv_cache_tokens=8192,
            kv_budget_ratio=1.0 / 4096.0,
            decode_budget_ratio=1.0,
            page_size=16,
            max_batched=8192,
        )
        == 16
    )
    assert (
        kv_decode_page(
            decode_tokens=3,
            kv_cache_tokens=0,
            kv_budget_ratio=1.0 / 4096.0,
            decode_budget_ratio=1.0,
            page_size=16,
            max_batched=8192,
        )
        == 16
    )


def test_kv_decode_page_clamped_to_remaining():
    assert (
        kv_decode_page(
            decode_tokens=100,
            kv_cache_tokens=0,
            decode_budget_ratio=1000.0,
            page_size=16,
            max_batched=128,
        )
        == 28
    )


def test_get_budget_fn_kv_decode_page():
    assert get_budget_fn("kv_decode_page") is kv_decode_page


def test_get_budget_fn_unknown_raises():
    with pytest.raises(ValueError):
        get_budget_fn("does-not-exist")


# ---------------------------------------------------------------------------
# Scheduler-level tests
# ---------------------------------------------------------------------------


def test_config_resolution_from_extra_config():
    sched = _make_scheduler(alpha=8.0, budget_fn="fixed_cap", prefill_min=32)
    assert sched.alpha == 8.0
    assert sched.budget_fn_name == "fixed_cap"
    assert sched.prefill_min == 32
    assert sched.budget_fn is get_budget_fn("fixed_cap")


def test_default_budget_fn_is_kv_decode_page():
    sched = _make_scheduler(budget_fn=None)
    assert sched.budget_fn_name == "kv_decode_page"
    assert sched.kv_budget_ratio == pytest.approx(1.0 / 4096.0)
    assert sched.decode_budget_ratio == pytest.approx(1.0)
    assert sched.budget_fn is get_budget_fn("kv_decode_page")


def test_kv_decode_page_caps_prefill_against_decode_count():
    """3 decodes + 1 long waiting prefill -> prefill capped at one page."""
    sched = _make_scheduler(budget_fn="kv_decode_page", max_num_batched_tokens=8192)
    decodes = _drive_to_decode(sched, n_decodes=3, prompt_len=10)

    new_prefill = create_requests(num_requests=1, num_tokens=2000, max_tokens=8)
    for req in new_prefill:
        sched.add_request(req)

    out = sched.schedule()
    decode_tokens = sum(out.num_scheduled_tokens[r.request_id] for r in decodes)
    prefill_tokens = sum(
        out.num_scheduled_tokens.get(r.request_id, 0) for r in new_prefill
    )
    assert decode_tokens == 3, "all decodes must be admitted first"
    assert prefill_tokens == 16


def test_no_decodes_uses_full_budget():
    """Pure cold start: only WAITING prefill, no decodes -> full max budget."""
    sched = _make_scheduler(alpha=2.0, max_num_batched_tokens=512)
    requests = create_requests(num_requests=1, num_tokens=400, max_tokens=8)
    for req in requests:
        sched.add_request(req)

    out = sched.schedule()
    # Without our cap, base scheduler would schedule all 400 tokens (<= 512).
    # With our cap, alpha*0=0 + 0 decodes would be 0; we explicitly bail to the
    # original budget when decode_estimate == 0, so we expect 400.
    assert out.num_scheduled_tokens[requests[0].request_id] == 400


def test_pure_decode_step_unaffected():
    """When only decodes are running, every decode token still gets through."""
    sched = _make_scheduler(alpha=16.0, max_num_batched_tokens=512)
    decodes = _drive_to_decode(sched, n_decodes=3, prompt_len=10)

    out = sched.schedule()
    # 1 token per decode (no spec decoding configured).
    total = sum(out.num_scheduled_tokens.values())
    assert total == 3
    for r in decodes:
        assert out.num_scheduled_tokens[r.request_id] == 1


def test_alpha_caps_prefill_against_decode_count():
    """3 decodes + 1 long waiting prefill, alpha=16 -> prefill capped at 48."""
    sched = _make_scheduler(alpha=16.0, max_num_batched_tokens=8192)
    decodes = _drive_to_decode(sched, n_decodes=3, prompt_len=10)

    new_prefill = create_requests(num_requests=1, num_tokens=2000, max_tokens=8)
    for req in new_prefill:
        sched.add_request(req)

    out = sched.schedule()
    decode_tokens = sum(out.num_scheduled_tokens[r.request_id] for r in decodes)
    prefill_tokens = sum(
        out.num_scheduled_tokens.get(r.request_id, 0) for r in new_prefill
    )
    assert decode_tokens == 3, "all decodes must be admitted first"
    # alpha=16, 3 decodes -> 48 prefill tokens this step.
    assert prefill_tokens == 48


def test_alpha_zero_blocks_new_prefill():
    """alpha=0 with running decodes -> no new prefill admitted this step."""
    sched = _make_scheduler(alpha=0.0, max_num_batched_tokens=8192)
    _drive_to_decode(sched, n_decodes=2, prompt_len=10)

    new_prefill = create_requests(num_requests=1, num_tokens=2000, max_tokens=8)
    for req in new_prefill:
        sched.add_request(req)

    out = sched.schedule()
    prefill_tokens = sum(
        out.num_scheduled_tokens.get(r.request_id, 0) for r in new_prefill
    )
    assert prefill_tokens == 0, (
        f"alpha=0 should starve prefill; got {prefill_tokens} prefill tokens"
    )


def test_large_alpha_matches_default_scheduler():
    """alpha very large -> behaves like default chunked prefill (no extra cap)."""
    sched = _make_scheduler(alpha=10_000.0, max_num_batched_tokens=512)
    decodes = _drive_to_decode(sched, n_decodes=2, prompt_len=10)

    new_prefill = create_requests(num_requests=1, num_tokens=2000, max_tokens=8)
    for req in new_prefill:
        sched.add_request(req)

    out = sched.schedule()
    total = sum(out.num_scheduled_tokens.values())
    # 2 decodes + remainder of max_num_batched_tokens = 512.
    assert total == 512
    decode_tokens = sum(out.num_scheduled_tokens[r.request_id] for r in decodes)
    assert decode_tokens == 2
    prefill_tokens = sum(
        out.num_scheduled_tokens.get(r.request_id, 0) for r in new_prefill
    )
    assert prefill_tokens == 510


def test_prefill_min_floor():
    """prefill_min raises the budget when alpha*decodes is too small."""
    sched = _make_scheduler(alpha=1.0, prefill_min=128, max_num_batched_tokens=8192)
    _drive_to_decode(sched, n_decodes=2, prompt_len=10)

    new_prefill = create_requests(num_requests=1, num_tokens=2000, max_tokens=8)
    for req in new_prefill:
        sched.add_request(req)

    out = sched.schedule()
    prefill_tokens = sum(
        out.num_scheduled_tokens.get(r.request_id, 0) for r in new_prefill
    )
    # alpha*decodes = 2, but prefill_min=128 lifts it.
    assert prefill_tokens == 128


def test_prefill_max_ceiling():
    """prefill_max caps the budget when alpha*decodes is too large."""
    sched = _make_scheduler(alpha=1000.0, prefill_max=64, max_num_batched_tokens=8192)
    _drive_to_decode(sched, n_decodes=4, prompt_len=10)

    new_prefill = create_requests(num_requests=1, num_tokens=2000, max_tokens=8)
    for req in new_prefill:
        sched.add_request(req)

    out = sched.schedule()
    prefill_tokens = sum(
        out.num_scheduled_tokens.get(r.request_id, 0) for r in new_prefill
    )
    # alpha*decodes would be 4000, but prefill_max=64 caps it.
    assert prefill_tokens == 64


def test_total_does_not_exceed_max_batched():
    """Sanity: the per-step cap must always respect max_num_batched_tokens."""
    sched = _make_scheduler(alpha=1000.0, max_num_batched_tokens=128)
    _drive_to_decode(sched, n_decodes=1, prompt_len=4)

    new_prefill = create_requests(num_requests=1, num_tokens=2000, max_tokens=8)
    for req in new_prefill:
        sched.add_request(req)

    out = sched.schedule()
    assert out.total_num_scheduled_tokens <= 128
