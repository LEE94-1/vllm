# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive P/D-aware proxy for homogeneous_pd backends (kv_role=kv_both).

Different from the stock ``round_robin_proxy`` (which forwards the whole
request to one backend), this proxy:

1. **Picks a prefill backend** by one of three strategies (env-configurable):
   * ``min_pending_prefill``  (default) — least proxy-tracked in-flight prefill tokens.
   * ``min_waiting_reqs``               — least Prometheus-reported waiting requests.
   * ``round_robin``                    — plain cyclic.

2. **Picks a decode backend** by lowest *KV-aware load score*::

       load_score(i) = num_requests_running(i)
                     + decode_kv_weight * kv_cache_usage_perc(i)

   The lower the score, the lighter the load — KV cache pressure is folded
   in so we don't dump a new decode onto a backend that's about to run out
   of KV blocks.

3. **Short-circuit when P == D** (i.e. the same backend wins both picks):
   the request is forwarded as a single, unsplit call to that backend.
   ``HomogeneousScheduler`` will then handle decode-first within that node
   and let prefill ride along.

4. **Two-stage decision when P != D**:
   * Stage 1: send a prefill sub-request (``max_tokens=1``,
     ``do_remote_decode=True``) to the chosen P backend; receive
     ``kv_transfer_params`` back.
   * Stage 2: re-pick D *after* prefill returns (load picture is fresher),
     forward the full request with the prefill's ``kv_transfer_params``
     attached. The decoder's ``NixlConnector`` pulls KV blocks from P
     over NIXL/UCX (cross-GPU-group if P != D physically).

A background coroutine polls ``/metrics`` on every backend at
``POLL_INTERVAL_MS`` cadence (default 200 ms) and parses the three relevant
Prometheus gauges.

Usage::

    python -m homogeneous_pd.proxy.adaptive_pd_proxy \\
        --host 0.0.0.0 --port 8000 \\
        --backend-hosts 127.0.0.1 127.0.0.1 \\
        --backend-ports 8100 8101

Environment overrides::

    PREFILL_STRATEGY      min_pending_prefill (default) | min_waiting_reqs | round_robin
    DECODE_KV_WEIGHT      float, default 10.0 — beta in score = running + beta * cache_usage
    POLL_INTERVAL_MS      int, default 200
    PROMPT_TOKEN_DIVISOR  int, default 3 — rough chars→tokens divisor used until the
                          real prompt_tokens count arrives in the prefill response.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass, field

import aiohttp
from aiohttp import web

logger = logging.getLogger("homogeneous_pd.adaptive_proxy")

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


def _filter_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


# ---------- Prometheus text-format mini-parser ------------------------------
# We only need three gauges; ignore labels other than to disambiguate samples.

_METRIC_LINE_RE = re.compile(
    r"""^
    (?P<name>vllm:[a-zA-Z_:][a-zA-Z0-9_:]*)
    (?:\{[^}]*\})?      # optional labels
    \s+
    (?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|NaN|\+?Inf)
    \s*$
    """,
    re.VERBOSE,
)

_WANTED = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
)


def parse_prometheus(text: str) -> dict[str, float]:
    """Return the LAST sample of each wanted metric. Good enough for gauges."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        if name not in _WANTED:
            continue
        try:
            out[name] = float(m.group("value"))
        except ValueError:
            continue
    return out


# ---------- Per-backend state -----------------------------------------------


@dataclass
class BackendStats:
    host: str
    port: int
    # ---- metric fields (only written by _poll_once) ----
    num_running: float = 0.0
    num_waiting: float = 0.0
    kv_usage: float = 0.0  # in [0, 1]
    # ---- proxy-tracked fields ----
    pending_prefill_tokens: int = 0  # in-flight prefill cost (split path)
    # Requests dispatched by this proxy since the last metric poll. We bump
    # this on dispatch and clear it on each successful poll. It exists to plug
    # the visibility hole during sub-poll-cadence bursts: a freshly dispatched
    # request will not show up in vLLM's /metrics until the engine adds it to
    # the running queue and the next scrape arrives (~poll_interval_ms). Both
    # pick_prefill and pick_decode treat (num_running + pending_dispatches)
    # as the effective load.
    pending_dispatches: int = 0
    last_poll_ok: bool = False
    last_poll_error: str = ""
    in_flight_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def effective_load(self) -> float:
        """Best estimate of in-flight request count (metric + freshly dispatched)."""
        return self.num_running + self.pending_dispatches


# ---------- Proxy core ------------------------------------------------------


class AdaptivePDProxy:
    def __init__(
        self,
        backends: list[tuple[str, int]],
        prefill_strategy: str = "min_pending_prefill",
        decode_kv_weight: float = 10.0,
        poll_interval_ms: int = 200,
        prompt_token_divisor: int = 3,
    ) -> None:
        if not backends:
            raise ValueError("AdaptivePDProxy requires at least one backend")
        if prefill_strategy not in (
            "min_pending_prefill",
            "min_waiting_reqs",
            "round_robin",
        ):
            raise ValueError(f"unknown prefill strategy: {prefill_strategy}")
        self.backends = backends
        self.stats = [BackendStats(host=h, port=p) for h, p in backends]
        self.prefill_strategy = prefill_strategy
        self.decode_kv_weight = decode_kv_weight
        self.poll_interval_s = max(poll_interval_ms, 50) / 1000.0
        self.prompt_token_divisor = max(prompt_token_divisor, 1)
        self._rr_cycle = itertools.cycle(range(len(backends)))
        self._session: aiohttp.ClientSession | None = None
        self._poll_task: asyncio.Task | None = None
        self._request_counter = 0
        # diagnostic counters for /proxy/healthcheck
        self._counter_short_circuit = 0
        self._counter_split_same = 0  # P!=D initial pick, but re-pick landed back on P
        self._counter_split_cross = 0

    # ---- lifecycle ----
    async def startup(self, _: web.Application) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30),
            connector=aiohttp.TCPConnector(limit=0, force_close=False),
        )
        self._poll_task = asyncio.create_task(self._poll_loop(), name="metrics_poll")
        logger.info(
            "AdaptivePDProxy ready: %d backend(s)=%s prefill_strategy=%s "
            "decode_kv_weight=%.2f poll_interval=%dms",
            len(self.backends),
            self.backends,
            self.prefill_strategy,
            self.decode_kv_weight,
            int(self.poll_interval_s * 1000),
        )

    async def cleanup(self, _: web.Application) -> None:
        if self._poll_task is not None:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
        if self._session is not None:
            await self._session.close()

    # ---- background metrics poll ----
    async def _poll_loop(self) -> None:
        try:
            while True:
                await self._poll_once()
                await asyncio.sleep(self.poll_interval_s)
        except asyncio.CancelledError:
            return

    async def _poll_once(self) -> None:
        assert self._session is not None

        async def _one(i: int) -> None:
            url = f"http://{self.stats[i].host}:{self.stats[i].port}/metrics"
            try:
                async with self._session.get(
                    url, timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status != 200:
                        self.stats[i].last_poll_ok = False
                        self.stats[i].last_poll_error = f"HTTP {resp.status}"
                        return
                    text = await resp.text()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                self.stats[i].last_poll_ok = False
                self.stats[i].last_poll_error = f"{type(e).__name__}: {e}"
                return
            metrics = parse_prometheus(text)
            s = self.stats[i]
            s.num_running = metrics.get("vllm:num_requests_running", 0.0)
            s.num_waiting = metrics.get("vllm:num_requests_waiting", 0.0)
            s.kv_usage = metrics.get("vllm:kv_cache_usage_perc", 0.0)
            # Fresh metric snapshot has just landed, so any request we
            # accounted for via pending_dispatches is now reflected in
            # num_running (or has already completed). Clear the local counter
            # to avoid double-counting from this point on.
            s.pending_dispatches = 0
            s.last_poll_ok = True
            s.last_poll_error = ""

        await asyncio.gather(*(_one(i) for i in range(len(self.stats))))

    # ---- pick strategies ----
    def pick_prefill(self) -> int:
        """Select a backend to send the prefill stage to.

        Strategy hierarchy (lower is better at every level):

        * ``round_robin``: pure cycle, ignores load.
        * ``min_pending_prefill``: lowest in-flight prefill cost first.
        * ``min_waiting_reqs``: lowest scheduler waiting queue first.

        For every load-aware strategy, ties cascade to the same secondary keys
        — the alternate load metric and then ``effective_load`` (= metric
        ``num_running`` plus dispatches the proxy has issued since the last
        poll). That third key is what prevents whole bursts from piling onto
        ``i == 0`` during a single 200 ms metric-poll window.
        """
        strat = self.prefill_strategy
        if strat == "round_robin":
            return next(self._rr_cycle)
        if strat == "min_pending_prefill":
            key = lambda i: (  # noqa: E731
                self.stats[i].pending_prefill_tokens,
                self.stats[i].num_waiting,
                self.stats[i].effective_load,
                i,
            )
        else:  # min_waiting_reqs
            key = lambda i: (  # noqa: E731
                self.stats[i].num_waiting,
                self.stats[i].pending_prefill_tokens,
                self.stats[i].effective_load,
                i,
            )
        return min(range(len(self.stats)), key=key)

    def pick_decode(self, exclude: int | None = None) -> int:
        """Pick the backend with the lightest decode-side load.

        Score is ``effective_load + decode_kv_weight * kv_usage`` so that KV
        pressure is folded in alongside in-flight request count.

        ``exclude`` optionally suppresses a particular index. It is available
        as an escape hatch but currently unused by ``_handle_pd``: the
        post-prefill stage 2 re-pick deliberately allows the prefill node
        back in, because "P is genuinely lightest right now" is a valid
        outcome that we handle via the SAME-NODE-re-pick plain-forward path
        (prefix cache covers most of the duplicated prefill cost).
        """
        scores = [
            (
                s.effective_load + self.decode_kv_weight * s.kv_usage,
                s.num_waiting,
                idx,
            )
            for idx, s in enumerate(self.stats)
            if exclude is None or idx != exclude
        ]
        if not scores:
            # Single-backend case: nothing to exclude against.
            return exclude if exclude is not None else 0
        return min(scores)[2]

    # ---- token estimate ----
    def _estimate_prompt_tokens(self, req_data: dict) -> int:
        # If client pre-tokenized, trust that.
        ids = req_data.get("prompt_token_ids")
        if isinstance(ids, list) and ids:
            if ids and isinstance(ids[0], list):
                return sum(len(x) for x in ids)
            return len(ids)
        prompt = req_data.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "".join(p if isinstance(p, str) else "" for p in prompt)
        msgs = req_data.get("messages") or []
        for m in msgs:
            c = m.get("content")
            if isinstance(c, str):
                prompt += "\n" + c
            elif isinstance(c, list):
                for piece in c:
                    if isinstance(piece, dict) and isinstance(piece.get("text"), str):
                        prompt += "\n" + piece["text"]
        return max(1, len(prompt) // self.prompt_token_divisor)

    # ---- HTTP handlers ----
    async def healthcheck(self, _: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "backends": [
                    {
                        "host": s.host,
                        "port": s.port,
                        "num_running": s.num_running,
                        "num_waiting": s.num_waiting,
                        "kv_usage": s.kv_usage,
                        "pending_prefill_tokens": s.pending_prefill_tokens,
                        "pending_dispatches": s.pending_dispatches,
                        "effective_load": s.effective_load,
                        "decode_load_score": (
                            s.effective_load + self.decode_kv_weight * s.kv_usage
                        ),
                        "last_poll_ok": s.last_poll_ok,
                        "last_poll_error": s.last_poll_error,
                    }
                    for s in self.stats
                ],
                "config": {
                    "prefill_strategy": self.prefill_strategy,
                    "decode_kv_weight": self.decode_kv_weight,
                    "poll_interval_ms": int(self.poll_interval_s * 1000),
                },
                "served": self._request_counter,
                "decisions": {
                    "short_circuit": self._counter_short_circuit,
                    "split_same_node_repick": self._counter_split_same,
                    "split_cross_node": self._counter_split_cross,
                },
            }
        )

    async def handle_completions(self, request: web.Request) -> web.StreamResponse:
        return await self._handle_pd("/v1/completions", request)

    async def handle_chat_completions(
        self, request: web.Request
    ) -> web.StreamResponse:
        return await self._handle_pd("/v1/chat/completions", request)

    async def handle_other(self, request: web.Request) -> web.StreamResponse:
        # Anything that isn't a P/D split path: stick on the least-loaded backend.
        idx = self.pick_decode()
        return await self._forward_full(idx, request)

    async def _forward_full(
        self,
        idx: int,
        request: web.Request,
        payload_json: dict | None = None,
    ) -> web.StreamResponse:
        """Forward the unmodified request to ``idx`` and stream the response.

        If ``payload_json`` is provided (callers that have already parsed the
        body via ``request.json()``), use it directly instead of reading the
        request body again — re-reading after ``request.json()`` can return an
        empty payload depending on aiohttp's body caching behaviour, which
        breaks the upstream POST.
        """
        assert self._session is not None
        host, port = self.stats[idx].host, self.stats[idx].port
        target = f"http://{host}:{port}{request.path_qs}"
        headers = _filter_headers(request.headers)
        send_kwargs: dict
        if payload_json is not None:
            send_kwargs = {"json": payload_json}
        else:
            body = await request.read() if request.can_read_body else None
            send_kwargs = {"data": body}
        try:
            async with self._session.request(
                request.method,
                target,
                headers=headers,
                allow_redirects=False,
                **send_kwargs,
            ) as upstream:
                resp_headers = _filter_headers(upstream.headers)
                response = web.StreamResponse(
                    status=upstream.status, headers=resp_headers
                )
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    if chunk:
                        await response.write(chunk)
                await response.write_eof()
                return response
        except aiohttp.ClientError as exc:
            logger.error("forward to %s:%d failed: %s", host, port, exc)
            return web.json_response(
                {"error": {"message": f"Bad gateway: {exc}", "type": "proxy_error"}},
                status=502,
            )

    async def _handle_pd(
        self, api: str, request: web.Request
    ) -> web.StreamResponse:
        assert self._session is not None
        req_id = uuid.uuid4().hex
        self._request_counter += 1

        # Parse the JSON body up front; we always need it for P/D split decisions.
        try:
            req_data = await request.json()
        except Exception as e:
            return web.json_response(
                {"error": {"message": f"bad request: {e}", "type": "proxy_error"}},
                status=400,
            )

        # Stage 0: snapshot picks (decides short-circuit vs split).
        p_idx = self.pick_prefill()
        d_idx_initial = self.pick_decode()

        # SHORT-CIRCUIT: same backend wins both -> forward whole request.
        # HomogeneousScheduler on the backend handles internal P/D balance
        # (decode-first; prefill gets a smaller token budget).
        if p_idx == d_idx_initial:
            self._counter_short_circuit += 1
            s = self.stats[p_idx]
            logger.info(
                "[%s] short-circuit on backend[%d]=%s:%d "
                "(running=%.0f dispatched=%d waiting=%.0f kv=%.2f "
                "pending_p_tok=%d)",
                req_id[:8],
                p_idx,
                s.host,
                s.port,
                s.num_running,
                s.pending_dispatches,
                s.num_waiting,
                s.kv_usage,
                s.pending_prefill_tokens,
            )
            # Bump pending_dispatches so burst-arrival requests within one
            # metrics-poll window (default 200 ms) see this backend as more
            # loaded and shift to the idle peer. The next /metrics poll
            # zeros this out, so there is no double-counting once the request
            # shows up in vLLM's own num_requests_running gauge.
            self.stats[p_idx].pending_dispatches += 1
            return await self._forward_full(
                p_idx, request, payload_json=req_data
            )

        # SPLIT path
        prompt_tokens_est = self._estimate_prompt_tokens(req_data)
        self.stats[p_idx].pending_prefill_tokens += prompt_tokens_est

        # Build prefill sub-request body
        p_body = dict(req_data)
        p_body["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        p_body["stream"] = False
        p_body["max_tokens"] = 1
        if "max_completion_tokens" in p_body:
            p_body["max_completion_tokens"] = 1
        p_body.pop("stream_options", None)
        # These args aren't supported by the prefill stage
        min_tokens = p_body.pop("min_tokens", None)
        min_completion_tokens = p_body.pop("min_completion_tokens", None)

        p_host, p_port = self.stats[p_idx].host, self.stats[p_idx].port
        prefill_url = f"http://{p_host}:{p_port}{api}"
        try:
            async with self._session.post(
                prefill_url,
                json=p_body,
                headers={"X-Request-Id": req_id},
            ) as p_resp:
                if p_resp.status != 200:
                    body_text = await p_resp.text()
                    return web.json_response(
                        {
                            "error": {
                                "message": f"prefill backend {p_host}:{p_port} "
                                f"returned {p_resp.status}: {body_text[:400]}",
                                "type": "proxy_error",
                            }
                        },
                        status=502,
                    )
                p_json = await p_resp.json()
        except aiohttp.ClientError as e:
            self.stats[p_idx].pending_prefill_tokens = max(
                0,
                self.stats[p_idx].pending_prefill_tokens - prompt_tokens_est,
            )
            return web.json_response(
                {
                    "error": {
                        "message": f"prefill connection error: {e}",
                        "type": "proxy_error",
                    }
                },
                status=502,
            )
        finally:
            # Release the estimated prefill cost regardless of outcome
            self.stats[p_idx].pending_prefill_tokens = max(
                0,
                self.stats[p_idx].pending_prefill_tokens - prompt_tokens_est,
            )

        # Optional refinement using the real prompt_tokens (not used downstream
        # except by /proxy/healthcheck stats).
        usage = p_json.get("usage") or {}
        real_p_tokens = usage.get("prompt_tokens")
        if isinstance(real_p_tokens, int):
            logger.debug(
                "[%s] prefill prompt_tokens real=%d, est=%d",
                req_id[:8],
                real_p_tokens,
                prompt_tokens_est,
            )

        kv_transfer_params = p_json.get("kv_transfer_params") or {}
        if not kv_transfer_params:
            logger.warning(
                "[%s] prefill backend returned no kv_transfer_params; "
                "falling back to single-node decode on backend[%d]",
                req_id[:8],
                p_idx,
            )
            # Cannot do remote prefill -> just re-execute the request locally on
            # the prefill backend (kv_both mode will re-prefill + decode there).
            return await self._forward_full(p_idx, request, payload_json=req_data)

        # Stage 2: re-pick decode AFTER prefill, with the fresh load picture.
        d_idx = self.pick_decode()
        if d_idx == p_idx:
            # SAME-NODE re-pick: the engine cannot register *itself* as a
            # remote KV source (vLLM asserts engine_id != remote_engine_id and
            # the EngineCore dies if we try). Skip the kv_transfer hand-off
            # and just re-issue the original request to that backend. The
            # local prefix cache absorbs most of the duplicated prefill cost.
            self._counter_split_same += 1
            logger.info(
                "[%s] split: P=backend[%d]=%s:%d D=backend[%d]=%s:%d "
                "SAME-NODE re-pick -> plain forward (no kv_transfer), "
                "real_prompt_tokens=%s",
                req_id[:8],
                p_idx,
                p_host,
                p_port,
                d_idx,
                self.stats[d_idx].host,
                self.stats[d_idx].port,
                real_p_tokens,
            )
            return await self._forward_full(d_idx, request, payload_json=req_data)

        self._counter_split_cross += 1

        d_body = dict(req_data)
        d_body["kv_transfer_params"] = kv_transfer_params
        if min_tokens is not None:
            d_body["min_tokens"] = min_tokens
        if min_completion_tokens is not None:
            d_body["min_completion_tokens"] = min_completion_tokens

        d_host, d_port = self.stats[d_idx].host, self.stats[d_idx].port
        decode_url = f"http://{d_host}:{d_port}{api}"
        is_stream = bool(req_data.get("stream"))

        logger.info(
            "[%s] split: P=backend[%d]=%s:%d D=backend[%d]=%s:%d "
            "CROSS-NODE (real_prompt_tokens=%s)",
            req_id[:8],
            p_idx,
            p_host,
            p_port,
            d_idx,
            d_host,
            d_port,
            real_p_tokens,
        )

        # Account for this dispatch so concurrent picks see d_idx as busier;
        # cleared on the next metric poll.
        self.stats[d_idx].pending_dispatches += 1
        try:
            async with self._session.post(
                decode_url,
                json=d_body,
                headers={"X-Request-Id": req_id},
            ) as d_resp:
                response = web.StreamResponse(
                    status=d_resp.status,
                    headers=_filter_headers(d_resp.headers),
                )
                await response.prepare(request)
                if is_stream:
                    async for chunk in d_resp.content.iter_any():
                        if chunk:
                            await response.write(chunk)
                else:
                    # Non-streaming: drain whole body, then write once
                    body = await d_resp.read()
                    await response.write(body)
                await response.write_eof()
                return response
        except aiohttp.ClientError as e:
            return web.json_response(
                {
                    "error": {
                        "message": f"decode connection error: {e}",
                        "type": "proxy_error",
                    }
                },
                status=502,
            )


# ---------- argparse / wiring -----------------------------------------------


def _env_or_default(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val else default


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n", 1)[0])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--backend-hosts",
        nargs="+",
        required=True,
    )
    p.add_argument(
        "--backend-ports",
        nargs="+",
        type=int,
        required=True,
    )
    p.add_argument(
        "--prefill-strategy",
        default=_env_or_default("PREFILL_STRATEGY", "min_pending_prefill"),
        choices=("min_pending_prefill", "min_waiting_reqs", "round_robin"),
    )
    p.add_argument(
        "--decode-kv-weight",
        type=float,
        default=float(_env_or_default("DECODE_KV_WEIGHT", "10.0")),
    )
    p.add_argument(
        "--poll-interval-ms",
        type=int,
        default=int(_env_or_default("POLL_INTERVAL_MS", "200")),
    )
    p.add_argument(
        "--prompt-token-divisor",
        type=int,
        default=int(_env_or_default("PROMPT_TOKEN_DIVISOR", "3")),
    )
    p.add_argument(
        "--log-level",
        default=_env_or_default("LOG_LEVEL", "INFO"),
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = p.parse_args(argv)
    if len(args.backend_hosts) != len(args.backend_ports):
        p.error("--backend-hosts and --backend-ports must have the same length")
    return args


def build_app(proxy: AdaptivePDProxy) -> web.Application:
    app = web.Application(client_max_size=0)
    app.on_startup.append(proxy.startup)
    app.on_cleanup.append(proxy.cleanup)
    app.router.add_get("/proxy/healthcheck", proxy.healthcheck)
    app.router.add_post("/v1/completions", proxy.handle_completions)
    app.router.add_post("/v1/chat/completions", proxy.handle_chat_completions)
    app.router.add_route("*", "/{tail:.*}", proxy.handle_other)
    return app


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    backends = list(zip(args.backend_hosts, args.backend_ports, strict=True))
    proxy = AdaptivePDProxy(
        backends=backends,
        prefill_strategy=args.prefill_strategy,
        decode_kv_weight=args.decode_kv_weight,
        poll_interval_ms=args.poll_interval_ms,
        prompt_token_divisor=args.prompt_token_divisor,
    )
    app = build_app(proxy)
    logger.info("Listening on http://%s:%d", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":  # pragma: no cover
    with contextlib.suppress(KeyboardInterrupt):
        main()
