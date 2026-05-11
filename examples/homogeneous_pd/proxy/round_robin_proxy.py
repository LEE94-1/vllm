# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming round-robin OpenAI-compatible proxy.

Used in the homogeneous-PD setup: every backend node serves both prefill and
decode in the same forward pass (via :class:`HomogeneousScheduler`), so a
request only needs to be sent to one node end-to-end. The proxy distributes
requests across N homogeneous backends in round-robin fashion.

Run::

    python -m homogeneous_pd.proxy.round_robin_proxy \
        --port 8000 \
        --backend-hosts 127.0.0.1 127.0.0.1 \
        --backend-ports 8101 8102

The proxy forwards every path verbatim (``/v1/completions``,
``/v1/chat/completions``, ``/v1/models``, ``/health``, ...) and streams the
response body back to the client without buffering, so SSE / OpenAI streaming
responses pass through correctly.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import logging
import sys

import aiohttp
from aiohttp import web

logger = logging.getLogger("homogeneous_pd.proxy")

# Hop-by-hop headers must not be forwarded (RFC 7230, section 6.1).
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


class RoundRobinProxy:
    def __init__(self, backends: list[tuple[str, int]]) -> None:
        if not backends:
            raise ValueError("RoundRobinProxy requires at least one backend")
        self.backends = backends
        self._cycle = itertools.cycle(range(len(backends)))
        self._session: aiohttp.ClientSession | None = None
        self._counter = 0

    async def startup(self, _: web.Application) -> None:
        # No total timeout: long generations may take many minutes.
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30),
            connector=aiohttp.TCPConnector(limit=0, force_close=False),
        )
        logger.info(
            "Proxy ready with %d backend(s): %s", len(self.backends), self.backends
        )

    async def cleanup(self, _: web.Application) -> None:
        if self._session is not None:
            await self._session.close()

    def _pick(self) -> tuple[int, tuple[str, int]]:
        idx = next(self._cycle)
        self._counter += 1
        return idx, self.backends[idx]

    async def handle(self, request: web.Request) -> web.StreamResponse:
        assert self._session is not None
        idx, (host, port) = self._pick()
        target = f"http://{host}:{port}{request.path_qs}"
        headers = _filter_headers(request.headers)
        body = await request.read() if request.can_read_body else None

        logger.debug(
            "request #%d -> backend[%d]=%s:%d %s %s",
            self._counter,
            idx,
            host,
            port,
            request.method,
            request.path_qs,
        )

        try:
            async with self._session.request(
                request.method,
                target,
                headers=headers,
                data=body,
                allow_redirects=False,
            ) as upstream:
                resp_headers = _filter_headers(upstream.headers)
                response = web.StreamResponse(
                    status=upstream.status, headers=resp_headers
                )
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    if not chunk:
                        continue
                    await response.write(chunk)
                await response.write_eof()
                return response
        except aiohttp.ClientError as exc:
            logger.error("upstream error talking to %s:%d: %s", host, port, exc)
            return web.json_response(
                {"error": {"message": f"Bad gateway: {exc}", "type": "proxy_error"}},
                status=502,
            )

    async def healthcheck(self, _: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "backends": [{"host": h, "port": p} for h, p in self.backends],
                "served": self._counter,
            }
        )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n", 1)[0])
    p.add_argument("--host", default="0.0.0.0", help="Address to bind the proxy on.")
    p.add_argument("--port", type=int, default=8000, help="Port to bind the proxy on.")
    p.add_argument(
        "--backend-hosts",
        nargs="+",
        required=True,
        help="Backend hostnames, paired positionally with --backend-ports.",
    )
    p.add_argument(
        "--backend-ports",
        nargs="+",
        type=int,
        required=True,
        help="Backend ports, one per host.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = p.parse_args(argv)
    if len(args.backend_hosts) != len(args.backend_ports):
        p.error("--backend-hosts and --backend-ports must have the same length")
    return args


def build_app(backends: list[tuple[str, int]]) -> web.Application:
    proxy = RoundRobinProxy(backends)
    app = web.Application(client_max_size=0)
    app.on_startup.append(proxy.startup)
    app.on_cleanup.append(proxy.cleanup)
    app.router.add_get("/proxy/healthcheck", proxy.healthcheck)
    app.router.add_route("*", "/{tail:.*}", proxy.handle)
    return app


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    backends = list(zip(args.backend_hosts, args.backend_ports, strict=True))
    app = build_app(backends)
    logger.info("Listening on http://%s:%d", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":  # pragma: no cover
    with contextlib.suppress(KeyboardInterrupt):
        main()
