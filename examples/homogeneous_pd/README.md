# Homogeneous PD vs PD-Disaggregation Experiment

A self-contained package for comparing two serving topologies on the same
hardware:

- **PD-disaggregated (baseline)** — separate prefill and decode instances
  (`kv_role=kv_producer` / `kv_role=kv_consumer`) connected by NIXL.
  Configurations: **1P1D** and **3P1D**.
- **Homogeneous PD (experimental)** — every node serves both prefill and
  decode (`kv_role=kv_both`) with a custom decode-first scheduler. KV cache
  is mutually transferable between any pair of nodes.
  Configurations: **2 nodes** (matched to 1P1D) and **4 nodes** (matched to
  3P1D).

Nothing in the upstream `vllm/` source tree is modified; the new scheduler is
loaded via the public `--scheduler-cls` flag.

## Layout

```
examples/homogeneous_pd/
├── README.md                              (this file)
├── configs/                               topology configs (per-experiment)
│   ├── README.md
│   ├── 1p1d.example.conf
│   ├── 3p1d.example.conf
│   ├── homogeneous_2node.example.conf
│   └── homogeneous_4node.example.conf
├── scheduler/
│   ├── homogeneous_scheduler.py           decode-first scheduler
│   └── budget_fn.py                       pluggable prefill-budget functions
├── proxy/
│   └── round_robin_proxy.py               OpenAI-compatible round-robin proxy
├── launchers/
│   ├── install.sh                         one-click installer
│   ├── common.sh                          shared bash helpers + defaults
│   ├── launch_node.sh                     start ONE node from a config
│   ├── launch_router.sh                   start the router from a config
│   ├── launch_pd_disagg.sh                single-machine PD-disagg launcher
│   ├── launch_homogeneous.sh              single-machine homogeneous launcher
│   ├── run_1p1d.sh                        1P + 1D wrapper (single machine)
│   ├── run_3p1d.sh                        3P + 1D wrapper (single machine)
│   ├── run_homogeneous_2node.sh           2-node homogeneous wrapper
│   ├── run_homogeneous_4node.sh           4-node homogeneous wrapper
│   └── stop_all.sh                        best-effort cleanup
└── tests/
    └── test_homogeneous_scheduler.py      CPU-only pytest suite
```

## Installation

One command on every machine that will run a node and on the router host:

```bash
./examples/homogeneous_pd/launchers/install.sh
```

What it does:

1. Installs `uv` (if missing).
2. Creates `.venv` at the repo root with Python 3.12.
3. Installs vLLM via the precompiled wheel (`VLLM_USE_PRECOMPILED=1`).
4. Installs `aiohttp` / `httpx` / `fastapi` / `uvicorn[standard]` / `pytest`.
5. Best-effort `pip install nixl` (required for cross-node KV transfer).
6. Smoke-tests the scheduler import.

Useful flags:

```bash
PYTHON_VERSION=3.11 ./install.sh    # different Python
SKIP_VLLM=1        ./install.sh    # only proxy deps + nixl
SKIP_NIXL=1        ./install.sh    # skip nixl (install manually later)
SKIP_DEPS=1        ./install.sh    # skip aiohttp/httpx/fastapi/uvicorn/pytest
```

If your distribution requires a system-level NIXL build, follow
[ai-dynamo/nixl](https://github.com/ai-dynamo/nixl) instead and re-run with
`SKIP_NIXL=1`.

After install, mark the launchers executable on the server:

```bash
chmod +x examples/homogeneous_pd/launchers/*.sh
```

## How the homogeneous scheduler works

`HomogeneousScheduler` subclasses `vllm.v1.core.sched.scheduler.Scheduler`.
On every step it does:

1. Walk `self.running` and conservatively estimate the number of decode
   tokens that the base scheduler will issue (skipping requests that are
   still in chunked-prefill).
2. Compute the per-step prefill budget via a pluggable function:

   ```
   prefill_budget = budget_fn(num_decode_tokens, alpha=…, max_batched=…)
   ```

   The default function `linear_ratio` returns `num_decode_tokens * alpha`,
   clamped to the remaining batch.
3. Temporarily set `self.max_num_scheduled_tokens` to
   `min(num_decode_tokens + prefill_budget, max_num_batched_tokens)` and
   delegate to the base scheduler. Because the base loop already iterates
   RUNNING (decode + chunked-prefill continuations) before WAITING (new
   prefill), the clamp guarantees room for all decodes first and then gives
   prefill the remainder up to the budget.

When there are no decodes (cold start), the clamp is bypassed so prefill can
ramp up unrestricted.

### Configuration knobs

All four are read from `kv_connector_extra_config` first, then from the
matching `HOMOGENEOUS_*` environment variable, and finally fall back to a
default:

| Key                         | Env var                  | Default        | Meaning                                                      |
| --------------------------- | ------------------------ | -------------- | ------------------------------------------------------------ |
| `homogeneous_alpha`         | `HOMOGENEOUS_ALPHA`      | `16.0`         | Slope of `linear_ratio`. Default matches default page size.  |
| `homogeneous_budget_fn`     | `HOMOGENEOUS_BUDGET_FN`  | `linear_ratio` | Name registered in `budget_fn._REGISTRY`.                    |
| `homogeneous_prefill_min`   | `HOMOGENEOUS_PREFILL_MIN`| `0`            | Floor on per-step prefill tokens (0 = no floor).             |
| `homogeneous_prefill_max`   | `HOMOGENEOUS_PREFILL_MAX`| `0`            | Ceiling on per-step prefill tokens (0 = no ceiling).         |

The launchers bake the values into `kv_connector_extra_config` so they are
respected even if the launcher process does not also export them.

### Adding a new budget function

```python
# my_module.py
from homogeneous_pd.scheduler import register_budget_fn

def quadratic(*, decode_tokens, alpha, max_batched, **_):
    raw = int(decode_tokens * decode_tokens * alpha)
    return max(0, min(raw, max_batched - decode_tokens))

register_budget_fn("quadratic", quadratic)
```

Then start the scheduler with:

```
HOMOGENEOUS_BUDGET_FN=quadratic ...
```

(Make sure `my_module` is imported before `vllm serve` boots — easiest is to
place a top-level `import my_module` next to the launcher.)

## Running an experiment

There are two deployment modes. The **distributed** mode is the
production-style path you will use when nodes live on different machines and
the router is colocated with neither. The **single-machine** mode bundles
everything into one launcher for quick local checks.

### Mode 1: Distributed (one machine per node + a router)

1. Pick the topology you want and copy its example config:

   ```bash
   cd examples/homogeneous_pd
   cp configs/homogeneous_2node.example.conf my.conf
   $EDITOR my.conf      # edit NODE_HOSTS, NODE_PORTS, NODE_GPUS for your hardware
   ```

   The four shipped templates correspond to the four planned setups:

   | Config file                                 | Topology              |
   | ------------------------------------------- | --------------------- |
   | `configs/1p1d.example.conf`                 | PD-disagg, 1P + 1D    |
   | `configs/3p1d.example.conf`                 | PD-disagg, 3P + 1D    |
   | `configs/homogeneous_2node.example.conf`    | Homogeneous, 2 nodes  |
   | `configs/homogeneous_4node.example.conf`    | Homogeneous, 4 nodes  |

   See `configs/README.md` for the full schema.

2. On every node host (with the index that matches that machine's row):

   ```bash
   ./launchers/launch_node.sh my.conf 0      # on NODE_HOSTS[0]
   ./launchers/launch_node.sh my.conf 1      # on NODE_HOSTS[1]
   ./launchers/launch_node.sh my.conf 2      # on NODE_HOSTS[2] (3p1d, 4-node only)
   ./launchers/launch_node.sh my.conf 3      # on NODE_HOSTS[3] (3p1d, 4-node only)
   ```

   `vllm serve` runs in the foreground; logs are also tee'd to
   `${LOG_DIR}/node_<i>.log`. Press Ctrl-C to stop the node.

3. On the router host (any machine with HTTP reachability to all nodes):

   ```bash
   ./launchers/launch_router.sh my.conf
   ```

   - For `TYPE=pd_disagg`, the router is the upstream NIXL toy proxy
     (`tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`); it
     sends every request to a prefill node first (with `max_tokens=1`) and
     then forwards to a decode node.
   - For `TYPE=homogeneous`, the router is
     `homogeneous_pd.proxy.round_robin_proxy`, a stateless OpenAI-compatible
     reverse proxy that round-robins requests across all backends.

   The router binds `${ROUTER_HOST}:${ROUTER_PORT}` (defaults `0.0.0.0:8000`).

4. Send your benchmark traffic to `http://<router-host>:${ROUTER_PORT}`.

### Mode 2: Single machine (everything on one box)

For quick sanity checks on a single multi-GPU box. Each launcher spawns the
nodes, waits for them to boot, and starts a colocated proxy:

```bash
cd examples/homogeneous_pd/launchers

MODEL=meta-llama/Llama-3.1-8B-Instruct ./run_1p1d.sh                 # baseline A1
MODEL=meta-llama/Llama-3.1-8B-Instruct ./run_3p1d.sh                 # baseline A2
MODEL=meta-llama/Llama-3.1-8B-Instruct ALPHA=16 ./run_homogeneous_2node.sh
MODEL=meta-llama/Llama-3.1-8B-Instruct ALPHA=16 ./run_homogeneous_4node.sh
```

Each launcher prints the proxy URL on stdout (default
`http://127.0.0.1:8000`) and blocks until you Ctrl-C it. On exit, all child
processes are killed automatically. If a launcher dies uncleanly, run
`./stop_all.sh` from the same directory.

### Common environment overrides

Defined and documented in `launchers/common.sh`. Either export them before
running, or set them inside your topology config:

| Variable                  | Default                                  |
| ------------------------- | ---------------------------------------- |
| `MODEL`                   | `meta-llama/Llama-3.1-8B-Instruct`       |
| `BLOCK_SIZE`              | `16`                                     |
| `MAX_MODEL_LEN`           | `16384`                                  |
| `MAX_NUM_BATCHED_TOKENS`  | `8192`                                   |
| `MAX_NUM_SEQS`            | `256`                                    |
| `GPU_MEMORY_UTILIZATION`  | `0.85`                                   |
| `KV_BUFFER_DEVICE`        | `cuda`                                   |
| `ALPHA` (homogeneous)     | `16`                                     |
| `BUDGET_FN` (homogeneous) | `linear_ratio`                           |
| `PROXY_PORT` / `ROUTER_PORT` | `8000`                                |
| `LOG_DIR`                 | `/tmp/homogeneous_pd_logs`               |

For a fair comparison, keep `MODEL`, `BLOCK_SIZE`, `MAX_MODEL_LEN`,
`MAX_NUM_BATCHED_TOKENS`, `MAX_NUM_SEQS`, and `GPU_MEMORY_UTILIZATION`
identical across the four configurations.

### Plugging in your benchmark

Both routers expose the standard OpenAI HTTP API. Any benchmark that targets
that URL works out of the box — for example:

```bash
vllm bench serve \
    --base-url http://<router-host>:8000 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name random \
    --random-input-len 4096 --random-output-len 256 \
    --num-prompts 500 --request-rate 8
```

### Health checks

```bash
# Backends individually
curl http://<node-host>:8100/v1/models

# Round-robin router (homogeneous mode only)
curl http://<router-host>:8000/proxy/healthcheck
```

## Running the tests

The CPU-only unit tests exercise the scheduler logic directly:

```bash
.venv/bin/python -m pytest examples/homogeneous_pd/tests -v
```

What they cover:

- `linear_ratio` / `fixed_cap` arithmetic and clamping.
- Config resolution from `kv_connector_extra_config`.
- "No decodes" cold start uses the full budget.
- Pure decode steps schedule exactly one token per request.
- `alpha=16` with 3 decodes schedules exactly 48 prefill tokens.
- `alpha=0` with running decodes admits zero prefill.
- A very large `alpha` reproduces stock chunked-prefill behaviour.
- `prefill_min` / `prefill_max` floor/ceiling apply correctly.
- The total scheduled tokens never exceed `max_num_batched_tokens`.

These run in a few seconds without GPU, network, or external services.

## What is *not* included on purpose

- **Workload generators / plotting**: deferred to your existing benchmark
  pipeline (the routers speak plain OpenAI HTTP).
- **Cross-node prefix-aware routing**: the round-robin router is intentionally
  stateless. Cross-node KV transfer still happens whenever the underlying
  NIXL connector decides to (e.g. on prefix-cache hits) because all backends
  run with `kv_role=kv_both`. A smarter router can be added later without
  touching the scheduler.
- **Async scheduling support**: `HomogeneousScheduler` inherits from the
  synchronous `Scheduler`. For a clean apples-to-apples comparison run both
  topologies in synchronous mode (`async_scheduling=False`).

## Troubleshooting

| Symptom                                                     | Likely cause / fix                                                                  |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `ModuleNotFoundError: homogeneous_pd`                       | The launcher prepends `examples/` to `PYTHONPATH` automatically; run launchers via their script paths (do not `cd` inside `homogeneous_pd/scheduler/` and re-run). |
| `Unknown homogeneous prefill budget function: ...`          | Typo in `BUDGET_FN`; valid names are listed in `scheduler/budget_fn.py`.            |
| Node hangs on startup                                       | Tail `${LOG_DIR}/node_<i>.log`. NIXL needs each instance's `VLLM_NIXL_SIDE_CHANNEL_PORT` (`NODE_SIDE_CHANNEL_PORTS[i]`) to be free; bump it if it collides. |
| Router returns 502 immediately                              | A backend died — `tail -F ${LOG_DIR}/*.log`.                                        |
| `pip install nixl` failed                                   | Install nixl from source (see [ai-dynamo/nixl](https://github.com/ai-dynamo/nixl)) and re-run install with `SKIP_NIXL=1`. |
| Same GPU is used by every instance                          | Edit `NODE_GPUS` in your topology config so each row has a distinct `CUDA_VISIBLE_DEVICES` value. |
