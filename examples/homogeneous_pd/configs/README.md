# Topology configuration files

A *topology config* is a small bash file (sourced by the launchers) that
describes:

- which IPs / ports / GPUs make up the cluster,
- whether the cluster is **PD-disaggregated** (`TYPE=pd_disagg`) or
  **homogeneous PD** (`TYPE=homogeneous`),
- the model and homogeneous-scheduler knobs (`ALPHA`, `BUDGET_FN`, ...),
- where the router listens (`ROUTER_HOST`, `ROUTER_PORT`).

Pick one of the four `*.example.conf` files in this directory, copy it to
`<my_setup>.conf`, edit the IPs / GPUs / ports for your hardware, then:

```bash
# On each node host (with NODE_IDX = the row this machine should run):
./launchers/launch_node.sh configs/<my_setup>.conf $NODE_IDX

# On the router host:
./launchers/launch_router.sh configs/<my_setup>.conf
```

## Required arrays (all configs)

| Variable                    | Type            | Meaning                                              |
| --------------------------- | --------------- | ---------------------------------------------------- |
| `TYPE`                      | string          | `pd_disagg` or `homogeneous`.                        |
| `ROLES`                     | array of string | Per-node `kv_role` (`kv_producer` / `kv_consumer` / `kv_both`). |
| `NODE_HOSTS`                | array of string | Per-node IP / hostname (must be reachable from the router). |
| `NODE_PORTS`                | array of int    | Per-node vLLM serve port.                            |
| `NODE_GPUS`                 | array of string | Per-node `CUDA_VISIBLE_DEVICES` value.               |
| `NODE_SIDE_CHANNEL_PORTS`   | array of int    | Per-node `VLLM_NIXL_SIDE_CHANNEL_PORT`.              |
| `MODEL`                     | string          | HuggingFace repo or local path.                      |
| `ROUTER_HOST` / `ROUTER_PORT` | string / int  | Where `launch_router.sh` binds.                      |

All five `NODE_*` arrays plus `ROLES` MUST have the same length; index `i`
describes the same node across all of them.

## Homogeneous-only knobs

| Variable          | Default        | Meaning                                                              |
| ----------------- | -------------- | -------------------------------------------------------------------- |
| `ALPHA`           | `16`           | Slope of the default `linear_ratio` budget function.                 |
| `BUDGET_FN`       | `linear_ratio` | Name registered in `homogeneous_pd/scheduler/budget_fn.py`.          |
| `PREFILL_MIN`     | `0`            | Floor on per-step prefill tokens (0 = no floor).                     |
| `PREFILL_MAX`     | `0`            | Ceiling on per-step prefill tokens (0 = no ceiling).                 |

## Optional global knobs (defaults from `launchers/common.sh`)

`BLOCK_SIZE`, `MAX_MODEL_LEN`, `MAX_NUM_BATCHED_TOKENS`, `MAX_NUM_SEQS`,
`GPU_MEMORY_UTILIZATION`, `KV_BUFFER_DEVICE`, `LOG_DIR`. Set them in the
config to override.
