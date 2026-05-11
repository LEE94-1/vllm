# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# shellcheck shell=bash
#
# Cluster networking environment. Sourced automatically by
# launchers/common.sh when present at:
#
#   ${HOMOGENEOUS_DIR}/configs/cluster_env.sh
#
# Copy this file once per cluster (NOT per experiment) and tune the device
# names to match your NICs:
#
#   cp configs/cluster_env.example.sh configs/cluster_env.sh
#   $EDITOR configs/cluster_env.sh
#
# The variables below cover three subsystems used at runtime:
#
#   * NCCL  鈥?tensor-parallel allreduce inside each vLLM instance.
#   * GLOO  鈥?torch RPC initialisation.
#   * UCX   鈥?transport used by NIXL for KV-cache transfer (intra- and
#             inter-node). UCX_NET_DEVICES is what makes NIXL pick the
#             RDMA NICs you want.
#
# Example below is for an A800 box with 4x ConnectX RDMA NICs
# (mlx5_2, mlx5_3, mlx5_6, mlx5_7) and an Ethernet management interface
# (ens22f0).

# ---- Generic RDMA / verbs ----------------------------------------------------
export RDMAV_HARDWARE_MODE=${RDMAV_HARDWARE_MODE:-0}

# ---- NCCL / GLOO ------------------------------------------------------------
# Pick the Ethernet interface that the head/router host can reach.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens22f0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ens22f0}

# IB HCAs used by NCCL for the *intra-node* TP collective (only matters if
# TP collective spills onto IB, e.g. multi-node TP 鈥?keep it consistent
# with UCX_NET_DEVICES below for predictability).
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_2,mlx5_3,mlx5_6,mlx5_7}

# ---- UCX (NIXL transport) ---------------------------------------------------
# UCX_NET_DEVICES selects which RDMA devices NIXL is allowed to use. Format
# is "device:port,device:port,...". Use port 1 on each HCA unless your
# hardware is wired differently.
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-mlx5_2:1,mlx5_3:1,mlx5_6:1,mlx5_7:1}

# Transports: prefer RDMA (`rc`) + CUDA copy + CUDA IPC; fall back to TCP
# only if explicitly desired. Drop `cuda_ipc` if your GPUs are not
# peer-accessible (rare on a single SXM box, common on a multi-socket PCIe
# box without PCIe-P2P).
export UCX_TLS=${UCX_TLS:-rc,cuda_copy,cuda_ipc}

# Memory types UCX understands. NIXL transfers CUDA tensors, so we must
# enable the cuda registration cache.
export UCX_MEMTYPE_CACHE=${UCX_MEMTYPE_CACHE:-n}
export UCX_CUDA_COPY_ASYNC_MEM_TYPE=${UCX_CUDA_COPY_ASYNC_MEM_TYPE:-cuda}

# ---- Convenience aliases used by some Mooncake-aware code paths -------------
# Harmless even when not using Mooncake; useful for parity with sglang
# scripts that target the same NICs.
export IB_DEVICE_LIST=${IB_DEVICE_LIST:-${NCCL_IB_HCA}}
