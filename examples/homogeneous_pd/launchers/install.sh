#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Installer for the homogeneous_pd experiment.
#
# This script intentionally uses ONLY stdlib `python -m venv` and `pip`, so
# every package fetch goes through whatever you have configured in
# ~/.pip/pip.conf / /etc/pip.conf (or the PIP_INDEX_URL env var). Nothing
# else is downloaded from the public internet: no uv installer from
# astral.sh, no precompiled wheel from wheels.vllm.ai, no extra PyTorch
# index, no python-build-standalone tarball.
#
# Packages installed via pip (each of these must exist on your mirror):
#
#   * vllm                                  -> the engine itself
#   * aiohttp, httpx, fastapi,
#     uvicorn[standard], pytest             -> proxy + test deps
#   * flashinfer-python   (best-effort)     -> only needed for ATTENTION_BACKEND=FLASHINFER
#   * nixl                (best-effort)     -> required for actual cross-node KV transfer
#
# Usage:
#   ./install.sh                     # full install (everything above)
#   SKIP_VLLM=1        ./install.sh  # only proxy deps + nixl + flashinfer
#   SKIP_DEPS=1        ./install.sh  # skip aiohttp/httpx/fastapi/uvicorn/pytest
#   SKIP_NIXL=1        ./install.sh  # skip nixl
#   SKIP_FLASHINFER=1  ./install.sh  # skip flashinfer
#   PYTHON_BIN=python3.12 ./install.sh   # pick a specific system Python interpreter
#
# Notes
# -----
# 1. We do NOT install vLLM in editable mode (`pip install -e .`) because
#    that path needs either the CUDA toolkit to compile or VLLM_USE_PRECOMPILED
#    which fetches from wheels.vllm.ai. Plain `pip install vllm` instead
#    gets a CUDA-ready wheel from your configured mirror, and the local
#    homogeneous_pd/ scheduler+proxy modules are loaded via PYTHONPATH
#    (see launchers/common.sh) so the on-disk source still drives behaviour.
# 2. If your mirror does not carry `flashinfer-python` or `nixl`, the
#    script prints a warning and continues; install them manually later.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
EXAMPLES_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

PYTHON_BIN=${PYTHON_BIN:-python3}
SKIP_VLLM=${SKIP_VLLM:-0}
SKIP_DEPS=${SKIP_DEPS:-0}
SKIP_NIXL=${SKIP_NIXL:-0}
SKIP_FLASHINFER=${SKIP_FLASHINFER:-0}

step() { printf '\n==> %s\n' "$*"; }

step "Repo root  : ${REPO_ROOT}"
step "Python bin : $(command -v "${PYTHON_BIN}" || echo '<missing>')"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "${PYTHON_BIN} not found. Install Python 3.10+ via your apt repo and retry," >&2
  echo "or set PYTHON_BIN to a working interpreter (e.g. PYTHON_BIN=python3.12)." >&2
  exit 1
fi

step "Python version: $("${PYTHON_BIN}" --version)"

# ---------------------------------------------------------------------------
# 1. Create a venv (stdlib only; no network).
# ---------------------------------------------------------------------------
VENV_DIR=${VENV_DIR:-${REPO_ROOT}/.venv}
if [[ ! -d "${VENV_DIR}" ]]; then
  step "Creating virtualenv at ${VENV_DIR} ..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  step "Re-using existing virtualenv at ${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
step "Active interpreter: $(command -v python) ($(python --version))"

# Print the pip index URL the user has configured so misconfiguration is
# obvious from the install log.
step "pip config (effective):"
python -m pip config list 2>/dev/null || true

# ---------------------------------------------------------------------------
# 2. Upgrade pip itself (still goes through your mirror).
# ---------------------------------------------------------------------------
step "Upgrading pip / setuptools / wheel ..."
python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 3. vLLM (from whatever your pip mirror has tagged as `vllm`).
# ---------------------------------------------------------------------------
if [[ "${SKIP_VLLM}" != "1" ]]; then
  step "Installing vllm ..."
  python -m pip install vllm
fi

# ---------------------------------------------------------------------------
# 4. Proxy + test dependencies.
# ---------------------------------------------------------------------------
if [[ "${SKIP_DEPS}" != "1" ]]; then
  step "Installing proxy + test dependencies ..."
  python -m pip install \
    aiohttp \
    httpx \
    fastapi \
    "uvicorn[standard]" \
    pytest
fi

# ---------------------------------------------------------------------------
# 5. FlashInfer (only needed for ATTENTION_BACKEND=FLASHINFER).
# ---------------------------------------------------------------------------
if [[ "${SKIP_FLASHINFER}" != "1" ]]; then
  step "Installing flashinfer-python (best-effort) ..."
  if python -m pip install flashinfer-python; then
    echo "    flashinfer-python installed."
  else
    cat <<'EOF'

WARNING: pip install flashinfer-python failed (probably missing from the
configured pip mirror). This is fine if you do NOT plan to use
ATTENTION_BACKEND=FLASHINFER. Otherwise ask your mirror admin to mirror the
flashinfer-python wheel for your CUDA version.
EOF
  fi
fi

# ---------------------------------------------------------------------------
# 6. NIXL (required for cross-node KV transfer).
# ---------------------------------------------------------------------------
if [[ "${SKIP_NIXL}" != "1" ]]; then
  step "Installing nixl (best-effort) ..."
  if python -m pip install nixl; then
    echo "    nixl installed."
  else
    cat <<'EOF'

WARNING: pip install nixl failed (probably missing from the configured pip
mirror). NIXL is required for KV-cache transfer between vLLM instances.
Either ask your mirror admin to mirror the `nixl` wheel, or build it from
source from a checkout of ai-dynamo/nixl that you have synced into your
environment manually.

CPU-only scheduler unit tests (`pytest examples/homogeneous_pd/tests`) do
not need nixl.
EOF
  fi
fi

# ---------------------------------------------------------------------------
# 7. Smoke test: scheduler import + budget function.
# ---------------------------------------------------------------------------
step "Verifying scheduler import ..."
PYTHONPATH="${EXAMPLES_DIR}:${PYTHONPATH:-}" python - <<'PY'
from homogeneous_pd.scheduler.budget_fn import linear_ratio
from homogeneous_pd.scheduler.homogeneous_scheduler import HomogeneousScheduler  # noqa: F401
assert linear_ratio(decode_tokens=4, alpha=16, max_batched=8192) == 64
print("HomogeneousScheduler import OK ; linear_ratio(4,16,8192)=64")
PY

# ---------------------------------------------------------------------------
# 8. Next steps.
# ---------------------------------------------------------------------------
cat <<EOF

============================================================
Install complete.

Activate the virtualenv in any new shell with:
  source ${VENV_DIR}/bin/activate

Run the CPU-only unit tests:
  python -m pytest examples/homogeneous_pd/tests -v

Distributed deployment (run on each machine):
  1. Configure cluster networking once per host:
       cp examples/homogeneous_pd/configs/cluster_env.example.sh \\
          examples/homogeneous_pd/configs/cluster_env.sh
       \$EDITOR examples/homogeneous_pd/configs/cluster_env.sh
  2. Copy and edit a topology config:
       cp examples/homogeneous_pd/configs/homogeneous_2node.example.conf my.conf
       \$EDITOR my.conf      # set NODE_HOSTS / NODE_PORTS / NODE_GPUS
  3. On every node host:
       examples/homogeneous_pd/launchers/launch_node.sh my.conf <NODE_INDEX>
  4. On the router host:
       examples/homogeneous_pd/launchers/launch_router.sh my.conf
============================================================
EOF
