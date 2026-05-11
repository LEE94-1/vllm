#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# One-click installer for the homogeneous_pd experiment.
#
#   curl + uv  ->  .venv (Python 3.12)  ->  vLLM (precompiled)
#                  + proxy deps (aiohttp / httpx / fastapi / uvicorn / pytest)
#                  + nixl  (best-effort; required for cross-node KV transfer)
#
# Usage:
#   ./install.sh                     # full install
#   SKIP_VLLM=1 ./install.sh         # only proxy deps + nixl
#   SKIP_NIXL=1 ./install.sh         # skip nixl install (manual later)
#   PYTHON_VERSION=3.11 ./install.sh # pick a different Python
#
# The script must be run from any working directory; it locates the repo root
# relative to itself.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
EXAMPLES_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

PYTHON_VERSION=${PYTHON_VERSION:-3.12}
SKIP_VLLM=${SKIP_VLLM:-0}
SKIP_NIXL=${SKIP_NIXL:-0}
SKIP_DEPS=${SKIP_DEPS:-0}

step() { printf '\n==> %s\n' "$*"; }

step "Repo root: ${REPO_ROOT}"

# ---------------------------------------------------------------------------
# 1. Ensure `uv` is available
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  step "Installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installs into ~/.local/bin by default
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv install failed. Install it manually from https://docs.astral.sh/uv/ and re-run." >&2
  exit 1
fi
step "uv version: $(uv --version)"

# ---------------------------------------------------------------------------
# 2. Create the virtualenv if it does not exist yet
# ---------------------------------------------------------------------------
if [[ ! -d "${REPO_ROOT}/.venv" ]]; then
  step "Creating virtualenv at ${REPO_ROOT}/.venv (Python ${PYTHON_VERSION}) ..."
  (cd "${REPO_ROOT}" && uv venv --python "${PYTHON_VERSION}")
else
  step "Re-using existing virtualenv at ${REPO_ROOT}/.venv"
fi

# Activate so subsequent `uv pip` calls target this venv.
# shellcheck source=/dev/null
source "${REPO_ROOT}/.venv/bin/activate"
step "Python interpreter: $(command -v python) ($(python --version))"

# ---------------------------------------------------------------------------
# 3. Install vLLM (precompiled wheel = no CUDA toolkit needed locally)
# ---------------------------------------------------------------------------
if [[ "${SKIP_VLLM}" != "1" ]]; then
  step "Installing vLLM (editable, precompiled wheels for torch backend = auto) ..."
  (cd "${REPO_ROOT}" && VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto)
fi

# ---------------------------------------------------------------------------
# 4. Proxy + test dependencies
# ---------------------------------------------------------------------------
if [[ "${SKIP_DEPS}" != "1" ]]; then
  step "Installing proxy + test dependencies ..."
  uv pip install \
    aiohttp \
    httpx \
    fastapi \
    "uvicorn[standard]" \
    pytest
fi

# ---------------------------------------------------------------------------
# 5. NIXL (required for actual cross-node KV transfer)
# ---------------------------------------------------------------------------
if [[ "${SKIP_NIXL}" != "1" ]]; then
  step "Installing nixl (best-effort) ..."
  if uv pip install nixl; then
    echo "    nixl installed via pip."
  else
    cat <<'EOF'

WARNING: pip install nixl failed.
NIXL is required to actually transfer KV cache between nodes (both
PD-disaggregated and homogeneous-PD setups depend on it).

Install it manually following:
    https://github.com/ai-dynamo/nixl

If you only want to run the CPU-only scheduler unit tests
(`pytest examples/homogeneous_pd/tests`), nixl is not required.
EOF
  fi
fi

# ---------------------------------------------------------------------------
# 6. Smoke test: scheduler import + budget function
# ---------------------------------------------------------------------------
step "Verifying scheduler import ..."
PYTHONPATH="${EXAMPLES_DIR}:${PYTHONPATH:-}" python - <<'PY'
from homogeneous_pd.scheduler.budget_fn import linear_ratio
from homogeneous_pd.scheduler.homogeneous_scheduler import HomogeneousScheduler
assert linear_ratio(decode_tokens=4, alpha=16, max_batched=8192) == 64
print("HomogeneousScheduler import OK ; linear_ratio(4,16,8192)=64")
PY

# ---------------------------------------------------------------------------
# 7. Print next steps
# ---------------------------------------------------------------------------
cat <<EOF

============================================================
Install complete.

Activate the virtualenv in any new shell with:
  source ${REPO_ROOT}/.venv/bin/activate

Run the CPU-only unit tests:
  python -m pytest examples/homogeneous_pd/tests -v

Distributed deployment (run on each machine):
  1. Copy and edit a topology config:
       cp examples/homogeneous_pd/configs/homogeneous_2node.example.conf my.conf
       \$EDITOR my.conf      # set NODE_HOSTS / NODE_PORTS / NODE_GPUS to your cluster
  2. On every node host:
       examples/homogeneous_pd/launchers/launch_node.sh my.conf <NODE_INDEX>
  3. On the router host:
       examples/homogeneous_pd/launchers/launch_router.sh my.conf

Single-machine quick start (everything on one box) is also supported via:
  examples/homogeneous_pd/launchers/run_homogeneous_2node.sh
  examples/homogeneous_pd/launchers/run_1p1d.sh
  ...
============================================================
EOF
