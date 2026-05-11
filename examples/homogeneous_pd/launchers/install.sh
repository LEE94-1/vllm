#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Installer for the homogeneous_pd experiment.
#
# Designed for container environments: installs straight into whatever
# Python is already on PATH, no venv, no uv, no Python download. Every
# package fetch goes through `pip`, which means whatever you have
# configured in /etc/pip.conf / ~/.pip/pip.conf / PIP_INDEX_URL is
# honoured. Nothing is downloaded from the public internet outside pip:
# no uv installer from astral.sh, no precompiled wheel from
# wheels.vllm.ai, no extra PyTorch index, no python-build-standalone
# tarball.
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
#   PYTHON_BIN=python3.10 ./install.sh   # pick a specific python interpreter
#   PIP_FLAGS="--user" ./install.sh  # extra pip flags (e.g. --user, --break-system-packages)
#
# Notes
# -----
# 1. vLLM is installed via plain `pip install vllm` (NOT `-e .` from this
#    repo), so the install does not need a CUDA toolkit and does not hit
#    wheels.vllm.ai. The in-repo `examples/homogeneous_pd/` package is
#    loaded by the launchers via PYTHONPATH (see launchers/common.sh) so
#    edits to the on-disk scheduler/proxy still drive behaviour.
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
# Anything the caller wants to forward to every `pip install`. Use this for
# --user, --break-system-packages, --no-build-isolation, etc.
PIP_FLAGS=${PIP_FLAGS:-}

step() { printf '\n==> %s\n' "$*"; }

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "${PYTHON_BIN} not found. Install Python 3.10+ via your apt repo and retry," >&2
  echo "or set PYTHON_BIN to a working interpreter (e.g. PYTHON_BIN=python3.10)." >&2
  exit 1
fi

step "Repo root  : ${REPO_ROOT}"
step "Python bin : $(command -v "${PYTHON_BIN}") ($("${PYTHON_BIN}" --version))"

# Shorthand: every install command goes through "${PYTHON_BIN}" -m pip with
# the user's optional PIP_FLAGS appended.
pip_install() {
  # shellcheck disable=SC2086  # we deliberately word-split PIP_FLAGS
  "${PYTHON_BIN}" -m pip install ${PIP_FLAGS} "$@"
}

# ---------------------------------------------------------------------------
# 1. Print the pip config that is actually in effect, so misconfigured
#    mirrors are obvious from the install log.
# ---------------------------------------------------------------------------
step "pip config (effective):"
"${PYTHON_BIN}" -m pip config list 2>/dev/null || true

# ---------------------------------------------------------------------------
# 2. Upgrade pip itself (also via the mirror).
# ---------------------------------------------------------------------------
step "Upgrading pip / setuptools / wheel ..."
pip_install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 3. vLLM (from whatever your pip mirror has tagged as `vllm`).
# ---------------------------------------------------------------------------
if [[ "${SKIP_VLLM}" != "1" ]]; then
  step "Installing vllm ..."
  pip_install vllm
fi

# ---------------------------------------------------------------------------
# 4. Proxy + test dependencies.
# ---------------------------------------------------------------------------
if [[ "${SKIP_DEPS}" != "1" ]]; then
  step "Installing proxy + test dependencies ..."
  pip_install \
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
  if pip_install flashinfer-python; then
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
  if pip_install nixl; then
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
PYTHONPATH="${EXAMPLES_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<'PY'
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
Install complete (system / container Python: $(command -v "${PYTHON_BIN}")).

Run the CPU-only unit tests:
  ${PYTHON_BIN} -m pytest examples/homogeneous_pd/tests -v

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
