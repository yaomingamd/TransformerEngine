#!/usr/bin/env bash
# Build, install, and smoke-test TE_HD256 with hd256 ASM (gfx950) for JAX and/or PyTorch.
# Resumable: re-run after salloc on mi355-gpu-058 / ym_hd256 container.
set -euo pipefail

TE_DIR=/dockerx/TE_HD256
SCRIPT_DIR="${TE_DIR}/scripts"
LOG_DIR="${TE_DIR}/logs"
mkdir -p "${LOG_DIR}"
BUILD_LOG="${LOG_DIR}/build_te.log"
TEST_JAX_LOG="${LOG_DIR}/test_hd256_jax.log"
TEST_PYTORCH_LOG="${LOG_DIR}/test_hd256_pytorch.log"
STATE_FILE="${LOG_DIR}/pipeline.state"

if [ -n "${TE_VENV:-}" ] && [ -f "${TE_VENV}/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "${TE_VENV}/bin/activate"
elif [ -f /root/hd256/bin/activate ]; then
  source /root/hd256/bin/activate
elif [ -f /root/root/venv/torch/bin/activate ]; then
  source /root/root/venv/torch/bin/activate
else
  echo "No Python venv found (set TE_VENV)" >&2
  exit 1
fi
pip install -q cmake wheel pybind11 ninja 2>/dev/null || true

cd "${TE_DIR}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_hd256.sh"

export ROCM_PATH="${ROCM_PATH:-/opt/rocm/core}"
export HIP_PATH="${HIP_PATH:-/opt/rocm/core}"

export USE_ROCM=1
export CMAKE_BUILD_PARALLEL_LEVEL=64
export PYTORCH_ROCM_ARCH=gfx950
export NVTE_ROCM_ARCH=gfx950
export NVTE_USE_ROCM=1
export NVTE_FUSED_ATTN_AOTRITON=0
export NVTE_ROCM_ENABLE_MXFP8=1
export NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1

# Default: build both JAX and PyTorch (shared lib + both framework extensions).
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-jax,pytorch}"

# Link against system ROCm during wheel build only (not at Python import time).
te_hd256_build_ld_path() {
  export ROCM_PATH="${ROCM_PATH:-/opt/rocm/core}"
  export HIP_PATH="${HIP_PATH:-/opt/rocm/core}"
  export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
}

stage="${1:-all}"

detect_aiter_cache() {
  local aiter_sha
  aiter_sha=$(git -C "${TE_DIR}/3rdparty/aiter" rev-parse HEAD)
  local cache_root="${TE_DIR}/build/aiter-prebuilts"
  local cache_dir
  cache_dir=$(find "${cache_root}" -maxdepth 1 -type d -name "rocm-*_aiter-${aiter_sha}" 2>/dev/null | while read -r d; do
    if [ -f "${d}/libmha_fwd.so" ] && [ -f "${d}/libmha_bwd.so" ]; then
      echo "${d}"
      break
    fi
  done)
  if [ -n "${cache_dir}" ]; then
    export NVTE_CK_FUSED_ATTN_PATH="${cache_dir}"
    echo "[$(date -Is)] Reusing AITER cache: ${cache_dir}" | tee -a "${BUILD_LOG}"
  fi
}

build_te() {
  local clean="${1:-1}"
  te_hd256_build_ld_path
  echo "[$(date -Is)] BUILD start (clean=${clean}, NVTE_FRAMEWORK=${NVTE_FRAMEWORK})" | tee -a "${BUILD_LOG}"
  if [ "${clean}" = "1" ]; then
    rm -rf build dist
    unset NVTE_CK_FUSED_ATTN_PATH || true
  else
    rm -rf dist build/cmake
    detect_aiter_cache
  fi
  python3 setup.py bdist_wheel 2>&1 | tee -a "${BUILD_LOG}"
  WHEEL=(dist/transformer_engine*.whl)
  if [ ! -f "${WHEEL[0]}" ]; then
    echo "ERROR: wheel not found in dist/" | tee -a "${BUILD_LOG}"
    return 1
  fi
  echo "[$(date -Is)] BUILD done: ${WHEEL[0]}" | tee -a "${BUILD_LOG}"
  echo "built" > "${STATE_FILE}"
}

install_te() {
  WHEEL=(dist/transformer_engine*.whl)
  if [ ! -f "${WHEEL[0]}" ]; then
    echo "No wheel; run build first" >&2
    return 1
  fi
  echo "[$(date -Is)] INSTALL ${WHEEL[0]}" | tee -a "${BUILD_LOG}"
  pip install --force-reinstall --no-deps "${WHEEL[0]}" 2>&1 | tee -a "${BUILD_LOG}"
  python3 -c "import transformer_engine.jax as te; print('TE jax', te.__file__)" 2>/dev/null || true
  python3 -c "import transformer_engine_torch as tex; print('TE torch', tex.__file__)" 2>/dev/null || true
  echo "installed" > "${STATE_FILE}"
}

test_hd256_jax() {
  export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
  pip install -q pytest pytest-timeout flax 2>/dev/null || true

  echo "[$(date -Is)] TEST hd256 JAX start" | tee -a "${TEST_JAX_LOG}"
  cd "${TE_DIR}"
  python3 scripts/test_hd256_jax_smoke.py 2>&1 | tee -a "${TEST_JAX_LOG}"
  echo "[$(date -Is)] TEST hd256 JAX PASS" | tee -a "${TEST_JAX_LOG}"
}

test_hd256_pytorch() {
  export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
  pip install -q pytest pytest-timeout 2>/dev/null || true

  echo "[$(date -Is)] TEST hd256 PyTorch start" | tee -a "${TEST_PYTORCH_LOG}"
  cd "${TE_DIR}"
  python3 scripts/test_hd256_pytorch_smoke.py 2>&1 | tee -a "${TEST_PYTORCH_LOG}"
  echo "[$(date -Is)] TEST hd256 PyTorch PASS" | tee -a "${TEST_PYTORCH_LOG}"
}

test_hd256() {
  test_hd256_jax
  test_hd256_pytorch
  echo "tested" > "${STATE_FILE}"
}

case "${stage}" in
  build) build_te ;;
  install) install_te ;;
  test) test_hd256 ;;
  test_jax) test_hd256_jax ;;
  test_pytorch) test_hd256_pytorch ;;
  all)
    build_te
    install_te
    test_hd256
    ;;
  retry)
    build_te 0
    install_te
    test_hd256
    ;;
  resume)
    if compgen -G "dist/transformer_engine*.whl" > /dev/null; then
      install_te
      test_hd256
    elif [ -d build/aiter-prebuilts ] && find build/aiter-prebuilts -name libmha_fwd.so -print -quit | grep -q .; then
      build_te 0
      install_te
      test_hd256
    else
      build_te 1
      install_te
      test_hd256
    fi
    ;;
  *)
    echo "Usage: $0 {build|install|test|test_jax|test_pytorch|all|retry|resume}"
    echo "  NVTE_FRAMEWORK=jax|pytorch|jax,pytorch (default: jax,pytorch)"
    exit 1
    ;;
esac
