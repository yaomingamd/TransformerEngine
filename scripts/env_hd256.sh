#!/usr/bin/env bash
# Runtime environment for TE_HD256 JAX/PyTorch on gfx950 (ym_hd256 / hd256 venv).
# hd256 ASM kernels are in the shared transformer_engine lib; both backends use
# fused_attn_ck.cpp (NVTE_CK_USES_* env vars).
#
# Do NOT put /opt/rocm/core/lib on LD_LIBRARY_PATH at Python import time:
# jax_rocm7_plugin already ships LLVM bits, and loading system ROCm LLVM
# triggers "spirv-expand-step registered more than once".
#
# The pip rocm_sdk_* wheels bundle libroctx, hipblaslt, etc. under site-packages.
#
# Usage (after venv activate):
#   source /dockerx/TE_HD256/scripts/env_hd256.sh

_te_hd256_detect_sdk_libs() {
  python3 - <<'PY'
import site
from pathlib import Path

paths = []
for sp in site.getsitepackages():
    for sub in ("_rocm_sdk_core/lib", "_rocm_sdk_libraries/lib"):
        p = Path(sp) / sub
        if p.is_dir():
            paths.append(str(p))
print(":".join(paths))
PY
}

TE_DIR="${TE_DIR:-/dockerx/TE_HD256}"

# Container default ROCM_PATH=/opt/rocm lacks .info/version; always use core install.
export ROCM_PATH=/opt/rocm/core
export HIP_PATH=/opt/rocm/core
export HIP_PLATFORM="${HIP_PLATFORM:-amd}"

_te_hd256_sdk_libs="$(_te_hd256_detect_sdk_libs)"
if [ -n "${_te_hd256_sdk_libs}" ]; then
  if [ -z "${_TE_HD256_ENV_ACTIVE:-}" ]; then
    export _OLD_TE_HD256_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    export _TE_HD256_ENV_ACTIVE=1
  fi
  export LD_LIBRARY_PATH="${_te_hd256_sdk_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
  echo "WARN: rocm_sdk pip libs not found in active venv; TE may fail to load native libs" >&2
fi

export PYTHONPATH="${TE_DIR}/3rdparty/hipify_torch${PYTHONPATH:+:${PYTHONPATH}}"

export NVTE_CK_USES_FWD_V3="${NVTE_CK_USES_FWD_V3:-1}"
export NVTE_CK_USES_BWD_V3="${NVTE_CK_USES_BWD_V3:-1}"
export NVTE_CK_IS_V3_ATOMIC_FP32="${NVTE_CK_IS_V3_ATOMIC_FP32:-0}"
export NVTE_LOG_CK_CONFIG="${NVTE_LOG_CK_CONFIG:-0}"

te_hd256_deactivate() {
  if [ -n "${_TE_HD256_ENV_ACTIVE:-}" ]; then
    if [ -n "${_OLD_TE_HD256_LD_LIBRARY_PATH+x}" ]; then
      export LD_LIBRARY_PATH="${_OLD_TE_HD256_LD_LIBRARY_PATH}"
      unset _OLD_TE_HD256_LD_LIBRARY_PATH
    fi
    unset _TE_HD256_ENV_ACTIVE
  fi
}
