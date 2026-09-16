#!/usr/bin/env python3
"""Smoke test TE PyTorch fused attention at hd=256 bf16 (gfx950 ASM via shared CK path)."""
import os
import sys

import torch

import transformer_engine.pytorch  # noqa: F401 — loads transformer_engine_torch extension

TE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(TE_ROOT, "tests", "pytorch"))
sys.path.insert(0, os.path.join(TE_ROOT, "tests", "pytorch", "attention"))

from utils import ModelConfig, get_available_attention_backends  # noqa: E402
from test_attention import _run_dot_product_attention  # noqa: E402
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends
from transformer_engine.pytorch.cpp_extensions.fused_attn import FusedAttnBackend


def _setup_env() -> None:
    os.environ["NVTE_FUSED_ATTN_CK"] = "1"
    os.environ["NVTE_FUSED_ATTN_AOTRITON"] = "0"
    os.environ["NVTE_CK_USES_FWD_V3"] = "1"
    os.environ["NVTE_CK_USES_BWD_V3"] = "1"
    os.environ["NVTE_CK_IS_V3_ATOMIC_FP32"] = "0"
    _attention_backends["backend_selection_requires_update"] = True


def run_case(name: str, mask_type: str, run_backward: bool = True) -> None:
    dtype = torch.bfloat16
    config = ModelConfig(2, 512, 8, 256, attn_mask_type=mask_type)
    qkv_layout = "thd_thd_thd"
    print(
        f"\n=== {name} b={config.batch_size} h={config.num_heads} s={config.max_seqlen_q} "
        f"d=256 bf16 THD mask={mask_type} ==="
    )
    _, _, fused_backends = get_available_attention_backends(
        config,
        qkv_dtype=dtype,
        qkv_layout=qkv_layout,
        pad_between_seqs=True,
        is_training=run_backward,
    )
    if FusedAttnBackend["CK"] not in fused_backends:
        raise RuntimeError(f"CK fused attention required, got backends={fused_backends}")

    _run_dot_product_attention(
        dtype,
        config,
        "FusedAttention",
        ckpt_attn=False,
        qkv_layout=qkv_layout,
        workspace_opt=False,
        pad_between_seqs=True,
        is_training=run_backward,
    )
    print(f"=== {name} PASS ===")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device required")
    _setup_env()
    print("PyTorch", torch.__version__, "| device", torch.cuda.get_device_name(0))
    print(
        "ENV:",
        f"NVTE_CK_USES_FWD_V3={os.environ.get('NVTE_CK_USES_FWD_V3')}",
        f"NVTE_CK_USES_BWD_V3={os.environ.get('NVTE_CK_USES_BWD_V3')}",
        f"NVTE_CK_IS_V3_ATOMIC_FP32={os.environ.get('NVTE_CK_IS_V3_ATOMIC_FP32')}",
    )
    run_case("hd256_padding", "padding", run_backward=True)
    run_case("hd256_causal", "padding_causal", run_backward=False)
    print("\nAll hd256 PyTorch smoke tests passed.")


if __name__ == "__main__":
    main()
