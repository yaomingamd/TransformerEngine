# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
"""Accuracy coverage for gfx950 BF16 D=256 AITER FMHA ASM kernels."""

import pathlib
import sys

import pytest
import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION

from transformer_engine.pytorch import get_device_compute_capability
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends
from transformer_engine.pytorch.cpp_extensions.fused_attn import FusedAttnBackend

_current_file = pathlib.Path(__file__).resolve()
sys.path.append(str(_current_file.parent.parent))
from utils import ModelConfig, get_available_attention_backends
from test_attention import run_dot_product_attention


pytestmark = [
    pytest.mark.skipif(not IS_HIP_EXTENSION, reason="ROCm TE specific test."),
    pytest.mark.skipif(
        get_device_compute_capability() != (9, 5),
        reason="gfx950 (compute capability 9.5) required.",
    ),
]


_CONFIGS = {
    "fwd_noncausal_bwd": (
        ModelConfig(2, 256, 8, 256),
        True,
    ),
    "fwd_causal_sq_gt_sk": (
        ModelConfig(
            2,
            320,
            8,
            256,
            max_seqlen_kv=256,
            attn_mask_type="padding_causal_bottom_right",
        ),
        False,
    ),
}


def _set_backend(monkeypatch: pytest.MonkeyPatch, use_v3: bool) -> None:
    """Select CK and toggle AITER v3 ASM dispatch."""
    for name in (
        "NVTE_FLASH_ATTN",
        "NVTE_FUSED_ATTN",
        "NVTE_UNFUSED_ATTN",
        "NVTE_FUSED_ATTN_CK",
        "NVTE_FUSED_ATTN_AOTRITON",
        "NVTE_CK_USES_FWD_V3",
        "NVTE_CK_USES_BWD_V3",
        "NVTE_CK_IS_V3_ATOMIC_FP32",
    ):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("NVTE_FUSED_ATTN", "1")
    monkeypatch.setenv("NVTE_FUSED_ATTN_CK", "1")
    monkeypatch.setenv("NVTE_FLASH_ATTN", "0")
    monkeypatch.setenv("NVTE_FUSED_ATTN_AOTRITON", "0")
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
    monkeypatch.setenv("NVTE_CK_USES_FWD_V3", str(int(use_v3)))
    monkeypatch.setenv("NVTE_CK_USES_BWD_V3", str(int(use_v3)))
    # The HD256 backward ASM kernel supports A16 only.
    monkeypatch.setenv("NVTE_CK_IS_V3_ATOMIC_FP32", "0")
    _attention_backends["backend_selection_requires_update"] = True


@pytest.mark.parametrize("name", sorted(_CONFIGS))
def test_gfx950_bf16_hd256_thd_asm_accuracy(monkeypatch, name):
    """Compare AITER v3 BF16 HD256 THD ASM results with the unfused reference."""
    config, is_training = _CONFIGS[name]

    _set_backend(monkeypatch, use_v3=True)
    _, _, fused_backends = get_available_attention_backends(
        config,
        qkv_dtype=torch.bfloat16,
        qkv_layout="thd_thd_thd",
        is_training=is_training,
    )
    if FusedAttnBackend["CK"] not in fused_backends:
        pytest.skip("CK fused attention is unavailable for this configuration.")

    v3_out, _, v3_grads = run_dot_product_attention(
        torch.bfloat16,
        config,
        "FusedAttention",
        False,
        "thd_thd_thd",
        False,
        is_training,
    )

    ref_out, _, ref_grads = run_dot_product_attention(
        torch.bfloat16,
        config,
        "UnfusedDotProductAttention",
        False,
        "thd_thd_thd",
        False,
        is_training,
    )

    # The D=256 forward kernel has a larger tile and therefore extra BF16 online-softmax
    # rescaling steps. A 1/32 ULP tolerance covers its validated long-sequence error bound.
    torch.testing.assert_close(v3_out, ref_out, atol=3.2e-2, rtol=3.2e-2)
    if is_training:
        for v3_grad, ref_grad in zip(v3_grads[:3], ref_grads[:3]):
            torch.testing.assert_close(v3_grad, ref_grad, atol=1e-2, rtol=1e-2)
