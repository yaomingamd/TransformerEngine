# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
"""Accuracy coverage for gfx950 BF16 D=256 AITER FMHA ASM kernels."""

import pytest

import jax.numpy as jnp

from transformer_engine.jax.attention import AttnBiasType, AttnMaskType, AttnSoftmaxType, QKVLayout
from transformer_engine.jax.cpp_extensions.misc import is_hip_extension
from transformer_engine_jax import get_device_compute_capability

from test_fused_attn import FusedAttnRunner, SeqDescFormat


pytestmark = [
    pytest.mark.skipif(not is_hip_extension(), reason="ROCm TE specific test."),
    pytest.mark.skipif(
        get_device_compute_capability(0) != 95,
        reason="gfx950 (compute capability 9.5) required.",
    ),
]

_ASM_ENV = {
    "NVTE_FUSED_ATTN": "1",
    "NVTE_FUSED_ATTN_CK": "1",
    "NVTE_FUSED_ATTN_AOTRITON": "0",
    "NVTE_FLASH_ATTN": "0",
    "NVTE_CK_USES_FWD_V3": "1",
    "NVTE_CK_USES_BWD_V3": "1",
    # The D=256 backward ASM kernel is registered for atomic16 only.
    "NVTE_CK_IS_V3_ATOMIC_FP32": "0",
}


def _enable_asm(monkeypatch):
    for name, value in _ASM_ENV.items():
        monkeypatch.setenv(name, value)


def _runner(mask_type, max_seqlen_q, max_seqlen_kv, is_training):
    return FusedAttnRunner(
        batch_size=2,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        num_heads_q=8,
        num_heads_kv=8,
        head_dim_qk=256,
        head_dim_v=256,
        attn_bias_type=AttnBiasType.NO_BIAS,
        attn_mask_type=mask_type,
        softmax_type=AttnSoftmaxType.VANILLA_SOFTMAX,
        dropout_prob=0.0,
        dtype=jnp.bfloat16,
        is_training=is_training,
        qkv_layout=QKVLayout.THD_THD_THD,
        bias_shape=None,
        window_size=None,
        seq_desc_format=SeqDescFormat.Seqlens,
        # D=256 online softmax uses more BF16 rescaling steps than smaller head sizes.
        rtol=3.2e-2,
        atol=3.2e-2,
    )


def test_gfx950_bf16_hd256_thd_noncausal_forward_backward(monkeypatch):
    """Compare noncausal THD forward, dQ, dK, and dV with the JAX reference."""
    _enable_asm(monkeypatch)
    _runner(AttnMaskType.PADDING_MASK, 256, 256, True).test_backward()


def test_gfx950_bf16_hd256_thd_causal_sq_gt_sk(monkeypatch):
    """Compare bottom-right causal THD forward when each query is longer than its key."""
    _enable_asm(monkeypatch)
    _runner(AttnMaskType.PADDING_CAUSAL_BOTTOM_RIGHT_MASK, 320, 256, False).test_forward()
