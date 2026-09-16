#!/usr/bin/env python3
"""Smoke test TE JAX fused attention at hd=256 bf16 (gfx950 ASM path)."""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

# Source scripts/env_hd256.sh in ym_hd256 (rocm_sdk LD_LIBRARY_PATH) before import.
# jax.devices() first is still a safe fallback on other venvs.
_ = jax.devices()

# tests live under TE_HD256/tests/jax
TE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(TE_ROOT, "tests", "jax"))

from test_fused_attn import (  # noqa: E402
    AttnBiasType,
    AttnMaskType,
    AttnSoftmaxType,
    BiasShape,
    FusedAttnRunner,
    QKVLayout,
    SeqDescFormat,
)


def run_case(
    name: str,
    batch_size: int,
    num_heads: int,
    max_seqlen: int,
    attn_mask_type: AttnMaskType = AttnMaskType.PADDING_MASK,
    run_backward: bool = True,
) -> None:
    print(
        f"\n=== {name} b={batch_size} h={num_heads} s={max_seqlen} "
        f"d=256 bf16 THD mask={attn_mask_type.name} ==="
    )
    runner = FusedAttnRunner(
        batch_size=batch_size,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        num_heads_q=num_heads,
        num_heads_kv=num_heads,
        head_dim_qk=256,
        head_dim_v=256,
        attn_bias_type=AttnBiasType.NO_BIAS,
        attn_mask_type=attn_mask_type,
        softmax_type=AttnSoftmaxType.VANILLA_SOFTMAX,
        dropout_prob=0.0,
        use_old_rng=True,
        dtype=jnp.bfloat16,
        is_training=True,
        qkv_layout=QKVLayout.THD_THD_THD,
        bias_shape=BiasShape._BHSS,
        window_size=None,
        seq_desc_format=SeqDescFormat.Seqlens,
    )
    runner.test_forward()
    if run_backward:
        runner.test_backward()
    print(f"=== {name} PASS ===")


def main() -> None:
    # Keep JAX from grabbing all VRAM; FusedAttnRunner also runs a reference path.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

    print("JAX devices:", jax.devices())
    print(
        "ENV:",
        f"NVTE_CK_USES_FWD_V3={os.environ.get('NVTE_CK_USES_FWD_V3')}",
        f"NVTE_CK_USES_BWD_V3={os.environ.get('NVTE_CK_USES_BWD_V3')}",
        f"NVTE_CK_IS_V3_ATOMIC_FP32={os.environ.get('NVTE_CK_IS_V3_ATOMIC_FP32')}",
    )
    # Smoke size fits single-GPU VRAM with TE + reference; still hits hd256 ASM kernels.
    run_case("hd256_noncausal", batch_size=2, num_heads=8, max_seqlen=512)
    # hd256 bwd ASM is non-causal only; causal bwd uses CK — fwd-only smoke here.
    run_case(
        "hd256_causal",
        batch_size=2,
        num_heads=8,
        max_seqlen=512,
        attn_mask_type=AttnMaskType.PADDING_CAUSAL_MASK,
        run_backward=False,
    )
    print("\nAll hd256 JAX smoke tests passed.")


if __name__ == "__main__":
    main()
