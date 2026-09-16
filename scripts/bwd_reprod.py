"""TE fused attention on MI350X: forward/backward timing and bwd/fwd ratio.

NVTE_CK_USES_FWD_V3 / NVTE_CK_USES_BWD_V3 / NVTE_CK_IS_V3_ATOMIC_FP32 are enabled by
default so TE prefers gfx950 ASM when available and falls back to CK otherwise. The
``kernel`` column shows the routed backend per direction (fwd/bwd), e.g. ASM/CK.

  source /dockerx/NVTE_env.sh
  python3 bwd_reprod.py [--seqs 2048,8192,20480] [--reps 5]
  python3 bwd_reprod.py --cases thd
  python3 bwd_reprod.py --cases dense,thd

Expected for a healthy flash-attention backward: ~2–2.5x the forward.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Optional

# Prefer ASM v3 paths; C++ gates fall back to CK when no kernel matches.
os.environ.setdefault("NVTE_CK_USES_FWD_V3", "1")
os.environ.setdefault("NVTE_CK_USES_BWD_V3", "1")
os.environ.setdefault("NVTE_CK_IS_V3_ATOMIC_FP32", "0")

import jax
import jax.numpy as jnp
from transformer_engine.jax.attention import (
    AttnBiasType,
    AttnMaskType,
    AttnSoftmaxType,
    QKVLayout,
    SequenceDescriptor,
    fused_attn,
)

Layout = Literal["bshd", "thd"]

p = argparse.ArgumentParser()
p.add_argument("--seqs", default="2048,8192,20480")
p.add_argument("--reps", type=int, default=5)
p.add_argument("--heads", type=int, default=18)
p.add_argument("--batch", type=int, default=1)
p.add_argument(
    "--cases",
    default="dense,thd",
    help="comma-separated: dense (BSHD no mask), thd (THD varlen padding/causal); "
    "alias: asm -> thd",
)
p.add_argument(
    "--pad-ratio",
    type=float,
    default=0.3,
    help="trailing padding ratio for THD cases (0 = no padding)",
)
p.add_argument(
    "--thd-heads",
    type=int,
    default=8,
    help="heads for THD cases (MHA; hd256 ASM requires h_q == h_kv)",
)
p.add_argument(
    "--thd-batch",
    type=int,
    default=2,
    help="batch size for THD cases",
)
a = p.parse_args()

SOFTMAX = getattr(AttnSoftmaxType, "VANILLA", list(AttnSoftmaxType)[0])
CASES = {("thd" if c.strip() == "asm" else c.strip()) for c in a.cases.split(",") if c.strip()}

print("jax", jax.__version__, "| device", jax.devices()[0].device_kind, flush=True)
print(
    "NVTE env:",
    f"FWD_V3={os.environ.get('NVTE_CK_USES_FWD_V3')}",
    f"BWD_V3={os.environ.get('NVTE_CK_USES_BWD_V3')}",
    f"ATOMIC_FP32={os.environ.get('NVTE_CK_IS_V3_ATOMIC_FP32')}",
    flush=True,
)


def v3_enabled(name: str) -> bool:
    return os.environ.get(name, "1") not in ("0", "false", "False")


def routed_backend(
    phase: Literal["fwd", "bwd"],
    hd: int,
    layout: Layout,
    mask_type: AttnMaskType,
    heads: int,
    heads_kv: int,
) -> str:
    """Mirror TE gfx950 ASM gates (fused_attn_ck.cpp); otherwise CK."""
    mha = heads == heads_kv
    if phase == "fwd":
        if not v3_enabled("NVTE_CK_USES_FWD_V3"):
            return "CK"
        if hd == 128:
            return "ASM"
        if hd == 256 and layout == "thd" and mha and mask_type.is_padding():
            return "ASM"
        return "CK"

    if not v3_enabled("NVTE_CK_USES_BWD_V3"):
        return "CK"
    if hd == 128:
        return "ASM"
    if hd == 256 and layout == "thd" and mha and mask_type == AttnMaskType.PADDING_MASK:
        return "ASM"
    return "CK"


def format_kernel_col(fwd_backend: str, bwd_backend: Optional[str]) -> str:
    if bwd_backend is None:
        return fwd_backend
    if fwd_backend == bwd_backend:
        return fwd_backend
    return f"{fwd_backend}/{bwd_backend}"


def timeit(f: Callable, *args) -> float:
    jax.block_until_ready(f(*args))
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*args))
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def print_row(
    case: str,
    hd: int,
    seq: int,
    fwd_ms: Optional[float],
    total_ms: Optional[float],
    kernel: str = "—",
    err: str = "",
) -> None:
    if fwd_ms is None or (total_ms is None and err):
        msg = f"  FAILED: {err}" if err else "  FAILED"
        print(f"{case:>16} {hd:>8} {seq:>6}{msg}", flush=True)
        return
    if total_ms is None:
        print(
            f"{case:>16} {hd:>8} {seq:>6} {fwd_ms:8.1f} {'—':>11} {'—':>8} {'—':>8} {kernel:>10}",
            flush=True,
        )
        return
    bwd_ms = total_ms - fwd_ms
    ratio = bwd_ms / fwd_ms if fwd_ms > 0 else float("nan")
    print(
        f"{case:>16} {hd:>8} {seq:>6} {fwd_ms:8.1f} {total_ms:11.1f} "
        f"{bwd_ms:8.1f} {ratio:8.1f}x {kernel:>10}",
        flush=True,
    )


@dataclass(frozen=True)
class ThdCase:
    name: str
    mask_type: AttnMaskType


def get_seqlens_and_offsets(segment_ids: jnp.ndarray):
    """THD seqlens/offsets from per-token segment ids (0 = pad)."""
    batch, max_seqlen = segment_ids.shape
    bincount_vmap = jax.vmap(partial(jnp.bincount, length=max_seqlen))
    seqlens_with_zero = bincount_vmap(segment_ids.astype(jnp.int32))
    seqlens = seqlens_with_zero[..., 1:]

    def _find_offsets(x):
        same_as_previous = jnp.logical_and(x[..., 1:] != x[..., :-1], x[..., 1:] != 0)
        first_column = x[..., :1] != 0
        same_as_previous = jnp.hstack((first_column, same_as_previous))
        return jax.vmap(partial(jnp.argwhere, size=x.shape[1], fill_value=-1))(
            same_as_previous
        ).squeeze(-1)

    offsets = _find_offsets(segment_ids)
    offsets = jnp.insert(offsets, offsets.shape[-1], values=-1, axis=-1)
    seqlens = jnp.insert(seqlens, seqlens.shape[-1], values=0, axis=-1)
    seqlens = jnp.where(seqlens, seqlens, -1)
    return seqlens, offsets


def make_thd_inputs(
    batch: int,
    max_seqlen: int,
    heads: int,
    hd: int,
    pad_ratio: float,
    key: jax.Array,
):
    if pad_ratio > 0:
        pad_len = int(max_seqlen * pad_ratio)
        valid_len = max_seqlen - pad_len
        segment_ids = jnp.concatenate(
            [
                jnp.ones((batch, valid_len), dtype=jnp.int32),
                jnp.zeros((batch, pad_len), dtype=jnp.int32),
            ],
            axis=-1,
        )
    else:
        segment_ids = jnp.ones((batch, max_seqlen), dtype=jnp.int32)

    seqlens, offsets = get_seqlens_and_offsets(segment_ids)
    seq_desc = SequenceDescriptor.from_seqlens_and_offsets(
        (seqlens, seqlens), (offsets, offsets)
    )
    pad = segment_ids == 0

    shape = (batch, max_seqlen, heads, hd)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, shape, jnp.bfloat16)
    k = jax.random.normal(k2, shape, jnp.bfloat16)
    v = jax.random.normal(k3, shape, jnp.bfloat16)
    return q, k, v, seq_desc, pad


def bench_dense_bshd(hd: int, seq: int) -> None:
    shape = (a.batch, seq, a.heads, hd)
    key = jax.random.PRNGKey(hd * 10_000 + seq)
    k1, k2, k3 = jax.random.split(key, 3)
    q, kk, v = (jax.random.normal(kk_, shape, jnp.bfloat16) for kk_ in (k1, k2, k3))
    kernel = format_kernel_col(
        routed_backend("fwd", hd, "bshd", AttnMaskType.NO_MASK, a.heads, a.heads),
        routed_backend("bwd", hd, "bshd", AttnMaskType.NO_MASK, a.heads, a.heads),
    )

    def attn(q, k, v):
        return fused_attn(
            (q, k, v),
            None,
            None,
            None,
            AttnBiasType.NO_BIAS,
            AttnMaskType.NO_MASK,
            QKVLayout.BSHD_BSHD_BSHD,
            SOFTMAX,
            1.0 / (hd**0.5),
            0.0,
            True,
        )

    fwd = jax.jit(attn)
    fwd_bwd = jax.jit(
        jax.value_and_grad(
            lambda q, k, v: jnp.sum(attn(q, k, v).astype(jnp.float32)),
            argnums=(0, 1, 2),
        )
    )
    try:
        tf_ = timeit(fwd, q, kk, v)
        tb = timeit(fwd_bwd, q, kk, v)
        print_row("bshd_nomask", hd, seq, tf_, tb, kernel)
    except Exception as e:  # noqa: BLE001
        print_row("bshd_nomask", hd, seq, None, None, kernel, err=f"{type(e).__name__}: {str(e)[:100]}")


def bench_thd(case: ThdCase, hd: int, seq: int) -> None:
    q, k, v, seq_desc, pad = make_thd_inputs(
        a.thd_batch, seq, a.thd_heads, hd, a.pad_ratio, jax.random.PRNGKey(hd * 100 + seq)
    )
    scale = 1.0 / (hd**0.5)
    kernel = format_kernel_col(
        routed_backend("fwd", hd, "thd", case.mask_type, a.thd_heads, a.thd_heads),
        routed_backend("bwd", hd, "thd", case.mask_type, a.thd_heads, a.thd_heads),
    )

    def attn(q, k, v):
        return fused_attn(
            (q, k, v),
            None,
            seq_desc,
            None,
            AttnBiasType.NO_BIAS,
            case.mask_type,
            QKVLayout.THD_THD_THD,
            SOFTMAX,
            scale,
            0.0,
            True,
            max_segments_per_seq=2,
        )

    def loss(q, k, v):
        out = attn(q, k, v)
        out = jnp.where(pad[..., jnp.newaxis, jnp.newaxis], 0, out)
        return jnp.sum(out.astype(jnp.float32))

    fwd = jax.jit(attn)
    fwd_bwd = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))
    try:
        tf_ = timeit(fwd, q, k, v)
        tb = timeit(fwd_bwd, q, k, v)
        print_row(case.name, hd, seq, tf_, tb, kernel)
    except Exception as e:  # noqa: BLE001
        print_row(case.name, hd, seq, None, None, kernel, err=f"{type(e).__name__}: {str(e)[:100]}")


THD_CASES = (
    ThdCase("thd_padding", AttnMaskType.PADDING_MASK),
    ThdCase("thd_causal", AttnMaskType.PADDING_CAUSAL_MASK),
)

print(
    f"{'case':>16} {'head_dim':>8} {'seq':>6} {'fwd ms':>8} "
    f"{'fwd+bwd ms':>11} {'bwd ms':>8} {'bwd/fwd':>8} {'kernel':>10}",
    flush=True,
)

seqs = [int(s) for s in a.seqs.split(",")]

if "dense" in CASES:
    for hd in (256, 128):
        for seq in seqs:
            bench_dense_bshd(hd, seq)

if "thd" in CASES:
    for hd in (256, 128):
        for seq in seqs:
            for case in THD_CASES:
                bench_thd(case, hd, seq)
