#!/usr/bin/env python3
"""TE PyTorch fused-attention bwd/fwd timing (mirror of bwd_reprod.py for JAX).

Uses DotProductAttention (TE PyTorch interface) on gfx950 CK/ASM path.

  source /dockerx/TE_HD256/scripts/env_hd256.sh
  source /dockerx/NVTE_env.sh
  python3 bwd_reprod_pytorch.py [--seqs 2048,8192,20480] [--reps 5]
  python3 bwd_reprod_pytorch.py --cases thd
  python3 bwd_reprod_pytorch.py --cases dense,thd

Expected for a healthy flash-attention backward: ~2–2.5x the forward.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Literal, Optional

# Prefer ASM v3 paths; C++ gates fall back to CK when no kernel matches.
os.environ.setdefault("NVTE_CK_USES_FWD_V3", "1")
os.environ.setdefault("NVTE_CK_USES_BWD_V3", "1")
os.environ.setdefault("NVTE_CK_IS_V3_ATOMIC_FP32", "0")
os.environ.setdefault("NVTE_FUSED_ATTN", "1")
os.environ.setdefault("NVTE_FLASH_ATTN", "0")
os.environ.setdefault("NVTE_FUSED_ATTN_CK", "1")
os.environ.setdefault("NVTE_FUSED_ATTN_AOTRITON", "0")

import torch

import transformer_engine.pytorch  # noqa: F401 — loads transformer_engine_torch
from transformer_engine.pytorch import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends

Layout = Literal["bshd", "thd"]
MaskName = Literal["no_mask", "padding", "padding_causal"]

p = argparse.ArgumentParser()
p.add_argument("--seqs", default="2048,8192,20480")
p.add_argument("--reps", type=int, default=5)
p.add_argument("--heads", type=int, default=18)
p.add_argument("--batch", type=int, default=1)
p.add_argument(
    "--cases",
    default="dense,thd",
    help="comma-separated: dense (BSHD no mask), thd (THD varlen padding/causal)",
)
p.add_argument("--pad-ratio", type=float, default=0.3)
p.add_argument("--thd-heads", type=int, default=8)
p.add_argument("--thd-batch", type=int, default=2)
a = p.parse_args()

CASES = {c.strip() for c in a.cases.split(",") if c.strip()}
DTYPE = torch.bfloat16

_attention_backends["backend_selection_requires_update"] = True

print(
    "PyTorch",
    torch.__version__,
    "| device",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    flush=True,
)
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
    mask: MaskName,
    heads: int,
    heads_kv: int,
) -> str:
    """Mirror TE gfx950 ASM gates (fused_attn_ck.cpp); otherwise CK."""
    mha = heads == heads_kv
    is_padding = mask in ("padding", "padding_causal")
    if phase == "fwd":
        if not v3_enabled("NVTE_CK_USES_FWD_V3"):
            return "CK"
        if hd == 128:
            return "ASM"
        if hd == 256 and layout == "thd" and mha and is_padding:
            return "ASM"
        return "CK"

    if not v3_enabled("NVTE_CK_USES_BWD_V3"):
        return "CK"
    if hd == 128:
        return "ASM"
    if hd == 256 and layout == "thd" and mha and mask == "padding":
        return "ASM"
    return "CK"


def format_kernel_col(fwd_backend: str, bwd_backend: Optional[str]) -> str:
    if bwd_backend is None:
        return fwd_backend
    if fwd_backend == bwd_backend:
        return fwd_backend
    return f"{fwd_backend}/{bwd_backend}"


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fwd_fn: Callable[[], torch.Tensor]) -> float:
    sync()
    fwd_fn()
    sync()
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        fwd_fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def timeit_fwd_bwd(fwd_bwd_fn: Callable[[], None]) -> float:
    sync()
    fwd_bwd_fn()
    sync()
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        fwd_bwd_fn()
        sync()
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
    mask: MaskName


def make_attention(hd: int, heads: int, mask: MaskName, qkv_format: str) -> DotProductAttention:
    return DotProductAttention(
        num_attention_heads=heads,
        kv_channels=hd,
        attention_dropout=0.0,
        attn_mask_type=mask,
        qkv_format=qkv_format,
    ).to(dtype=DTYPE, device="cuda")


def make_bshd_tensors(batch: int, seq: int, heads: int, hd: int, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    shape = (batch, seq, heads, hd)
    q = torch.randn(shape, dtype=DTYPE, device="cuda", generator=gen)
    k = torch.randn(shape, dtype=DTYPE, device="cuda", generator=gen)
    v = torch.randn(shape, dtype=DTYPE, device="cuda", generator=gen)
    return q, k, v


def make_thd_tensors(batch: int, max_seqlen: int, heads: int, hd: int, pad_ratio: float, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    pad_len = int(max_seqlen * pad_ratio) if pad_ratio > 0 else 0
    valid_len = max_seqlen - pad_len
    total = batch * max_seqlen
    shape = (total, heads, hd)
    q = torch.randn(shape, dtype=DTYPE, device="cuda", generator=gen)
    k = torch.randn(shape, dtype=DTYPE, device="cuda", generator=gen)
    v = torch.randn(shape, dtype=DTYPE, device="cuda", generator=gen)
    if pad_len > 0:
        for i in range(batch):
            sl = slice(i * max_seqlen + valid_len, (i + 1) * max_seqlen)
            q[sl].zero_()
            k[sl].zero_()
            v[sl].zero_()

    cu_seqlens_q_padded = torch.arange(
        0, (batch + 1) * max_seqlen, max_seqlen, dtype=torch.int32, device="cuda"
    )
    cu_seqlens_kv_padded = cu_seqlens_q_padded.clone()
    cu_seqlens_q = torch.arange(
        0, (batch + 1) * valid_len, valid_len, dtype=torch.int32, device="cuda"
    )
    cu_seqlens_kv = cu_seqlens_q.clone()
    return q, k, v, cu_seqlens_q, cu_seqlens_kv, cu_seqlens_q_padded, cu_seqlens_kv_padded


def bench_dense_bshd(hd: int, seq: int) -> None:
    mask: MaskName = "no_mask"
    kernel = format_kernel_col(
        routed_backend("fwd", hd, "bshd", mask, a.heads, a.heads),
        routed_backend("bwd", hd, "bshd", mask, a.heads, a.heads),
    )
    attn = make_attention(hd, a.heads, mask, "bshd")
    q, k, v = make_bshd_tensors(a.batch, seq, a.heads, hd, hd * 10_000 + seq)

    try:
        tf_ = timeit(lambda: attn(q, k, v, qkv_format="bshd"))

        qg = q.detach().clone().requires_grad_(True)
        kg = k.detach().clone().requires_grad_(True)
        vg = v.detach().clone().requires_grad_(True)

        def fwd_bwd():
            out = attn(qg, kg, vg, qkv_format="bshd")
            out.float().sum().backward()

        tb = timeit_fwd_bwd(fwd_bwd)
        print_row("bshd_nomask", hd, seq, tf_, tb, kernel)
    except Exception as e:  # noqa: BLE001
        print_row("bshd_nomask", hd, seq, None, None, kernel, err=f"{type(e).__name__}: {str(e)[:100]}")


def bench_thd(case: ThdCase, hd: int, seq: int) -> None:
    kernel = format_kernel_col(
        routed_backend("fwd", hd, "thd", case.mask, a.thd_heads, a.thd_heads),
        routed_backend("bwd", hd, "thd", case.mask, a.thd_heads, a.thd_heads),
    )
    attn = make_attention(hd, a.thd_heads, case.mask, "thd")
    q, k, v, cu_q, cu_kv, cu_q_pad, cu_kv_pad = make_thd_tensors(
        a.thd_batch, seq, a.thd_heads, hd, a.pad_ratio, hd * 100 + seq
    )
    fwd_kwargs = {
        "qkv_format": "thd",
        "cu_seqlens_q": cu_q,
        "cu_seqlens_kv": cu_kv,
        "cu_seqlens_q_padded": cu_q_pad,
        "cu_seqlens_kv_padded": cu_kv_pad,
        "max_seqlen_q": seq,
        "max_seqlen_kv": seq,
        "pad_between_seqs": True,
    }

    try:
        tf_ = timeit(lambda: attn(q, k, v, **fwd_kwargs))

        qg = q.detach().clone().requires_grad_(True)
        kg = k.detach().clone().requires_grad_(True)
        vg = v.detach().clone().requires_grad_(True)

        def fwd_bwd():
            out = attn(qg, kg, vg, **fwd_kwargs)
            out.float().sum().backward()

        tb = timeit_fwd_bwd(fwd_bwd)
        print_row(case.name, hd, seq, tf_, tb, kernel)
    except Exception as e:  # noqa: BLE001
        print_row(case.name, hd, seq, None, None, kernel, err=f"{type(e).__name__}: {str(e)[:100]}")


THD_CASES = (
    ThdCase("thd_padding", "padding"),
    ThdCase("thd_causal", "padding_causal"),
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
