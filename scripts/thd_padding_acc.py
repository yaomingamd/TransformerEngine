#!/usr/bin/env python3
"""Compare THD fused-attn numerics: gfx950 ASM (v3) vs CK fallback.

ASM:  NVTE_CK_USES_FWD_V3=1 (+ BWD_V3=1 when running backward)
CK:   NVTE_CK_USES_FWD_V3=0 NVTE_CK_USES_BWD_V3=0

Each backend runs in a fresh subprocess so env routing is isolated from JAX caches.

  source /dockerx/NVTE_env.sh
  python3 thd_padding_acc.py [--head-dims 128,256] [--seqs 512,2048]
  python3 thd_padding_acc.py --masks padding          # fwd+bwd
  python3 thd_padding_acc.py --masks causal           # fwd only
  python3 thd_padding_acc.py --masks padding,causal
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--head-dims", default="128,256")
p.add_argument("--seqs", default="512,2048")
p.add_argument("--batch", type=int, default=2)
p.add_argument("--heads", type=int, default=8)
p.add_argument("--pad-ratio", type=float, default=0.3)
p.add_argument("--seed", type=int, default=0)
p.add_argument(
    "--masks",
    default="padding,causal",
    help="comma-separated: padding (fwd+bwd), causal (fwd only)",
)
p.add_argument("--worker", choices=("asm", "ck"), help=argparse.SUPPRESS)
p.add_argument("--out", type=Path, help=argparse.SUPPRESS)
p.add_argument("--hd", type=int, help=argparse.SUPPRESS)
p.add_argument("--seq", type=int, help=argparse.SUPPRESS)
p.add_argument("--mask", choices=("padding", "causal"), help=argparse.SUPPRESS)
p.add_argument("--fwd-only", action="store_true", help=argparse.SUPPRESS)
a = p.parse_args()

BF16_RTOL = 0.0625
BF16_ATOL = 0.01
SCRIPT = Path(__file__).resolve()
MASKS = {m.strip() for m in a.masks.split(",") if m.strip()}


def set_backend(mode: str, fwd_only: bool) -> None:
    if mode == "asm":
        os.environ["NVTE_CK_USES_FWD_V3"] = "1"
        os.environ["NVTE_CK_USES_BWD_V3"] = "0" if fwd_only else "1"
        os.environ["NVTE_CK_IS_V3_ATOMIC_FP32"] = "0"
    elif mode == "ck":
        os.environ["NVTE_CK_USES_FWD_V3"] = "0"
        os.environ["NVTE_CK_USES_BWD_V3"] = "0"
        os.environ["NVTE_CK_IS_V3_ATOMIC_FP32"] = "0"
    else:
        raise ValueError(mode)


def get_seqlens_and_offsets(segment_ids):
    import jax
    import jax.numpy as jnp

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


def worker_main() -> None:
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

    set_backend(a.worker, a.fwd_only)
    hd, seq = a.hd, a.seq
    mask_type = (
        AttnMaskType.PADDING_MASK
        if a.mask == "padding"
        else AttnMaskType.PADDING_CAUSAL_MASK
    )

    if a.pad_ratio > 0:
        pad_len = int(seq * a.pad_ratio)
        valid_len = seq - pad_len
        segment_ids = jnp.concatenate(
            [
                jnp.ones((a.batch, valid_len), dtype=jnp.int32),
                jnp.zeros((a.batch, pad_len), dtype=jnp.int32),
            ],
            axis=-1,
        )
    else:
        segment_ids = jnp.ones((a.batch, seq), dtype=jnp.int32)

    seqlens, offsets = get_seqlens_and_offsets(segment_ids)
    seq_desc = SequenceDescriptor.from_seqlens_and_offsets(
        (seqlens, seqlens), (offsets, offsets)
    )
    valid = segment_ids != 0

    key = jax.random.PRNGKey(a.seed + hd * 1000 + seq + (1 if a.mask == "causal" else 0))
    k1, k2, k3 = jax.random.split(key, 3)
    shape = (a.batch, seq, a.heads, hd)
    q = jax.random.normal(k1, shape, jnp.bfloat16)
    k = jax.random.normal(k2, shape, jnp.bfloat16)
    v = jax.random.normal(k3, shape, jnp.bfloat16)
    scale = 1.0 / (hd**0.5)

    def attn(q, k, v):
        return fused_attn(
            (q, k, v),
            None,
            seq_desc,
            None,
            AttnBiasType.NO_BIAS,
            mask_type,
            QKVLayout.THD_THD_THD,
            AttnSoftmaxType.VANILLA_SOFTMAX,
            scale,
            0.0,
            True,
            max_segments_per_seq=2,
        )

    def masked_sum(out):
        out = jnp.where(valid[..., None, None], out, 0)
        return jnp.sum(out.astype(jnp.float32))

    fwd = jax.jit(attn)
    out = fwd(q, k, v)
    if a.fwd_only:
        dq = dk = dv = jnp.zeros_like(q)
    else:
        bwd = jax.jit(jax.value_and_grad(lambda q, k, v: masked_sum(attn(q, k, v)), (0, 1, 2)))
        _, (dq, dk, dv) = bwd(q, k, v)
    jax.block_until_ready((out, dq, dk, dv))

    def to_f32(x):
        return np.asarray(x, dtype=np.float32)

    np.savez(
        a.out,
        out=to_f32(out),
        dq=to_f32(dq),
        dk=to_f32(dk),
        dv=to_f32(dv),
        valid=np.asarray(valid, dtype=bool),
    )


def run_worker(mode: str, hd: int, seq: int, mask: str, fwd_only: bool, out: Path) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--worker",
        mode,
        "--out",
        str(out),
        "--hd",
        str(hd),
        "--seq",
        str(seq),
        "--mask",
        mask,
        "--batch",
        str(a.batch),
        "--heads",
        str(a.heads),
        "--pad-ratio",
        str(a.pad_ratio),
        "--seed",
        str(a.seed),
    ]
    if fwd_only:
        cmd.append("--fwd-only")
    subprocess.run(cmd, check=True)


def compare_arrays(name: str, asm: np.ndarray, ck: np.ndarray, valid: np.ndarray) -> bool:
    mask = np.asarray(valid, dtype=bool)[..., None, None]
    asm_v = np.where(mask, np.asarray(asm, dtype=np.float32), 0.0)
    ck_v = np.where(mask, np.asarray(ck, dtype=np.float32), 0.0)
    diff = np.abs(asm_v - ck_v)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    denom = np.maximum(np.abs(ck_v), 1e-6)
    max_rel = float(np.max(diff / denom))
    passed = max_diff <= BF16_ATOL or max_rel <= BF16_RTOL
    status = "PASS" if passed else "FAIL"
    print(
        f"  {name:8} {status}  max_abs={max_diff:.6f}  mean_abs={mean_diff:.6f}  "
        f"max_rel={max_rel:.6f}  (atol={BF16_ATOL}, rtol={BF16_RTOL})"
    )
    return passed


def run_case(mask: str, hd: int, seq: int) -> bool:
    fwd_only = mask == "causal"
    case_name = f"thd_{mask}" + (" (fwd only)" if fwd_only else "")

    with tempfile.TemporaryDirectory() as tmp:
        asm_path = Path(tmp) / "asm.npz"
        ck_path = Path(tmp) / "ck.npz"
        run_worker("asm", hd, seq, mask, fwd_only, asm_path)
        run_worker("ck", hd, seq, mask, fwd_only, ck_path)
        asm = np.load(asm_path)
        ck = np.load(ck_path)
        valid = asm["valid"]

    ok = True
    print(f"\n=== {case_name} hd={hd} seq={seq} b={a.batch} h={a.heads} pad={a.pad_ratio} ===")
    labels = {"out": "fwd_out"}
    if not fwd_only:
        labels.update({"dq": "dQ", "dk": "dK", "dv": "dV"})
    for name, label in labels.items():
        ok = compare_arrays(label, asm[name], ck[name], valid) and ok
    return ok


def main() -> int:
    if a.worker:
        worker_main()
        return 0

    import jax

    print("jax", jax.__version__, "|", jax.devices()[0], flush=True)
    all_ok = True
    for mask in ("padding", "causal"):
        if mask not in MASKS:
            continue
        for hd in [int(x) for x in a.head_dims.split(",")]:
            for seq in [int(x) for x in a.seqs.split(",")]:
                try:
                    all_ok = run_case(mask, hd, seq) and all_ok
                except Exception as exc:  # noqa: BLE001
                    all_ok = False
                    print(f"\n=== thd_{mask} hd={hd} seq={seq} ERROR: {exc} ===", flush=True)

    print("\n" + ("ALL PASS" if all_ok else "SOME FAILED"), flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
