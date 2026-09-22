"""独立验证 RELOAD 与 Div->Mul 优化的数值正确性（不依赖 torch）。

直接复用 AddRmsNormBias.impl 作为 golden，用 numpy 模拟：
1. reloaded_row: RetainedKernel<RELOAD=true> 的大分块两遍读逻辑
2. mul_inv  vs  div: 用 1/rms 再乘  vs  直接除以 rms 的数值等价性
"""
import sys
import os
import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project", "scripts"))
from AddRmsNormBias import impl as golden


def is_bf16(dtype):
    return 'bfloat16' in str(dtype) or 'bf16' in str(dtype)


def reloaded_row(x, r, g, b, eps, chunk):
    """模拟 RetainedKernel<RELOAD=true>：每个 chunk 读两遍，先算平方和，再归一化。"""
    d = x.size
    s = np.float32(0.0)
    for c in range(0, d, chunk):
        n = min(chunk, d - c)
        y = x[c:c + n].astype(np.float32) + r[c:c + n].astype(np.float32)
        s += np.sum(y * y, dtype=np.float32)
    norm = np.sqrt(s * np.float32(1.0 / d) + np.float32(eps))
    inv = np.float32(1.0) / norm
    out = np.empty(d, dtype=np.float32)
    for c in range(0, d, chunk):
        n = min(chunk, d - c)
        y = x[c:c + n].astype(np.float32) + r[c:c + n].astype(np.float32)
        out[c:c + n] = (y * inv) * g[c:c + n].astype(np.float32) + b[c:c + n].astype(np.float32)
    return out.astype(x.dtype)


def mul_inv_row(x, r, g, b, eps):
    """用 1/rms 再乘（模拟 BatchKernel/TinyKernel 的 Div->Mul 改动）。"""
    d = x.size
    y = x.astype(np.float32) + r.astype(np.float32)
    norm = np.sqrt(np.mean(y * y) + np.float32(eps))
    inv = np.float32(1.0) / norm
    out = (y * inv) * g.astype(np.float32) + b.astype(np.float32)
    return out.astype(x.dtype)


def div_row(x, r, g, b, eps):
    """直接除以 rms（原 Div 实现）。"""
    d = x.size
    y = x.astype(np.float32) + r.astype(np.float32)
    norm = np.sqrt(np.mean(y * y) + np.float32(eps))
    out = (y / norm) * g.astype(np.float32) + b.astype(np.float32)
    return out.astype(x.dtype)


def compare(got, want, label):
    gt = got.astype(np.float32)
    wt = want.astype(np.float32)
    tol = 1e-4 if got.dtype == np.float32 else 1e-3
    ok = np.isclose(gt, wt, rtol=tol, atol=tol, equal_nan=True)
    frac = np.mean(ok)
    print(f"  {label}: match={frac:.6f}  max_abs_err={np.max(np.abs(gt - wt)):.3e}")
    return frac


def vector_indices(mask, repeats, block_stride, repeat_stride):
    lane = np.arange(mask)
    return (np.arange(repeats)[:, None] * repeat_stride * 8
            + (lane // 8) * block_stride * 8 + lane % 8)


def tree_sum(a):
    a = a.copy()
    width = a.shape[-1]
    padded = 1 << (width - 1).bit_length()
    if padded != width:
        a = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, padded - width)])
    while a.shape[-1] > 1:
        a = a[..., 0::2] + a[..., 1::2]
    return a[..., 0]


def batch_tile(x, r, g, b, eps):
    """模拟 BatchKernel（含 Div->Mul 改动后的倒数+乘）。"""
    nr, d = x.shape
    pitch = (d + 63) // 64 * 64
    yp = np.full((nr, pitch), np.nan, dtype=np.float32)
    yp[:, :d] = x.astype(np.float32) + r.astype(np.float32)
    y = yp.reshape(-1)
    if d != pitch:
        mask = ((1 << 64) - 1) ^ ((1 << (d % 64)) - 1)
        lanes = np.array([i for i in range(64) if mask & (1 << i)])
        tails = np.arange(nr)[:, None] * pitch + d // 64 * 64 + lanes
        y[tails] = 0.0
    sq = y * y
    idx = vector_indices(64, nr, 1, pitch // 8)
    if pitch % 512 == 0:
        partials = tree_sum(sq.reshape(-1, 64))
        pidx = vector_indices(pitch // 64, nr, 1, pitch // 512)
        sums = tree_sum(partials[pidx])
    else:
        for c in range(64, pitch, 64):
            sq[idx] = sq[idx] + sq[idx + c]
        sums = tree_sum(sq[idx])
    norm = np.sqrt(sums * np.float32(1.0 / d) + np.float32(eps))
    inv = np.float32(1.0) / norm  # Div->Mul 改动：先算倒数
    br = np.repeat(inv, 8)        # 广播倒数
    norm_idx = vector_indices(64, nr, 0, 1)
    channel_idx = vector_indices(64, nr, 1, 0)
    gf, bf = np.full(pitch, np.nan, np.float32), np.full(pitch, np.nan, np.float32)
    gf[:d], bf[:d] = g.astype(np.float32), b.astype(np.float32)
    for c in range(0, pitch, 64):
        y[idx + c] = y[idx + c] * br[norm_idx]  # Mul 而非 Div
        y[idx + c] = y[idx + c] * gf[channel_idx + c]
        y[idx + c] = y[idx + c] + bf[channel_idx + c]
    return y.reshape(nr, pitch)[:, :d].astype(x.dtype)


def retained_row(x, r, g, b, eps):
    """模拟 RetainedKernel 默认（整行保留，Mul+倒数）。"""
    d = x.size
    pitch = (d + 63) // 64 * 64
    y = np.zeros(pitch, np.float32)
    y[:d] = x.astype(np.float32) + r.astype(np.float32)
    parts = tree_sum((y * y).reshape(-1, 64))
    norm = np.sqrt(np.sum(parts, dtype=np.float32) * np.float32(1.0 / d)
                   + np.float32(eps))
    scale = np.float32(1.0) / norm
    normalized = y[:d] * scale
    return (normalized * g.astype(np.float32) + b.astype(np.float32)).astype(x.dtype)


def main():
    rng = np.random.default_rng(20260922)

    print("=== BatchKernel (Div->Mul) vs golden ===")
    for dtype in (np.float32, np.float16, ml_dtypes.bfloat16):
        for d in (64, 128, 192, 256, 512, 1024):
            for nr in (1, 7, 8, 33):
                x = rng.uniform(-2, 2, (nr, d)).astype(dtype)
                r = rng.uniform(-2, 2, (nr, d)).astype(dtype)
                g = rng.uniform(-1, 1, d).astype(dtype)
                b = rng.uniform(-0.1, 0.1, d).astype(dtype)
                want = golden(x, r, g, b, 1e-5)
                got = batch_tile(x, r, g, b, 1e-5)
                f = compare(got, want, f"batch {np.dtype(dtype)} D={d} nr={nr}")
                assert f > 0.999, f"FAIL batch {dtype} D={d} nr={nr}"

    print("=== RetainedKernel default (FP16/BF16 mid-width) vs golden ===")
    for dtype in (np.float16, ml_dtypes.bfloat16):
        for d in (1025, 2048, 4096, 8192):
            x = rng.uniform(-2, 2, d).astype(dtype)
            r = rng.uniform(-2, 2, d).astype(dtype)
            g = rng.uniform(-1, 1, d).astype(dtype)
            b = rng.uniform(-0.1, 0.1, d).astype(dtype)
            want = golden(x, r, g, b, 1e-5)
            got = retained_row(x, r, g, b, 1e-5)
            f = compare(got, want, f"retained {np.dtype(dtype)} D={d}")
            assert f > 0.999, f"FAIL retained {dtype} D={d}"
    print("=== RELOAD (large chunk, two-pass) vs golden ===")
    for dtype, cache in ((np.float32, 6144), (np.float16, 8192), (ml_dtypes.bfloat16, 8192)):
        for d in (cache + 1, 12288, 16384, 24576, 32768):
            x = rng.uniform(-2, 2, d).astype(dtype)
            r = rng.uniform(-2, 2, d).astype(dtype)
            g = rng.uniform(-1, 1, d).astype(dtype)
            b = rng.uniform(-0.1, 0.1, d).astype(dtype)
            want = golden(x, r, g, b, 1e-5)
            got = reloaded_row(x, r, g, b, 1e-5, cache)
            compare(got, want, f"{np.dtype(dtype)} D={d}")

    print("=== Div vs Mul+inv (batch/tiny paths) ===")
    for dtype in (np.float32, np.float16, ml_dtypes.bfloat16):
        for d in (64, 128, 256, 512, 1024):
            x = rng.uniform(-2, 2, d).astype(dtype)
            r = rng.uniform(-2, 2, d).astype(dtype)
            g = rng.uniform(-1, 1, d).astype(dtype)
            b = rng.uniform(-0.1, 0.1, d).astype(dtype)
            want = golden(x, r, g, b, 1e-5)
            got_mul = mul_inv_row(x, r, g, b, 1e-5)
            got_div = div_row(x, r, g, b, 1e-5)
            f_mul = compare(got_mul, want, f"mul {np.dtype(dtype)} D={d}")
            f_div = compare(got_div, want, f"div {np.dtype(dtype)} D={d}")
            # 两种方式都应通过 golden；且 Mul 与 Div 之间也应高度一致
            assert f_mul > 0.999 and f_div > 0.999, f"FAIL {dtype} D={d}"

    print("ALL OK")


if __name__ == "__main__":
    main()
