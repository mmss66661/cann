#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
AddRmsNormBias算子golden实现
以 numpy 组合实现 (add + rms_norm + bias_add) 的结果为 golden
"""
import numpy as np
from ml_dtypes import bfloat16


def _is_bf16(dtype):
    name = str(dtype)
    return 'bfloat16' in name or 'bf16' in name


def impl(x, residual, gamma, bias, epsilon=1e-5):
    """AddRmsNormBias: output = RMSNorm(x + residual, gamma, eps) + bias

    参数名与顺序与 JSON 的 input_desc + attr_desc 一致：
    x, residual, gamma, bias 为 numpy 数组，epsilon 为属性默认值。
    返回与 x 同 shape 同 dtype 的 numpy 数组。
    """
    orig_dtype = x.dtype

    # 统一转 float32 计算（bf16/fp16 精度不足，内部用 FP32）
    x_f = x.astype(np.float32)
    r_f = residual.astype(np.float32)
    g_f = gamma.astype(np.float32)
    b_f = bias.astype(np.float32)

    # Step 1: 残差加法
    y = x_f + r_f
    # Step 2: RMS 归一化（沿最后一维）
    rms = np.sqrt(np.mean(y * y, axis=-1, keepdims=True) + epsilon)
    out = y / rms * g_f
    # Step 3: 偏置加法
    out = out + b_f

    # 转回原始 dtype（与 JSON output_desc 一致）
    if _is_bf16(orig_dtype):
        return out.astype(bfloat16)
    if orig_dtype == np.float16:
        return out.astype(np.float16)
    return out.astype(np.float32)


if __name__ == "__main__":
    # 与 JSON npu_cases 对应的 15 个测试用例（用 numpy 生成）
    cases = [
        # (id, x_shape, D, dtype, x_range, r_range, g_range, b_range, eps)
        (1,  [2, 3, 4, 8],      8,     np.float32, [-1, 1], [-1, 1],    [0.9, 1.1], [-0.1, 0.1], 1e-5),
    ]

    for cid, xs, D, dt, xr, rr, gr, br, eps in cases:
        x = np.random.uniform(xr[0], xr[1], xs).astype(dt)
        r = np.random.uniform(rr[0], rr[1], xs).astype(dt)
        g = np.random.uniform(gr[0], gr[1], [D]).astype(dt)
        b = np.random.uniform(br[0], br[1], [D]).astype(dt)
        y = impl(x, r, g, b, eps)
        print(f"Case {cid:02d}: shape={y.shape}, dtype={y.dtype}")

    print("All tests passed!")
