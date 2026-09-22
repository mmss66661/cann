"""CPU checks of the fast paths' vector addresses, arithmetic and UB bounds.

This models the documented repeat strides, not the Ascend instruction latency,
compiler, synchronization or device rounding. Online validation remains required.
"""
import unittest

import numpy as np
import ml_dtypes

from test_design import downloaded_golden, partition


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
    br = np.repeat(norm, 8)
    norm_idx = vector_indices(64, nr, 0, 1)
    channel_idx = vector_indices(64, nr, 1, 0)
    gf, bf = np.full(pitch, np.nan, np.float32), np.full(pitch, np.nan, np.float32)
    gf[:d], bf[:d] = g.astype(np.float32), b.astype(np.float32)
    for c in range(0, pitch, 64):
        y[idx + c] = y[idx + c] / br[norm_idx]
        y[idx + c] = y[idx + c] * gf[channel_idx + c]
        y[idx + c] = y[idx + c] + bf[channel_idx + c]
    return y.reshape(nr, pitch)[:, :d].astype(x.dtype)


def retained_row(x, r, g, b, eps, direct_div=False):
    d = x.size
    pitch = (d + 63) // 64 * 64
    y = np.zeros(pitch, np.float32)
    y[:d] = x.astype(np.float32) + r.astype(np.float32)
    parts = tree_sum((y * y).reshape(-1, 64))
    norm = np.sqrt(np.sum(parts, dtype=np.float32) * np.float32(1.0 / d)
                   + np.float32(eps))
    scale = norm if direct_div else np.float32(1.0) / norm
    normalized = y[:d] / scale if direct_div else y[:d] * scale
    return (normalized * g.astype(np.float32) + b.astype(np.float32)).astype(x.dtype)


def reloaded_row(x, r, g, b, eps, chunk):
    """Model RetainedKernel<RELOAD>: each chunk is read in both passes."""
    d = x.size
    sum_of_squares = np.float32(0.0)
    for c in range(0, d, chunk):
        n = min(chunk, d - c)
        y = x[c:c + n].astype(np.float32) + r[c:c + n].astype(np.float32)
        sum_of_squares += np.sum(y * y, dtype=np.float32)
    norm = np.sqrt(sum_of_squares * np.float32(1.0 / d) + np.float32(eps))
    out = np.empty(d, dtype=np.float32)
    for c in range(0, d, chunk):
        n = min(chunk, d - c)
        y = x[c:c + n].astype(np.float32) + r[c:c + n].astype(np.float32)
        out[c:c + n] = (y / norm * g[c:c + n].astype(np.float32)
                        + b[c:c + n].astype(np.float32))
    return out.astype(x.dtype)


def wide_batch_tile(x, r, g, b, eps):
    nr, d = x.shape
    p = (d + 63) // 64 * 64
    y = np.zeros((nr, p), np.float32)
    y[:, :d] = x.astype(np.float32) + r.astype(np.float32)
    sq = y * y
    norm = np.ones((nr, 8), np.float32)
    if p % 512 == 0 and p <= 4096:
        parts = tree_sum(sq.reshape(-1, 64))
        pidx = vector_indices(p // 64, nr, 1, p // 512)
        norm[:, 0] = tree_sum(parts[pidx])
    else:
        for row in range(nr):
            norm[row, 0] = np.sum(tree_sum(sq[row].reshape(-1, 64)), dtype=np.float32)
    inv = np.float32(1.0) / np.sqrt(norm * np.float32(1.0 / d) + np.float32(eps))
    # Brcb expands each of the 8 values per row into a separate 8-float block.
    br = np.repeat(inv.reshape(-1), 8)
    gpad, bpad = np.zeros(p, np.float32), np.zeros(p, np.float32)
    gpad[:d], bpad[:d] = g.astype(np.float32), b.astype(np.float32)
    laneidx = vector_indices(64, p // 64, 1, 8)
    bridx = vector_indices(64, p // 64, 0, 0)
    for row in range(nr):
        y[row, laneidx] *= br[bridx + row * 64]
        y[row] *= gpad
        y[row] += bpad
    return y[:, :d].astype(x.dtype)


def compare(test, got, golden):
    tol = 1e-4 if got.dtype == np.float32 else 1e-3
    ok = np.isclose(got.astype(np.float32), golden.astype(np.float32),
                    rtol=tol, atol=tol, equal_nan=True)
    # The downloaded verifier allows at most 0.1% mismatches.
    test.assertLessEqual(1.0 - np.mean(ok), 0.001)


class TestFastPaths(unittest.TestCase):
    def test_wide_batch_broadcast_and_partial_batches_against_golden(self):
        rng = np.random.default_rng(20260922)
        for dtype in (np.float32, np.float16, ml_dtypes.bfloat16):
            limit = 6144 if dtype == np.float32 else 8192
            for d in (1025, 1088, 1473, 1535, 1536, 1984, 1985, 2048, 2049,
                      2497, 2559, 2560, 2752, 3072, 3457, 3583, 3584,
                      4033, 4095, 4096, 4097, 6144, 6145, 8191, 8192):
                if d > limit:
                    continue
                tile_rows = 8192 // ((d + 63) // 64 * 64)
                for nr in (1, tile_rows, tile_rows + 1, 2 * tile_rows + 1):
                    with self.subTest(dtype=dtype, d=d, rows=nr):
                        x = rng.uniform(-2, 2, (nr, d)).astype(dtype)
                        r = rng.uniform(-2, 2, (nr, d)).astype(dtype)
                        g = rng.uniform(-1, 1, d).astype(dtype)
                        b = rng.uniform(-0.1, 0.1, d).astype(dtype)
                        output = np.empty_like(x)
                        for first, last in partition(nr, d, x.itemsize, 3):
                            for row in range(first, last, tile_rows):
                                stop = min(last, row + tile_rows)
                                output[row:stop] = wide_batch_tile(x[row:stop], r[row:stop], g, b, 1e-5)
                        compare(self, output, downloaded_golden(x, r, g, b))

    def test_batch_addresses_and_tail_batches_against_golden(self):
        rng = np.random.default_rng(20260921)
        for dtype in (np.float32, np.float16, ml_dtypes.bfloat16):
            for d in (8, 63, 64, 65, 71, 72, 79, 80, 95, 96, 127, 128,
                      129, 191, 192, 193, 255, 256, 320, 575, 576, 577, 960,
                      1000, 1023, 1024):
                capacity = 4096 if dtype == np.float32 else 8192
                tile_rows = capacity // ((d + 63) // 64 * 64)
                for nr in (1, 7, 8, 9, tile_rows, tile_rows + 1, 2 * tile_rows + 3):
                    with self.subTest(dtype=dtype, d=d, rows=nr):
                        x = rng.uniform(-2, 2, (nr, d)).astype(dtype)
                        r = rng.uniform(-2, 2, (nr, d)).astype(dtype)
                        g = rng.uniform(-1, 1, d).astype(dtype)
                        b = rng.uniform(-0.1, 0.1, d).astype(dtype)
                        output = np.empty_like(x)
                        for begin, end in partition(nr, d, x.itemsize, 3):
                            for row in range(begin, end, tile_rows):
                                stop = min(end, row + tile_rows)
                                output[row:stop] = batch_tile(x[row:stop], r[row:stop], g, b, 1e-5)
                        compare(self, output, downloaded_golden(x, r, g, b))

    def test_retained_rows_and_chunk_boundaries_against_golden(self):
        rng = np.random.default_rng(42)
        for dtype in (np.float32, np.float16, ml_dtypes.bfloat16):
            for d in (1025, 1088, 2047, 2048, 2049, 4095, 4096, 4097,
                      6143, 6144, 6145, 6208, 8191, 8192, 8193, 8256,
                      16383, 16384, 16385, 32704, 32767, 32768):
                with self.subTest(dtype=dtype, d=d):
                    x = rng.uniform(-2, 2, d).astype(dtype)
                    r = rng.uniform(-2, 2, d).astype(dtype)
                    g = rng.uniform(-1, 1, d).astype(dtype)
                    b = rng.uniform(-0.1, 0.1, d).astype(dtype)
                    compare(self, retained_row(x, r, g, b, 1e-6),
                            downloaded_golden(x, r, g, b, 1e-6))

    def test_reloaded_long_rows_and_direct_div_against_golden(self):
        rng = np.random.default_rng(20260923)
        for dtype in (np.float32, np.float16, ml_dtypes.bfloat16):
            cache = 6144 if dtype == np.float32 else 8192
            for d in (cache + 1, cache + 65, 12287, 12288, 12289,
                      16383, 16384, 24576, 32767, 32768):
                with self.subTest(dtype=dtype, d=d):
                    x = rng.uniform(-2, 2, d).astype(dtype)
                    r = rng.uniform(-2, 2, d).astype(dtype)
                    g = rng.uniform(-1, 1, d).astype(dtype)
                    b = rng.uniform(-0.1, 0.1, d).astype(dtype)
                    want = downloaded_golden(x, r, g, b, 1e-6)
                    got = reloaded_row(x, r, g, b, 1e-6, cache)
                    compare(self, got, want)
                    if d <= cache:
                        direct = retained_row(x, r, g, b, 1e-6, direct_div=True)
                        compare(self, direct, want)

    def test_all_dimensions_fit_192k_ub_and_instruction_limits(self):
        peak = 0
        for size in (2, 4):
            for d in range(1, 32769):
                pitch = (d + 63) // 64 * 64
                if d <= 1024:
                    nr = (4096 if size == 4 else 8192) // pitch
                    tile = nr * pitch
                    padded = (nr + 7) // 8 * 8
                    br_bytes = max(padded * 32, tile // 64 * 4)
                    used = 6 * tile * size + 8 * tile + 8 * pitch + 4 * padded + br_bytes
                    self.assertLessEqual(nr, 255)
                    self.assertLessEqual(pitch // 8, 255)
                    self.assertLessEqual(tile // 64, 255)
                    if pitch % 512 == 0:
                        self.assertLessEqual(nr * pitch // 64 * 4, br_bytes)
                elif d <= (6144 if size == 4 else 8192):
                    nr = 8192 // pitch
                    tile = nr * pitch
                    used = 3 * size * tile + 4 * tile + 8 * pitch + 2560 + 288 * nr
                    if size != 4:
                        used += 4 * tile
                    self.assertGreaterEqual(nr, 1)
                    self.assertLessEqual(pitch // 64, 128)
                    self.assertLessEqual(nr * 8 * 4, 2048)
                    self.assertLessEqual(nr, 255)
                else:
                    chunk = pitch if pitch <= (6144 if size == 4 else 8192) else 2048
                    used = 3 * chunk * size + pitch * 4 + chunk * 12 + 4096 + 32 + 256
                    self.assertLessEqual(chunk // 64, 255)
                    self.assertLessEqual(pitch // 64, 512)
                    for c in range(0, d, chunk):
                        count = min(chunk, d - c)
                        aligned = (count + 63) // 64 * 64
                        self.assertLessEqual(aligned, chunk)
                        self.assertLessEqual(c // 64 + aligned // 64, 512)
                    reload_chunk = min(pitch, 6144 if size == 4 else 8192)
                    reload_used = (3 * reload_chunk * size + 4 * reload_chunk
                                   + 4096 + 32 + 256)
                    self.assertLessEqual(reload_used, 192 * 1024)
                if d <= 4096:
                    self.assertLessEqual((5 * size + 20) * pitch + 288, 192 * 1024)
                peak = max(peak, used)
                self.assertLessEqual(used, 192 * 1024, (size, d, used))
        print(f"Fast-path peak modeled UB allocation: {peak} / {192 * 1024} bytes")

    def test_padded_dma_stride_and_exact_output_extent(self):
        for size in (2, 4):
            for d in (*range(1, 1025), 1025, 2047, 2049, 4095, 4097,
                      6145, 8193, 16385, 32767, 32768):
                pitch = (d + 63) // 64 * 64
                width_bytes = d * size
                gap_blocks = pitch * size // 32 - (width_bytes + 31) // 32
                step_bytes = ((width_bytes + 31) // 32 + gap_blocks) * 32
                self.assertEqual(step_bytes, pitch * size)
                nr = 3
                packed = np.arange(nr * width_bytes, dtype=np.int64).astype(np.uint8)
                ub = np.full(nr * step_bytes, 0xA5, np.uint8)
                for row in range(nr):
                    ub[row * step_bytes:row * step_bytes + width_bytes] = packed[
                        row * width_bytes:(row + 1) * width_bytes]
                output = np.full(nr * width_bytes + 32, 0xCC, np.uint8)
                for row in range(nr):
                    output[row * width_bytes:(row + 1) * width_bytes] = ub[
                        row * step_bytes:row * step_bytes + width_bytes]
                np.testing.assert_array_equal(output[:-32], packed)
                np.testing.assert_array_equal(output[-32:], 0xCC)


if __name__ == '__main__':
    unittest.main()
