"""Properties of v9's cached parameters and packed RMS coefficient layout.

These tests exercise the address mapping and FP32 formula on CPU, not CANN
compilation, device instructions or asynchronous execution.
"""
import unittest

import ml_dtypes
import numpy as np

from test_design import downloaded_golden
from test_fast_paths import compare, tree_sum, vector_indices


def expand_parameters(g, b, tile_rows):
    d = len(g)
    p = (d + 63) // 64 * 64
    tmp = np.full(2 * p, np.nan, np.float32)
    tmp[:d] = g.astype(np.float32)
    tmp[p:p + d] = b.astype(np.float32)
    gc = np.full(tile_rows * p, np.nan, np.float32)
    bc = gc.copy()
    dst = vector_indices(64, tile_rows, 1, p // 8)
    src = vector_indices(64, tile_rows, 1, 0)
    for c in range(0, p, 64):
        gc[dst + c] = tmp[src + c] + np.float32(0)
        bc[dst + c] = tmp[src + p + c] + np.float32(0)
    return gc, bc


def expand_rms(norm, p):
    nr = len(norm)
    br = np.repeat(norm, 8)
    coeff_width = p // 8
    coeff = np.full(nr * coeff_width, np.nan, np.float32)
    for c in range(0, coeff_width, 64):
        count = min(64, coeff_width - c)
        dst = vector_indices(count, nr, 1, p // 64) + c
        src = vector_indices(count, nr, 0, 1)
        coeff[dst] = br[src] + np.float32(0)
    div_src = vector_indices(64, nr * p // 64, 0, 1)
    return coeff[div_src].reshape(nr, p)


def dense_tile(x, r, g, b, eps, cached):
    nr, d = x.shape
    p = (d + 63) // 64 * 64
    y = np.zeros((nr, p), np.float32)
    y[:, :d] = x.astype(np.float32) + r.astype(np.float32)
    sq = y * y
    if p % 512 == 0:
        sums = tree_sum(tree_sum(sq.reshape(nr, -1, 64)))
    else:
        folded = sq[:, :64].copy()
        for c in range(64, p, 64):
            folded += sq[:, c:c + 64]
        sums = tree_sum(folded)
    norm = np.sqrt(sums * np.float32(1.0 / d) + np.float32(eps))
    scale = expand_rms(norm, p)
    gc, bc = (a[:nr * p].reshape(nr, p) for a in cached)
    return ((y / scale) * gc + bc)[:, :d].astype(x.dtype)


class TestDenseBatch(unittest.TestCase):
    def test_layout_for_every_width_and_partial_batch(self):
        for d in range(129, 1025):
            p = (d + 63) // 64 * 64
            nr = 6144 // p
            g = np.arange(d, dtype=np.float32)
            b = -g - 3
            gc, bc = expand_parameters(g, b, nr)
            np.testing.assert_array_equal(gc.reshape(nr, p)[:, :d],
                                          np.broadcast_to(g, (nr, d)))
            np.testing.assert_array_equal(bc.reshape(nr, p)[:, :d],
                                          np.broadcast_to(b, (nr, d)))
            for actual in (1, max(1, nr - 1), nr):
                norm = np.arange(1, actual + 1, dtype=np.float32)
                got = expand_rms(norm, p)
                np.testing.assert_array_equal(got, np.broadcast_to(norm[:, None], got.shape))

    def test_ub_and_vector_limits(self):
        peak = 0
        for size in (2, 4):
            for d in range(129, 1025):
                p = (d + 63) // 64 * 64
                nr = (4096 if size == 4 else 6144) // p
                tile = nr * p
                padded_rows = (nr + 7) // 8 * 8
                used = (6 * size * tile + (12 if size == 4 else 16) * tile
                        + padded_rows * 4 + max(tile // 64 * 4, padded_rows * 32))
                self.assertLessEqual(used, 192 * 1024)
                self.assertGreaterEqual(tile, 2 * p)
                self.assertLessEqual(tile // 64, 255)
                self.assertLessEqual(nr, 255)
                self.assertLessEqual(p // 8, 255)
                self.assertLessEqual(nr * p // 8, tile)
                peak = max(peak, used)
        print(f"Dense batch peak explicit UB: {peak} / {192 * 1024} bytes")

    def test_arithmetic_and_cached_parameters_across_tiles(self):
        rng = np.random.default_rng(2026092214)
        for dtype in (np.float32, np.float16, ml_dtypes.bfloat16):
            for d in (129, 191, 192, 193, 255, 256, 257, 320, 384,
                      448, 511, 512, 513, 575, 576, 577, 768, 960, 1000, 1023, 1024):
                p = (d + 63) // 64 * 64
                nr = (4096 if dtype == np.float32 else 6144) // p
                x = rng.uniform(-2, 2, (2 * nr + 1, d)).astype(dtype)
                r = rng.uniform(-2, 2, x.shape).astype(dtype)
                g = rng.uniform(-1, 1, d).astype(dtype)
                b = rng.uniform(-0.1, 0.1, d).astype(dtype)
                cache = expand_parameters(g, b, nr)
                copies = tuple(a.copy() for a in cache)
                out = np.empty_like(x)
                for start in range(0, len(x), nr):
                    end = min(start + nr, len(x))
                    out[start:end] = dense_tile(x[start:end], r[start:end], g, b, 1e-5, cache)
                compare(self, out, downloaded_golden(x, r, g, b))
                for before, after in zip(copies, cache):
                    np.testing.assert_array_equal(before, after)

    def test_nonfinite_values_remain_row_local(self):
        for d in (192, 576, 1024):
            x = np.ones((4, d), np.float32)
            r = np.zeros_like(x)
            x[0, 0] = np.nan
            x[1, 3] = np.inf
            x[2] = 0
            g, b = np.ones(d, np.float32), np.zeros(d, np.float32)
            with np.errstate(invalid='ignore'):
                got = dense_tile(x, r, g, b, 1e-5, expand_parameters(g, b, 4))
                want = downloaded_golden(x, r, g, b)
            np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4, equal_nan=True)


if __name__ == '__main__':
    unittest.main()
