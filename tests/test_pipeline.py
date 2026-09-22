"""CPU model of v8 ping-pong tiles; this does not emulate NPU synchronization."""
import unittest

import ml_dtypes
import numpy as np

from test_design import downloaded_golden, partition
from test_fast_paths import compare, wide_batch_tile


def pipeline_rows(d, size):
    if not 1024 < d <= (6144 if size == 4 else 8192):
        return 0
    p = (d + 63) // 64 * 64
    per_element = 6 * size + (4 if size == 4 else 8)
    return min((176 * 1024 - 8 * p - 3072) // (per_element * p + 288),
               255 * 64 // p)


class TestPipeline(unittest.TestCase):
    def test_ub_and_reduce_capacity_for_every_supported_width(self):
        peak = 0
        for size in (2, 4):
            for d in range(1025, 8193):
                nr = pipeline_rows(d, size)
                if nr == 0:
                    continue
                p = (d + 63) // 64 * 64
                tile = p * nr
                # Sizes of every allocation in WideBatchKernel<T, true>.
                allocations = [2 * tile * size] * 2 + [tile * size] * 2
                if size != 4:
                    allocations.append(tile * 4)
                allocations += [tile * 4, p * 4, p * 4, 1024, 2048, nr * 32, nr * 256]
                self.assertTrue(all(n % 32 == 0 for n in allocations))
                self.assertLessEqual(sum(allocations), 176 * 1024)
                self.assertLessEqual(tile // 64, 255)
                self.assertLessEqual(tile // 64 * 4, 1024)
                self.assertLessEqual(nr * 8 * 4, 2048)
                self.assertLessEqual(p // 64, 255)
                peak = max(peak, sum(allocations))
        print(f"Pipelined path peak explicit UB: {peak} / {192 * 1024} bytes")

    def test_prefetch_tail_tiles_and_output_ranges(self):
        rng = np.random.default_rng(20260922)
        for dtype in (np.float32, np.float16, ml_dtypes.bfloat16):
            for d in (1025, 1088, 1473, 1536, 1985, 2048, 2049, 2560,
                      3072, 3457, 3584, 4033, 4096, 4097, 4608, 4864,
                      4928, 5120, 6143, 6144, 6208, 8192):
                size = np.dtype(dtype).itemsize
                nr = pipeline_rows(d, size)
                if nr == 0:
                    continue
                p = (d + 63) // 64 * 64
                # Short core, exactly full tile, two tiles, partial final tile.
                for rows in (1, nr, 2 * nr, 3 * nr + 1, 6 * nr + 3):
                    with self.subTest(dtype=dtype, d=d, rows=rows):
                        x = rng.uniform(-2, 2, (rows, d)).astype(dtype)
                        r = rng.uniform(-2, 2, (rows, d)).astype(dtype)
                        g = rng.uniform(-1, 1, d).astype(dtype)
                        b = rng.uniform(-0.1, 0.1, d).astype(dtype)
                        out = np.full(rows * d + 16, 123, dtype=dtype)
                        written = np.zeros(rows, np.int32)
                        for begin, end in partition(rows, d, size, 3):
                            # Two raw input allocations, each holding x then r.
                            slots = np.full((2, 2, nr, p), np.nan, dtype=dtype)
                            live = [False, False]

                            def copy_in(tile_index, start):
                                slot = tile_index % 2
                                self.assertFalse(live[slot])
                                count = min(nr, end - start)
                                slots[slot, 0, :count, :d] = x[start:start + count]
                                slots[slot, 1, :count, :d] = r[start:start + count]
                                live[slot] = True

                            copy_in(0, begin)
                            for k, row in enumerate(range(begin, end, nr)):
                                if row + nr < end:
                                    copy_in(k + 1, row + nr)
                                count = min(nr, end - row)
                                slot = k % 2
                                self.assertTrue(live[slot])
                                got = wide_batch_tile(slots[slot, 0, :count, :d],
                                                      slots[slot, 1, :count, :d],
                                                      g, b, 1e-5)
                                out[row * d:(row + count) * d] = got.reshape(-1)
                                written[row:row + count] += 1
                                live[slot] = False
                            self.assertFalse(any(live))
                        np.testing.assert_array_equal(written, 1)
                        np.testing.assert_array_equal(out[-16:], 123)
                        compare(self, out[:-16].reshape(rows, d),
                                downloaded_golden(x, r, g, b))


if __name__ == "__main__":
    unittest.main()
