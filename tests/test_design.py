"""CPU checks for the reference arithmetic and the host row partition.

These tests do not execute Ascend C. Run the downloaded judge package on a
910B to validate compilation, device precision, and speed.
"""

import math
import sys
import unittest
from pathlib import Path

import ml_dtypes
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project" / "scripts"))
from AddRmsNormBias import impl as downloaded_golden


def partition(rows: int, hidden: int, itemsize: int, available_cores: int):
    elems_per_block = 32 // itemsize
    group = elems_per_block // math.gcd(hidden, elems_per_block)
    groups = (rows + group - 1) // group
    cores = min(max(available_cores, 1), groups)
    rows_per_core = ((groups + cores - 1) // cores) * group
    intervals = [
        (i * rows_per_core, min((i + 1) * rows_per_core, rows))
        for i in range(cores)
        if i * rows_per_core < rows
    ]
    return intervals


def numpy_input(tensor):
    if tensor.dtype == torch.bfloat16:
        return tensor.float().numpy().astype(ml_dtypes.bfloat16)
    return tensor.numpy()


def chunked_arithmetic(x, residual, gamma, bias, epsilon, chunk=4096):
    hidden = x.shape[-1]
    sum_of_squares = torch.zeros((*x.shape[:-1], 1), dtype=torch.float32)
    for col in range(0, hidden, chunk):
        y = x[..., col : col + chunk].float() + residual[..., col : col + chunk].float()
        sum_of_squares += y.square().sum(dim=-1, keepdim=True)
    inv = 1.0 / torch.sqrt(sum_of_squares / hidden + epsilon)
    pieces = []
    for col in range(0, hidden, chunk):
        y = x[..., col : col + chunk].float() + residual[..., col : col + chunk].float()
        pieces.append(
            (y * inv * gamma[col : col + chunk].float()
             + bias[col : col + chunk].float()).to(x.dtype)
        )
    return torch.cat(pieces, dim=-1)


class TestDesign(unittest.TestCase):
    def test_partition_covers_rows_without_cross_core_block_sharing(self):
        for hidden in (64, 65, 192, 255, 576, 4096, 4097, 32768):
            for itemsize in (2, 4):
                for rows in (1, 2, 3, 31, 64, 65, 1001):
                    spans = partition(rows, hidden, itemsize, 48)
                    self.assertEqual(spans[0][0], 0)
                    self.assertEqual(spans[-1][1], rows)
                    for left, right in zip(spans, spans[1:]):
                        self.assertEqual(left[1], right[0])
                    for left, _ in spans[1:]:
                        self.assertEqual(left * hidden * itemsize % 32, 0)

    def test_chunked_arithmetic_matches_downloaded_package_golden(self):
        torch.manual_seed(2026)
        shapes = ((2, 64), (2, 3, 192), (1, 2, 3, 576),
                  (2, 4096), (2, 4097), (1, 32768))
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for shape in shapes:
                hidden = shape[-1]
                x = torch.randn(shape, dtype=dtype)
                residual = torch.randn_like(x)
                gamma = torch.randn(hidden, dtype=dtype)
                bias = torch.randn(hidden, dtype=dtype)
                got = chunked_arithmetic(x, residual, gamma, bias, 1e-5)
                want_np = downloaded_golden(
                    numpy_input(x), numpy_input(residual),
                    numpy_input(gamma), numpy_input(bias), 1e-5
                )
                want = torch.from_numpy(want_np.astype(np.float32))
                tol = 1e-4 if dtype == torch.float32 else 1e-3
                torch.testing.assert_close(got.float(), want, rtol=tol, atol=tol)


if __name__ == "__main__":
    unittest.main()
