import numpy as np
import os
import sys

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None

sys.path.insert(0, os.path.dirname(__file__))
from AddRmsNormBias import impl

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)


# --- Case 0 ---
np.random.seed(42 + 0)
os.makedirs("input/case0", exist_ok=True)
x = np.random.uniform(low=-2, high=2, size=(1, 64)).astype(np.float16)
x.tofile("input/case0/x.bin")
residual = np.random.uniform(low=-2, high=2, size=(1, 64)).astype(np.float16)
residual.tofile("input/case0/residual.bin")
gamma = np.random.uniform(low=0.8, high=1.2, size=(64)).astype(np.float16)
gamma.tofile("input/case0/gamma.bin")
bias = np.random.uniform(low=-0.3, high=0.3, size=(64)).astype(np.float16)
bias.tofile("input/case0/bias.bin")

epsilon = 1e-05

os.makedirs("output/golden_case0", exist_ok=True)
golden = impl(x, residual, gamma, bias, epsilon=epsilon)
if golden is not None:
    golden.tofile("output/golden_case0/golden_output.bin")

print(f"Generated test data and golden output for 1 cases.")
