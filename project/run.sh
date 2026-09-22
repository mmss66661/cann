#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="add_rms_norm_bias_custom"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "ERROR: ASCEND_HOME_PATH is not set. Please run:"
    echo "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    echo "or set ASCEND_HOME_PATH to your CANN toolkit path."
    exit 1
fi

echo "=== [1/4] Set CANN env ==="
source "${ASCEND_HOME_PATH}/set_env.sh"

echo "=== [2/4] Build ==="
rm -rf build
mkdir -p build
cd build
cmake ..
make -j4
cd ..

echo "=== [3/4] Gen test data ==="
cd build
python3 ../scripts/gen_data.py

echo "=== [4/4] Run + Verify ==="
rm -f input/*.bin
cp input/case0/* input/ 2>/dev/null || true
cp output/golden_case0/* output/ 2>/dev/null || true
find output -name '*.bin' ! -name 'golden_*' -delete 2>/dev/null || true
if timeout 120 "./${OP_NAME}"; then
    if python3 ../scripts/verify_result.py 0; then
        echo "=== PASSED ==="
    else
        echo "=== FAILED ==="
        exit 1
    fi
else
    echo "=== FAILED (kernel exited non-zero or timed out) ==="
    exit 1
fi
