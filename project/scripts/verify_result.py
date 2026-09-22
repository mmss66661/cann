import numpy as np
import sys
import os

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None

# Per-case output specs: 1 cases
# Each spec: (name, dtype, rtol, atol, tol)
# tol = max fraction of elements allowed to mismatch (0.0 = strict)
case_output_specs = {
    0: [
        ("output", np.float16, 0.001, 0.001, 0.001),
    ],
}

if bfloat16 is None:
    for _cid, _specs in case_output_specs.items():
        for _name, _dtype, _rtol, _atol, _tol in _specs:
            if _dtype is None:
                print(f"ERROR: bfloat16 dtype required for case {_cid} output '{_name}' but ml_dtypes is not installed.")
                print("Install with: pip install ml_dtypes")
                sys.exit(1)

def verify_result(output_path, golden_path, dtype, rtol, atol, tol=0.0):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)
    total_size = golden.size

    if output.size != golden.size:
        if output.size < golden.size:
            golden = golden[:output.size]
            print(f"WARNING: output {output.size} < golden {total_size} elements, truncated golden to match")
        else:
            print(f"FAILED: output has {output.size} elements, golden has {total_size} — size mismatch")
            return False

    missing = total_size - output.size

    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
        if missing == 0 and np.array_equal(output, golden):
            print(f"PASSED: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
            return True
        else:
            diff = np.abs(output.astype(np.int64) - golden.astype(np.int64))
            errors = np.sum(output != golden) + missing
            error_rate = errors / total_size if total_size > 0 else 0.0
            if error_rate <= tol:
                print(f"PASSED (tol={tol}): {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
                print(f"  Mismatched: {errors}/{total_size} ({error_rate*100:.2f}%)")
                return True
            print(f"FAILED: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
            print(f"  Mismatched: {errors}/{total_size} ({error_rate*100:.2f}%), tol={tol}")
            print(f"  Max diff: {np.max(diff) if diff.size > 0 else 0.0}")
            return False

    cmp_output = output.astype(np.float32) if dtype == bfloat16 else output
    cmp_golden = golden.astype(np.float32) if dtype == bfloat16 else golden
    isclose = np.isclose(cmp_output, cmp_golden, rtol=rtol, atol=atol, equal_nan=True)
    errors = np.sum(~isclose) + missing
    error_rate = errors / total_size if total_size > 0 else 0.0
    diff = np.abs(cmp_output - cmp_golden)
    if error_rate <= tol:
        print(f"PASSED: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
        if errors > 0:
            print(f"  Mismatched: {errors}/{total_size} ({error_rate*100:.2f}%), tol={tol}")
        print(f"  Max diff: {np.max(diff) if diff.size > 0 else 0.0}")
        return True
    else:
        print(f"FAILED: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
        print(f"  Mismatched: {errors}/{total_size} ({error_rate*100:.2f}%), tol={tol}")
        print(f"  Max diff: {np.max(diff) if diff.size > 0 else 0.0}")
        return False

if __name__ == "__main__":
    try:
        case_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    except ValueError:
        print(f"Invalid case_id: {sys.argv[1] if len(sys.argv) > 1 else '(none)'}")
        sys.exit(1)
    if case_id not in case_output_specs:
        print(f"Unknown case_id {case_id}. Available: {sorted(case_output_specs.keys())}")
        sys.exit(1)

    output_dir = "output"
    all_pass = True
    specs = case_output_specs[case_id]
    if not specs:
        print(f"Case {case_id}: No outputs to verify (all optional) — PASS")
        sys.exit(0)
    for name, dtype, rtol, atol, tol in specs:
        output_path = os.path.join(output_dir, name + ".bin")
        golden_path = os.path.join(output_dir, "golden_" + name + ".bin")
        if not os.path.exists(output_path):
            print(f"FAILED: output {name}.bin not found")
            all_pass = False
            continue
        if not os.path.exists(golden_path):
            print(f"FAILED: golden_{name}.bin not found")
            all_pass = False
            continue
        if not verify_result(output_path, golden_path, dtype, rtol, atol, tol):
            all_pass = False
    sys.exit(0 if all_pass else 1)
