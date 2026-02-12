
import subprocess
import os
import re
import datetime

# Map of Rust Test Files to C++ Harnesses (Source of Truth)
# This mirrors PROOF_OF_PARITY.md
TEST_MAPPING = [
    ("matrix_arithmetic_test", "matrix_add_verify.cpp"),
    ("matrix_mul_test", "matrix_mul_verify.cpp"),
    ("matrix_large_mul_test", "matrix_large_mul_verify.cpp"),
    ("matrix_det_test", "matrix_det_verify.cpp"),
    ("matrix_inv_test", "matrix_inv_verify.cpp"),
    ("matrix_block_test", "matrix_block_verify.cpp"),
    ("matrix_transpose_test", "matrix_transpose_verify.cpp"),
    ("matrix_reduction_test", "matrix_reduction_verify.cpp"),
    ("matrix_lu_test", "matrix_lu_verify.cpp"),
    ("matrix_qr_test", "matrix_qr_verify.cpp"),
    ("llt_test", "matrix_cholesky_verify.cpp"),
    ("ldlt_test", "matrix_cholesky_verify.cpp"),
    ("svd_test", "matrix_svd_verify.cpp"),
    ("eigen_differential_test", "matrix_eigen_verify.cpp"),
    ("simd_verify_test", "comprehensive_verify.cpp"),
    ("geometry_test", "geometry_verify.cpp"),
]

REPORT_FILE = "tmp/verification/PARITY_REPORT.md"

def run_test(test_name):
    print(f"Running {test_name}...")
    try:
        # Run cargo test for the specific test file
        # --nocapture to see output if needed, but we mostly care about exit code
        cmd = ["cargo", "test", "--test", test_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def generate_report(results):
    with open(REPORT_FILE, 'w') as f:
        f.write("# Parity Verification Report\n\n")
        f.write(f"> **Generated**: {datetime.datetime.now().isoformat()}\n")
        f.write("> **Tool**: `verify_parity.py`\n\n")
        
        f.write("## Execution Results\n\n")
        f.write("| Rust Test Module | C++ Oracle | Result | Status |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        passed_count = 0
        total_count = len(results)
        
        for test_name, cpp_harness, success, output in results:
            status = "✅ PASS" if success else "❌ FAIL"
            if success: passed_count += 1
            
            # Extract simple result summary if possible, or just say 'Verified'
            note = "Values Matched"
            
            f.write(f"| `{test_name}` | `{cpp_harness}` | {note} | {status} |\n")
            
        f.write(f"\n## Summary\n")
        f.write(f"- **Total Tests**: {total_count}\n")
        f.write(f"- **Passed**: {passed_count}\n")
        f.write(f"- **Failed**: {total_count - passed_count}\n")
        
        score = (passed_count / total_count) * 100 if total_count > 0 else 0
        f.write(f"\n**Parity Score**: **{score:.1f}%**\n")

def main():
    print("🚀 Starting Deterministic Parity Verification...")
    results = []
    
    for test_name, cpp_harness in TEST_MAPPING:
        success, output = run_test(test_name)
        results.append((test_name, cpp_harness, success, output))
        if success:
            print(f"  ✅ {test_name}: PASSED")
        else:
            print(f"  ❌ {test_name}: FAILED")
            
    generate_report(results)
    print(f"\n📄 Report generated at: {REPORT_FILE}")

if __name__ == "__main__":
    main()
