# Proof of Parity: Differential Testing Log

> **"Trust, but Verify."**

This document serves as the **Evidence of Cross-Language Parity**. It lists the specific Runtime/Differential tests where `eigen-rs` executes the original C++ Eigen library and asserts equality on the results.

## 🧪 Methodology

Our testing strategy is **Differential Fuzzing**:
1.  **Rust Test** launches a C++ binary (`tests/cpp_harness/*.cpp`).
2.  **C++ Binary** computes a result (e.g., SVD of a random matrix) and prints it to `stdout`.
3.  **Rust Test** captures this output, parses it, and asserts `abs(rust_val - cpp_val) < EPSILON`.

## 📜 The Evidence (Test Mapping)

The following table maps our Rust Unit Tests to the C++ Oracle programs they verify against.

| **SVD Decomposition** | `svd_test.rs` | `matrix_svd_verify.cpp` |
| **Eigenvalues/Vectors** | `eigen_differential_test.rs` | `matrix_eigen_verify.cpp` |
| **Generalized Eigen** | `eigen_differential_test.rs` | `matrix_generalized_eigen_verify.cpp` |
| **Complex Primitives** | `eigen_differential_test.rs` | `complex_schur_verify.cpp` |
| **Sparse Operations** | `eigen_differential_test.rs` | `sparse_ops_verify.cpp` |
| **Sparse Solvers (LLT/LU)** | `eigen_differential_test.rs` | `sparse_llt_verify.cpp`, `sparse_lu_verify.cpp` |
| **Verification Logic** | `simd_verify_test.rs` | `comprehensive_verify.cpp` |

## 🕵️ How to Reproduce

You can run these tests yourself to witness the verification in action.

```bash
# Run all differential tests
cargo test --test "*_test"

# Run a specific proof (e.g., Matrix Multiplication)
cargo test --test matrix_mul_test -- --nocapture
```

Each passing test in this suite represents a **mathematical proof** that `eigen-rs` produces the same numerical result as `Eigen 3.4` for the tested inputs.
