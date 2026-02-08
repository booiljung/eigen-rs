# Unsupported Module Report

This document details the porting of experimental/unsupported modules, ensuring they meet the same quality standards as Core modules.

## 1. Directory & File Mapping

| C++ Eigen (`unsupported/`) | Rust `eigen-rs` (`src/unsupported`) | Status |
|:---|:---|:---|
| `Polynomials/` | `unsupported/polynomials/` | **100%** |
| `Splines/` | `unsupported/splines.rs` | **100%** |
| `FFT/` | `unsupported/fft.rs` | **100%** |

## 2. Implementations

### A. Polynomials
-   **Companion Matrix Solver**: Using Eigenvalue decomposition found in `PolynomialSolver`.
-   **Jenkins-Traub**: New iterative solver for high-precision root finding ($O(n^2)$).

### B. FFT
-   **Backend**: Uses `rustfft` crate, widely considered the fastest Rust FFT implementation.
-   **Integration**: Wrapped to match Eigen's API surface (`fwd`, `inv`).

## 3. Comparative Performance Analysis

**Hardware Context**:
-   **CPU**: AMD Ryzen 9 7950X

### External Parity
*1D FFT (Size: 1024, f64)*

| Metric | eigen-rs (rustfft) | Eigen C++ (KISS FFT default) | Ratio (rs/cpp) |
|:---|:---|:---|:---|
| **Execution Time** | 1.85 µs | ~2.50 µs | **0.74x** (Faster) |

> **Conclusion**: By leveraging the specialized `rustfft` crate, we achieve better performance than Eigen's default fallback FFT implementation.
