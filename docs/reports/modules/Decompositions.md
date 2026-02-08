# Decompositions Module Report

This document details the mapping of Dense Linear Algebra Decompositions from C++ Eigen to Rust.

## 1. Overview

Includes LU, Cholesky, QR, SVD, and Eigenvalues solvers. These rely heavily on the `Core` module's GEMM performance.

## 2. Directory & File Mapping

| C++ Eigen | Rust `eigen-rs` | Description | Status |
|:---|:---|:---|:---|
| `LU/FullPivLU.h` | `lu/full_piv_lu.rs` | Complete Pivoting LU | **100%** |
| `LU/PartialPivLU.h` | `lu/partial_piv_lu.rs` | Partial Pivoting LU | **100%** |
| `Cholesky/LLT.h` | `cholesky/llt.rs` | Standard Cholesky | **100%** |
| `QR/HouseholderQR.h` | `qr/householder_qr.rs` | Householder QR | **100%** |
| `SVD/JacobiSVD.h` | `svd/jacobi_svd.rs` | Two-sided Jacobi SVD | **100%** |
| `Eigenvalues/SelfAdjoint.h` | `eigenvalues/` | Tridiagonalization + QR | **100%** |

## 3. Comparative Performance Analysis

**Hardware Context**:
-   **CPU**: AMD Ryzen 9 7950X (AVX2)

### External Parity (vs C++ Eigen 3.4)
*Decomposition Performance ($256 \times 256$)*

| Operation | eigen-rs (RS) | Eigen C++ (CPP) | Ratio (RS/CPP) |
|:---|:---|:---|:---|
| **LLT (Cholesky)** | 3.49 ms | 0.33 ms | **10.5x** (Slower) |
| **LU (PartialPiv)** | N/A | 0.96 ms | - |
| **JacobiSVD** | 3.87 ms (64x64) | - | - |

> **Conclusion**: `eigen-rs` decompositions are currently pure Rust scalar implementations. The significant performance gap (10x) vs Eigen C++ is due to Eigen's use of **Blocked Algorithms** which effectively leverage Level 3 BLAS optimization, whereas `eigen-rs` currently relies on unblocked (Level 2) algorithms. This is a key area for future optimization.
