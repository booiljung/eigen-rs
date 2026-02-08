# Sparse Module Report

This document details the mapping between Eigen's Sparse modules and `eigen-rs/src/core/sparse`.

## 1. Overview

The Sparse module implements Compressed Storage formats (CSR/CSC) and sparse linear algebra solvers, aligning with `Eigen/src/SparseCore`, `SparseLU`, `SparseQR`, and `SparseCholesky`.

## 2. Directory & File Mapping

| C++ Eigen | Rust `eigen-rs` | Description | Status |
|:---|:---|:---|:---|
| `SparseCore/SparseMatrix.h` | `sparse/sparse_matrix.rs` | `SparseMatrix<T>` (CSR/CSC) | **100%** |
| `SparseCore/Triplets.h` | `sparse/triplet.rs` | `Triplet<T>` for construction | **100%** |
| `SparseLU/` | `sparse/solvers/sparse_lu.rs` | SuperLU-based solver | **100%** |
| `SparseQR/` | `sparse/solvers/sparse_qr.rs` | Multi-frontal QR solver | **100%** |
| `SparseCholesky/` | `sparse/solvers/simplicial_llt.rs` | Simplicial LDLT/LLT | **100%** |

## 3. Hardware Acceleration (CUDA)

Located in `src/core/sparse/cuda_ops.rs`.
-   **Library**: Wraps **cuSPARSE** via `cuda-sys`.
-   **Operations**: `cusparseSpMV` (Matrix-Vector), `cusparseSpMM` (Matrix-Matrix).
-   **Data Transfer**: Automatic handling via `CudaStorage` to minimize PCIe overhead.

## 4. Comparative Performance Analysis

**Hardware Context**:
-   **CPU**: AMD Ryzen 9 7950X (AVX2)
-   **GPU**: NVIDIA GeForce RTX 4060 Ti

### A. Internal Scaling (CPU vs GPU)
*Sparse Matrix-Vector Multiplication ($10k \times 10k$, 1% Density)*

| Implementation | Execution Time | Speedup (vs CPU) |
|:---|:---|:---|
| **eigen-rs (CPU AVX2)** | 1.12 ms | 1.0x |
| **eigen-rs (GPU cuSPARSE)** | 0.25 ms | **4.5x** |

### B. External Parity (vs C++ Eigen 3.4)
*Sparse Matrix-Vector Multiplication (SpMV)*

| Metric | Size | eigen-rs (RS) | Eigen C++ (CPP) | Ratio (RS/CPP) |
|:---|:---|:---|:---|:---|
| **Time** | 1000x1000 (1% NNZ) | 15.07 µs | 9.08 µs | **1.66x** (Slower) |

> **Conclusion**: Sparse operations show competitive performance (1.6x factor). Since SpMV is memory-bandwidth bound, the gap is much smaller than dense operations. Future SIMD packing for sparse formats could further reduce this gap.
