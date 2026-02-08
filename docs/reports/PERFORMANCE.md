# Performance Evaluation Report

**Run Date**: 2026-02-08
**Hardware**: AMD EPYC 7B13 (AVX2/FMA)

This report details the performance comparison between `eigen-rs` (Rust) and **Eigen 3.4.0** (C++).

## 1. Executive Summary

| Operation | Metric | eigen-rs (Native) | Eigen C++ (Native) | Ratio | Status |
|:---|:---|:---|:---|:---|:---|
| **GEMM (f32)** | 256x256 Time | **2.49 ms** | **0.81 ms** | **3.07x** | ⚠️ Gap (>1.5x) |
| **GEMM (f64)** | 256x256 Time | **3.04 ms** | **0.67 ms** | **4.53x** | ⚠️ Gap (>1.5x) |
| **SpMV (f64)** | 1k, 1% nnz | **16.8 µs** | **6.5 µs** | **2.58x** | ⚠️ Gap (>1.5x) |

> **Note**: Current CPU performance shows a 3x-4x gap compared to Eigen C++. This indicates that while the architecture is sound, the low-level micro-kernels (SIMD/Assembly) and blocking parameters need significant tuning or assembly implementation (Phase 56) to match Eigen's hand-optimized efficiency.

## 2. Methodology

-   **Rust**: `RUSTFLAGS="-C target-cpu=native" cargo bench` (Criterion).
-   **C++**: `g++ -O3 -march=native -DNDEBUG` (Eigen 3.4.0 headers).
-   **Environment**: Linux / AMD EPYC 7B13.

## 3. Detailed Results

### 3.1 Dense Algebra (GEMM)
*Matrix Multiplication: $C = A \times B$*

#### Single Precision (f32)
| Size | eigen-rs (Time) | Eigen C++ (Time) | Ratio (eigen-rs / C++) |
|:---|:---|:---|:---|
| 64x64 | 137 µs | 17 µs | **8.0x** |
| 128x128 | 516 µs | - | - |
| 256x256 | 2.49 ms | 0.81 ms | **3.07x** |
| 512x512 | - | 3.98 ms | - |

#### Double Precision (f64)
| Size | eigen-rs (Time) | Eigen C++ (Time) | Ratio (eigen-rs / C++) |
|:---|:---|:---|:---|
| 64x64 | 111 µs | 12 µs | **9.2x** |
| 128x128 | - | - | - |
| 256x256 | 3.04 ms | 0.67 ms | **4.53x** |
| 512x512 | - | 5.23 ms | - |

### 3.2 Sparse Algebra (SpMV)
*Sparse Matrix-Vector Multiplication: $y = A x$ (CSR format)*

| Matrix Size | Non-Zeros | eigen-rs (f64) | Eigen C++ (f64) | Ratio |
|:---|:---|:---|:---|:---|
| 1000 x 1000 | 10 per row (1%) | **16.8 µs** | **6.5 µs** | **2.58x** |

## 4. Analysis & Next Steps

### Analysis
-   **Performance Gap**: The 3x-4x gap suggests that `eigen-rs` is not yet effectively utilizing the full AVX2 pipeline or memory hierarchy compared to Eigen C++.
-   **Small Matrix overhead**: The 8x-9x gap for 64x64 matrices indicates significant overhead in the `gemm_blocked` logic (packing, kernel dispatch) which dominates small computations.
-   **SpMV**: The 2.5x gap in SpMV points to potential missed optimizations in memory access patterns or loop unrolling, as this operation should be memory-bound.

### Recommended Actions (Phase 56)
1.  **Assembly Micro-kernels**: Move from Rust intrinsics to inline assembly (`asm!`) for f32/f64 kernels to ensure optimal register allocation and pipeline scheduling.
2.  **Prefetching**: Implement software prefetching in the macro-kernel loops to hide memory latency.
3.  **Small Matrix Optimization**: Implement a dedicated "small matrix" path that skips packing and processes directly (or use a different blocking strategy).
4.  **Tuning**: Re-evaluate `MC`, `KC`, `NC` blocking parameters for the target architecture.
