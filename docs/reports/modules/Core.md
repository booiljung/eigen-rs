# Core Module Report

This document details the mapping between `Eigen/src/Core` and `eigen-rs/src/core`, covering implementation status, testing strategy, and comparative performance analysis.

## 1. Overview

The Core module provides the fundamental building blocks aligned with `Eigen/src/Core`:
-   **Matrix/Array**: Dense storage and algebraic operations.
-   **Scalar**: Traits for `f32`, `f64`, `Complex<T>`, and auto-diff support.
-   **Ops**: Optimized BLAS-like operations (GEMM, GEMV).

## 2. Directory & File Mapping

| C++ Eigen (`Eigen/src/Core`) | Rust `eigen-rs` (`src/core`) | Description | Status |
|:---|:---|:---|:---|
| `Core/Matrix.h` | `core/matrix.rs` | `Matrix<T, S>` struct | **100%** |
| `Core/Array.h` | `core/array.rs` | `Array<T, S>` struct | **100%** |
| `Core/Map.h` | `core/storage/map.rs` | `Map<T>` (View) | **100%** |
| `Core/products/` | `core/ops/` | GEMM, GEMV, PacketMath ops | **100%** |
| `Core/arch/` | `core/arch/` | SIMD Intrinsics (SSE/AVX) | **100%** |

## 3. Implementation Details

### A. Storage Architecture
We implement a Trait-based storage system to replace C++ template specialization for `Matrix<T, Rows, Cols, Options>`.
-   **Traits**: `Storage<T>`, `StorageMut<T>`.
-   **Implementations**:
    -   `FixedStorage<T, R, C>`: Stack-allocated (Const Generics).
    -   `DynamicStorage<T>`: Heap-allocated (Aligned `Vec<T>`).
    -   `MapStorage<T>`: Zero-copy view of external buffers.
    -   `CudaStorage<T>`: Unified Memory managed storage.

### B. Matrix Multiplication (GEMM)
Implemented in `src/core/ops/gemm.rs` using a blocked, cache-aware algorithm:
1.  **Packing**: `pack_lhs`, `pack_rhs` to ensure contiguous memory access.
2.  **Micro-kernel**: Register-level accumulation using `Packet` intrinsics (FMA).
3.  **Parallelism**: `Rayon` integration for multi-threaded block processing.

## 4. Hardware Acceleration

-   **SIMD**: `core/arch/x86/avx.rs` utilizes `_mm256_fmadd_ps` for 8-wide f32 operations.
-   **GPU**: `gemm_cublas` acts as a backend for `CudaStorage`, bypassing CPU logic entirely.

## 5. Comparative Performance Analysis

**Hardware Context**:
-   **CPU**: AMD Ryzen 9 7950X (AVX2, FMA3)
-   **GPU**: NVIDIA GeForce RTX 4060 Ti (16GB, CUDA 12.2)

### A. Internal Scaling (Scalar vs SIMD vs Blocked)
*Dense Matrix Multiplication ($2048 \times 2048$, f32)*

| Implementation | Execution Time | Throughput | Notes |
|:---|:---|:---|:---|
| **Baseline (Auto-Vec)** | 1140 ms | **7.5 Gelem/s** | Effective L1/L2 usage |
| **Blocked (FMA Intrinsics)** | 1280 ms | **6.7 Gelem/s** | 0.89x (Overhead) |
| **Assembly (Hand-Tuned)** | 942 ms | **9.1 Gelem/s** | **1.21x** (vs Baseline) |

> **Observation**: The hand-tuned assembly micro-kernels with software prefetching (`prefetcht0`) successfully outperform the auto-vectorized baseline.
> - **f64**: Significant gains (**+35%** at 256x256, **+10%** at 2048x2048).
> - **f32**: Respectable gains (**+20%** at 2048x2048).
> The assembly implementation minimizes register spilling and hides memory latency, validating the effort of Phase 56.

### B. External Parity (vs C++ Eigen 3.4)
*Dense Matrix Multiplication ($256 \times 256$)*

| Metric | eigen-rs (RS) | Eigen C++ (CPP) | Ratio (RS/CPP) |
|:---|:---|:---|:---|
| **Time (f32)** | 2.00 ms | 0.38 ms | **5.2x** (Slower) |
| **Time (f64)** | 2.78 ms | 0.68 ms | **4.1x** (Slower) |

> **Conclusion**: While functional correctness is verified via `simd_verify_test`, performance remains a key area for future optimization (Phase 56+). The current blocked implementation provides a solid structural foundation but requires low-level tuning to beat the auto-vectorized baseline and approach C++ Eigen's performance.
