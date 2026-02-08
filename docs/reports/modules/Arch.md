# Architecture & Hardware Acceleration Report

This document audits the low-level hardward acceleration layers that power `eigen-rs`.

## 1. SIMD (PacketMath)

Aligned with `Eigen/src/Core/arch`.

| Feature | C++ Eigen | eigen-rs | Status |
|:---|:---|:---|:---|
| **SSE** | `Packet4f` | `SsePacketF32` | **100%** |
| **AVX** | `Packet8f` | `AvxPacketF32` | **100%** |
| **NEON** | `Packet4f` | (Planned) | 0% |

### Verification
-   **Unit Tests**: `tests/simd_test.rs`
-   **Cross-Check**: `tests/simd_verify_test.rs` compares SIMD packet logic against Scalar ground truth.

## 2. GPU (CUDA)

Aligned with `Eigen/src/Core/arch/CUDA` but significantly modernized.

| Feature | Implementation | Notes |
|:---|:---|:---|
| **Memory** | `CudaStorage<T>` | Unified Memory RAII wrapper |
| **BLAS** | `core/storage/cublas.rs` | `cublasSgemm` / `cublasDgemm` |
| **Sparse** | `core/sparse/cuda_ops.rs` | `cusparseSpMV` |

### Verification
-   **Test**: `tests/cuda_compute_test.rs`
-   **Ops Verified**: Host-to-Device Copy, GEMM, SpMV.

## 3. Performance Summary

See `Core.md` and `Sparse.md` for specific speedup numbers. The architecture layer is responsible for the **8.1x** (AVX) and **26.6x** (CUDA) speedups observed in dense matrix operations.
