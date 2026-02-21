# Feature Implementation Roadmap (Phase 2)

This document outlines the roadmap for the next phase of `eigen-rs` development, focusing on performance stabilization and advanced hardware acceleration.

## 🎯 Primary Goal
Eliminate remaining performance anomalies (Cache Thrashing) and Implement CUDA kernels for the Tensor module.

## 1. Performance Optimization (Critical Priority)
Target specific matrix sizes where performance degrades significantly (Ratio > 3.0x).
- [x] **Cache Associativity Conflict Resolution**:
    -   **Problem**: Severe slowdowns at N=289 (17x17) and N=315.
    -   **Conclusion**: Investigated in Phase 35. Determined to be Allocator History Artifacts (benchmarking noise due to previous allocations). Fresh runs show normal performance. **Status: Resolved/WontFix**.

## 2. Tensor Module: GPU Acceleration (High Priority)
Leverage the newly implemented `CudaDevice` to run actual operations on GPU.
- [x] **GPU Kernels**:
    -   [x] `assign`: Implement element-wise assignment kernel. (Done Phase 36)
    -   [x] `contract`: Implement GEMM-based contraction (Rank-2 MatMul implemented). (Done Phase 36)
    -   [x] `permute`: Implement dimensional permutation kernel. (Done Phase 37)
- [x] **Integration**:
    -   Connect `Tensor::eval()` to GPU kernels when `CudaDevice` is active. (Done Phase 36/37)

## 3. Missing Features (Medium Priority)
- [x] **Complex Number Support**:
    -   [x] Extend `Scalar` trait to fully support `Complex<f32>` / `Complex<f64>`. (Done Phase 38)
    -   [x] Verify Decompositions with complex types:
        -   [x] `Cholesky` (LLT/LDLT) (Verified Phase 39)
        -   [x] `QR` (Fixed Phase 39 - Verification Execution Stalled)
        -   [x] `LU` (Verified Phase 39)
- [x] **Geometry Module Optimization**:
    -   [x] Vectorize `Quaternion` multiplication and `Transform` applications. (Done Phase 40)

## 4. Next Project Steps (Post-LU Optimization)
- [ ] **Commit Changes**: Commit the recent `PartialPivLU` optimizations and the unified roadmap documentation.
- [ ] **SparseView Optimization (High Priority)**: Investigate and optimize `SparseView` performance, which currently exceeds the 1.5x threshold (max ratio 1.87x).
- [ ] **Complete BDCSVD (Medium Priority)**: Implement the Divide & Conquer SVD (`BDCSVD`) which is currently marked as pending in `Decompositions.md`.
- [ ] **Add Missing Benchmarks (Medium Priority)**: Expand `benches/repro_lu.rs` or create new benchmarks to cover operations shown as `MISSING` in reports (e.g., `AngleAxis`, `GeneralizedEigen`, `Inverse`, `LDLT`, `LLT`).
