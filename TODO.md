# Optimization Roadmap (Phase 3)

This document outlines the plan to bridge the performance gap between `eigen-rs` and C++ Eigen 3.4, based on the findings from `verify_perf.py`.

## 🎯 Goal
Achieve performance parity (Ratio < 1.0~1.5x) for core operations.

## 1. Algorithmic Optimizations (High Impact)
Currently, decompositions like LLT use naive $O(N^3)$ loops.
- [x] **Blocked LLT Decomposition**: Implemented Right-Looking Cholesky (Block Size 32). Performance improved 4x.
    -   **Next Bottleneck**: Improve `MatMul` baseline (< 2.0x parity required) and optimize `Trsm` ($O(N^3)$ naive loop).
- [ ] **Blocked LDLT**: Extend blocking variable algorithm to LDLT with pivoting.
- [ ] **Blocked SVD**: Replace Jacobi SVD with Divide & Conquer or bidiagonalization-based SVD for large matrices.

## 2. GEMM & Memory Optimizations (Medium Impact)
`verify_perf.py` revealed significant allocation overhead in `gemm_blocked`.
- [ ] **Workspace Reuse**: Eliminate `vec![...]` allocation in `gemm_blocked` by using a thread-local workspace or stack allocation for small matrices.
- [ ] **Small Matrix Specialization**: For $N < 32$, skip blocking and use unrolled register-based kernels directly.
- [ ] **Pointer Aliasing**: Verify `restrict` usage (or Rust equivalent check) to ensure autovectorization doesn't fail due to aliasing fears.

## 3. SIMD Integration (Low Level)
- [x] **Micro-kernel Tuning**: Logic moved to `asm_kernel.rs` with Fused Loop and 4x Unrolling.
    -   **Result**: MatMul 6x slower (was >20x).
- [ ] **SIMD Packing**: Implement AVX-based packing (`pack_lhs`, `pack_rhs`) to remove scalar copy overhead.
- [ ] **Target Specific Dispatch**: Ensure `target_feature` detection is zero-overhead at runtime (use `#[cfg]` where possible).
- [ ] **Unsafe unchecked_get**: Replace `get().unwrap()` with `unsafe { get_unchecked() }` in hot inner loops of decompositions.

## 4. Infrastructure
- [ ] **Autotuning Script**: Create a script to find optimal `MC`, `KC`, `NC` blocking parameters for the target machine.
