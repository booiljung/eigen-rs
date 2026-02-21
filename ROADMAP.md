# eigen-rs Roadmap

This document serves as the high-level index for the `eigen-rs` project roadmap. Details for each module are tracked in their respective documents.

## Module Roadmaps

| Module | Description | Status | Link |
|:---|:---|:---|:---|
| **Core** | Dense Matrix features, Evaluators, Storage | **100%** | [See Details](docs/roadmap/Core.md) |
| **Architecture** | SIMD (AVX/SSE/NEON), GPU (CUDA), Parallelism | **91%** | [See Details](docs/roadmap/Arch.md) |
| **Geometry** | Rotations, Transforms, Spatial primitives | **100%** | [See Details](docs/roadmap/Geometry.md) |
| **Decompositions** | LU, QR, Cholesky, SVD, Eigensolvers | **100%** | [See Details](docs/roadmap/Decompositions.md) |
| **Sparse** | Sparse Storage (CSR/CSC), Solvers | **100%** | [See Details](docs/roadmap/Sparse.md) |
| **Unsupported** | FFT, Splines, Polynomials, Matrix functions | **100%** | [See Details](docs/roadmap/Unsupported.md) |
| **Neural Networks** | NN Layers, Operations | **100%** | [See Details](docs/roadmap/NN.md) |
| **Tensors** | Multidimensional Arrays, Contraction | **100%** | [See Details](docs/roadmap/Tensor.md) |
| **Optimization** | Levenberg-Marquardt, Hybrid, AutoDiff | **100%** | [See Details](docs/roadmap/Optimization.md) |

## Strategic Philosophy: Pure Rust Core + Optional FFI

The project follows a **"Pure Rust First, FFI Optional"** hybrid approach:
1.  **Core Logic (Pure Rust)**:
    - Matrix arithmetic, decompositions (LU/QR/Cholesky), and geometry are implemented in 100% safe/unsafe Rust.
    - **Goal**: Generic extensibility (supporting `Complex`, `DualNumber`, etc.) and zero-dependency portability (WASM/Embedded).
2.  **Special Functions (Hybrid/FFI)**:
    - Complex functions (`erf`, `bessel`, `gamma`) leverage `libc` (system `libm`) for accuracy and performance.
    - Future goal: Gradual transition to `libm` crate for pure Rust portability.
3.  **High-Performance Backends (Opt-in FFI)**:
    - Users can enable features like `mkl`, `lapack`, `cuda` to offload heavy computations (GEMM, SVD) to optimized vendor libraries.
    - Default behavior remains pure Rust for maximum compatibility.

## Numerical Stability & Quality Assurance

- [x] **Fuzzy Comparison**: `isApprox`, `isMuchSmallerThan` with configurable precision.
- [x] **Condition Estimation**: `ConditionEstimator` for solver reliability.
- [x] **Scalar Traits**: `NumTraits` for floating point, integer, and complex types.
- [x] **Validation**: Differential testing framework against C++ Eigen binaries.

## Recent Optimizations (2026 Campaign)

| Phase | Component | Speedup (vs C++ Eigen) | Status |
|:---|:---|:---|:---|
| **Phase 10** | `VecDot` / `VecNorm` | **0.76x / 0.42x** (Faster) | ✅ Complete |
| **Phase 12** | `GeneralizedEigen` | **1.27x - 0.79x** (Faster) | ✅ Complete |
| **Phase 13** | `RealSchur` (Speed) | **4x Speedup** (Execution) | ✅ Complete |
| **Phase 14** | `LDLT` / `Tridiagonal` | **1.44x / 1.24x** | ✅ Complete |
| **Phase 15** | `Hessenberg` Decomposition | **Optimal** (No allocs) | ✅ Complete |
| **Phase 17** | `RealSchur` (Convergence) | **0.24x** (Faster) | ✅ Complete |
