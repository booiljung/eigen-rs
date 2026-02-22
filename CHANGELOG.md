# Changelog

All notable changes to `eigen-rs` will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-22
### Added
- **Total Parity with Eigen 3.4**: Complete C++ to Rust translation across Core, Geometry, Splines, Fft, and Polynomials.
- **Advanced Decompositions**: Parity for LU, QR, SVD, Cholesky (LLT/LDLT), EigenSolver, and Schur forms.
- **Sparse Module**: Functional parity for Sparse matrices (CSR/CSC formats), iterators, and sparse direct solvers (`SimplicialLLT`, `SparseLU`).
- **Tensor Module**: Multi-dimensional N-way lazy evaluation and GEMM contractions for Neural Networks.
- **Hardware Acceleration (SIMD)**: Zero-cost abstraction AVX/FMA hand-optimized micro-kernels targeting x86_64 and ARM64 via `target-cpu=native`.
- **GPU Acceleration (CUDA)**: Dynamic FFI bridging via `libloading` for `cuBLAS` (dense GEMM), `cuSOLVER` (factorizations), and `cuSPARSE` (MVM and iterators).
- **Multithreading**: Rayon-based parallel evaluator applied natively to `Cwise` loops and partitioned Matrix multiplications.
- **Differential Testing**: A robust C++ test harness proving numerical honesty to the final IEEE-754 epsilon.
