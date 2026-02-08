# Architecture & Hardware Roadmap

**Status**: 91% Complete

## Hardware Acceleration & Parallelism
- **Parallelism (OS-Dependent)**
    - [x] **Work-Stealing**: `Rayon` integration for multi-threaded execution.
    - [x] **Parallel Evaluators**: Parallel `assign` and expression evaluation.
    - [x] **Parallel Reductions**: `sum`, `min`, `max` optimized for large matrices.
    - [x] **Parallel Sparse**: Parallel SpMV (CSR/CSC) for high-performance sparse algebra.
- **SIMD (PacketMath)**
    - [x] **Foundations**: Architecture-agnostic `Packet` layer.
    - [x] **ISA Support**: SSE2-4.2, AVX/AVX2, FMA, NEON specializations.
    - [x] **Special Functions**: Packet-level `sqrt`, `rsqrt`, `abs`, `exp`, `log`.
- **Level 2/3 BLAS Optimization**
    - [x] **GEMM**: L1/L2/L3 cache-aware blocking and tiling.
    - [x] **GEMV**: Highly optimized matrix-vector products.
    - [x] **Packing**: Internal data reordering for contiguous SIMD access.
- **GPU Acceleration (CUDA/ROCm)**
    - [x] **Unified Memory**: `CudaStorage` for host-device synchronization.
    - [x] **Element-wise JIT**: PTX kernel generation for arbitrary expressions.
    - [x] **External Bridge**: cuBLAS integration (`src/core/storage/cublas.rs`).
