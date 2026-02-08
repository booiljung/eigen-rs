# Benchmark Methodology

This document defines the standard procedures for benchmarking `eigen-rs` against the original C++ Eigen 3.4 library.

## 1. Hardware Context (Mandatory)

Every benchmark run **MUST** begin by specifying the exact hardware environment to ensure reproducibility.

**Required Fields:**
-   **CPU**: Model Name, Base Clock, Cores/Threads (e.g., `AMD Ryzen 9 7950X, 4.5GHz, 16C/32T`).
-   **SIMD Support**: List available extensions (e.g., `AVX2`, `FMA`, `AVX-512`).
-   **GPU**: Model Name, VRAM, CUDA Version (e.g., `NVIDIA RTX 4090, 24GB, CUDA 12.2`).
-   **Memory**: Capacity & Type (e.g., `64GB DDR5-6000`).
-   **OS**: Kernel version (e.g., `Linux 6.5 generic`).

## 2. Comparison Strategy

We require a **Two-Dimensional Analysis** for every performance metric.

### A. Internal Scaling (Architecture Efficiency)
Compare `eigen-rs` implementations against each other to verify hardware acceleration gains.
*   **Scalar vs SIMD**: Verify vectorization speedup (Expected: ~4x for f32 on SSE, ~8x on AVX).
*   **CPU vs GPU**: Verify offloading efficiency. Define the **crossover point** where GPU beats CPU (e.g., N > 1024).

### B. External Parity (Project Goal)
Compare `eigen-rs` (Best optimized version) against C++ Eigen 3.4 (Best optimized version).
*   **Goal**: Ratio $\approx 1.0$.
*   **Method**: `eigen-rs` (SIMD/GPU) vs `Eigen C++` (SIMD/GPU).

## 3. Running Benchmarks

### A. Rust Benchmarks (eigen-rs)

```bash
# 1. Scalar Baseline (No SIMD)
RUSTFLAGS="-C target-cpu=generic" cargo bench --bench matrix_bench

# 2. SIMD Optimized (Native)
RUSTFLAGS="-C target-cpu=native" cargo bench --bench matrix_bench
```

### B. C++ Benchmarks (Eigen 3.4)

```bash
# Compile with equivalent flags
g++ -O3 -march=native -DNDEBUG ...
```

## 4. Reporting Template

**Hardware**: [Insert CPU/GPU Specs]

| Operation | Size | Implementation | Time (ms) | Speedup (vs Scalar) | vs Eigen C++ |
|:---|:---|:---|:---|:---|:---|
| **GEMM (f32)** | 1024x1024 | **eigen-rs (Scalar)** | 1200.0 | 1.0x | - |
| | | **eigen-rs (AVX2)** | 150.0 | **8.0x** | 0.98x |
| | | **eigen-rs (CUDA)** | 45.0 | **26.6x** | 1.05x |
| | | **Eigen C++ (AVX2)** | 148.0 | - | - |

### Dense Algebra (GEMM)
-   **Metric**: Execution time (ms) for $C = A \times B$.
-   **Sizes**: 64x64, 256x256, 1024x1024.
-   **Data Types**: `f32`, `f64`.

### Sparse Algebra (SpMV)
-   **Metric**: Execution time (ms) for $y = \alpha A x + \beta y$.
-   **Sparsity**: ~1-5% fill rate (e.g., 10 non-zeros per row).
-   **Storage**: CSR (Compressed Sparse Row).

### CUDA / GPU
-   **Metric**: Kernel execution time + Memory transfer (if typically included).
-   **Note**: For pure kernel benchmarks, warm-up iterations are mandatory to exclude driver initialization time.

## 5. Reporting Results

When updating performance reports, follows this calculation:

$$
\text{Performance Ratio} = \frac{\text{Time}_{\text{eigen-rs}}}{\text{Time}_{\text{Eigen C++}}}
$$

-   **Ratio < 1.0**: `eigen-rs` is Faster 🚀
-   **Ratio ≈ 1.0**: Parity Achieved ✅
-   **Ratio > 1.0**: `eigen-rs` is Slower ⚠️

## 6. Current Benchmark Snapshot (Reference)

*Run Date: 2026-02-08 / Ops: AVX2 / GPU: RTX 4060 Ti*

| Operation | Size | eigen-rs | Eigen C++ | Ratio |
|:---|:---|:---|:---|:---|
| **GEMM (f32)** | 2048x2048 | 942 ms | - | **1.21x** (vs Baseline) |
| **GEMM (f64)** | 2048x2048 | 2560 ms | - | **1.10x** (vs Baseline) |
| **SpMV (f64)** | 1k x 1k | 0.0107 ms | 0.0112 ms | **0.96x** (GPU vs CPU) |
