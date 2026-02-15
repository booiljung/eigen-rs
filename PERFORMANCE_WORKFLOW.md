# Performance Optimization Workflow

This document outlines the standard procedure for optimizing `eigen-rs` performance. Follow these steps iteratively.

## Workflow Steps

1.  **Code Optimization (Implementation)**
    - Identify a bottleneck (e.g., GEMM, SIMD packing).
    - Modify the code to implement optimization.
    - Ensure code quality: `cargo fmt` and `cargo clippy`.

2.  **Compilation & Build**
    - `cargo build --release`

3.  **Correctness Verification (Debug)**
    - **Crucial Step**: Verify correctness **before** measuring performance.
    - Run standard tests: `cargo test`
    - Run differential tests against C++ (e.g., `cargo test --test matrix_large_mul_test`).
    - **Note**: Fix any logic errors or panics before proceeding.

4.  **Performance Measurement (Benchmark)**
    - Execute the benchmark script as defined in [BENCHMARK.md](BENCHMARK.md).
    - **Command**: `python3 verification/verify_perf.py`
    - *Refer to [BENCHMARK.md](BENCHMARK.md) for detailed usage (e.g., `--random-sweep`).*

5.  **Analysis & Reporting**
    - Review the generated report in `docs/reports/`.
    - Compare the `Ratio` against the baseline.
    - **Success (Improved & Correct)**:
        - Commit the code.
        - If it's a milestone, follow the "Reporting Standards" in [BENCHMARK.md](BENCHMARK.md) to log it.
    - **Failure**: Repeat from Step 1.
