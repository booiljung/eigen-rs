# Performance Benchmark Guide

This document outlines the standard methodology for benchmarking `eigen-rs` against the original C++ Eigen library. Our primary goal is to achieve **performance parity** or better.

## 1. Objective
The goal is to ensure `eigen-rs` provides comparable performance efficiently using Rust's safety and zero-cost abstractions.

- **Target Metric**: Execution Time Ratio ($\text{Time}_{\text{Rust}} / \text{Time}_{\text{C++}}$)
- **Goal**: Ratio $\le 1.0$ (Ideal) to $1.5$ (Acceptable).

## 2. Methodology
We use a differential benchmarking approach where the same linear algebra operations are executed in both Rust (`eigen-rs`) and C++ (`Eigen 3.4`), compiled with identical optimization levels.

### Tools
- **`verification/verify_perf.py`**: The main driver. It compilies both benchmarks, runs them, parses output, and generates reports.
- **Criteria**:
    - **Rust**: `criterion` (release mode, `target-cpu=native` recommended).
    - **C++**: `g++ -O3 -march=native`.

## 3. How to Run Benchmarks

### Standard Run (Comparative)
To run the full suite and verify performance against C++:

```bash
# Run benchmarks and generate a report in docs/reports/
python3 verification/verify_perf.py

# Run and automatically append the summary to this file (Use carefully)
python3 verification/verify_perf.py --update-benchmark
```

### System Metadata
The tool automatically captures:
- CPU Model & Cores
- OS Kernel Version
- Rustc & G++ Versions

## 4. Reporting Standards

### Report Location
Detailed, timestamped reports are generated in:
`docs/reports/performance_YYYY-MM-DD_HHMMSS.md`

### Interpreting Results
The key metric is the **Ratio**:
- **Ratio < 1.0**: 🚀 `eigen-rs` is Faster.
- **Ratio ≈ 1.0**: ✅ Parity Achieved.
- **Ratio > 1.5**: ⚠️ Performance Gap (Optimization Required).

### When to Commit
Do not commit every local benchmark run. Commit changes to `BENCHMARK_LOG.md` or `docs/reports/` only when:
1.  **A Milestone is Reached**: Significant performance improvement (e.g., SIMD implementation).
2.  **A Regression is Fixed**: Documenting the recovery.
3.  **Reference Update**: Updating the baseline for a new release.



## 6. Current Performance Status
*Refer to the latest report in `docs/reports/LATEST_PERFORMANCE.md` for the most up-to-date numbers.*