# Performance Verification Report: 2026-02-13_035046
> **Generated**: 2026-02-13T03:50:46.931543
> **Threshold**: < 2.0x of C++ Eigen

## System Context
- **CPU**: 13th Gen Intel(R) Core(TM) i5-13500
- **OS**: Linux 6.17.0-14-generic
- **Rust**: rustc 1.91.1 (ed61e7d7e 2025-11-07)
- **C++**: g++ (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0

## Benchmark Results
| Operation | Size | C++ (ns) | Rust (ns) | Ratio | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| EigenValues | 32 | 32633 | 63926 | **1.96x** | ✅ PASS |
| EigenValues | 64 | 131999 | 392254 | **2.97x** | ❌ FAIL (Slow) |
| LLT | 64 | 5899 | 24856 | **4.21x** | ❌ FAIL (Slow) |
| LLT | 128 | 25320 | 234179 | **9.25x** | ❌ FAIL (Slow) |
| LLT | 256 | 118267 | 623116 | **5.27x** | ❌ FAIL (Slow) |
| MatMul | 64 | 12288 | 107203 | **8.72x** | ❌ FAIL (Slow) |
| MatMul | 128 | 92621 | 635285 | **6.86x** | ❌ FAIL (Slow) |
| MatMul | 256 | 358401 | 2828514 | **7.89x** | ❌ FAIL (Slow) |
| SVD | 32 | 202896 | 776388 | **3.83x** | ❌ FAIL (Slow) |
| SVD | 64 | 1167849 | 4100065 | **3.51x** | ❌ FAIL (Slow) |