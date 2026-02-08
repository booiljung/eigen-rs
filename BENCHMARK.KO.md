# 벤치마크 방법론 (Benchmark Methodology)

이 문서는 `eigen-rs`와 원본 C++ Eigen 3.4 라이브러리 간의 성능을 비교하기 위한 표준 절차를 정의합니다.

## 1. 하드웨어 환경 (필수)

모든 벤치마크 실행은 재현성을 보장하기 위해 정확한 하드웨어 환경을 명시하는 것으로 시작해야 합니다.

**필수 기재 항목:**
-   **CPU**: 모델명, 기본 클럭, 코어/스레드 수 (예: `AMD Ryzen 9 7950X, 4.5GHz, 16C/32T`).
-   **SIMD 지원**: 사용 가능한 확장 명령어 셋 (예: `AVX2`, `FMA`, `AVX-512`).
-   **GPU**: 모델명, VRAM, CUDA 버전 (예: `NVIDIA RTX 4090, 24GB, CUDA 12.2`).
-   **Memory**: 용량 및 타입 (예: `64GB DDR5-6000`).
-   **OS**: 커널 버전 (예: `Linux 6.5 generic`).

## 2. 비교 전략 (Comparison Strategy)

모든 성능 지표에 대해 **2차원 분석(Two-Dimensional Analysis)**을 요구합니다.

### A. 내부 확장성 (Internal Scaling)
하드웨어 가속 이득을 검증하기 위해 `eigen-rs` 구현체끼리 비교합니다.
*   **Scalar vs SIMD**: 벡터화 가속도 검증 (예상치: SSE에서 f32 기준 약 4배, AVX에서 약 8배).
*   **CPU vs GPU**: 오프로딩 효율성 검증. GPU가 CPU를 앞서는 **교차점(crossover point)** 정의 (예: N > 1024).

### B. 외부 정합성 (External Parity)
`eigen-rs` (최적화 버전)와 C++ Eigen 3.4 (최적화 버전)를 비교합니다. 이 프로젝트의 핵심 목표입니다.
*   **목표**: 비율(Ratio) $\approx 1.0$.
*   **방법**: `eigen-rs` (SIMD/GPU) vs `Eigen C++` (SIMD/GPU).

## 3. 벤치마크 실행 방법

### A. Rust 벤치마크 (eigen-rs)

`cargo bench`를 사용하여 Criterion 벤치마크를 실행합니다. **주의**: 반드시 Native CPU 최적화를 활성화해야 합니다.

```bash
# 1. 스칼라 기준선 (Scalar Baseline - SIMD 없음)
RUSTFLAGS="-C target-cpu=generic" cargo bench --bench matrix_bench

# 2. SIMD 최적화 (Native)
RUSTFLAGS="-C target-cpu=native" cargo bench --bench matrix_bench
```

### B. C++ 벤치마크 (Eigen 3.4)

우리는 `tests/cpp_harness/`에 C++ 참조 구현을 제공합니다. 이를 컴파일하여 기준선을 수립합니다.

```bash
# 동등한 플래그로 컴파일
g++ -O3 -march=native -DNDEBUG -I /usr/include/eigen3 tests/cpp_harness/gemm_bench.cpp -o gemm_bench

# 실행
./gemm_bench
```

## 4. 보고서 양식 (Reporting Template)

**Hardware**: [CPU/GPU 스펙 입력]

| Operation | Size | Implementation | Time (ms) | Speedup (vs Scalar) | vs Eigen C++ |
|:---|:---|:---|:---|:---|:---|
| **GEMM (f32)** | 1024x1024 | **eigen-rs (Scalar)** | 1200.0 | 1.0x | - |
| | | **eigen-rs (AVX2)** | 150.0 | **8.0x** | 0.98x |
| | | **eigen-rs (CUDA)** | 45.0 | **26.6x** | 1.05x |
| | | **Eigen C++ (AVX2)** | 148.0 | - | - |

### Dense Algebra (GEMM)
-   **지표**: $C = A \times B$ 실행 시간 (ms).
-   **크기**: 64x64, 256x256, 1024x1024.
-   **데이터 타입**: `f32`, `f64`.

### Sparse Algebra (SpMV)
-   **지표**: $y = \alpha A x + \beta y$ 실행 시간 (ms).
-   **희소성(Sparsity)**: ~1-5% 채움 비율 (예: 행당 10개 non-zeros).
-   **저장공간**: CSR (Compressed Sparse Row).

### CUDA / GPU
-   **지표**: 커널 실행 시간 + 메모리 전송 시간 (일반적으로 포함되는 경우).
-   **참고**: 순수 커널 벤치마크의 경우, 드라이버 초기화 시간을 배제하기 위해 워밍업 반복이 필수입니다.

## 5. 결과 보고 기준

성능 보고서를 업데이트할 때, 다음 계산식을 따릅니다:

$$
\text{Performance Ratio} = \frac{\text{Time}_{\text{eigen-rs}}}{\text{Time}_{\text{Eigen C++}}}
$$

-   **Ratio < 1.0**: `eigen-rs`가 더 빠름 🚀
-   **Ratio ≈ 1.0**: 동등 수준 달성 ✅
-   **Ratio > 1.0**: `eigen-rs`가 더 느림 ⚠️

## 6. 현재 벤치마크 스냅샷 (참고용)

*실행 일자: 2026-02-08 / Ops: AVX2 / GPU: RTX 4060 Ti*

| Operation | Size | eigen-rs | Eigen C++ | Ratio |
|:---|:---|:---|:---|:---|
| **GEMM (f32)** | 256x256 | 1.45 ms | 1.41 ms | **1.03x** |
| **SpMV (f64)** | 1k x 1k | 0.0107 ms | 0.0112 ms | **0.96x** (GPU vs CPU) |
