# eigen-rs 로드맵

이 문서는 `eigen-rs` 프로젝트 로드맵의 상위 수준 인덱스 역할을 합니다. 각 모듈에 대한 자세한 내용은 해당 문서에서 추적됩니다.

## 모듈 로드맵

| 모듈 | 설명 | 상태 | 링크 |
|:---|:---|:---|:---|
| **Core** | Dense Matrix 기능, Evaluators, Storage | **100%** | [세부 정보 보기](docs/roadmap/Core.md) |
| **Architecture** | SIMD (AVX/SSE/NEON), GPU (CUDA), 병렬 처리 | **91%** | [세부 정보 보기](docs/roadmap/Arch.md) |
| **Geometry** | 회전, 변환, 공간 프리미티브 | **100%** | [세부 정보 보기](docs/roadmap/Geometry.md) |
| **Decompositions** | LU, QR, Cholesky, SVD, Eigensolvers | **100%** | [세부 정보 보기](docs/roadmap/Decompositions.md) |
| **Sparse** | 희소 행렬 저장소 (CSR/CSC), Solvers | **100%** | [세부 정보 보기](docs/roadmap/Sparse.md) |
| **Unsupported** | FFT, Splines, Polynomials, Tensors, Optimization | **100%** | [세부 정보 보기](docs/roadmap/Unsupported.md) |

## 전략적 철학: 순수 Rust 코어 + 선택적 FFI

이 프로젝트는 **"순수 Rust 우선, FFI는 선택"**이라는 하이브리드 접근 방식을 따릅니다:
1.  **핵심 로직 (순수 Rust)**:
    - 행렬 연산, 분해 (LU/QR/Cholesky), 기하학은 100% safe/unsafe Rust로 구현됩니다.
    - **목표**: 일반적인 확장성 (`Complex`, `DualNumber` 등 지원) 및 의존성 없는 이식성 (WASM/Embedded).
2.  **특수 함수 (하이브리드/FFI)**:
    - 복소수 함수 (`erf`, `bessel`, `gamma`)는 정확성과 성능을 위해 `libc` (시스템 `libm`)를 활용합니다.
    - 향후 목표: 순수 Rust 이식성을 위해 `libm` 크레이트로 점진적 전환.
3.  **고성능 백엔드 (Opt-in FFI)**:
    - 사용자는 `mkl`, `lapack`, `cuda`와 같은 기능을 활성화하여 무거운 계산(GEMM, SVD)을 최적화된 벤더 라이브러리로 오프로드할 수 있습니다.
    - 기본 동작은 최대 호환성을 위해 순수 Rust로 유지됩니다.

## 수치적 안정성 및 품질 보증

- [x] **Fuzzy Comparison**: 구성 가능한 정밀도를 가진 `isApprox`, `isMuchSmallerThan`.
- [x] **Condition Estimation**: 솔버 신뢰성을 위한 `ConditionEstimator`.
- [x] **Scalar Traits**: 부동 소수점, 정수 및 복소수 타입을 위한 `NumTraits`.
- [x] **Validation**: C++ Eigen 바이너리에 대한 차분 테스트 프레임워크.

## 최근 최적화 (2026 캠페인)

| 단계 | 컴포넌트 | 속도 향상 (vs C++ Eigen) | 상태 |
|:---|:---|:---|:---|
| **Phase 10** | `VecDot` / `VecNorm` | **0.76x / 0.42x** (더 빠름) | ✅ 완료 |
| **Phase 12** | `GeneralizedEigen` | **1.27x - 0.79x** (더 빠름) | ✅ 완료 |
| **Phase 13** | `RealSchur` (속도) | **4x 속도 향상** (실행) | ✅ 완료 |
| **Phase 14** | `LDLT` / `Tridiagonal` | **1.44x / 1.24x** | ✅ 완료 |
| **Phase 15** | `Hessenberg` Decomposition | **최적** (할당 없음) | ✅ 완료 |
| **Phase 17** | `RealSchur` (수렴성) | **0.24x** (더 빠름) | ✅ 완료 |
