## 프로젝트 개요

- 목적: C++ **Eigen 3.4** 라이브러리의 Rust 이식
- 프로젝트명: **`eigen-rs`**
- **Acknowledgement**: 본 프로젝트는 원본 [Eigen](https://eigen.tuxfamily.org/) 라이브러리의 업적을 존중하며 그 정신을 계승합니다.

## 기술 스택

- 언어: **Rust 1.75+**
- 도구: **Cargo** (빌드/의존성 보관)
- 최적화:
  - **SIMD** (x86_64, ARM64 지원)
  - **CUDA 11.0+** (GPU 가속, 시스템 독립적 호환성 유지)
- 인프라: **GitHub Actions** (CI/CD)

## 개발 규칙

### 0. 운영 및 윤리
- **SemVer 준수**: 시맨틱 버저닝을 엄격히 따라 하위 호환성을 보호함
- **투명성**: 구현된 기능과 미구현 기능을 명확히 고지하여 사용자 혼란 방지
- **기여 예우**: 오픈소스 기여자에 대한 명확한 가이드 및 기술적 존중 유지

### 1. 코드 스타일
- **Rustfmt**: `cargo fmt` 통과 필수
- **Clippy**: 모든 경고 수정 필수 (`cargo clippy`)
- **주석**: 모든 public API에 doc comment 작성
- **에러 처리**:
  - `unwrap()`, `expect()` 사용 금지
  - `Result`, `Option` 기반의 명시적 처리 원칙
- **소유권 관리**:
  - 불필요한 `clone()` 최소화
  - 참조(Borrowing) 및 라이프타임 활용 최우선

### 2. 테스트
- **전수 테스트**: 모든 기능 단위 테스트 포함
- **교차 검증(Differential Testing)**:
  - 동일한 입력에 대해 C++ Eigen의 결과와 Rust 구현의 결과를 실시간으로 비교함
  - **Rationale**: 시스템/컴파일러 환경마다 미세하게 다를 수 있는 부동소수점 연산 결과를 동일 환경에서 대조하여 수치적 정직성을 보장하기 위함 (Runtime C++ Harness 방식 채택)
- **모듈화**: `#[cfg(test)]` 내 구현
- **유연성**: SIMD/CUDA 환경 부재 시 자동 fallback 제공
- **CI 연동**: `cargo test` 통과 시에만 브런치 병합 가능

### 3. 문서화
- **README.md**: 개요, 설치, 예제 포함
- **CHANGELOG.md**: 변경 이력 관리 
- **API 문서**: `cargo doc` 기준 최신화 유지

### 4. 외부 소스 관리
- **Submodule 활용**: 외부 소스 코드(Eigen 원본 등)를 가져올 때는 `git clone` 대신 반드시 `git submodule add`를 사용

### 5. 성능 관리
- **측정 도구**: `criterion` 라이브러리를 사용한 정밀 벤치마크 수행
- **최적화 빌드**: 하드웨어 가속(SIMD)의 공정한 성능 평가를 위해 반드시 `RUSTFLAGS="-C target-cpu=native"` 환경에서 측정함
- **결과 기록**: 모든 성능 측정 결과는 `BENCHMARK.md`에 공식 기록하며, 다음 항목을 포함함:
  - 연산 유형 및 행렬 크기 (예: GEMM 256x256)
  - `eigen-rs` vs `Eigen 3.4` 수행 시간 대조
  - 성능 비율 (Ratio = eigen-rs / Eigen 3.4) 명시
- **목표 가치**: 핵심 연산(GEMM, Decompositions)에 대해 Eigen 대비 **1.5x 이내**의 성능 유지를 기본 목표로 함

## 핵심 설계 원칙 (Advanced)

### 1. Zero-Cost 추상화
- **지연 연산**: Rust Traits/GATs를 활용하여 중간 임시 객체 생성을 방지하는 Expression Templates 구현
- **메모리 정렬**: SIMD 최적화를 위해 16/32/64바이트 단위의 정렬된 메모리 할당 관리
- **Const Generics**: 고정 크기(Fixed)와 동적 크기(Dynamic) 행렬의 우아한 통합

### 2. 수치적 안정성 및 확장성
- **Scalar Generics**: `f32`, `f64`, `Complex` 및 사용자 정의 수치 타입 지원
- **수치적 정직성**: 연산 오차 허용 범위(epsilon)를 명확히 하고, 정밀도 희생 시 사용자에게 경고
- **벤치마크**: `criterion`을 사용한 원본 C++ Eigen과의 객관적 성능 비교 인프라 구축
- **테스트 커버리지**: `tarpaulin` 등을 활용한 높은 코드 커버리지 유지 및 관리
- **라이선스**: 원작의 정신을 계승하여 MPL 2.0 정책 검토

## CI/CD
- **자동화**: GitHub Actions 기반 빌드/테스트
- **트리거**: main 브랜치 push 및 PR 생성 시 실행
- **병합 조건**: CI 통과 필수

## 6. 벤치마크 방법론 (Benchmark Methodology)

이 섹션은 `eigen-rs`와 원본 C++ Eigen 3.4 라이브러리 간의 성능을 비교하기 위한 표준 절차를 정의합니다.

### A. 하드웨어 환경 (필수)
모든 벤치마크 실행은 재현성을 보장하기 위해 정확한 하드웨어 환경을 명시하는 것으로 시작해야 합니다.

**필수 기재 항목:**
-   **CPU**: 모델명, 기본 클럭, 코어/스레드 수 (예: `AMD Ryzen 9 7950X, 4.5GHz, 16C/32T`).
-   **SIMD 지원**: 사용 가능한 확장 명령어 셋 (예: `AVX2`, `FMA`, `AVX-512`).
-   **GPU**: 모델명, VRAM, CUDA 버전 (예: `NVIDIA RTX 4090, 24GB, CUDA 12.2`).
-   **Memory**: 용량 및 타입 (예: `64GB DDR5-6000`).
-   **OS**: 커널 버전 (예: `Linux 6.5 generic`).

### B. 비교 전략 (Comparison Strategy)
모든 성능 지표에 대해 **2차원 분석(Two-Dimensional Analysis)**을 요구합니다.

#### 1. 내부 확장성 (Internal Scaling)
하드웨어 가속 이득을 검증하기 위해 `eigen-rs` 구현체끼리 비교합니다.
*   **Scalar vs SIMD**: 벡터화 가속도 검증 (예상치: SSE에서 f32 기준 약 4배, AVX에서 약 8배).
*   **CPU vs GPU**: 오프로딩 효율성 검증. GPU가 CPU를 앞서는 **교차점(crossover point)** 정의 (예: N > 1024).

#### 2. 외부 정합성 (External Parity)
`eigen-rs` (최적화 버전)와 C++ Eigen 3.4 (최적화 버전)를 비교합니다. 이 프로젝트의 핵심 목표입니다.
*   **목표**: 비율(Ratio) $\approx 1.0$.
*   **방법**: `eigen-rs` (SIMD/GPU) vs `Eigen C++` (SIMD/GPU).

### C. 벤치마크 실행 방법

#### 1. Rust 벤치마크 (eigen-rs)
```bash
# 스칼라 기준선 (Scalar Baseline - SIMD 없음)
RUSTFLAGS="-C target-cpu=generic" cargo bench --bench matrix_bench

# SIMD 최적화 (Native)
RUSTFLAGS="-C target-cpu=native" cargo bench --bench matrix_bench
```

#### 2. C++ 벤치마크 (Eigen 3.4)
```bash
# 동등한 플래그로 컴파일
g++ -O3 -march=native -DNDEBUG ...
```

### D. 보고서 양식 (Reporting Template)
**Hardware**: [CPU/GPU 스펙 입력]

| Operation | Size | Implementation | Time (ms) | Speedup (vs Scalar) | vs Eigen C++ |
|:---|:---|:---|:---|:---|:---|
| **GEMM (f32)** | 1024x1024 | **eigen-rs (Scalar)** | 1200.0 | 1.0x | - |
| | | **eigen-rs (AVX2)** | 150.0 | **8.0x** | 0.98x |
| | | **eigen-rs (CUDA)** | 45.0 | **26.6x** | 1.05x |
| | | **Eigen C++ (AVX2)** | 148.0 | - | - |

### E. 결과 보고 기준
성능 보고서를 업데이트할 때, 다음 계산식을 따릅니다:

$$
\text{Performance Ratio} = \frac{\text{Time}_{\text{eigen-rs}}}{\text{Time}_{\text{Eigen C++}}}
$$

-   **Ratio < 1.0**: `eigen-rs`가 더 빠름 🚀
-   **Ratio ≈ 1.0**: 동등 수준 달성 ✅
-   **Ratio > 1.0**: `eigen-rs`가 더 느림 ⚠️
