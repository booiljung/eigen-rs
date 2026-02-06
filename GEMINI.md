## 프로젝트 개요

- 목적: C++ **Eigen 3.4** 라이브러리의 Rust 이식
- 프로젝트명: **`eigen-rs`**

## 기술 스택

- 언어: **Rust 1.75+**
- 도구: **Cargo** (빌드/의존성 보관)
- 최적화:
  - **SIMD** (x86_64, ARM64 지원)
  - **CUDA 11.0+** (GPU 가속, 시스템 독립적 호환성 유지)
- 인프라: **GitHub Actions** (CI/CD)

## 개발 규칙

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
- **교차 검증**: 동일한 입력에 대해 C++ Eigen의 결과와 Rust 구현의 결과를 비교하는 테스트 수행 (Differential Testing)
- **모듈화**: `#[cfg(test)]` 내 구현
- **유연성**: SIMD/CUDA 환경 부재 시 자동 fallback 제공
- **CI 연동**: `cargo test` 통과 시에만 브런치 병합 가능

### 3. 문서화
- **README.md**: 개요, 설치, 예제 포함
- **CHANGELOG.md**: 변경 이력 관리
- **API 문서**: `cargo doc` 기준 최신화 유지

### 4. 외부 소스 관리
- **Submodule 활용**: 외부 소스 코드(Eigen 원본 등)를 가져올 때는 `git clone` 대신 반드시 `git submodule add`를 사용

## 핵심 설계 원칙 (Advanced)

### 1. Zero-Cost 추상화
- **지연 연산**: Rust Traits/GATs를 활용하여 중간 임시 객체 생성을 방지하는 Expression Templates 구현
- **메모리 정렬**: SIMD 최적화를 위해 16/32/64바이트 단위의 정렬된 메모리 할당 관리
- **Const Generics**: 고정 크기(Fixed)와 동적 크기(Dynamic) 행렬의 우아한 통합

### 2. 수치적 안정성 및 확장성
- **Scalar Generics**: `f32`, `f64`, `Complex` 및 사용자 정의 수치 타입 지원
- **벤치마크**: `criterion`을 사용한 원본 C++ Eigen과의 객관적 성능 비교 인프라 구축
- **테스트 커버리지**: `tarpaulin` 등을 활용한 높은 코드 커버리지 유지 및 관리
- **라이선스**: 원작의 정신을 계승하여 MPL 2.0 정책 검토

## CI/CD
- **자동화**: GitHub Actions 기반 빌드/테스트
- **트리거**: main 브랜치 push 및 PR 생성 시 실행
- **병합 조건**: CI 통과 필수
