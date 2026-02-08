# eigen-rs

[![CI](https://github.com/user/eigen-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/user/eigen-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/eigen-rs.svg)](https://crates.io/crates/eigen-rs)
[![Docs.rs](https://docs.rs/eigen-rs/badge.svg)](https://docs.rs/eigen-rs)
[![License](https://img.shields.io/crates/l/eigen-rs.svg)](https://github.com/user/eigen-rs/blob/main/LICENSE)
[![English](https://img.shields.io/badge/Language-English-blue.svg)](README.md)

**Eigen 3.4** C++ 라이브러리의 고성능 Rust 포트입니다.

> **주의 (Warning)**
> 이 프로젝트는 **AI에 의해 생성되고 검증되었습니다**. C++ Eigen에 대한 포괄적인 자동화 테스트와 벤치마크를 통과했으나, 아직 **사람에 의해 검증되지 않았습니다**. 프로덕션 환경에서의 사용에 주의하십시오.

`eigen-rs`는 원본 Eigen 라이브러리와 동일한 수치적 안정성, 성능, 그리고 우아한 API를 제공하는 것을 목표로 하며, Rust의 안전성과 Const Generics, Trait 기반 SIMD/CUDA 디스패칭과 같은 현대적인 기능을 활용합니다.

## 기여하기 (Contributing)
기여에 관심이 있으신가요? 브랜치 전략 및 개발 워크플로우에 대한 자세한 내용은 [기여 가이드 (Contributing Guide)](CONTRIBUTING.md)를 참조하세요.

## 특징 (Features)

- **Header-only Spirit**: Zero-Cost 추상화를 위해 최적화된 가벼운 구조.
- **Lazy Evaluation**: 임시 할당을 최소화하기 위한 고급 표현식 템플릿 시스템.
- **하드웨어 가속 (Hardware Acceleration)**:
    - **SIMD**: 아키텍처 불문 Trait을 통한 SSE, AVX, NEON 지원.
    - **CUDA**: 대규모 산술 표현식에 대한 자동 GPU 오프로딩.
- **수치 분해 (Numerical Decompositions)**: LU, QR, Cholesky (LLT/LDLT), JacobiSVD를 포함한 강력한 솔버.
- **기하 모듈 (Geometry Module)**: 로보틱스 및 컴퓨터 그래픽스를 위한 쿼터니언 및 선형 변환.

## 사용법 (Usage)

`Cargo.toml`에 다음을 추가하세요:

```toml
[dependencies]
eigen-rs = "3.4.0"
```

### 예제: 행렬 연산

```rust
use eigen_rs::core::matrix::MatrixX;

fn main() {
    let mut a = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    let mut b = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    
    // 행렬 채우기...
    
    let c = &a + &b * 2.0;
    println!("Matrix C:\n{}", c);
}
```

### 예제: 선형 시스템 풀이

```rust
let llt = matrix.llt().expect("Matrix must be positive-definite");
let x = llt.solve(&b).expect("Solve failed");
```

## 문서 (Documentation)

-   [**Benchmarks**](BENCHMARK.md): 방법론 및 성능 결과 (영문).
-   [**벤치마크 (Korean)**](BENCHMARK.KR.md): 벤치마크 방법론 국문 번역.
-   [**Roadmap**](ROADMAP.md): 핵심 모듈별 기능 동등성 달성 현황.

## 로드맵 (Roadmap)

기능 동등성 진행 상황에 대한 자세한 내용은 [ROADMAP.md](ROADMAP.md)를 참조하세요.

## 감사의 글 (Acknowledgements)

이 프로젝트는 Benoît Jacob과 Gaël Guennebaud가 개발한 **Eigen**을 존중하는 마음으로 포팅한 결과물입니다. `eigen-rs`는 그들의 성능과 수학적 우아함에 대한 헌신을 계승합니다. C++ 선형대수의 표준을 정립해 준 모든 Eigen 커뮤니티에 깊은 감사를 표합니다.

