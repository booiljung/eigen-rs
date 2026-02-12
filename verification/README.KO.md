# 검증 및 테스트 (Verification & Testing)

> **현재 상태**: **100.0% 완성도 (Perfection Score)**  
> **최종 검증일**: 2026-02-10

이 디렉토리는 `eigen-rs`가 원본 C++ Eigen 라이브러리와 엄격한 동등성을 유지하도록 보장하는 "검증 오라클(Verification Oracle)" 툴체인을 포함하고 있습니다.

## 📊 검증 리포트 (Verification Report)

-   [**커버리지 리포트 (정적)**](../tmp/verification/VERIFICATION_REPORT.md): API 존재 여부를 확인합니다.
-   [**커버리지 리포트 (정적)**](../tmp/verification/VERIFICATION_REPORT.md): API 존재 여부를 확인합니다.
-   [**동등성 리포트 (동적)**](../tmp/verification/PARITY_REPORT.md): 런타임 수치 동등성을 확인합니다.
-   [**일관성 리포트 (규칙 기반)**](../tmp/verification/CONSISTENCY_REPORT.md): 커버리지와 동등성 간의 일치를 확인합니다.
-   [**성능 리포트 (벤치마크)**](../tmp/verification/PERFORMANCE_REPORT.md): C++ 대비 실행 속도를 비교합니다.

## 📘 방법론 (검증 오라클)

우리는 로직을 재작성하는 것이 아니라, 동작을 복제합니다. 우리의 방법론은 C++ API 표면을 추출하고, 모든 Rust 대응부가 차분 테스팅(Differential Testing)을 통해 정확히 동일하게 동작함을 보장하는 것입니다.

-   [**검증 전략 (English)**](PORTING_STRATEGY.md)
-   [**검증 전략 (Korean)**](PORTING_STRATEGY_KO.md)
-   [**교차 검증 증명 (Proof of Parity)**](PROOF_OF_PARITY.md)

## 🛠️ 툴체인 (Toolchain)

검증 프로세스는 3단계로 구성됩니다:

1.  **C++ API 추출**: 원본 Eigen 헤더를 파싱합니다.
2.  **Rust API 추출**: 우리의 Rust 구현체를 파싱합니다.
3.  **커버리지 검증**: API를 매칭하고 차분 테스트 사용 여부를 확인합니다.
4.  **동등성 검증**: 차분 테스트를 실행하고 수치적 동등성을 확정합니다.
5.  **일관성 검증**: 추출된 API와 검증된 API, 실행된 테스트 간의 3자 일치를 규칙 기반으로 검사합니다.
6.  **성능 검증**: C++ 기준점 대비 실행 속도를 벤치마킹합니다.

### 실행 방법

전체 검증 스위트(커버리지 + 동등성 + 일관성 + 성능)를 실행하려면:

```bash
./run.sh
```

## 📂 디렉토리 구조

-   `run.sh`: 메인 진입점 스크립트.
-   `verify_coverage.py`: API 매칭 심판 로직.
-   `verify_parity.py`: 차분 테스트 실행기.
-   `verify_three_way.py`: 3자 일관성 검사기.
-   `verify_perf.py`: 성능 벤치마크 실행기.
-   `extract_cpp_api.py`: C++ 헤더 파서.
-   `extract_rust_api.py`: Rust 소스 파서.
