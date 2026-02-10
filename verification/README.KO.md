# 검증 및 테스트 (Verification & Testing)

> **현재 상태**: **100.0% 완성도 (Perfection Score)**  
> **최종 검증일**: 2026-02-10

이 디렉토리는 `eigen-rs`가 원본 C++ Eigen 라이브러리와 엄격한 동등성을 유지하도록 보장하는 "검증 오라클(Verification Oracle)" 툴체인을 포함하고 있습니다.

## 📊 검증 리포트 (Verification Report)

모든 API와 그 검증 상태를 나열한 상세 리포트가 자동으로 생성됩니다.

-   [**전체 리포트 보기**](../tmp/verification/VERIFICATION_REPORT.md)

## 📘 방법론 (검증 오라클)

우리는 로직을 재작성하는 것이 아니라, 동작을 복제합니다. 우리의 방법론은 C++ API 표면을 추출하고, 모든 Rust 대응부가 차분 테스팅(Differential Testing)을 통해 정확히 동일하게 동작함을 보장하는 것입니다.

-   [**검증 전략 (English)**](PORTING_STRATEGY.md)
-   [**검증 전략 (Korean)**](PORTING_STRATEGY_KO.md)

## 🛠️ 툴체인 (Toolchain)

검증 프로세스는 3단계로 구성됩니다:

1.  **C++ API 추출**: 원본 Eigen 헤더를 파싱합니다.
2.  **Rust API 추출**: 우리의 Rust 구현체를 파싱합니다.
3.  **커버리지 검증**: API를 매칭하고 차분 테스트 사용 여부를 확인합니다.

### 실행 방법

전체 검증 스위트를 실행하고 리포트를 재생성하려면:

```bash
./run.sh
```

## 📂 디렉토리 구조

-   `run.sh`: 메인 진입점 스크립트.
-   `verify_coverage.py`: API를 매칭하고 점수를 계산하는 심판 로직.
-   `extract_cpp_api.py`: C++ 헤더 파서.
-   `extract_rust_api.py`: Rust 소스 파서.
