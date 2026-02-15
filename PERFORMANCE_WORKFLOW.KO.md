# 성능 최적화 워크플로우 (Performance Optimization Workflow)

이 문서는 `eigen-rs`의 성능 최적화를 위한 표준 절차를 설명합니다.

## 워크플로우 단계 (Steps)

1.  **코드 최적화 (Code Optimization)**
    - 병목 지점 식별 및 코드 수정 (예: GEMM, SIMD).
    - `cargo fmt` 및 `cargo clippy` 준수.

2.  **컴파일 및 빌드 (Compilation & Build)**
    - `cargo build --release`

3.  **정확성 검증 및 수정 (Correctness Verification)**
    - **필수**: 성능 측정 전 정확성 검증.
    - `cargo test` 및 C++ 대조 테스트 실행.
    - 오류 수정 후 다음 단계로 진행.

4.  **성능 측정 (Benchmark Measurement)**
    - [BENCHMARK.KO.md](BENCHMARK.KO.md)에 정의된 표준 스크립트를 실행합니다.
    - **명령어**: `python3 verification/verify_perf.py`
    - *상세 옵션(예: `--random-sweep`)은 [BENCHMARK.KO.md](BENCHMARK.KO.md)를 참조하십시오.*

5.  **분석 및 보고 (Analysis & Reporting)**
    - `docs/reports/`의 보고서를 검토하고 `Ratio`를 비교합니다.
    - **성공 시**:
        - 코드 커밋.
        - 마일스톤인 경우 [BENCHMARK.KO.md](BENCHMARK.KO.md)의 "보고 표준"에 따라 로그를 남깁니다.
    - **실패 시**: 1단계로 복귀.
