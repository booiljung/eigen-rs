# 성능 벤치마크 가이드

이 문서는 `eigen-rs`와 원본 C++ Eigen 라이브러 간의 표준 벤치마크 방법론을 설명합니다. 우리의 주요 목표는 **성능 동등성(Performance Parity)** 이상을 달성하는 것입니다.

## 1. 목표 (Objective)
`eigen-rs`가 Rust의 안전성과 제로 코스트 추상화를 활용하면서도 효율적인 성능을 제공하는지 확인하는 것입니다.

- **핵심 지표**: 실행 시간 비율 ($\text{Time}_{\text{Rust}} / \text{Time}_{\text{C++}}$)
- **목표**: 비율 $\le 1.0$ (이상적) ~ $1.5$ (허용 범위).

## 2. 방법론 (Methodology)
우리는 동일한 선형 대수 연산을 Rust(`eigen-rs`)와 C++(`Eigen 3.4`)에서 동일한 최적화 레벨로 컴파일하여 실행하는 차분 벤치마킹(Differential Benchmarking) 방식을 사용합니다.

### 도구 (Tools)
- **`verification/verify_perf.py`**: 메인 드라이버입니다. 두 벤치마크를 컴파일하고 실행하며, 출력을 파싱하여 보고서를 생성합니다.
- **기준 (Criteria)**:
    - **Rust**: `criterion` (release 모드, `target-cpu=native` 권장).
    - **C++**: `g++ -O3 -march=native`.

## 3. 벤치마크 실행 방법 (How to Run Benchmarks)

### 표준 실행 (비교 분석)
전체 스위트를 실행하고 C++ 대비 성능을 검증하려면:

```bash
# 벤치마크를 실행하고 docs/reports/에 보고서 생성
python3 verification/verify_perf.py

# 무작위 크기로 실행 (과적합 방지)
python3 verification/verify_perf.py --random-sweep

# 실행 후 결과를 이 파일(BENCHMARK_LOG.md)에 자동 추가 (주의해서 사용)
python3 verification/verify_perf.py --update-benchmark
```

### 시스템 메타데이터
도구는 다음 정보를 자동으로 캡처합니다:
- CPU 모델 및 코어 수
- OS 커널 버전
- Rustc 및 G++ 버전

## 4. 보고 표준 (Reporting Standards)

### 보고서 위치
상세한 타임스탬프가 찍힌 보고서는 다음 경로에 생성됩니다:
`docs/reports/performance_YYYY-MM-DD_HHMMSS.md`

> **엄격한 규칙**: 이 파일들을 수동으로 생성하거나 이름을 변경하지 마십시오. 벤치마크 보고서는 데이터 무결성과 포맷 준수를 위해 반드시 `verify_perf.py` 스크립트에 의해 생성되어야 합니다.

### 결과 해석
핵심 지표는 **비율(Ratio)**입니다:
- **Ratio < 1.0**: 🚀 `eigen-rs`가 더 빠름.
- **Ratio ≈ 1.0**: ✅ 동등성 달성.
- **Ratio > 1.5**: ⚠️ 성능 격차 (최적화 필요).

### 커밋 시점 (When to Commit)
로컬 벤치마크 결과를 매번 커밋하지 마십시오. `BENCHMARK_LOG.md` 또는 `docs/reports/` 변경 사항은 다음 경우에만 커밋하십시오:
1.  **마일스톤 달성**: 유의미한 성능 향상 (예: SIMD 구현).
2.  **회귀 수정**: 성능 저하 복구 기록.
3.  **기준 업데이트**: 새로운 릴리스를 위한 베이스라인 업데이트.



## 6. 현재 성능 상태
*최신 수치는 `docs/reports/LATEST_PERFORMANCE.md` 보고서를 참조하십시오.*
