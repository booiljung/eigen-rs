# eigen-rs 검증 도구 체인 (Verification Toolchain)

이 디렉토리는 원본 C++ Eigen 라이브러리와 비교하여 `eigen-rs` 라이브러리의 "완벽성(Perfection)"을 증명하기 위해 설계된 규칙 기반 검증 도구 체인을 포함하고 있습니다.

## 디렉토리 구조 (Directory Structure)

- **`extract_cpp_api.py`**: C++ Eigen 헤더에서 공개(public) API를 추출합니다.
- **`extract_rust_api.py`**: `eigen-rs` Rust 소스에서 공개(public) API를 추출합니다.
- **`verify_coverage.py`**: 추출된 API를 1:1로 매칭하고, Differential Test 존재 여부를 확인하여 검증 리포트를 생성합니다.
- **`meta_verify.py`**: 무결성 검사 스크립트입니다. 합성 더미 데이터를 사용하여 추출기(Extractor)와 매처(Matcher)가 올바르게 작동하는지 검증합니다.
- **`data/`**: 삭제됨. 대신 `tmp/verification/`을 사용하십시오.

## 출력 (Output)

모든 생성된 파일은 `tmp/verification/` 디렉토리에 저장됩니다:
- `tmp/verification/cpp_api_list.json`
- `tmp/verification/rust_api_list.json`
- `tmp/verification/VERIFICATION_REPORT.md`

## 실행 방법 (How to Run)

### 1. 전체 검증 (Full Verification)
전체 검증 파이프라인을 실행하고 `VERIFICATION_REPORT.md`를 생성하려면:

```bash
./run.sh
```

또는 수동으로 실행:
```bash
python3 extract_cpp_api.py
python3 extract_rust_api.py
python3 verify_coverage.py
```

비교 리포트는 `tmp/verification/VERIFICATION_REPORT.md`에 생성됩니다.

### 2. 무결성 검사 (Meta-Verification)
도구 자체가 올바르게 작동하는지 검증하려면:

```bash
./run.sh --meta
```

또는 수동으로 실행:
```bash
python3 meta_verify.py
```

## 방법론 (Methodology)

1.  **추출 (Extraction)**: C++ 헤더에서는 `EIGEN_DEVICE_FUNC`가 붙은 공개 메서드를, Rust 소스에서는 `pub fn`을 정규식(Regex)으로 스캔합니다.
2.  **매칭 (Matching)**: Rust 메서드 이름을 C++ 대응 항목으로 매핑합니다 (예: `add` -> `operator+`, `rows` -> `rows`).
3.  **검증 (Verification)**: 모든 매칭된 항목에 대해 Differential Test가 존재하는지 확인합니다 (`run_cpp_harness` 호출 또는 특정 테스트 패턴 포함 여부).
4.  **리포팅 (Reporting)**: 각 API를 다음과 같이 분류합니다:
    - `PASS`: 매칭됨 & 테스트됨.
    - `WARN`: Rust 전용 확장 기능 (기능적 검증됨).
    - `FAIL`: 테스트 커버리지 없음.
