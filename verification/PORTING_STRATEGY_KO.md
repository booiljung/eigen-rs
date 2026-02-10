# 검증 오라클 방법론: C++ 라이브러리의 Rust 포팅을 위한 청사진

> **대상 독자**: 미래의 AI 에이전트 및 개발자  
> **목적**: Eigen, PCL, OpenCV와 같은 복잡한 C++ 라이브러리를 100% 신뢰도로 Rust로 포팅하기 위한 엄격하고 자동화된, 그리고 게임화된 전략을 제공합니다.

## 1. 핵심 철학: "신뢰하되 검증하라 (Trust but Verify)"

성숙한 C++ 라이브러리를 Rust로 포팅할 때, 원본 라이브러리는 단순한 참고 자료가 아닙니다. 그것은 **오라클(Oracle, 절대적 진리)**입니다. 당신의 목표는 이해를 바탕으로 로직을 '재작성'하는 것이 아니라, 증거를 바탕으로 **동작을 복제**하는 것입니다.

### 황금률 (The Golden Rules)
1.  **완전한 동등성 (Total Parity)**: Rust 구현체는 동일한 입력에 대해 C++ 원본과 *정확히* 동일하게 동작해야 합니다 (정의된 부동 소수점 오차 범위 내).
2.  **수치적 정직성 (Numerical Honesty)**: C++ 라이브러리가 특정 알고리즘(예: Column-major 저장소, 특정 분해 단계)을 사용하는 경우, Rust 관용구(Idioms)를 위해 명시적으로 변경하지 않는 한 Rust 포팅도 이를 따라야 합니다. (변경 시 반드시 문서화 필요).
3.  **게임화 (Gamification)**: "완성도 점수(Perfection Score)" (0-100%)를 사용하여 진행 상황을 추적하십시오. 이는 완료 동기를 부여하고 갭을 시각화합니다.

---

## 2. 차이점 다루기: 왜 "Rust 확장(Extensions)"이 필요한가?

C++에는 존재하지 않는 API를 필연적으로 만들게 될 것입니다. **이는 정상이자 필수적입니다.**

### Rust 전용 API가 필요한 일반적인 이유:
1.  **생명주기 관리 (Lifecycle Management)**: C++는 생성자/소멸자를 사용합니다. Rust는 명시적인 `new()`, `default()`, `clone()`, `to_owned()` 메서드가 필요합니다.
2.  **안전성 추상화 (Safety Abstractions)**: C++는 원시 포인터(`double*`)를 사용합니다. Rust는 `as_slice()`, `as_ptr()`, `iter()`와 같은 안전한 래퍼를 요구합니다.
3.  **트레이트 구현 (Trait Implementations)**: `Display` (`fmt`), `Debug`, `IntoIterator`와 같은 Rust 관용적 기능을 위해서는 특정 메서드가 필요합니다.
4.  **헬퍼 접근자 (Helper Accessors)**: C++의 `friend` 클래스 없이 내부 상태를 테스트하기 위해, Rust는 종종 C++에서는 숨겨져 있거나 암시적인 값을 읽기 전용 접근자(예: `matrix_l()`, `bias()`)로 노출합니다.

**규칙**: 모든 "Rust 확장"은 C++와 차분 테스트를 할 수 없으므로 반드시 **단위 테스트(Unit Tests)**로 검증해야 합니다.

---

## 3. 검증 워크플로우

중요한 점은 검증이 나중에 하는 생각이 아니라는 것입니다. 검증은 개발을 주도하는 드라이버입니다.

### 1단계: API 표면 자동 추출
코드를 작성하기 전에, 영토 지도를 그리십시오.
-   **도구**: `extract_cpp_api.py`
-   **로직**: C++ 헤더를 파싱하여 모든 공개 클래스, 메서드, 고유 오버로드를 추출합니다.
-   **출력**: 포팅해야 할 특정 기능 목록이 담긴 JSON 파일 (예: `Matrix::block`, `LLT::solve`).

### 2단계: 구현 및 자가 추출
Rust로 기능을 구현하면서 수행합니다.
-   **도구**: `extract_rust_api.py`
-   **로직**: Rust 소스 코드(`syn` 또는 정규식 사용)를 파싱하여 당신이 구현했다고 *주장*하는 것을 식별합니다.
-   **출력**: 당신의 Rust 공개 API 목록이 담긴 JSON 파일.

### 3단계: 매처(Matcher) 및 갭 분석 (The Gap Analysis)
-   **도구**: `verify_coverage.py` (심판)
-   **로직**: C++ 목록 vs. Rust 목록 비교.
    -   **일치 (Match)**: `inverse()` == `inv()` (표준 이름 변경 규칙 허용).
    -   **누락 (Missing)**: C++에는 있으나 Rust에는 없음.
    -   **확장 (Extra)**: Rust에는 있으나 C++에는 없음 (Rust Extensions - 기능적으로 검증되어야 함).
-   **게임화**: `완성도(Completeness) = (일치 항목 / 전체 C++ API) * 100`.

### 4단계: 차분 테스팅 (Differential Testing - The "Oracle" Check)
정확성 없는 커버리지는 무의미합니다.
-   **개념**: *정확히 동일한* 데이터를 두 라이브러리에 통과시킵니다.
-   **메커니즘**:
    1.  **데이터 생성**: Rust에서 무작위 행렬/입력을 생성합니다.
    2.  **수출/FFI**: 원본 라이브러리와 링크된 최소한의 C++ 바이너리로 데이터를 전송합니다.
    3.  **연산**: C++ 라이브러리가 연산을 수행합니다.
    4.  **비교**: Rust가 자신의 결과와 C++ 결과가 일치하는지 단언(Assert)합니다 (~1e-6 오차).
-   **엄격한 규칙**: API는 차분 테스팅을 위해 지정된 테스트 파일에서 **명시적으로 사용된 경우에만** `PASS`로 표시됩니다.

---

## 3. 미래 프로젝트 적용 가이드

### 시나리오 A: PCL (Point Cloud Library) 포팅
-   **오라클**: PCL C++ 1.1x
-   **데이터 구조**: `pcl::PointCloud<PointT>`, `pcl::KdTree`
-   **적용**:
    1.  **추출**: `pcl/point_cloud.h`, `pcl/search/kdtree.h` 파싱.
    2.  **차분 테스트**: `.pcd` 파일을 직렬화합니다. C++ PCL에서 로드하여 `VoxelGridFilter` 실행. Rust에서 로드하여 구현체 실행. 포인트 좌표 비교.

### 시나리오 B: OpenCV 포팅
-   **오라클**: OpenCV 4.x
-   **데이터 구조**: `cv::Mat`
-   **적용**:
    1.  **추출**: `opencv2/core.hpp`, `opencv2/imgproc.hpp` 파싱.
    2.  **차분 테스트**:
        -   입력: 표준 이미지 (Lena.jpg).
        -   연산: `cv::Canny` 또는 `cv::warpAffine`.
        -   비교: C++ 출력과 Rust 출력 간의 픽셀 단위 차이(Pixel-by-pixel difference).

---

## 4. "완성도 보고서" 아티팩트

프로젝트 상태의 진실 공급원(Source of Truth) 역할을 하는 `VERIFICATION_REPORT.md`를 생성하십시오.

| API 이름 | C++ 일치 | 차분 테스트 여부? | 상태 |
| :--- | :--- | :--- | :--- |
| `compute()` | `compute()` | ✅ `test_kdtree_diff.rs` | **PASS** |
| `indices()` | `indices()` | ❌ | **FAIL** |
| `to_owned()`| *없음* | ✅ `test_internals.rs` | **PASS (Rust Ext)** |

**최종 완성도 점수**: **98.5%**

---

> **사용자/AI에게**: PCL/OpenCV 프로젝트를 시작할 때, 이 문서를 AI에게 제공하십시오. 그리고 이렇게 명령하십시오: *"이 `PORTING_STRATEGY_KO.md` 파일을 기반으로 검증 툴체인을 먼저 구축해줘."*
