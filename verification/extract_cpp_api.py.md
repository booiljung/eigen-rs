# C++ API Extractor (`extract_cpp_api.py`)

This script is the **First Pillar** of the Verification Oracle toolchain. Its purpose is to map the "territory" of the original C++ Eigen library by extracting the list of public methods that `eigen-rs` must implement.

## 🎯 Purpose (목적)

*   To parse the original Eigen 3.4 source code headers.
*   To identify all public classes (e.g., `Matrix`, `LLT`) and their methods.
*   To generate a "Golden Standard" list (`cpp_api_list.json`) against which the Rust port is measured.

## ⚙️ How it Works (동작 원리)

Since C++ is complex to parse, this script uses a **heuristic Regex-based approach** tailored specifically for Eigen's coding style.

1.  **Input Source**: Scans `eigen-src/Eigen/src` recursively (All modules: Core, Geometry, LU, SVD, Sparse, etc.).
2.  **Class Detection**: Looks for `class ClassName {` patterns.
3.  **Access Control**: Tracks `public:` and `private:` sections to ensure only public APIs are extracted.
4.  **Method Extraction**:
    *   Searches for the `EIGEN_DEVICE_FUNC` macro, which fronts almost all Eigen methods.
    *   Extracts method names and operator overloads (e.g., `operator+`).
    *   Ignores constructors, destructors, and private helpers.

## 🚀 Usage (사용법)

```bash
python3 verification/extract_cpp_api.py
```

## 📄 Output Format (출력 형식)

The script generates `tmp/verification/cpp_api_list.json`:

```json
{
  "Matrix": [
    "block",
    "col",
    "determinant",
    "inverse",
    "operator*",
    "transpose",
    ...
  ],
  "LLT": [
    "matrixL",
    "solve",
    ...
  ]
}
```

## 🧩 Role in Verification (검증에서의 역할)

This output acts as the **"Expected"** side of the equation. The `verify_coverage.py` script compares this list against the Rust implementation to detect:
*   **Missing Features**: APIs present in C++ but missing in Rust.
*   **Naming Mismatches**: APIs that exist but have different names.
