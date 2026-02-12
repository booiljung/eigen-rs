# Rust API Extractor (`extract_rust_api.py`)

This script is the **Second Pillar** of the Verification Oracle toolchain. Its purpose is to map the "Reality" of the `eigen-rs` implementation by extracting the list of public methods currently available to users.

## 🎯 Purpose (목적)

*   To parse the `eigen-rs` source code (`src/`).
*   To identify all public structs (e.g., `Matrix`, `LLT`) and their public methods.
*   To generate a "Reality Check" list (`rust_api_list.json`) representing the actual implemented surface.

## ⚙️ How it Works (동작 원리)

The script uses a **custom text parser** (Regex + State Machine) to handle Rust's `impl` block structure without requiring a heavy compiler frontend.

1.  **Input Source**: Scans `src/` recursively.
2.  **Struct Detection**: Parses `impl StructName` lines.
    *   Handles generics: `impl<T> Matrix<T>` → `Matrix`
    *   Handles traits: `impl Display for Matrix` → `Matrix`
3.  **Method Extraction**:
    *   Searches for `pub fn method_name(...)` patterns within `impl` blocks.
    *   Ignores private functions.
    *   Groups methods by their owning Struct.

## 🚀 Usage (사용법)

```bash
python3 verification/extract_rust_api.py
```

## 📄 Output Format (출력 형식)

The script generates `tmp/verification/rust_api_list.json`:

```json
{
  "Matrix": [
    "as_slice",
    "block",
    "determinant",
    "inverse",
    "iter",
    "new",
    ...
  ],
  "LLT": [
    "matrix_l",
    "solve",
    ...
  ]
}
```

## 🧩 Role in Verification (검증에서의 역할)

This output acts as the **"Actual"** side of the equation. The `verify_coverage.py` script compares this list against the C++ Golden Standard to:
*   **Verify Compliance**: Does `eigen-rs` implement what Eigen C++ offers?
*   **Identify Extensions**: What extra APIs (`as_slice`, `new`) does Rust have?
