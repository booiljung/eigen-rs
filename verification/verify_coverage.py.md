# Verification Judge (`verify_coverage.py`)

This script is the **Third and Final Pillar** of the Verification Oracle toolchain. It acts as the "Judge", comparing the "Reality" (`rust_api_list.json`) against the "Golden Standard" (`cpp_api_list.json`).

## 🎯 Purpose (목적)

*   To match every Rust API to its C++ equivalent.
*   To check if the Rust API is covered by a **Differential Test** (proving parity) or a **Unit Test** (proving functionality).
*   To calculate the final **Perfection Score**.

## 🔑 Key Data Structures (핵심 데이터 구조)

### 1. `RUST_TO_CPP_MAP`
**"The Translation Dictionary"**

Rust and C++ often use different naming conventions or operator paradigms. This map bridges those gaps so `verify_coverage.py` doesn't falsely report a mismatch.

*   **Operator Mapping**: Rust uses traits (`add`), C++ uses operators (`operator+`).
    *   `'add': 'operator+'`
    *   `'index': 'operator[]'`
*   **Naming Convention**: Rust is `snake_case`, C++ is `camelCase`.
    *   `'is_approx': 'isApprox'`
    *   `'set_zero': 'setZero'`
*   **Logical Aliases**: Different names for the same concept.
    *   `'len': 'size'`
    *   `'inv': 'inverse'`

### 2. `RUST_EXTENSIONS`
**"The Whitelist"**

These are APIs that **do not existence** in C++ but are **required** in Rust. Without this whitelist, the judge would mark them as `FAIL` because it can't find a C++ match.

*   **Lifecycle**: `new`, `default`, `clone`, `to_owned`.
*   **Safety**: `as_ptr`, `as_slice` (C++ just uses pointers).
*   **Traits**: `fmt` (Debug/Display), `iter` (Iterator).
*   **Helpers**: `matrix_l`, `step` (Internal state accessors for testing).

**Verification Logic for Extensions:**
*   If in `RUST_EXTENSIONS` AND has `Unit Test` -> **PASS (Rust Ext)**
*   If in `RUST_EXTENSIONS` AND NO Test -> **FAIL** (Even extensions need tests!)

## 🚀 Usage (사용법)

```bash
python3 verification/verify_coverage.py
```

## 📄 Output (출력)

*   Generates `tmp/verification/VERIFICATION_REPORT.md`.
*   Prints the final score to stdout.
