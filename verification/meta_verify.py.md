# Meta-Verifier (`meta_verify.py`)

This script is the **Guardian of the Watchmen**. It verifies that the verification tools themselves (`extract_cpp_api.py` and `extract_rust_api.py`) are working correctly.

## 🎯 Purpose (목적)

*   To prevent "False Positives" where the extractors might miss APIs due to parsing errors.
*   To ensure that changes to the regex logic in extractors do not break existing functionality.
*   To validate edge cases like multi-line function declarations or complex macro usage.

## ⚙️ How it Works (동작 원리)

1.  **Setup**: Creates a temporary directory `tmp/verification/test_data`.
2.  **Mock Generation**:
    *   Generates a `Dummy.h` (C++) with known patterns:
        *   Public/Private/Protected methods.
        *   Multi-line declarations.
        *   Operator overloads.
    *   Generates a `dummy.rs` (Rust) with known patterns:
        *   `impl` blocks.
        *   Public/Private functions.
3.  **Execution**: Runs the actual `extract_cpp_api` and `extract_rust_api` functions on these dummy files.
4.  **Assertion**: Checks if the extracted JSON data matches exactly what is expected (e.g., "Did it find `testMethod`?", "Did it ignore `privateMethod`?").

## 🚀 Usage (사용법)

Run this before trusting the main verification report, especially after modifying extractor logic.

```bash
python3 verification/meta_verify.py
```

**Output:**
```text
Running C++ Extractor on Dummy.h...
Running Rust Extractor on dummy.rs...
✅ Meta-Verification PASSED: Tools are accurately extracting APIs.
```
