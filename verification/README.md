# eigen-rs Verification Toolchain

This directory contains the rule-based verification toolchain designed to prove the "Perfection" of the `eigen-rs` library by comparing it against the original C++ Eigen library.

## Directory Structure

- **`extract_cpp_api.py`**: Extracts public APIs from C++ Eigen headers.
- **`extract_rust_api.py`**: Extracts public APIs from `eigen-rs` Rust source.
- **`verify_coverage.py`**: Matches the extracted APIs 1:1, checks for Differential Tests, and generates the Verification Report.
- **`meta_verify.py`**: Integrity check script. Verifies that the extractors and matchers work correctly using synthetic dummy data.
- **`data/`**: REMOVED. Use `tmp/verification/` instead.

## Output

All generated files are stored in `tmp/verification/`:
- `tmp/verification/cpp_api_list.json`
- `tmp/verification/rust_api_list.json`
- `tmp/verification/VERIFICATION_REPORT.md`

## How to Run

### 1. Full Verification (Generates Report)
To run the full verification pipeline and generate `VERIFICATION_REPORT.md`:

```bash
./run.sh
```

Or manually:
```bash
python3 extract_cpp_api.py
python3 extract_rust_api.py
python3 verify_coverage.py
```

The report will be generated at `verification/VERIFICATION_REPORT.md`.

### 2. Integrity Check (Meta-Verification)
To verify that the tools themselves are working correctly:

```bash
./run.sh --meta
```

Or manually:
```bash
python3 meta_verify.py
```

## Methodology

1.  **Extraction**: We regex-scan C++ headers for `EIGEN_DEVICE_FUNC` public methods and Rust sources for `pub fn`.
2.  **Matching**: We map Rust method names to C++ equivalents (e.g., `add` -> `operator+`, `rows` -> `rows`).
3.  **Verification**: For every match, we check if a Differential Test exists (containing `run_cpp_harness` or specific test patterns).
4.  **Reporting**: We classify each API as:
    - `PASS`: Matched & Tested.
    - `WARN`: Rust-only extension (Functionally Verified).
    - `FAIL`: No coverage.
