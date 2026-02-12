# Verification & Testing

> **Current Status**: **100.0% Perfection Score**  
> **Last Verified**: 2026-02-10

This directory contains the "Verification Oracle" toolchain used to ensure `eigen-rs` maintains strict parity with the original C++ Eigen library.

## 📊 Verification Report

-   [**Coverage Report (Static)**](../tmp/verification/VERIFICATION_REPORT.md): Checks API existence.
-   [**Parity Report (Dynamic)**](../tmp/verification/PARITY_REPORT.md): Checks runtime numerical equality.

## 📘 methodology (Verification Oracle)

We typically do not rewrite logic; we replicate behavior. Our methodology involves extracting the C++ API surface and ensuring every Rust equivalent behaves exactly the same via Differential Testing.

-   [**Porting Strategy (English)**](PORTING_STRATEGY.md)
-   [**검증 전략 (Korean)**](PORTING_STRATEGY_KO.md)
-   [**Proof of Parity (Cross-Comparison Evidence)**](PROOF_OF_PARITY.md)

## 🛠️ Toolchain

The verification process consists of three stages:

1.  **Extract C++ API**: Parses original Eigen headers.
2.  **Extract Rust API**: Parses our Rust implementation.
3.  **Verify Coverage**: Matches APIs and checks for Differential Test usage.
4.  **Verify Parity**: Executes Differential Tests and confirms numerical parity.

### How to Run

To run the full verification suite (Coverage + Parity):

```bash
./run.sh
```

## 📂 Directory Structure

-   `run.sh`: Main entry point script.
-   `verify_coverage.py`: The judge logic that matches APIs.
-   `verify_parity.py`: The executioner that runs differential tests.
-   `extract_cpp_api.py`: C++ header parser.
-   `extract_rust_api.py`: Rust source parser.
