#!/bin/bash
set -e

echo "🧪 Running Proof of Parity: Differential Testing..."
echo "==================================================="

# 1. Matrix Multiplication
echo ""
echo "👉 Proving Matrix Multiplication (Result Parity)..."
cargo test --test matrix_mul_test -- --nocapture

# 2. Cholesky Decomposition
echo ""
echo "👉 Proving Cholesky Decomposition (LLT/LDLT Parity)..."
cargo test --test llt_test -- --nocapture

# 3. SVD Decomposition
echo ""
echo "👉 Proving SVD Decomposition (Singular Values Parity)..."
cargo test --test svd_test -- --nocapture

echo ""
echo "✅ PROOF COMPLETE: Run 'cargo test' to execute the full suite."
