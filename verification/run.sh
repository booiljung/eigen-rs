#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

cd "$ROOT_DIR"

if [ "$1" == "--meta" ]; then
    echo "Running Meta-Verification (Integrity Check)..."
    python3 verification/meta_verify.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Meta-Verification Passed!${NC}"
    else
        echo -e "${RED}❌ Meta-Verification Failed!${NC}"
        exit 1
    fi
else
    echo "Running Full Verification Toolchain..."
    
    echo "1. Extracting C++ API..."
    python3 verification/extract_cpp_api.py
    
    echo "2. Extracting Rust API..."
    python3 verification/extract_rust_api.py
    
    echo "3. Matching & verifying Coverage..."
    python3 verification/verify_coverage.py
    
    echo "4. Verifying Parity (Differential Testing)..."
    python3 verification/verify_parity.py

    echo -e "${GREEN}✅ Full Verification Suite Completed.${NC}"
fi
