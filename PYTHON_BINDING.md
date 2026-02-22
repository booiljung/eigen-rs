# Python Bindings (PyO3) Implementation and Verification Plan

## 1. Goal
Create a `python-eigen-rs` wrapper crate that exposes the high-performance Rust port of Eigen (`eigen-rs`) to Python. This expands the ecosystem and allows Python users to leverage the SIMD/CUDA optimized matrix operations and decompositions.

## 2. Architecture
1. **New Crate**: Create a workspace member named `python-eigen-rs` within the `eigen-rs` project.
2. **PyO3 Integration**: Use `pyo3` to bridge Rust and Python, building a native python extension (`cdylib`).
3. **Rust Types Wrapper**: Create PyO3 wrapper structs (e.g., `PyMatrixF32`, `PyMatrixF64`) that wrap the native `MatrixX<f32>` and `MatrixX<f64>` types.
4. **Exposed Methods**:
   - Initialization (from NumPy arrays via `rust-numpy`).
   - Core Arithmetic (`+`, `-`, `*`).
   - Matrix Multiplication (`mat.matmul(other)`).
   - Decompositions (`.lu()`, `.qr()`, `.svd()`).
   - Accessors (convert back to NumPy arrays).

## 3. Verification Plan

### 3.1. Build and Setup Environment
- Use `maturin` (a build system for PyO3) to compile the Rust crate into a Python `.whl` package or install it directly into a virtual environment (`maturin develop`).
- Create an isolated Python virtual environment (`python -m venv .venv`).

### 3.2. Automated Python Tests (`pytest`)
- Write a Python test suite (e.g., `tests/test_eigen_py.py`) using `pytest`.
- The tests will rigorously verify the correctness of the exposed APIs by comparing the results against `numpy` or `scipy.linalg`.

**Example Test Strategy:**
```python
import numpy as np
import python_eigen_rs as e_rs

def test_matrix_multiplication():
    a_np = np.random.rand(10, 10).astype(np.float64)
    b_np = np.random.rand(10, 10).astype(np.float64)
    
    a_rs = e_rs.PyMatrixF64.from_numpy(a_np)
    b_rs = e_rs.PyMatrixF64.from_numpy(b_np)
    
    # Compute in Rust
    c_rs = a_rs.matmul(b_rs)
    
    # Compute in Numpy
    c_np = a_np @ b_np
    
    # Compare correctness
    np.testing.assert_allclose(c_rs.to_numpy(), c_np, rtol=1e-5, atol=1e-8)

def test_svd_decomposition():
    a_np = np.random.rand(5, 5).astype(np.float64)
    a_rs = e_rs.PyMatrixF64.from_numpy(a_np)
    
    # Rust SVD
    u_rs, s_rs, v_rs = a_rs.svd()
    
    # Numpy SVD
    u_np, s_np, vh_np = np.linalg.svd(a_np)
    
    # Check singular values
    np.testing.assert_allclose(s_rs.to_numpy(), s_np, rtol=1e-5, atol=1e-8)
```

### 3.3. Differential Benchmarking
- A performance comparison script (`verify_python_perf.py`) will be created to benchmark the `python-eigen-rs` calls directly against `numpy` using `timeit` or `pytest-benchmark`.

### 3.4. CI/CD Integration
- Update `.github/workflows/ci.yml` pipeline:
  1. Install `python` and `maturin`.
  2. Build the `python-eigen-rs` extension.
  3. Run `pytest` and confirm successful validation on all PRs.
