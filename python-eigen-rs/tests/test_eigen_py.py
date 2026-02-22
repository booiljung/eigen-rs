import numpy as np
import pytest
import python_eigen_rs as e_rs

def test_initialization():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    mat = e_rs.PyMatrixF64.from_numpy(arr)
    
    arr_out = mat.to_numpy()
    np.testing.assert_array_equal(arr, arr_out)

def test_matmul():
    a_np = np.random.rand(10, 10).astype(np.float64)
    b_np = np.random.rand(10, 10).astype(np.float64)
    
    a_rs = e_rs.PyMatrixF64.from_numpy(a_np)
    b_rs = e_rs.PyMatrixF64.from_numpy(b_np)
    
    c_rs = a_rs.matmul(b_rs)
    c_np = a_np @ b_np
    
    np.testing.assert_allclose(c_rs.to_numpy(), c_np, rtol=1e-5, atol=1e-8)
