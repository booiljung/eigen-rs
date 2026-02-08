use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::FixedStorage;
use eigen_rs::core::xpr::MatrixXpr;
mod common;

#[test]
fn test_matrix_transpose_differential() {
    // 1. Get reference values from C++ Eigen
    let cpp_results = common::run_cpp_harness("tests/cpp_harness/matrix_transpose_verify.cpp")
        .expect("Failed to run C++ harness");

    // 2. Setup same matrix in Rust (2x3)
    let mut m = Matrix::<f32, FixedStorage<f32, 2, 3, 6>>::new_fixed();
    *m.get_mut(0, 0).unwrap() = 1.0; *m.get_mut(0, 1).unwrap() = 2.0; *m.get_mut(0, 2).unwrap() = 3.0;
    *m.get_mut(1, 0).unwrap() = 4.0; *m.get_mut(1, 1).unwrap() = 5.0; *m.get_mut(1, 2).unwrap() = 6.0;

    // 3. Perform transpose (Lazy)
    let m_t = m.transpose();
    assert_eq!(m_t.rows(), 3);
    assert_eq!(m_t.cols(), 2);

    // 4. Compare with C++ (which will be 3x2)
    for (r, c, v_cpp) in cpp_results {
        let v_rust = m_t.eval(r, c);
        assert!((v_rust - v_cpp).abs() < 1e-6, 
            "Transpose mismatch at ({},{}): Rust={} != C++={}", r, c, v_rust, v_cpp);
    }
}

#[test]
fn test_matrix_transpose_chaining() {
    // (A + B).transpose()
    let mut a = Matrix::<f32, FixedStorage<f32, 2, 2, 4>>::new_fixed();
    let mut b = Matrix::<f32, FixedStorage<f32, 2, 2, 4>>::new_fixed();

    *a.get_mut(0, 0).unwrap() = 1.0; *a.get_mut(0, 1).unwrap() = 2.0;
    *a.get_mut(1, 0).unwrap() = 3.0; *a.get_mut(1, 1).unwrap() = 4.0;

    *b.get_mut(0, 0).unwrap() = 10.0; *b.get_mut(0, 1).unwrap() = 20.0;
    *b.get_mut(1, 0).unwrap() = 30.0; *b.get_mut(1, 1).unwrap() = 40.0;

    let sum = &a + &b;
    let res_expr = sum.transpose();
    
    // Result should be [[11, 33], [22, 44]]
    assert_eq!(res_expr.eval(0, 0), 11.0);
    assert_eq!(res_expr.eval(0, 1), 33.0);
    assert_eq!(res_expr.eval(1, 0), 22.0);
    assert_eq!(res_expr.eval(1, 1), 44.0);
}
