use eigen_rs::Matrix2;
mod common;

#[test]
fn test_matrix2f_differential() {
    // 1. Get reference values from C++ Eigen
    let cpp_results = common::run_cpp_harness("tests/cpp_harness/matrix2f_verify.cpp")
        .expect("Failed to run C++ harness");

    // 2. Setup same matrix in Rust eigen-rs
    // Note: matrix2f_verify.cpp uses m << 1, 2, 3, 4; which is row-major input
    // but Eigen defaults to col-major storage.
    // m(0,0)=1, m(0,1)=2, m(1,0)=3, m(1,1)=4
    let mut m_rust = Matrix2::<f32>::new_fixed();
    *m_rust.get_mut(0, 0).unwrap() = 1.0;
    *m_rust.get_mut(0, 1).unwrap() = 2.0;
    *m_rust.get_mut(1, 0).unwrap() = 3.0;
    *m_rust.get_mut(1, 1).unwrap() = 4.0;

    // 3. Compare results
    for (r, c, v_cpp) in cpp_results {
        let v_rust = *m_rust.get(r, c).expect("Rust matrix missing element");
        assert!((v_rust - v_cpp).abs() < 1e-6, 
            "Value mismatch at ({},{}): Rust={} != C++={}", r, c, v_rust, v_cpp);
    }
}
