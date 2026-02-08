use eigen_rs::Matrix2;
mod common;

#[test]
fn test_matrix_addition_differential() {
    // 1. Get reference values from C++ Eigen for (A + B)
    let cpp_results = common::run_cpp_harness("tests/cpp_harness/matrix_add_verify.cpp")
        .expect("Failed to run C++ harness");

    // 2. Setup same matrices in Rust
    let mut a = Matrix2::<f32>::new_fixed();
    let mut b = Matrix2::<f32>::new_fixed();

    // A = [1 2; 3 4], B = [10 20; 30 40]
    *a.get_mut(0, 0).unwrap() = 1.0; *a.get_mut(0, 1).unwrap() = 2.0;
    *a.get_mut(1, 0).unwrap() = 3.0; *a.get_mut(1, 1).unwrap() = 4.0;

    *b.get_mut(0, 0).unwrap() = 10.0; *b.get_mut(0, 1).unwrap() = 20.0;
    *b.get_mut(1, 0).unwrap() = 30.0; *b.get_mut(1, 1).unwrap() = 40.0;

    // 3. Perform addition (Lazy) and assign to result
    let mut res = Matrix2::<f32>::new_fixed();
    let add_expr = &a + &b;
    res.assign(&add_expr).expect("Assignment failed");

    // 4. Compare with C++
    for (r, c, v_cpp) in cpp_results {
        let v_rust = *res.get(r, c).expect("Rust matrix missing element");
        assert!((v_rust - v_cpp).abs() < 1e-6, 
            "Addition mismatch at ({},{}): Rust={} != C++={}", r, c, v_rust, v_cpp);
    }
}
