use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::FixedStorage;
mod common;

#[test]
fn test_matrix_mul_differential() {
    // 1. Get reference values from C++ Eigen
    let cpp_results = common::run_cpp_harness("tests/cpp_harness/matrix_mul_verify.cpp")
        .expect("Failed to run C++ harness");

    // 2. Setup same matrices in Rust
    // A: 2x3, B: 3x2, Res: 2x2
    let mut a = Matrix::<f32, FixedStorage<f32, 2, 3, 6>>::new_fixed();
    let mut b = Matrix::<f32, FixedStorage<f32, 3, 2, 6>>::new_fixed();

    *a.get_mut(0, 0).unwrap() = 1.0; *a.get_mut(0, 1).unwrap() = 2.0; *a.get_mut(0, 2).unwrap() = 3.0;
    *a.get_mut(1, 0).unwrap() = 4.0; *a.get_mut(1, 1).unwrap() = 5.0; *a.get_mut(1, 2).unwrap() = 6.0;

    *b.get_mut(0, 0).unwrap() = 7.0; *b.get_mut(0, 1).unwrap() = 8.0;
    *b.get_mut(1, 0).unwrap() = 9.0; *b.get_mut(1, 1).unwrap() = 10.0;
    *b.get_mut(2, 0).unwrap() = 11.0; *b.get_mut(2, 1).unwrap() = 12.0;

    // 3. Perform multiplication
    let mut res = eigen_rs::Matrix2::<f32>::new_fixed();
    res.assign(&(&a * &b)).unwrap();

    // 4. Compare
    for (r, c, v_cpp) in cpp_results {
        let v_rust = *res.get(r, c).expect("Rust matrix missing element");
        assert!((v_rust - v_cpp).abs() < 1e-5, 
            "Mul mismatch at ({},{}): Rust={} != C++={}", r, c, v_rust, v_cpp);
    }
}
