use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::FixedStorage;
mod common;

#[test]
fn test_matrix_lu_differential() {
    // 1. Get reference values from C++ Eigen
    let cpp_output = common::run_cpp_harness_stdout("tests/cpp_harness/matrix_lu_verify.cpp")
        .expect("Failed to run C++ harness");

    // 2. Setup Rust matrix (4x4 tridiagonal)
    let mut m = Matrix::<f32, FixedStorage<f32, 4, 4, 16>>::new_fixed();
    *m.get_mut(0, 0).unwrap() = 2.0; *m.get_mut(0, 1).unwrap() = -1.0;
    *m.get_mut(1, 0).unwrap() = -1.0; *m.get_mut(1, 1).unwrap() = 2.0; *m.get_mut(1, 2).unwrap() = -1.0;
    *m.get_mut(2, 1).unwrap() = -1.0; *m.get_mut(2, 2).unwrap() = 2.0; *m.get_mut(2, 3).unwrap() = -1.0;
    *m.get_mut(3, 2).unwrap() = -1.0; *m.get_mut(3, 3).unwrap() = 2.0;

    // 3. Perform LU decomposition
    let lu = m.partial_piv_lu().expect("LU decomposition failed");

    // 4. Compare
    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts[0] == "DET" {
            let v_cpp: f32 = parts[1].parse().unwrap();
            let v_rust = lu.determinant();
            assert!((v_rust - v_cpp).abs() < 1e-5, "Det mismatch: Rust={} != C++={}", v_rust, v_cpp);
        }
    }
}
