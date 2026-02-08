use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::FixedStorage;
use eigen_rs::core::xpr::MatrixXpr;
mod common;

#[test]
fn test_matrix_determinant_differential() {
    // 1. Get reference values from C++ Eigen
    let cpp_output = common::run_cpp_harness_stdout("tests/cpp_harness/matrix_det_verify.cpp")
        .expect("Failed to run C++ harness");

    // 2. Setup Rust matrices
    // 2x2
    let mut m2 = Matrix::<f32, FixedStorage<f32, 2, 2, 4>>::new_fixed();
    *m2.get_mut(0, 0).unwrap() = 1.0; *m2.get_mut(0, 1).unwrap() = 2.0;
    *m2.get_mut(1, 0).unwrap() = 3.0; *m2.get_mut(1, 1).unwrap() = 4.0;

    // 3x3
    let mut m3 = Matrix::<f32, FixedStorage<f32, 3, 3, 9>>::new_fixed();
    *m3.get_mut(0, 0).unwrap() = 1.0; *m3.get_mut(0, 1).unwrap() = 2.0; *m3.get_mut(0, 2).unwrap() = 3.0;
    *m3.get_mut(1, 0).unwrap() = 0.0; *m3.get_mut(1, 1).unwrap() = 1.0; *m3.get_mut(1, 2).unwrap() = 4.0;
    *m3.get_mut(2, 0).unwrap() = 5.0; *m3.get_mut(2, 1).unwrap() = 6.0; *m3.get_mut(2, 2).unwrap() = 0.0;

    // 3. Compare with C++
    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 2 {
            let key = parts[0];
            let v_cpp: f32 = parts[1].parse().unwrap();
            
            match key {
                "DET2" => {
                    let v_rust = m2.determinant();
                    assert!((v_rust - v_cpp).abs() < 1e-5, "Det2 mismatch: Rust={} != C++={}", v_rust, v_cpp);
                },
                "DET3" => {
                    let v_rust = m3.determinant();
                    assert!((v_rust - v_cpp).abs() < 1e-5, "Det3 mismatch: Rust={} != C++={}", v_rust, v_cpp);
                },
                _ => {}
            }
        }
    }
}
