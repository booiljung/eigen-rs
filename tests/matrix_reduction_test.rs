use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::FixedStorage;
use eigen_rs::core::xpr::MatrixXpr;
mod common;

#[test]
fn test_matrix_reductions_differential() {
    // 1. Get reference values from C++ Eigen
    let cpp_output =
        common::run_cpp_harness_stdout("tests/cpp_harness/matrix_reduction_verify.cpp")
            .expect("Failed to run C++ harness");

    // 2. Setup same matrix in Rust
    let mut m = Matrix::<f32, FixedStorage<f32, 2, 2, 4>>::new_fixed();
    *m.get_mut(0, 0).unwrap() = 1.5;
    *m.get_mut(0, 1).unwrap() = 2.5;
    *m.get_mut(1, 0).unwrap() = 3.5;
    *m.get_mut(1, 1).unwrap() = 4.5;

    // 3. Compare with C++
    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 2 {
            let key = parts[0];
            let v_cpp: f32 = parts[1].parse().unwrap();

            match key {
                "SUM" => {
                    let v_rust = m.sum();
                    assert!(
                        (v_rust - v_cpp).abs() < 1e-5,
                        "Sum mismatch: Rust={} != C++={}",
                        v_rust,
                        v_cpp
                    );
                }
                "MIN" => {
                    let v_rust = m.min();
                    assert_eq!(
                        v_rust, v_cpp,
                        "Min mismatch: Rust={} != C++={}",
                        v_rust, v_cpp
                    );
                }
                "MAX" => {
                    let v_rust = m.max();
                    assert_eq!(
                        v_rust, v_cpp,
                        "Max mismatch: Rust={} != C++={}",
                        v_rust, v_cpp
                    );
                }
                "MEAN" => {
                    let v_rust = m.mean();
                    assert!(
                        (v_rust - v_cpp).abs() < 1e-5,
                        "Mean mismatch: Rust={} != C++={}",
                        v_rust,
                        v_cpp
                    );
                }
                _ => {}
            }
        }
    }
}
