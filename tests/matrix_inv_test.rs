use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::FixedStorage;
mod common;

#[test]
fn test_matrix_inverse_differential() {
    // 1. Get reference values from C++ Eigen
    let cpp_output = common::run_cpp_harness_stdout("tests/cpp_harness/matrix_inv_verify.cpp")
        .expect("Failed to run C++ harness");

    // 2. Setup Rust matrices
    let mut m2 = Matrix::<f32, FixedStorage<f32, 2, 2, 4>>::new_fixed();
    *m2.get_mut(0, 0).unwrap() = 1.0;
    *m2.get_mut(0, 1).unwrap() = 2.0;
    *m2.get_mut(1, 0).unwrap() = 3.0;
    *m2.get_mut(1, 1).unwrap() = 4.0;

    let mut m4 = Matrix::<f32, FixedStorage<f32, 4, 4, 16>>::new_fixed();
    *m4.get_mut(0, 0).unwrap() = 2.0;
    *m4.get_mut(0, 1).unwrap() = -1.0;
    *m4.get_mut(1, 0).unwrap() = -1.0;
    *m4.get_mut(1, 1).unwrap() = 2.0;
    *m4.get_mut(1, 2).unwrap() = -1.0;
    *m4.get_mut(2, 1).unwrap() = -1.0;
    *m4.get_mut(2, 2).unwrap() = 2.0;
    *m4.get_mut(2, 3).unwrap() = -1.0;
    *m4.get_mut(3, 2).unwrap() = -1.0;
    *m4.get_mut(3, 3).unwrap() = 2.0;

    // 3. Compute inverses
    let inv2 = m2.inverse().expect("Inverse of M2 failed");
    let inv4 = m4.inverse().expect("Inverse of M4 failed");

    // 4. Compare with C++
    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        match parts[0] {
            "INV2" => {
                let r: usize = parts[1].parse().unwrap();
                let c: usize = parts[2].parse().unwrap();
                let v_cpp: f32 = parts[3].parse().unwrap();
                let v_rust = *inv2.get(r, c).unwrap();
                assert!(
                    (v_rust - v_cpp).abs() < 1e-5,
                    "Inv2 mismatch at {},{}: Rust={} != C++={}",
                    r,
                    c,
                    v_rust,
                    v_cpp
                );
            }
            "INV4" => {
                let r: usize = parts[1].parse().unwrap();
                let c: usize = parts[2].parse().unwrap();
                let v_cpp: f32 = parts[3].parse().unwrap();
                let v_rust = *inv4.get(r, c).unwrap();
                assert!(
                    (v_rust - v_cpp).abs() < 1e-4,
                    "Inv4 mismatch at {},{}: Rust={} != C++={}",
                    r,
                    c,
                    v_rust,
                    v_cpp
                );
            }
            _ => {}
        }
    }
}

#[test]
fn test_matrix_inverse_identity() {
    let mut m = Matrix::<f32, FixedStorage<f32, 3, 3, 9>>::new_fixed();
    *m.get_mut(0, 0).unwrap() = 2.0;
    *m.get_mut(1, 1).unwrap() = 4.0;
    *m.get_mut(2, 2).unwrap() = 8.0;

    let inv = m.inverse().expect("Inverse failed");

    // Inverse of diag(2,4,8) is diag(0.5, 0.25, 0.125)
    assert_eq!(*inv.get(0, 0).unwrap(), 0.5);
    assert_eq!(*inv.get(1, 1).unwrap(), 0.25);
    assert_eq!(*inv.get(2, 2).unwrap(), 0.125);
    assert_eq!(*inv.get(0, 1).unwrap(), 0.0);
}
