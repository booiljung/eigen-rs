use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::DynamicStorage;
mod common;

#[test]
fn test_matrix_large_mul_differential() {
    let n = 100;
    
    // 1. Get C++ Oracle Result
    let cpp_results = common::run_cpp_harness("tests/cpp_harness/matrix_large_mul_verify.cpp")
        .expect("Failed to run C++ harness");

    // 2. Setup Rust Matrices (Same Deterministic Logic)
    let mut a = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(n, n).unwrap();
    let mut b = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(n, n).unwrap();
    
    for i in 0..n {
        for j in 0..n {
            *a.get_mut(i, j).unwrap() = (i + j) as f32;
            *b.get_mut(i, j).unwrap() = (i as isize - j as isize) as f32;
        }
    }

    // 3. Compute Rust Result
    let mut res = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(n, n).unwrap();
    res.assign(&(&a * &b)).unwrap();
    
    // 4. Compare
    for (r, c, v_cpp) in cpp_results {
        let v_rust = *res.get(r, c).expect("Rust matrix missing element");
        // Larger error tolerance for larger operations
        assert!(
            (v_rust - v_cpp).abs() < 1e-3, 
            "Large Mul mismatch at ({},{}): Rust={} != C++={}",
            r, c, v_rust, v_cpp
        );
    }
}
