use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::FixedStorage;
use eigen_rs::core::xpr::MatrixXpr;
mod common;

#[test]
fn test_matrix_qr_differential() {
    // 1. Get reference values from C++ Eigen
    let cpp_output = common::run_cpp_harness_stdout("tests/cpp_harness/matrix_qr_verify.cpp")
        .expect("Failed to run C++ harness");

    // 2. Setup Rust matrix (3x2)
    let mut m = Matrix::<f32, FixedStorage<f32, 3, 2, 6>>::new_fixed();
    *m.get_mut(0, 0).unwrap() = 1.0;
    *m.get_mut(0, 1).unwrap() = 2.0;
    *m.get_mut(1, 0).unwrap() = 3.0;
    *m.get_mut(1, 1).unwrap() = 4.0;
    *m.get_mut(2, 0).unwrap() = 5.0;
    *m.get_mut(2, 1).unwrap() = 6.0;

    // 3. Perform QR decomposition
    let qr = m.householder_qr().expect("QR decomposition failed");
    let q = qr.matrix_q();
    let r = qr.matrix_r();

    // 4. Compare
    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        match parts[0] {
            "Q" => {
                let row: usize = parts[1].parse().unwrap();
                let col: usize = parts[2].parse().unwrap();
                let v_cpp: f32 = parts[3].parse().unwrap();
                let v_rust = *q.get(row, col).unwrap();
                // Eigen's Q might have flipped signs for columns compared to standard Householder
                // but HouseholderQR should be consistent.
                assert!(
                    (v_rust.abs() - v_cpp.abs()).abs() < 1e-5,
                    "Q abs mismatch at {},{}: Rust={} != C++={}",
                    row,
                    col,
                    v_rust,
                    v_cpp
                );
                // Also check reconstructability if signs are flipped
            }
            "R" => {
                let row: usize = parts[1].parse().unwrap();
                let col: usize = parts[2].parse().unwrap();
                let v_cpp: f32 = parts[3].parse().unwrap();
                let v_rust = *r.get(row, col).unwrap();
                assert!(
                    (v_rust.abs() - v_cpp.abs()).abs() < 1e-5,
                    "R abs mismatch at {},{}: Rust={} != C++={}",
                    row,
                    col,
                    v_rust,
                    v_cpp
                );
            }
            _ => {}
        }
    }

    // 5. Reconstruction check: A = QR
    let mut qr_prod =
        Matrix::<f32, eigen_rs::core::storage::DynamicStorage<f32>>::new_dynamic(3, 2).unwrap();
    qr_prod.assign(&(&q.block(0, 0, 3, 3) * &r)).unwrap();

    for i in 0..3 {
        for j in 0..2 {
            let v_orig = *m.get(i, j).unwrap();
            let v_recon = *qr_prod.get(i, j).unwrap();
            assert!(
                (v_orig - v_recon).abs() < 1e-5,
                "Reconstruction failed at {},{}: Orig={} Recon={}",
                i,
                j,
                v_orig,
                v_recon
            );
        }
    }
}

#[test]
fn test_matrix_qr_orthogonality() {
    let mut m = Matrix::<f32, FixedStorage<f32, 4, 3, 12>>::new_fixed();
    for i in 0..4 {
        for j in 0..3 {
            *m.get_mut(i, j).unwrap() = (i + j * 2) as f32;
        }
    }

    let qr = m.householder_qr().unwrap();
    let q = qr.matrix_q();

    // Q^T * Q = I
    let qt = q.transpose();
    let res_expr = &qt * &q;

    for i in 0..4 {
        for j in 0..4 {
            let v = res_expr.eval(i, j);
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (v - expected).abs() < 1e-5,
                "Orthogonality failed at {},{}: v={}",
                i,
                j,
                v
            );
        }
    }
}
