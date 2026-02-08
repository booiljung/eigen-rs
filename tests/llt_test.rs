use eigen_rs::core::matrix::MatrixX;
mod common;

#[test]
fn test_llt_reconstruction() {
    let mut m = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    // Symmetric positive definite matrix
    // [ 4, 12, -16 ]
    // [ 12, 37, -43 ]
    // [ -16, -43, 98 ]
    *m.get_mut(0, 0).unwrap() = 4.0;
    *m.get_mut(0, 1).unwrap() = 12.0;
    *m.get_mut(0, 2).unwrap() = -16.0;
    *m.get_mut(1, 0).unwrap() = 12.0;
    *m.get_mut(1, 1).unwrap() = 37.0;
    *m.get_mut(1, 2).unwrap() = -43.0;
    *m.get_mut(2, 0).unwrap() = -16.0;
    *m.get_mut(2, 1).unwrap() = -43.0;
    *m.get_mut(2, 2).unwrap() = 98.0;

    let llt = m.llt().unwrap();
    let l = llt.matrix_l();

    // Verify L * L.T = M
    // Matrix Multiplication logic for MatrixX should work.
    let lt = l.transpose();
    let mut m_reconstructed = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    m_reconstructed.assign_product(&(l * &lt)).unwrap();

    for i in 0..3 {
        for j in 0..3 {
            assert!((m_reconstructed.get(i, j).unwrap() - m.get(i, j).unwrap()).abs() < 1e-4);
        }
    }

    // Differential Testing with C++ Eigen
    let cpp_output =
        common::run_cpp_harness_stdout("tests/cpp_harness/matrix_cholesky_verify.cpp").unwrap();
    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts[0] == "LLT_L" {
            let row: usize = parts[1].parse().unwrap();
            let col: usize = parts[2].parse().unwrap();
            let val_cpp: f32 = parts[3].parse().unwrap();
            let val_rust = *l.get(row, col).unwrap();
            assert!(
                (val_rust - val_cpp).abs() < 1e-5,
                "LLT L mismatch at {},{}",
                row,
                col
            );
        }
    }
}

#[test]
fn test_llt_solve() {
    let mut a = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    *a.get_mut(0, 0).unwrap() = 4.0;
    *a.get_mut(0, 1).unwrap() = 12.0;
    *a.get_mut(0, 2).unwrap() = -16.0;
    *a.get_mut(1, 0).unwrap() = 12.0;
    *a.get_mut(1, 1).unwrap() = 37.0;
    *a.get_mut(1, 2).unwrap() = -43.0;
    *a.get_mut(2, 0).unwrap() = -16.0;
    *a.get_mut(2, 1).unwrap() = -43.0;
    *a.get_mut(2, 2).unwrap() = 98.0;

    let mut b = MatrixX::<f32>::new_dynamic(3, 1).unwrap();
    *b.get_mut(0, 0).unwrap() = 1.0;
    *b.get_mut(1, 0).unwrap() = 2.0;
    *b.get_mut(2, 0).unwrap() = 3.0;

    let llt = a.llt().unwrap();
    let x = llt.solve(&b).unwrap();

    // Verify Ax = b
    let mut ax = MatrixX::<f32>::new_dynamic(3, 1).unwrap();
    ax.assign_product(&(&a * &x)).unwrap();

    for i in 0..3 {
        assert!((ax.get(i, 0).unwrap() - b.get(i, 0).unwrap()).abs() < 1e-4);
    }

    // Differential Testing
    let cpp_output =
        common::run_cpp_harness_stdout("tests/cpp_harness/matrix_cholesky_verify.cpp").unwrap();
    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts[0] == "LLT_SOLVE" {
            let idx: usize = parts[1].parse().unwrap();
            let val_cpp: f32 = parts[2].parse().unwrap();
            let val_rust = *x.get(idx, 0).unwrap();
            assert!(
                (val_rust - val_cpp).abs() < 1e-5,
                "LLT solve mismatch at {}",
                idx
            );
        }
    }
}
