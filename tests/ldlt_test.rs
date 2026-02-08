use eigen_rs::core::matrix::MatrixX;
mod common;

#[test]
fn test_ldlt_reconstruction() {
    let mut m = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    // Symmetric positive definite matrix
    *m.get_mut(0, 0).unwrap() = 4.0; *m.get_mut(0, 1).unwrap() = 12.0; *m.get_mut(0, 2).unwrap() = -16.0;
    *m.get_mut(1, 0).unwrap() = 12.0; *m.get_mut(1, 1).unwrap() = 37.0; *m.get_mut(1, 2).unwrap() = -43.0;
    *m.get_mut(2, 0).unwrap() = -16.0; *m.get_mut(2, 1).unwrap() = -43.0; *m.get_mut(2, 2).unwrap() = 98.0;

    let ldlt = m.ldlt().unwrap();
    let l = ldlt.matrix_l();
    let d = ldlt.vector_d();

    // Verify L * D * L.T = M
    let rows = 3;
    let mut d_mat = MatrixX::<f32>::new_dynamic(rows, rows).unwrap();
    for i in 0..rows {
        *d_mat.get_mut(i, i).unwrap() = d[i];
    }
    
    let lt = l.transpose();
    let mut ld = MatrixX::<f32>::new_dynamic(rows, rows).unwrap();
    ld.assign_product(&(l * &d_mat)).unwrap();
    
    let mut m_reconstructed = MatrixX::<f32>::new_dynamic(rows, rows).unwrap();
    m_reconstructed.assign_product(&(&ld * &lt)).unwrap();

    // Manual reconstruction for verification: A = P^T * (L * D * L^T) * P
    let p = ldlt.permutation();
    let mut final_reconstructed = MatrixX::<f32>::new_dynamic(rows, rows).unwrap();
    // In Rust, P is stored such that mat_pivoted(i, j) = A(p[i], p[j])
    // So A(p[i], p[j]) = (L*D*L^T)(i, j)
    for i in 0..rows {
        for j in 0..rows {
            *final_reconstructed.get_mut(p[i], p[j]).unwrap() = *m_reconstructed.get(i, j).unwrap();
        }
    }

    for i in 0..rows {
        for j in 0..rows {
            assert!((final_reconstructed.get(i, j).unwrap() - m.get(i, j).unwrap()).abs() < 1e-4);
        }
    }

    // Differential Testing
    let cpp_output = common::run_cpp_harness_stdout("tests/cpp_harness/matrix_cholesky_verify.cpp").unwrap();
    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        match parts[0] {
            "LDLT_L" => {
                let row: usize = parts[1].parse().unwrap();
                let col: usize = parts[2].parse().unwrap();
                let val_cpp: f32 = parts[3].parse().unwrap();
                let val_rust = *l.get(row, col).unwrap();
                assert!((val_rust - val_cpp).abs() < 1e-5, "LDLT L mismatch at {},{}", row, col);
            },
            "LDLT_D" => {
                let idx: usize = parts[1].parse().unwrap();
                let val_cpp: f32 = parts[2].parse().unwrap();
                let val_rust = d[idx];
                assert!((val_rust - val_cpp).abs() < 1e-5, "LDLT D mismatch at {}", idx);
            },
            _ => {}
        }
    }
    // Differential Testing
    let cpp_output = common::run_cpp_harness_stdout("tests/cpp_harness/matrix_cholesky_verify.cpp").unwrap();
    // Eigen's P contains transpositions. We need to be careful.
    // Actually, for simplicity, let's just check if L*D*L^T = A and if D values match in some order.
    // BUT we want to match EXACTLY if possible.
    // Eigen's P for this matrix: [2, 1, 2]? No, P indices: 2, 1, 2.
    // Let's just check D values and reconstruction for now as permutations in Eigen LDLT
    // are stored as Transpositions.
    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        match parts[0] {
            "LDLT_D" => {
                let idx: usize = parts[1].parse().unwrap();
                let val_cpp: f32 = parts[2].parse().unwrap();
                let val_rust = d[idx];
                assert!((val_rust - val_cpp).abs() < 1e-4, "LDLT D mismatch at {}: Rust={} C++={}", idx, val_rust, val_cpp);
            },
            "LDLT_L" => {
                let r: usize = parts[1].parse().unwrap();
                let c: usize = parts[2].parse().unwrap();
                let val_cpp: f32 = parts[3].parse().unwrap();
                let val_rust = *l.get(r, c).unwrap();
                assert!((val_rust - val_cpp).abs() < 1e-4, "LDLT L mismatch at {},{}: Rust={} C++={}", r, c, val_rust, val_cpp);
            },
            _ => {}
        }
    }
}
