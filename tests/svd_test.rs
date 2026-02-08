use eigen_rs::core::matrix::MatrixX;
mod common;

#[test]
fn test_svd_reconstruction() {
    let mut m = MatrixX::<f32>::new_dynamic(3, 2).unwrap();
    *m.get_mut(0, 0).unwrap() = 1.0; *m.get_mut(0, 1).unwrap() = 2.0;
    *m.get_mut(1, 0).unwrap() = 3.0; *m.get_mut(1, 1).unwrap() = 4.0;
    *m.get_mut(2, 0).unwrap() = 5.0; *m.get_mut(2, 1).unwrap() = 6.0;

    let svd = m.jacobi_svd().unwrap();
    let u = svd.matrix_u();
    let v = svd.matrix_v();
    let s = svd.singular_values();

    // Verify U * S * V.T = M
    let mut sigma = MatrixX::<f32>::new_dynamic(3, 2).unwrap();
    for i in 0..s.len() {
        if i < 3 && i < 2 {
            *sigma.get_mut(i, i).unwrap() = s[i];
        }
    }

    let vt = v.transpose();
    let mut us = MatrixX::<f32>::new_dynamic(3, 2).unwrap();
    us.assign_product(&(u * &sigma)).unwrap();

    let mut m_reconstructed = MatrixX::<f32>::new_dynamic(3, 2).unwrap();
    m_reconstructed.assign_product(&(&us * &vt)).unwrap();

    for i in 0..3 {
        for j in 0..2 {
            assert!((m_reconstructed.get(i, j).unwrap() - m.get(i, j).unwrap()).abs() < 1e-4);
        }
    }

    // Differential Testing
    let cpp_output = common::run_cpp_harness_stdout("tests/cpp_harness/matrix_svd_verify.cpp").unwrap();
    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        match parts[0] {
            "S" => {
                let idx: usize = parts[1].parse().unwrap();
                let val_cpp: f32 = parts[2].parse().unwrap();
                assert!((s[idx] - val_cpp).abs() < 1e-5, "SVD singular value mismatch at {}", idx);
            },
            "U" => {
                let r: usize = parts[1].parse().unwrap();
                let c: usize = parts[2].parse().unwrap();
                let val_cpp: f32 = parts[3].parse().unwrap();
                // Only compare first 2 columns for 3x2 matrix
                if c < 2 {
                    let val_rust = *u.get(r, c).unwrap();
                    assert!((val_rust.abs() - val_cpp.abs()).abs() < 1e-4, "SVD U mismatch at {},{}", r, c);
                }
            },
            "V" => {
                let r: usize = parts[1].parse().unwrap();
                let c: usize = parts[2].parse().unwrap();
                let val_cpp: f32 = parts[3].parse().unwrap();
                if c < 2 {
                    let val_rust = *v.get(r, c).unwrap();
                    assert!((val_rust.abs() - val_cpp.abs()).abs() < 1e-4, "SVD V mismatch at {},{}", r, c);
                }
            },
            _ => {}
        }
    }
}

#[test]
fn test_svd_orthogonality() {
    let mut m = MatrixX::<f32>::new_dynamic(3, 2).unwrap();
    *m.get_mut(0, 0).unwrap() = 1.0; *m.get_mut(0, 1).unwrap() = 2.0;
    *m.get_mut(1, 0).unwrap() = 3.0; *m.get_mut(1, 1).unwrap() = 4.0;
    *m.get_mut(2, 0).unwrap() = 5.0; *m.get_mut(2, 1).unwrap() = 6.0;

    let svd = m.jacobi_svd().unwrap();
    let u = svd.matrix_u();
    let v = svd.matrix_v();

    // Check first n columns of U are orthogonal
    let n = 2;
    let ut = u.transpose();
    let mut utu = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    utu.assign_product(&(&ut * u)).unwrap();
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((utu.get(i, j).unwrap() - expected).abs() < 1e-4);
        }
    }

    // V.T * V = I
    let vt = v.transpose();
    let mut vtv = MatrixX::<f32>::new_dynamic(2, 2).unwrap();
    vtv.assign_product(&(&vt * v)).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((vtv.get(i, j).unwrap() - expected).abs() < 1e-4);
        }
    }
}

#[test]
fn test_bdcsvd_reconstruction() {
    let mut m = MatrixX::<f32>::new_dynamic(3, 2).unwrap();
    *m.get_mut(0, 0).unwrap() = 1.0; *m.get_mut(0, 1).unwrap() = 2.0;
    *m.get_mut(1, 0).unwrap() = 3.0; *m.get_mut(1, 1).unwrap() = 4.0;
    *m.get_mut(2, 0).unwrap() = 5.0; *m.get_mut(2, 1).unwrap() = 6.0;

    use eigen_rs::core::decompositions::BDCSVD;
    let svd = BDCSVD::new(&m).unwrap();
    let u = svd.matrix_u();
    let v = svd.matrix_v();
    let s = svd.singular_values();

    // Verify U * S * V.T = M
    let mut sigma = MatrixX::<f32>::new_dynamic(3, 2).unwrap();
    for i in 0..s.len() {
        if i < 3 && i < 2 {
            *sigma.get_mut(i, i).unwrap() = s[i];
        }
    }

    let vt = v.transpose();
    let mut us = MatrixX::<f32>::new_dynamic(3, 2).unwrap();
    us.assign_product(&(u * &sigma)).unwrap();

    let mut m_reconstructed = MatrixX::<f32>::new_dynamic(3, 2).unwrap();
    m_reconstructed.assign_product(&(&us * &vt)).unwrap();

    for i in 0..3 {
        for j in 0..2 {
            assert!((m_reconstructed.get(i, j).unwrap() - m.get(i, j).unwrap()).abs() < 1e-4, 
                "Mismatch at {},{}", i, j);
        }
    }
}

#[test]
fn test_bdcsvd_orthogonality() {
    let mut m = MatrixX::<f32>::new_dynamic(4, 3).unwrap();
    // Fill with random-ish data
    for i in 0..4 {
        for j in 0..3 {
            *m.get_mut(i, j).unwrap() = (i + j) as f32 + 1.0;
        }
    }

    use eigen_rs::core::decompositions::BDCSVD;
    let svd = BDCSVD::new(&m).unwrap();
    let u = svd.matrix_u();
    let v = svd.matrix_v();

    // Check U is orthogonal (U^T U = I)
    let ut = u.transpose();
    let mut utu = MatrixX::<f32>::new_dynamic(4, 4).unwrap();
    utu.assign_product(&(&ut * u)).unwrap();
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((utu.get(i, j).unwrap() - expected).abs() < 1e-4);
        }
    }

    // Check V is orthogonal
    let vt = v.transpose();
    let mut vtv = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    vtv.assign_product(&(&vt * v)).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((vtv.get(i, j).unwrap() - expected).abs() < 1e-4);
        }
    }
}
