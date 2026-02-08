use eigen_rs::core::decompositions::{ComputationInfo, Tridiagonalization};
use eigen_rs::core::matrix::MatrixX;

#[test]
fn test_eigen_decomposition_3x3() {
    let mut m = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    // Symmetric matrix
    *m.get_mut(0, 0).unwrap() = 2.0;
    *m.get_mut(0, 1).unwrap() = 1.0;
    *m.get_mut(0, 2).unwrap() = 0.0;
    *m.get_mut(1, 0).unwrap() = 1.0;
    *m.get_mut(1, 1).unwrap() = 2.0;
    *m.get_mut(1, 2).unwrap() = 1.0;
    *m.get_mut(2, 0).unwrap() = 0.0;
    *m.get_mut(2, 1).unwrap() = 1.0;
    *m.get_mut(2, 2).unwrap() = 2.0;

    let solver = m.self_adjoint_eigen_solver(true).unwrap();
    if solver.info() != ComputationInfo::Success {
        panic!("Solver failed to converge: {:?}", solver.info());
    }

    let eivals = solver.eigenvalues();
    let eivecs = solver.eigenvectors().unwrap();

    // Verify reconstruction: A * V = V * D
    for i in 0..3 {
        let lambda = *eivals.get(i, 0).unwrap();
        // A * v_i
        let mut av = [0.0; 3];
        for r in 0..3 {
            for c in 0..3 {
                av[r] += (*m.get(r, c).unwrap()) * (*eivecs.get(c, i).unwrap());
            }
        }

        // lambda * v_i
        for r in 0..3 {
            let lv = lambda * (*eivecs.get(r, i).unwrap());
            assert!(
                (av[r] - lv).abs() < 1e-4,
                "Reconstruction failed at row {}, col {}: av={} lv={}",
                r,
                i,
                av[r],
                lv
            );
        }
    }

    // Verify orthogonality: V^T * V = I
    for i in 0..3 {
        for j in 0..3 {
            let mut dot = 0.0;
            for k in 0..3 {
                dot += (*eivecs.get(k, i).unwrap()) * (*eivecs.get(k, j).unwrap());
            }
            if i == j {
                assert!(
                    (dot - 1.0).abs() < 1e-4,
                    "Orthogonality failed at ({}, {}): dot={}",
                    i,
                    j,
                    dot
                );
            } else {
                assert!(
                    dot.abs() < 1e-4,
                    "Orthogonality failed at ({}, {}): dot={}",
                    i,
                    j,
                    dot
                );
            }
        }
    }
}

#[test]
fn test_eigenvalues_2x2() {
    let mut m = MatrixX::<f32>::new_dynamic(2, 2).unwrap();
    *m.get_mut(0, 0).unwrap() = 1.0;
    *m.get_mut(0, 1).unwrap() = 2.0;
    *m.get_mut(1, 0).unwrap() = 2.0;
    *m.get_mut(1, 1).unwrap() = 1.0;

    let solver = m.self_adjoint_eigen_solver(true).unwrap();
    assert_eq!(solver.info(), ComputationInfo::Success);

    let eivals = solver.eigenvalues();
    let val1 = *eivals.get(0, 0).unwrap();
    let val2 = *eivals.get(1, 0).unwrap();

    let (v1, v2) = if val1 < val2 {
        (val1, val2)
    } else {
        (val2, val1)
    };
    assert!((v1 - (-1.0)).abs() < 1e-4, "Expected -1.0, got {}", v1);
    assert!((v2 - 3.0).abs() < 1e-4, "Expected 3.0, got {}", v2);

    // Verify reconstruction
    let eivecs = solver.eigenvectors().unwrap();
    for i in 0..2 {
        let lambda = *eivals.get(i, 0).unwrap();
        let mut av = [0.0; 2];
        for r in 0..2 {
            for c in 0..2 {
                av[r] += (*m.get(r, c).unwrap()) * (*eivecs.get(c, i).unwrap());
            }
        }
        for r in 0..2 {
            let lv = lambda * (*eivecs.get(r, i).unwrap());
            assert!((av[r] - lv).abs() < 1e-4);
        }
    }
}

#[test]
fn test_tridiagonalization() {
    let mut m = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    *m.get_mut(0, 0).unwrap() = 4.0;
    *m.get_mut(0, 1).unwrap() = 1.0;
    *m.get_mut(0, 2).unwrap() = -2.0;
    *m.get_mut(1, 0).unwrap() = 1.0;
    *m.get_mut(1, 1).unwrap() = 3.0;
    *m.get_mut(1, 2).unwrap() = 0.0;
    *m.get_mut(2, 0).unwrap() = -2.0;
    *m.get_mut(2, 1).unwrap() = 0.0;
    *m.get_mut(2, 2).unwrap() = 5.0;

    let tri = Tridiagonalization::new(&m).unwrap();
    let t = tri.matrix_t();
    let q = tri.matrix_q();

    // Verify T is tridiagonal
    for i in 0..3 {
        for j in 0..3 {
            if (i as isize - j as isize).abs() > 1 {
                assert!(
                    t.get(i, j).unwrap().abs() < 1e-5,
                    "T is not tridiagonal at ({}, {}): val={}",
                    i,
                    j,
                    t.get(i, j).unwrap()
                );
            }
        }
    }

    // Verify reconstruction: A = Q * T * Q^T
    let mut q_eval = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    q_eval.assign(&q).unwrap();
    let mut t_eval = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    t_eval.assign(&t).unwrap();

    let qt_expr = &q_eval * &t_eval;
    let mut qt = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    qt.assign(&qt_expr).unwrap();

    let q_trans = q_eval.transpose();
    let res_expr = &qt * &q_trans;
    let mut res = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
    res.assign(&res_expr).unwrap();

    for i in 0..3 {
        for j in 0..3 {
            let val = *res.get(i, j).unwrap();
            let expected = *m.get(i, j).unwrap();
            assert!(
                (val - expected).abs() < 1e-4,
                "Reconstruction failed at ({}, {}): got {} expected {}",
                i,
                j,
                val,
                expected
            );
        }
    }
}
