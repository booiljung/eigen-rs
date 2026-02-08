use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::DynamicStorage;
mod common;

#[test]
fn test_matrix_eigen_differential() {
    // 1. Get reference values from C++ Eigen
    let cpp_output = common::run_cpp_harness_stdout("tests/cpp_harness/matrix_eigen_verify.cpp")
        .expect("Failed to run C++ harness");

    // 2. Setup Rust matrix (4x4)
    let mut m = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(4, 4).unwrap();
    let data = [
        10.0, 1.0, 2.0, 3.0, 1.0, 20.0, 4.0, 5.0, 2.0, 4.0, 30.0, 6.0, 3.0, 5.0, 6.0, 40.0,
    ];
    for i in 0..4 {
        for j in 0..4 {
            *m.get_mut(i, j).unwrap() = data[i * 4 + j];
        }
    }

    // 3. Compute in Rust
    let solver = m
        .self_adjoint_eigen_solver(true)
        .expect("Rust solver failed");
    let rust_vals = solver.eigenvalues();
    let rust_vecs = solver.eigenvectors().unwrap();

    // 4. Parse C++ output and compare
    let mut cpp_vals = [0.0f32; 4];
    let mut cpp_vecs = vec![vec![0.0f32; 4]; 4];

    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        match parts[0] {
            "VAL" => {
                let idx: usize = parts[1].parse().unwrap();
                let val: f32 = parts[2].parse().unwrap();
                cpp_vals[idx] = val;
            }
            "VEC" => {
                let row: usize = parts[1].parse().unwrap();
                let col: usize = parts[2].parse().unwrap();
                let val: f32 = parts[3].parse().unwrap();
                cpp_vecs[row][col] = val;
            }
            _ => {}
        }
    }

    // 5. Compare Eigenvalues
    for i in 0..4 {
        let rv = *rust_vals.get(i, 0).unwrap();
        let cv = cpp_vals[i];
        assert!(
            (rv - cv).abs() < 1e-4,
            "Eigenvalue mismatch at {}: Rust={} C++={}",
            i,
            rv,
            cv
        );
    }

    // 6. Compare Eigenvectors (Careful with sign flip!)
    for j in 0..4 {
        // Rust's j-th column vs C++'s j-th column
        // Check if v_rust ≈ v_cpp OR v_rust ≈ -v_cpp
        let mut diff_pos = 0.0f32;
        let mut diff_neg = 0.0f32;
        for i in 0..4 {
            let rv = *rust_vecs.get(i, j).unwrap();
            let cv = cpp_vecs[i][j];
            diff_pos += (rv - cv).abs();
            diff_neg += (rv + cv).abs();
        }

        // Sum of absolute differences should be small for one of the cases
        assert!(
            diff_pos < 1e-4 || diff_neg < 1e-4,
            "Eigenvector mismatch at column {}: diff_pos={}, diff_neg={}",
            j,
            diff_pos,
            diff_neg
        );
    }
}
#[test]
fn test_complex_schur_differential() {
    use eigen_rs::core::complex::Complex;
    use eigen_rs::core::decompositions::ComplexSchur;

    let cpp_output = common::run_cpp_harness_stdout("tests/cpp_harness/complex_schur_verify.cpp")
        .expect("Failed to run C++ harness");

    let n = 4;
    let mut m = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(n, n).unwrap();
    // Same values as in C++
    let data = [
        Complex::new(0.35, 0.45),
        Complex::new(0.45, -0.14),
        Complex::new(-0.14, 0.25),
        Complex::new(-0.17, 0.11),
        Complex::new(0.09, 0.07),
        Complex::new(0.07, 0.35),
        Complex::new(-0.54, -0.13),
        Complex::new(0.35, 0.17),
        Complex::new(-0.44, -0.33),
        Complex::new(-0.33, 0.11),
        Complex::new(-0.03, 0.17),
        Complex::new(0.17, 0.09),
        Complex::new(0.25, -0.32),
        Complex::new(-0.32, 0.09),
        Complex::new(-0.13, 0.07),
        Complex::new(0.11, 0.11),
    ];
    for i in 0..n {
        for j in 0..n {
            *m.get_mut(i, j).unwrap() = data[i * n + j];
        }
    }

    let schur = ComplexSchur::new(&m).expect("Rust Schur failed");
    let rust_t = schur.matrix_t();
    let _rust_u = schur.matrix_u();

    let mut cpp_vals = Vec::new();
    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts[0] == "T" {
            let r: usize = parts[1].parse().unwrap();
            let c: usize = parts[2].parse().unwrap();
            if r == c {
                let re: f64 = parts[3].parse().unwrap();
                let im: f64 = parts[4].parse().unwrap();
                cpp_vals.push(Complex::new(re, im));
            }
        }
    }

    let mut rv_vals: Vec<Complex<f64>> = (0..n).map(|i| *rust_t.get(i, i).unwrap()).collect();
    rv_vals.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
    cpp_vals.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());

    for i in 0..n {
        let rv = rv_vals[i];
        let cv = cpp_vals[i];
        assert!(
            (rv.re - cv.re).abs() < 1e-8,
            "Eigenvalue mismatch at {}: Rust={} C++={}",
            i,
            rv,
            cv
        );
        assert!(
            (rv.im - cv.im).abs() < 1e-8,
            "Eigenvalue mismatch at {}: Rust={} C++={}",
            i,
            rv,
            cv
        );
    }
}

#[test]
fn test_generalized_eigen_differential() {
    use eigen_rs::core::complex::Complex;
    use eigen_rs::core::decompositions::GeneralizedEigenSolver;

    let cpp_output =
        common::run_cpp_harness_stdout("tests/cpp_harness/matrix_generalized_eigen_verify.cpp")
            .expect("Failed to run C++ harness");

    let n = 4;
    let mut a = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(n, n).unwrap();
    let mut b = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(n, n).unwrap();

    let data_a = [
        Complex::new(1.0, 1.0),
        Complex::new(2.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(0.5, 0.5),
        Complex::new(0.5, 0.5),
        Complex::new(3.0, 2.0),
        Complex::new(1.0, -1.0),
        Complex::new(0.1, 0.2),
        Complex::new(1.0, 0.0),
        Complex::new(1.0, 1.0),
        Complex::new(2.0, 2.0),
        Complex::new(0.3, 0.4),
        Complex::new(0.2, 0.3),
        Complex::new(0.4, 0.5),
        Complex::new(0.6, 0.1),
        Complex::new(1.5, 1.2),
    ];
    let data_b = [
        Complex::new(5.0, 0.0),
        Complex::new(1.0, 1.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.1, 0.1),
        Complex::new(1.0, -1.0),
        Complex::new(4.0, 2.0),
        Complex::new(2.0, 1.0),
        Complex::new(0.2, 0.3),
        Complex::new(3.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(5.0, -1.0),
        Complex::new(0.5, 0.5),
        Complex::new(0.1, 0.2),
        Complex::new(0.2, 0.1),
        Complex::new(0.3, 0.4),
        Complex::new(2.0, 1.0),
    ];

    for i in 0..n {
        for j in 0..n {
            *a.get_mut(i, j).unwrap() = data_a[i * n + j];
            *b.get_mut(i, j).unwrap() = data_b[i * n + j];
        }
    }

    let solver = GeneralizedEigenSolver::new(&a, &b, false).expect("Rust solver failed");
    let rust_vals = solver.eigenvalues();

    let mut cpp_vals = vec![Complex::new(0.0, 0.0); n];

    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts[0] == "VAL" {
            let idx: usize = parts[1].parse().unwrap();
            let re: f64 = parts[2].parse().unwrap();
            let im: f64 = parts[3].parse().unwrap();
            cpp_vals[idx] = Complex::new(re, im);
        }
    }

    // Sort to compare
    let mut rv_sorted: Vec<Complex<f64>> = (0..n).map(|i| *rust_vals.get(i, 0).unwrap()).collect();
    rv_sorted.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
    cpp_vals.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());

    for i in 0..n {
        let rv = rv_sorted[i];
        let cv = cpp_vals[i];
        assert!(
            (rv.re - cv.re).abs() < 1e-8,
            "Eigenvalue real mismatch at {}: Rust={} C++={}",
            i,
            rv,
            cv
        );
        assert!(
            (rv.im - cv.im).abs() < 1e-8,
            "Eigenvalue imag mismatch at {}: Rust={} C++={}",
            i,
            rv,
            cv
        );
    }
}

#[test]
fn test_sparse_ops_differential() {
    use eigen_rs::core::sparse::{SparseMatrix, StorageOrder, Triplet};

    let cpp_output = common::run_cpp_harness_stdout("tests/cpp_harness/sparse_ops_verify.cpp")
        .expect("Failed to run C++ harness");

    let mut a = SparseMatrix::<f64>::new(2, 3, StorageOrder::RowMajor);
    a.set_from_triplets(vec![
        Triplet::new(0, 0, 1.0),
        Triplet::new(1, 1, 2.0),
        Triplet::new(0, 2, 3.0),
    ]);

    let mut b = SparseMatrix::<f64>::new(2, 3, StorageOrder::RowMajor);
    b.set_from_triplets(vec![
        Triplet::new(0, 0, 10.0),
        Triplet::new(0, 1, 5.0),
        Triplet::new(1, 1, 1.0),
    ]);

    let rust_add = (&a + &b).unwrap();
    let rust_sub = (&a - &b).unwrap();
    let rust_scale = &a * 2.5;
    let rust_transpose = a.transpose();

    let mut b2 = SparseMatrix::<f64>::new(3, 2, StorageOrder::RowMajor);
    b2.set_from_triplets(vec![
        Triplet::new(0, 0, 10.0),
        Triplet::new(0, 1, 5.0),
        Triplet::new(2, 0, 1.0),
    ]);
    let rust_mul = (&a * &b2).unwrap();

    // Helper to get value from sparse matrix (for testing only, inefficient)
    let get_val = |m: &SparseMatrix<f64>, r: usize, c: usize| {
        use eigen_rs::core::sparse::InnerIterator;
        let outer = if m.order() == StorageOrder::RowMajor {
            r
        } else {
            c
        };
        let mut it = InnerIterator::new(m, outer);
        while it.is_valid() {
            if it.row() == r && it.col() == c {
                return it.value();
            }
            it.next();
        }
        0.0
    };

    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 4 || !parts[0].ends_with("_VAL") {
            continue;
        }

        let prefix = parts[0].split('_').next().unwrap();
        let r: usize = parts[1].parse().unwrap();
        let c: usize = parts[2].parse().unwrap();
        let cv: f64 = parts[3].parse().unwrap();

        let rv = match prefix {
            "ADD" => get_val(&rust_add, r, c),
            "SUB" => get_val(&rust_sub, r, c),
            "SCALE" => get_val(&rust_scale, r, c),
            "TRANSPOSE" => get_val(&rust_transpose, r, c),
            "MUL" => get_val(&rust_mul, r, c),
            _ => panic!("Unknown prefix"),
        };

        assert!(
            (rv - cv).abs() < 1e-10,
            "{} mismatch at ({}, {}): Rust={} C++={}",
            prefix,
            r,
            c,
            rv,
            cv
        );
    }
}

#[test]
fn test_sparse_llt_differential() {
    use eigen_rs::core::matrix::Matrix;
    use eigen_rs::core::sparse::solvers::SimplicialLLT;
    use eigen_rs::core::sparse::{InnerIterator, SparseMatrix, StorageOrder, Triplet};
    use eigen_rs::core::storage::DynamicStorage;

    let cpp_output = common::run_cpp_harness_stdout("tests/cpp_harness/sparse_llt_verify.cpp")
        .expect("Failed to run C++ harness");

    let mut expected_l_triplets = Vec::new();
    let mut expected_x = Vec::new();
    let mut expected_b = Vec::new();
    let mut n = 0;

    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        match parts[0] {
            "L_VAL" => {
                let r = parts[1].parse::<usize>().unwrap();
                let c = parts[2].parse::<usize>().unwrap();
                let v = parts[3].parse::<f64>().unwrap();
                expected_l_triplets.push(Triplet::new(r, c, v));
                if r + 1 > n {
                    n = r + 1;
                }
                if c + 1 > n {
                    n = c + 1;
                }
            }
            "B_VAL" => {
                let v = parts[2].parse::<f64>().unwrap();
                expected_b.push(v);
            }
            "X_VAL" => {
                let v = parts[2].parse::<f64>().unwrap();
                expected_x.push(v);
            }
            _ => {}
        }
    }

    // Construct matrix A on Rust side (same as C++ side: A = M*Mt + I)
    let mut m = SparseMatrix::<f64>::new(n, n, StorageOrder::ColMajor);
    let mut triplets = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if (i + j) % 3 == 0 {
                triplets.push(Triplet::new(i, j, (i + j + 1) as f64 / 10.0));
            }
        }
    }
    m.set_from_triplets(triplets);

    let mt = m.transpose_reordered();
    let res_m_mt = (&m * &mt).unwrap();

    // Add I to M*Mt
    let mut a_triplets = Vec::new();
    for j in 0..n {
        let mut found_diag = false;
        let mut it = InnerIterator::new(&res_m_mt, j);
        while it.is_valid() {
            let mut val = it.value();
            if it.row() == j {
                val += 1.0;
                found_diag = true;
            }
            a_triplets.push(Triplet::new(it.row(), j, val));
            it.next();
        }
        if !found_diag {
            a_triplets.push(Triplet::new(j, j, 1.0));
        }
    }

    let mut final_a = SparseMatrix::<f64>::new(n, n, StorageOrder::ColMajor);
    final_a.set_from_triplets(a_triplets);

    let mut llt = SimplicialLLT::new();
    llt.compute(&final_a).expect("Rust factorization failed");

    let rust_l = llt.matrix_l();
    // Verify L
    let mut expected_l = SparseMatrix::<f64>::new(n, n, StorageOrder::ColMajor);
    expected_l.set_from_triplets(expected_l_triplets);

    for j in 0..n {
        let mut it_rust = InnerIterator::new(rust_l, j);
        let mut it_expected = InnerIterator::new(&expected_l, j);
        while it_rust.is_valid() && it_expected.is_valid() {
            assert_eq!(it_rust.row(), it_expected.row());
            assert!(
                (it_rust.value() - it_expected.value()).abs() < 1e-10,
                "L mismatch at ({}, {}): Rust={} C++={}",
                it_rust.row(),
                j,
                it_rust.value(),
                it_expected.value()
            );
            it_rust.next();
            it_expected.next();
        }
        assert!(
            !it_rust.is_valid() && !it_expected.is_valid(),
            "L structure mismatch at column {}",
            j
        );
    }

    // Verify Solve
    let b = Matrix::<f64, DynamicStorage<f64>>::from_vec(n, 1, expected_b).unwrap();
    let x = llt.solve(&b).expect("Rust solve failed");

    for i in 0..n {
        let rv = *x.get(i, 0).unwrap();
        let cv = expected_x[i];
        assert!(
            (rv - cv).abs() < 1e-10,
            "Solution mismatch at index {}: Rust={} C++={}",
            i,
            rv,
            cv
        );
    }
}

#[test]
fn test_sparse_lu_differential() {
    use eigen_rs::core::matrix::Matrix;
    use eigen_rs::core::sparse::solvers::SparseLU;
    use eigen_rs::core::sparse::{SparseMatrix, StorageOrder, Triplet};
    use eigen_rs::core::storage::DynamicStorage;

    let cpp_output = common::run_cpp_harness_stdout("tests/cpp_harness/sparse_lu_verify.cpp")
        .expect("Failed to run C++ harness");

    let mut expected_x = Vec::new();
    let mut expected_b = Vec::new();
    let mut n = 0;

    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        match parts[0] {
            "B_VAL" => {
                let v = parts[2].parse::<f64>().unwrap();
                expected_b.push(v);
            }
            "X_VAL" => {
                let v = parts[2].parse::<f64>().unwrap();
                expected_x.push(v);
                n += 1;
            }
            _ => {}
        }
    }

    // Construct matrix A on Rust side (same as C++ side)
    let mut a = SparseMatrix::<f64>::new(n, n, StorageOrder::ColMajor);
    let mut triplets = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let val = ((i * 7 + j * 3) % 11) as f64 - 5.0;
            if val != 0.0 && (i + j) % 2 == 0 {
                triplets.push(Triplet::new(i, j, val));
            }
        }
        triplets.push(Triplet::new(i, i, 10.0));
    }
    a.set_from_triplets(triplets);

    let mut lu = SparseLU::new();
    lu.compute(&a).expect("Rust SparseLU factorization failed");

    let b = Matrix::<f64, DynamicStorage<f64>>::from_vec(n, 1, expected_b).unwrap();
    let x = lu.solve(&b).expect("Rust SparseLU solve failed");

    for i in 0..n {
        let rv = *x.get(i, 0).unwrap();
        let cv = expected_x[i];
        assert!(
            (rv - cv).abs() < 1e-10,
            "SparseLU Solution mismatch at index index {}: Rust={} C++={}",
            i,
            rv,
            cv
        );
    }
}

#[test]
fn test_sparse_iterative_differential() {
    use eigen_rs::core::matrix::Matrix;
    use eigen_rs::core::sparse::solvers::{BiCGSTAB, ConjugateGradient, DiagonalPreconditioner};
    use eigen_rs::core::sparse::{SparseMatrix, StorageOrder, Triplet};
    use eigen_rs::core::storage::DynamicStorage;

    let cpp_output =
        common::run_cpp_harness_stdout("tests/cpp_harness/sparse_iterative_verify.cpp")
            .expect("Failed to run C++ harness");

    let mut cg_b = Vec::new();
    let mut cg_x = Vec::new();
    let mut bicg_b = Vec::new();
    let mut bicg_x = Vec::new();

    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        match parts[0] {
            "CG_B_VAL" => cg_b.push(parts[2].parse::<f64>().unwrap()),
            "CG_X_VAL" => cg_x.push(parts[2].parse::<f64>().unwrap()),
            "BICG_B_VAL" => bicg_b.push(parts[2].parse::<f64>().unwrap()),
            "BICG_X_VAL" => bicg_x.push(parts[2].parse::<f64>().unwrap()),
            _ => {}
        }
    }

    let n = cg_b.len();

    // 1. CG Test
    let mut a_spd = SparseMatrix::<f64>::new(n, n, StorageOrder::ColMajor);
    let mut triplets_spd = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                triplets_spd.push(Triplet::new(i, j, 20.0 + (i % 5) as f64));
            } else if (i as isize - j as isize).abs() == 1 {
                triplets_spd.push(Triplet::new(i, j, -1.0));
            }
        }
    }
    a_spd.set_from_triplets(triplets_spd);

    let b_cg_mat = Matrix::<f64, DynamicStorage<f64>>::from_vec(n, 1, cg_b).unwrap();
    let mut cg = ConjugateGradient::with_preconditioner(DiagonalPreconditioner::new());
    cg.set_tolerance(1e-12);
    let x_cg_rust = cg.solve(&a_spd, &b_cg_mat).unwrap();

    for i in 0..n {
        let rv = *x_cg_rust.get(i, 0).unwrap();
        let cv = cg_x[i];
        assert!(
            (rv - cv).abs() < 1e-9,
            "CG Solution mismatch at index index {}: Rust={} C++={}",
            i,
            rv,
            cv
        );
    }

    // 2. BiCGSTAB Test
    let mut a_gen = SparseMatrix::<f64>::new(n, n, StorageOrder::ColMajor);
    let mut triplets_gen = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let val = ((i * 13 + j * 7) % 17) as f64 - 8.0;
            if val != 0.0 && (i as isize - j as isize).abs() <= 3 {
                triplets_gen.push(Triplet::new(i, j, val));
            }
        }
        triplets_gen.push(Triplet::new(i, i, 50.0));
    }
    a_gen.set_from_triplets(triplets_gen);

    let b_bicg_mat = Matrix::<f64, DynamicStorage<f64>>::from_vec(n, 1, bicg_b).unwrap();
    let mut bicg = BiCGSTAB::with_preconditioner(DiagonalPreconditioner::new());
    bicg.set_tolerance(1e-12);
    let x_bicg_rust = bicg.solve(&a_gen, &b_bicg_mat).unwrap();

    for i in 0..n {
        let rv = *x_bicg_rust.get(i, 0).unwrap();
        let cv = bicg_x[i];
        assert!(
            (rv - cv).abs() < 1e-9,
            "BiCGSTAB Solution mismatch at index index {}: Rust={} C++={}",
            i,
            rv,
            cv
        );
    }
}
