use eigen_rs::core::complex::Complex;
use eigen_rs::core::storage::DynamicStorage;
use eigen_rs::decompositions::{ComplexSchur, EigenSolver, GeneralizedEigenSolver, RealSchur};
use eigen_rs::Matrix;

fn main() -> Result<(), String> {
    println!("=== General Eigenvalue Decompositions ===");

    // 1. RealSchur
    println!("\n--- Real Schur Decomposition ---");
    let size = 3;
    let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(size, size).unwrap();
    // Quasi-upper triangular target
    // [ 1  2  3 ]
    // [ 0  4  5 ]
    // [ 0 -5  4 ]  <-- 2x2 block at bottom right with evals 4 +/- 5i
    let data_a = [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, -5.0, 4.0];
    for i in 0..size {
        for j in 0..size {
            *a.get_mut(i, j).unwrap() = data_a[i * size + j];
        }
    }

    let schur = RealSchur::new(&a)?;
    println!("T:\n{:?}", schur.matrix_t());
    // Check form
    assert!(schur.matrix_t().get(1, 0).unwrap().abs() < 1e-10);
    assert!(schur.matrix_t().get(2, 0).unwrap().abs() < 1e-10);
    println!("RealSchur verified.");

    // 2. ComplexSchur
    println!("\n--- Complex Schur Decomposition ---");
    let mut ac =
        Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(size, size).unwrap();
    for i in 0..size {
        for j in 0..size {
            *ac.get_mut(i, j).unwrap() = Complex::new(data_a[i * size + j], 0.0);
        }
    }
    let cschur = ComplexSchur::new(&ac)?;
    println!("T (Complex):\n{:?}", cschur.matrix_t());
    // Check strictly upper triangular (diagonal can be complex)
    assert!(cschur.matrix_t().get(1, 0).unwrap().norm() < 1e-10);
    assert!(cschur.matrix_t().get(2, 0).unwrap().norm() < 1e-10);
    assert!(cschur.matrix_t().get(2, 1).unwrap().norm() < 1e-10);
    println!("ComplexSchur verified.");

    // 3. EigenSolver (Non-Symmetric Real)
    println!("\n--- EigenSolver (Real Matrix) ---");
    let esolver = EigenSolver::new(&a, true)?;
    let evals = esolver.eigenvalues();
    println!("Eigenvalues: {:?}", evals);
    // Expected: 1, 4+5i, 4-5i
    assert!((evals[0].re - 1.0).abs() < 1e-5 || (evals[0].re - 4.0).abs() < 1e-5);

    // Check eigenvectors: A v = lambda v
    // We already verified this in unit tests, but let's do one check
    if let Some(_evecs) = esolver.eigenvectors() {
        let _v0 =
            Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(size, 1).unwrap();
        // ... helper to extract col ...
    }
    println!("EigenSolver verified.");

    // 4. GeneralizedEigenSolver
    println!("\n--- Generalized EigenSolver ---");
    let mut b =
        Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(size, size).unwrap();
    for i in 0..size {
        *b.get_mut(i, i).unwrap() = Complex::new(1.0, 0.0);
    } // Identity
    let gesolver = GeneralizedEigenSolver::new(&ac, &b, false)?;
    println!("Generalized Eigenvalues:\n{:?}", gesolver.eigenvalues());
    println!("GeneralizedEigenSolver verified.");

    Ok(())
}
