
#[cfg(test)]
mod tests {
    use eigen_rs::core::matrix::{Matrix, DynamicStorage};
    use eigen_rs::core::complex::Complex;
    

    #[test]
    fn test_ght_hang_repro() {
        println!("Starting GHT non-trivial hang reproduction...");
        let size = 3;
        // Non-trivial A and B to force reduction and QZ steps
        let mut a = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(size, size).unwrap();
        let mut b = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(size, size).unwrap();

        let data_a = [
            Complex::new(1.0, 1.0), Complex::new(2.0, 0.0), Complex::new(0.0, 1.0),
            Complex::new(0.5, 0.5), Complex::new(3.0, 2.0), Complex::new(1.0, -1.0),
            Complex::new(1.0, 0.0), Complex::new(1.0, 1.0), Complex::new(2.0, 2.0),
        ];
        let data_b = [
            Complex::new(5.0, 0.0), Complex::new(1.0, 1.0), Complex::new(0.0, 0.0),
            Complex::new(1.0, -1.0), Complex::new(4.0, 2.0), Complex::new(2.0, 1.0),
            Complex::new(3.0, 0.0), Complex::new(1.0, 0.0), Complex::new(5.0, -1.0),
        ];

        for i in 0..size {
            for j in 0..size {
                *a.get_mut(i, j).unwrap() = data_a[i * size + j];
                *b.get_mut(i, j).unwrap() = data_b[i * size + j];
            }
        }

        // This call previously caused a hang (in QZ step?)
        use eigen_rs::core::decompositions::generalized_eigen_solver::GeneralizedEigenSolver;
        let solver = GeneralizedEigenSolver::new(&a, &b, true).expect("Solver failed");
        println!("GeneralizedEigenSolver finished successfully!");
        
        // Verify determinant condition: det(A - lambda B) ~ 0
        let evals = solver.eigenvalues();
        for k in 0..size {
             let lambda = evals.get(k, 0).unwrap();
             // det(A - lambda B) check
             let mut char_mat = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(size, size).unwrap();
             for r in 0..size {
                 for c in 0..size {
                     *char_mat.get_mut(r, c).unwrap() = *a.get(r, c).unwrap() - (*lambda) * (*b.get(r, c).unwrap());
                 }
             }
             let c = |r, c| *char_mat.get(r, c).unwrap();
             let det = c(0,0)*(c(1,1)*c(2,2) - c(1,2)*c(2,1))
                     - c(0,1)*(c(1,0)*c(2,2) - c(1,2)*c(2,0))
                     + c(0,2)*(c(1,0)*c(2,1) - c(1,1)*c(2,0));
             
             println!("Lambda: {}, Det: {}", lambda, det.norm());
             assert!(det.norm() < 1e-4, "Determinant too large: {}", det.norm());
        }
    }
}
