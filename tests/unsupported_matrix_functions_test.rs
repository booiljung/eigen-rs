#[cfg(test)]
mod tests {
    use eigen_rs::core::matrix::{DynamicStorage, Matrix};

    use eigen_rs::core::xpr::MatrixXpr;
    use eigen_rs::unsupported::matrix_functions::{MatrixExponential, MatrixLogarithm};

    #[test]
    fn test_matrix_exp_identity() {
        let size = 3;
        let diff = 1e-4; // Pade approx error

        // exp(0) = I
        let z = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(size, size).unwrap();
        let exp_z = MatrixExponential::new(&z).compute().unwrap();

        for i in 0..size {
            for j in 0..size {
                let val = exp_z.eval(i, j);
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((val - expected).abs() < diff, "Mismatch at {},{}", i, j);
            }
        }
    }

    #[test]
    fn test_matrix_exp_diagonal() {
        let size = 2;
        let mut d = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(size, size).unwrap();
        *d.get_mut(0, 0).unwrap() = 1.0;
        *d.get_mut(1, 1).unwrap() = 2.0;

        // exp(diag(1, 2)) = diag(e, e^2)
        let exp_d = MatrixExponential::new(&d).compute().unwrap();

        let e1 = 1.0f64.exp();
        let e2 = 2.0f64.exp();

        assert!((exp_d.eval(0, 0) - e1).abs() < 1e-4);
        assert!((exp_d.eval(1, 1) - e2).abs() < 1e-4);
        assert!(exp_d.eval(0, 1).abs() < 1e-4);
        assert!(exp_d.eval(1, 0).abs() < 1e-4);
    }

    #[test]
    fn test_matrix_log_identity() {
        let size = 3;
        // log(I) = 0
        let i_mat = Matrix::<f64, DynamicStorage<f64>>::identity(size, size);
        let log_i = MatrixLogarithm::new(&i_mat).compute().unwrap();

        for i in 0..size {
            for j in 0..size {
                assert!(log_i.eval(i, j).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn test_matrix_exp_log_roundtrip() {
        let size = 2;
        let mut m = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(size, size).unwrap();
        // Small values for better convergence of simple implementations
        *m.get_mut(0, 0).unwrap() = 0.5;
        *m.get_mut(0, 1).unwrap() = 0.1;
        *m.get_mut(1, 0).unwrap() = -0.1;
        *m.get_mut(1, 1).unwrap() = 0.5;

        let exp_m = MatrixExponential::new(&m).compute().unwrap();
        let log_exp_m = MatrixLogarithm::new(&exp_m).compute().unwrap();

        for i in 0..size {
            for j in 0..size {
                let orig = m.eval(i, j);
                let roundtrip = log_exp_m.eval(i, j);
                assert!(
                    (orig - roundtrip).abs() < 1e-3,
                    "Roundtrip mismatch at {},{}: {} vs {}",
                    i,
                    j,
                    orig,
                    roundtrip
                );
            }
        }
    }
}
