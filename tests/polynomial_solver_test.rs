#[cfg(test)]
mod tests {
    use eigen_rs::unsupported::polynomials::{poly_eval, PolynomialSolver, JenkinsTraubSolver};
    
    #[test]
    fn test_poly_eval() {
        // P(x) = 1 + 2x + 3x^2
        let poly = vec![1.0, 2.0, 3.0];
        let x = 2.0;
        // 1 + 4 + 12 = 17
        assert_eq!(poly_eval(&poly, x), 17.0);
    }
    
    #[test]
    fn test_companion_matrix_roots() {
        // (x - 2)(x + 2) = x^2 - 4. Coeffs: [-4, 0, 1]
        let poly = vec![-4.0, 0.0, 1.0]; 
        let mut solver = PolynomialSolver::new();
        solver.compute(&poly).unwrap();
        let roots = solver.roots();
        
        assert_eq!(roots.len(), 2);
        // Roots should be 2, -2
        let mut found_2 = false;
        let mut found_neg_2 = false;
        
        for r in roots {
            if (r.re - 2.0_f64).abs() < 1e-5 { found_2 = true; }
            if (r.re + 2.0_f64).abs() < 1e-5 { found_neg_2 = true; }
        }
        assert!(found_2 && found_neg_2);
    }

    #[test]
    fn test_jenkins_traub_roots() {
        // (x - 2)(x + 2) = x^2 - 4. Coeffs: [-4, 0, 1]
        let poly = vec![-4.0, 0.0, 1.0];
        let mut solver = JenkinsTraubSolver::new();
        solver.compute(&poly).unwrap();
        let roots = solver.roots();
        
        assert_eq!(roots.len(), 2);
        
        let mut found_2 = false;
        let mut found_neg_2 = false;
        
        for r in roots {
            if (r.re - 2.0_f64).abs() < 1e-5 { found_2 = true; }
            if (r.re + 2.0_f64).abs() < 1e-5 { found_neg_2 = true; }
        }
        assert!(found_2 && found_neg_2);
    }
}
