use eigen_rs::unsupported::{poly_eval, PolynomialSolver};
use eigen_rs::core::complex::Complex;

#[test]
fn test_poly_eval() {
    // P(x) = 1 + 2x + 3x^2
    let poly = vec![1.0, 2.0, 3.0];
    let x = 2.0;
    // 1 + 4 + 12 = 17
    let val = poly_eval(&poly, x);
    assert!((val - 17.0_f64).abs() < 1e-10);

    // P(x) = 5
    let poly_const = vec![5.0];
    assert!((poly_eval(&poly_const, 10.0) - 5.0_f64).abs() < 1e-10);
}

#[test]
fn test_polynomial_solver_real_roots() {
    // x^2 - 1 = 0 => roots: 1, -1
    // coeffs: [-1, 0, 1]
    let poly = vec![-1.0, 0.0, 1.0];
    
    let mut solver = PolynomialSolver::<f64>::new();
    solver.compute(&poly).expect("Solver failed");
    
    let roots = solver.roots();
    assert_eq!(roots.len(), 2);
    
    let r1 = Complex::new(1.0, 0.0);
    let r2 = Complex::new(-1.0, 0.0);
    
    // Check if distinct roots exist
    let found_r1 = roots.iter().any(|r| (*r - r1).norm() < 1e-6);
    let found_r2 = roots.iter().any(|r| (*r - r2).norm() < 1e-6);
    
    assert!(found_r1, "Root 1.0 not found");
    assert!(found_r2, "Root -1.0 not found");
}

#[test]
fn test_polynomial_solver_complex_roots() {
    // x^2 + 1 = 0 => roots: i, -i
    // coeffs: [1, 0, 1]
    let poly = vec![1.0, 0.0, 1.0];
    
    let mut solver = PolynomialSolver::<f64>::new();
    solver.compute(&poly).expect("Solver failed");
    
    let roots = solver.roots();
    assert_eq!(roots.len(), 2);
    
    let r1 = Complex::new(0.0, 1.0);
    let r2 = Complex::new(0.0, -1.0);
    
    let found_r1 = roots.iter().any(|r| (*r - r1).norm() < 1e-6);
    let found_r2 = roots.iter().any(|r| (*r - r2).norm() < 1e-6);
    
    assert!(found_r1, "Root i not found");
    assert!(found_r2, "Root -i not found");
}

#[test]
fn test_polynomial_solver_edge_cases() {
    let mut solver = PolynomialSolver::<f64>::new();
    
    // Empty polynomial
    assert!(solver.compute(&[]).is_err());
    
    // Constant polynomial (P(x) = 5) -> No roots
    assert!(solver.compute(&[5.0]).is_ok());
    assert!(solver.roots().is_empty());
    
    // Leading zeros (0x^3 + 0x^2 + x - 1) -> x - 1 = 0 -> root 1
    // coeffs: [-1, 1, 0, 0]
    let poly_zeros = vec![-1.0, 1.0, 0.0, 0.0];
    solver.compute(&poly_zeros).expect("Should handle leading zeros");
    assert_eq!(solver.roots().len(), 1);
    assert!((solver.roots()[0].re - 1.0).abs() < 1e-6);
    
    // Zero polynomial (P(x) = 0) -> technically infinite roots, but implementation returns empty/ok
    assert!(solver.compute(&[0.0, 0.0]).is_ok()); 
    assert!(solver.roots().is_empty());
}

#[test]
fn test_polynomial_solver_cubic() {
    // x^3 - 1 = 0 -> roots: 1, -0.5 +/- i*sqrt(3)/2
    // coeffs: [-1, 0, 0, 1]
    let poly = vec![-1.0, 0.0, 0.0, 1.0];
    let mut solver = PolynomialSolver::<f64>::new();
    solver.compute(&poly).expect("Cubic failed");
    
    let roots = solver.roots();
    assert_eq!(roots.len(), 3);
    
    // Check for root 1
    let found_one = roots.iter().any(|r| (r.re - 1.0).abs() < 1e-6 && r.im.abs() < 1e-6);
    assert!(found_one, "Root 1 not found");
}
