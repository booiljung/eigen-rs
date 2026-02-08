use eigen_rs::unsupported::special_functions::SpecialFunctions;

#[test]
fn test_erf() {
    let tol = 1e-6;
    assert!((0.0f64.erf() - 0.0).abs() < tol);
    assert!((1.0f64.erf() - 0.84270079).abs() < tol);
    assert!((-1.0f64.erf() + 0.84270079).abs() < tol);
    
    // f32
    assert!((1.0f32.erf() - 0.84270079f32).abs() < tol as f32);
}

#[test]
fn test_erfc() {
    let tol = 1e-6;
    assert!((1.0f64.erfc() - 0.15729921).abs() < tol);
}

#[test]
fn test_lgamma() {
    let tol = 1e-6;
    assert!((1.0f64.lgamma() - 0.0).abs() < tol);
    assert!((2.0f64.lgamma() - 0.0).abs() < tol);
    assert!((3.0f64.lgamma() - 0.69314718).abs() < tol);
    assert!((4.0f64.lgamma() - 1.79175947).abs() < tol);
}

#[test]
fn test_bessel_j0() {
    let tol = 1e-4;
    assert!((0.0f64.bessel_j0() - 1.0).abs() < tol);
    assert!((2.4048f64.bessel_j0()).abs() < tol); 
}

#[test]
fn test_bessel_j1() {
    let tol = 1e-4;
    assert!((0.0f64.bessel_j1() - 0.0).abs() < tol);
    assert!((3.8317f64.bessel_j1()).abs() < tol);
}
