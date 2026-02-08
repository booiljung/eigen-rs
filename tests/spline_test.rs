use eigen_rs::unsupported::Spline;
use eigen_rs::core::matrix::MatrixX;


#[test]
fn test_spline_linear_eval() {
    // Linear B-Spline (Degree 1)
    // 2 Control points: (0,0) and (1,1)
    // Knots: [0, 0, 1, 1] (Clamped)
    // Degree k=1. n=2 (points). Len = n+k+1 = 4.
    
    let knots = vec![0.0, 0.0, 1.0, 1.0];
    
    // Control Points: [[0, 1], [0, 1]] (Column 0: (0,0), Column 1: (1,1))
    // Linear interpolation f(u) = (u, u)
    let mut ctrl = MatrixX::<f64>::new_dynamic(2, 2).unwrap();
    *ctrl.get_mut(0, 0).unwrap() = 0.0; *ctrl.get_mut(1, 0).unwrap() = 0.0;
    *ctrl.get_mut(0, 1).unwrap() = 1.0; *ctrl.get_mut(1, 1).unwrap() = 1.0;
    
    let spline = Spline::new(knots, ctrl, 1);
    
    // Eval at 0.0
    let res0 = spline.eval(0.0);
    assert!((res0.get(0,0).unwrap() - 0.0).abs() < 1e-10);
    assert!((res0.get(1,0).unwrap() - 0.0).abs() < 1e-10);

    // Eval at 0.5 -> expect (0.5, 0.5)
    let res5 = spline.eval(0.5);
    assert!((res5.get(0,0).unwrap() - 0.5).abs() < 1e-10);
    assert!((res5.get(1,0).unwrap() - 0.5).abs() < 1e-10);

    // Eval at 1.0
    let res1 = spline.eval(1.0);
    assert!((res1.get(0,0).unwrap() - 1.0).abs() < 1e-10);
    assert!((res1.get(1,0).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_spline_quadratic_eval() {
    // Quadratic B-Spline (Degree 2)
    // 3 Control points
    // Knots: [0, 0, 0, 1, 1, 1] (Clamped)
    // k=2, n=3. Len = 3+2+1 = 6.
    
    let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    
    // Points: P0=(0,0), P1=(0.5, 1), P2=(1,0)
    let mut ctrl = MatrixX::<f64>::new_dynamic(2, 3).unwrap();
    *ctrl.get_mut(0, 0).unwrap() = 0.0; *ctrl.get_mut(1, 0).unwrap() = 0.0;
    *ctrl.get_mut(0, 1).unwrap() = 0.5; *ctrl.get_mut(1, 1).unwrap() = 1.0;
    *ctrl.get_mut(0, 2).unwrap() = 1.0; *ctrl.get_mut(1, 2).unwrap() = 0.0;

    let spline = Spline::new(knots, ctrl, 2);
    
    // Midpoint u=0.5
    // For quadratic clamped bezier-like: 
    // B(0.5) = (1/4)P0 + (1/2)P1 + (1/4)P2
    // x = 0 + 0.25 + 0.25 = 0.5
    // y = 0 + 0.5 + 0 = 0.5
    let res = spline.eval(0.5);
    
    assert!((res.get(0,0).unwrap() - 0.5).abs() < 1e-10, "X val: {}", res.get(0,0).unwrap());
    assert!((res.get(1,0).unwrap() - 0.5).abs() < 1e-10, "Y val: {}", res.get(1,0).unwrap());
}

#[test]
fn test_spline_interpolation_linear() {
    let mut points = MatrixX::<f64>::new_dynamic(2, 3).unwrap();
    // P0
    *points.get_mut(0, 0).unwrap() = 0.0; *points.get_mut(1, 0).unwrap() = 0.0;
    // P1
    *points.get_mut(0, 1).unwrap() = 1.0; *points.get_mut(1, 1).unwrap() = 1.0;
    // P2
    *points.get_mut(0, 2).unwrap() = 2.0; *points.get_mut(1, 2).unwrap() = 0.0;
    
    let spline = Spline::interpolate(&points, 1).expect("Interpolation failed");
    
    let p0 = spline.eval(0.0);
    assert!((p0.get(0,0).unwrap() - 0.0).abs() < 1e-6);
    assert!((p0.get(1,0).unwrap() - 0.0).abs() < 1e-6);
    
    let p2 = spline.eval(1.0);
    assert!((p2.get(0,0).unwrap() - 2.0).abs() < 1e-6);
}

#[test]
fn test_spline_robustness() {
    // 1. Insufficient points
    let points_small = MatrixX::<f64>::new_dynamic(2, 2).unwrap(); // 2 points
    // Degree 2 requires at least 3 points
    let res = Spline::interpolate(&points_small, 2);
    assert!(res.is_err(), "Should fail with insufficient points");

    // 2. Collinear points (should work)
    let mut points_collinear = MatrixX::<f64>::new_dynamic(2, 3).unwrap();
    // (0,0), (1,1), (2,2)
    *points_collinear.get_mut(0, 0).unwrap() = 0.0; *points_collinear.get_mut(1, 0).unwrap() = 0.0;
    *points_collinear.get_mut(0, 1).unwrap() = 1.0; *points_collinear.get_mut(1, 1).unwrap() = 1.0;
    *points_collinear.get_mut(0, 2).unwrap() = 2.0; *points_collinear.get_mut(1, 2).unwrap() = 2.0;

    let spline = Spline::interpolate(&points_collinear, 2).expect("Collinear interpolation failed");
    // Midpoint (1,1)
    let mid = spline.eval(0.5); // Parameter t=0.5 corresponds to middle point for symmetric spacing
    assert!((mid.get(0,0).unwrap() - 1.0).abs() < 1e-6);
    assert!((mid.get(1,0).unwrap() - 1.0).abs() < 1e-6);

    // 3. Out of bounds (Extrapolation)
    // The implementation clamps span index, effectively extrapolating using the first/last polynomial.
    // For a line (0,0)-(2,2), extrapolation should follow the line.
    let out_neg = spline.eval(-0.5);
    // Depending on basis, B-spline extrapolation might not be linear if knots are clamped.
    // But for clamped knots, the first basis function starts at 0.
    // Actually, evaluating outside [t_k, t_{n+1}] is mathematically undefined but our code clamps index.
    // Let's just ensure it doesn't panic.
    let _ = out_neg;
    
    // 4. Closed Loop (First = Last)
    let mut points_loop = MatrixX::<f64>::new_dynamic(2, 4).unwrap();
    // (0,0) -> (1,1) -> (1,0) -> (0,0)
    *points_loop.get_mut(0, 0).unwrap() = 0.0; *points_loop.get_mut(1, 0).unwrap() = 0.0;
    *points_loop.get_mut(0, 1).unwrap() = 1.0; *points_loop.get_mut(1, 1).unwrap() = 1.0;
    *points_loop.get_mut(0, 2).unwrap() = 1.0; *points_loop.get_mut(1, 2).unwrap() = 0.0;
    *points_loop.get_mut(0, 3).unwrap() = 0.0; *points_loop.get_mut(1, 3).unwrap() = 0.0;

    let spline_loop = Spline::interpolate(&points_loop, 3).expect("Loop interpolation failed");
    let p_start = spline_loop.eval(0.0);
    let p_end = spline_loop.eval(1.0);
    
    assert!((p_start.get(0,0).unwrap() - p_end.get(0,0).unwrap()).abs() < 1e-6);
    assert!((p_start.get(1,0).unwrap() - p_end.get(1,0).unwrap()).abs() < 1e-6);
}
