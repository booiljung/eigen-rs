use eigen_rs::core::matrix::MatrixX;
use eigen_rs::core::optimization::hybrid::Hybrid;
use eigen_rs::core::optimization::levenberg_marquardt::LevenbergMarquardt;
use eigen_rs::core::optimization::{Functor, Status};
// use eigen_rs::core::scalar::Scalar;

/// Problem: Find the center (xc, yc) and radius r of a circle
/// that passes through points (2, 5), (0, 3), (2, 1).
/// Expected: xc=2, yc=3, r=2.
struct CircleFitting {
    points: Vec<(f64, f64)>,
}

impl Functor<f64> for CircleFitting {
    fn inputs(&self) -> usize {
        3
    } // xc, yc, r
    fn values(&self) -> usize {
        3
    } // 3 points

    fn operator(&self, x: &MatrixX<f64>, fvec: &mut MatrixX<f64>) -> Result<(), String> {
        let xc = *x.get(0, 0).unwrap();
        let yc = *x.get(1, 0).unwrap();
        let r = *x.get(2, 0).unwrap();

        for i in 0..3 {
            let (px, py) = self.points[i];
            let res = (xc - px).powi(2) + (yc - py).powi(2) - r.powi(2);
            *fvec.get_mut(i, 0).unwrap() = res;
        }
        Ok(())
    }

    fn jacobian(&self, x: &MatrixX<f64>, fjac: &mut MatrixX<f64>) -> Result<(), String> {
        let xc = *x.get(0, 0).unwrap();
        let yc = *x.get(1, 0).unwrap();
        let r = *x.get(2, 0).unwrap();

        for i in 0..3 {
            let (px, py) = self.points[i];
            // df/dxc = 2(xc - px)
            *fjac.get_mut(i, 0).unwrap() = 2.0 * (xc - px);
            // df/dyc = 2(yc - py)
            *fjac.get_mut(i, 1).unwrap() = 2.0 * (yc - py);
            // df/dr = -2r
            *fjac.get_mut(i, 2).unwrap() = -2.0 * r;
        }
        Ok(())
    }
}

#[test]
fn test_levenberg_marquardt_circle() {
    let problem = CircleFitting {
        points: vec![(2.0, 5.0), (0.0, 3.0), (2.0, 1.0)],
    };

    let lm = LevenbergMarquardt::new();
    let mut x = MatrixX::<f64>::from_vec(3, 1, vec![1.0, 1.0, 1.0]).unwrap(); // Initial guess

    let status = lm.minimize(&problem, &mut x).unwrap();
    assert_eq!(status, Status::Converged);

    let xc = *x.get(0, 0).unwrap();
    let yc = *x.get(1, 0).unwrap();
    let r = *x.get(2, 0).unwrap();

    assert!((xc - 2.0).abs() < 1e-6);
    assert!((yc - 3.0).abs() < 1e-6);
    assert!((r.abs() - 2.0).abs() < 1e-6);
}

/// f1(x, y) = x^2 + y^2 - 4 = 0
/// f2(x, y) = e^x + y - 1 = 0
struct RootFinding;

impl Functor<f64> for RootFinding {
    fn inputs(&self) -> usize {
        2
    }
    fn values(&self) -> usize {
        2
    }

    fn operator(&self, x: &MatrixX<f64>, fvec: &mut MatrixX<f64>) -> Result<(), String> {
        let xv = *x.get(0, 0).unwrap();
        let yv = *x.get(1, 0).unwrap();
        *fvec.get_mut(0, 0).unwrap() = xv.powi(2) + yv.powi(2) - 4.0;
        *fvec.get_mut(1, 0).unwrap() = xv.exp() + yv - 1.0;
        Ok(())
    }

    fn jacobian(&self, x: &MatrixX<f64>, fjac: &mut MatrixX<f64>) -> Result<(), String> {
        let xv = *x.get(0, 0).unwrap();
        let yv = *x.get(1, 0).unwrap();
        // df1/dx = 2x, df1/dy = 2y
        *fjac.get_mut(0, 0).unwrap() = 2.0 * xv;
        *fjac.get_mut(0, 1).unwrap() = 2.0 * yv;
        // df2/dx = e^x, df2/dy = 1
        *fjac.get_mut(1, 0).unwrap() = xv.exp();
        *fjac.get_mut(1, 1).unwrap() = 1.0;
        Ok(())
    }
}

#[test]
fn test_hybrid_root() {
    let problem = RootFinding;
    let hybrid = Hybrid::new();
    let mut x = MatrixX::<f64>::from_vec(2, 1, vec![1.0, -1.0]).unwrap();

    let status = hybrid.solve(&problem, &mut x).unwrap();
    assert_eq!(status, Status::Converged);

    let xv = *x.get(0, 0).unwrap();
    let yv = *x.get(1, 0).unwrap();

    // Verify residue
    assert!((xv.powi(2) + yv.powi(2) - 4.0).abs() < 1e-6);
    assert!((xv.exp() + yv - 1.0).abs() < 1e-6);
}
