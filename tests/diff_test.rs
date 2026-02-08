use eigen_rs::core::matrix::MatrixX;
use eigen_rs::core::optimization::autodiff::{AdResiduals, AutoDiff};
use eigen_rs::core::optimization::hybrid::Hybrid;
use eigen_rs::core::optimization::levenberg_marquardt::LevenbergMarquardt;
use eigen_rs::core::optimization::numerical_diff::{NumericalDiff, Residuals};
use eigen_rs::core::optimization::Status;
use eigen_rs::core::scalar::Scalar;

/// Problem: Find the center (xc, yc) and radius r of a circle
/// that passes through points (2, 5), (0, 3), (2, 1).
/// This implementation is generic over D: Scalar to support AutoDiff.
struct CircleFittingAd {
    points: Vec<(f64, f64)>,
}

impl AdResiduals<f64> for CircleFittingAd {
    fn inputs(&self) -> usize {
        3
    }
    fn values(&self) -> usize {
        3
    }

    fn operator<D: Scalar>(&self, x: &MatrixX<D>, fvec: &mut MatrixX<D>) -> Result<(), String> {
        let xc = *x.get(0, 0).unwrap();
        let yc = *x.get(1, 0).unwrap();
        let r = *x.get(2, 0).unwrap();

        for i in 0..3 {
            let (px, py) = self.points[i];
            let px_d = D::from_f64(px);
            let py_d = D::from_f64(py);

            // f = (xc - px)^2 + (yc - py)^2 - r^2
            let res = (xc - px_d) * (xc - px_d) + (yc - py_d) * (yc - py_d) - r * r;
            *fvec.get_mut(i, 0).unwrap() = res;
        }
        Ok(())
    }
}

/// A wrapper to use AdResiduals with NumericalDiff (which expects Residuals<f64>).
struct AdToRes<'a, F: AdResiduals<f64>>(&'a F);

impl<'a, F: AdResiduals<f64>> Residuals<f64> for AdToRes<'a, F> {
    fn inputs(&self) -> usize {
        self.0.inputs()
    }
    fn values(&self) -> usize {
        self.0.values()
    }
    fn operator(&self, x: &MatrixX<f64>, fvec: &mut MatrixX<f64>) -> Result<(), String> {
        self.0.operator(x, fvec)
    }
}

#[test]
fn test_autodiff_circle_fitting() {
    let problem = CircleFittingAd {
        points: vec![(2.0, 5.0), (0.0, 3.0), (2.0, 1.0)],
    };

    // Wrap with AutoDiff
    let ad_functor = AutoDiff::new(problem);

    let lm = LevenbergMarquardt::new();
    let mut x = MatrixX::<f64>::from_vec(3, 1, vec![1.0, 1.0, 1.0]).unwrap();

    let status = lm.minimize(&ad_functor, &mut x).unwrap();
    assert_eq!(status, Status::Converged);

    let xc = *x.get(0, 0).unwrap();
    let yc = *x.get(1, 0).unwrap();
    let r = *x.get(2, 0).unwrap();

    assert!((xc - 2.0).abs() < 1e-6);
    assert!((yc - 3.0).abs() < 1e-6);
    assert!((r.abs() - 2.0).abs() < 1e-6);
}

#[test]
fn test_numerical_diff_circle_fitting() {
    let problem = CircleFittingAd {
        points: vec![(2.0, 5.0), (0.0, 3.0), (2.0, 1.0)],
    };

    // Wrap with NumericalDiff
    let wrapper = AdToRes(&problem);
    let nd_functor = NumericalDiff::new(wrapper);

    let lm = LevenbergMarquardt::new();
    let mut x = MatrixX::<f64>::from_vec(3, 1, vec![1.0, 1.0, 1.0]).unwrap();

    let status = lm.minimize(&nd_functor, &mut x).unwrap();
    assert_eq!(status, Status::Converged);

    let xc = *x.get(0, 0).unwrap();
    let yc = *x.get(1, 0).unwrap();
    let r = *x.get(2, 0).unwrap();

    assert!((xc - 2.0).abs() < 1e-6);
    assert!((yc - 3.0).abs() < 1e-6);
    assert!((r.abs() - 2.0).abs() < 1e-6);
}

/// Test with non-linear root finding (Hybrid solver)
struct RootFindingAd;

impl AdResiduals<f64> for RootFindingAd {
    fn inputs(&self) -> usize {
        2
    }
    fn values(&self) -> usize {
        2
    }

    fn operator<D: Scalar>(&self, x: &MatrixX<D>, fvec: &mut MatrixX<D>) -> Result<(), String> {
        let xv = *x.get(0, 0).unwrap();
        let yv = *x.get(1, 0).unwrap();

        // f1 = x^2 + y^2 - 4
        *fvec.get_mut(0, 0).unwrap() = xv * xv + yv * yv - D::from_f64(4.0);
        // f2 = exp(x) + y - 1
        *fvec.get_mut(1, 0).unwrap() = xv.exp() + yv - D::from_f64(1.0);
        Ok(())
    }
}

#[test]
fn test_autodiff_hybrid_root() {
    let problem = RootFindingAd;
    let ad_functor = AutoDiff::new(problem);

    let hybrid = Hybrid::new();
    let mut x = MatrixX::<f64>::from_vec(2, 1, vec![1.0, -1.0]).unwrap();

    let status = hybrid.solve(&ad_functor, &mut x).unwrap();
    assert_eq!(status, Status::Converged);

    let xv = *x.get(0, 0).unwrap();
    let yv = *x.get(1, 0).unwrap();

    assert!((xv * xv + yv * yv - 4.0).abs() < 1e-6);
    assert!((xv.exp() + yv - 1.0).abs() < 1e-6); // f64 has exp()
}
