//! Levenberg-Marquardt non-linear least squares solver.

use crate::core::scalar::Scalar;
use crate::core::matrix::MatrixX;
use crate::core::optimization::{Functor, Status};
use crate::core::xpr::MatrixXpr;

/// Levenberg-Marquardt solver.
pub struct LevenbergMarquardt<T: Scalar> {
    max_iter: usize,
    xtol: f64,
    ftol: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + 'static> Default for LevenbergMarquardt<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + 'static> LevenbergMarquardt<T> {
    pub fn new() -> Self {
        Self {
            max_iter: 1000,
            xtol: 1e-8,
            ftol: 1e-10,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iter = max_iter;
    }

    pub fn minimize<F: Functor<T>>(&self, functor: &F, x: &mut MatrixX<T>) -> Result<Status, String> {
        let n = functor.inputs();
        let m = functor.values();
        
        let mut fvec = MatrixX::<T>::new_dynamic(m, 1)?;
        let mut fjac = MatrixX::<T>::new_dynamic(m, n)?;
        
        functor.operator(x, &mut fvec)?;
        let mut current_err = self.total_error(&fvec);
        
        let mut lambda = T::default();
        let mut lambda_initialized = false;
        
        for _iter in 0..self.max_iter {
            functor.jacobian(x, &mut fjac)?;
            
            // J^T * J
            let mut jtj = MatrixX::<T>::new_dynamic(n, n)?;
            let fjac_t = fjac.transpose();
            let prod = &fjac_t * &fjac;
            jtj.assign(&prod)?;
            
            if !lambda_initialized {
                // Initialize lambda = 1e-3 * max(diag(J^T J))
                let mut max_diag = T::default();
                for i in 0..n {
                    let d = *jtj.get(i, i).unwrap();
                    if d > max_diag { max_diag = d; }
                }
                lambda = max_diag * T::from_f64(1e-3);
                if lambda.to_f64() < 1e-6 { lambda = T::from_f64(1e-3); }
                lambda_initialized = true;
            }

            // RHS = -J^T * f
            let mut rhs = MatrixX::<T>::new_dynamic(n, 1)?;
            let rhs_prod = &fjac_t * &fvec;
            rhs.assign_product(&rhs_prod)?;
            for i in 0..n {
                if let Some(val) = rhs.get_mut(i, 0) {
                    *val = -*val;
                }
            }
            
            // (J^T * J + lambda * I) * delta = rhs
            let mut a = jtj.clone();
            for i in 0..n {
                if let Some(val) = a.get_mut(i, i) {
                    *val += lambda;
                }
            }
            
            // Solve using QR
            let qr = a.householder_qr()?;
            let delta = qr.solve(&rhs)?;
            
            // Try new x
            let mut x_new = x.clone();
            for i in 0..n {
                if let Some(val) = x_new.get_mut(i, 0) {
                    *val += *delta.get(i, 0).unwrap();
                }
            }
            
            let mut fvec_new = MatrixX::<T>::new_dynamic(m, 1)?;
            functor.operator(&x_new, &mut fvec_new)?;
            let new_err = self.total_error(&fvec_new);
            
            if new_err < current_err {
                // Convergence check before accepting
                if (current_err - new_err).to_f64().abs() < self.ftol * current_err.to_f64().abs() + 1e-14 {
                    *x = x_new;
                    return Ok(Status::Converged);
                }

                // Accept step
                *x = x_new;
                fvec = fvec_new;
                current_err = new_err;
                lambda /= T::from_f64(10.0);
            } else {
                // Reject step, increase damping
                lambda *= T::from_f64(10.0);
            }
            
            if lambda.to_f64() > 1e16 {
                return Ok(Status::MaxIterationsReached);
            }
        }
        let _ = self.xtol; // Silence warning
        
        if current_err.to_f64() < 1e-12 {
            Ok(Status::Converged)
        } else {
            Ok(Status::MaxIterationsReached)
        }
    }

    fn total_error(&self, fvec: &MatrixX<T>) -> T {
        let mut sum = T::default();
        for i in 0..fvec.rows() {
            let val = fvec.eval(i, 0);
            sum += val * val;
        }
        sum
    }
}
