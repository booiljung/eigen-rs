//! Hybrid non-linear equation solver (Newton-LineSearch).

use crate::core::matrix::MatrixX;
use crate::core::optimization::{Functor, Status};
use crate::core::scalar::Scalar;
use crate::core::xpr::MatrixXpr;

/// Hybrid solver using Newton method with backtracking line search.
pub struct Hybrid<T: Scalar> {
    max_iter: usize,
    xtol: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + std::cmp::PartialOrd + 'static> Default for Hybrid<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + 'static + std::cmp::PartialOrd> Hybrid<T> {
    pub fn new() -> Self {
        Self {
            max_iter: 100,
            xtol: 1e-8,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn solve<F: Functor<T>>(&self, functor: &F, x: &mut MatrixX<T>) -> Result<Status, String> {
        let n = functor.inputs();
        if n != functor.values() {
            return Err("Hybrid solver requires square system (inputs == values)".to_string());
        }

        let mut fvec = MatrixX::<T>::new_dynamic(n, 1)?;
        let mut fjac = MatrixX::<T>::new_dynamic(n, n)?;

        for _iter in 0..self.max_iter {
            functor.operator(x, &mut fvec)?;
            let current_norm = self.norm(&fvec);

            if current_norm.to_f64() < self.xtol {
                return Ok(Status::Converged);
            }

            functor.jacobian(x, &mut fjac)?;

            // Newton step: J * delta = -f
            let mut rhs = fvec.clone();
            for i in 0..n {
                if let Some(val) = rhs.get_mut(i, 0) {
                    *val = -*val;
                }
            }

            let qr = fjac.householder_qr()?;
            let delta = qr.solve(&rhs)?;

            // Backtracking Line Search (Armijo condition simplified)
            let mut alpha = 1.0;
            let mut accepted = false;

            for _ in 0..10 {
                // Max 10 line search steps
                let mut x_new = x.clone();
                for i in 0..n {
                    if let Some(val) = x_new.get_mut(i, 0) {
                        *val += *delta.get(i, 0).unwrap() * T::from_f64(alpha);
                    }
                }

                let mut fvec_new = MatrixX::<T>::new_dynamic(n, 1)?;
                functor.operator(&x_new, &mut fvec_new)?;
                let new_norm = self.norm(&fvec_new);

                if new_norm < current_norm {
                    *x = x_new;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }

            if !accepted {
                // Step was not accepted even with line search, try a small perturbation or stop
                return Ok(Status::MaxIterationsReached);
            }
        }

        Ok(Status::MaxIterationsReached)
    }

    fn norm(&self, fvec: &MatrixX<T>) -> T {
        let mut sum = T::default();
        for i in 0..fvec.rows() {
            let val = fvec.eval(i, 0);
            sum += val * val;
        }
        sum.sqrt()
    }
}
