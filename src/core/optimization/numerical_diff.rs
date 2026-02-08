//! Numerical differentiation using finite differences.

use crate::core::scalar::Scalar;
use crate::core::matrix::MatrixX;
use crate::core::optimization::Functor;
use crate::core::xpr::MatrixXpr;

/// A trait for functions that only provide residuals.
/// 
/// `NumericalDiff` and `AutoDiff` (if implemented for this) can wrap this
/// to provide the full `Functor` interface.
pub trait Residuals<T: Scalar> {
    fn inputs(&self) -> usize;
    fn values(&self) -> usize;
    fn operator(&self, x: &MatrixX<T>, fvec: &mut MatrixX<T>) -> Result<(), String>;
}

/// A wrapper that adds numerical Jacobian computation to a `Residuals` implementation.
pub struct NumericalDiff<F: Residuals<T>, T: Scalar> {
    func: F,
    eps: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<F: Residuals<T>, T: Scalar> NumericalDiff<F, T> {
    pub fn new(func: F) -> Self {
        Self {
            func,
            eps: 1e-7, // sqrt(machine epsilon) for f64
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn set_epsilon(&mut self, eps: f64) {
        self.eps = eps;
    }
}

impl<F: Residuals<T>, T: Scalar + 'static> Functor<T> for NumericalDiff<F, T> {
    fn inputs(&self) -> usize { self.func.inputs() }
    fn values(&self) -> usize { self.func.values() }

    fn operator(&self, x: &MatrixX<T>, fvec: &mut MatrixX<T>) -> Result<(), String> {
        self.func.operator(x, fvec)
    }

    fn jacobian(&self, x: &MatrixX<T>, fjac: &mut MatrixX<T>) -> Result<(), String> {
        let n = self.inputs();
        let m = self.values();
        let mut fvec = MatrixX::<T>::new_dynamic(m, 1)?;
        let mut fvec_eps = MatrixX::<T>::new_dynamic(m, 1)?;
        
        // Base evaluation
        self.func.operator(x, &mut fvec)?;
        
        let h = T::from_f64(self.eps);
        let inv_h = T::from_f64(1.0 / self.eps);

        for j in 0..n {
            let mut x_eps = x.clone();
            if let Some(val) = x_eps.get_mut(j, 0) {
                *val += h;
            }
            
            self.func.operator(&x_eps, &mut fvec_eps)?;
            
            for i in 0..m {
                let diff = (fvec_eps.eval(i, 0) - fvec.eval(i, 0)) * inv_h;
                *fjac.get_mut(i, j).unwrap() = diff;
            }
        }
        
        Ok(())
    }
}
