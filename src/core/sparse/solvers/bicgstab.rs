//! BiCGSTAB (Bi-Conjugate Gradient Stabilized) solver for general non-symmetric sparse matrices.

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::sparse::solvers::iterative_solver_base::{IdentityPreconditioner, Preconditioner};
use crate::core::sparse::sparse_matrix::SparseMatrix;
use crate::core::storage::{DynamicStorage, Storage};

/// BiCGSTAB solver.
/// Solves Ax = b for general non-symmetric matrices A.
pub struct BiCGSTAB<T: Scalar, P: Preconditioner<T> = IdentityPreconditioner> {
    preconditioner: P,
    max_iterations: usize,
    tolerance: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Default for BiCGSTAB<T, IdentityPreconditioner> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> BiCGSTAB<T, IdentityPreconditioner> {
    /// Creates a new BiCGSTAB solver with default Identity preconditioner.
    pub fn new() -> Self {
        Self {
            preconditioner: IdentityPreconditioner::new(),
            max_iterations: 1000,
            tolerance: 1e-10,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar, P: Preconditioner<T>> BiCGSTAB<T, P> {
    /// Creates a new BiCGSTAB solver with a custom preconditioner.
    pub fn with_preconditioner(preconditioner: P) -> Self {
        Self {
            preconditioner,
            max_iterations: 1000,
            tolerance: 1e-10,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = max_iter;
    }

    pub fn set_tolerance(&mut self, tol: f64) {
        self.tolerance = tol;
    }

    /// Solves Ax = b iteratively.
    pub fn solve<S: Storage<T>>(
        &mut self,
        matrix: &SparseMatrix<T>,
        b: &Matrix<T, S>,
    ) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        let n = matrix.rows();
        if b.rows() != n {
            return Err("Incompatible dimensions".to_string());
        }

        #[cfg(feature = "cuda")]
        {
            if crate::core::tensor::device::cuda::is_cuda_device_active() {
                use crate::core::sparse::cuda_sparse_bridge::CudaSparseExt;
                // Attempt to solve on GPU. If it fails (e.g. missing memory, unsupported type), ignore and fall back to CPU.
                if let Ok(Some(x)) = matrix.try_bicgstab_cuda(b, self.max_iterations, self.tolerance) {
                    return Ok(x);
                }
            }
        }

        self.preconditioner.compute(matrix)?;

        let mut x = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, b.cols())?; // Initial x = 0

        for k in 0..b.cols() {
            let mut b_vec = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, 1)?;
            for i in 0..n {
                *b_vec.get_mut(i, 0).unwrap() = *b.get(i, k).unwrap();
            }

            // r = b - A*x (since x=0, r=b)
            let mut r = b_vec.clone();
            let r_tilde = r.clone(); // Shadow residual

            let mut rho = T::from_f64(1.0);
            let mut alpha = T::from_f64(1.0);
            let mut omega = T::from_f64(1.0);

            let mut v = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, 1)?;
            let mut p = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, 1)?;

            let b_norm = r.norm_f64();
            if b_norm < 1e-16 {
                continue;
            }

            for iter in 0..self.max_iterations {
                // rho_next = r_tilde^T * r
                let rho_next = self.dot_product(&r_tilde, &r);
                if rho_next.abs().to_f64() < 1e-20 {
                    return Err("BiCGSTAB stalled (rho)".to_string());
                }

                if iter == 0 {
                    p = r.clone();
                } else {
                    let beta = (rho_next / rho) * (alpha / omega);
                    // p = r + beta * (p - omega * v)
                    for i in 0..n {
                        *p.get_mut(i, 0).unwrap() = *r.get(i, 0).unwrap()
                            + beta * (*p.get(i, 0).unwrap() - omega * *v.get(i, 0).unwrap());
                    }
                }

                // p_hat = M^-1 * p
                let p_hat = self.preconditioner.solve(&p)?;

                // v = A * p_hat
                v = matrix.mul_dense(&p_hat)?;

                // alpha = rho_next / (r_tilde^T * v)
                alpha = rho_next / self.dot_product(&r_tilde, &v);

                // s = r - alpha * v
                let mut s = r.clone();
                for i in 0..n {
                    *s.get_mut(i, 0).unwrap() =
                        *s.get(i, 0).unwrap() - alpha * *v.get(i, 0).unwrap();
                }

                if s.norm_f64() / b_norm < self.tolerance {
                    // x = x + alpha * p_hat
                    for i in 0..n {
                        *x.get_mut(i, k).unwrap() =
                            *x.get(i, k).unwrap() + alpha * *p_hat.get(i, 0).unwrap();
                    }
                    break;
                }

                // s_hat = M^-1 * s
                let s_hat = self.preconditioner.solve(&s)?;

                // t = A * s_hat
                let t = matrix.mul_dense(&s_hat)?;

                // omega = (t^T * s) / (t^T * t)
                omega = self.dot_product(&t, &s) / self.dot_product(&t, &t);

                // x = x + alpha * p_hat + omega * s_hat
                for i in 0..n {
                    *x.get_mut(i, k).unwrap() = *x.get(i, k).unwrap()
                        + alpha * *p_hat.get(i, 0).unwrap()
                        + omega * *s_hat.get(i, 0).unwrap();
                }

                // r = s - omega * t
                for i in 0..n {
                    *r.get_mut(i, 0).unwrap() =
                        *s.get(i, 0).unwrap() - omega * *t.get(i, 0).unwrap();
                }

                rho = rho_next;

                if r.norm_f64() / b_norm < self.tolerance {
                    break;
                }

                if omega.abs().to_f64() < 1e-20 {
                    return Err("BiCGSTAB stalled (omega)".to_string());
                }
            }
        }

        Ok(x)
    }

    fn dot_product<S1: Storage<T>, S2: Storage<T>>(
        &self,
        a: &Matrix<T, S1>,
        b: &Matrix<T, S2>,
    ) -> T {
        let mut sum = T::default();
        for i in 0..a.rows() {
            sum += a.get(i, 0).unwrap().conj() * *b.get(i, 0).unwrap();
        }
        sum
    }
}

// Reuse MatrixNorm trait (local for now, maybe move to library if needed)
trait MatrixNorm<T: Scalar, S: Storage<T>> {
    fn norm_f64(&self) -> f64;
}

impl<T: Scalar, S: Storage<T>> MatrixNorm<T, S> for Matrix<T, S> {
    fn norm_f64(&self) -> f64 {
        let mut sum_sq = 0.0;
        for j in 0..self.cols() {
            for i in 0..self.rows() {
                sum_sq += self.get(i, j).unwrap().to_f64().powi(2);
            }
        }
        sum_sq.sqrt()
    }
}
