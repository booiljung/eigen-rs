//! Conjugate Gradient (CG) solver for symmetric positive-definite (SPD) matrices.

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::sparse::solvers::iterative_solver_base::{IdentityPreconditioner, Preconditioner};
use crate::core::sparse::sparse_matrix::SparseMatrix;
use crate::core::storage::{DynamicStorage, Storage};

/// Conjugate Gradient solver.
/// Solves Ax = b for symmetric positive-definite matrices A.
pub struct ConjugateGradient<T: Scalar, P: Preconditioner<T> = IdentityPreconditioner> {
    preconditioner: P,
    max_iterations: usize,
    tolerance: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Default for ConjugateGradient<T, IdentityPreconditioner> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> ConjugateGradient<T, IdentityPreconditioner> {
    /// Creates a new CG solver with default Identity preconditioner.
    pub fn new() -> Self {
        Self {
            preconditioner: IdentityPreconditioner::new(),
            max_iterations: 1000,
            tolerance: 1e-10,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar, P: Preconditioner<T>> ConjugateGradient<T, P> {
    /// Creates a new CG solver with a custom preconditioner.
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

        self.preconditioner.compute(matrix)?;

        let mut x = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, b.cols())?; // Initial x = 0

        for k in 0..b.cols() {
            // b_vec = b[:, k]
            let mut b_vec = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, 1)?;
            for i in 0..n {
                *b_vec.get_mut(i, 0).unwrap() = *b.get(i, k).unwrap();
            }

            // r = b - A*x (since x=0, r=b)
            let mut r = b_vec.clone();

            // z = M^-1 * r
            let mut z = self.preconditioner.solve(&r)?;

            // p = z
            let mut p = z.clone();

            // rho = r^T * z
            let mut rho = self.dot_product(&r, &z);
            let b_norm = r.norm_f64();
            if b_norm < 1e-16 {
                continue; // x = 0 is a good solution
            }

            for _iter in 0..self.max_iterations {
                // v = A * p
                let v = matrix.mul_dense(&p)?;

                // alpha = rho / (p^T * v)
                let p_dot_v = self.dot_product(&p, &v);
                let alpha = rho / p_dot_v;

                // x = x + alpha * p
                for i in 0..n {
                    *x.get_mut(i, k).unwrap() =
                        *x.get(i, k).unwrap() + alpha * *p.get(i, 0).unwrap();
                }

                // r = r - alpha * v
                for i in 0..n {
                    *r.get_mut(i, 0).unwrap() =
                        *r.get(i, 0).unwrap() - alpha * *v.get(i, 0).unwrap();
                }

                let err = r.norm_f64() / b_norm;
                if err < self.tolerance {
                    break;
                }

                // z = M^-1 * r
                z = self.preconditioner.solve(&r)?;

                // rho_next = r^T * z
                let rho_next = self.dot_product(&r, &z);

                // beta = rho_next / rho
                let beta = rho_next / rho;

                // p = z + beta * p
                for i in 0..n {
                    *p.get_mut(i, 0).unwrap() =
                        *z.get(i, 0).unwrap() + beta * *p.get(i, 0).unwrap();
                }

                rho = rho_next;
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

// Matrix extension to help with norms (already exists for dense usually, but let's make sure or add one)
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
