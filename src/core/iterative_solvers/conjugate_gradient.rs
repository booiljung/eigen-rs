use crate::core::iterative_solvers::traits::IterativeSolver;
use crate::core::matrix::{Matrix, MatrixX};
use crate::core::scalar::Scalar;
use crate::core::storage::{DynamicStorage, Storage};
use std::marker::PhantomData;

/// A Conjugate Gradient solver for self-adjoint positive definite problems.
pub struct ConjugateGradient<T: Scalar, M> {
    max_iterations: usize,
    tolerance: T,
    iterations: std::cell::RefCell<usize>,
    error: std::cell::RefCell<T>,
    matrix: Option<M>, // Store the matrix A
    _phantom: PhantomData<T>,
}

impl<T: Scalar, M: Clone> ConjugateGradient<T, M> {
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            // Tolerance default is type dependent, using 1e-12 roughly
            tolerance: T::from_f64(1e-12),
            iterations: std::cell::RefCell::new(0),
            error: std::cell::RefCell::new(T::zero()),
            matrix: None,
            _phantom: PhantomData,
        }
    }

    /// Sets the maximum number of iterations.
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Sets the tolerance for the stopping criteria.
    pub fn with_tolerance(mut self, tolerance: T) -> Self {
        self.tolerance = tolerance;
        self
    }
}

// We need a trait bound on M that allows Matrix-Vector multiplication.
// For now, let's assume M is &Matrix<T, S> or similar. 
// Ideally M implements specific Mul traits.

// Hardcoding for M = Matrix<T, S> for now to demonstrate.
impl<T: Scalar<Real = T> + PartialOrd + 'static, S: Storage<T> + 'static> IterativeSolver<T, DynamicStorage<T>> 
    for ConjugateGradient<T, Matrix<T, S>> 
    where S: Clone // Requirement to store Matrix
{
    type MatrixType = Matrix<T, S>;

    fn compute(&mut self, matrix: &Self::MatrixType) {
        self.matrix = Some(matrix.clone());
    }

    fn solve<S2>(&self, b: &Matrix<T, S2>) -> Matrix<T, DynamicStorage<T>>
    where
        S2: Storage<T>,
    {
        let rows = b.rows();
        let cols = b.cols();
        let mut x0 = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, cols).unwrap();
        x0.set_zero();
        self.solve_with_guess(b, &x0)
    }

    fn solve_with_guess<S2, S3>(
        &self,
        b: &Matrix<T, S2>,
        x0: &Matrix<T, S3>,
    ) -> Matrix<T, DynamicStorage<T>>
    where
        S2: Storage<T>,
        S3: Storage<T>,
    {
        let a = self.matrix.as_ref().expect("Matrix not initialized. Call compute() first.");
        let rows = b.rows();
        let cols = b.cols();
        
        // x = x0
        let mut x = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        x.assign(x0).unwrap();

        // R = b - A * x
        // 1. Evaluate Ax
        let mut ax = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        ax.assign(&(a * &x)).unwrap();
        
        // 2. Evaluate R = b - Ax
        let mut r = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        r.assign(&(b - &ax)).unwrap();
        
        // P = R
        let mut p = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        p.assign(&r).unwrap();
        
        let mut rs_old = r.squared_norm();
        let tol_sq = self.tolerance * self.tolerance;

        *self.iterations.borrow_mut() = 0;
        *self.error.borrow_mut() = rs_old.sqrt();
        
        if rs_old < tol_sq {
             return x;
        }

        // Pre-allocate Ap
        let mut ap = MatrixX::<T>::new_dynamic(rows, cols).unwrap();

        for i in 0..self.max_iterations {
            *self.iterations.borrow_mut() = i + 1;
            
            // Ap = A * p
            ap.assign(&(a * &p)).unwrap();
            
            // alpha = rs_old / (p . Ap)
            let p_ap = p.dot(&ap);
            
            if p_ap.abs() < T::from_f64(1e-30) {
                 break; // Avoid division by zero
            }

            let alpha = rs_old / p_ap;
            
            // Vector updates:
            // x = x + alpha * p
            // r = r - alpha * Ap
            
            // Fix Borrow Checker: Get size first
            let size = x.size(); 
            
            let x_data = x.storage_mut().data_mut();
            let p_data = p.storage().data();
            let r_data = r.storage_mut().data_mut();
            let ap_data = ap.storage().data();

            for k in 0..size {
                x_data[k] += alpha * p_data[k];
                r_data[k] -= alpha * ap_data[k];
            }
            
            let rs_new = r.squared_norm();
            *self.error.borrow_mut() = rs_new.sqrt();

            if rs_new < tol_sq {
                break;
            }
            
            let beta = rs_new / rs_old;
            
            // p = r + beta * p
            let p_data_mut = p.storage_mut().data_mut();
            let r_data_read = r.storage().data();
            
            // No need to re-fetch size, it's consistent
            for k in 0..size {
                p_data_mut[k] = r_data_read[k] + beta * p_data_mut[k];
            }
            
            rs_old = rs_new;
        }
        
        x
    }

    fn iterations(&self) -> usize {
        *self.iterations.borrow()
    }

    fn error(&self) -> T {
        *self.error.borrow()
    }
}

use crate::core::sparse::SparseMatrix;

impl<T: Scalar<Real = T> + PartialOrd + 'static> IterativeSolver<T, DynamicStorage<T>> 
    for ConjugateGradient<T, SparseMatrix<T>> 
{
    type MatrixType = SparseMatrix<T>;

    fn compute(&mut self, matrix: &Self::MatrixType) {
        self.matrix = Some(matrix.clone());
    }

    fn solve<S2>(&self, b: &Matrix<T, S2>) -> Matrix<T, DynamicStorage<T>>
    where
        S2: Storage<T>,
    {
        let rows = b.rows();
        let cols = b.cols();
        let mut x0 = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, cols).unwrap();
        x0.set_zero();
        self.solve_with_guess(b, &x0)
    }

    fn solve_with_guess<S2, S3>(
        &self,
        b: &Matrix<T, S2>,
        x0: &Matrix<T, S3>,
    ) -> Matrix<T, DynamicStorage<T>>
    where
        S2: Storage<T>,
        S3: Storage<T>,
    {
        let a = self.matrix.as_ref().expect("Matrix not initialized. Call compute() first.");
        let rows = b.rows();
        let cols = b.cols();
        
        // x = x0
        let mut x = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        x.assign(x0).unwrap();

        // R = b - A * x
        // 1. Evaluate Ax = A * x
        // For SparseMatrix, use mul_dense_into
        let mut ax = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        ax.set_zero();
        a.mul_dense_into(&x, &mut ax).expect("Sparse-Dense multiplication failed");
        
        // 2. Evaluate R = b - Ax
        let mut r = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        r.assign(&(b - &ax)).unwrap();
        
        // P = R
        let mut p = MatrixX::<T>::new_dynamic(rows, cols).unwrap();
        p.assign(&r).unwrap();
        
        let mut rs_old = r.squared_norm();
        let tol_sq = self.tolerance * self.tolerance;

        *self.iterations.borrow_mut() = 0;
        *self.error.borrow_mut() = rs_old.sqrt();
        
        if rs_old < tol_sq {
             return x;
        }

        // Pre-allocate Ap
        let mut ap = MatrixX::<T>::new_dynamic(rows, cols).unwrap();

        for i in 0..self.max_iterations {
            *self.iterations.borrow_mut() = i + 1;
            
            // Ap = A * p
            a.mul_dense_into(&p, &mut ap).expect("Sparse-Dense multiplication failed");
            
            // alpha = rs_old / (p . Ap)
            let p_ap = p.dot(&ap);
            
            if p_ap.abs() < T::from_f64(1e-30) {
                 break; // Avoid division by zero
            }

            let alpha = rs_old / p_ap;
            
            // Vector updates:
            // x = x + alpha * p
            // r = r - alpha * Ap
            
            let size = x.size();
            let x_data = x.storage_mut().data_mut();
            let p_data = p.storage().data();
            let r_data = r.storage_mut().data_mut();
            let ap_data = ap.storage().data();

            for k in 0..size {
                x_data[k] += alpha * p_data[k];
                r_data[k] -= alpha * ap_data[k];
            }
            
            let rs_new = r.squared_norm();
            *self.error.borrow_mut() = rs_new.sqrt();

            if rs_new < tol_sq {
                break;
            }
            
            let beta = rs_new / rs_old;
            
            // p = r + beta * p
            let p_data_mut = p.storage_mut().data_mut();
            let r_data_read = r.storage().data();
            
            for k in 0..size {
                p_data_mut[k] = r_data_read[k] + beta * p_data_mut[k];
            }
            
            rs_old = rs_new;
        }
        
        x
    }
    
    fn iterations(&self) -> usize {
        *self.iterations.borrow()
    }

    fn error(&self) -> T {
        *self.error.borrow()
    }
}
