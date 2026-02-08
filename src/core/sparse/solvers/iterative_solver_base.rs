//! Base trait and common implementations for preconditioners used by iterative solvers.

use crate::core::scalar::Scalar;
use crate::core::sparse::sparse_matrix::SparseMatrix;
use crate::core::matrix::Matrix;
use crate::core::storage::{Storage, DynamicStorage};
use crate::core::sparse::iterators::InnerIterator;

/// Trait for preconditioners that can be used with iterative solvers.
/// A preconditioner M is an approximation of A such that M^-1 * A has a better condition number.
pub trait Preconditioner<T: Scalar> {
    /// Checks if the preconditioner is ready for use.
    fn is_initialized(&self) -> bool { true }

    /// Sets up the preconditioner for the given matrix A.
    fn compute(&mut self, matrix: &SparseMatrix<T>) -> Result<(), String>;

    /// Solves the preconditioning system M * x = b.
    fn solve<S: Storage<T>>(&self, b: &Matrix<T, S>) -> Result<Matrix<T, DynamicStorage<T>>, String>;
}

/// Identity preconditioner (M = I). Does nothing.
#[derive(Default)]
pub struct IdentityPreconditioner;

impl IdentityPreconditioner {
    pub fn new() -> Self {
        Self
    }
}

impl<T: Scalar> Preconditioner<T> for IdentityPreconditioner {
    fn compute(&mut self, _matrix: &SparseMatrix<T>) -> Result<(), String> {
        Ok(())
    }

    fn solve<S: Storage<T>>(&self, b: &Matrix<T, S>) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        // Return a copy of b
        let mut x = Matrix::<T, DynamicStorage<T>>::new_dynamic(b.rows(), b.cols())?;
        for j in 0..b.cols() {
            for i in 0..b.rows() {
                *x.get_mut(i, j).unwrap() = *b.get(i, j).unwrap();
            }
        }
        Ok(x)
    }
}

/// Diagonal (Jacobi) preconditioner (M = diag(A)).
#[derive(Default)]
pub struct DiagonalPreconditioner<T: Scalar> {
    inv_diag: Vec<T>,
    is_initialized: bool,
}

impl<T: Scalar> DiagonalPreconditioner<T> {
    pub fn new() -> Self {
        Self {
            inv_diag: Vec::new(),
            is_initialized: false,
        }
    }
}

impl<T: Scalar> Preconditioner<T> for DiagonalPreconditioner<T> {
    fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    fn compute(&mut self, matrix: &SparseMatrix<T>) -> Result<(), String> {
        let n = matrix.rows();
        self.inv_diag = vec![T::default(); n];
        
        for j in 0..n {
            let mut it = InnerIterator::new(matrix, j);
            let mut found = false;
            while it.is_valid() {
                if it.row() == j {
                    let val = it.value();
                    if val.abs().to_f64() < 1e-16 {
                        return Err(format!("Zero or near-zero diagonal element at index {}", j));
                    }
                    self.inv_diag[j] = T::from_f64(1.0) / val;
                    found = true;
                    break;
                }
                it.next();
            }
            if !found {
                return Err(format!("Missing diagonal element at index {}", j));
            }
        }
        
        self.is_initialized = true;
        Ok(())
    }

    fn solve<S: Storage<T>>(&self, b: &Matrix<T, S>) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        if !self.is_initialized {
            return Err("Preconditioner not initialized".to_string());
        }
        let n = b.rows();
        let mut x = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, b.cols())?;
        for j in 0..b.cols() {
            for i in 0..n {
                *x.get_mut(i, j).unwrap() = *b.get(i, j).unwrap() * self.inv_diag[i];
            }
        }
        Ok(x)
    }
}
