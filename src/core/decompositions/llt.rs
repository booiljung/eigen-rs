//! LLT Cholesky decomposition ($A = LL^T$).
//! Best suited for symmetric/Hermitian positive definite matrices.

use crate::core::matrix::Matrix;
use crate::core::storage::Storage;
use crate::core::scalar::Scalar;
use crate::core::storage::DynamicStorage;

/// Result of an LLT Cholesky decomposition.
pub struct LLT<T: Scalar, S: Storage<T>> {
    l: Matrix<T, DynamicStorage<T>>,
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar + 'static, S: Storage<T> + 'static> LLT<T, S> {
    /// Computes the LLT decomposition of the given matrix.
    /// The matrix MUST be symmetric positive definite.
    pub fn new(matrix: &Matrix<T, S>) -> Result<Self, String> {
        let rows = matrix.rows();
        let cols = matrix.cols();
        if rows != cols {
            return Err("LLT decomposition requires a square matrix".to_string());
        }

        let mut l = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, cols)?;
        l.assign(matrix)?;

        for j in 0..rows {
            let mut s = T::from_usize(0);
            for k in 0..j {
                let l_jk = *l.get(j, k).unwrap();
                s += l_jk * l_jk;
            }
            
            let diag = *l.get(j, j).unwrap() - s;
            if diag <= T::from_usize(0) {
                return Err("Matrix is not positive definite".to_string());
            }
            
            let l_jj = diag.sqrt();
            *l.get_mut(j, j).unwrap() = l_jj;

            for i in j + 1..rows {
                let mut s = T::from_usize(0);
                for k in 0..j {
                    s += (*l.get(i, k).unwrap()) * (*l.get(j, k).unwrap());
                }
                *l.get_mut(i, j).unwrap() = (*l.get(i, j).unwrap() - s) / l_jj;
                // Zero out the upper triangle
                *l.get_mut(j, i).unwrap() = T::from_usize(0);
            }
        }

        Ok(Self {
            l,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Returns the factor L.
    pub fn matrix_l(&self) -> &Matrix<T, DynamicStorage<T>> {
        &self.l
    }

    /// Solves Ax = b for x using the LLT decomposition.
    pub fn solve<S2: Storage<T> + 'static>(&self, b: &Matrix<T, S2>) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        let rows = self.l.rows();
        if b.rows() != rows {
            return Err("Dimension mismatch in LLT solve".to_string());
        }

        let b_cols = b.cols();
        let mut x = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, b_cols)?;
        x.assign(b)?;

        // Solve Ly = b (Forward substitution)
        // L is lower triangular, but NOT unit diagonal in LLT.
        for i in 0..rows {
            let l_ii = *self.l.get(i, i).unwrap();
            for j in 0..b_cols {
                let mut val = *x.get(i, j).unwrap();
                for k in 0..i {
                    val -= (*self.l.get(i, k).unwrap()) * (*x.get(k, j).unwrap());
                }
                *x.get_mut(i, j).unwrap() = val / l_ii;
            }
        }
        
        // Solve L^T x = y (Backward substitution)
        // L^T is upper triangular.
        for i in (0..rows).rev() {
            let l_ii = *self.l.get(i, i).unwrap();
            for j in 0..b_cols {
                let mut val = *x.get(i, j).unwrap();
                for k in i + 1..rows {
                    val -= (*self.l.get(k, i).unwrap()) * (*x.get(k, j).unwrap());
                }
                *x.get_mut(i, j).unwrap() = val / l_ii;
            }
        }

        Ok(x)
    }
}
