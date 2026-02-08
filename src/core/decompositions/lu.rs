//! Partial Pivoting LU decomposition (PA = LU).

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::DynamicStorage;
use crate::core::storage::Storage;

/// Result of a Partial Pivoting LU decomposition.
pub struct PartialPivLU<T: Scalar, S: Storage<T>> {
    lu: Matrix<T, DynamicStorage<T>>,
    p: Vec<usize>, // Permutation vector
    det_p: T,      // Determinant of permutation matrix (+1 or -1)
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar, S: Storage<T>> PartialPivLU<T, S> {
    /// Computes the LU decomposition of the given square matrix.
    pub fn new(matrix: &Matrix<T, S>) -> Result<Self, String> {
        let rows = matrix.rows();
        let cols = matrix.cols();
        if rows != cols {
            return Err("LU decomposition requires a square matrix".to_string());
        }

        let mut lu = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, cols)?;
        lu.assign(matrix)?;

        let mut p: Vec<usize> = (0..rows).collect();
        let mut det_p = T::from_usize(1);
        let neg_one = T::from_usize(0) - T::from_usize(1);

        for k in 0..rows {
            // Partial pivoting: find max in column k
            let mut max_val = T::from_usize(0);
            let mut imax = k;

            for i in k..rows {
                let val = *lu.get(i, k).unwrap();
                let abs_val = val.abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    imax = i;
                }
            }

            if imax != k {
                // Swap rows imax and k in LU
                // Note: We need a row swap method in Matrix.
                // For now, let's do it manually.
                for j in 0..cols {
                    let tmp = *lu.get(k, j).unwrap();
                    let val_imax = *lu.get(imax, j).unwrap();
                    *lu.get_mut(k, j).unwrap() = val_imax;
                    *lu.get_mut(imax, j).unwrap() = tmp;
                }
                // Update permutation
                p.swap(k, imax);
                det_p *= neg_one;
            }

            let pivot = *lu.get(k, k).unwrap();
            if pivot != T::from_usize(0) {
                for i in k + 1..rows {
                    let factor = *lu.get(i, k).unwrap() / pivot;
                    *lu.get_mut(i, k).unwrap() = factor;
                    for j in k + 1..cols {
                        let sub = factor * (*lu.get(k, j).unwrap());
                        *lu.get_mut(i, j).unwrap() -= sub;
                    }
                }
            }
        }

        Ok(Self {
            lu,
            p,
            det_p,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Returns the determinant of the original matrix.
    pub fn determinant(&self) -> T {
        let mut det = self.det_p;
        for i in 0..self.lu.rows() {
            det *= *self.lu.get(i, i).unwrap();
        }
        det
    }

    /// Solves Ax = b for x.
    pub fn solve<S2: Storage<T>>(
        &self,
        b: &Matrix<T, S2>,
    ) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        let rows = self.lu.rows();
        if b.rows() != rows {
            return Err("Dimension mismatch in LU solve".to_string());
        }

        let b0_cols = b.cols();
        let mut x = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, b0_cols)?;

        // 1. Apply permutation P to b (x = Pb)
        for i in 0..rows {
            let pi = self.p[i];
            for j in 0..b0_cols {
                *x.get_mut(i, j).unwrap() = *b.get(pi, j).unwrap();
            }
        }

        // 2. Forward substitution for L (Ly = Pb)
        // L is unit lower triangular (diagonal elements are 1)
        for k in 0..rows {
            for i in k + 1..rows {
                let factor = *self.lu.get(i, k).unwrap();
                for j in 0..b0_cols {
                    let sub = factor * (*x.get(k, j).unwrap());
                    *x.get_mut(i, j).unwrap() -= sub;
                }
            }
        }

        // 3. Backward substitution for U (Ux = y)
        for k in (0..rows).rev() {
            let pivot = *self.lu.get(k, k).unwrap();
            if pivot == T::from_usize(0) {
                return Err("Matrix is singular, cannot solve".to_string());
            }
            for j in 0..b0_cols {
                *x.get_mut(k, j).unwrap() /= pivot;
            }
            for i in 0..k {
                let factor = *self.lu.get(i, k).unwrap();
                for j in 0..b0_cols {
                    let sub = factor * (*x.get(k, j).unwrap());
                    *x.get_mut(i, j).unwrap() -= sub;
                }
            }
        }

        Ok(x)
    }
}
