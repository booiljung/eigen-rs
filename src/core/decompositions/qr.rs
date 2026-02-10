//! Householder QR decomposition (A = QR).

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::{DynamicStorage, Storage};

/// Result of a Householder QR decomposition.
///
/// The result is stored in a compact way compatible with LAPACK/Eigen:
/// - The upper triangular part of `qr` is the matrix R.
/// - The strict lower triangular part of `qr` contains the Householder vectors v.
/// - `h_coeffs` contains the Householder coefficients tau.
pub struct HouseholderQR<T: Scalar, S: Storage<T>> {
    qr: Matrix<T, DynamicStorage<T>>,
    h_coeffs: Vec<T>,
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar, S: Storage<T>> HouseholderQR<T, S> {
    /// Computes the Householder QR decomposition of the given matrix.
    pub fn new(matrix: &Matrix<T, S>) -> Result<Self, String> {
        let rows = matrix.rows();
        let cols = matrix.cols();
        let size = std::cmp::min(rows, cols);

        let mut qr = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, cols)?;
        qr.assign(matrix)?;

        let mut h_coeffs = vec![T::default(); size];

        for (k, h_coeff) in h_coeffs.iter_mut().enumerate().take(size) {
            // 1. Compute norm of the tail of column k
            let mut norm_sq = T::default();
            for i in k..rows {
                let val = *qr.get(i, k).unwrap();
                norm_sq += val * val;
            }
            let norm = norm_sq.sqrt();

            if norm != T::default() {
                let v0 = *qr.get(k, k).unwrap();
                let sigma = if v0 >= T::default() {
                    norm
                } else {
                    T::default() - norm
                };

                let v0_new = v0 + sigma;
                // Correct tau for normalized v (where v[0] = 1)
                let tau = v0_new / sigma;
                *h_coeff = tau;

                // Scale remaining elements of column k by (v0 + sigma)^-1
                let inv_v0_new = v0_new.recip();
                for i in k + 1..rows {
                    *qr.get_mut(i, k).unwrap() *= inv_v0_new;
                }

                // R[k, k] = -sigma
                *qr.get_mut(k, k).unwrap() = T::default() - sigma;

                // Apply reflection to remaining columns from the left
                for j in k + 1..cols {
                    let mut dot = *qr.get(k, j).unwrap();
                    for i in k + 1..rows {
                        dot += (*qr.get(i, k).unwrap()) * (*qr.get(i, j).unwrap());
                    }

                    let factor = tau * dot;
                    *qr.get_mut(k, j).unwrap() -= factor;
                    for i in k + 1..rows {
                        let vi = *qr.get(i, k).unwrap();
                        *qr.get_mut(i, j).unwrap() -= factor * vi;
                    }
                }
            } else {
                *h_coeff = T::default();
            }
        }

        Ok(Self {
            qr,
            h_coeffs,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Returns the upper triangular matrix R.
    pub fn matrix_r(&self) -> Matrix<T, DynamicStorage<T>> {
        let rows = self.qr.rows();
        let cols = self.qr.cols();
        let mut r = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, cols).unwrap();

        for i in 0..rows {
            for j in 0..cols {
                if j >= i {
                    *r.get_mut(i, j).unwrap() = *self.qr.get(i, j).unwrap();
                } else {
                    *r.get_mut(i, j).unwrap() = T::default();
                }
            }
        }
        r
    }

    /// Returns the orthogonal matrix Q (Full).
    pub fn matrix_q(&self) -> Matrix<T, DynamicStorage<T>> {
        let rows = self.qr.rows();
        let cols = self.qr.cols();
        let size = std::cmp::min(rows, cols);
        let mut q = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, rows).unwrap();

        // Initialize Q as Identity
        for i in 0..rows {
            for j in 0..rows {
                *q.get_mut(i, j).unwrap() = if i == j {
                    T::from_usize(1)
                } else {
                    T::default()
                };
            }
        }

        // Q = H1 * H2 * ... * Hn
        // We apply them in reverse order to build Q from I.
        for k in (0..size).rev() {
            let tau = self.h_coeffs[k];
            if tau != T::default() {
                // Apply Hk = I - tau * v * v^T to Q from the left
                // Q[k:rows, :] = (I - tau * v * v^T) * Q[k:rows, :]
                for j in 0..rows {
                    // dot = v^T * Q[k:rows, j] = Q[k, j] + sum_{i=k+1}^rows v_i * Q[i, j]
                    let mut dot = *q.get(k, j).unwrap();
                    for i in k + 1..rows {
                        dot += (*self.qr.get(i, k).unwrap()) * (*q.get(i, j).unwrap());
                    }

                    let factor = tau * dot;
                    *q.get_mut(k, j).unwrap() -= factor;
                    for i in k + 1..rows {
                        let vi = *self.qr.get(i, k).unwrap();
                        *q.get_mut(i, j).unwrap() -= factor * vi;
                    }
                }
            }
        }
        q
    }

    /// Solves the system Ax = b using the QR decomposition.
    pub fn solve<S2: Storage<T>>(
        &self,
        b: &Matrix<T, S2>,
    ) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        let rows = self.qr.rows();
        let cols = self.qr.cols();
        if b.rows() != rows {
            return Err(format!(
                "Dimension mismatch in QR solve: b.rows() {} != A.rows() {}",
                b.rows(),
                rows
            ));
        }

        let mut x = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, b.cols())?;
        x.assign(b)?;

        let size = self.h_coeffs.len();

        // 1. Apply Q^T to b: b' = Q^T * b = H_n * ... * H_1 * b
        for k in 0..size {
            let tau = self.h_coeffs[k];
            if tau != T::default() {
                for j in 0..b.cols() {
                    // dot = v^T * x[k:rows, j] = x[k, j] + sum_{i=k+1}^rows v_i * x[i, j]
                    let mut dot = *x.get(k, j).unwrap();
                    for i in k + 1..rows {
                        dot += (*self.qr.get(i, k).unwrap()) * (*x.get(i, j).unwrap());
                    }

                    let factor = tau * dot;
                    *x.get_mut(k, j).unwrap() -= factor;
                    for i in k + 1..rows {
                        let vi = *self.qr.get(i, k).unwrap();
                        *x.get_mut(i, j).unwrap() -= factor * vi;
                    }
                }
            }
        }

        // 2. Solve Rx = b' using back-substitution
        // R is stored in the upper triangle of self.qr
        for j in 0..b.cols() {
            for i in (0..size).rev() {
                let diag = *self.qr.get(i, i).unwrap();
                if diag.abs().to_f64() < 1e-18 {
                    return Err(format!(
                        "QR solve failed: singular matrix (zero diagonal at {})",
                        i
                    ));
                }

                let mut sum = T::default();
                for k in i + 1..cols {
                    sum += (*self.qr.get(i, k).unwrap()) * (*x.get(k, j).unwrap());
                }

                let val = (*x.get(i, j).unwrap() - sum) / diag;
                *x.get_mut(i, j).unwrap() = val;
            }
        }

        // The result x should have dimensions (cols, b.cols())
        if cols < rows {
            // Trim x if A was tall
            let mut x_final = Matrix::<T, DynamicStorage<T>>::new_dynamic(cols, b.cols())?;
            for j in 0..b.cols() {
                for i in 0..cols {
                    *x_final.get_mut(i, j).unwrap() = *x.get(i, j).unwrap();
                }
            }
            Ok(x_final)
        } else {
            Ok(x)
        }
    }
}
