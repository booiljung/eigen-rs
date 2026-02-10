//! Tridiagonal decomposition of a selfadjoint matrix.
//! A = Q * T * Q^T

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::{DynamicStorage, Storage};

/// Tridiagonal decomposition of a selfadjoint matrix.
pub struct Tridiagonalization<T: Scalar, S: Storage<T>> {
    packed_matrix: Matrix<T, DynamicStorage<T>>,
    h_coeffs: Vec<T>,
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar, S: Storage<T>> Tridiagonalization<T, S> {
    /// Computes the tridiagonal decomposition of the given symmetric matrix.
    /// Only the lower triangular part of the matrix is used.
    pub fn new(matrix: &Matrix<T, S>) -> Result<Self, String> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err("Tridiagonalization requires a square matrix".to_string());
        }

        let mut packed_matrix = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n)?;
        packed_matrix.assign(matrix)?;

        let mut h_coeffs = vec![T::default(); if n > 1 { n - 1 } else { 1 }];

        if n > 1 {
            Self::tridiagonalization_inplace(&mut packed_matrix, &mut h_coeffs);
        }

        Ok(Self {
            packed_matrix,
            h_coeffs,
            _phantom: std::marker::PhantomData,
        })
    }

    fn tridiagonalization_inplace(mat_a: &mut Matrix<T, DynamicStorage<T>>, h_coeffs: &mut [T]) {
        let n = mat_a.rows();

        for (i, h_coeff) in h_coeffs.iter_mut().enumerate().take(n - 1) {
            // 1. Compute Householder reflection for column i starting from i+1
            let mut norm_sq = T::default();
            for k in i + 1..n {
                let val = *mat_a.get(k, i).unwrap();
                norm_sq += val * val;
            }
            let norm = norm_sq.sqrt();

            if norm != T::default() {
                let v0 = *mat_a.get(i + 1, i).unwrap();
                let beta = if v0 >= T::default() { -norm } else { norm };

                let v0_minus_beta = v0 - beta;
                let inv_v0_minus_beta = v0_minus_beta.recip();

                // Scale Householder vector: v[0] becomes 1, rest stored in mat_a
                for k in i + 2..n {
                    *mat_a.get_mut(k, i).unwrap() *= inv_v0_minus_beta;
                }

                // Householder coefficient h (tau) for the scaled vector
                let h = (v0.abs() + norm) / norm;
                *h_coeff = h;

                // 2. Similarity transformation: A = H A H^T
                // Compute p = (h A v) - (h/2 * v^T (h A v)) v
                let remaining_size = n - i - 1;
                let mut w = vec![T::default(); remaining_size];
                for (row, val) in w.iter_mut().enumerate().take(remaining_size) {
                    let mut dot = T::default();
                    for col in 0..remaining_size {
                        let r = row + i + 1;
                        let c = col + i + 1;
                        let val = if r >= c {
                            *mat_a.get(r, c).unwrap()
                        } else {
                            *mat_a.get(c, r).unwrap()
                        };

                        // v[0] is 1
                        let v_col = if col == 0 {
                            T::from_f64(1.0)
                        } else {
                            *mat_a.get(c, i).unwrap()
                        };
                        dot += val * v_col;
                    }
                    *val = h * dot;
                }

                let mut vt_w = T::default();
                for (k, val) in w.iter().enumerate().take(remaining_size) {
                    let v_k = if k == 0 {
                        T::from_f64(1.0)
                    } else {
                        *mat_a.get(k + i + 1, i).unwrap()
                    };
                    vt_w += v_k * *val;
                }
                let scale = h * vt_w * T::from_f64(0.5);

                for (k, val) in w.iter_mut().enumerate().take(remaining_size) {
                    let v_k = if k == 0 {
                        T::from_f64(1.0)
                    } else {
                        *mat_a.get(k + i + 1, i).unwrap()
                    };
                    *val -= scale * v_k;
                }

                for col in 0..remaining_size {
                    for row in col..remaining_size {
                        let v_row = if row == 0 {
                            T::from_f64(1.0)
                        } else {
                            *mat_a.get(row + i + 1, i).unwrap()
                        };
                        let p_row = w[row];
                        let v_col = if col == 0 {
                            T::from_f64(1.0)
                        } else {
                            *mat_a.get(col + i + 1, i).unwrap()
                        };
                        let p_col = w[col];

                        *mat_a.get_mut(row + i + 1, col + i + 1).unwrap() -=
                            v_row * p_col + p_row * v_col;
                    }
                }

                // Restore beta as the tridiagonal sub-diagonal element
                *mat_a.get_mut(i + 1, i).unwrap() = beta;
            } else {
                *h_coeff = T::default();
            }
        }
    }

    /// Returns the tridiagonal matrix T.
    pub fn matrix_t(&self) -> Matrix<T, DynamicStorage<T>> {
        let n = self.packed_matrix.rows();
        let mut t = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n).unwrap();

        for j in 0..n {
            for i in 0..n {
                if i == j || i == j + 1 || j == i + 1 {
                    // Always read from the lower triangular part
                    let r = std::cmp::max(i, j);
                    let c = std::cmp::min(i, j);
                    *t.get_mut(i, j).unwrap() = *self.packed_matrix.get(r, c).unwrap();
                } else {
                    *t.get_mut(i, j).unwrap() = T::default();
                }
            }
        }
        t
    }

    /// Returns the orthogonal matrix Q.
    pub fn matrix_q(&self) -> Matrix<T, DynamicStorage<T>> {
        let n = self.packed_matrix.rows();
        let mut q = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n).unwrap();

        for i in 0..n {
            for j in 0..n {
                *q.get_mut(i, j).unwrap() = if i == j {
                    T::from_usize(1)
                } else {
                    T::default()
                };
            }
        }

        for i in (0..n - 1).rev() {
            let h = self.h_coeffs[i];
            if h != T::default() {
                for j in i + 1..n {
                    let mut dot = *q.get(i + 1, j).unwrap();
                    for k in i + 2..n {
                        dot += (*self.packed_matrix.get(k, i).unwrap()) * (*q.get(k, j).unwrap());
                    }

                    let factor = h * dot;
                    *q.get_mut(i + 1, j).unwrap() -= factor;
                    for k in i + 2..n {
                        let vk = *self.packed_matrix.get(k, i).unwrap();
                        *q.get_mut(k, j).unwrap() -= factor * vk;
                    }
                }
            }
        }
        q
    }
}
