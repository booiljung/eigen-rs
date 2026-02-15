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
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("fma") {
                        Self::apply_householder_avx(&mut qr, k, tau, rows, cols);
                        continue;
                    }
                }

                // Scalar Fallback
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

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn apply_householder_avx(
        qr: &mut Matrix<T, DynamicStorage<T>>,
        k: usize,
        tau: T,
        rows: usize,
        cols: usize,
    ) {
        use std::any::TypeId;
        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f32>() {
            use std::arch::x86_64::*;
            // f32 path
            unsafe {
                let start_ptr = qr.storage().data().as_ptr() as *const f32;
                let mut_ptr = qr.storage_mut().data_mut().as_mut_ptr() as *mut f32;
                let rows_stride = rows;

                // Pointer to v (k-th column)
                let v_ptr = start_ptr.add(k * rows_stride);
                let tau_f32 = *(&tau as *const T as *const f32);

                let mut j = k + 1;
                while j + 4 <= cols {
                    let c0_ptr = mut_ptr.add(j * rows_stride);
                    let c1_ptr = mut_ptr.add((j + 1) * rows_stride);
                    let c2_ptr = mut_ptr.add((j + 2) * rows_stride);
                    let c3_ptr = mut_ptr.add((j + 3) * rows_stride);

                    let mut dot0 = *c0_ptr.add(k);
                    let mut dot1 = *c1_ptr.add(k);
                    let mut dot2 = *c2_ptr.add(k);
                    let mut dot3 = *c3_ptr.add(k);

                    let mut dot0_vec = _mm256_setzero_ps();
                    let mut dot1_vec = _mm256_setzero_ps();
                    let mut dot2_vec = _mm256_setzero_ps();
                    let mut dot3_vec = _mm256_setzero_ps();

                    let mut i = k + 1;
                    while i + 7 < rows {
                        let v_vec = _mm256_loadu_ps(v_ptr.add(i));

                        let c0_vec = _mm256_loadu_ps(c0_ptr.add(i));
                        dot0_vec = _mm256_fmadd_ps(v_vec, c0_vec, dot0_vec);

                        let c1_vec = _mm256_loadu_ps(c1_ptr.add(i));
                        dot1_vec = _mm256_fmadd_ps(v_vec, c1_vec, dot1_vec);

                        let c2_vec = _mm256_loadu_ps(c2_ptr.add(i));
                        dot2_vec = _mm256_fmadd_ps(v_vec, c2_vec, dot2_vec);

                        let c3_vec = _mm256_loadu_ps(c3_ptr.add(i));
                        dot3_vec = _mm256_fmadd_ps(v_vec, c3_vec, dot3_vec);

                        i += 8;
                    }

                    // HSum
                    let mut arr0 = [0.0; 8];
                    _mm256_storeu_ps(arr0.as_mut_ptr(), dot0_vec);
                    let mut arr1 = [0.0; 8];
                    _mm256_storeu_ps(arr1.as_mut_ptr(), dot1_vec);
                    let mut arr2 = [0.0; 8];
                    _mm256_storeu_ps(arr2.as_mut_ptr(), dot2_vec);
                    let mut arr3 = [0.0; 8];
                    _mm256_storeu_ps(arr3.as_mut_ptr(), dot3_vec);

                    for x in arr0 {
                        dot0 += x;
                    }
                    for x in arr1 {
                        dot1 += x;
                    }
                    for x in arr2 {
                        dot2 += x;
                    }
                    for x in arr3 {
                        dot3 += x;
                    }

                    for ii in i..rows {
                        let v_val = *v_ptr.add(ii);
                        dot0 += v_val * (*c0_ptr.add(ii));
                        dot1 += v_val * (*c1_ptr.add(ii));
                        dot2 += v_val * (*c2_ptr.add(ii));
                        dot3 += v_val * (*c3_ptr.add(ii));
                    }

                    let factor0 = tau_f32 * dot0;
                    let factor1 = tau_f32 * dot1;
                    let factor2 = tau_f32 * dot2;
                    let factor3 = tau_f32 * dot3;

                    let f0_vec = _mm256_set1_ps(factor0);
                    let f1_vec = _mm256_set1_ps(factor1);
                    let f2_vec = _mm256_set1_ps(factor2);
                    let f3_vec = _mm256_set1_ps(factor3);

                    *c0_ptr.add(k) -= factor0;
                    *c1_ptr.add(k) -= factor1;
                    *c2_ptr.add(k) -= factor2;
                    *c3_ptr.add(k) -= factor3;

                    i = k + 1;
                    while i + 7 < rows {
                        let v_vec = _mm256_loadu_ps(v_ptr.add(i));

                        let mut c0_vec = _mm256_loadu_ps(c0_ptr.add(i));
                        c0_vec = _mm256_fnmadd_ps(v_vec, f0_vec, c0_vec);
                        _mm256_storeu_ps(c0_ptr.add(i), c0_vec);

                        let mut c1_vec = _mm256_loadu_ps(c1_ptr.add(i));
                        c1_vec = _mm256_fnmadd_ps(v_vec, f1_vec, c1_vec);
                        _mm256_storeu_ps(c1_ptr.add(i), c1_vec);

                        let mut c2_vec = _mm256_loadu_ps(c2_ptr.add(i));
                        c2_vec = _mm256_fnmadd_ps(v_vec, f2_vec, c2_vec);
                        _mm256_storeu_ps(c2_ptr.add(i), c2_vec);

                        let mut c3_vec = _mm256_loadu_ps(c3_ptr.add(i));
                        c3_vec = _mm256_fnmadd_ps(v_vec, f3_vec, c3_vec);
                        _mm256_storeu_ps(c3_ptr.add(i), c3_vec);

                        i += 8;
                    }

                    for ii in i..rows {
                        let v_val = *v_ptr.add(ii);
                        *c0_ptr.add(ii) -= factor0 * v_val;
                        *c1_ptr.add(ii) -= factor1 * v_val;
                        *c2_ptr.add(ii) -= factor2 * v_val;
                        *c3_ptr.add(ii) -= factor3 * v_val;
                    }
                    j += 4;
                }

                for j_col in j..cols {
                    let c_ptr = mut_ptr.add(j_col * rows_stride);

                    let mut dot = *c_ptr.add(k);
                    let mut dot_vec = _mm256_setzero_ps();
                    let mut i = k + 1;
                    while i + 7 < rows {
                        let v_vec = _mm256_loadu_ps(v_ptr.add(i));
                        let c_vec = _mm256_loadu_ps(c_ptr.add(i));
                        dot_vec = _mm256_fmadd_ps(v_vec, c_vec, dot_vec);
                        i += 8;
                    }
                    let mut temp_arr = [0.0; 8];
                    _mm256_storeu_ps(temp_arr.as_mut_ptr(), dot_vec);
                    for x in temp_arr {
                        dot += x;
                    }

                    for ii in i..rows {
                        dot += (*v_ptr.add(ii)) * (*c_ptr.add(ii));
                    }

                    let factor = tau_f32 * dot;
                    let factor_vec = _mm256_set1_ps(factor);

                    *c_ptr.add(k) -= factor;

                    i = k + 1;
                    while i + 7 < rows {
                        let v_vec = _mm256_loadu_ps(v_ptr.add(i));
                        let mut c_vec = _mm256_loadu_ps(c_ptr.add(i));
                        c_vec = _mm256_fnmadd_ps(v_vec, factor_vec, c_vec);
                        _mm256_storeu_ps(c_ptr.add(i), c_vec);
                        i += 8;
                    }
                    for ii in i..rows {
                        *c_ptr.add(ii) -= factor * (*v_ptr.add(ii));
                    }
                }
            }
        } else if tid == TypeId::of::<f64>() {
            use std::arch::x86_64::*;
            // f64 path
            unsafe {
                let start_ptr = qr.storage().data().as_ptr() as *const f64;
                let mut_ptr = qr.storage_mut().data_mut().as_mut_ptr() as *mut f64;
                let rows_stride = rows;

                // Pointer to v (k-th column)
                let v_ptr = start_ptr.add(k * rows_stride);
                let tau_f64 = *(&tau as *const T as *const f64);

                let mut j = k + 1;
                while j + 4 <= cols {
                    let c0_ptr = mut_ptr.add(j * rows_stride);
                    let c1_ptr = mut_ptr.add((j + 1) * rows_stride);
                    let c2_ptr = mut_ptr.add((j + 2) * rows_stride);
                    let c3_ptr = mut_ptr.add((j + 3) * rows_stride);

                    let mut dot0 = *c0_ptr.add(k);
                    let mut dot1 = *c1_ptr.add(k);
                    let mut dot2 = *c2_ptr.add(k);
                    let mut dot3 = *c3_ptr.add(k);

                    let mut dot0_vec = _mm256_setzero_pd();
                    let mut dot1_vec = _mm256_setzero_pd();
                    let mut dot2_vec = _mm256_setzero_pd();
                    let mut dot3_vec = _mm256_setzero_pd();

                    let mut i = k + 1;
                    while i + 3 < rows {
                        let v_vec = _mm256_loadu_pd(v_ptr.add(i));

                        let c0_vec = _mm256_loadu_pd(c0_ptr.add(i));
                        dot0_vec = _mm256_fmadd_pd(v_vec, c0_vec, dot0_vec);

                        let c1_vec = _mm256_loadu_pd(c1_ptr.add(i));
                        dot1_vec = _mm256_fmadd_pd(v_vec, c1_vec, dot1_vec);

                        let c2_vec = _mm256_loadu_pd(c2_ptr.add(i));
                        dot2_vec = _mm256_fmadd_pd(v_vec, c2_vec, dot2_vec);

                        let c3_vec = _mm256_loadu_pd(c3_ptr.add(i));
                        dot3_vec = _mm256_fmadd_pd(v_vec, c3_vec, dot3_vec);

                        i += 4;
                    }

                    let mut arr0 = [0.0; 4];
                    _mm256_storeu_pd(arr0.as_mut_ptr(), dot0_vec);
                    let mut arr1 = [0.0; 4];
                    _mm256_storeu_pd(arr1.as_mut_ptr(), dot1_vec);
                    let mut arr2 = [0.0; 4];
                    _mm256_storeu_pd(arr2.as_mut_ptr(), dot2_vec);
                    let mut arr3 = [0.0; 4];
                    _mm256_storeu_pd(arr3.as_mut_ptr(), dot3_vec);

                    for x in arr0 {
                        dot0 += x;
                    }
                    for x in arr1 {
                        dot1 += x;
                    }
                    for x in arr2 {
                        dot2 += x;
                    }
                    for x in arr3 {
                        dot3 += x;
                    }

                    for ii in i..rows {
                        let v_val = *v_ptr.add(ii);
                        dot0 += v_val * (*c0_ptr.add(ii));
                        dot1 += v_val * (*c1_ptr.add(ii));
                        dot2 += v_val * (*c2_ptr.add(ii));
                        dot3 += v_val * (*c3_ptr.add(ii));
                    }

                    let factor0 = tau_f64 * dot0;
                    let factor1 = tau_f64 * dot1;
                    let factor2 = tau_f64 * dot2;
                    let factor3 = tau_f64 * dot3;

                    let f0_vec = _mm256_set1_pd(factor0);
                    let f1_vec = _mm256_set1_pd(factor1);
                    let f2_vec = _mm256_set1_pd(factor2);
                    let f3_vec = _mm256_set1_pd(factor3);

                    *c0_ptr.add(k) -= factor0;
                    *c1_ptr.add(k) -= factor1;
                    *c2_ptr.add(k) -= factor2;
                    *c3_ptr.add(k) -= factor3;

                    i = k + 1;
                    while i + 3 < rows {
                        let v_vec = _mm256_loadu_pd(v_ptr.add(i));

                        let mut c0_vec = _mm256_loadu_pd(c0_ptr.add(i));
                        c0_vec = _mm256_fnmadd_pd(v_vec, f0_vec, c0_vec);
                        _mm256_storeu_pd(c0_ptr.add(i), c0_vec);

                        let mut c1_vec = _mm256_loadu_pd(c1_ptr.add(i));
                        c1_vec = _mm256_fnmadd_pd(v_vec, f1_vec, c1_vec);
                        _mm256_storeu_pd(c1_ptr.add(i), c1_vec);

                        let mut c2_vec = _mm256_loadu_pd(c2_ptr.add(i));
                        c2_vec = _mm256_fnmadd_pd(v_vec, f2_vec, c2_vec);
                        _mm256_storeu_pd(c2_ptr.add(i), c2_vec);

                        let mut c3_vec = _mm256_loadu_pd(c3_ptr.add(i));
                        c3_vec = _mm256_fnmadd_pd(v_vec, f3_vec, c3_vec);
                        _mm256_storeu_pd(c3_ptr.add(i), c3_vec);

                        i += 4;
                    }

                    for ii in i..rows {
                        let v_val = *v_ptr.add(ii);
                        *c0_ptr.add(ii) -= factor0 * v_val;
                        *c1_ptr.add(ii) -= factor1 * v_val;
                        *c2_ptr.add(ii) -= factor2 * v_val;
                        *c3_ptr.add(ii) -= factor3 * v_val;
                    }
                    j += 4;
                }

                for j_col in j..cols {
                    let c_ptr = mut_ptr.add(j_col * rows_stride);

                    let mut dot = *c_ptr.add(k);
                    let mut dot_vec = _mm256_setzero_pd();

                    let mut i = k + 1;
                    while i + 3 < rows {
                        let v_vec = _mm256_loadu_pd(v_ptr.add(i));
                        let c_vec = _mm256_loadu_pd(c_ptr.add(i));
                        dot_vec = _mm256_fmadd_pd(v_vec, c_vec, dot_vec);
                        i += 4;
                    }

                    let mut temp_arr = [0.0; 4];
                    _mm256_storeu_pd(temp_arr.as_mut_ptr(), dot_vec);
                    for x in temp_arr {
                        dot += x;
                    }

                    for ii in i..rows {
                        dot += (*v_ptr.add(ii)) * (*c_ptr.add(ii));
                    }

                    let factor = tau_f64 * dot;
                    let factor_vec = _mm256_set1_pd(factor);

                    *c_ptr.add(k) -= factor;

                    i = k + 1;
                    while i + 3 < rows {
                        let v_vec = _mm256_loadu_pd(v_ptr.add(i));
                        let mut c_vec = _mm256_loadu_pd(c_ptr.add(i));
                        c_vec = _mm256_fnmadd_pd(v_vec, factor_vec, c_vec);
                        _mm256_storeu_pd(c_ptr.add(i), c_vec);
                        i += 4;
                    }

                    for ii in i..rows {
                        *c_ptr.add(ii) -= factor * (*v_ptr.add(ii));
                    }
                }
            }
        }
    }
}
