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
                let mut v_buf = vec![T::default(); remaining_size];
                // Fill v_buf
                v_buf[0] = T::from_f64(1.0);
                for k in 1..remaining_size {
                     v_buf[k] = *mat_a.get(k + i + 1, i).unwrap();
                }

                unsafe {
                    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() && is_x86_feature_detected!("fma") {
                         Self::compute_p_symmv_f64(mat_a, &mut w, &v_buf, i, remaining_size, h);
                    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() && is_x86_feature_detected!("fma") {
                         Self::compute_p_symmv_f32(mat_a, &mut w, &v_buf, i, remaining_size, h);
                    } else {
                         // Scalar Fallback
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
                                dot += val * v_buf[col];
                            }
                            *val = h * dot;
                        }
                    }
                }
                
                // update_w logic
                let mut vt_w = T::default();
                for (k, val) in w.iter().enumerate().take(remaining_size) {
                     vt_w += v_buf[k] * *val;
                }
                let scale = h * vt_w * T::from_f64(0.5);

                unsafe {
                     if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() && is_x86_feature_detected!("fma") {
                         Self::update_w_vectorized_f64(&mut w, &v_buf, scale);
                     } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() && is_x86_feature_detected!("fma") {
                         Self::update_w_vectorized_f32(&mut w, &v_buf, scale);
                     } else {
                        for (k, val) in w.iter_mut().enumerate().take(remaining_size) {
                            *val -= scale * v_buf[k];
                        }
                     }
                }

                Self::rank2_update(mat_a, &w, i, remaining_size);

                // Restore beta as the tridiagonal sub-diagonal element
                *mat_a.get_mut(i + 1, i).unwrap() = beta;
            } else {
                *h_coeff = T::default();
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn compute_p_symmv_f32(
        mat_a: &Matrix<T, DynamicStorage<T>>,
        w: &mut [T],
        v: &[T],
        i: usize,
        size: usize,
        h: T,
    ) {
        use std::arch::x86_64::*;
        let mat_ptr = mat_a.storage().data().as_ptr() as *const f32;
        let w_ptr = w.as_mut_ptr() as *mut f32;
        let v_ptr = v.as_ptr() as *const f32;
        let rows = mat_a.rows();
        let h_val = *(std::mem::transmute::<&T, &f32>(&h));

        // w is assumed zero initialized
        
        for col in 0..size {
            let v_val = *v_ptr.add(col);
            let v_vec = _mm256_set1_ps(v_val);
            
            // Pointer to A(i+1, col+i+1) [actually start of column vector part]
            // We want A(row+i+1, col+i+1). 
            // In Col-Major, A(r_idx, c_idx) is at c_idx*rows + r_idx.
            // c_idx = col + i + 1.
            // Ptr to start of column c_idx: mat_ptr + c_idx * rows.
            // element at r_idx = row + i + 1.
            // So ptr = mat_ptr + c_idx*rows + (i+1).
            // Then logic adds 'row' to it.
            let col_ptr = mat_ptr.add((col + i + 1) * rows + (i + 1));

            // Diagonal element A(col, col) * v[col]
            // Indices is symmetric. A(col+i+1, col+i+1).
            let val = *col_ptr.add(col);
            *w_ptr.add(col) += val * v_val;

            // Loop row > col
            let mut dot_vec = _mm256_setzero_ps();
            
            let start_row = col + 1;
            let mut r = start_row;
            
            while r + 8 <= size {
                let a_vec = _mm256_loadu_ps(col_ptr.add(r));
                let mut w_vec = _mm256_loadu_ps(w_ptr.add(r));
                let vr_vec = _mm256_loadu_ps(v_ptr.add(r));

                // w[row] += A * v[col]
                w_vec = _mm256_fmadd_ps(a_vec, v_vec, w_vec);
                _mm256_storeu_ps(w_ptr.add(r), w_vec);

                // dot += A * v[row] (for w[col])
                dot_vec = _mm256_fmadd_ps(a_vec, vr_vec, dot_vec);

                r += 8;
            }
            
            // Horizontal sum of dot_vec
            // We can accumulate scalar tail into dot_vec? No, simple scalar sum.
            let mut dot_scalar = 0.0;
            // Reduce dot_vec
            let mut temp = [0.0f32; 8];
            _mm256_storeu_ps(temp.as_mut_ptr(), dot_vec);
            dot_scalar += temp.iter().sum::<f32>();

            for rr in r..size {
                let val = *col_ptr.add(rr);
                // w[rr] += val * v_val
                *w_ptr.add(rr) += val * v_val;
                // w[col] += val * v[rr]
                dot_scalar += val * *v_ptr.add(rr);
            }
            
            *w_ptr.add(col) += dot_scalar;
        }

        // Apply scale h
        let h_vec = _mm256_set1_ps(h_val);
        let mut r = 0;
        while r + 8 <= size {
            let mut w_vec = _mm256_loadu_ps(w_ptr.add(r));
            w_vec = _mm256_mul_ps(w_vec, h_vec);
            _mm256_storeu_ps(w_ptr.add(r), w_vec);
            r += 8;
        }
        for rr in r..size {
            *w_ptr.add(rr) *= h_val;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn update_w_vectorized_f32(w: &mut [T], v: &[T], scale: T) {
        use std::arch::x86_64::*;
        let w_ptr = w.as_mut_ptr() as *mut f32;
        let v_ptr = v.as_ptr() as *const f32;
        let scale_val = *(std::mem::transmute::<&T, &f32>(&scale));
        let scale_vec = _mm256_set1_ps(scale_val);
        let n = w.len();
        
        let mut r = 0;
        while r + 8 <= n {
            let mut w_vec = _mm256_loadu_ps(w_ptr.add(r));
            let v_vec = _mm256_loadu_ps(v_ptr.add(r));
            // w -= scale * v
            // w = w - scale * v = -(scale*v - w) = nmadd? 
            // standard: w - (scale * v)
            // fnmsub: -(a*b) + c = c - a*b.
            w_vec = _mm256_fnmadd_ps(scale_vec, v_vec, w_vec);
            _mm256_storeu_ps(w_ptr.add(r), w_vec);
            r += 8;
        }
        for rr in r..n {
            *w_ptr.add(rr) -= scale_val * *v_ptr.add(rr);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn update_w_vectorized_f64(w: &mut [T], v: &[T], scale: T) {
        use std::arch::x86_64::*;
        let w_ptr = w.as_mut_ptr() as *mut f64;
        let v_ptr = v.as_ptr() as *const f64;
        let scale_val = *(std::mem::transmute::<&T, &f64>(&scale));
        let scale_vec = _mm256_set1_pd(scale_val);
        let n = w.len();
        
        let mut r = 0;
        while r + 4 <= n {
            let mut w_vec = _mm256_loadu_pd(w_ptr.add(r));
            let v_vec = _mm256_loadu_pd(v_ptr.add(r));
            w_vec = _mm256_fnmadd_pd(scale_vec, v_vec, w_vec);
            _mm256_storeu_pd(w_ptr.add(r), w_vec);
            r += 4;
        }
        for rr in r..n {
            *w_ptr.add(rr) -= scale_val * *v_ptr.add(rr);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn compute_p_symmv_f64(
        mat_a: &Matrix<T, DynamicStorage<T>>,
        w: &mut [T],
        v: &[T],
        i: usize,
        size: usize,
        h: T,
    ) {
        use std::arch::x86_64::*;
        let mat_ptr = mat_a.storage().data().as_ptr() as *const f64;
        let w_ptr = w.as_mut_ptr() as *mut f64;
        let v_ptr = v.as_ptr() as *const f64;
        let rows = mat_a.rows();
        let h_val = *(std::mem::transmute::<&T, &f64>(&h));

        for col in 0..size {
            let v_val = *v_ptr.add(col);
            let v_vec = _mm256_set1_pd(v_val);
            
            let col_ptr = mat_ptr.add((col + i + 1) * rows + (i + 1));

            let val = *col_ptr.add(col);
            *w_ptr.add(col) += val * v_val;

            let mut dot_vec = _mm256_setzero_pd();
            
            let start_row = col + 1;
            let mut r = start_row;
            
            while r + 4 <= size {
                let a_vec = _mm256_loadu_pd(col_ptr.add(r));
                let mut w_vec = _mm256_loadu_pd(w_ptr.add(r));
                let vr_vec = _mm256_loadu_pd(v_ptr.add(r));

                w_vec = _mm256_fmadd_pd(a_vec, v_vec, w_vec);
                _mm256_storeu_pd(w_ptr.add(r), w_vec);

                dot_vec = _mm256_fmadd_pd(a_vec, vr_vec, dot_vec);

                r += 4;
            }
            
            let mut dot_scalar = 0.0;
            let mut temp = [0.0f64; 4];
            _mm256_storeu_pd(temp.as_mut_ptr(), dot_vec);
            dot_scalar += temp.iter().sum::<f64>();

            for rr in r..size {
                let val = *col_ptr.add(rr);
                *w_ptr.add(rr) += val * v_val;
                dot_scalar += val * *v_ptr.add(rr);
            }
            
            *w_ptr.add(col) += dot_scalar;
        }

        let h_vec = _mm256_set1_pd(h_val);
        let mut r = 0;
        while r + 4 <= size {
            let mut w_vec = _mm256_loadu_pd(w_ptr.add(r));
            w_vec = _mm256_mul_pd(w_vec, h_vec);
            _mm256_storeu_pd(w_ptr.add(r), w_vec);
            r += 4;
        }
        for rr in r..size {
            *w_ptr.add(rr) *= h_val;
        }
    }

    fn rank2_update(mat_a: &mut Matrix<T, DynamicStorage<T>>, w: &[T], i: usize, size: usize) {
        let rows = mat_a.rows();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
                && is_x86_feature_detected!("fma")
            {
                use std::arch::x86_64::*;
                unsafe {
                    let mat_ptr = mat_a.storage_mut().data_mut().as_mut_ptr() as *mut f32;
                    let w_ptr = w.as_ptr() as *const f32;
                    // v stored part starts at mat(i+2, i)
                    // If size < 2, i+2 might be out of bounds if i=n-2.
                    // But size = n - i - 1. If size=1, n=i+2. i+2 is valid (end).
                    // logical v[1] is at mat(i+2, i).
                    let v_base_ptr = mat_ptr.add(i * rows + i + 2);

                    // Col 0 Special Case
                    {
                        // col=0. v_col = 1.0. p_col = w[0].
                        let col = 0;
                        let v_c = 1.0f32;
                        let p_c = *w_ptr;

                        // row=0 special case: v_row=1.0.
                        let val_00 = *mat_ptr.add((col + i + 1) * rows + (col + i + 1)); // mat(i+1, i+1)
                                                                                         // val -= 1*p_0 + p_0*1 = 2*p_0
                        *mat_ptr.add((col + i + 1) * rows + (col + i + 1)) = val_00 - 2.0 * p_c;

                        // row 1..size
                        let col_ptr = mat_ptr.add((col + i + 1) * rows + (col + i + 1));
                        let mut r = 1;

                        let vc_vec = _mm256_set1_ps(v_c);
                        let pc_vec = _mm256_set1_ps(p_c);

                        while r + 8 <= size {
                            // v_row from v_base_ptr[r-1]
                            let vr_vec = _mm256_loadu_ps(v_base_ptr.add(r - 1));
                            let pr_vec = _mm256_loadu_ps(w_ptr.add(r));
                            let mut a_vec = _mm256_loadu_ps(col_ptr.add(r));

                            // A -= vr*pc + pr*vc
                            let term1 = _mm256_mul_ps(vr_vec, pc_vec);
                            let term2 = _mm256_mul_ps(pr_vec, vc_vec);
                            let sum = _mm256_add_ps(term1, term2);
                            a_vec = _mm256_sub_ps(a_vec, sum);

                            _mm256_storeu_ps(col_ptr.add(r), a_vec);
                            r += 8;
                        }
                        for rr in r..size {
                            let vr = *v_base_ptr.add(rr - 1);
                            let pr = *w_ptr.add(rr);
                            *col_ptr.add(rr) -= vr * p_c + pr * v_c;
                        }
                    }

                    // Cols 1..size
                    for col in 1..size {
                        let v_c = *v_base_ptr.add(col - 1);
                        let p_c = *w_ptr.add(col);

                        let vc_vec = _mm256_set1_ps(v_c);
                        let pc_vec = _mm256_set1_ps(p_c);

                        // Fix: col_ptr should point to the start of the block (row 0 of the block, which is i+1 global)
                        // We want A[row+i+1, col+i+1].
                        // r iterates row. col_ptr.add(r) should give A[r+i+1].
                        // So col_ptr should be &A[i+1, col+i+1].
                        // mat_ptr at (col+i+1)*rows + (i+1).
                        let col_ptr = mat_ptr.add((col + i + 1) * rows + (i + 1));
                        let mut r = col; // row starts at col

                        while r + 8 <= size {
                            // v_row from v_base_ptr[r-1]
                            let vr_vec = _mm256_loadu_ps(v_base_ptr.add(r - 1));
                            let pr_vec = _mm256_loadu_ps(w_ptr.add(r));
                            let mut a_vec = _mm256_loadu_ps(col_ptr.add(r)); // A is shifted by row

                            // A -= vr*pc + pr*vc
                            let term1 = _mm256_mul_ps(vr_vec, pc_vec);
                            let term2 = _mm256_mul_ps(pr_vec, vc_vec);
                            let sum = _mm256_add_ps(term1, term2);
                            a_vec = _mm256_sub_ps(a_vec, sum);

                            _mm256_storeu_ps(col_ptr.add(r), a_vec);
                            r += 8;
                        }
                        for rr in r..size {
                            let vr = *v_base_ptr.add(rr - 1);
                            let pr = *w_ptr.add(rr);
                            *col_ptr.add(rr) -= vr * p_c + pr * v_c;
                        }
                    }
                    return;
                }
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()
                && is_x86_feature_detected!("fma")
            {
                use std::arch::x86_64::*;
                unsafe {
                    let mat_ptr = mat_a.storage_mut().data_mut().as_mut_ptr() as *mut f64;
                    let w_ptr = w.as_ptr() as *const f64;
                    let v_base_ptr = mat_ptr.add(i * rows + i + 2);

                    // Col 0
                    {
                        let col = 0;
                        let v_c = 1.0f64;
                        let p_c = *w_ptr;

                        let val_00 = *mat_ptr.add((col + i + 1) * rows + (col + i + 1));
                        *mat_ptr.add((col + i + 1) * rows + (col + i + 1)) = val_00 - 2.0 * p_c;

                        // Fix col 0 ptr for consistency (though it was correct before because col=0)
                        let col_ptr = mat_ptr.add((col + i + 1) * rows + (i + 1));
                        let mut r = 1;

                        let vc_vec = _mm256_set1_pd(v_c);
                        let pc_vec = _mm256_set1_pd(p_c);

                        while r + 4 <= size {
                            let vr_vec = _mm256_loadu_pd(v_base_ptr.add(r - 1));
                            let pr_vec = _mm256_loadu_pd(w_ptr.add(r));
                            let mut a_vec = _mm256_loadu_pd(col_ptr.add(r));

                            let term1 = _mm256_mul_pd(vr_vec, pc_vec);
                            let term2 = _mm256_mul_pd(pr_vec, vc_vec);
                            let sum = _mm256_add_pd(term1, term2);
                            a_vec = _mm256_sub_pd(a_vec, sum);

                            _mm256_storeu_pd(col_ptr.add(r), a_vec);
                            r += 4;
                        }
                        for rr in r..size {
                            let vr = *v_base_ptr.add(rr - 1);
                            let pr = *w_ptr.add(rr);
                            *col_ptr.add(rr) -= vr * p_c + pr * v_c;
                        }
                    }

                    for col in 1..size {
                        let v_c = *v_base_ptr.add(col - 1);
                        let p_c = *w_ptr.add(col);

                        let vc_vec = _mm256_set1_pd(v_c);
                        let pc_vec = _mm256_set1_pd(p_c);

                        // Fix col_ptr for f64 too
                        let col_ptr = mat_ptr.add((col + i + 1) * rows + (i + 1));
                        let mut r = col;

                        while r + 4 <= size {
                            let vr_vec = _mm256_loadu_pd(v_base_ptr.add(r - 1));
                            let pr_vec = _mm256_loadu_pd(w_ptr.add(r));
                            let mut a_vec = _mm256_loadu_pd(col_ptr.add(r));

                            let term1 = _mm256_mul_pd(vr_vec, pc_vec);
                            let term2 = _mm256_mul_pd(pr_vec, vc_vec);
                            let sum = _mm256_add_pd(term1, term2);
                            a_vec = _mm256_sub_pd(a_vec, sum);

                            _mm256_storeu_pd(col_ptr.add(r), a_vec);
                            r += 4;
                        }
                        for rr in r..size {
                            let vr = *v_base_ptr.add(rr - 1);
                            let pr = *w_ptr.add(rr);
                            *col_ptr.add(rr) -= vr * p_c + pr * v_c;
                        }
                    }
                    return;
                }
            }
        }

        // Refined Implementation
        Self::rank2_update_scalar(mat_a, w, i, size);
    }

    fn rank2_update_scalar(
        mat_a: &mut Matrix<T, DynamicStorage<T>>,
        w: &[T],
        i: usize,
        size: usize,
    ) {
        for col in 0..size {
            for row in col..size {
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

                *mat_a.get_mut(row + i + 1, col + i + 1).unwrap() -= v_row * p_col + p_row * v_col;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::matrix::Matrix;
    use crate::core::storage::DynamicStorage;
    use crate::core::decompositions::Tridiagonalization;

    #[test]
    fn test_tridiagonal_decomposition() {
        let n: usize = 4;
        let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
        // Symmetric matrix
        let val = vec![
            4.0, 1.0, -2.0, 2.0,
            1.0, 2.0, 0.0, 1.0,
            -2.0, 0.0, 3.0, -2.0,
            2.0, 1.0, -2.0, -1.0
        ];
        
        for i in 0..n {
            for j in 0..n {
                *a.get_mut(i, j).unwrap() = val[i * n + j];
            }
        }

        let tridiag = Tridiagonalization::new(&a).unwrap();
        let t = tridiag.matrix_t();
        let q = tridiag.matrix_q();

        // Check T is tridiagonal
        for i in 0..n {
            for j in 0..n {
                if (i as isize - j as isize).abs() > 1 {
                    assert!(t.get(i, j).unwrap().abs() < 1e-10, "T not tridiagonal at ({}, {}): {}", i, j, t.get(i, j).unwrap());
                }
            }
        }

        // Check Q is orthogonal: Q * Q^T = I
        let qt = q.transpose();
        
        let mut q_qt_res = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
        q_qt_res.assign_product(&(&q * &qt)).unwrap();

        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((q_qt_res.get(i, j).unwrap() - expected).abs() < 1e-10, "Q not orthogonal at ({}, {})", i, j);
            }
        }

        // Check A = Q * T * Q^T
        let mut t_qt_res = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
        t_qt_res.assign_product(&(&t * &qt)).unwrap();
        
        let mut recon = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n).unwrap();
        recon.assign_product(&(&q * &t_qt_res)).unwrap();

        for i in 0..n {
            for j in 0..n {
                let diff = (recon.get(i, j).unwrap() - a.get(i, j).unwrap()).abs();
                assert!(diff < 1e-10, "Reconstruction failed at ({}, {}): expected {}, got {}, diff {}", i, j, a.get(i, j).unwrap(), recon.get(i, j).unwrap(), diff);
            }
        }
    }
}
