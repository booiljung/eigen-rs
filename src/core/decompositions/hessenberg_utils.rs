
use crate::core::matrix::Matrix;
use crate::core::storage::{DynamicStorage, Storage};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

// Vectorized helper to apply Householder reflection from the left: A = (I - tau v v^T) A
// A -= tau * v * (v^T * A)
// Logic:
// 1. Compute workspace w = v^T * A. (w is a row vector, w[j] = v . A_j)
//    Since A is column major, A_j is contiguous. calculating dot product v . A_j is efficient.
// 2. Update A: A_j -= (tau * w[j]) * v.
//    A_j -= factor * v. simple axpy.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn apply_householder_on_the_left_vectorized_f64<S: Storage<f64>>(
    mat_a: &mut Matrix<f64, S>,
    v_buf: &[f64],
    tau: f64,
    start_row: usize,
    cols_start: usize,
    cols_end: usize,
) {
    let rows = mat_a.rows();
    let v_len = v_buf.len(); // Length of Householder vector
    
    // Safety check?
    // start_row + v_len should fall within rows.

    for j in cols_start..cols_end {
        let col_ptr = mat_a.storage_mut().data_mut().as_mut_ptr().add(j * rows + start_row);
        
        // 1. Compute dot product: dot = v . col
        let mut dot = 0.0;
        let mut r = 0;
        
        // AVX2 Dot Product loop
        let mut sum_vec = _mm256_setzero_pd();
        while r + 4 <= v_len {
            let v_val = _mm256_loadu_pd(v_buf.as_ptr().add(r));
            let col_val = _mm256_loadu_pd(col_ptr.add(r));
            sum_vec = _mm256_fmadd_pd(v_val, col_val, sum_vec);
            r += 4;
        }
        // Horizontal sum
        let temp = _mm256_add_pd(sum_vec, _mm256_permute2f128_pd(sum_vec, sum_vec, 1));
        let temp = _mm256_add_pd(temp, _mm256_permute_pd(temp, 5));
        dot += _mm256_cvtsd_f64(temp);

        // Scalar fallback for remainder
        while r < v_len {
            dot += *v_buf.get_unchecked(r) * *col_ptr.add(r);
            r += 1;
        }

        // 2. Update column: col -= (tau * dot) * v
        let factor = tau * dot;
        let factor_vec = _mm256_set1_pd(factor);
        
        let mut r = 0;
        while r + 4 <= v_len {
            let v_val = _mm256_loadu_pd(v_buf.as_ptr().add(r));
            let mut col_val = _mm256_loadu_pd(col_ptr.add(r));
            col_val = _mm256_fnmadd_pd(factor_vec, v_val, col_val); // col = col - factor * v
            _mm256_storeu_pd(col_ptr.add(r), col_val);
            r += 4;
        }
        
        while r < v_len {
            *col_ptr.add(r) -= factor * *v_buf.get_unchecked(r);
            r += 1;
        }
    }
}


// Vectorized helper to apply Householder reflection from the right: A = A (I - tau v v^T)
// A -= tau * (A v) * v^T
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn apply_householder_on_the_right_vectorized_f64<S: Storage<f64>>(
    mat_a: &mut Matrix<f64, S>,
    v_buf: &[f64],
    tau: f64,
    start_col: usize,
    rows_end: usize, 
    w: &mut [f64], // Pre-allocated workspace
) {
    let rows = mat_a.rows();
    let v_len = v_buf.len();
    
    // Initialize w to zero
    std::ptr::write_bytes(w.as_mut_ptr(), 0, rows_end);

    // 1. Compute w = A * v = sum(A_j * v_j)
    for j in 0..v_len {
        let v_val = *v_buf.get_unchecked(j); // Scalar
        let col_idx = start_col + j;
        let col_ptr = mat_a.storage().data().as_ptr().add(col_idx * rows);
        
        if v_val != 0.0 {
             let v_vec = _mm256_set1_pd(v_val);
             let mut r = 0;
             while r + 4 <= rows_end {
                 let col_val = _mm256_loadu_pd(col_ptr.add(r));
                 let mut w_val = _mm256_loadu_pd(w.as_mut_ptr().add(r));
                 w_val = _mm256_fmadd_pd(col_val, v_vec, w_val); // w += col * v
                 _mm256_storeu_pd(w.as_mut_ptr().add(r), w_val);
                 r += 4;
             }
             while r < rows_end {
                 *w.get_unchecked_mut(r) += *col_ptr.add(r) * v_val;
                 r += 1;
             }
        }
    }

    // 2. Update A: A_j -= (tau * v_j) * w
    for j in 0..v_len {
        let v_val = *v_buf.get_unchecked(j);
        let factor = tau * v_val; // Scalar
        let col_idx = start_col + j;
        let col_ptr = mat_a.storage_mut().data_mut().as_mut_ptr().add(col_idx * rows);

        if factor != 0.0 {
            let factor_vec = _mm256_set1_pd(factor);
            let mut r = 0;
            while r + 4 <= rows_end {
                let mut col_val = _mm256_loadu_pd(col_ptr.add(r));
                let w_val = _mm256_loadu_pd(w.as_ptr().add(r));
                col_val = _mm256_fnmadd_pd(factor_vec, w_val, col_val); // col -= factor * w
                _mm256_storeu_pd(col_ptr.add(r), col_val);
                r += 4;
            }
            while r < rows_end {
                *col_ptr.add(r) -= factor * *w.get_unchecked(r);
                r += 1;
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn apply_householder_on_the_left_vectorized_f32<S: Storage<f32>>(
    mat_a: &mut Matrix<f32, S>,
    v_buf: &[f32],
    tau: f32,
    start_row: usize,
    cols_start: usize,
    cols_end: usize,
) {
    let rows = mat_a.rows();
    let v_len = v_buf.len();

    for j in cols_start..cols_end {
        let col_ptr = mat_a.storage_mut().data_mut().as_mut_ptr().add(j * rows + start_row);
        
        let mut dot = 0.0;
        let mut r = 0;
        
        let mut sum_vec = _mm256_setzero_ps();
        while r + 8 <= v_len {
            let v_val = _mm256_loadu_ps(v_buf.as_ptr().add(r));
            let col_val = _mm256_loadu_ps(col_ptr.add(r));
            sum_vec = _mm256_fmadd_ps(v_val, col_val, sum_vec);
            r += 8;
        }
        let temp = _mm256_add_ps(sum_vec, _mm256_permute2f128_ps(sum_vec, sum_vec, 1));
        let temp = _mm256_hadd_ps(temp, temp); 
        let temp = _mm256_hadd_ps(temp, temp); 
        dot += _mm_cvtss_f32(_mm256_castps256_ps128(temp));

        while r < v_len {
            dot += *v_buf.get_unchecked(r) * *col_ptr.add(r);
            r += 1;
        }

        let factor = tau * dot;
        let factor_vec = _mm256_set1_ps(factor);
        
        let mut r = 0;
        while r + 8 <= v_len {
            let v_val = _mm256_loadu_ps(v_buf.as_ptr().add(r));
            let mut col_val = _mm256_loadu_ps(col_ptr.add(r));
            col_val = _mm256_fnmadd_ps(factor_vec, v_val, col_val);
            _mm256_storeu_ps(col_ptr.add(r), col_val);
            r += 8;
        }
        
        while r < v_len {
            *col_ptr.add(r) -= factor * *v_buf.get_unchecked(r);
            r += 1;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn apply_householder_on_the_right_vectorized_f32<S: Storage<f32>>(
    mat_a: &mut Matrix<f32, S>,
    v_buf: &[f32],
    tau: f32,
    start_col: usize,
    rows_end: usize,
    w: &mut [f32],
) {
    let rows = mat_a.rows();
    let v_len = v_buf.len();
    
    std::ptr::write_bytes(w.as_mut_ptr(), 0, rows_end);

    for j in 0..v_len {
        let v_val = *v_buf.get_unchecked(j);
        let col_idx = start_col + j;
        let col_ptr = mat_a.storage().data().as_ptr().add(col_idx * rows);
        
        if v_val != 0.0 {
             let v_vec = _mm256_set1_ps(v_val);
             let mut r = 0;
             while r + 8 <= rows_end {
                 let col_val = _mm256_loadu_ps(col_ptr.add(r));
                 let mut w_val = _mm256_loadu_ps(w.as_mut_ptr().add(r));
                 w_val = _mm256_fmadd_ps(col_val, v_vec, w_val);
                 _mm256_storeu_ps(w.as_mut_ptr().add(r), w_val);
                 r += 8;
             }
             while r < rows_end {
                 *w.get_unchecked_mut(r) += *col_ptr.add(r) * v_val;
                 r += 1;
             }
        }
    }

    for j in 0..v_len {
        let v_val = *v_buf.get_unchecked(j);
        let factor = tau * v_val;
        let col_idx = start_col + j;
        let col_ptr = mat_a.storage_mut().data_mut().as_mut_ptr().add(col_idx * rows);

        if factor != 0.0 {
            let factor_vec = _mm256_set1_ps(factor);
            let mut r = 0;
            while r + 8 <= rows_end {
                let mut col_val = _mm256_loadu_ps(col_ptr.add(r));
                let w_val = _mm256_loadu_ps(w.as_ptr().add(r));
                col_val = _mm256_fnmadd_ps(factor_vec, w_val, col_val);
                _mm256_storeu_ps(col_ptr.add(r), col_val);
                r += 8;
            }
            while r < rows_end {
                *col_ptr.add(r) -= factor * *w.get_unchecked(r);
                r += 1;
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn compute_householder_vectorized_f64<S: Storage<f64>>(
    mat_a: &mut Matrix<f64, S>,
    i: usize,
    v_buf: &mut [f64],
) -> (f64, f64) {
    let n = mat_a.rows();
    let col_ptr = mat_a.storage_mut().data_mut().as_mut_ptr().add(i * n); // Start of column i
    
    let tail_ptr = col_ptr.add(i + 2);
    let tail_len = if n > i + 2 { n - (i + 2) } else { 0 };
    
    let mut norm_sq = 0.0;
    
    if tail_len > 0 {
        let mut sum_vec = _mm256_setzero_pd();
        let mut r = 0;
        while r + 4 <= tail_len {
            let val = _mm256_loadu_pd(tail_ptr.add(r));
            sum_vec = _mm256_fmadd_pd(val, val, sum_vec);
            r += 4;
        }
        let temp = _mm256_add_pd(sum_vec, _mm256_permute2f128_pd(sum_vec, sum_vec, 1));
        let temp = _mm256_add_pd(temp, _mm256_permute_pd(temp, 5));
        norm_sq += _mm256_cvtsd_f64(temp);

        while r < tail_len {
            let val = *tail_ptr.add(r);
            norm_sq += val * val;
            r += 1;
        }
    }
    
    let v0 = *col_ptr.add(i + 1);
    norm_sq += v0 * v0;
    let norm = norm_sq.sqrt();
    
    if norm == 0.0 {
        return (0.0, 0.0);
    }
    
    // In hessenberg.rs logic:
    // let sigma = if v0 >= T::default() { norm } else { T::default() - norm }; NOTE: hessenberg.rs uses beta=if >=0 {-norm} else {norm} and sigma=beta? No.
    // hessenberg.rs:
    // sigma = if v0 >= 0 { norm } else { -norm }; ??
    // v0_plus_sigma = v0 + sigma;
    // tau = v0_plus_sigma.conj() / sigma;
    // Let's stick to hessenberg.rs exactly.
    let sigma = if v0 >= 0.0 { norm } else { -norm };
    let v0_plus_sigma = v0 + sigma;
    let inv_v0_plus_sigma = 1.0 / v0_plus_sigma;
    let tau = v0_plus_sigma / sigma; // For Real, conj() is identity.
    
    // Scale tail
    let scale_vec = _mm256_set1_pd(inv_v0_plus_sigma);
    if tail_len > 0 {
        let mut r = 0;
        while r + 4 <= tail_len {
            let mut val = _mm256_loadu_pd(tail_ptr.add(r));
            val = _mm256_mul_pd(val, scale_vec);
            _mm256_storeu_pd(tail_ptr.add(r), val);
            r += 4;
        }
        while r < tail_len {
            *tail_ptr.add(r) *= inv_v0_plus_sigma;
            r += 1;
        }
    }
    
    v_buf[0] = 1.0;
    if tail_len > 0 {
        let mut r = 0;
        while r + 4 <= tail_len {
             let val = _mm256_loadu_pd(tail_ptr.add(r));
             _mm256_storeu_pd(v_buf.as_mut_ptr().add(1 + r), val);
             r += 4;
        }
        while r < tail_len {
            v_buf[1 + r] = *tail_ptr.add(r);
            r += 1;
        }
    }
    
    // Return (tau, restoration_value)
    // hessenberg.rs: *mat_a.get_mut(i + 1, i) = T::default() - sigma;
    (tau, -sigma)
}


#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn compute_householder_vectorized_f32<S: Storage<f32>>(
    mat_a: &mut Matrix<f32, S>,
    i: usize,
    v_buf: &mut [f32],
) -> (f32, f32) {
    let n = mat_a.rows();
    let col_ptr = mat_a.storage_mut().data_mut().as_mut_ptr().add(i * n);
    
    let tail_ptr = col_ptr.add(i + 2);
    let tail_len = if n > i + 2 { n - (i + 2) } else { 0 };
    
    let mut norm_sq = 0.0;
    
    if tail_len > 0 {
        let mut sum_vec = _mm256_setzero_ps();
        let mut r = 0;
        while r + 8 <= tail_len {
            let val = _mm256_loadu_ps(tail_ptr.add(r));
            sum_vec = _mm256_fmadd_ps(val, val, sum_vec);
            r += 8;
        }
        let temp = _mm256_add_ps(sum_vec, _mm256_permute2f128_ps(sum_vec, sum_vec, 1));
        let temp = _mm256_hadd_ps(temp, temp);
        let temp = _mm256_hadd_ps(temp, temp);
        norm_sq += _mm_cvtss_f32(_mm256_castps256_ps128(temp));

        while r < tail_len {
            let val = *tail_ptr.add(r);
            norm_sq += val * val;
            r += 1;
        }
    }
    
    let v0 = *col_ptr.add(i + 1);
    norm_sq += v0 * v0;
    let norm = norm_sq.sqrt();
    
    if norm == 0.0 {
        return (0.0, 0.0);
    }
    
    let sigma = if v0 >= 0.0 { norm } else { -norm };
    let v0_plus_sigma = v0 + sigma;
    let inv_v0_plus_sigma = 1.0 / v0_plus_sigma;
    let tau = v0_plus_sigma / sigma; 
    
    let scale_vec = _mm256_set1_ps(inv_v0_plus_sigma);
    if tail_len > 0 {
        let mut r = 0;
        while r + 8 <= tail_len {
            let mut val = _mm256_loadu_ps(tail_ptr.add(r));
            val = _mm256_mul_ps(val, scale_vec);
            _mm256_storeu_ps(tail_ptr.add(r), val);
            r += 8;
        }
        while r < tail_len {
            *tail_ptr.add(r) *= inv_v0_plus_sigma;
            r += 1;
        }
    }
    
    v_buf[0] = 1.0;
    if tail_len > 0 {
        let mut r = 0;
        while r + 8 <= tail_len {
             let val = _mm256_loadu_ps(tail_ptr.add(r));
             _mm256_storeu_ps(v_buf.as_mut_ptr().add(1 + r), val);
             r += 8;
        }
        while r < tail_len {
            v_buf[1 + r] = *tail_ptr.add(r);
            r += 1;
        }
    }
    
    (tau, -sigma)
}
