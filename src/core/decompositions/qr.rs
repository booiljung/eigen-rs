//! Householder QR decomposition (A = QR).

use crate::core::matrix::Matrix;
use num_traits::Zero;
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

        // Tuning parameter for Blocked QR
        // FIXME: Blocked QR (Level-3 BLAS) is currently disabled due to unresolved heap corruption (Segfault/Double Free) at N=128.
        // Reverting to Unblocked (Level-2 BLAS) for stability.
        const BLOCK_SIZE: usize = 64; // Block size for potential future enablement

        // Force Unblocked execution
        if false && cols >= BLOCK_SIZE * 2 {
             Self::compute_blocked(&mut qr, &mut h_coeffs, BLOCK_SIZE);
        } else {
             Self::compute_unblocked(&mut qr, &mut h_coeffs, 0, size);
        }

        Ok(Self {
            qr,
            h_coeffs,
            _phantom: std::marker::PhantomData,
        })
    }

    fn compute_unblocked(
        qr: &mut Matrix<T, DynamicStorage<T>>,
        h_coeffs: &mut [T],
        start_col: usize,
        end_col: usize
    ) {
        let rows = qr.rows();
        let cols = qr.cols();

        for k in start_col..end_col {
            // 1. Compute norm of the tail of column k
            let mut norm_sq = T::Real::zero();
            for i in k..rows {
                let val = *qr.get(i, k).unwrap();
                norm_sq += val.norm_sq();
            }
            let norm = norm_sq.sqrt();

            if norm != T::Real::zero() {
                let v0 = *qr.get(k, k).unwrap();
                // For Real: if v0 >= 0, sigma = +norm.
                // For Complex: we align with the phase of v0.
                // General safe choice: sigma = sign(v0) * norm.
                // But simplified for types: check real part.
                let v0_abs = v0.abs();
                let sigma = if v0_abs == T::Real::zero() {
                    T::from_real(norm)
                } else {
                    // Align sigma with v0: sigma = norm * (v0 / |v0|)
                    (v0 * T::from_real(norm)) / T::from_real(v0_abs)
                };

                let v0_new = v0 + sigma;
                // Correct tau for normalized v (where v[0] = 1)
                let tau = v0_new / sigma;
                h_coeffs[k] = tau;

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
                        use std::any::TypeId;
                        let tid = TypeId::of::<T>();
                        if tid == TypeId::of::<f32>() || tid == TypeId::of::<f64>() {
                            Self::apply_householder_avx(qr, k, tau, rows, cols);
                            continue;
                        }
                    }
                }

                // Scalar Fallback
                for j in k + 1..cols {
                    let mut dot = *qr.get(k, j).unwrap();
                    for i in k + 1..rows {
                        dot += (*qr.get(i, k).unwrap()).conj() * (*qr.get(i, j).unwrap());
                    }

                    let factor = tau * dot;
                    *qr.get_mut(k, j).unwrap() -= factor;
                    for i in k + 1..rows {
                        let vi = *qr.get(i, k).unwrap();
                        *qr.get_mut(i, j).unwrap() -= factor * vi;
                    }
                }
            } else {
                h_coeffs[k] = T::default();
            }
        }
    }

    fn compute_blocked(
        qr: &mut Matrix<T, DynamicStorage<T>>,
        h_coeffs: &mut [T],
        block_size: usize
    ) {
        let rows = qr.rows();
        let cols = qr.cols();
        let size = h_coeffs.len();

        let mut k = 0;
        while k < size {
            let bs = std::cmp::min(block_size, size - k);
            let end_k = k + bs;

            // 1. Panel Factorization (Unblocked)
            Self::compute_panel_unblocked(qr, h_coeffs, k, end_k);

            // 2. Trailing Matrix Update (if exists)
            if end_k < cols {
                // T: block_size x block_size upper triangular
                let mut t_mat = vec![T::default(); bs * bs];
                Self::compute_t(qr, h_coeffs, k, bs, &mut t_mat);

                // Y: rows x block_size (Unit lower trapezoidal)
                // We need to extract Y explicitly for GEMM.
                // Or use the fact strictly lower part is in A.
                // Construct compact Y for GEMM.
                let mut y_mat = vec![T::default(); rows * bs];
                for j in 0..bs {
                    let global_j = k + j;
                    for i in 0..rows {
                        let val = if i < global_j {
                            T::default()
                        } else if i == global_j {
                            T::from_usize(1)
                        } else {
                             *qr.get(i, global_j).unwrap()
                        };
                        y_mat[i + j * rows] = val; // Col-Major
                    }
                }
                
                // Target: A_trail (rows x (cols - end_k)) starting at (0, end_k)
                // We actually only need to update rows >= k. But Householder vectors start at k.
                // Y is zero for rows < k. So we can update starting at row k.
                // A_trail_sub = A[k:rows, end_k:cols]
                
                // Blocked update: A_trail -= Y * T^T * Y^T * A_trail
                Self::apply_block_update(qr, &t_mat, k, end_k, rows, cols, &y_mat);
            }

            k += bs;
        }
    }

    // Unblocked QR that ONLY updates columns within the range [start, end) (Panel)
    // But it must maintain the full height Householder vectors.
    fn compute_panel_unblocked(
        qr: &mut Matrix<T, DynamicStorage<T>>,
        h_coeffs: &mut [T],
        start_col: usize,
        end_col: usize
    ) {
        let rows = qr.rows();
        let cols = qr.cols(); // Global cols needed for stride calculation if AVX? 
        // Actually for panel, we only update columns up to `end_col`.

        for k in start_col..end_col {
            // ... same norm logic ...
            // ... same norm logic ...
            let mut norm_sq = T::Real::zero();
            for i in k..rows {
                let val = *qr.get(i, k).unwrap();
                norm_sq += val.norm_sq();
            }
            let norm = norm_sq.sqrt();

            if norm != T::Real::zero() {
                let v0 = *qr.get(k, k).unwrap();
                let sigma = if v0.real() >= T::Real::zero() { T::from_real(norm) } else { T::from_real(-norm) };
                let v0_new = v0 + sigma;
                let tau = v0_new / sigma;
                h_coeffs[k] = tau;
                let inv_v0_new = v0_new.recip();
                for i in k + 1..rows {
                    *qr.get_mut(i, k).unwrap() *= inv_v0_new;
                }
                *qr.get_mut(k, k).unwrap() = T::default() - sigma;

                // Update only columns within the panel [k+1, end_col)
                // Use AVX if available
                 #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("fma") {
                        // We need a version of apply which limits the columns.
                        // The existing apply_householder_avx takes `cols`. 
                        // If we pass `end_col`, it will work only on valid range.
                        Self::apply_householder_avx(qr, k, tau, rows, end_col);
                        continue;
                    }
                }
                
                // Scalar Fallback
                for j in k + 1..end_col {
                    let mut dot = *qr.get(k, j).unwrap();
                    for i in k + 1..rows {
                        dot += (*qr.get(i, k).unwrap()).conj() * (*qr.get(i, j).unwrap());
                    }
                    let factor = tau * dot;
                    *qr.get_mut(k, j).unwrap() -= factor;
                    for i in k + 1..rows {
                        let vi = *qr.get(i, k).unwrap();
                        *qr.get_mut(i, j).unwrap() -= factor * vi;
                    }
                }
            } else {
                h_coeffs[k] = T::default();
            }
        }
    }

    // Computes T matrix (size bs x bs) for the block.
    // T is upper triangular.
    // Ref: LAPACK DLARFT
    fn compute_t(
        qr: &Matrix<T, DynamicStorage<T>>,
        h_coeffs: &[T],
        k_start: usize,
        bs: usize,
        t_mat: &mut [T]
    ) {
        // T(i, i) = tau[i]
        // T(i, j) = -tau[i] * (v[i]^T * v[j] * T(j, j) + ...)
        // A simpler recursion:
        // T_0 = tau_0
        // For i = 1..bs-1:
        //   T(0..i, i) = -tau[i] * T(0..i, 0..i) * (v[i]^T * V(0..rows, 0..i))^T ... tricky.
        // Let's use the explicit loop structure from LAPACK DLARFT.
        
        // T is bs x bs.
        let rows = qr.rows();
        
        for i in 0..bs {
            let tau = h_coeffs[k_start + i];
            if tau == T::default() {
                // T row/col i is zero
                continue;
            }
            
            // T(i, i) = tau
            t_mat[i + i * bs] = tau;
            
            // For j = 0..i
            // T(j, i) = -tau * (V(:, i)^T * V(:, j)) * ... wait.
            // Actually: T(0:i, i) = -tau * T(0:i, 0:i) * (V(:, 0:i)^T * v_i)
            
            // 1. Compute w = V(:, 0:i)^T * v_i
            // V is implicitly unit lower trapezoidal. 
            // V_j is column k_start + j.
            // v_i is column k_start + i.
            // dot product range: start from row k_start + i + 1? No, from k_start + max(i, j)?
            // V matrix structure:
            // 1
            // v 1
            // v v 1
            // ...
            
            // w is length i.
            for j in 0..i {
                let mut sum = T::default();
                // v_j starts at row k_start + j + 1 (with implicit 1 at k_start + j)
                // v_i starts at row k_start + i + 1 (with implicit 1 at k_start + i)
                
                // intersection starts at row k_start + i.
                // At row k_start + j: v_j=1, v_i=0 (since i > j). 
                // Actually v_i is 0 for rows < k_start + i.
                // So dot product is only for rows >= k_start + i.
                
                // At row k_start + i: v_j has value, v_i is 1.
                // sum += v_j[k_start+i] * 1
                if k_start + i < rows {
                    sum += *qr.get(k_start + i, k_start + j).unwrap();
                }
                
                // Remaining rows
                for r in (k_start + i + 1)..rows {
                    sum += (*qr.get(r, k_start + j).unwrap()) * (*qr.get(r, k_start + i).unwrap());
                }
                
                t_mat[j + i * bs] = sum;
            }
            
            // 2. T(0:i, i) = -tau * T(0:i, 0:i) * w
            // We can compute this utilizing the existing triangular T structure.
            // vector z = -tau * w
            // T_col_i = T_prev * z
            
            // Implement GEMV-like T * z logic
            for j in 0..i {
                t_mat[j + i * bs] *= -tau;
            }
            
            // Now multiply by T(0:i, 0:i) which is upper triangular
            // We overwrite column i.
            // Work backwards or use buffer? Use buffer (scalar is cheap).
            let mut col_res = vec![T::default(); i];
            for r in 0..i {
                let mut acc = T::default();
                for c in r..i {
                    // T is upper triangular, so only c >= r matters
                     acc += t_mat[r + c * bs] * t_mat[c + i * bs];
                }
                col_res[r] = acc;
            }
            for r in 0..i {
                t_mat[r + i * bs] = col_res[r];
            }
        }
    }

    // Applies the update A_trail -= Y * T^T * Y^T * A_trail
    // Equivalent to:
    // 1. W = Y^T * A_trail
    // 2. W = -T^T * W  (Fuse negation here to handle gemm's additive nature)
    // 3. A_trail += Y * W
    // Applies the update A_trail -= Y * T^T * Y^T * A_trail
    // Equivalent to:
    // 1. W = Y^T * A_trail
    // 2. W = -T^T * W  (Fuse negation here to handle gemm's additive nature)
    // 3. A_trail += Y * W
    fn apply_block_update(
        qr: &mut Matrix<T, DynamicStorage<T>>,
        t_mat: &[T],
        k: usize,
        end_k: usize,
        rows: usize,
        cols: usize,
        y_mat: &[T] 
    ) {
         use crate::core::ops::gemm::gemm_dispatch_pointers;

         let bs = end_k - k;
         let trail_cols = cols - end_k;
         
         if trail_cols == 0 { return; }

         // 1. W = Y^T * A_trail
         // Y is (rows x bs) Col-Major: Y(i, j) at `i + j * rows`.
         // We want Y^T (bs x rows). Y^T(j, i) = Y(i, j).
         // To view `y_mat` as Y^T:
         // Element (r, c) of Y^T is Y(c, r).
         // Y is stored with stride rs=1 (step in c), cs=rows (step in r).
         // So Y(c, r) is at address `c * 1 + r * rows`.
         // For Y^T, we want element (r, c) = Y(c, r).
         // Address = `c * 1 + r * rows`.
         // Stride required: `r * rs_yt + c * cs_yt`.
         // MATCHING: rs_yt = rows, cs_yt = 1.
         // So we pass `y_mat` with rs=rows, cs=1 to treat it as Transposed Y.
         
         let mut w = vec![T::default(); bs * trail_cols];
         
         unsafe {
             // A_ptr points to A(0, end_k)
             let a_ptr = qr.storage().data().as_ptr().add(end_k * rows);
             
             let _ = gemm_dispatch_pointers(
                 bs, trail_cols, rows, // m=bs, n=trail_cols, k=rows
                 y_mat.as_ptr(),
                 rows as isize, 1, // A (Y^T): rs=rows, cs=1
                 
                 a_ptr, 
                 1, rows as isize, // B (A_trail): rs=1, cs=rows
                 
                 w.as_mut_ptr(),
                 1, bs as isize // C (W): rs=1, cs=bs
             );
         }
         
         // 2. W = -T^T * W
         // T is upper triangular (bs x bs).
         // W is (bs x trail_cols).
         // We compute W_new = -1.0 * T^T * W_old.
         
         let w_copy = w.clone();
         for c in 0..trail_cols {
             for r in 0..bs {
                 let mut sum = T::default();
                 // Row r of T^T is Col r of T.
                 // Elements T(k, r) for k <= r.
                 for k in 0..=r {
                      let t_val = t_mat[k + r * bs]; // T(k, r)
                      let w_val = w_copy[k + c * bs];
                      sum += t_val * w_val;
                 }
                 w[r + c * bs] = -sum; // Apply negation
             }
         }
         
         // 3. A_trail += Y * W
         // Y: (rows x bs). rs=1, cs=rows.
         // W: (bs x trail_cols). rs=1, cs=bs.
         // A_trail: (rows x trail_cols). rs=1, cs=rows.
         
         unsafe {
             let a_ptr = qr.storage_mut().data_mut().as_mut_ptr().add(end_k * rows);
             
             let _ = gemm_dispatch_pointers(
                 rows, trail_cols, bs,
                 y_mat.as_ptr(),
                 1, rows as isize, // Y
                 
                 w.as_ptr(),
                 1, bs as isize, // W
                 
                 a_ptr,
                 1, rows as isize // C += A * B
             );
         }
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
                        dot += (*self.qr.get(i, k).unwrap()).conj() * (*x.get(i, j).unwrap());
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
