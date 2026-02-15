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

impl<T: Scalar + num_traits::One, S: Storage<T>> PartialPivLU<T, S> {
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

        // Threshold for blocking
        const BLOCK_SIZE: usize = 64;

        if rows <= BLOCK_SIZE {
            Self::lu_unblocked(&mut lu, &mut p, 0, rows, &mut det_p);
        } else {
            Self::lu_blocked(&mut lu, &mut p, BLOCK_SIZE, &mut det_p);
        }

        Ok(Self {
            lu,
            p,
            det_p,
            _phantom: std::marker::PhantomData,
        })
    }

    // Unblocked LU (Panel Factorization)
    fn lu_unblocked(
        mat: &mut Matrix<T, DynamicStorage<T>>,
        p: &mut Vec<usize>,
        offset: usize,
        end: usize,
        det_p: &mut T,
    ) {
        let rows = mat.rows();
        let cols = mat.cols();
        let neg_one = T::from_usize(0) - T::from_usize(1);

        for k in offset..end {
            // println!("LU Unblocked k={}", k);
            let mut max_val = T::from_usize(0);
            let mut imax = k;

            for i in k..rows {
                let val = *mat.get(i, k).unwrap();
                let abs_val = val.abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    imax = i;
                }
            }

            if imax != k {
                for j in 0..cols {
                    let tmp = *mat.get(k, j).unwrap();
                    let val_imax = *mat.get(imax, j).unwrap();
                    *mat.get_mut(k, j).unwrap() = val_imax;
                    *mat.get_mut(imax, j).unwrap() = tmp;
                }
                p.swap(k, imax);
                *det_p *= neg_one;
            }

            let pivot = *mat.get(k, k).unwrap();
            if pivot != T::from_usize(0) {
                let inv_pivot = T::one() / pivot;
                for i in k + 1..rows {
                    *mat.get_mut(i, k).unwrap() *= inv_pivot;
                }

                // 3. Update trailing submatrix (Rank-1 update)
                // Vectorized AXPY: col(j) -= col(k) * factor
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("fma") {
                        Self::panel_update_avx(mat, k, end);
                        continue;
                    }
                }

                // Scalar Fallback
                for j in k + 1..end {
                    let factor = *mat.get(k, j).unwrap();
                    for i in k + 1..rows {
                        let val = *mat.get(i, k).unwrap();
                        *mat.get_mut(i, j).unwrap() -= val * factor;
                    }
                }
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn panel_update_avx(mat: &mut Matrix<T, DynamicStorage<T>>, k: usize, end: usize) {
        // println!("Panel update AVX k={} end={}", k, end);
        let rows = mat.rows();
        use std::any::TypeId;
        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f32>() {
            use std::arch::x86_64::*;
            unsafe {
                let start_ptr = mat.storage().data().as_ptr() as *const f32; // Assuming contiguous ColumnMajor
                let mut_ptr = mat.storage_mut().data_mut().as_mut_ptr() as *mut f32;
                let rows_stride = rows;

                let col_k_ptr = start_ptr.add(k * rows_stride);

                for j in k + 1..end {
                    let col_j_ptr = mut_ptr.add(j * rows_stride);

                    let factor_val = *col_j_ptr.add(k); // mat(k, j)
                    let factor_vec = _mm256_set1_ps(factor_val);

                    let mut i = k + 1;
                    while i + 7 < rows {
                        let val_k = _mm256_loadu_ps(col_k_ptr.add(i));
                        let mut val_j = _mm256_loadu_ps(col_j_ptr.add(i));

                        // val_j -= val_k * factor
                        val_j = _mm256_fnmadd_ps(val_k, factor_vec, val_j);

                        _mm256_storeu_ps(col_j_ptr.add(i), val_j);
                        i += 8;
                    }
                    for ii in i..rows {
                        let val_k = *col_k_ptr.add(ii);
                        *col_j_ptr.add(ii) -= val_k * factor_val;
                    }
                }
            }
        } else if tid == TypeId::of::<f64>() {
            use std::arch::x86_64::*;
            unsafe {
                let start_ptr = mat.storage().data().as_ptr() as *const f64;
                let mut_ptr = mat.storage_mut().data_mut().as_mut_ptr() as *mut f64;
                let rows_stride = rows;

                let col_k_ptr = start_ptr.add(k * rows_stride);

                for j in k + 1..end {
                    let col_j_ptr = mut_ptr.add(j * rows_stride);

                    let factor_val = *col_j_ptr.add(k);
                    let factor_vec = _mm256_set1_pd(factor_val);

                    let mut i = k + 1;
                    while i + 3 < rows {
                        let val_k = _mm256_loadu_pd(col_k_ptr.add(i));
                        let mut val_j = _mm256_loadu_pd(col_j_ptr.add(i));
                        val_j = _mm256_fnmadd_pd(val_k, factor_vec, val_j);
                        _mm256_storeu_pd(col_j_ptr.add(i), val_j);
                        i += 4;
                    }
                    for ii in i..rows {
                        let val_k = *col_k_ptr.add(ii);
                        *col_j_ptr.add(ii) -= val_k * factor_val;
                    }
                }
            }
        }
    }

    // Blocked LU (Driver)
    fn lu_blocked(
        mat: &mut Matrix<T, DynamicStorage<T>>,
        p: &mut Vec<usize>,
        bs: usize,
        det_p: &mut T,
    ) {
        let n = mat.rows();
        for k in (0..n).step_by(bs) {
            let kb = std::cmp::min(n - k, bs);
            let end = k + kb;

            Self::lu_unblocked(mat, p, k, end, det_p);

            if end < n {
                Self::trsm_unit_lower(mat, k, kb, n);
                Self::gemm_update(mat, k, kb, n);
            }
        }
    }

    // Solve L11 * B = B where B = A12
    fn trsm_unit_lower(mat: &mut Matrix<T, DynamicStorage<T>>, k: usize, kb: usize, n: usize) {
        // Optimized AVX path
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("fma") {
                Self::trsm_unit_lower_avx(mat, k, kb, n);
                return;
            }
        }

        // Scalar Fallback
        for j in k + kb..n {
            for i in 0..kb {
                let global_i = k + i;
                let mut val = *mat.get(global_i, j).unwrap();
                for l in 0..i {
                    let global_l = k + l;
                    let val_l = *mat.get(global_l, global_i).unwrap();
                    val -= val_l * (*mat.get(global_l, j).unwrap());
                }
                *mat.get_mut(global_i, j).unwrap() = val;
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn trsm_unit_lower_avx(mat: &mut Matrix<T, DynamicStorage<T>>, k: usize, kb: usize, n: usize) {
        let rows = mat.rows();
        use std::any::TypeId;
        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f32>() {
            use std::arch::x86_64::*;
            unsafe {
                // We use ColumnMajor layout property.
                // For each col j in k+kb..n
                let mut_ptr = mat.storage_mut().data_mut().as_mut_ptr() as *mut f32;
                let rows_stride = rows;

                // Base ptr to L block matrix (starts at k, k)
                // L col l starts at (k + l * rows) + k

                for j in k + kb..n {
                    let col_j_ptr = mut_ptr.add(j * rows_stride);

                    // For each column of L (l from 0 to kb)
                    for l in 0..kb {
                        // X[l] is known. load it.
                        let x_l = *col_j_ptr.add(k + l);
                        let x_l_vec = _mm256_set1_ps(x_l);

                        // L col ptr
                        let col_l_ptr = mut_ptr.add((k + l) * rows_stride);
                        // Update future rows i > l
                        // X[l+1..kb] -= L[l+1..kb, l] * X[l]

                        let mut i = l + 1;
                        // l is relative to k.
                        // We update indices k + i

                        while i + 7 < kb {
                            let l_vec = _mm256_loadu_ps(col_l_ptr.add(k + i));
                            let mut x_vec = _mm256_loadu_ps(col_j_ptr.add(k + i));

                            x_vec = _mm256_fnmadd_ps(l_vec, x_l_vec, x_vec);

                            _mm256_storeu_ps(col_j_ptr.add(k + i), x_vec);
                            i += 8;
                        }
                        for ii in i..kb {
                            let l_val = *col_l_ptr.add(k + ii);
                            *col_j_ptr.add(k + ii) -= l_val * x_l;
                        }
                    }
                }
            }
        } else if tid == TypeId::of::<f64>() {
            use std::arch::x86_64::*;
            unsafe {
                let mut_ptr = mat.storage_mut().data_mut().as_mut_ptr() as *mut f64;
                let rows_stride = rows;

                for j in k + kb..n {
                    let col_j_ptr = mut_ptr.add(j * rows_stride);
                    for l in 0..kb {
                        let x_l = *col_j_ptr.add(k + l);
                        let x_l_vec = _mm256_set1_pd(x_l);
                        let col_l_ptr = mut_ptr.add((k + l) * rows_stride);

                        let mut i = l + 1;
                        while i + 3 < kb {
                            let l_vec = _mm256_loadu_pd(col_l_ptr.add(k + i));
                            let mut x_vec = _mm256_loadu_pd(col_j_ptr.add(k + i));

                            x_vec = _mm256_fnmadd_pd(l_vec, x_l_vec, x_vec);

                            _mm256_storeu_pd(col_j_ptr.add(k + i), x_vec);
                            i += 4;
                        }
                        for ii in i..kb {
                            let l_val = *col_l_ptr.add(k + ii);
                            *col_j_ptr.add(k + ii) -= l_val * x_l;
                        }
                    }
                }
            }
        }
    }

    // A22 -= L21 * U12
    fn gemm_update(mat: &mut Matrix<T, DynamicStorage<T>>, k: usize, kb: usize, n: usize) {
        use crate::core::ops::gemm::gemm_blocked;

        let m_size = n - (k + kb); // Rows of A22 / L21
        let k_size = kb; // Cols of L21 / Rows of U12
        let n_size = n - (k + kb); // Cols of A22 / U12

        // L21 starts at (k+kb, k)
        let l21_ptr = mat.get(k + kb, k).unwrap() as *const T;
        let rs_a = 1;
        let cs_a = mat.rows() as isize;

        // U12 starts at (k, k+kb)
        let u12_ptr = mat.get(k, k + kb).unwrap() as *const T;
        let rs_b = 1;
        let cs_b = mat.rows() as isize;

        // A22 starts at (k+kb, k+kb)
        let c_ptr = mat.get_mut(k + kb, k + kb).unwrap() as *mut T;
        let rs_c = 1;
        let cs_c = mat.rows() as isize;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let tid = std::any::TypeId::of::<T>();
            // F32 Path
            if tid == std::any::TypeId::of::<f32>() && is_x86_feature_detected!("fma") {
                use crate::core::ops::gemm::arch::x86::asm_kernel::AsmFmaKernelF32;
                let alpha_f32: f32 = -1.0;
                unsafe {
                    let _ = gemm_blocked::<f32, AsmFmaKernelF32>(
                        m_size,
                        k_size,
                        n_size,
                        l21_ptr as *const f32,
                        rs_a,
                        cs_a,
                        u12_ptr as *const f32,
                        rs_b,
                        cs_b,
                        c_ptr as *mut f32,
                        rs_c,
                        cs_c,
                        alpha_f32,
                    );
                }
                return;
            }
            // F64 Path
            if tid == std::any::TypeId::of::<f64>() && is_x86_feature_detected!("fma") {
                use crate::core::ops::gemm::arch::x86::asm_kernel::AsmFmaKernelF64;
                let alpha_f64: f64 = -1.0;
                unsafe {
                    let _ = gemm_blocked::<f64, AsmFmaKernelF64>(
                        m_size,
                        k_size,
                        n_size,
                        l21_ptr as *const f64,
                        rs_a,
                        cs_a,
                        u12_ptr as *const f64,
                        rs_b,
                        cs_b,
                        c_ptr as *mut f64,
                        rs_c,
                        cs_c,
                        alpha_f64,
                    );
                }
                return;
            }
        }

        // Fallback GEMM
        for i in 0..m_size {
            for j in 0..n_size {
                let mut sum = T::from_usize(0);
                for l in 0..k_size {
                    sum += (*mat.get(k + kb + i, k + l).unwrap())
                        * (*mat.get(k + l, k + kb + j).unwrap());
                }
                *mat.get_mut(k + kb + i, k + kb + j).unwrap() -= sum;
            }
        }
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

        // Check for AVX optimization
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("fma") {
                unsafe {
                    Self::solve_avx(&self.lu, &mut x);
                }
                return Ok(x);
            }
        }

        // 2. Forward substitution for L (Ly = x)
        // L is unit lower triangular
        // Swap loops for cache locality (j, k, i)
        for j in 0..b0_cols {
            for k in 0..rows {
                let factor = *x.get(k, j).unwrap();
                for i in k + 1..rows {
                    *x.get_mut(i, j).unwrap() -= factor * (*self.lu.get(i, k).unwrap());
                }
            }
        }

        // 3. Backward substitution for U (Ux = y)
        for j in 0..b0_cols {
            for k in (0..rows).rev() {
                let pivot = *self.lu.get(k, k).unwrap();
                *x.get_mut(k, j).unwrap() /= pivot;
                let factor = *x.get(k, j).unwrap();
                for i in 0..k {
                    *x.get_mut(i, j).unwrap() -= factor * (*self.lu.get(i, k).unwrap());
                }
            }
        }

        Ok(x)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe fn solve_avx(lu: &Matrix<T, DynamicStorage<T>>, x: &mut Matrix<T, DynamicStorage<T>>) {
        use std::any::TypeId;
        let tid = TypeId::of::<T>();
        let rows = lu.rows();
        let cols = x.cols();

        if tid == TypeId::of::<f32>() {
            use std::arch::x86_64::*;
            let lu_ptr = lu.storage().data().as_ptr() as *const f32;
            let x_ptr = x.storage_mut().data_mut().as_mut_ptr() as *mut f32;

            // Block over j (columns of x)
            let mut j = 0;
            while j + 8 <= cols {
                let x_ptr0 = x_ptr.add(j * rows);
                let x_ptr1 = x_ptr.add((j + 1) * rows);
                let x_ptr2 = x_ptr.add((j + 2) * rows);
                let x_ptr3 = x_ptr.add((j + 3) * rows);
                let x_ptr4 = x_ptr.add((j + 4) * rows);
                let x_ptr5 = x_ptr.add((j + 5) * rows);
                let x_ptr6 = x_ptr.add((j + 6) * rows);
                let x_ptr7 = x_ptr.add((j + 7) * rows);

                // Forward L
                for k in 0..rows {
                    let f0 = *x_ptr0.add(k);
                    let f0_vec = _mm256_set1_ps(f0);
                    let f1 = *x_ptr1.add(k);
                    let f1_vec = _mm256_set1_ps(f1);
                    let f2 = *x_ptr2.add(k);
                    let f2_vec = _mm256_set1_ps(f2);
                    let f3 = *x_ptr3.add(k);
                    let f3_vec = _mm256_set1_ps(f3);
                    let f4 = *x_ptr4.add(k);
                    let f4_vec = _mm256_set1_ps(f4);
                    let f5 = *x_ptr5.add(k);
                    let f5_vec = _mm256_set1_ps(f5);
                    let f6 = *x_ptr6.add(k);
                    let f6_vec = _mm256_set1_ps(f6);
                    let f7 = *x_ptr7.add(k);
                    let f7_vec = _mm256_set1_ps(f7);

                    let lu_col = lu_ptr.add(k * rows);

                    let mut i = k + 1;
                    while i + 7 < rows {
                        let l_vec = _mm256_loadu_ps(lu_col.add(i));

                        _mm256_storeu_ps(
                            x_ptr0.add(i),
                            _mm256_fnmadd_ps(l_vec, f0_vec, _mm256_loadu_ps(x_ptr0.add(i))),
                        );
                        _mm256_storeu_ps(
                            x_ptr1.add(i),
                            _mm256_fnmadd_ps(l_vec, f1_vec, _mm256_loadu_ps(x_ptr1.add(i))),
                        );
                        _mm256_storeu_ps(
                            x_ptr2.add(i),
                            _mm256_fnmadd_ps(l_vec, f2_vec, _mm256_loadu_ps(x_ptr2.add(i))),
                        );
                        _mm256_storeu_ps(
                            x_ptr3.add(i),
                            _mm256_fnmadd_ps(l_vec, f3_vec, _mm256_loadu_ps(x_ptr3.add(i))),
                        );
                        _mm256_storeu_ps(
                            x_ptr4.add(i),
                            _mm256_fnmadd_ps(l_vec, f4_vec, _mm256_loadu_ps(x_ptr4.add(i))),
                        );
                        _mm256_storeu_ps(
                            x_ptr5.add(i),
                            _mm256_fnmadd_ps(l_vec, f5_vec, _mm256_loadu_ps(x_ptr5.add(i))),
                        );
                        _mm256_storeu_ps(
                            x_ptr6.add(i),
                            _mm256_fnmadd_ps(l_vec, f6_vec, _mm256_loadu_ps(x_ptr6.add(i))),
                        );
                        _mm256_storeu_ps(
                            x_ptr7.add(i),
                            _mm256_fnmadd_ps(l_vec, f7_vec, _mm256_loadu_ps(x_ptr7.add(i))),
                        );

                        i += 8;
                    }
                    for ii in i..rows {
                        let l_val = *lu_col.add(ii);
                        *x_ptr0.add(ii) -= l_val * f0;
                        *x_ptr1.add(ii) -= l_val * f1;
                        *x_ptr2.add(ii) -= l_val * f2;
                        *x_ptr3.add(ii) -= l_val * f3;
                        *x_ptr4.add(ii) -= l_val * f4;
                        *x_ptr5.add(ii) -= l_val * f5;
                        *x_ptr6.add(ii) -= l_val * f6;
                        *x_ptr7.add(ii) -= l_val * f7;
                    }
                }

                // Backward U
                for k in (0..rows).rev() {
                    let pivot = *lu_ptr.add(k * rows + k);
                    *x_ptr0.add(k) /= pivot;
                    let f0 = *x_ptr0.add(k);
                    let f0_vec = _mm256_set1_ps(f0);
                    *x_ptr1.add(k) /= pivot;
                    let f1 = *x_ptr1.add(k);
                    let f1_vec = _mm256_set1_ps(f1);
                    *x_ptr2.add(k) /= pivot;
                    let f2 = *x_ptr2.add(k);
                    let f2_vec = _mm256_set1_ps(f2);
                    *x_ptr3.add(k) /= pivot;
                    let f3 = *x_ptr3.add(k);
                    let f3_vec = _mm256_set1_ps(f3);
                    *x_ptr4.add(k) /= pivot;
                    let f4 = *x_ptr4.add(k);
                    let f4_vec = _mm256_set1_ps(f4);
                    *x_ptr5.add(k) /= pivot;
                    let f5 = *x_ptr5.add(k);
                    let f5_vec = _mm256_set1_ps(f5);
                    *x_ptr6.add(k) /= pivot;
                    let f6 = *x_ptr6.add(k);
                    let f6_vec = _mm256_set1_ps(f6);
                    *x_ptr7.add(k) /= pivot;
                    let f7 = *x_ptr7.add(k);
                    let f7_vec = _mm256_set1_ps(f7);

                    let lu_col = lu_ptr.add(k * rows);

                    let mut i = 0;
                    while i + 7 < k {
                        let u_vec = _mm256_loadu_ps(lu_col.add(i));

                        _mm256_storeu_ps(
                            x_ptr0.add(i),
                            _mm256_fnmadd_ps(u_vec, f0_vec, _mm256_loadu_ps(x_ptr0.add(i))),
                        );
                        _mm256_storeu_ps(
                            x_ptr1.add(i),
                            _mm256_fnmadd_ps(u_vec, f1_vec, _mm256_loadu_ps(x_ptr1.add(i))),
                        );
                        _mm256_storeu_ps(
                            x_ptr2.add(i),
                            _mm256_fnmadd_ps(u_vec, f2_vec, _mm256_loadu_ps(x_ptr2.add(i))),
                        );
                        _mm256_storeu_ps(
                            x_ptr3.add(i),
                            _mm256_fnmadd_ps(u_vec, f3_vec, _mm256_loadu_ps(x_ptr3.add(i))),
                        );
                        _mm256_storeu_ps(
                            x_ptr4.add(i),
                            _mm256_fnmadd_ps(u_vec, f4_vec, _mm256_loadu_ps(x_ptr4.add(i))),
                        );
                        _mm256_storeu_ps(
                            x_ptr5.add(i),
                            _mm256_fnmadd_ps(u_vec, f5_vec, _mm256_loadu_ps(x_ptr5.add(i))),
                        );
                        _mm256_storeu_ps(
                            x_ptr6.add(i),
                            _mm256_fnmadd_ps(u_vec, f6_vec, _mm256_loadu_ps(x_ptr6.add(i))),
                        );
                        _mm256_storeu_ps(
                            x_ptr7.add(i),
                            _mm256_fnmadd_ps(u_vec, f7_vec, _mm256_loadu_ps(x_ptr7.add(i))),
                        );

                        i += 8;
                    }
                    for ii in i..k {
                        let u_val = *lu_col.add(ii);
                        *x_ptr0.add(ii) -= u_val * f0;
                        *x_ptr1.add(ii) -= u_val * f1;
                        *x_ptr2.add(ii) -= u_val * f2;
                        *x_ptr3.add(ii) -= u_val * f3;
                        *x_ptr4.add(ii) -= u_val * f4;
                        *x_ptr5.add(ii) -= u_val * f5;
                        *x_ptr6.add(ii) -= u_val * f6;
                        *x_ptr7.add(ii) -= u_val * f7;
                    }
                }

                j += 8;
            }

            // Remainder loop
            for j_rem in j..cols {
                let x_col = x_ptr.add(j_rem * rows);
                // Forward L
                for k in 0..rows {
                    let factor = *x_col.add(k);
                    let factor_vec = _mm256_set1_ps(factor);
                    let lu_col = lu_ptr.add(k * rows);

                    let mut i = k + 1;
                    while i + 7 < rows {
                        let l_vec = _mm256_loadu_ps(lu_col.add(i));
                        let mut x_vec = _mm256_loadu_ps(x_col.add(i));
                        x_vec = _mm256_fnmadd_ps(l_vec, factor_vec, x_vec);
                        _mm256_storeu_ps(x_col.add(i), x_vec);
                        i += 8;
                    }
                    for ii in i..rows {
                        *x_col.add(ii) -= (*lu_col.add(ii)) * factor;
                    }
                }
                // Backward U
                for k in (0..rows).rev() {
                    let pivot = *lu_ptr.add(k * rows + k);
                    *x_col.add(k) /= pivot;
                    let factor = *x_col.add(k);
                    let factor_vec = _mm256_set1_ps(factor);
                    let lu_col = lu_ptr.add(k * rows);
                    let mut i = 0;
                    while i + 7 < k {
                        let u_vec = _mm256_loadu_ps(lu_col.add(i));
                        let mut x_vec = _mm256_loadu_ps(x_col.add(i));
                        x_vec = _mm256_fnmadd_ps(u_vec, factor_vec, x_vec);
                        _mm256_storeu_ps(x_col.add(i), x_vec);
                        i += 8;
                    }
                    for ii in i..k {
                        *x_col.add(ii) -= (*lu_col.add(ii)) * factor;
                    }
                }
            }
        } else if tid == TypeId::of::<f64>() {
            // ... f64 scalar/simple vec fallback for now ...
            // Or apply same blocking logic (4 columns for f64?)
            // I'll keep the previous f64 simple vec impl but could block 4 cols.
            // For now, I'll paste the previous f64 implementation to avoid error.
            use std::arch::x86_64::*;
            let lu_ptr = lu.storage().data().as_ptr() as *const f64;
            let x_ptr = x.storage_mut().data_mut().as_mut_ptr() as *mut f64;

            for j in 0..cols {
                let x_col = x_ptr.add(j * rows);

                // Forward L
                for k in 0..rows {
                    let factor = *x_col.add(k);
                    let factor_vec = _mm256_set1_pd(factor);
                    let lu_col = lu_ptr.add(k * rows);

                    let mut i = k + 1;
                    while i + 3 < rows {
                        let l_vec = _mm256_loadu_pd(lu_col.add(i));
                        let mut x_vec = _mm256_loadu_pd(x_col.add(i));
                        x_vec = _mm256_fnmadd_pd(l_vec, factor_vec, x_vec);
                        _mm256_storeu_pd(x_col.add(i), x_vec);
                        i += 4;
                    }
                    for ii in i..rows {
                        *x_col.add(ii) -= (*lu_col.add(ii)) * factor;
                    }
                }

                // Backward U
                for k in (0..rows).rev() {
                    let pivot = *lu_ptr.add(k * rows + k);
                    *x_col.add(k) /= pivot;
                    let factor = *x_col.add(k);
                    let factor_vec = _mm256_set1_pd(factor);
                    let lu_col = lu_ptr.add(k * rows);

                    let mut i = 0;
                    while i + 3 < k {
                        let u_vec = _mm256_loadu_pd(lu_col.add(i));
                        let mut x_vec = _mm256_loadu_pd(x_col.add(i));
                        x_vec = _mm256_fnmadd_pd(u_vec, factor_vec, x_vec);
                        _mm256_storeu_pd(x_col.add(i), x_vec);
                        i += 4;
                    }
                    for ii in i..k {
                        *x_col.add(ii) -= (*lu_col.add(ii)) * factor;
                    }
                }
            }
        }
    }
}
