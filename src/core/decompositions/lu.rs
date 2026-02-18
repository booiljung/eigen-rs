//! Partial Pivoting LU decomposition (PA = LU).

use crate::core::matrix::Matrix;
use num_traits::Zero;
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
        // Increased to 64 to improve AVX2 utilization and reduce unblocked overhead
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
            let mut max_val = T::Real::zero();
            let mut imax = k;

            for i in k..rows {
                let val = *mat.get(i, k).unwrap();
                let abs_val = val.abs(); // Returns T::Real
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
                
                // 2. Scale Column k (below diagonal)
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    // Runtime check for AVX2/FMA
                    if (std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() 
                        || std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>())
                        && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
                    {
                        unsafe {
                            Self::scale_col_avx(mat, k, inv_pivot);
                        }
                    } else {
                        // Scalar Fallback
                        for i in k + 1..rows {
                            *mat.get_mut(i, k).unwrap() *= inv_pivot;
                        }
                    }
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                for i in k + 1..rows {
                    *mat.get_mut(i, k).unwrap() *= inv_pivot;
                }

                // 3. Update trailing submatrix (Rank-1 update)
                // Vectorized AXPY: col(j) -= col(k) * factor
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if (std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
                        || std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>())
                        && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
                    {
                        unsafe {
                            Self::panel_update_avx(mat, k, end);
                        }
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
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn scale_col_avx(mat: &mut Matrix<T, DynamicStorage<T>>, k: usize, val: T) {
        let rows = mat.rows();
        use std::any::TypeId;
        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f32>() {
            use std::arch::x86_64::*;
            unsafe {
                let mut_ptr = mat.storage_mut().data_mut().as_mut_ptr() as *mut f32;
                let rows_stride = rows;
                // col k starts at k * rows
                let col_ptr = mut_ptr.add(k * rows_stride);
                
                // We want elements k+1..rows
                // Access is contiguous: col_ptr[k+1], col_ptr[k+2] ...
                
                let val_f32 = *(&val as *const T as *const f32);
                let val_vec = _mm256_set1_ps(val_f32);
                
                let mut i = k + 1;
                while i + 7 < rows {
                    let ptr = col_ptr.add(i);
                    let v = _mm256_loadu_ps(ptr);
                    _mm256_storeu_ps(ptr, _mm256_mul_ps(v, val_vec));
                    i += 8;
                }
                for ii in i..rows {
                    *col_ptr.add(ii) *= val_f32;
                }
            }
        } else if tid == TypeId::of::<f64>() {
             use std::arch::x86_64::*;
            unsafe {
                let mut_ptr = mat.storage_mut().data_mut().as_mut_ptr() as *mut f64;
                let rows_stride = rows;
                let col_ptr = mut_ptr.add(k * rows_stride);
                
                let val_f64 = *(&val as *const T as *const f64);
                let val_vec = _mm256_set1_pd(val_f64);
                
                let mut i = k + 1;
                while i + 3 < rows {
                    let ptr = col_ptr.add(i);
                    let v = _mm256_loadu_pd(ptr);
                    _mm256_storeu_pd(ptr, _mm256_mul_pd(v, val_vec));
                    i += 4;
                }
                for ii in i..rows {
                    *col_ptr.add(ii) *= val_f64;
                }
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn panel_update_avx(mat: &mut Matrix<T, DynamicStorage<T>>, k: usize, end: usize) {
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
            if (std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
                || std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>())
                && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
            {
                unsafe {
                    Self::trsm_unit_lower_avx(mat, k, kb, n);
                }
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
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn trsm_unit_lower_avx(mat: &mut Matrix<T, DynamicStorage<T>>, k: usize, kb: usize, n: usize) {
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

    // Blocked Backward Substitution (U * X = Y)
    fn solve_u_blocked(&self, x: &mut Matrix<T, DynamicStorage<T>>) {
        let rows = self.lu.rows();
        let cols = x.cols();
        let lu_ptr = self.lu.storage().data().as_ptr();
        let x_ptr = x.storage_mut().data_mut().as_mut_ptr();
        
        let block_size = 64;

        // Check AVX availability once
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let use_avx = (std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() 
                      || std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>())
                      && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let use_avx = false;

        // Iterate backwards
        for k in (0..rows).step_by(block_size).rev() {
            let kb = std::cmp::min(block_size, rows - k); // logic needs care for step_by rev?
            // rev() of step_by: e.g. 0, 64, 128 (N=200) -> 0, 64, 128. Rev -> 128, 64, 0.
            // When k=128, kb = min(64, 200-128=72) = 64.
            // When k=0, kb = 64.
            // Wait, step_by starts from 0.
            // If N=200. k=0, 64, 128, 192.
            // rev: 192, 128, 64, 0.
            // k=192: kb = min(64, 200-192=8) = 8. Correct.
            
            // 1. Solve diagonal block
            unsafe {
                let l_diag_ptr = lu_ptr.add(k * rows + k);
                let x_block_ptr = x_ptr.add(k);
                
                let done = if use_avx {
                     #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                     {
                        Self::solve_u_avx_ptr(kb, cols, l_diag_ptr, 1, rows, x_block_ptr, 1, rows);
                        true
                     }
                     #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                     false
                } else {
                    false
                };

                if !done {
                    // Scalar fallback
                    for j in 0..cols {
                        let col_ptr = x_block_ptr.add(j * rows);
                        // Backward U in block
                        for i in (0..kb).rev() {
                            let pivot = *l_diag_ptr.add(i * rows + i);
                            *col_ptr.add(i) /= pivot;
                            let factor = *col_ptr.add(i);
                            let u_col = l_diag_ptr.add(i * rows);
                            for ii in 0..i {
                                *col_ptr.add(ii) -= *u_col.add(ii) * factor;
                            }
                        }
                    }
                }
            }

            // 2. GEMM Update for Upper part
            // X(0..k, :) -= U(0..k, k..k+kb) * X(k..k+kb, :)
            if k > 0 {
                let m = k; // rows to update
                let n_gemm = cols;
                let k_gemm = kb;
                
                unsafe {
                    let a_ptr = lu_ptr.add(k * rows); // U(0, k) -> col k start.
                    // But we want U(0..k, k..k+kb).
                    // This is block starting at (0, k).
                    // Pointer to (0, k): lu_ptr.add(k * rows + 0). Correct. (ColMajor: col*rows + row)
                    
                    let b_ptr = x_ptr.add(k); // X block (k..k+kb)
                    let c_ptr = x_ptr; // X top (0..k)

                     #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if use_avx {
                         let tid = std::any::TypeId::of::<T>();
                        #[cfg(target_feature = "avx2")]
                        if tid == std::any::TypeId::of::<f32>() {
                             use crate::core::ops::gemm::arch::x86::asm_kernel::AsmFmaKernelF32;
                             use crate::core::ops::gemm::gemm_blocked;
                             let _ = gemm_blocked::<f32, AsmFmaKernelF32>(
                                m, k_gemm, n_gemm,
                                a_ptr as *const f32, 1, rows as isize,
                                b_ptr as *const f32, 1, rows as isize,
                                c_ptr as *mut f32, 1, rows as isize,
                                -1.0, 
                             );
                        } else if tid == std::any::TypeId::of::<f64>() {
                             use crate::core::ops::gemm::arch::x86::asm_kernel::AsmFmaKernelF64;
                             use crate::core::ops::gemm::gemm_blocked;
                             let _ = gemm_blocked::<f64, AsmFmaKernelF64>(
                                m, k_gemm, n_gemm,
                                a_ptr as *const f64, 1, rows as isize,
                                b_ptr as *const f64, 1, rows as isize,
                                c_ptr as *mut f64, 1, rows as isize,
                                -1.0, 
                             );
                        }
                    } else {
                         // Generic GEMM Fallback
                         for j in 0..n_gemm {
                             let x1_col = b_ptr.add(j * rows);
                             let x2_col = c_ptr.add(j * rows);
                             for l in 0..k_gemm { 
                                 let factor = *x1_col.add(l);
                                 let u_col = a_ptr.add(l * rows);
                                 for i in 0..m { 
                                     *x2_col.add(i) -= *u_col.add(i) * factor;
                                 }
                             }
                         }
                    }
                     #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                    {
                         for j in 0..n_gemm {
                             let x1_col = b_ptr.add(j * rows);
                             let x2_col = c_ptr.add(j * rows);
                             for l in 0..k_gemm { 
                                 let factor = *x1_col.add(l);
                                 let u_col = a_ptr.add(l * rows);
                                 for i in 0..m { 
                                     *x2_col.add(i) -= *u_col.add(i) * factor;
                                 }
                             }
                         }
                    }
                }
            }
        }
    }
    fn solve_l_blocked(&self, x: &mut Matrix<T, DynamicStorage<T>>) {
        let rows = self.lu.rows();
        let cols = x.cols();
        let lu_ptr = self.lu.storage().data().as_ptr();
        let x_ptr = x.storage_mut().data_mut().as_mut_ptr();
        
        let block_size = 64;

        // Check AVX availability once
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let use_avx = (std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() 
                      || std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>())
                      && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let use_avx = false;

        for k in (0..rows).step_by(block_size) {
            let kb = std::cmp::min(block_size, rows - k);

            // 1. Solve diagonal block
            // L(k..k+kb, k..k+kb) * X(k..k+kb, :) = X(...)
            unsafe {
                let l_diag_ptr = lu_ptr.add(k * rows + k);
                let x_block_ptr = x_ptr.add(k); // row k
                
                let done = if use_avx {
                     #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                     {
                        Self::solve_l_avx_ptr(kb, cols, l_diag_ptr, 1, rows, x_block_ptr, 1, rows);
                        true
                     }
                     #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                     false
                } else {
                    false
                };

                if !done {
                    // Scalar fallback for diagonal block
                    for j in 0..cols {
                        let col_ptr = x_block_ptr.add(j * rows);
                        for i in 0..kb { // relative row in block
                            let factor = *col_ptr.add(i);
                            // Subtract L column from remaining elements in block
                            let l_col = l_diag_ptr.add(i * rows); 
                            for ii in i+1..kb {
                                *col_ptr.add(ii) -= *l_col.add(ii) * factor;
                            }
                        }
                    }
                }
            }

            // 2. GEMM Update
            // X(k+kb..n, :) -= L(k+kb..n, k..k+kb) * X(k..k+kb, :)
            if k + kb < rows {
                let m = rows - (k + kb);
                let n_gemm = cols;
                let k_gemm = kb;
                
                unsafe {
                    let a_ptr = lu_ptr.add(k * rows + (k + kb)); // L21
                    let b_ptr = x_ptr.add(k); // X1 (block we just solved)
                    let c_ptr = x_ptr.add(k + kb); // X2 (target)

                    // Dispatch GEMM
                     #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if use_avx {
                        // specialized AVX GEMM
                         let tid = std::any::TypeId::of::<T>();
                        #[cfg(target_feature = "avx2")]
                        if tid == std::any::TypeId::of::<f32>() {
                             use crate::core::ops::gemm::arch::x86::asm_kernel::AsmFmaKernelF32;
                             use crate::core::ops::gemm::gemm_blocked;
                             let _ = gemm_blocked::<f32, AsmFmaKernelF32>(
                                m, k_gemm, n_gemm,
                                a_ptr as *const f32, 1, rows as isize,
                                b_ptr as *const f32, 1, rows as isize,
                                c_ptr as *mut f32, 1, rows as isize,
                                -1.0, 
                             );
                        } else if tid == std::any::TypeId::of::<f64>() {
                             use crate::core::ops::gemm::arch::x86::asm_kernel::AsmFmaKernelF64;
                             use crate::core::ops::gemm::gemm_blocked;
                             let _ = gemm_blocked::<f64, AsmFmaKernelF64>(
                                m, k_gemm, n_gemm,
                                a_ptr as *const f64, 1, rows as isize,
                                b_ptr as *const f64, 1, rows as isize,
                                c_ptr as *mut f64, 1, rows as isize,
                                -1.0, 
                             );
                        }
                    } else {
                        // Generic GEMM Fallback
                         // Use simple loop for now to avoid importing generic kernel machinery if complex
                         // Or implement simple blocked loop here
                         // X2 -= L21 * X1
                         // L21: m x k_gemm
                         // X1: k_gemm x n_gemm
                         // X2: m x n_gemm
                         // For cache locality, block over n_gemm?
                         for j in 0..n_gemm {
                             let x1_col = b_ptr.add(j * rows);
                             let x2_col = c_ptr.add(j * rows);
                             for l in 0..k_gemm { // cols of L21
                                 let factor = *x1_col.add(l);
                                 let l_col = a_ptr.add(l * rows);
                                 for i in 0..m { // rows of L21
                                     *x2_col.add(i) -= *l_col.add(i) * factor;
                                 }
                             }
                         }
                    }
                     #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                    {
                         for j in 0..n_gemm {
                             let x1_col = b_ptr.add(j * rows);
                             let x2_col = c_ptr.add(j * rows);
                             for l in 0..k_gemm { 
                                 let factor = *x1_col.add(l);
                                 let l_col = a_ptr.add(l * rows);
                                 for i in 0..m { 
                                     *x2_col.add(i) -= *l_col.add(i) * factor;
                                 }
                             }
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
            #[cfg(target_feature = "avx2")]
            if tid == std::any::TypeId::of::<f32>() {
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
            #[cfg(target_feature = "avx2")]
            if tid == std::any::TypeId::of::<f64>() {
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

        // Blocked Solve
        self.solve_l_blocked(&mut x);
        self.solve_u_blocked(&mut x);

        Ok(x)
    }



    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn solve_l_avx_ptr(
        rows: usize,
        cols: usize,
        lu_ptr: *const T,
        rs_lu: usize,
        cs_lu: usize,
        x_ptr: *mut T,
        rs_x: usize,
        cs_x: usize,
    ) {
        use std::any::TypeId;
        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f32>() {
            use std::arch::x86_64::*;
            let lu_ptr = lu_ptr as *const f32;
            let x_ptr = x_ptr as *mut f32;
            
            // Block over j (columns of x)
            let mut j = 0;
            while j + 8 <= cols {
                let x_ptr0 = x_ptr.add(j * cs_x);
                let x_ptr1 = x_ptr.add((j + 1) * cs_x);
                let x_ptr2 = x_ptr.add((j + 2) * cs_x);
                let x_ptr3 = x_ptr.add((j + 3) * cs_x);
                let x_ptr4 = x_ptr.add((j + 4) * cs_x);
                let x_ptr5 = x_ptr.add((j + 5) * cs_x);
                let x_ptr6 = x_ptr.add((j + 6) * cs_x);
                let x_ptr7 = x_ptr.add((j + 7) * cs_x);

                // Forward L
                for k in 0..rows {
                    let f0 = *x_ptr0.add(k * rs_x);
                    let f0_vec = _mm256_set1_ps(f0);
                    let f1 = *x_ptr1.add(k * rs_x);
                    let f1_vec = _mm256_set1_ps(f1);
                    let f2 = *x_ptr2.add(k * rs_x);
                    let f2_vec = _mm256_set1_ps(f2);
                    let f3 = *x_ptr3.add(k * rs_x);
                    let f3_vec = _mm256_set1_ps(f3);
                    let f4 = *x_ptr4.add(k * rs_x);
                    let f4_vec = _mm256_set1_ps(f4);
                    let f5 = *x_ptr5.add(k * rs_x);
                    let f5_vec = _mm256_set1_ps(f5);
                    let f6 = *x_ptr6.add(k * rs_x);
                    let f6_vec = _mm256_set1_ps(f6);
                    let f7 = *x_ptr7.add(k * rs_x);
                    let f7_vec = _mm256_set1_ps(f7);

                    let lu_col = lu_ptr.add(k * cs_lu);

                    let mut i = k + 1;
                    while i + 7 < rows {
                        // Assuming rs_lu == 1 (contiguous column)
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
                j += 8;
            }

            // Remainder loop
            for j_rem in j..cols {
                let x_col = x_ptr.add(j_rem * cs_x);
                // Forward L
                for k in 0..rows {
                    let factor = *x_col.add(k * rs_x);
                    let factor_vec = _mm256_set1_ps(factor);
                    let lu_col = lu_ptr.add(k * cs_lu);

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
            }
        } else if tid == TypeId::of::<f64>() {
            use std::arch::x86_64::*;
            let lu_ptr = lu_ptr as *const f64;
            let x_ptr = x_ptr as *mut f64;

            for j in 0..cols {
                let x_col = x_ptr.add(j * cs_x);
                // Forward L
                for k in 0..rows {
                     let factor = *x_col.add(k * rs_x);
                     let factor_vec = _mm256_set1_pd(factor);
                     let lu_col = lu_ptr.add(k * cs_lu);
                     
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
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn solve_u_avx_ptr(
        rows: usize,
        cols: usize,
        lu_ptr: *const T,
        rs_lu: usize,
        cs_lu: usize,
        x_ptr: *mut T,
        rs_x: usize,
        cs_x: usize,
    ) {
         use std::any::TypeId;
        let tid = TypeId::of::<T>();

        if tid == TypeId::of::<f32>() {
            use std::arch::x86_64::*;
            let lu_ptr = lu_ptr as *const f32;
            let x_ptr = x_ptr as *mut f32;

            let mut j = 0;
            while j + 8 <= cols {
                let x_ptr0 = x_ptr.add(j * cs_x);
                let x_ptr1 = x_ptr.add((j + 1) * cs_x);
                let x_ptr2 = x_ptr.add((j + 2) * cs_x);
                let x_ptr3 = x_ptr.add((j + 3) * cs_x);
                let x_ptr4 = x_ptr.add((j + 4) * cs_x);
                let x_ptr5 = x_ptr.add((j + 5) * cs_x);
                let x_ptr6 = x_ptr.add((j + 6) * cs_x);
                let x_ptr7 = x_ptr.add((j + 7) * cs_x);

                // Backward U
                for k in (0..rows).rev() {
                    let pivot = *lu_ptr.add(k * cs_lu + k);
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

                    let lu_col = lu_ptr.add(k * cs_lu);

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
                let x_col = x_ptr.add(j_rem * cs_x);
                 // Backward U
                for k in (0..rows).rev() {
                    let pivot = *lu_ptr.add(k * cs_lu + k);
                    *x_col.add(k) /= pivot;
                    let factor = *x_col.add(k);
                    let factor_vec = _mm256_set1_ps(factor);
                    let lu_col = lu_ptr.add(k * cs_lu);
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
            use std::arch::x86_64::*;
            let lu_ptr = lu_ptr as *const f64;
            let x_ptr = x_ptr as *mut f64;

            for j in 0..cols {
                 let x_col = x_ptr.add(j * cs_x);
                  // Backward U
                for k in (0..rows).rev() {
                     let pivot = *lu_ptr.add(k * cs_lu + k);
                     *x_col.add(k) /= pivot;
                     let factor = *x_col.add(k);
                     let factor_vec = _mm256_set1_pd(factor);
                     let lu_col = lu_ptr.add(k * cs_lu);
                     
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
