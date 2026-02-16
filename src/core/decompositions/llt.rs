//! LLT Cholesky decomposition ($A = LL^T$).
//! Best suited for symmetric/Hermitian positive definite matrices.

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::DynamicStorage;
use crate::core::storage::Storage;

/// Result of an LLT Cholesky decomposition.
pub struct LLT<T: Scalar, S: Storage<T>> {
    l: Matrix<T, DynamicStorage<T>>,
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar + num_traits::One + 'static, S: Storage<T> + 'static> LLT<T, S> {
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

        // Threshold for blocking.
        // Reduced to 32 to improve L1 cache hit rate for TRSM/SYRK
        const BLOCK_SIZE: usize = 32;

        if rows <= BLOCK_SIZE {
            Self::llt_unblocked(&mut l, 0, rows)?;
            // Zero out upper triangle explicitly
            for j in 0..rows {
                for i in 0..j {
                    *l.get_mut(i, j).unwrap() = T::default();
                }
            }
        } else {
            Self::llt_blocked(&mut l, BLOCK_SIZE)?;
        }

        Ok(Self {
            l,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Unblocked LLT (Right-Looking / Outer Product) with AVX2 optimization.
    /// Operates on the submatrix starting at (offset, offset) with size `size`.
    /// Unblocked LLT (Right-Looking / Outer Product) with AVX2 optimization.
    /// Operates on the submatrix starting at (offset, offset) with size `size`.
    fn llt_unblocked(
        mat: &mut Matrix<T, DynamicStorage<T>>,
        offset: usize,
        size: usize,
    ) -> Result<(), String> {
        // Optimization for f32 AVX2
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(target_feature = "avx2")]
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                unsafe {
                    Self::llt_unblocked_f32_avx(mat, offset, size)?;
                }
                return Ok(());
            }
        }

        // Fallback (Right-Looking Scalar)
        // Better cache locality for ColMajor than Inner-Product
        for j in 0..size {
            let global_j = offset + j;

            // 1. Square Root Diagonal
            let diag_val = *mat.get(global_j, global_j).unwrap();
            if diag_val <= T::from_usize(0) {
                return Err("Matrix is not positive definite".to_string());
            }
            let l_jj = diag_val.sqrt();
            *mat.get_mut(global_j, global_j).unwrap() = l_jj;

            // 2. Scale Column j (below diagonal)
            let inv_jj = T::one() / l_jj;
            for i in j + 1..size {
                *mat.get_mut(offset + i, global_j).unwrap() *= inv_jj;
            }

            // 3. Update Trailing Matrix (Rank-1 Update)
            // A(j+1:n, j+1:n) -= L(j+1:n, j) * L(j+1:n, j)^T
            // Symmetry: Update lower triangle only.
            for k in j + 1..size {
                // Col k of trailing submatrix
                let global_k = offset + k;
                let scale = *mat.get(global_k, global_j).unwrap(); // L_kj

                for i in k..size {
                    // Row i of trailing submatrix (>= k)
                    let global_i = offset + i;
                    let val = *mat.get(global_i, global_j).unwrap(); // L_ij
                    *mat.get_mut(global_i, global_k).unwrap() -= val * scale;
                }
            }
        }
        Ok(())
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn llt_unblocked_f32_avx(
        mat: &mut Matrix<T, DynamicStorage<T>>,
        offset: usize,
        size: usize,
    ) -> Result<(), String> {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let ptr = mat.storage_mut().data_mut().as_mut_ptr() as *mut f32;
        let rows = mat.rows();

        for j in 0..size {
            let global_j = offset + j;

            // 1. Diagonal
            let diag_ptr = ptr.add(global_j * rows + global_j);
            let mut diag_val = *diag_ptr;
            if diag_val <= 0.0 {
                return Err("Matrix is not positive definite".to_string());
            }
            let l_jj = diag_val.sqrt();
            *diag_ptr = l_jj;

            let inv_jj = 1.0 / l_jj;
            let inv_vec = _mm256_set1_ps(inv_jj);

            // 2. Scale Column j (below diagonal) using AVX
            // Elements: offset + i, global_j for i in j+1..size
            // Start row: offset + j + 1
            let col_start = offset + j + 1;
            let col_end = offset + size; // = offset + size
            let count = if col_end > col_start {
                col_end - col_start
            } else {
                0
            };

            let col_ptr = ptr.add(global_j * rows + col_start);

            let mut x = 0;
            while x + 7 < count {
                let p = col_ptr.add(x);
                let v = _mm256_loadu_ps(p);
                _mm256_storeu_ps(p, _mm256_mul_ps(v, inv_vec));
                x += 8;
            }
            while x < count {
                *col_ptr.add(x) *= inv_jj;
                x += 1;
            }

            // 3. Update Trailing Matrix: Rank-1 Update
            // For k from j+1 to size-1
            //   col_k[k..size] -= col_j[k..size] * scalar(col_j[k])
            for k in j + 1..size {
                let global_k = offset + k;

                // Scalar multiplier: L_kj = mat(global_k, global_j)
                let l_kj = *ptr.add(global_j * rows + global_k);
                let l_kj_vec = _mm256_set1_ps(l_kj);

                // Vector to subtract: col_j[k..size]
                // Dest vector: col_k[k..size]
                // Start row: offset + k
                let start_row = offset + k;
                let count_update = offset + size - start_row; // Rows to update (lower triangle)

                let src_ptr = ptr.add(global_j * rows + start_row);
                let dst_ptr = ptr.add(global_k * rows + start_row);

                let mut y = 0;
                while y + 7 < count_update {
                    let v_src = _mm256_loadu_ps(src_ptr.add(y));
                    let v_dst = _mm256_loadu_ps(dst_ptr.add(y));
                    // dst = dst - src * scale
                    let v_res = _mm256_fnmadd_ps(v_src, l_kj_vec, v_dst);
                    _mm256_storeu_ps(dst_ptr.add(y), v_res);
                    y += 8;
                }
                while y < count_update {
                    *dst_ptr.add(y) -= *src_ptr.add(y) * l_kj;
                    y += 1;
                }
            }
        }
        Ok(())
    }

    /// Blocked LLT (Right-Looking).
    fn llt_blocked(mat: &mut Matrix<T, DynamicStorage<T>>, bs: usize) -> Result<(), String> {
        let n = mat.rows();
        for k in (0..n).step_by(bs) {
            let kb = std::cmp::min(n - k, bs);

            // 1. Factorize diagonal block A(k:k+kb, k:k+kb)
            Self::llt_unblocked(mat, k, kb)?;

            if k + kb < n {
                // 2. Panel Solve (Trsm): A(k+kb:n, k:k+kb) * L(k:k+kb, k:k+kb)^T = A(k+kb:n, k:k+kb)
                // We implement a naive vectorized trsm for now as it's O(N^2)
                Self::trsm_right_transpose(mat, k, kb, n);

                // 3. Rank-k Update (GEMM): A(k+kb:n, k+kb:n) -= A(k+kb:n, k:k+kb) * A(k+kb:n, k:k+kb)^T
                // This is the heavy lifting O(N^3) part.
                Self::syrk_update(mat, k, kb, n);
            }
        }

        // Zero out upper triangle explicitly at the end
        for j in 0..n {
            for i in 0..j {
                *mat.get_mut(i, j).unwrap() = T::from_usize(0);
            }
        }

        Ok(())
    }

    // Solve X * L^T = B where B is A(k+kb:n, k:k+kb) and L is A(k:k+kb, k:k+kb)
    fn trsm_right_transpose(mat: &mut Matrix<T, DynamicStorage<T>>, k: usize, kb: usize, n: usize) {
        // Optimization for f32 AVX2
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(target_feature = "avx2")]
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                unsafe {
                    Self::trsm_right_transpose_f32_avx(mat, k, kb, n);
                }
                return;
            }
        }

        // Fallback
        for i in k + kb..n {
            // For each row in the panel below diagonal
            for j in 0..kb {
                // Column in the panel (local index)
                let global_j = k + j;
                let mut val = *mat.get(i, global_j).unwrap();

                // Subtract knowns
                for l in 0..j {
                    let global_l = k + l;
                    val -=
                        (*mat.get(i, global_l).unwrap()) * (*mat.get(global_j, global_l).unwrap());
                }

                // Divide by diagonal L^T(j, j) = L(j, j)
                let diag = *mat.get(global_j, global_j).unwrap();
                *mat.get_mut(i, global_j).unwrap() = val / diag;
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn trsm_right_transpose_f32_avx(
        mat: &mut Matrix<T, DynamicStorage<T>>,
        k: usize,
        kb: usize,
        n: usize,
    ) {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        // Cast T pointer to f32
        let ptr = mat.storage_mut().data_mut().as_mut_ptr() as *mut f32;
        let rows = mat.rows(); // stride

        let b_start_row = k + kb;
        let b_rows = n - b_start_row;
        // kb is cols
        // We operate on the panel A(k+kb:n, k:k+kb)

        // Process B in blocks of 16 rows (2 x AVX YMM) to hide FMA latency
        // Unrolling outer loop `i` helps because `x_vec` dependency in inner loop `l` is broken across `i`.
        let row_block = 16;
        let mut i_blk = 0;

        while i_blk + 15 < b_rows {
            let row_offset = b_start_row + i_blk;

            // Two accumulators for 16 rows
            // x_vec0 for rows [0..8]
            // x_vec1 for rows [8..16]

            for j in 0..kb {
                let global_j = k + j;
                let b_col_ptr0 = ptr.add(global_j * rows + row_offset);
                let b_col_ptr1 = ptr.add(global_j * rows + row_offset + 8);

                let mut x_vec0 = _mm256_loadu_ps(b_col_ptr0);
                let mut x_vec1 = _mm256_loadu_ps(b_col_ptr1);

                for l in 0..j {
                    let global_l = k + l;

                    // L_jl = L(global_j, global_l)
                    // Since both L and B are ColMajor:
                    // L(row, col) is ptr[col * rows + row]
                    // We need L(global_j, global_l).
                    let ljl_val = *ptr.add(global_l * rows + global_j);
                    let ljl_vec = _mm256_set1_ps(ljl_val);

                    // X(:, l) is B(:, l)
                    // We need X(row_offset..row_offset+8, l)
                    let x_prev_ptr0 = ptr.add(global_l * rows + row_offset);
                    let x_prev_vec0 = _mm256_loadu_ps(x_prev_ptr0);

                    let x_prev_ptr1 = ptr.add(global_l * rows + row_offset + 8);
                    let x_prev_vec1 = _mm256_loadu_ps(x_prev_ptr1);

                    // x_j -= x_l * L_jl
                    x_vec0 = _mm256_fnmadd_ps(x_prev_vec0, ljl_vec, x_vec0);
                    x_vec1 = _mm256_fnmadd_ps(x_prev_vec1, ljl_vec, x_vec1);
                }

                // OPTIMIZATION: Multiply by inverse diagonal instead of division
                let ljj_val = *ptr.add(global_j * rows + global_j);
                let inv_ljj = 1.0 / ljj_val;
                let inv_ljj_vec = _mm256_set1_ps(inv_ljj);

                x_vec0 = _mm256_mul_ps(x_vec0, inv_ljj_vec);
                x_vec1 = _mm256_mul_ps(x_vec1, inv_ljj_vec);

                _mm256_storeu_ps(b_col_ptr0, x_vec0);
                _mm256_storeu_ps(b_col_ptr1, x_vec1);
            }
            i_blk += 16;
        }

        // Handle remaining blocks of 8 if any
        if i_blk + 7 < b_rows {
            let row_offset = b_start_row + i_blk;
            for j in 0..kb {
                let global_j = k + j;
                let b_col_ptr = ptr.add(global_j * rows + row_offset);
                let mut x_vec = _mm256_loadu_ps(b_col_ptr);

                for l in 0..j {
                    let global_l = k + l;
                    let ljl_val = *ptr.add(global_l * rows + global_j);
                    let ljl_vec = _mm256_set1_ps(ljl_val);

                    let x_prev_ptr = ptr.add(global_l * rows + row_offset);
                    let x_prev_vec = _mm256_loadu_ps(x_prev_ptr);

                    x_vec = _mm256_fnmadd_ps(x_prev_vec, ljl_vec, x_vec);
                }
                // OPTIMIZATION: Multiply by inverse diagonal
                let ljj_val = *ptr.add(global_j * rows + global_j);
                let inv_ljj = 1.0 / ljj_val;
                let inv_ljj_vec = _mm256_set1_ps(inv_ljj);

                x_vec = _mm256_mul_ps(x_vec, inv_ljj_vec);
                _mm256_storeu_ps(b_col_ptr, x_vec);
            }
            i_blk += 8;
        }

        // Handle remaining rows with scalar loop
        if i_blk < b_rows {
            for i in (b_start_row + i_blk)..n {
                for j in 0..kb {
                    let global_j = k + j;
                    let mut val = *mat.get(i, global_j).unwrap();

                    for l in 0..j {
                        let global_l = k + l;
                        val -= (*mat.get(i, global_l).unwrap())
                            * (*mat.get(global_j, global_l).unwrap());
                    }

                    let diag = *mat.get(global_j, global_j).unwrap();
                    *mat.get_mut(i, global_j).unwrap() = val / diag;
                }
            }
        }
    }

    // Update trailing matrix: C -= A * A^T
    // C = A(k+kb:n, k+kb:n)
    // A = A(k+kb:n, k:k+kb)
    fn syrk_update(mat: &mut Matrix<T, DynamicStorage<T>>, k: usize, kb: usize, n: usize) {
        use crate::core::ops::gemm::gemm_blocked;

        let m_size = n - (k + kb); // Rows of A21 and A22
        let k_size = kb; // Cols of A21

        // Pointers
        // A21 starts at (k+kb, k)
        let a21_ptr = mat.get(k + kb, k).unwrap() as *const T;
        let rs_a = 1;
        let cs_a = mat.rows() as isize;

        // C (A22) starts at (k+kb, k+kb)
        let c_ptr = mat.get_mut(k + kb, k + kb).unwrap() as *mut T;
        let rs_c = 1;
        let cs_c = mat.rows() as isize;

        // Check for F32/F64 optimization availability
        // Only run optimized path if FMA is available and Types match
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let tid = std::any::TypeId::of::<T>();

            // F32 Path
            #[cfg(target_feature = "avx2")]
            if tid == std::any::TypeId::of::<f32>() {
                use crate::core::ops::gemm::arch::x86::asm_kernel::AsmFmaKernelF32;
                // Alpha = -1.0
                // Transmutate -1.0f32 to T
                // Since we know T is f32 here, we can use transmute_copy or pointer cast logic
                // But simplified: Just cast.
                let alpha_f32: f32 = -1.0;

                unsafe {
                    let _ = gemm_blocked::<f32, AsmFmaKernelF32>(
                        m_size,
                        k_size,
                        m_size,                // n_size = m_size
                        a21_ptr as *const f32, // A
                        rs_a,
                        cs_a,
                        a21_ptr as *const f32, // B (A^T handling: swap rs/cs)
                        cs_a,                  // rs_b = cs_a (Transpose)
                        rs_a,                  // cs_b = rs_a (Transpose) -> effectively reading A^T
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
                        m_size,
                        a21_ptr as *const f64,
                        rs_a,
                        cs_a,
                        a21_ptr as *const f64,
                        cs_a,
                        rs_a,
                        c_ptr as *mut f64,
                        rs_c,
                        cs_c,
                        alpha_f64,
                    );
                }
                return;
            }
        }

        // Fallback (Naive Loop)
        for i in k + kb..n {
            for j in k + kb..=i {
                // Lower triangle only
                let mut sum = T::from_usize(0);
                for l in 0..kb {
                    let global_l = k + l;
                    sum += (*mat.get(i, global_l).unwrap()) * (*mat.get(j, global_l).unwrap());
                }
                *mat.get_mut(i, j).unwrap() -= sum;
            }
        }
    }

    /// Returns the factor L.
    pub fn matrix_l(&self) -> &Matrix<T, DynamicStorage<T>> {
        &self.l
    }

    /// Solves Ax = b for x using the LLT decomposition.
    pub fn solve<S2: Storage<T> + 'static>(
        &self,
        b: &Matrix<T, S2>,
    ) -> Result<Matrix<T, DynamicStorage<T>>, String> {
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

    /// Solves L * X = B in-place, where B is stored in `mat`.
    /// `mat` is overwritten with X.
    /// L is lower triangular (from this decomposition).
    pub fn solve_inplace_l(&self, mat: &mut Matrix<T, DynamicStorage<T>>) -> Result<(), String> {
        let rows = self.l.rows();
        if mat.rows() != rows {
            return Err("Dimension mismatch in LLT solve_inplace_l".to_string());
        }
        let cols = mat.cols();
        
        // Forward substitution
        for i in 0..rows {
            let l_ii = *self.l.get(i, i).unwrap();
            let inv_l_ii = T::one() / l_ii;
            
            for j in 0..cols {
                let mut val = *mat.get(i, j).unwrap();
                for k in 0..i {
                    val -= (*self.l.get(i, k).unwrap()) * (*mat.get(k, j).unwrap());
                }
                *mat.get_mut(i, j).unwrap() = val * inv_l_ii;
            }
        }
        Ok(())
    }

    /// Solves L^T * X = B in-place, where B is stored in `mat`.
    /// `mat` is overwritten with X.
    /// L^T is upper triangular.
    pub fn solve_inplace_lt(&self, mat: &mut Matrix<T, DynamicStorage<T>>) -> Result<(), String> {
        let rows = self.l.rows();
        if mat.rows() != rows {
            return Err("Dimension mismatch in LLT solve_inplace_lt".to_string());
        }
        let cols = mat.cols();
        
        // Backward substitution
        for i in (0..rows).rev() {
            let l_ii = *self.l.get(i, i).unwrap();
            let inv_l_ii = T::one() / l_ii;
            
            for j in 0..cols {
                let mut val = *mat.get(i, j).unwrap();
                for k in i + 1..rows {
                     val -= (*self.l.get(k, i).unwrap()) * (*mat.get(k, j).unwrap());
                }
                *mat.get_mut(i, j).unwrap() = val * inv_l_ii;
            }
        }
        Ok(())
    }
}
