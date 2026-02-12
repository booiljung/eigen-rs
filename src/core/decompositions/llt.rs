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
        const BLOCK_SIZE: usize = 32;

        if rows <= BLOCK_SIZE {
            Self::llt_unblocked(&mut l, 0, rows)?;
        } else {
            Self::llt_blocked(&mut l, BLOCK_SIZE)?;
        }

        Ok(Self {
            l,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Unblocked LLT (Naive) for small blocks/panels.
    /// Operates on the submatrix starting at (offset, offset) with size `size`.
    fn llt_unblocked(mat: &mut Matrix<T, DynamicStorage<T>>, offset: usize, size: usize) -> Result<(), String> {
        for j in 0..size {
            let global_j = offset + j;
            let mut s = T::from_usize(0);
            for k in 0..j {
                let global_k = offset + k;
                let l_jk = *mat.get(global_j, global_k).unwrap();
                s += l_jk * l_jk;
            }

            let diag = *mat.get(global_j, global_j).unwrap() - s;
            if diag <= T::from_usize(0) {
                return Err("Matrix is not positive definite".to_string());
            }

            let l_jj = diag.sqrt();
            *mat.get_mut(global_j, global_j).unwrap() = l_jj;
            let l_jj_inv = T::one() / l_jj;

            for i in j + 1..size {
                let global_i = offset + i;
                let mut s = T::from_usize(0);
                for k in 0..j {
                    let global_k = offset + k;
                    s += (*mat.get(global_i, global_k).unwrap()) * (*mat.get(global_j, global_k).unwrap());
                }
                *mat.get_mut(global_i, global_j).unwrap() = (*mat.get(global_i, global_j).unwrap() - s) * l_jj_inv;
                // Zero out the upper triangle happens implicitly or explicitly outside
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
        // L is lower triangular at (k, k). L^T is upper triangular.
        // X_row * L^T = B_row
        // Back-substitution for each row of the panel.
        
        for i in k + kb..n { // For each row in the panel below diagonal
            for j in 0..kb { // Column in the panel (local index)
                let global_j = k + j;
                let mut val = *mat.get(i, global_j).unwrap();
                
                // Subtract knowns
                for l in 0..j {
                    let global_l = k + l;
                    // L^T(l, j) is L(j, l)
                    val -= (*mat.get(i, global_l).unwrap()) * (*mat.get(global_j, global_l).unwrap());
                }
                
                // Divide by diagonal L^T(j, j) = L(j, j)
                let diag = *mat.get(global_j, global_j).unwrap();
                *mat.get_mut(i, global_j).unwrap() = val / diag;
            }
        }
    }

    // Update trailing matrix: C -= A * A^T
    // C = A(k+kb:n, k+kb:n)
    // A = A(k+kb:n, k:k+kb)
    fn syrk_update(mat: &mut Matrix<T, DynamicStorage<T>>, k: usize, kb: usize, n: usize) {
        use crate::core::ops::gemm::gemm_blocked;
        
        let m_size = n - (k + kb); // Rows of A21 and A22
        let k_size = kb;           // Cols of A21
        
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
            if tid == std::any::TypeId::of::<f32>() && is_x86_feature_detected!("fma") {
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
                        m_size, // n_size = m_size
                        a21_ptr as *const f32, // A
                        rs_a,
                        cs_a,
                        a21_ptr as *const f32, // B (A^T handling: swap rs/cs)
                        cs_a, // rs_b = cs_a (Transpose)
                        rs_a, // cs_b = rs_a (Transpose) -> effectively reading A^T
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
            for j in k + kb..=i { // Lower triangle only
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
}
