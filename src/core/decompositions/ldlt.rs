//! LDLT Cholesky decomposition ($A = LDL^T$).
//! Robust decomposition that avoids square roots and handles semi-definite matrices.

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::DynamicStorage;
use crate::core::storage::Storage;

/// Result of an LDLT Cholesky decomposition.
pub struct LDLT<T: Scalar, S: Storage<T>> {
    l: Matrix<T, DynamicStorage<T>>,
    d: Vec<T>,
    p: Vec<usize>, // Permutation indices
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar + 'static, S: Storage<T> + 'static> LDLT<T, S> {
    /// Computes the LDLT decomposition of the given matrix with diagonal pivoting.
    pub fn new(matrix: &Matrix<T, S>) -> Result<Self, String> {
        let rows = matrix.rows();
        let cols = matrix.cols();
        if rows != cols {
            return Err("LDLT decomposition requires a square matrix".to_string());
        }

        let mut mat = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, cols)?;
        mat.assign(matrix)?;

        let mut d = vec![T::from_usize(0); rows];
        let mut p = (0..rows).collect::<Vec<_>>();

        // Epsilon for singularity check
        let eps = T::epsilon();

        // Safe pointer for hot loops
        let mat_ptr = mat.storage_mut().data_mut().as_mut_ptr();

        for j in 0..rows {
            // 1. Diagonal Pivoting: Find max diagonal in the remaining submatrix
            // Optimization: Only scan the diagonal, avoid excessive memory moves if no swap needed.
            let mut max_idx = j;
            let mut max_val = unsafe { *mat_ptr.add(j * rows + j) }.abs(); // Returns T::Real

            for k in j + 1..rows {
                let v = unsafe { *mat_ptr.add(k * rows + k) }.abs(); // mat(k,k)
                if v > max_val {
                    max_val = v;
                    max_idx = k;
                }
            }

            if max_idx != j {
                // Swap pivot indices
                p.swap(j, max_idx);

                // Symmetric Swap: Swap row/col j and max_idx
                // This is O(N) but indispensable for pivot stability.
                // We utilize the fact that we only need to swap up to 'rows' and the structure is symmetric 
                // in the lower triangle, but we store full matrix.

                // Swap rows (j, max_idx)
                for k in 0..rows {
                    unsafe {
                        let ptr_j = mat_ptr.add(k * rows + j);
                        let ptr_max = mat_ptr.add(k * rows + max_idx);
                        std::ptr::swap(ptr_j, ptr_max);
                    }
                }
                // Swap cols (j, max_idx)
                for k in 0..rows {
                    unsafe {
                        let ptr_j = mat_ptr.add(j * rows + k);
                        let ptr_max = mat_ptr.add(max_idx * rows + k);
                        std::ptr::swap(ptr_j, ptr_max);
                    }
                }
            }

            // 2. Blocked / Vectorized Update
            // L(j:N, j) -= sum_{k=0..j-1} L(j:N, k) * (L(j, k) * D(k))
            // Current Approach: Rank-1 update (right looking) or Dot product (left looking).
            // Eigen uses "Left Looking": For current column j, gather contributions from k < j.

            unsafe {
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() && is_x86_feature_detected!("fma") {
                    let ptr = mat_ptr as *mut f64;
                    // Gather contributions from previous columns
                    // Col j = Col j - L(:, 0..j) * (D(0..j) * L(j, 0..j))^T
                    // This inner loop is the bottleneck.
                    // Improving locality: Process in blocks of K columns?
                    // For now, let's just ensure inner loop is tight and vectorized.
                    
                    for k in 0..j {
                        let l_jk_val = *ptr.add(k * rows + j); // mat(j, k)
                        let d_k_val = *(d.as_ptr() as *const f64).add(k);
                        let val_kj = l_jk_val * d_k_val;

                        // Vectorized Update: Col(j) -= val * Col(k)
                        Self::update_column_vectorized_f64(ptr, rows, j, k, val_kj);
                    }
                } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() && is_x86_feature_detected!("fma") {
                     let ptr = mat_ptr as *mut f32;
                     for k in 0..j {
                        let l_jk_val = *ptr.add(k * rows + j);
                        let d_k_val = *(d.as_ptr() as *const f32).add(k);
                        let val_kj = l_jk_val * d_k_val;
                        Self::update_column_vectorized_f32(ptr, rows, j, k, val_kj);
                     }
                } else {
                     // Scalar Fallback
                     for k in 0..j {
                         let val = unsafe { *mat_ptr.add(k * rows + j) }.conj() * d[k];
                         // Update column j starting from row j
                         for i in j..rows {
                             unsafe {
                                *mat_ptr.add(j * rows + i) -= val * *mat_ptr.add(k * rows + i);
                             }
                         }
                     }
                }
            }

            // 3. Diagonal update and scaling
            let dj = unsafe { *mat_ptr.add(j * rows + j) };
            d[j] = dj;
            unsafe { *mat_ptr.add(j * rows + j) = T::from_usize(1) }; // L diagonal is 1

            if dj.abs() > eps {
                let inv_dj = T::from_usize(1) / dj;
                // Scale the column below diagonal
                unsafe {
                    for i in j + 1..rows {
                        *mat_ptr.add(j * rows + i) *= inv_dj;
                    }
                }
            } else {
                 unsafe {
                    for i in j + 1..rows {
                        *mat_ptr.add(j * rows + i) = T::from_usize(0);
                    }
                 }
            }
        }

        // Clean up upper triangle
        for i in 0..rows {
            for j in i + 1..rows {
                unsafe { *mat_ptr.add(j * rows + i) = T::from_usize(0) };
            }
        }

        Ok(Self {
            l: mat,
            d,
            p,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn matrix_l(&self) -> &Matrix<T, DynamicStorage<T>> {
        &self.l
    }
    pub fn vector_d(&self) -> &[T] {
        &self.d
    }
    pub fn permutation(&self) -> &[usize] {
        &self.p
    }

    /// Solves Ax = b for x using the LDLT decomposition.
    pub fn solve<S2: Storage<T> + 'static>(
        &self,
        b: &Matrix<T, S2>,
    ) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        let rows = self.l.rows();
        if b.rows() != rows {
            return Err("Dimension mismatch in LDLT solve".to_string());
        }

        let b_cols = b.cols();
        let mut x = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, b_cols)?;

        // Apply permutation P to b: x = P * b
        for i in 0..rows {
            let src_idx = self.p[i];
            for j in 0..b_cols {
                *x.get_mut(i, j).unwrap() = *b.get(src_idx, j).unwrap();
            }
        }

        // Solve L y = x (Forward substitution)
        for i in 0..rows {
            for j in 0..b_cols {
                let mut val = *x.get(i, j).unwrap();
                for k in 0..i {
                    val -= (*self.l.get(i, k).unwrap()) * (*x.get(k, j).unwrap());
                }
                *x.get_mut(i, j).unwrap() = val;
            }
        }

        // Solve D z = y
        for i in 0..rows {
            let di = self.d[i];
            if di.abs() > T::epsilon() {
                for j in 0..b_cols {
                    *x.get_mut(i, j).unwrap() /= di;
                }
            }
        }

        // Solve L^T w = z (Backward substitution)
        for i in (0..rows).rev() {
            for j in 0..b_cols {
                let mut val = *x.get(i, j).unwrap();
                for k in i + 1..rows {
                    val -= (*self.l.get(k, i).unwrap()).conj() * (*x.get(k, j).unwrap());
                }
                *x.get_mut(i, j).unwrap() = val;
            }
        }

        // Apply inverse permutation P^T to w: out = P^T * w
        let mut result = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, b_cols)?;
        for i in 0..rows {
            let dest_idx = self.p[i];
            for j in 0..b_cols {
               *result.get_mut(dest_idx, j).unwrap() = *x.get(i, j).unwrap();
            }
        }

        Ok(result)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn update_column_vectorized_f32(
        mat_ptr: *mut f32,
        rows: usize,
        j: usize,
        k: usize,
        val: f32,
    ) {
        use std::arch::x86_64::*;
        // Col pointers. Matrix is Column-Major. 
        // Col j (Target) starts at j * rows.
        // Col k (Source) starts at k * rows.
        let col_j_ptr = mat_ptr.add(j * rows);
        let col_k_ptr = mat_ptr.add(k * rows);
        
        // Update range: rows [j..rows]
        let len = rows - j;
        let mut r = 0;
        
        let val_vec = _mm256_set1_ps(val);
        
        while r + 8 <= len {
            let idx = j + r; // Row index
            let mut y_vec = _mm256_loadu_ps(col_j_ptr.add(idx));
            let x_vec = _mm256_loadu_ps(col_k_ptr.add(idx));
            y_vec = _mm256_fnmadd_ps(val_vec, x_vec, y_vec);
            _mm256_storeu_ps(col_j_ptr.add(idx), y_vec);
            r += 8;
        }
        
        for rr in r..len {
            let idx = j + rr;
            *col_j_ptr.add(idx) -= val * *col_k_ptr.add(idx);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn update_column_vectorized_f64(
        mat_ptr: *mut f64,
        rows: usize,
        j: usize,
        k: usize,
        val: f64,
    ) {
        use std::arch::x86_64::*;
        let col_j_ptr = mat_ptr.add(j * rows);
        let col_k_ptr = mat_ptr.add(k * rows);
        
        let len = rows - j;
        let mut r = 0;
        
        let val_vec = _mm256_set1_pd(val);
        
        while r + 4 <= len {
            let idx = j + r;
            let mut y_vec = _mm256_loadu_pd(col_j_ptr.add(idx));
            let x_vec = _mm256_loadu_pd(col_k_ptr.add(idx));
            y_vec = _mm256_fnmadd_pd(val_vec, x_vec, y_vec);
            _mm256_storeu_pd(col_j_ptr.add(idx), y_vec);
            r += 4;
        }
        
        for rr in r..len {
            let idx = j + rr;
            *col_j_ptr.add(idx) -= val * *col_k_ptr.add(idx);
        }
    }
}
