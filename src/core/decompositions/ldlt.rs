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

        for j in 0..rows {
            // 1. Diagonal Pivoting: Find max diagonal in the remaining submatrix
            let mut max_idx = j;
            let mut max_val = mat.get(j, j).unwrap().abs();
            for k in j + 1..rows {
                let v = mat.get(k, k).unwrap().abs();
                if v > max_val {
                    max_val = v;
                    max_idx = k;
                }
            }

            if max_idx != j {
                // Swap rows and columns max_idx and j
                p.swap(j, max_idx);

                // Swap rows
                for k in 0..rows {
                    let tmp = *mat.get(j, k).unwrap();
                    *mat.get_mut(j, k).unwrap() = *mat.get(max_idx, k).unwrap();
                    *mat.get_mut(max_idx, k).unwrap() = tmp;
                }
                // Swap columns
                for k in 0..rows {
                    let tmp = *mat.get(k, j).unwrap();
                    *mat.get_mut(k, j).unwrap() = *mat.get(k, max_idx).unwrap();
                    *mat.get_mut(k, max_idx).unwrap() = tmp;
                }
            }

            // 2. Standard LDLT step on the pivoted matrix (Vectorized Left-Looking)
            let mut ptr_mut = mat.storage_mut().data_mut().as_mut_ptr(); // Safe? We need raw pointer.
            // Actually proper way:
            
            unsafe {
                 // Check if we can use AVX2
                 // Assuming f64 mostly, but handle f32
                 if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() && is_x86_feature_detected!("fma") {
                    let ptr = mat.storage_mut().data_mut().as_mut_ptr() as *mut f64;
                    for k in 0..j {
                        let l_jk_val = *mat.get(j, k).unwrap();
                        let d_k_val = d[k];
                        let val_kj = l_jk_val * d_k_val; // Value to propagate
                        
                        let val_f64 = *(std::mem::transmute::<&T, &f64>(&val_kj));
                        
                        // Apply update to column j using column k
                        // L(j:N, j) -= val * L(j:N, k)
                        Self::update_column_vectorized_f64(ptr, rows, j, k, val_f64);
                    }
                 } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() && is_x86_feature_detected!("fma") {
                    let ptr = mat.storage_mut().data_mut().as_mut_ptr() as *mut f32;
                    for k in 0..j {
                        let l_jk_val = *mat.get(j, k).unwrap();
                        let d_k_val = d[k];
                        let val_kj = l_jk_val * d_k_val;
                        let val_f32 = *(std::mem::transmute::<&T, &f32>(&val_kj));
                        Self::update_column_vectorized_f32(ptr, rows, j, k, val_f32);
                    }
                 } else {
                     // Scalar Fallback (Column-Oriented)
                     for k in 0..j {
                         let val = *mat.get(j, k).unwrap() * d[k];
                         for i in j..rows {
                             let lik = *mat.get(i, k).unwrap();
                             *mat.get_mut(i, j).unwrap() -= val * lik;
                         }
                     }
                 }
            }

            let dj = *mat.get(j, j).unwrap();
            d[j] = dj;
            *mat.get_mut(j, j).unwrap() = T::from_usize(1);

            if dj.abs() > T::epsilon() {
                let inv_dj = T::from_usize(1) / dj;
                for i in j + 1..rows {
                    *mat.get_mut(i, j).unwrap() *= inv_dj;
                }
            } else {
                 for i in j + 1..rows {
                    *mat.get_mut(i, j).unwrap() = T::from_usize(0);
                }
            }


        }

        // Clean up upper triangle
        for i in 0..rows {
            for j in i + 1..rows {
                *mat.get_mut(i, j).unwrap() = T::from_usize(0);
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
                    val -= (*self.l.get(k, i).unwrap()) * (*x.get(k, j).unwrap());
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
        let col_j_ptr = mat_ptr.add(j * rows);
        let col_k_ptr = mat_ptr.add(k * rows);
        
        // We update rows j..rows (actually we only need j..rows. But algorithm updates j for D[j])
        // Let's stick to update j..rows.
        let len = rows - j;
        let mut r = 0;
        
        let val_vec = _mm256_set1_ps(val);
        
        while r + 8 <= len {
            let row_idx = j + r;
            // Load column j (destination)
            let mut y_vec = _mm256_loadu_ps(col_j_ptr.add(row_idx));
            // Load column k (source)
            let x_vec = _mm256_loadu_ps(col_k_ptr.add(row_idx));
            
            // y = y - val * x
            y_vec = _mm256_fnmadd_ps(val_vec, x_vec, y_vec);
            
            _mm256_storeu_ps(col_j_ptr.add(row_idx), y_vec);
            r += 8;
        }
        
        for rr in r..len {
            let row_idx = j + rr;
            *col_j_ptr.add(row_idx) -= val * *col_k_ptr.add(row_idx);
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
            let row_idx = j + r;
            let mut y_vec = _mm256_loadu_pd(col_j_ptr.add(row_idx));
            let x_vec = _mm256_loadu_pd(col_k_ptr.add(row_idx));
            
            y_vec = _mm256_fnmadd_pd(val_vec, x_vec, y_vec);
            
            _mm256_storeu_pd(col_j_ptr.add(row_idx), y_vec);
            r += 4;
        }
        
        for rr in r..len {
            let row_idx = j + rr;
            *col_j_ptr.add(row_idx) -= val * *col_k_ptr.add(row_idx);
        }
    }
}
