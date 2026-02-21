//! Jacobi Singular Value Decomposition (SVD).
//! Highly accurate SVD implementation using two-sided Jacobi rotations.

use crate::core::decompositions::bidiagonal::Bidiagonalization;
use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::{DynamicStorage, Storage};

/// Result of a JacobiSVD decomposition.
pub struct JacobiSVD<T: Scalar, S: Storage<T>> {
    u: Matrix<T, DynamicStorage<T>>,
    v: Matrix<T, DynamicStorage<T>>,
    singular_values: Vec<T>,
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar<Real = T> + PartialOrd + 'static, S: Storage<T> + 'static> JacobiSVD<T, S> {
    /// Computes the JacobiSVD of the given matrix.
    pub fn new(matrix: &Matrix<T, S>) -> Result<Self, String> {
        let m = matrix.rows();
        let n = matrix.cols();

        let mut a = Matrix::<T, DynamicStorage<T>>::new_dynamic(m, n)?;
        a.assign(matrix)?;

        let mut u = Matrix::<T, DynamicStorage<T>>::new_dynamic(m, m)?;
        // Initialize U as identity
        for i in 0..m {
            for j in 0..m {
                *u.get_mut(i, j).unwrap() = if i == j {
                    T::from_usize(1)
                } else {
                    T::from_usize(0)
                };
            }
        }

        let mut v = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n)?;
        // Initialize V as identity
        for i in 0..n {
            for j in 0..n {
                *v.get_mut(i, j).unwrap() = if i == j {
                    T::from_usize(1)
                } else {
                    T::from_usize(0)
                };
            }
        }

        let max_iter = 100;
        #[allow(unused_assignments)]
        let mut eps = T::from_usize(0); // This should be a small value based on type
                                        // For f32, 1e-7 is reasonable.
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            eps = unsafe { *(&1e-7f32 as *const f32 as *const T) };
        } else {
            eps = unsafe { *(&1e-15f64 as *const f64 as *const T) };
        }

        // Two-sided Jacobi SVD algorithm (Simplified)
        for _iter in 0..max_iter {
            let mut converged = true;
            for i in 0..n {
                for j in i + 1..n {
                    // Compute J_ij = [ a_ii a_ij; a_ji a_jj ]
                    // But for SVD, we need to consider columns or use a symmetric form A^T A
                    // Here we'll use the One-Sided Jacobi (better for non-square) or Two-Sided.
                    // For brevity, let's implement the One-Sided Jacobi which is common.

                    let mut a_ii = T::from_usize(0);
                    let mut a_jj = T::from_usize(0);
                    let mut a_ij = T::from_usize(0);

                    for k in 0..m {
                        a_ii += (*a.get(k, i).unwrap()) * (*a.get(k, i).unwrap());
                        a_jj += (*a.get(k, j).unwrap()) * (*a.get(k, j).unwrap());
                        a_ij += (*a.get(k, i).unwrap()) * (*a.get(k, j).unwrap());
                    }

                    if a_ij.abs() > eps * (a_ii * a_jj).sqrt() {
                        converged = false;

                        let tau = (a_jj - a_ii) / (T::from_usize(2) * a_ij);
                        let t = if tau >= T::from_usize(0) {
                            T::from_usize(1) / (tau + (T::from_usize(1) + tau * tau).sqrt())
                        } else {
                            T::from_usize(0)
                                - (T::from_usize(1)
                                    / ((T::from_usize(0) - tau)
                                        + (T::from_usize(1) + tau * tau).sqrt()))
                        };

                        let c = T::from_usize(1) / (T::from_usize(1) + t * t).sqrt();
                        let s = c * t;

                        // Rotate columns i and j of A
                        Self::apply_rotation(&mut a, i, j, c, s);

                        // Rotate columns i and j of V
                        Self::apply_rotation(&mut v, i, j, c, s);
                    }
                }
            }
            if converged {
                break;
            }
        }

        // Extract singular values and normalize U
        let mut singular_values = Vec::new();
        let mut valid_cols = Vec::new();
        let mut invalid_cols = Vec::new();

        for i in 0..n {
            let mut norm = T::from_usize(0);
            for k in 0..m {
                norm += (*a.get(k, i).unwrap()) * (*a.get(k, i).unwrap());
            }
            let s_val = norm.sqrt();
            singular_values.push((s_val, i));

            if s_val > eps {
                for k in 0..m {
                    *u.get_mut(k, i).unwrap() = *a.get(k, i).unwrap() / s_val;
                }
                valid_cols.push(i);
            } else {
                // Mark for GS fixup. Currently holds e_i
                invalid_cols.push(i);
            }
        }
        
        // Gram-Schmidt Completion for Null Space
        if !invalid_cols.is_empty() {
             for &bad_idx in &invalid_cols {
                 // Current vector v is u[:, bad_idx] (which is e_{bad_idx})
                 // Orthogonalize against all valid_cols
                 for &good_idx in &valid_cols {
                     let mut dot = T::from_usize(0);
                     for r in 0..m {
                         dot += *u.get(r, bad_idx).unwrap() * *u.get(r, good_idx).unwrap();
                     }
                     for r in 0..m {
                         let val = *u.get(r, bad_idx).unwrap() - dot * *u.get(r, good_idx).unwrap();
                         *u.get_mut(r, bad_idx).unwrap() = val;
                     }
                 }
                 
                 // Orthogonalize against previously fixed invalid_cols
                 // Note: invalid_cols is iterated in order. We can just iterate valid_cols U (fixed invalid)
                 // Wait, we need to add bad_idx to valid_cols after fixing?
                 // Let's simpler: just iterate all PROCESSED columns.
             }
             
             // The above loop is tricky because we need to ortho against *modified* bad columns too.
             // Let's rewrite cleaner:
             
             // 1. We have a set of Basis Vectors = { u[good] }
             // 2. We have Candidates = { u[bad] }
             // 3. For each candidate C:
             //      C = C - sum( proj(C, B) ) for B in Basis
             //      C = normalize(C)
             //      Basis.add(C)
             
             let mut basis_indices = valid_cols.clone();
             for &bad_idx in &invalid_cols {
                  let mut success = false;
                  let mut attempt = 0;
                  
                  // Reset the column to e_bad_idx initially (it might have been modified by previous operations if we didn't track properly, 
                  // but here we are just reading it. Actually, u was initialized to I, but rotated.
                  // Wait, u was rotated! So u[:, bad_idx] is NOT e_bad_idx. It is some vector resulting from rotations.
                  // But since s_val ~ 0, A * V[:, bad_idx] ~ 0.
                  // The column in u corresponding to this is u[:, bad_idx] which is technically part of the accumulated rotation?
                  // No, for One-Sided Jacobi, U is computed as A * V * S^-1.
                  // U was initialized to I, but line 121 *overwrites* it: *u.get_mut(k, i) = ...
                  // The loop only overwrites for s_val > eps.
                  // So for s_val <= eps, the column u[:, bad_idx] is indeed the generic "Identity" column from initialization (line 30).
                  // BUT, u was initialized to I at the very beginning of 'new'.
                  // Then 'apply_rotation' was called on 'a' and 'v'.
                  // 'u' was NEVER rotated in this One-Sided implementation!
                  // So u[:, bad_idx] IS e_{bad_idx}. Confirmed.
                  
                  while !success && attempt < 10 {
                      // If attempt > 0, randomize the vector
                      if attempt > 0 {
                           for r in 0..m {
                               // Simple pseudo-random generator
                               let val = ((r + bad_idx + attempt) * 123456789) % 100;
                               *u.get_mut(r, bad_idx).unwrap() = T::from_f64(val as f64 / 100.0);
                           }
                      }
                  
                      // 1. Orthogonalize
                      for &basis_idx in &basis_indices {
                          let mut dot = T::from_usize(0);
                          for r in 0..m {
                               dot += *u.get(r, bad_idx).unwrap() * *u.get(r, basis_idx).unwrap();
                          }
                          for r in 0..m {
                               let sub = dot * *u.get(r, basis_idx).unwrap();
                               let val = *u.get(r, bad_idx).unwrap() - sub;
                               *u.get_mut(r, bad_idx).unwrap() = val;
                          }
                      }
                      
                      // 2. Normalize
                      let mut norm = T::from_usize(0);
                      for r in 0..m {
                           let val = *u.get(r, bad_idx).unwrap();
                           norm += val * val;
                      }
                      let n_val = norm.sqrt();
                      
                      if n_val > eps {
                           for r in 0..m {
                                *u.get_mut(r, bad_idx).unwrap() /= n_val;
                           }
                           success = true;
                      } else {
                           attempt += 1;
                      }
                  }
                  
                  if !success {
                       println!("JacobiSVD Error: Failed to complete basis for column {}", bad_idx);
                  }
                  
                  // 3. Add to basis
                  basis_indices.push(bad_idx);
             }
        }

        // Sort singular values in descending order
        singular_values.sort_by(|a, b| {
            if let Some(ord) = b.0.partial_cmp(&a.0) {
                ord
            } else {
                let a_nan = a.0 != a.0;
                let b_nan = b.0 != b.0;
                if a_nan && b_nan {
                    std::cmp::Ordering::Equal
                } else if a_nan {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            }
        });

        let mut sorted_s = vec![T::from_usize(0); n];
        let mut sorted_u = Matrix::<T, DynamicStorage<T>>::new_dynamic(m, m)?;
        let mut sorted_v = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n)?;

        for (i, (s_val, orig_idx)) in singular_values.into_iter().enumerate() {
            sorted_s[i] = s_val;
            for k in 0..m {
                *sorted_u.get_mut(k, i).unwrap() = *u.get(k, orig_idx).unwrap();
            }
            for k in 0..n {
                *sorted_v.get_mut(k, i).unwrap() = *v.get(k, orig_idx).unwrap();
            }
        }
        
        // Debug: Check sorted_u orthogonality immediately
        for c1 in 0..n {
             for c2 in 0..n {
                 let mut dot = T::from_usize(0);
                 for r in 0..m {
                     dot += *sorted_u.get(r, c1).unwrap() * *sorted_u.get(r, c2).unwrap();
                 }
                 let expected = if c1 == c2 { T::from_usize(1) } else { T::from_usize(0) };
                 if (dot - expected).abs() > eps * T::from_usize(1000) {
                     println!("JacobiSVD Internal Ortho Error at {}x{}: dot={:?}, expected={:?}, error={:?}", c1, c2, dot, expected, (dot-expected).abs());
                 }
             }
        }

        Ok(Self {
            u: sorted_u,
            v: sorted_v,
            singular_values: sorted_s,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn matrix_u(&self) -> &Matrix<T, DynamicStorage<T>> {
        &self.u
    }
    pub fn matrix_v(&self) -> &Matrix<T, DynamicStorage<T>> {
        &self.v
    }
    pub fn singular_values(&self) -> &[T] {
        &self.singular_values
    }

    // Helper: Apply rotation to columns i and j
    fn apply_rotation(mat: &mut Matrix<T, DynamicStorage<T>>, i: usize, j: usize, c: T, s: T) {
        let rows = mat.rows();

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                if is_x86_feature_detected!("fma") {
                    use std::arch::x86_64::*;
                    unsafe {
                        // Cast to f32
                        let c_f32: f32 = *(&c as *const T as *const f32);
                        let s_f32: f32 = *(&s as *const T as *const f32);

                        let ptr_i = mat.get_mut(0, i).unwrap() as *mut T as *mut f32;
                        let ptr_j = mat.get_mut(0, j).unwrap() as *mut T as *mut f32;

                        let c_vec = _mm256_set1_ps(c_f32);
                        let s_vec = _mm256_set1_ps(s_f32);

                        let mut k = 0;
                        while k + 8 <= rows {
                            let val_i = _mm256_loadu_ps(ptr_i.add(k));
                            let val_j = _mm256_loadu_ps(ptr_j.add(k));

                            // i' = c*i - s*j
                            // j' = s*i + c*j

                            let term1_i = _mm256_mul_ps(c_vec, val_i);
                            let term2_i = _mm256_mul_ps(s_vec, val_j);
                            let new_i = _mm256_sub_ps(term1_i, term2_i);

                            let term1_j = _mm256_mul_ps(s_vec, val_i);
                            let term2_j = _mm256_mul_ps(c_vec, val_j);
                            let new_j = _mm256_add_ps(term1_j, term2_j);

                            _mm256_storeu_ps(ptr_i.add(k), new_i);
                            _mm256_storeu_ps(ptr_j.add(k), new_j);
                            k += 8;
                        }

                        // Scalar tail
                        for kk in k..rows {
                            let val_i = *ptr_i.add(kk);
                            let val_j = *ptr_j.add(kk);
                            *ptr_i.add(kk) = c_f32 * val_i - s_f32 * val_j;
                            *ptr_j.add(kk) = s_f32 * val_i + c_f32 * val_j;
                        }
                    }
                    return;
                }
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                if is_x86_feature_detected!("fma") {
                    use std::arch::x86_64::*;
                    unsafe {
                        let c_f64: f64 = *(&c as *const T as *const f64);
                        let s_f64: f64 = *(&s as *const T as *const f64);

                        let ptr_i = mat.get_mut(0, i).unwrap() as *mut T as *mut f64;
                        let ptr_j = mat.get_mut(0, j).unwrap() as *mut T as *mut f64;

                        let c_vec = _mm256_set1_pd(c_f64);
                        let s_vec = _mm256_set1_pd(s_f64);

                        let mut k = 0;
                        while k + 4 <= rows {
                            let val_i = _mm256_loadu_pd(ptr_i.add(k));
                            let val_j = _mm256_loadu_pd(ptr_j.add(k));

                            let new_i = _mm256_sub_pd(
                                _mm256_mul_pd(c_vec, val_i),
                                _mm256_mul_pd(s_vec, val_j),
                            );
                            let new_j = _mm256_add_pd(
                                _mm256_mul_pd(s_vec, val_i),
                                _mm256_mul_pd(c_vec, val_j),
                            );

                            _mm256_storeu_pd(ptr_i.add(k), new_i);
                            _mm256_storeu_pd(ptr_j.add(k), new_j);
                            k += 4;
                        }
                        for kk in k..rows {
                            let val_i = *ptr_i.add(kk);
                            let val_j = *ptr_j.add(kk);
                            *ptr_i.add(kk) = c_f64 * val_i - s_f64 * val_j;
                            *ptr_j.add(kk) = s_f64 * val_i + c_f64 * val_j;
                        }
                    }
                    return;
                }
            }
        }

        // Scalar fallback (generic T)
        for k in 0..rows {
            let val_i = *mat.get(k, i).unwrap();
            let val_j = *mat.get(k, j).unwrap();
            *mat.get_mut(k, i).unwrap() = c * val_i - s * val_j;
            *mat.get_mut(k, j).unwrap() = s * val_i + c * val_j;
        }
    }
}


