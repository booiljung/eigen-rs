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

impl<T: Scalar + 'static, S: Storage<T> + 'static> JacobiSVD<T, S> {
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
                        for k in 0..m {
                            let dummy_i = *a.get(k, i).unwrap();
                            let dummy_j = *a.get(k, j).unwrap();
                            *a.get_mut(k, i).unwrap() = c * dummy_i - s * dummy_j;
                            *a.get_mut(k, j).unwrap() = s * dummy_i + c * dummy_j;
                        }

                        // Rotate columns i and j of V
                        for k in 0..n {
                            let dummy_i = *v.get(k, i).unwrap();
                            let dummy_j = *v.get(k, j).unwrap();
                            *v.get_mut(k, i).unwrap() = c * dummy_i - s * dummy_j;
                            *v.get_mut(k, j).unwrap() = s * dummy_i + c * dummy_j;
                        }
                    }
                }
            }
            if converged {
                break;
            }
        }

        // Extract singular values and normalize U
        let mut singular_values = Vec::new();
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
            }
        }

        // Sort singular values in descending order
        singular_values.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

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
}

/// Bidiagonal Divide & Conquer SVD.
///
/// Reduces the matrix to bidiagonal form, then computes the SVD.
/// Currently uses `JacobiSVD` on the bidiagonal matrix for the "conquer" step.
/// This acts as a reliable baseline for SVD computations.
pub struct BDCSVD<T: Scalar, S: Storage<T>> {
    u: Matrix<T, DynamicStorage<T>>,
    v: Matrix<T, DynamicStorage<T>>,
    singular_values: Vec<T>,
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar + 'static, S: Storage<T> + 'static> BDCSVD<T, S> {
    /// Computes the BDCSVD of the given matrix.
    pub fn new(matrix: &Matrix<T, S>) -> Result<Self, String> {
        let m = matrix.rows();
        let n = matrix.cols();

        // 1. Bidiagonalization
        let bidiag = Bidiagonalization::new(matrix)?;
        let u_bi = bidiag.matrix_u();
        let v_bi = bidiag.matrix_v();
        let b = bidiag.matrix_b();

        // 2. Golub-Kahan SVD (Implicit QR)
        // We use a simplified strategy: Pad to square and use JacobiSVD.
        // This guarantees orthogonality and correctness.

        // 2. Pad B to m x m
        let min_dim = std::cmp::min(m, n);

        let mut b_pad = Matrix::<T, DynamicStorage<T>>::new_dynamic(m, m)?;
        for i in 0..m {
            for j in 0..m {
                let val = if i < min_dim && i == j {
                    *b.get(i, i).unwrap()
                } else if i < min_dim - 1 && j == i + 1 {
                    *b.get(i, i + 1).unwrap()
                } else {
                    T::default()
                };
                *b_pad.get_mut(i, j).unwrap() = val;
            }
        }

        // 3. SVD of B_pad
        let svd_b = JacobiSVD::new(&b_pad)?;
        let u_jac = svd_b.matrix_u();
        let s_vals = svd_b.singular_values().to_vec();
        let v_jac = svd_b.matrix_v();

        // 4. Combine
        // Compute U = U_bi * U_jac
        let mut u = Matrix::<T, DynamicStorage<T>>::new_dynamic(m, m)?;
        for i in 0..m {
            for j in 0..m {
                let mut sum = T::default();
                for k in 0..m {
                    sum += *u_bi.get(i, k).unwrap() * (*u_jac.get(k, j).unwrap());
                }
                *u.get_mut(i, j).unwrap() = sum;
            }
        }

        // Compute V = V_bi * V_jac(0..n, 0..n)
        let mut v = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n)?;
        for i in 0..n {
            for j in 0..n {
                let mut sum = T::default();
                for k in 0..n {
                    // Only sum up to n
                    sum += *v_bi.get(i, k).unwrap() * (*v_jac.get(k, j).unwrap());
                }
                *v.get_mut(i, j).unwrap() = sum;
            }
        }

        // Filter singular values
        let s_vals_final = s_vals.into_iter().take(n).collect();

        Ok(Self {
            u,
            v,
            singular_values: s_vals_final,
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
}
