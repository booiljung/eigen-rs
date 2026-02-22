//! Hessenberg decomposition of a square matrix.
//! A = Q * H * Q^T

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::{DynamicStorage, Storage};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// Hessenberg decomposition of a square matrix.
pub struct HessenbergDecomposition<T: Scalar, S: Storage<T>> {
    packed_matrix: Matrix<T, DynamicStorage<T>>,
    h_coeffs: Vec<T>,
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar, S: Storage<T>> HessenbergDecomposition<T, S> {
    /// Computes the Hessenberg decomposition of the given square matrix.
    pub fn new(matrix: &Matrix<T, S>) -> Result<Self, String> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err("Hessenberg decomposition requires a square matrix".to_string());
        }

        let mut packed_matrix = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n)?;
        packed_matrix.assign(matrix)?;

        let mut h_coeffs = vec![T::default(); if n > 2 { n - 2 } else { 1 }];

        if n > 2 {
            Self::hessenberg_inplace(&mut packed_matrix, &mut h_coeffs);
        }

        Ok(Self {
            packed_matrix,
            h_coeffs,
            _phantom: std::marker::PhantomData,
        })
    }

    fn hessenberg_inplace(mat_a: &mut Matrix<T, DynamicStorage<T>>, h_coeffs: &mut [T]) {
        let n = mat_a.rows();

        // Workspace buffers
        // v_buf: max length n.
        // w_buf: max length n (used for right update).
        let mut v_buf_alloc = vec![T::default(); n];
        let mut w_buf_alloc = vec![T::default(); n];

        for (i, h_coeff) in h_coeffs.iter_mut().enumerate().take(n - 2) {
            let v_len = n - (i + 1);

            // 1. Compute Householder reflection for column i starting from i+1
            // Try vectorized first
            let mut computed_vectorized = false;

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            unsafe {
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()
                    && is_x86_feature_detected!("fma")
                {
                    let mat_ptr: *mut Matrix<T, DynamicStorage<T>> = mat_a;
                    let mat_f64: &mut Matrix<f64, DynamicStorage<f64>> =
                        std::mem::transmute(mat_ptr);

                    let v_slice_f64: &mut [f64] = std::mem::transmute(&mut v_buf_alloc[0..v_len]);

                    // Compute Householder vector
                    let (tau_val, beta_val) = crate::core::decompositions::hessenberg_utils::compute_householder_vectorized_f64(
                        mat_f64, i, v_slice_f64
                    );

                    let tau: T = std::mem::transmute_copy(&tau_val);
                    *h_coeff = tau;

                    // Restore beta (sub-diagonal element)
                    *mat_a.get_mut(i + 1, i).unwrap() = std::mem::transmute_copy(&beta_val);

                    if tau_val != 0.0 {
                        let inputs_f64: &[f64] = v_slice_f64;
                        let w_slice_f64: &mut [f64] = std::mem::transmute(&mut w_buf_alloc[0..n]); // w needs up to n size? Apply right uses rows_end=n

                        crate::core::decompositions::hessenberg_utils::apply_householder_on_the_left_vectorized_f64(
                            mat_f64, inputs_f64, tau_val, i + 1, i + 1, n
                        );
                        // Right update: affects columns i+1..n? No, right update affects all rows, columns i+1..n?
                        // A = A (I - tau v v^T). v is size n-(i+1).
                        // It mixes columns i+1..n.
                        // Rows updated? All rows 0..n.
                        crate::core::decompositions::hessenberg_utils::apply_householder_on_the_right_vectorized_f64(
                            mat_f64, inputs_f64, tau_val, i + 1, n, w_slice_f64
                        );
                    }
                    computed_vectorized = true;
                } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
                    && is_x86_feature_detected!("fma")
                {
                    let mat_ptr: *mut Matrix<T, DynamicStorage<T>> = mat_a;
                    let mat_f32: &mut Matrix<f32, DynamicStorage<f32>> =
                        std::mem::transmute(mat_ptr);

                    let v_slice_f32: &mut [f32] = std::mem::transmute(&mut v_buf_alloc[0..v_len]);

                    let (tau_val, beta_val) = crate::core::decompositions::hessenberg_utils::compute_householder_vectorized_f32(
                        mat_f32, i, v_slice_f32
                    );

                    let tau: T = std::mem::transmute_copy(&tau_val);
                    *h_coeff = tau;
                    *mat_a.get_mut(i + 1, i).unwrap() = std::mem::transmute_copy(&beta_val);

                    if tau_val != 0.0 {
                        let inputs_f32: &[f32] = v_slice_f32;
                        let w_slice_f32: &mut [f32] = std::mem::transmute(&mut w_buf_alloc[0..n]);

                        crate::core::decompositions::hessenberg_utils::apply_householder_on_the_left_vectorized_f32(
                            mat_f32, inputs_f32, tau_val, i + 1, i + 1, n
                        );
                        crate::core::decompositions::hessenberg_utils::apply_householder_on_the_right_vectorized_f32(
                            mat_f32, inputs_f32, tau_val, i + 1, n, w_slice_f32
                        );
                    }
                    computed_vectorized = true;
                }
            }

            if !computed_vectorized {
                let mut norm_sq = T::default();
                for k in i + 1..n {
                    let val = *mat_a.get(k, i).unwrap();
                    norm_sq += T::from_real(val.norm_sq());
                }
                let norm = norm_sq.sqrt();

                if norm != T::default() {
                    let v0 = *mat_a.get(i + 1, i).unwrap();
                    let sigma = if v0.real() >= <T::Real as num_traits::Zero>::zero() {
                        norm
                    } else {
                        T::default() - norm
                    };

                    let v0_plus_sigma = v0 + sigma;
                    let inv_v0_plus_sigma = v0_plus_sigma.recip();

                    // Scale Householder vector: v[0] becomes 1, rest stored in mat_a
                    for k in i + 2..n {
                        *mat_a.get_mut(k, i).unwrap() *= inv_v0_plus_sigma;
                    }

                    // Householder coefficient tau
                    let tau = v0_plus_sigma.conj() / sigma;
                    *h_coeff = tau;

                    // Copy v to buffer for scalar application
                    v_buf_alloc[0] = T::from_f64(1.0);
                    for k in 1..v_len {
                        v_buf_alloc[k] = *mat_a.get(i + 1 + k, i).unwrap();
                    }
                    let v_slice = &v_buf_alloc[0..v_len];

                    // 2. Apply reflection from the left: A = (I - tau v v^T) A
                    for j in i + 1..n {
                        let mut dot = T::default();
                        for k in 0..v_len {
                            dot += v_slice[k].conj() * *mat_a.get(i + 1 + k, j).unwrap();
                        }

                        let factor = tau * dot;
                        for k in 0..v_len {
                            *mat_a.get_mut(i + 1 + k, j).unwrap() -= factor * v_slice[k];
                        }
                    }

                    // 3. Apply reflection from the right: A = A (I - tau v v^*)
                    let tau_conj = tau.conj();
                    for j in 0..n {
                        let mut dot = T::default();
                        for k in 0..v_len {
                            dot += *mat_a.get(j, i + 1 + k).unwrap() * v_slice[k];
                        }

                        let factor = tau_conj * dot;
                        for k in 0..v_len {
                            *mat_a.get_mut(j, i + 1 + k).unwrap() -= factor * v_slice[k].conj();
                        }
                    }

                    // Restore sub-diagonal element
                    *mat_a.get_mut(i + 1, i).unwrap() = T::default() - sigma;
                } else {
                    *h_coeff = T::default();
                }
            }
        }
    }

    /// Returns the Hessenberg matrix H.
    pub fn matrix_h(&self) -> Matrix<T, DynamicStorage<T>> {
        let n = self.packed_matrix.rows();
        let mut h = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n).unwrap();

        for j in 0..n {
            for i in 0..n {
                if i <= j + 1 {
                    *h.get_mut(i, j).unwrap() = *self.packed_matrix.get(i, j).unwrap();
                } else {
                    *h.get_mut(i, j).unwrap() = T::default();
                }
            }
        }
        h
    }

    /// Returns the orthogonal matrix Q.
    pub fn matrix_q(&self) -> Matrix<T, DynamicStorage<T>> {
        let n = self.packed_matrix.rows();
        let mut q = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n).unwrap();

        for i in 0..n {
            for j in 0..n {
                *q.get_mut(i, j).unwrap() = if i == j {
                    T::from_usize(1)
                } else {
                    T::default()
                };
            }
        }

        // Q = H0 * H1 * ... * H_{n-3}
        if n > 2 {
            for i in (0..n - 2).rev() {
                let tau = self.h_coeffs[i];
                if tau != T::default() {
                    let tau_conj = tau.conj();
                    for j in i + 1..n {
                        let mut dot = *q.get(i + 1, j).unwrap();
                        for k in i + 2..n {
                            dot += (*self.packed_matrix.get(k, i).unwrap()).conj()
                                * (*q.get(k, j).unwrap());
                        }

                        let factor = tau_conj * dot;
                        *q.get_mut(i + 1, j).unwrap() -= factor;
                        for k in i + 2..n {
                            let vk = *self.packed_matrix.get(k, i).unwrap();
                            *q.get_mut(k, j).unwrap() -= factor * vk;
                        }
                    }
                }
            }
        }
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::storage::DynamicStorage;

    #[test]
    fn test_hessenberg_basic() -> Result<(), String> {
        let n = 4;
        let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;
        let data = [
            4.0, 1.0, 2.0, 3.0, 1.0, 5.0, 1.0, 2.0, 2.0, 1.0, 6.0, 1.0, 3.0, 2.0, 1.0, 7.0,
        ];
        for i in 0..n {
            for j in 0..n {
                *a.get_mut(i, j).unwrap() = data[i * n + j];
            }
        }

        let decomp = HessenbergDecomposition::new(&a)?;
        let h = decomp.matrix_h();
        let q = decomp.matrix_q();

        // Check Q is orthogonal: Q^T * Q = I
        let mut qtq = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q.get(k, i).unwrap() * q.get(k, j).unwrap();
                }
                *qtq.get_mut(i, j).unwrap() = sum;
            }
        }

        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((qtq.get(i, j).unwrap() - expected).abs() < 1e-10);
            }
        }

        // Check A = Q * H * Q^T => Q^T * A * Q = H
        let mut qtaq = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;
        // Temporary: temp = A * Q
        let mut temp = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a.get(i, k).unwrap() * q.get(k, j).unwrap();
                }
                *temp.get_mut(i, j).unwrap() = sum;
            }
        }
        // qtaq = Q^T * temp
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q.get(k, i).unwrap() * temp.get(k, j).unwrap();
                }
                *qtaq.get_mut(i, j).unwrap() = sum;
            }
        }

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (qtaq.get(i, j).unwrap() - h.get(i, j).unwrap()).abs() < 1e-10,
                    "Mismatch at ({}, {}): {} != {}",
                    i,
                    j,
                    qtaq.get(i, j).unwrap(),
                    h.get(i, j).unwrap()
                );
            }
        }

        // Check H is upper Hessenberg
        for i in 0..n {
            for j in 0..n {
                if i > j + 1 {
                    assert!(
                        h.get(i, j).unwrap().abs() < 1e-10,
                        "H is not upper Hessenberg at ({}, {}): {}",
                        i,
                        j,
                        h.get(i, j).unwrap()
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_hessenberg_identity() -> Result<(), String> {
        let n = 3;
        let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;
        for i in 0..n {
            *a.get_mut(i, i).unwrap() = 1.0;
        }

        let decomp = HessenbergDecomposition::new(&a)?;
        let h = decomp.matrix_h();

        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((h.get(i, j).unwrap() - expected).abs() < 1e-10);
            }
        }
        Ok(())
    }

    #[test]
    fn test_hessenberg_random_non_symmetric() -> Result<(), String> {
        let n = 5;
        let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;
        // Simple "random" deterministic matrix
        for i in 0..n {
            for j in 0..n {
                *a.get_mut(i, j).unwrap() =
                    (i as f64 * 1.1 + j as f64 * 0.7 + (i * j) as f64 * 0.1).sin();
            }
        }

        let decomp = HessenbergDecomposition::new(&a)?;
        let h = decomp.matrix_h();
        let q = decomp.matrix_q();

        // Check A = Q H Q^T
        // Check A*Q == Q*H
        let mut aq = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;
        let mut qh = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;

        for i in 0..n {
            for j in 0..n {
                let mut sum_aq = 0.0;
                let mut sum_qh = 0.0;
                for k in 0..n {
                    sum_aq += a.get(i, k).unwrap() * q.get(k, j).unwrap();
                    sum_qh += q.get(i, k).unwrap() * h.get(k, j).unwrap();
                }
                *aq.get_mut(i, j).unwrap() = sum_aq;
                *qh.get_mut(i, j).unwrap() = sum_qh;
            }
        }

        for i in 0..n {
            for j in 0..n {
                assert!((aq.get(i, j).unwrap() - qh.get(i, j).unwrap()).abs() < 1e-10);
            }
        }
        Ok(())
    }
}
