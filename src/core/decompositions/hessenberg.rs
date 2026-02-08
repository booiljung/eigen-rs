//! Hessenberg decomposition of a square matrix.
//! A = Q * H * Q^T

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::{DynamicStorage, Storage};

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

        for i in 0..n - 2 {
            // 1. Compute Householder reflection for column i starting from i+1
            let mut norm_sq = T::default();
            for k in i + 1..n {
                let val = *mat_a.get(k, i).unwrap();
                norm_sq += val.norm_sq();
            }
            let norm = norm_sq.sqrt();

            if norm != T::default() {
                let v0 = *mat_a.get(i + 1, i).unwrap();
                let sigma = if v0 >= T::default() {
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
                h_coeffs[i] = tau;

                // 2. Apply reflection from the left: A = (I - tau v v^T) A
                // A[i+1:n, i+1:n] = (I - tau v v^T) A[i+1:n, i+1:n]
                // Note: We also apply it to the i-th column's tail (below i+1) but carefully.
                for j in i + 1..n {
                    let mut dot = *mat_a.get(i + 1, j).unwrap();
                    for k in i + 2..n {
                        dot += (*mat_a.get(k, i).unwrap()).conj() * (*mat_a.get(k, j).unwrap());
                    }

                    let factor = tau * dot;
                    *mat_a.get_mut(i + 1, j).unwrap() -= factor;
                    for k in i + 2..n {
                        let vk = *mat_a.get(k, i).unwrap();
                        *mat_a.get_mut(k, j).unwrap() -= factor * vk;
                    }
                }

                // 3. Apply reflection from the right: A = A (I - tau v v^*)
                // For similarity transform, we need A = H A H*
                // Right reflection is A = A H_i* = A (I - tau.conj() v v*)
                let tau_conj = tau.conj();
                for j in 0..n {
                    let mut dot = *mat_a.get(j, i + 1).unwrap();
                    for k in i + 2..n {
                        dot += (*mat_a.get(j, k).unwrap()) * (*mat_a.get(k, i).unwrap());
                    }

                    let factor = tau_conj * dot;
                    *mat_a.get_mut(j, i + 1).unwrap() -= factor;
                    for k in i + 2..n {
                        let vk = *mat_a.get(k, i).unwrap();
                        *mat_a.get_mut(j, k).unwrap() -= factor * vk.conj();
                    }
                }

                // Restore sub-diagonal element
                *mat_a.get_mut(i + 1, i).unwrap() = T::default() - sigma;
            } else {
                h_coeffs[i] = T::default();
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
