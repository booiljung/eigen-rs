//! Real Schur decomposition of a square matrix.
//! A = Q * T * Q^T

use crate::core::matrix::Matrix;
use crate::core::storage::{Storage, DynamicStorage};
use crate::core::scalar::Scalar;
use crate::core::decompositions::HessenbergDecomposition;

/// Real Schur decomposition of a square matrix.
pub struct RealSchur<T: Scalar, S: Storage<T>> {
    t: Matrix<T, DynamicStorage<T>>,
    q: Matrix<T, DynamicStorage<T>>,
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar, S: Storage<T>> RealSchur<T, S> {
    /// Computes the Real Schur decomposition of the given square matrix.
    pub fn new(matrix: &Matrix<T, S>) -> Result<Self, String> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err("Real Schur decomposition requires a square matrix".to_string());
        }

        // 1. Initial Hessenberg reduction: A = Q H Q^T
        let hessenberg = HessenbergDecomposition::new(matrix)?;
        let mut h = hessenberg.matrix_h();
        let mut q = hessenberg.matrix_q();

        if n > 2 {
            Self::compute_inplace(&mut h, &mut q)?;
        } else if n == 2 {
            // Already Hessenberg and thus quasi-upper triangular for 2x2
        }

        Ok(Self {
            t: h,
            q,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Iterative QR algorithm with shifts on Hessenberg matrix.
    fn compute_inplace(h: &mut Matrix<T, DynamicStorage<T>>, q: &mut Matrix<T, DynamicStorage<T>>) -> Result<(), String> {
        let n = h.rows();
        let max_iter = 40 * n;
        let mut iter = 0;
        let mut low = 0;
        let mut high = n - 1;

        // Numerical precision epsilon
        let eps = T::from_f64(if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() { 1e-7 } else { 1e-15 });

        while high > 0 {
            if iter > max_iter {
                return Err("RealSchur failed to converge".to_string());
            }

            // 1. Deflation: Check for small sub-diagonal elements
            let mut i = high;
            while i > low {
                let h_i_im1 = *h.get(i, i - 1).unwrap();
                let h_im1_im1 = *h.get(i - 1, i - 1).unwrap();
                let h_i_i = *h.get(i, i).unwrap();
                if h_i_im1.abs() <= eps * (h_im1_im1.abs() + h_i_i.abs()) {
                    *h.get_mut(i, i - 1).unwrap() = T::default();
                    break;
                }
                i -= 1;
            }

            if i == high {
                // Found a 1x1 block at the end
                high = high.saturating_sub(1);
                iter = 0;
            } else if i == high - 1 {
                // Found a 2x2 block at the end
                high = high.saturating_sub(2);
                iter = 0;
            } else {
                // 2. Francis QR step on the sub-matrix [i..high+1, i..high+1]
                let start = i;
                let end = high;

                // Shift strategies
                let s: T;
                let t: T;

                if iter > 0 && iter % 10 == 0 {
                    // Exceptional shift using sub-diagonals (Datta / LAPACK strategy) to break cycles
                    // s = 1.5 * (|h_n,n-1| + |h_n-1,n-2|) approximated
                    let h_high_m1 = h.get(high, high - 1).unwrap().abs();
                    let h_m1_m2 = if high > low + 1 { h.get(high - 1, high - 2).unwrap().abs() } else { T::default() };
                    let s_val = h_high_m1 + h_m1_m2;
                    
                    // Specific ad-hoc values to break symmetry
                    s = T::from_f64(1.5) * s_val;
                    t = s_val * s_val; 
                } else {
                     // Standard Francis double-shift from bottom 2x2 block
                    let h_mm = *h.get(end, end).unwrap();
                    let h_mm1 = *h.get(end - 1, end - 1).unwrap();
                    let h_m_m1 = *h.get(end, end - 1).unwrap();
                    let h_m1_m = *h.get(end - 1, end).unwrap();

                    // Characteristic polynomial: x^2 - s*x + t = 0
                    s = h_mm + h_mm1;      // Trace
                    t = h_mm * h_mm1 - h_m_m1 * h_m1_m; // Determinant
                }

                Self::francis_qr_step(h, q, start, end, s, t);
                iter += 1;
            }

            // Shrink low if leading 1x1 or 2x2 blocks are decoupled
            while low < high {
                let h_lp1_l = *h.get(low + 1, low).unwrap();
                if h_lp1_l.abs() > eps {
                    break;
                }
                low += 1;
            }
        }

        Ok(())
    }

    /// Performs one Francis double-shift QR step on the Hessenberg matrix H in the range [start, end].
    /// `s` (trace) and `t` (determinant) define the implicit shift polynomial x^2 - sx + t.
    fn francis_qr_step(h: &mut Matrix<T, DynamicStorage<T>>, q: &mut Matrix<T, DynamicStorage<T>>, start: usize, end: usize, s: T, t: T) {
        let n = h.rows();
        
        // 1. Shifts s and t are passed in.

        // 2. Compute first Householder reflection
        // The first column of (H - λ1 I)(H - λ2 I) = H^2 - s*H + t*I is:
        // [ h_00^2 + h_01*h_10 - s*h_00 + t ]
        // [ h_10*(h_00 + h_11 - s)          ]
        // [ h_21*h_10                       ]
        let h00 = *h.get(start, start).unwrap();
        let h10 = *h.get(start + 1, start).unwrap();
        let h01 = *h.get(start, start + 1).unwrap();
        let h11 = *h.get(start + 1, start + 1).unwrap();
        let h21 = if start + 2 <= end { *h.get(start + 2, start + 1).unwrap() } else { T::default() };

        let v1 = h00 * h00 + h01 * h10 - s * h00 + t;
        let v2 = h10 * (h00 + h11 - s);
        let v3 = h10 * h21;

        let mut v = [v1, v2, v3];
        
        // 3. Bulge chasing
        for k in start..end { // Loop up to 'end'
            let nr = std::cmp::min(3, end - k + 1);
            if nr < 2 { break; } // Stop if not enough elements for a 2x2 or 3x3 reflection
            
            if k > start {
                // If not the first step, the vector 'v' for the Householder reflection
                // is taken from the sub-diagonal elements of H
                v[0] = *h.get(k, k - 1).unwrap();
                v[1] = *h.get(k + 1, k - 1).unwrap();
                if nr == 3 {
                    v[2] = *h.get(k + 2, k - 1).unwrap();
                }
            }

            let (tau, house) = Self::make_householder(&v[..nr]);
            
            // Left: H = P H
            let left_start = if k == start { start } else { k - 1 };
            for j in left_start..n {
                let mut dot = *h.get(k, j).unwrap() + house[1] * (*h.get(k + 1, j).unwrap());
                if nr == 3 {
                    dot += house[2] * (*h.get(k + 2, j).unwrap());
                }
                let factor = tau * dot;
                *h.get_mut(k, j).unwrap() -= factor;
                *h.get_mut(k + 1, j).unwrap() -= factor * house[1];
                if nr == 3 {
                    *h.get_mut(k + 2, j).unwrap() -= factor * house[2];
                }
            }

            // Right: H = H P^T
            // For Right multiplication, all rows can be affected in columns k, k+1, k+2
            for j in 0..n {
                let mut dot = *h.get(j, k).unwrap() + house[1] * (*h.get(j, k + 1).unwrap());
                if nr == 3 {
                    dot += house[2] * (*h.get(j, k + 2).unwrap());
                }
                let factor = tau * dot;
                *h.get_mut(j, k).unwrap() -= factor;
                *h.get_mut(j, k + 1).unwrap() -= factor * house[1];
                if nr == 3 {
                    *h.get_mut(j, k + 2).unwrap() -= factor * house[2];
                }
            }

            // Clean up sub-diagonal elements that should be zero
            if k > start {
                *h.get_mut(k + 1, k - 1).unwrap() = T::default();
                if nr == 3 {
                    *h.get_mut(k + 2, k - 1).unwrap() = T::default();
                }
            }

            // Q = Q P^T
            for j in 0..n {
                let mut dot = *q.get(j, k).unwrap() + house[1] * (*q.get(j, k + 1).unwrap());
                if nr == 3 {
                    dot += house[2] * (*q.get(j, k + 2).unwrap());
                }
                let factor = tau * dot;
                *q.get_mut(j, k).unwrap() -= factor;
                *q.get_mut(j, k + 1).unwrap() -= factor * house[1];
                if nr == 3 {
                    *q.get_mut(j, k + 2).unwrap() -= factor * house[2];
                }
            }
        }
    }

    fn make_householder(v: &[T]) -> (T, Vec<T>) {
        let n = v.len();
        let mut norm_sq = T::default();
        for i in 1..n {
            norm_sq += v[i] * v[i];
        }
        
        let mut house = vec![T::default(); n];
        house[0] = T::from_f64(1.0);
        for i in 1..n {
            house[i] = v[i];
        }

        if norm_sq == T::default() {
            return (T::default(), house);
        }

        let mu = (v[0] * v[0] + norm_sq).sqrt();
        let v0 = if v[0] <= T::default() { v[0] - mu } else { (T::default() - norm_sq) / (v[0] + mu) };
        let tau = T::from_f64(2.0) * v0 * v0 / (v0 * v0 + norm_sq);
        
        let inv_v0 = v0.recip();
        for i in 1..n {
            house[i] *= inv_v0;
        }

        (tau, house)
    }

    pub fn matrix_t(&self) -> &Matrix<T, DynamicStorage<T>> { &self.t }
    pub fn matrix_q(&self) -> &Matrix<T, DynamicStorage<T>> { &self.q }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::storage::DynamicStorage;

    #[test]
    fn test_real_schur_basic() -> Result<(), String> {
        let n = 4;
        let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;
        let data = [
             0.35,  0.45, -0.14, -0.17,
             0.09,  0.07, -0.54,  0.35,
            -0.44, -0.33, -0.03,  0.17,
             0.25, -0.32, -0.13,  0.11,
        ];
        for i in 0..n {
            for j in 0..n {
                *a.get_mut(i, j).unwrap() = data[i * n + j];
            }
        }

        let schur = RealSchur::new(&a)?;
        let t = schur.matrix_t();
        let q = schur.matrix_q();

        // 1. Check Q is orthogonal: Q^T * Q = I
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

        // 2. Check A = Q * T * Q^T => A * Q = Q * T
        let mut aq = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;
        let mut qt = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;

        for i in 0..n {
            for j in 0..n {
                let mut sum_aq = 0.0;
                let mut sum_qt = 0.0;
                for k in 0..n {
                    sum_aq += a.get(i, k).unwrap() * q.get(k, j).unwrap();
                    sum_qt += q.get(i, k).unwrap() * t.get(k, j).unwrap();
                }
                *aq.get_mut(i, j).unwrap() = sum_aq;
                *qt.get_mut(i, j).unwrap() = sum_qt;
            }
        }

        for i in 0..n {
            for j in 0..n {
                assert!((aq.get(i, j).unwrap() - qt.get(i, j).unwrap()).abs() < 1e-10);
            }
        }

        // 3. Check T is quasi-upper triangular
        for i in 0..n {
            for j in 0..n {
                if i > j + 1 {
                    assert!(t.get(i, j).unwrap().abs() < 1e-10, "T is not quasi-upper triangular at ({}, {}): {}", i, j, t.get(i, j).unwrap());
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_real_schur_identity() -> Result<(), String> {
        let n = 5;
        let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;
        for i in 0..n { *a.get_mut(i, i).unwrap() = 1.0; }

        let schur = RealSchur::new(&a)?;
        let t = schur.matrix_t();
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((t.get(i, j).unwrap() - expected).abs() < 1e-10);
            }
        }
        Ok(())
    }

    #[test]
    fn test_real_schur_random() -> Result<(), String> {
        // Larger matrix to test convergence
        let n = 10;
        let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;
        for i in 0..n {
            for j in 0..n {
                *a.get_mut(i, j).unwrap() = ((i + j) as f64 * 0.5).cos() + (i as f64 * 0.1).sin();
            }
        }

        let schur = RealSchur::new(&a)?;
        let t = schur.matrix_t();
        let q = schur.matrix_q();

        // Parity check A*Q = Q*T
        for i in 0..n {
            for j in 0..n {
                let mut aq = 0.0;
                let mut qt = 0.0;
                for k in 0..n {
                    aq += a.get(i, k).unwrap() * q.get(k, j).unwrap();
                    qt += q.get(i, k).unwrap() * t.get(k, j).unwrap();
                }
                assert!((aq - qt).abs() < 1e-9);
            }
        }

        // Orthogonality check Q^T*Q = I
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q.get(k, i).unwrap() * q.get(k, j).unwrap();
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((sum - expected).abs() < 1e-10);
            }
        }

        // Quasi-triangularity check
        for i in 0..n {
            for j in 0..n {
                if i > j + 1 {
                    assert!(t.get(i, j).unwrap().abs() < 1e-9);
                }
            }
        }

        Ok(())
    }
}
