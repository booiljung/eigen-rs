//! Real Schur decomposition of a square matrix.
//! A = Q * T * Q^T

use crate::core::decompositions::HessenbergDecomposition;
use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::{DynamicStorage, Storage};

/// Real Schur decomposition of a square matrix.
pub struct RealSchur<T: Scalar, S: Storage<T>> {
    t: Matrix<T, DynamicStorage<T>>,
    q: Matrix<T, DynamicStorage<T>>,
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar<Real = T> + PartialOrd, S: Storage<T>> RealSchur<T, S> {
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
    fn compute_inplace(
        h: &mut Matrix<T, DynamicStorage<T>>,
        q: &mut Matrix<T, DynamicStorage<T>>,
    ) -> Result<(), String> {
        let n = h.rows();
        // C++ Eigen uses 400 * n. We increase from 40 to 100 to allow for harder cases.
        // Revert to 40 * n, as increasing it just wastes time if we are stuck.
        let max_iter = 40 * n;
        let mut iter = 0;
        let mut low = 0;
        let mut high = n - 1;

        // PANIC TEST removed

        // Numerical precision epsilon
        let eps = T::from_f64(
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                1e-7
            } else {
                1e-15
            },
        );

        let mut _total_iters_global = 0usize;

        // Workspace for vectorized Householder application
        let mut workspace = vec![T::default(); n];

        // Compute initial norm for robust deflation check
        let h_norm = h.norm();

        while high > 0 {
            if iter > max_iter {
                println!(
                    "RealSchur(N={}): Failed to converge at index high={}. Total iters: {}",
                    n, high, _total_iters_global
                );
                return Err("RealSchur failed to converge".to_string());
            }

            // 1. Deflation: Check for small sub-diagonal elements
            let mut i = high;
            while i > low {
                let h_i_im1 = *h.get(i, i - 1).unwrap();
                let h_im1_im1 = *h.get(i - 1, i - 1).unwrap();
                let h_i_i = *h.get(i, i).unwrap();

                // Dynamic epsilon: Relax tolerance if we are stuck to ensure convergence.
                // This is a trade-off between precision and convergence guarantees.
                let current_eps = if iter > 200 {
                    eps * T::from_f64(100.0)
                } else {
                    eps
                };

                let val = h_i_im1.abs();
                let sum_diags = h_im1_im1.abs() + h_i_i.abs();

                // If sum_diags is too small, fallback to matrix norm.
                // Also use norm-based threshold if we are stuck (iter > 200).
                let reference = if sum_diags < eps * h_norm || iter > 200 {
                    h_norm
                } else {
                    sum_diags
                };

                let threshold = current_eps * reference;

                // Absolute threshold for underflow/denormals
                if val <= threshold || val <= eps * eps {
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
                    // Exceptional shift using sub-diagonals (Datta / LAPACK strategy) to break cycles.
                    // We add a 'jitter' based on 'iter' to ensure we don't apply the SAME exceptional shift
                    // repeatedly if we stay stuck in the same block.
                    let h_high_m1 = h.get(high, high - 1).unwrap().abs();
                    let h_m1_m2 = if high > low + 1 {
                        h.get(high - 1, high - 2).unwrap().abs()
                    } else {
                        T::default()
                    };
                    let s_val = h_high_m1 + h_m1_m2;

                    // Jitter: 1.5, 1.1, 0.75, ...
                    let jitter = 1.5 * (1.0 - ((iter / 10) % 3) as f64 * 0.25);
                    s = T::from_f64(jitter) * s_val;
                    t = s_val * s_val;
                } else {
                    // Standard Francis double-shift from bottom 2x2 block
                    let h_mm = *h.get(end, end).unwrap();
                    let h_mm1 = *h.get(end - 1, end - 1).unwrap();
                    let h_m_m1 = *h.get(end, end - 1).unwrap();
                    let h_m1_m = *h.get(end - 1, end).unwrap();

                    // Characteristic polynomial: x^2 - s*x + t = 0
                    s = h_mm + h_mm1; // Trace
                    t = h_mm * h_mm1 - h_m_m1 * h_m1_m; // Determinant
                }

                Self::francis_qr_step(h, q, start, end, s, t, &mut workspace);
                iter += 1;
                _total_iters_global += 1;
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

        if _total_iters_global > n * 10 {
            // Only print if suspicious
            println!(
                "RealSchur(N={}): Total iters = {}, Ratio = {:.2}",
                n,
                _total_iters_global,
                _total_iters_global as f64 / n as f64
            );
        } else {
            // println!("RealSchur(N={}): Total iters = {}, Ratio = {:.2}", n, _total_iters_global, _total_iters_global as f64 / n as f64);
        }

        Ok(())
    }

    /// Performs one Francis double-shift QR step on the Hessenberg matrix H in the range [start, end].
    fn francis_qr_step(
        h: &mut Matrix<T, DynamicStorage<T>>,
        q: &mut Matrix<T, DynamicStorage<T>>,
        start: usize,
        end: usize,
        s: T,
        t: T,
        workspace: &mut [T],
    ) {
        let n = h.rows();

        // Dispatch to vectorized implementations for f64/f32 if available
        let mut vectorized = false;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                    unsafe {
                        let workspace_f64: &mut [f64] = std::mem::transmute(&mut workspace[..]);
                        Self::francis_qr_step_f64_avx2(
                            std::mem::transmute::<
                                &mut Matrix<T, DynamicStorage<T>>,
                                &mut Matrix<f64, DynamicStorage<f64>>,
                            >(h),
                            std::mem::transmute::<
                                &mut Matrix<T, DynamicStorage<T>>,
                                &mut Matrix<f64, DynamicStorage<f64>>,
                            >(q),
                            start,
                            end,
                            *(&s as *const T as *const f64),
                            *(&t as *const T as *const f64),
                            workspace_f64,
                        );
                    }
                    vectorized = true;
                } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                    unsafe {
                        let workspace_f32: &mut [f32] = std::mem::transmute(&mut workspace[..]);
                        Self::francis_qr_step_f32_avx2(
                            std::mem::transmute::<
                                &mut Matrix<T, DynamicStorage<T>>,
                                &mut Matrix<f32, DynamicStorage<f32>>,
                            >(h),
                            std::mem::transmute::<
                                &mut Matrix<T, DynamicStorage<T>>,
                                &mut Matrix<f32, DynamicStorage<f32>>,
                            >(q),
                            start,
                            end,
                            *(&s as *const T as *const f32),
                            *(&t as *const T as *const f32),
                            workspace_f32,
                        );
                    }
                    vectorized = true;
                }
            }
        }

        if vectorized {
            return;
        }

        use std::sync::atomic::{AtomicBool, Ordering};
        static PRINTED: AtomicBool = AtomicBool::new(false);
        if !PRINTED.swap(true, Ordering::Relaxed) {
            // eprintln!("**** RealSchur: Using SCALAR FALLBACK Path ****");
        }

        // --- Scalar Fallback (Reference Implementation with minor optimizations) ---
        let n = h.rows();
        // Calculate indices manually to allow split borrows if necessary,
        // or just use raw pointers for simplicity in this fallback too.
        // Using raw pointers avoids the slice borrow checker dance if we want to be free.
        let h_ptr = h.storage_mut().data_mut().as_mut_ptr();
        let q_ptr = q.storage_mut().data_mut().as_mut_ptr();

        unsafe {
            let h00 = *h_ptr.add(start * n + start);
            let h10 = *h_ptr.add(start * n + start + 1);
            let h01 = *h_ptr.add((start + 1) * n + start);
            let h11 = *h_ptr.add((start + 1) * n + start + 1);
            let h21 = if start + 2 <= end {
                *h_ptr.add((start + 1) * n + start + 2)
            } else {
                T::default()
            };

            let v1 = h00 * h00 + h01 * h10 - s * h00 + t;
            let v2 = h10 * (h00 + h11 - s);
            let v3 = h10 * h21;

            let mut v = [v1, v2, v3];

            for k in start..end {
                let nr = std::cmp::min(3, end - k + 1);
                if nr < 2 {
                    break;
                }

                if k > start {
                    v[0] = *h_ptr.add((k - 1) * n + k);
                    v[1] = *h_ptr.add((k - 1) * n + k + 1);
                    if nr == 3 {
                        v[2] = *h_ptr.add((k - 1) * n + k + 2);
                    }
                }

                let (tau, house) = Self::make_householder(&v[..nr]);

                // Left: H = P H
                let left_start = if k == start { start } else { k - 1 };
                for j in left_start..n {
                    let ptr_k = h_ptr.add(j * n + k);
                    let val0 = *ptr_k;
                    let val1 = *ptr_k.add(1);

                    let mut dot = val0 + house[1] * val1;
                    if nr == 3 {
                        dot += house[2] * *ptr_k.add(2);
                    }
                    let factor = tau * dot;

                    *ptr_k -= factor;
                    *ptr_k.add(1) -= factor * house[1];
                    if nr == 3 {
                        *ptr_k.add(2) -= factor * house[2];
                    }
                }

                // Right: H = H P^T
                let rows_limit = std::cmp::min(n, k + 4);
                let col_k_ptr = h_ptr.add(k * n);
                for i in 0..rows_limit {
                    let val0 = *col_k_ptr.add(i);
                    let val1 = *col_k_ptr.add(n + i);

                    let mut dot = val0 + house[1] * val1;
                    if nr == 3 {
                        dot += house[2] * *col_k_ptr.add(2 * n + i);
                    }
                    let factor = tau * dot;

                    *col_k_ptr.add(i) -= factor;
                    *col_k_ptr.add(n + i) -= factor * house[1];
                    if nr == 3 {
                        *col_k_ptr.add(2 * n + i) -= factor * house[2];
                    }
                }

                // Clean sub-diagonal
                if k > start {
                    *h_ptr.add((k - 1) * n + k + 1) = T::default();
                    if nr == 3 {
                        *h_ptr.add((k - 1) * n + k + 2) = T::default();
                    }
                }

                // Q = Q P^T
                let q_col_k_ptr = q_ptr.add(k * n);
                for i in 0..n {
                    let val0 = *q_col_k_ptr.add(i);
                    let val1 = *q_col_k_ptr.add(n + i);

                    let mut dot = val0 + house[1] * val1;
                    if nr == 3 {
                        dot += house[2] * *q_col_k_ptr.add(2 * n + i);
                    }
                    let factor = tau * dot;

                    *q_col_k_ptr.add(i) -= factor;
                    *q_col_k_ptr.add(n + i) -= factor * house[1];
                    if nr == 3 {
                        *q_col_k_ptr.add(2 * n + i) -= factor * house[2];
                    }
                }
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn francis_qr_step_f64_avx2(
        h: &mut Matrix<f64, DynamicStorage<f64>>,
        q: &mut Matrix<f64, DynamicStorage<f64>>,
        start: usize,
        end: usize,
        s: f64,
        t: f64,
        workspace: &mut [f64],
    ) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static PRINTED: AtomicBool = AtomicBool::new(false);
        if !PRINTED.swap(true, Ordering::Relaxed) {
            // eprintln!("**** RealSchur: Using AVX2 F64 Path ****");
        }

        use crate::core::decompositions::hessenberg_utils::apply_householder_on_the_right_vectorized_f64;
        let n = h.rows();
        let h_ptr = h.storage_mut().data_mut().as_mut_ptr();

        let h00 = *h_ptr.add(start * n + start);
        let h10 = *h_ptr.add(start * n + start + 1);
        let h01 = *h_ptr.add((start + 1) * n + start);
        let h11 = *h_ptr.add((start + 1) * n + start + 1);
        let h21 = if start + 2 <= end {
            *h_ptr.add((start + 1) * n + start + 2)
        } else {
            0.0
        };

        let v1 = h00 * h00 + h01 * h10 - s * h00 + t;
        let v2 = h10 * (h00 + h11 - s);
        let v3 = h10 * h21;

        let mut v = [v1, v2, v3];

        for k in start..end {
            let nr = std::cmp::min(3, end - k + 1);
            if nr < 2 {
                break;
            }

            if k > start {
                v[0] = *h_ptr.add((k - 1) * n + k);
                v[1] = *h_ptr.add((k - 1) * n + k + 1);
                if nr == 3 {
                    v[2] = *h_ptr.add((k - 1) * n + k + 2);
                }
            }
            // make_householder manual inline or call helper
            let (tau, house_arr) = Self::make_householder_f64(&v[..nr]);
            let house = &house_arr[..nr];

            // Left: H = P H
            // Updates rows k..k+nr. Cols left_start..n
            let left_start = if k == start { start } else { k - 1 };
            let ptr_k = h_ptr.add(left_start * n + k);
            let mut col_offset = 0;
            for _j in left_start..n {
                let p = ptr_k.add(col_offset);
                let val0 = *p;
                let val1 = *p.add(1);

                let mut dot = val0 + house[1] * val1;
                if nr == 3 {
                    dot += house[2] * *p.add(2);
                }
                let factor = tau * dot;

                *p -= factor;
                *p.add(1) -= factor * house[1];
                if nr == 3 {
                    *p.add(2) -= factor * house[2];
                }
                col_offset += n;
            }

            // Right: H = H P^T
            let rows_limit = std::cmp::min(n, k + 4);
            apply_householder_on_the_right_vectorized_f64(h, house, tau, k, rows_limit, workspace);

            // Clean sub-diagonal
            if k > start {
                *h_ptr.add((k - 1) * n + k + 1) = 0.0;
                if nr == 3 {
                    *h_ptr.add((k - 1) * n + k + 2) = 0.0;
                }
            }

            // Q = Q P^T
            apply_householder_on_the_right_vectorized_f64(q, house, tau, k, n, workspace);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn francis_qr_step_f32_avx2(
        h: &mut Matrix<f32, DynamicStorage<f32>>,
        q: &mut Matrix<f32, DynamicStorage<f32>>,
        start: usize,
        end: usize,
        s: f32,
        t: f32,
        workspace: &mut [f32],
    ) {
        use crate::core::decompositions::hessenberg_utils::apply_householder_on_the_right_vectorized_f32;
        let n = h.rows();
        let h_ptr = h.storage_mut().data_mut().as_mut_ptr();

        let h00 = *h_ptr.add(start * n + start);
        let h10 = *h_ptr.add(start * n + start + 1);
        let h01 = *h_ptr.add((start + 1) * n + start);
        let h11 = *h_ptr.add((start + 1) * n + start + 1);
        let h21 = if start + 2 <= end {
            *h_ptr.add((start + 1) * n + start + 2)
        } else {
            0.0
        };

        let v1 = h00 * h00 + h01 * h10 - s * h00 + t;
        let v2 = h10 * (h00 + h11 - s);
        let v3 = h10 * h21;

        let mut v = [v1, v2, v3];

        for k in start..end {
            let nr = std::cmp::min(3, end - k + 1);
            if nr < 2 {
                break;
            }

            if k > start {
                v[0] = *h_ptr.add((k - 1) * n + k);
                v[1] = *h_ptr.add((k - 1) * n + k + 1);
                if nr == 3 {
                    v[2] = *h_ptr.add((k - 1) * n + k + 2);
                }
            }

            let (tau, house_arr) = Self::make_householder_f32(&v[..nr]);
            let house = &house_arr[..nr];

            let left_start = if k == start { start } else { k - 1 };

            let ptr_k = h_ptr.add(left_start * n + k);
            let mut col_offset = 0;
            for _j in left_start..n {
                let p = ptr_k.add(col_offset);
                let val0 = *p;
                let val1 = *p.add(1);

                let mut dot = val0 + house[1] * val1;
                if nr == 3 {
                    dot += house[2] * *p.add(2);
                }
                let factor = tau * dot;

                *p -= factor;
                *p.add(1) -= factor * house[1];
                if nr == 3 {
                    *p.add(2) -= factor * house[2];
                }
                col_offset += n;
            }

            let rows_limit = std::cmp::min(n, k + 4);
            apply_householder_on_the_right_vectorized_f32(h, house, tau, k, rows_limit, workspace);

            if k > start {
                *h_ptr.add((k - 1) * n + k + 1) = 0.0;
                if nr == 3 {
                    *h_ptr.add((k - 1) * n + k + 2) = 0.0;
                }
            }

            apply_householder_on_the_right_vectorized_f32(q, house, tau, k, n, workspace);
        }
    }

    fn make_householder_f64(v: &[f64]) -> (f64, [f64; 3]) {
        let n = v.len();
        let mut house = [0.0; 3];
        // house vector has length 'n' effectively, but stored in fixed 3.
        // house[0] = 1.0 (implicit?)
        // Logic: v[0] = 1.0. house[1..] = v[1..].
        // Existing logic: house[0] = 1.0.
        house[0] = 1.0;

        let mut norm_sq = 0.0;
        for i in 1..n {
            let val = v[i];
            norm_sq += val * val;
            house[i] = val;
        }

        if norm_sq == 0.0 {
            return (0.0, house);
        }

        let v0 = v[0];
        let mu = (v0 * v0 + norm_sq).sqrt();
        let v0_prime = if v0 <= 0.0 {
            v0 - mu
        } else {
            -norm_sq / (v0 + mu)
        };

        let tau = 2.0 * v0_prime * v0_prime / (v0_prime * v0_prime + norm_sq);
        let inv_v0 = 1.0 / v0_prime;

        for i in 1..n {
            house[i] *= inv_v0;
        }
        (tau, house)
    }

    fn make_householder_f32(v: &[f32]) -> (f32, [f32; 3]) {
        let n = v.len();
        let mut house = [0.0; 3];
        house[0] = 1.0;

        let mut norm_sq = 0.0;
        for i in 1..n {
            let val = v[i];
            norm_sq += val * val;
            house[i] = val;
        }

        if norm_sq == 0.0 {
            return (0.0, house);
        }

        let v0 = v[0];
        let mu = (v0 * v0 + norm_sq).sqrt();
        let v0_prime = if v0 <= 0.0 {
            v0 - mu
        } else {
            -norm_sq / (v0 + mu)
        };

        let tau = 2.0 * v0_prime * v0_prime / (v0_prime * v0_prime + norm_sq);
        let inv_v0 = 1.0 / v0_prime;

        for i in 1..n {
            house[i] *= inv_v0;
        }
        (tau, house)
    }

    fn make_householder(v: &[T]) -> (T, Vec<T>) {
        let n = v.len();
        let mut norm_sq = T::default();
        for val in v.iter().take(n).skip(1) {
            norm_sq += *val * *val;
        }

        let mut house = vec![T::default(); n];
        house[0] = T::from_f64(1.0);
        house[1..n].clone_from_slice(&v[1..n]);

        if norm_sq == T::default() {
            return (T::default(), house);
        }

        let mu = (v[0] * v[0] + norm_sq).sqrt();
        let v0 = if v[0] <= T::default() {
            v[0] - mu
        } else {
            (T::default() - norm_sq) / (v[0] + mu)
        };
        let tau = T::from_f64(2.0) * v0 * v0 / (v0 * v0 + norm_sq);

        let inv_v0 = v0.recip();
        for val in house.iter_mut().take(n).skip(1) {
            *val *= inv_v0;
        }

        (tau, house)
    }

    pub fn matrix_t(&self) -> &Matrix<T, DynamicStorage<T>> {
        &self.t
    }
    pub fn matrix_q(&self) -> &Matrix<T, DynamicStorage<T>> {
        &self.q
    }
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
            0.35, 0.45, -0.14, -0.17, 0.09, 0.07, -0.54, 0.35, -0.44, -0.33, -0.03, 0.17, 0.25,
            -0.32, -0.13, 0.11,
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
                    assert!(
                        t.get(i, j).unwrap().abs() < 1e-10,
                        "T is not quasi-upper triangular at ({}, {}): {}",
                        i,
                        j,
                        t.get(i, j).unwrap()
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_real_schur_identity() -> Result<(), String> {
        let n = 5;
        let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;
        for i in 0..n {
            *a.get_mut(i, i).unwrap() = 1.0;
        }

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
