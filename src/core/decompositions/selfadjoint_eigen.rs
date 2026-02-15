//! Eigenvalue decomposition of a selfadjoint matrix.

use crate::core::decompositions::Tridiagonalization;
use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::{DynamicStorage, Storage};

/// Eigendecomposition of a selfadjoint matrix.
pub struct SelfAdjointEigenSolver<T: Scalar, S: Storage<T>> {
    eigenvectors: Option<Matrix<T, DynamicStorage<T>>>,
    eigenvalues: Matrix<T, DynamicStorage<T>>,
    info: ComputationInfo,
    _phantom: std::marker::PhantomData<S>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ComputationInfo {
    Success,
    NumericalIssue,
    NoConvergence,
    InvalidInput,
}

impl<T: Scalar, S: Storage<T>> SelfAdjointEigenSolver<T, S> {
    /// Computes the eigendecomposition of the given selfadjoint matrix.
    pub fn new(matrix: &Matrix<T, S>, compute_eigenvectors: bool) -> Result<Self, String> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err("SelfAdjointEigenSolver requires a square matrix".to_string());
        }

        let mut solver = Self {
            eigenvectors: None,
            eigenvalues: Matrix::<T, DynamicStorage<T>>::new_dynamic(n, 1)?,
            info: ComputationInfo::InvalidInput,
            _phantom: std::marker::PhantomData,
        };

        solver.compute(matrix, compute_eigenvectors)?;
        Ok(solver)
    }

    pub fn compute(
        &mut self,
        matrix: &Matrix<T, S>,
        compute_eigenvectors: bool,
    ) -> Result<(), String> {
        let n = matrix.rows();
        if n == 0 {
            self.info = ComputationInfo::Success;
            return Ok(());
        }

        if n == 1 {
            *self.eigenvalues.get_mut(0, 0).unwrap() = *matrix.get(0, 0).unwrap();
            if compute_eigenvectors {
                let mut vecs = Matrix::<T, DynamicStorage<T>>::new_dynamic(1, 1)?;
                *vecs.get_mut(0, 0).unwrap() = T::from_f64(1.0);
                self.eigenvectors = Some(vecs);
            }
            self.info = ComputationInfo::Success;
            return Ok(());
        }

        if n == 2 {
            self.compute_2x2(matrix, compute_eigenvectors)?;
            return Ok(());
        }

        if n == 3 {
            self.compute_3x3(matrix, compute_eigenvectors)?;
            return Ok(());
        }

        // General case: Reduce to tridiagonal form and use QR
        let tri = Tridiagonalization::new(matrix)?;
        let mut diag = vec![T::default(); n];
        let mut subdiag = vec![T::default(); n - 1];

        let t = tri.matrix_t();
        for i in 0..n {
            diag[i] = *t.get(i, i).unwrap();
            if i < n - 1 {
                subdiag[i] = *t.get(i + 1, i).unwrap();
            }
        }

        if compute_eigenvectors {
            self.eigenvectors = Some(tri.matrix_q());
        }

        self.compute_from_tridiagonal(&mut diag, &mut subdiag, compute_eigenvectors)?;
        self.sort_eigenvalues(n, compute_eigenvectors);

        Ok(())
    }

    fn compute_2x2(
        &mut self,
        matrix: &Matrix<T, S>,
        compute_eigenvectors: bool,
    ) -> Result<(), String> {
        let a = *matrix.get(0, 0).unwrap();
        let b = *matrix.get(1, 0).unwrap();
        let d = *matrix.get(1, 1).unwrap();

        let trace = a + d;
        let gap = a - d;
        let disc = (gap * gap + T::from_f64(4.0) * b * b).sqrt();

        let l1 = (trace - disc) * T::from_f64(0.5);
        let l2 = (trace + disc) * T::from_f64(0.5);

        *self.eigenvalues.get_mut(0, 0).unwrap() = l1;
        *self.eigenvalues.get_mut(1, 0).unwrap() = l2;

        if compute_eigenvectors {
            let mut vecs = Matrix::<T, DynamicStorage<T>>::new_dynamic(2, 2)?;
            if b == T::default() {
                *vecs.get_mut(0, 0).unwrap() = T::from_f64(1.0);
                *vecs.get_mut(1, 0).unwrap() = T::default();
                *vecs.get_mut(0, 1).unwrap() = T::default();
                *vecs.get_mut(1, 1).unwrap() = T::from_f64(1.0);
            } else {
                let (c, s) = {
                    let tau = gap / (T::from_f64(2.0) * b);
                    let t_val = if tau >= T::default() {
                        T::from_f64(1.0) / (tau + (T::from_f64(1.0) + tau * tau).sqrt())
                    } else {
                        T::from_f64(-1.0) / (tau.abs() + (T::from_f64(1.0) + tau * tau).sqrt())
                    };
                    let c_val = (T::from_f64(1.0) + t_val * t_val).sqrt().recip();
                    let s_val = t_val * c_val;
                    (c_val, s_val)
                };
                *vecs.get_mut(0, 0).unwrap() = c;
                *vecs.get_mut(1, 0).unwrap() = -s;
                *vecs.get_mut(0, 1).unwrap() = s;
                *vecs.get_mut(1, 1).unwrap() = c;
            }
            // Ensure eigenvectors are sorted along with eigenvalues
            // (l1 < l2)
            self.eigenvectors = Some(vecs);
        }

        self.info = ComputationInfo::Success;
        Ok(())
    }

    fn compute_3x3(
        &mut self,
        matrix: &Matrix<T, S>,
        compute_eigenvectors: bool,
    ) -> Result<(), String> {
        let m00 = *matrix.get(0, 0).unwrap();
        let m10 = *matrix.get(1, 0).unwrap();
        let m20 = *matrix.get(2, 0).unwrap();
        let m11 = *matrix.get(1, 1).unwrap();
        let m21 = *matrix.get(2, 1).unwrap();
        let m22 = *matrix.get(2, 2).unwrap();

        let shift = (m00 + m11 + m22) / T::from_f64(3.0);

        let s00 = m00 - shift;
        let s11 = m11 - shift;
        let s22 = m22 - shift;

        let norm = (s00 * s00
            + s11 * s11
            + s22 * s22
            + T::from_f64(2.0) * (m10 * m10 + m20 * m20 + m21 * m21))
            .sqrt();
        if norm < T::epsilon() {
            *self.eigenvalues.get_mut(0, 0).unwrap() = shift;
            *self.eigenvalues.get_mut(1, 0).unwrap() = shift;
            *self.eigenvalues.get_mut(2, 0).unwrap() = shift;
            if compute_eigenvectors {
                let mut _vecs = Matrix::<T, DynamicStorage<T>>::new_dynamic(3, 3)?;
                for i in 0..3 {
                    for j in 0..3 {
                        *_vecs.get_mut(i, j).unwrap() = if i == j {
                            T::from_f64(1.0)
                        } else {
                            T::default()
                        };
                    }
                }
                self.eigenvectors = Some(_vecs);
            }
            self.info = ComputationInfo::Success;
            return Ok(());
        }

        let scale = norm;
        let a00 = s00 / scale;
        let a10 = m10 / scale;
        let a20 = m20 / scale;
        let a11 = s11 / scale;
        let a21 = m21 / scale;
        let a22 = s22 / scale;

        let c0 = a00 * a11 * a22 + T::from_f64(2.0) * a10 * a20 * a21
            - a00 * a21 * a21
            - a11 * a20 * a20
            - a22 * a10 * a10;
        let c1 = a00 * a11 - a10 * a10 + a00 * a22 - a20 * a20 + a11 * a22 - a21 * a21;
        let _c2 = T::default(); // trace is zero by construction

        let a_over_3 = -c1 / T::from_f64(3.0);
        let half_b = c0 * T::from_f64(0.5);

        let q = a_over_3 * a_over_3 * a_over_3 - half_b * half_b;
        let q_clamped = if q > T::default() { q } else { T::default() };

        let rho = a_over_3.sqrt();
        let theta = (q_clamped.sqrt()).atan2(half_b) / T::from_f64(3.0);
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let sqrt3 = T::from_f64(3.0).sqrt();

        let r0 = -rho * (cos_theta + sqrt3 * sin_theta);
        let r1 = -rho * (cos_theta - sqrt3 * sin_theta);
        let r2 = T::from_f64(2.0) * rho * cos_theta;

        let mut roots = [r0, r1, r2];
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for (i, root) in roots.iter().enumerate() {
            *self.eigenvalues.get_mut(i, 0).unwrap() = *root * scale + shift;
        }

        if compute_eigenvectors {
            // To keep it simple and robust, if compute_eigenvectors is true for 3x3,
            // we'll run the general path.
            // In Eigen, they have a fully direct 3x3, but it's quite complex.
            // Let's just use the general path for now if eigenvectors are needed for N=3.
            return self.compute_general(matrix, compute_eigenvectors);
        }

        self.info = ComputationInfo::Success;
        Ok(())
    }

    fn compute_general(
        &mut self,
        matrix: &Matrix<T, S>,
        compute_eigenvectors: bool,
    ) -> Result<(), String> {
        let n = matrix.rows();
        let tri = Tridiagonalization::new(matrix)?;
        let mut diag = vec![T::default(); n];
        let mut subdiag = vec![T::default(); n - 1];
        let t = tri.matrix_t();
        for i in 0..n {
            diag[i] = *t.get(i, i).unwrap();
            if i < n - 1 {
                subdiag[i] = *t.get(i + 1, i).unwrap();
            }
        }
        if compute_eigenvectors {
            self.eigenvectors = Some(tri.matrix_q());
        }
        self.compute_from_tridiagonal(&mut diag, &mut subdiag, compute_eigenvectors)?;
        self.sort_eigenvalues(n, compute_eigenvectors);
        Ok(())
    }

    fn compute_from_tridiagonal(
        &mut self,
        diag: &mut [T],
        subdiag: &mut [T],
        compute_eigenvectors: bool,
    ) -> Result<(), String> {
        let n = diag.len();
        let mut end = n - 1;
        let mut iter = 0;
        let max_iter = 30 * n;
        let epsilon = T::from_f64(1e-12);

        while end > 0 {
            for i in (0..end).rev() {
                if subdiag[i].abs() <= epsilon * (diag[i].abs() + diag[i + 1].abs()) {
                    subdiag[i] = T::default();
                }
            }
            while end > 0 && subdiag[end - 1] == T::default() {
                end -= 1;
            }
            if end == 0 {
                break;
            }
            iter += 1;
            if iter > max_iter {
                self.info = ComputationInfo::NoConvergence;
                return Ok(());
            }
            let mut start = end - 1;
            while start > 0 && subdiag[start - 1] != T::default() {
                start -= 1;
            }
            self.tridiagonal_qr_step(diag, subdiag, start, end, compute_eigenvectors);
        }
        for (i, val) in diag.iter().enumerate().take(n) {
            *self.eigenvalues.get_mut(i, 0).unwrap() = *val;
        }
        self.info = ComputationInfo::Success;
        Ok(())
    }

    fn tridiagonal_qr_step(
        &mut self,
        diag: &mut [T],
        subdiag: &mut [T],
        start: usize,
        end: usize,
        compute_eigenvectors: bool,
    ) {
        let n_total = self.eigenvalues.rows();
        let d_prev = diag[end - 1];
        let d_last = diag[end];
        let e_last = subdiag[end - 1];
        let e2 = e_last * e_last;
        let td = (d_prev - d_last) * T::from_f64(0.5);
        let sign_td = if td >= T::default() {
            T::from_f64(1.0)
        } else {
            T::from_f64(-1.0)
        };
        let mu = d_last - e2 / (td + sign_td * (td * td + e2).sqrt());

        let mut f = diag[start] - mu;
        let mut g = subdiag[start];

        for k in start..end {
            let r = (f * f + g * g).sqrt();
            let (c, s) = if r == T::default() {
                (T::from_f64(1.0), T::default())
            } else {
                (f / r, g / r)
            };
            if k > start {
                subdiag[k - 1] = r;
            }
            let d1 = diag[k];
            let d2 = diag[k + 1];
            let e1 = subdiag[k];
            let c2 = c * c;
            let s2 = s * s;
            let cs = c * s;
            diag[k] = c2 * d1 + s2 * d2 + T::from_f64(2.0) * cs * e1;
            diag[k + 1] = s2 * d1 + c2 * d2 - T::from_f64(2.0) * cs * e1;
            subdiag[k] = (c2 - s2) * e1 + cs * (d2 - d1);
            if k < end - 1 {
                let e_next = subdiag[k + 1];
                f = subdiag[k];
                g = s * e_next;
                subdiag[k + 1] = c * e_next;
            }
            if compute_eigenvectors {
                let vecs = self.eigenvectors.as_mut().unwrap();
                Self::apply_rotation(vecs, k, k + 1, c, s);
            }
        }
    }

    // Helper: Apply rotation to columns i and j (EigenValues variant)
    // v1' = c*v1 + s*v2
    // v2' = -s*v1 + c*v2
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

                            // v1' = c*v1 + s*v2
                            // v2' = -s*v1 + c*v2

                            let term1_i = _mm256_mul_ps(c_vec, val_i);
                            let term2_i = _mm256_mul_ps(s_vec, val_j);
                            let new_i = _mm256_add_ps(term1_i, term2_i);

                            let term1_j = _mm256_mul_ps(s_vec, val_i); // s*v1
                            let term2_j = _mm256_mul_ps(c_vec, val_j); // c*v2
                            let new_j = _mm256_sub_ps(term2_j, term1_j); // c*v2 - s*v1

                            _mm256_storeu_ps(ptr_i.add(k), new_i);
                            _mm256_storeu_ps(ptr_j.add(k), new_j);
                            k += 8;
                        }

                        // Scalar tail
                        for kk in k..rows {
                            let val_i = *ptr_i.add(kk);
                            let val_j = *ptr_j.add(kk);
                            *ptr_i.add(kk) = c_f32 * val_i + s_f32 * val_j;
                            *ptr_j.add(kk) = c_f32 * val_j - s_f32 * val_i;
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

                            let term1_i = _mm256_mul_pd(c_vec, val_i);
                            let term2_i = _mm256_mul_pd(s_vec, val_j);
                            let new_i = _mm256_add_pd(term1_i, term2_i);

                            let term1_j = _mm256_mul_pd(s_vec, val_i);
                            let term2_j = _mm256_mul_pd(c_vec, val_j);
                            let new_j = _mm256_sub_pd(term2_j, term1_j);

                            _mm256_storeu_pd(ptr_i.add(k), new_i);
                            _mm256_storeu_pd(ptr_j.add(k), new_j);
                            k += 4;
                        }
                        for kk in k..rows {
                            let val_i = *ptr_i.add(kk);
                            let val_j = *ptr_j.add(kk);
                            *ptr_i.add(kk) = c_f64 * val_i + s_f64 * val_j;
                            *ptr_j.add(kk) = c_f64 * val_j - s_f64 * val_i;
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
            *mat.get_mut(k, i).unwrap() = c * val_i + s * val_j;
            *mat.get_mut(k, j).unwrap() = c * val_j - s * val_i;
        }
    }

    fn sort_eigenvalues(&mut self, n: usize, _compute_eigenvectors: bool) {
        if self.info != ComputationInfo::Success {
            return;
        }
        for i in 0..n {
            let mut min_idx = i;
            for j in i + 1..n {
                let val_j = *self.eigenvalues.get(j, 0).unwrap();
                let val_min = *self.eigenvalues.get(min_idx, 0).unwrap();
                if val_j < val_min {
                    min_idx = j;
                }
            }
            if min_idx != i {
                let temp = *self.eigenvalues.get(i, 0).unwrap();
                *self.eigenvalues.get_mut(i, 0).unwrap() =
                    *self.eigenvalues.get(min_idx, 0).unwrap();
                *self.eigenvalues.get_mut(min_idx, 0).unwrap() = temp;
                if let Some(ref mut vecs) = self.eigenvectors {
                    for row in 0..n {
                        let v1 = *vecs.get(row, i).unwrap();
                        let v2 = *vecs.get(row, min_idx).unwrap();
                        *vecs.get_mut(row, i).unwrap() = v2;
                        *vecs.get_mut(row, min_idx).unwrap() = v1;
                    }
                }
            }
        }
    }

    pub fn eigenvalues(&self) -> &Matrix<T, DynamicStorage<T>> {
        &self.eigenvalues
    }
    pub fn eigenvectors(&self) -> Option<&Matrix<T, DynamicStorage<T>>> {
        self.eigenvectors.as_ref()
    }
    pub fn info(&self) -> ComputationInfo {
        self.info
    }
}
