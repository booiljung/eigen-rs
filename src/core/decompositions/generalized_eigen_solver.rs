//! Generalized Eigenvalue Solver for the problem A*x = lambda*B*x.
//! Based on the QZ algorithm.

use crate::core::complex::Complex;
use crate::core::decompositions::GeneralizedHessenbergTriangular;
use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::{DynamicStorage, Storage};

/// Generalized Eigensolver for square complex matrices.
pub struct GeneralizedEigenSolver<T: Scalar, S: Storage<Complex<T>>> {
    eigenvalues: Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
    eigenvectors: Option<Matrix<Complex<T>, DynamicStorage<Complex<T>>>>,
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar, S: Storage<Complex<T>>> GeneralizedEigenSolver<T, S> {
    /// Computes the generalized eigenvalues and (optionally) eigenvectors of (A, B).
    pub fn new(
        a: &Matrix<Complex<T>, S>,
        b: &Matrix<Complex<T>, S>,
        compute_eigenvectors: bool,
    ) -> Result<Self, String> {
        let n = a.rows();
        if n != a.cols() || n != b.rows() || n != b.cols() {
            return Err(
                "Generalized eigensolver requires square matrices of the same size".to_string(),
            );
        }

        // 1. GHT Reduction: (A, B) -> (H, R)
        let ght = GeneralizedHessenbergTriangular::new(a, b)?;
        let mut h = ght.matrix_h().clone();
        let mut r = ght.matrix_r().clone();
        let mut q = ght.matrix_q().clone();
        let mut z = ght.matrix_z().clone();

        // 2. QZ Iteration to reach Generalized Schur form (S, T)
        Self::compute_qz_inplace(&mut h, &mut r, &mut q, &mut z)?;

        // 3. Eigenvalues: lambda_i = S_ii / T_ii
        let mut eigenvalues = Matrix::<Complex<T>, DynamicStorage<Complex<T>>>::new_dynamic(n, 1)?;
        for i in 0..n {
            let s_ii = *h.get(i, i).unwrap();
            let t_ii = *r.get(i, i).unwrap();
            if t_ii.norm_sq() < T::epsilon() {
                // Infinite eigenvalue or poorly conditioned
                // In a robust solver, we'd return (alpha, beta) pairs.
                // For now, return a large value or handle as needed.
                *eigenvalues.get_mut(i, 0).unwrap() =
                    Complex::new(T::from_f64(1e100), T::default()); // Placeholder
            } else {
                *eigenvalues.get_mut(i, 0).unwrap() = s_ii / t_ii;
            }
        }

        let mut eigenvectors = None;
        if compute_eigenvectors {
            let mut vecs = Matrix::<Complex<T>, DynamicStorage<Complex<T>>>::new_dynamic(n, n)?;
            let eps = T::epsilon();

            for k in 0..n {
                let lambda = *eigenvalues.get(k, 0).unwrap();
                let mut y = vec![Complex::default(); n];
                y[k] = Complex::from_f64(1.0);

                for i in (0..k).rev() {
                    let mut sum = Complex::default();
                    for (j, y_val) in y.iter().enumerate().take(k + 1).skip(i + 1) {
                        sum += (*h.get(i, j).unwrap() - lambda * (*r.get(i, j).unwrap())) * *y_val;
                    }
                    let diag = *h.get(i, i).unwrap() - lambda * (*r.get(i, i).unwrap());
                    if diag.norm_sq() > eps * eps {
                        y[i] = -sum / diag;
                    } else {
                        y[i] = -sum / Complex::new(eps, T::default());
                    }
                }

                // x = Z * y
                for i in 0..n {
                    let mut val = Complex::default();
                    for (j, y_val) in y.iter().enumerate().take(k + 1) {
                        val += (*z.get(i, j).unwrap()) * *y_val;
                    }
                    *vecs.get_mut(i, k).unwrap() = val;
                }
            }
            eigenvectors = Some(vecs);
        }

        Ok(Self {
            eigenvalues,
            eigenvectors,
            _phantom: std::marker::PhantomData,
        })
    }

    fn compute_qz_inplace(
        h: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
        r: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
        q: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
        z: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
    ) -> Result<(), String> {
        let n = h.rows();
        let mut high = n - 1;
        let max_iter = 30 * n;
        let mut iter = 0;

        let epsilon = T::epsilon();

        while high > 0 {
            if iter > max_iter {
                return Err("QZ algorithm failed to converge".to_string());
            }

            // 1. Deflation check
            let mut low = high;
            while low > 0 {
                if h.get(low, low - 1).unwrap().norm()
                    <= epsilon
                        * (h.get(low - 1, low - 1).unwrap().norm()
                            + h.get(low, low).unwrap().norm())
                {
                    *h.get_mut(low, low - 1).unwrap() = Complex::default();
                    break;
                }
                low -= 1;
            }

            if low == high {
                high -= 1;
                iter = 0;
                continue;
            }

            // 2. QZ step on block [low..high+1]
            Self::qz_step(h, r, q, z, low, high);
            iter += 1;
        }

        Ok(())
    }

    fn qz_step(
        h: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
        r: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
        q: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
        z: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
        start: usize,
        end: usize,
    ) {
        let n = h.rows();

        // 1. Shift: Wilkinson shift for (H * R^-1)
        // Shift lambda is an eigenvalue of the bottom 2x2 block of (H * R^-1)
        // i.e., eigenvalues of H[end-1:end+1, end-1:end+1] * R[end-1:end+1, end-1:end+1]^-1
        // Let's compute eigenvalues of H22 * inv(R22)
        let h22_a = *h.get(end - 1, end - 1).unwrap();
        let h22_b = *h.get(end - 1, end).unwrap();
        let h22_c = *h.get(end, end - 1).unwrap();
        let h22_d = *h.get(end, end).unwrap();

        let r22_a = *r.get(end - 1, end - 1).unwrap();
        let r22_b = *r.get(end - 1, end).unwrap();
        let r22_d = *r.get(end, end).unwrap();

        // inv(R22) = (1/det) * [[r22_d, -r22_b], [0, r22_a]]
        let det_r = r22_a * r22_d;
        let m_a = (h22_a * r22_d) / det_r;
        let m_b = (-h22_a * r22_b + h22_b * r22_a) / det_r;
        let m_c = (h22_c * r22_d) / det_r;
        let m_d = (-h22_c * r22_b + h22_d * r22_a) / det_r;

        // Eigenvalues of [[m_a, m_b], [m_c, m_d]]
        let tr = m_a + m_d;
        let det = m_a * m_d - m_b * m_c;
        let disc = tr * tr - Complex::from_f64(4.0) * det;
        let sqrt_disc = disc.sqrt();
        let l1 = (tr + sqrt_disc) / Complex::from_f64(2.0);
        let l2 = (tr - sqrt_disc) / Complex::from_f64(2.0);

        // Wilkinson shift: pick the eigenvalue closer to m_d
        let shift = if (l1 - m_d).norm_sq() < (l2 - m_d).norm_sq() {
            l1
        } else {
            l2
        };

        // 2. Initial rotation Q_1 (Left) to seed the bulge
        // We want Q_1 * (H * inv(R) - shift * I) * e_1 = [*, 0, ...]^T
        // u = (H - shift * R) * e_1 = [h11 - shift * r11, h21, 0, ...]^T
        let u1 = *h.get(start, start).unwrap() - shift * (*r.get(start, start).unwrap());
        let u2 = *h.get(start + 1, start).unwrap();

        let (cq1, sq1) = Self::givens_rotation(u1, u2);

        // Apply Q1 to rows start, start+1
        Self::apply_givens_left(h, start, start + 1, start, n, cq1, sq1);
        Self::apply_givens_left(r, start, start + 1, start, n, cq1, sq1);
        Self::apply_givens_left(q, start, start + 1, 0, n, cq1, sq1);

        // 3. Bulge chasing
        for k in start..end {
            // Restore triangularity of R: zero R[k+1, k] using column rotation Z_k (Right)
            if k < end {
                let r_kp1_k = *r.get(k + 1, k).unwrap();
                if r_kp1_k != Complex::default() {
                    let (cz, sz) = Self::givens_rotation(*r.get(k + 1, k + 1).unwrap(), -r_kp1_k);
                    Self::apply_givens_right(h, 0, n, k, k + 1, cz, sz);
                    Self::apply_givens_right(r, 0, k + 2, k, k + 1, cz, sz);
                    *r.get_mut(k + 1, k).unwrap() = Complex::default();
                    Self::apply_givens_right(z, 0, n, k, k + 1, cz, sz);
                }
            }

            // Restore Hessenberg form of H: zero H[k+2, k] using row rotation Q_k+1 (Left)
            // Wait, if k+2 <= end
            if k + 2 <= end {
                let h_kp2_k = *h.get(k + 2, k).unwrap();
                if h_kp2_k != Complex::default() {
                    let (cq, sq) = Self::givens_rotation(*h.get(k + 1, k).unwrap(), h_kp2_k);
                    Self::apply_givens_left(h, k + 1, k + 2, k, n, cq, sq);
                    *h.get_mut(k + 2, k).unwrap() = Complex::default();

                    Self::apply_givens_left(r, k + 1, k + 2, k + 1, n, cq, sq);
                    Self::apply_givens_left(q, k + 1, k + 2, 0, n, cq, sq);
                }
            }
        }
    }

    // Reuse Givens helpers from GHT (I should probably move them to a shared place eventually)
    fn givens_rotation(a: Complex<T>, b: Complex<T>) -> (T, Complex<T>) {
        if b == Complex::default() {
            return (T::from_f64(1.0), Complex::default());
        }
        let norm = (a.norm_sq() + b.norm_sq()).sqrt();
        let c = a.norm() / norm;
        let s = (a / Complex::new(a.norm(), T::default())).conj()
            * (b / Complex::new(norm, T::default()));
        (c, s)
    }

    fn apply_givens_left(
        m: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
        i: usize,
        j: usize,
        col_start: usize,
        col_end: usize,
        c: T,
        s: Complex<T>,
    ) {
        let cc = Complex::new(c, T::default());
        for k in col_start..col_end {
            let v1 = *m.get(i, k).unwrap();
            let v2 = *m.get(j, k).unwrap();
            *m.get_mut(i, k).unwrap() = cc * v1 + s.conj() * v2;
            *m.get_mut(j, k).unwrap() = -s * v1 + cc * v2;
        }
    }

    fn apply_givens_right(
        m: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
        row_start: usize,
        row_end: usize,
        i: usize,
        j: usize,
        c: T,
        s: Complex<T>,
    ) {
        let cc = Complex::new(c, T::default());
        for k in row_start..row_end {
            let v1 = *m.get(k, i).unwrap();
            let v2 = *m.get(k, j).unwrap();
            *m.get_mut(k, i).unwrap() = cc * v1 + s * v2;
            *m.get_mut(k, j).unwrap() = -s.conj() * v1 + cc * v2;
        }
    }

    pub fn eigenvalues(&self) -> &Matrix<Complex<T>, DynamicStorage<Complex<T>>> {
        &self.eigenvalues
    }

    pub fn eigenvectors(&self) -> Option<&Matrix<Complex<T>, DynamicStorage<Complex<T>>>> {
        self.eigenvectors.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::storage::DynamicStorage;

    #[test]
    fn test_generalized_eigen_solver_basic() -> Result<(), String> {
        // ... (keep existing basic test)
        let n = 2;
        let mut a = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(n, n)?;
        let mut b = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(n, n)?;

        *a.get_mut(0, 0).unwrap() = Complex::new(2.0, 0.0);
        *a.get_mut(0, 1).unwrap() = Complex::new(1.0, 0.0);
        *a.get_mut(1, 1).unwrap() = Complex::new(4.0, 0.0);

        *b.get_mut(0, 0).unwrap() = Complex::new(1.0, 0.0);
        *b.get_mut(1, 1).unwrap() = Complex::new(1.0, 0.0);

        let solver = GeneralizedEigenSolver::new(&a, &b, false)?;
        let evals = solver.eigenvalues();

        let val1 = *evals.get(0, 0).unwrap();
        let val2 = *evals.get(1, 0).unwrap();

        let mut vals = [val1.re, val2.re];
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((vals[0] - 2.0).abs() < 1e-10);
        assert!((vals[1] - 4.0).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_generalized_eigen_solver_eigenvectors() -> Result<(), String> {
        let n = 3;
        let mut a = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(n, n)?;
        let mut b = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(n, n)?;

        let data_a = [
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(0.0, 0.0),
            Complex::new(3.0, 2.0),
            Complex::new(1.0, -1.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.0, 2.0),
        ];
        let data_b = [
            Complex::new(5.0, 0.0),
            Complex::new(1.0, 1.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.0, 2.0),
            Complex::new(2.0, 1.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(3.0, -1.0),
        ];

        for i in 0..n {
            for j in 0..n {
                *a.get_mut(i, j).unwrap() = data_a[i * n + j];
                *b.get_mut(i, j).unwrap() = data_b[i * n + j];
            }
        }

        let solver = GeneralizedEigenSolver::new(&a, &b, true)?;
        let evals = solver.eigenvalues();
        let evecs = solver.eigenvectors().unwrap();

        for k in 0..n {
            let lambda = *evals.get(k, 0).unwrap();
            // Verify A * x = lambda * B * x
            for i in 0..n {
                let mut ax = Complex::default();
                let mut bx = Complex::default();
                for j in 0..n {
                    ax += (*a.get(i, j).unwrap()) * (*evecs.get(j, k).unwrap());
                    bx += (*b.get(i, j).unwrap()) * (*evecs.get(j, k).unwrap());
                }
                let lbx = lambda * bx;
                assert!(
                    (ax.re - lbx.re).abs() < 1e-9,
                    "A*x = lambda*B*x failure at row {} for eigenvalue {}",
                    i,
                    lambda
                );
                assert!(
                    (ax.im - lbx.im).abs() < 1e-9,
                    "A*x = lambda*B*x failure at row {} for eigenvalue {}",
                    i,
                    lambda
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_generalized_eigen_solver_random() -> Result<(), String> {
        let n = 3;
        let mut a = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(n, n)?;
        let mut b = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(n, n)?;

        let data_a = [
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(0.5, 0.5),
            Complex::new(3.0, 2.0),
            Complex::new(1.0, -1.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 2.0),
        ];
        let data_b = [
            Complex::new(5.0, 0.0),
            Complex::new(1.0, 1.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, -1.0),
            Complex::new(4.0, 2.0),
            Complex::new(2.0, 1.0),
            Complex::new(3.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(5.0, -1.0),
        ];

        for i in 0..n {
            for j in 0..n {
                *a.get_mut(i, j).unwrap() = data_a[i * n + j];
                *b.get_mut(i, j).unwrap() = data_b[i * n + j];
            }
        }

        let solver = GeneralizedEigenSolver::new(&a, &b, false)?;
        let evals = solver.eigenvalues();

        // Since we don't have eigenvectors yet, verify by checking det(A - lambda B) = 0
        // We'll compute the eigenvalues lambda_i and for each, check if (A - lambda_i B) is singular.
        // A simpler way: since n is 3, we can just check if any lambda matches Eigen's known result if we had one.
        // Or just check that they are finite and stable.

        for k in 0..n {
            let lambda = *evals.get(k, 0).unwrap();
            // Check det(A - lambda B) using partial pivoting LU
            let mut char_mat =
                Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(n, n)?;
            for i in 0..n {
                for j in 0..n {
                    *char_mat.get_mut(i, j).unwrap() =
                        *a.get(i, j).unwrap() - lambda * (*b.get(i, j).unwrap());
                }
            }

            // Singular check: det should be small
            let lu = char_mat.partial_piv_lu();
            assert!(lu.is_ok(), "LU failed for lambda {}", lambda);

            let det = lu.unwrap().determinant();
            assert!(
                det.norm() < 1e-7,
                "det(A - lambda B) = {} is too large for lambda {}",
                det,
                lambda
            );
        }

        Ok(())
    }
}
