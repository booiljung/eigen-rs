//! Eigenvalue and eigenvector solver for general square real matrices.

use crate::core::complex::Complex;
use crate::core::decompositions::RealSchur;
use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::{DynamicStorage, Storage};

/// Eigenvalue and eigenvector solver for general square real matrices.
/// Eigenvalue and eigenvector solver for general square real matrices.
pub struct EigenSolver<T: Scalar<Real = T> + PartialOrd, S: Storage<T>> {
    eigenvalues: Vec<Complex<T>>,
    eigenvectors: Option<Matrix<Complex<T>, DynamicStorage<Complex<T>>>>,
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar<Real = T> + PartialOrd, S: Storage<T>> EigenSolver<T, S> {
    /// Computes the eigenvalues and (optionally) eigenvectors of the given square real matrix.
    pub fn new(matrix: &Matrix<T, S>, compute_eigenvectors: bool) -> Result<Self, String> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err("EigenSolver requires a square matrix".to_string());
        }

        // 1. Compute Real Schur decomposition: A = Q T Q^T
        let schur = RealSchur::new(matrix)?;
        let t = schur.matrix_t();
        let q = schur.matrix_q();

        // 2. Extract eigenvalues from quasi-upper triangular matrix T
        let mut eigenvalues = Vec::with_capacity(n);
        let mut i = 0;
        let eps = T::epsilon();

        while i < n {
            if i + 1 < n {
                let h_ip1_i = *t.get(i + 1, i).unwrap();
                if h_ip1_i.abs()
                    > eps * (t.get(i, i).unwrap().abs() + t.get(i + 1, i + 1).unwrap().abs())
                {
                    // 2x2 block found: complex conjugate pair
                    let a = *t.get(i, i).unwrap();
                    let b = *t.get(i, i + 1).unwrap();
                    let c = *t.get(i + 1, i).unwrap();
                    let d = *t.get(i + 1, i + 1).unwrap();

                    let tr = a + d;
                    let det = a * d - b * c;
                    let disc = tr * tr - T::from_f64(4.0) * det;

                    if disc < T::default() {
                        let re = tr / T::from_f64(2.0);
                        let im = (-disc).sqrt() / T::from_f64(2.0);
                        eigenvalues.push(Complex::new(re, im));
                        eigenvalues.push(Complex::new(re, -im));
                    } else {
                        // Actually real (shouldn't happen with good Schur but for safety)
                        let re1 = (tr + disc.sqrt()) / T::from_f64(2.0);
                        let re2 = (tr - disc.sqrt()) / T::from_f64(2.0);
                        eigenvalues.push(Complex::new(re1, T::default()));
                        eigenvalues.push(Complex::new(re2, T::default()));
                    }
                    i += 2;
                    continue;
                }
            }
            // 1x1 block: real eigenvalue
            eigenvalues.push(Complex::new(*t.get(i, i).unwrap(), T::default()));
            i += 1;
        }

        let mut eigenvectors = None;
        if compute_eigenvectors {
            // Solve (T - lambda I) y = 0 by back-substitution
            // Then x = Q y
            let mut vecs = Matrix::<Complex<T>, DynamicStorage<Complex<T>>>::new_dynamic(n, n)?;

            // Temporary storage for real/imag parts of y
            let mut y_re = vec![T::default(); n];
            let mut y_im = vec![T::default(); n];

            let mut k = 0;
            while k < n {
                let lambda = eigenvalues[k];

                if lambda.im == T::default() {
                    // Real eigenvalue
                    y_re.fill(T::default());
                    y_re[k] = T::from_f64(1.0);

                    for i in (0..k).rev() {
                        let mut sum = T::default();
                        for (j, y_re_val) in y_re.iter().enumerate().take(k + 1).skip(i + 1) {
                            sum += *t.get(i, j).unwrap() * *y_re_val;
                        }
                        let diag = *t.get(i, i).unwrap() - lambda.re;
                        if diag.abs() > eps {
                            y_re[i] = -sum / diag;
                        } else {
                            y_re[i] = -sum / (diag + eps);
                        }
                    }

                    // x = Q * y_re
                    for i in 0..n {
                        let mut val = T::default();
                        for (j, y_re_val) in y_re.iter().enumerate().take(k + 1) {
                            val += *q.get(i, j).unwrap() * *y_re_val;
                        }
                        *vecs.get_mut(i, k).unwrap() = Complex::new(val, T::default());
                    }
                    k += 1;
                } else {
                    // Complex conjugate pair
                    y_re.fill(T::default());
                    y_im.fill(T::default());

                    let t21 = *t.get(k + 1, k).unwrap();
                    let t22 = *t.get(k + 1, k + 1).unwrap();

                    let y2 = Complex::new(T::from_f64(1.0), T::default());
                    let y1 = (Complex::new(lambda.re - t22, lambda.im))
                        / Complex::new(t21, T::default());

                    y_re[k] = y1.re;
                    y_im[k] = y1.im;
                    y_re[k + 1] = y2.re;
                    y_im[k + 1] = y2.im;

                    for i in (0..k).rev() {
                        let mut sum_re = T::default();
                        let mut sum_im = T::default();
                        for j in i + 1..k + 2 {
                            sum_re += *t.get(i, j).unwrap() * y_re[j];
                            sum_im += *t.get(i, j).unwrap() * y_im[j];
                        }
                        let diag = *t.get(i, i).unwrap() - lambda.re;
                        let den = diag * diag + lambda.im * lambda.im;
                        y_re[i] = (-sum_re * diag - sum_im * lambda.im) / den;
                        y_im[i] = (-sum_im * diag + sum_re * lambda.im) / den;
                    }

                    for i in 0..n {
                        let mut val_re = T::default();
                        let mut val_im = T::default();
                        for j in 0..k + 2 {
                            val_re += *q.get(i, j).unwrap() * y_re[j];
                            val_im += *q.get(i, j).unwrap() * y_im[j];
                        }
                        *vecs.get_mut(i, k).unwrap() = Complex::new(val_re, val_im);
                        *vecs.get_mut(i, k + 1).unwrap() = Complex::new(val_re, -val_im);
                    }
                    k += 2;
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

    pub fn eigenvalues(&self) -> &[Complex<T>] {
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
    fn test_eigen_solver_eigenvectors() -> Result<(), String> {
        let n = 3;
        let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;
        let data = [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0];
        for i in 0..n {
            for j in 0..n {
                *a.get_mut(i, j).unwrap() = data[i * n + j];
            }
        }

        let solver = EigenSolver::new(&a, true)?;
        let evs = solver.eigenvalues();
        let evecs = solver.eigenvectors().unwrap();

        for k in 0..n {
            let lambda = evs[k];
            // Verify A * v = lambda * v
            for i in 0..n {
                let mut av = Complex::<f64>::new(0.0, 0.0);
                for j in 0..n {
                    let a_ij = *a.get(i, j).unwrap();
                    let v_jk = *evecs.get(j, k).unwrap();
                    av += Complex::new(a_ij, 0.0) * v_jk;
                }
                let lv = lambda * (*evecs.get(i, k).unwrap());
                assert!((av.re - lv.re).abs() < 1e-9);
                assert!((av.im - lv.im).abs() < 1e-9);
            }
        }
        Ok(())
    }

    #[test]
    fn test_eigen_solver_real() -> Result<(), String> {
        let n = 3;
        let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;
        // Matrix with eigenvalues 1, 2, 3
        let data = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0];
        for i in 0..n {
            for j in 0..n {
                *a.get_mut(i, j).unwrap() = data[i * n + j];
            }
        }

        let solver = EigenSolver::new(&a, false)?;
        let evs = solver.eigenvalues();

        let mut vals: Vec<f64> = evs.iter().map(|c| c.re).collect();
        vals.sort_by(|a, b| {
            if let Some(ord) = a.partial_cmp(b) {
                ord
            } else {
                let a_nan = *a != *a;
                let b_nan = *b != *b;
                if a_nan && b_nan {
                    std::cmp::Ordering::Equal
                } else if a_nan {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            }
        });

        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 2.0).abs() < 1e-10);
        assert!((vals[2] - 3.0).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_eigen_solver_complex() -> Result<(), String> {
        let n = 2;
        let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;
        // Rotation matrix: eigenvalues +/- i
        let data = [0.0, -1.0, 1.0, 0.0];
        for i in 0..n {
            for j in 0..n {
                *a.get_mut(i, j).unwrap() = data[i * n + j];
            }
        }

        let solver = EigenSolver::new(&a, false)?;
        let evs = solver.eigenvalues();

        assert_eq!(evs.len(), 2);
        // Expect (0, 1) and (0, -1)
        let has_pos_i = evs
            .iter()
            .any(|c| (c.re).abs() < 1e-10 && (c.im - 1.0).abs() < 1e-10);
        let has_neg_i = evs
            .iter()
            .any(|c| (c.re).abs() < 1e-10 && (c.im + 1.0).abs() < 1e-10);

        assert!(has_pos_i);
        assert!(has_neg_i);

        Ok(())
    }
}
