//! Eigenvalue and eigenvector decomposition of a general complex matrix.
//! A = V * D * V^-1

use crate::core::complex::Complex;
use crate::core::decompositions::ComplexSchur;
use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::{DynamicStorage, Storage};

/// Eigensolver for general complex matrices.
pub struct ComplexEigenSolver<T: Scalar, S: Storage<Complex<T>>> {
    eigenvalues: Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
    eigenvectors: Option<Matrix<Complex<T>, DynamicStorage<Complex<T>>>>,
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar, S: Storage<Complex<T>>> ComplexEigenSolver<T, S> {
    /// Computes the eigenvalues and (optionally) eigenvectors of a square complex matrix.
    pub fn new(matrix: &Matrix<Complex<T>, S>, compute_eigenvectors: bool) -> Result<Self, String> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err("Complex eigensolver requires a square matrix".to_string());
        }

        let schur = ComplexSchur::new(matrix)?;
        let t = schur.matrix_t();
        let u = schur.matrix_u();

        // 1. Eigenvalues are the diagonal elements of T
        let mut eigenvalues = Matrix::<Complex<T>, DynamicStorage<Complex<T>>>::new_dynamic(n, 1)?;
        for i in 0..n {
            *eigenvalues.get_mut(i, 0).unwrap() = *t.get(i, i).unwrap();
        }

        let mut eigenvectors = None;
        if compute_eigenvectors {
            // Solve (T - lambda I) y = 0 for each eigenvalue
            // Since T is upper triangular, we use back-substitution.
            let mut vecs_y = Matrix::<Complex<T>, DynamicStorage<Complex<T>>>::new_dynamic(n, n)?;

            for j in 0..n {
                let lambda = *eigenvalues.get(j, 0).unwrap();

                // Solve (T[0:j+1, 0:j+1] - lambda I) y_j = 0
                // We set y_j[j] = 1.0 and solve for y_j[0..j]
                *vecs_y.get_mut(j, j).unwrap() = Complex::from_f64(1.0);

                for i in (0..j).rev() {
                    let mut sum = Complex::default();
                    for k in i + 1..=j {
                        sum += (*t.get(i, k).unwrap()) * (*vecs_y.get(k, j).unwrap());
                    }

                    let denom = lambda - (*t.get(i, i).unwrap());
                    if denom.norm_sq() < T::epsilon() {
                        // Singular or nearly singular. In a real solver, we'd use a small perturbation.
                        // For now, avoid division by zero.
                        *vecs_y.get_mut(i, j).unwrap() = Complex::default();
                    } else {
                        *vecs_y.get_mut(i, j).unwrap() = sum / denom;
                    }
                }

                // Normalize eigenvector y_j
                let mut norm_sq = T::default();
                for i in 0..=j {
                    norm_sq += vecs_y.get(i, j).unwrap().norm_sq();
                }
                let norm = norm_sq.sqrt();
                if norm != T::default() {
                    for i in 0..=j {
                        *vecs_y.get_mut(i, j).unwrap() /= Complex::new(norm, T::default());
                    }
                }
            }

            // Transform back: X = U * Y
            let mut vecs_x = Matrix::<Complex<T>, DynamicStorage<Complex<T>>>::new_dynamic(n, n)?;
            for i in 0..n {
                for j in 0..n {
                    let mut sum = Complex::default();
                    for k in 0..n {
                        sum += (*u.get(i, k).unwrap()) * (*vecs_y.get(k, j).unwrap());
                    }
                    *vecs_x.get_mut(i, j).unwrap() = sum;
                }
            }
            eigenvectors = Some(vecs_x);
        }

        Ok(Self {
            eigenvalues,
            eigenvectors,
            _phantom: std::marker::PhantomData,
        })
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
    fn test_complex_eigen_solver_basic() -> Result<(), String> {
        let n = 3;
        let mut a = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(n, n)?;
        let data = [
            Complex::new(1.0, 2.0),
            Complex::new(2.0, -1.0),
            Complex::new(0.0, 1.0),
            Complex::new(1.0, 1.0),
            Complex::new(4.0, 0.0),
            Complex::new(2.0, 3.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 2.0),
            Complex::new(5.0, -2.0),
        ];
        for i in 0..n {
            for j in 0..n {
                *a.get_mut(i, j).unwrap() = data[i * n + j];
            }
        }

        let solver = ComplexEigenSolver::new(&a, true)?;
        let evals = solver.eigenvalues();
        let evecs = solver.eigenvectors().unwrap();

        // Verify A * v = lambda * v
        for j in 0..n {
            let lambda = *evals.get(j, 0).unwrap();
            for i in 0..n {
                let mut av = Complex::default();
                for k in 0..n {
                    av += (*a.get(i, k).unwrap()) * (*evecs.get(k, j).unwrap());
                }
                let lv = lambda * (*evecs.get(i, j).unwrap());
                assert!(
                    (av.re - lv.re).abs() < 1e-8,
                    "Failed at col {} row {}: av={}, lv={}",
                    j,
                    i,
                    av,
                    lv
                );
                assert!(
                    (av.im - lv.im).abs() < 1e-8,
                    "Failed at col {} row {}: av={}, lv={}",
                    j,
                    i,
                    av,
                    lv
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_complex_eigen_solver_identity() -> Result<(), String> {
        let n = 3;
        let mut a = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(n, n)?;
        for i in 0..n {
            *a.get_mut(i, i).unwrap() = Complex::new(1.0, 0.0);
        }

        let solver = ComplexEigenSolver::new(&a, true)?;
        let evals = solver.eigenvalues();

        for j in 0..n {
            let val = evals.get(j, 0).unwrap();
            assert!((val.re - 1.0).abs() < 1e-10);
            assert!(val.im.abs() < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_complex_eigen_solver_repeated() -> Result<(), String> {
        let n = 2;
        let mut a = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(n, n)?;
        // Matrix with repeated eigenvalue 2.0
        // [2 1]
        // [0 2]
        *a.get_mut(0, 0).unwrap() = Complex::new(2.0, 0.0);
        *a.get_mut(0, 1).unwrap() = Complex::new(1.0, 0.0);
        *a.get_mut(1, 1).unwrap() = Complex::new(2.0, 0.0);

        let solver = ComplexEigenSolver::new(&a, true)?;
        let evals = solver.eigenvalues();

        for j in 0..n {
            let val = evals.get(j, 0).unwrap();
            assert!((val.re - 2.0).abs() < 1e-10);
        }

        // Note: For deficient matrices, we can't always recover a full set of eigenvectors.
        // Our current back-substitution will return [1, 0] for both.

        Ok(())
    }
}
