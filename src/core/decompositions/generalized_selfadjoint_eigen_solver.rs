//! Generalized Self-Adjoint Eigenvalue Solver.
//! Solves $A x = \lambda B x$ where $A$ is self-adjoint and $B$ is positive definite.
//!
//! Based on Cholesky decomposition of $B = L L^T$.
//! The problem is transformed to a standard symmetric eigenvalue problem:
//! $C y = \lambda y$ where $C = L^{-1} A L^{-T}$ and $x = L^{-T} y$.

use crate::core::decompositions::{LLT, SelfAdjointEigenSolver};
use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::DynamicStorage;
use crate::core::storage::Storage;

/// Generalized Self-Adjoint Eigensolver.
pub struct GeneralizedSelfAdjointEigenSolver<T: Scalar, S: Storage<T>> {
    eigenvalues: Matrix<T, DynamicStorage<T>>,
    eigenvectors: Option<Matrix<T, DynamicStorage<T>>>,
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar + num_traits::One + 'static, S: Storage<T> + 'static> GeneralizedSelfAdjointEigenSolver<T, S> {
    /// Computes the generalized eigenvalues and (optionally) eigenvectors of (A, B).
    /// A must be self-adjoint, B must be positive definite.
    pub fn new(
        a: &Matrix<T, S>,
        b: &Matrix<T, S>,
        compute_eigenvectors: bool,
    ) -> Result<Self, String> {
        let n = a.rows();
        if n != a.cols() || n != b.rows() || n != b.cols() {
            return Err("Generalized eigensolver requires square matrices".to_string());
        }

        // 1. Cholesky decomposition of B: B = L L^T
        let llt = LLT::new(b)?;
        let l = llt.matrix_l();

        // 2. Compute C = L^{-1} A L^{-T}
        // We do this by solving linear systems.
        // First compute Y = L^{-1} A => L Y = A => Forward substitution
        // Then C = Y L^{-T} => C L^T = Y => C = (Y^T L^{-1})^T => (L Y^T_trans)^T?
        // Easiest semantic way:
        // C = L.solve(A).solve_transpose(L) ? No, solve solves Ax=b.
        
        // Let's implement manually for efficiency.
        // Y = L^{-1} A
        let mut y = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n)?;
        y.assign(a)?;
        // Solve L Y = A (in-place in y, which holds A)
        llt.solve_inplace_l(&mut y)?;

        // Now C = Y L^{-T}.
        // C^T = L^{-1} Y^T. Since C is symmetric, C = C^T = L^{-1} Y^T.
        // So we can solve L X = Y^T.
        let mut c = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n)?;
        // Transpose Y into C
        for i in 0..n {
            for j in 0..n {
                *c.get_mut(i, j).unwrap() = *y.get(j, i).unwrap();
            }
        }
        // Solve L C = Y^T (in-place in c)
        llt.solve_inplace_l(&mut c)?;
        
        // Note: C should be symmetric.
        // The result of L^{-1} A L^{-T} is mathematically symmetric if A is symmetric.
        // However, numerical errors might make it slightly non-symmetric.
        // SelfAdjointEigenSolver only uses the lower/upper triangular part anyway.

        // 3. Solve standard eigenvalue problem for C
        let eigen = SelfAdjointEigenSolver::new(&c, compute_eigenvectors)?;
        let eigenvalues = eigen.eigenvalues().clone();

        let mut eigenvectors = None;
        if compute_eigenvectors {
            // Recover eigenvectors x = L^{-T} y
            // eigen_vecs returns y.
            // We need to solve L^T x = y.
            
            let y_vecs = eigen.eigenvectors().unwrap();
            let mut x_vecs = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n)?;
            x_vecs.assign(y_vecs)?; // Copy y
            
            // Solve L^T x = y (in-place in x_vecs)
            llt.solve_inplace_lt(&mut x_vecs)?;
            
            eigenvectors = Some(x_vecs);
        }

        Ok(Self {
            eigenvalues,
            eigenvectors,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn eigenvalues(&self) -> &Matrix<T, DynamicStorage<T>> {
        &self.eigenvalues
    }

    pub fn eigenvectors(&self) -> Option<&Matrix<T, DynamicStorage<T>>> {
        self.eigenvectors.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::storage::DynamicStorage;

    #[test]
    fn test_generalized_selfadjoint_eigen_basic() -> Result<(), String> {
        let n = 2;
        let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;
        let mut b = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;

        // A = [[2, 1], [1, 4]]
        *a.get_mut(0, 0).unwrap() = 2.0;
        *a.get_mut(0, 1).unwrap() = 1.0;
        *a.get_mut(1, 0).unwrap() = 1.0;
        *a.get_mut(1, 1).unwrap() = 4.0;

        // B = I
        *b.get_mut(0, 0).unwrap() = 1.0;
        *b.get_mut(1, 1).unwrap() = 1.0;

        // Eigenvalues of A with B=I should be standard eigenvalues of A
        // lambda^2 - 6 lambda + 7 = 0 => lambda = (6 +/- sqrt(36-28))/2 = 3 +/- sqrt(2)
        // 3 + 1.414 = 4.414, 3 - 1.414 = 1.586
        
        let solver = GeneralizedSelfAdjointEigenSolver::new(&a, &b, false)?;
        let evals = solver.eigenvalues();
        
        let v1 = *evals.get(0, 0).unwrap();
        let v2 = *evals.get(1, 0).unwrap();
        
        // Sorted? SelfAdjointEigenSolver sorts them.
        assert!((v1 - (3.0 - 2.0f64.sqrt())).abs() < 1e-6);
        assert!((v2 - (3.0 + 2.0f64.sqrt())).abs() < 1e-6);
        
        Ok(())
    }

    #[test]
    fn test_generalized_selfadjoint_eigen_general_b() -> Result<(), String> {
        let n = 2;
        let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;
        let mut b = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(n, n)?;

        // A = [[10, 0], [0, 20]]
        *a.get_mut(0, 0).unwrap() = 10.0;
        *a.get_mut(1, 1).unwrap() = 20.0;

        // B = [[2, 0], [0, 5]]
        *b.get_mut(0, 0).unwrap() = 2.0;
        *b.get_mut(1, 1).unwrap() = 5.0;

        // Ax = lambda Bx
        // 10 x1 = lambda 2 x1 => lambda = 5
        // 20 x2 = lambda 5 x2 => lambda = 4
        
        let solver = GeneralizedSelfAdjointEigenSolver::new(&a, &b, true)?;
        let evals = solver.eigenvalues();

        let v1 = *evals.get(0, 0).unwrap();
        let v2 = *evals.get(1, 0).unwrap();
        
        assert!((v1 - 4.0).abs() < 1e-6);
        assert!((v2 - 5.0).abs() < 1e-6);
        
        // Eigenvectors
        let evecs = solver.eigenvectors().unwrap();
        // For lambda=4, x2 != 0, x1 = 0. e.g. [0, 1]
        // For lambda=5, x1 != 0, x2 = 0. e.g. [1, 0]
        
        // Check orthogonality B-inner product?
        // <u, B v> = 0 if lambda_u != lambda_v
        
        // Let's just check the equations Ax = lambda Bx
        for k in 0..n {
            let lambda = *evals.get(k, 0).unwrap();
            for i in 0..n {
                 let mut ax = 0.0;
                 let mut bx = 0.0;
                 for j in 0..n {
                     ax += a.get(i, j).unwrap() * evecs.get(j, k).unwrap();
                     bx += b.get(i, j).unwrap() * evecs.get(j, k).unwrap();
                 }
                 assert!((ax - lambda * bx).abs() < 1e-4, "Ax = lambda Bx failed");
            }
        }
        
        Ok(())
    }
}
