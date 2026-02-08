//! GMRES (Generalized Minimal Residual) solver for sparse square problems.
//!
//! Solves Ax = b for square, non-symmetric matrices using the GMRES method.
//! This implementation uses restarts to limit memory usage.

use crate::core::scalar::Scalar;
use crate::core::matrix::{Matrix, DynamicStorage};
use crate::core::storage::Storage;
use crate::core::sparse::sparse_matrix::SparseMatrix;
use crate::core::sparse::solvers::iterative_solver_base::{Preconditioner, IdentityPreconditioner};

/// Generalized Minimal Residual (GMRES) solver.
///
/// Solves the linear system Ax = b using the GMRES algorithm.
/// This solver is suitable for general non-symmetric matrices.
pub struct GMRES<T: Scalar, P: Preconditioner<T> + Default = IdentityPreconditioner> {
    max_iter: usize,
    tolerance: T,
    restart: usize,
    preconditioner: P,
    iterations: usize,
    error: T,
}

impl<T: Scalar, P: Preconditioner<T> + Default> Default for GMRES<T, P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar, P: Preconditioner<T> + Default> GMRES<T, P> {
    /// Creates a new GMRES solver with default parameters.
    pub fn new() -> Self {
        Self {
            max_iter: 1000,
            tolerance: T::from_f64(1e-6),
            restart: 30, // Default restart parameter
            preconditioner: P::default(),
            iterations: 0,
            error: T::default(),
        }
    }
    
    // ... setters ...

    /// Sets the maximum number of iterations.
    pub fn set_max_iter(&mut self, max_iter: usize) {
        self.max_iter = max_iter;
    }

    /// Sets the tolerance for convergence.
    pub fn set_tolerance(&mut self, tolerance: T) {
        self.tolerance = tolerance;
    }
    
    /// Sets the restart parameter (m).
    pub fn set_restart(&mut self, restart: usize) {
        self.restart = restart;
    }

    /// Returns the number of iterations performed in the last solve.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Returns the error (residual norm) of the last solve.
    pub fn error(&self) -> T {
        self.error
    }
    
    /// Solves Ax = b.
    pub fn solve<S: Storage<T>>(&mut self, matrix: &SparseMatrix<T>, b: &Matrix<T, S>) -> Result<Matrix<T, DynamicStorage<T>>, String> 
    where T: Scalar + 'static 
    {
        let n = matrix.rows();
        if matrix.cols() != n {
            return Err("Matrix must be square".to_string());
        }
        if b.rows() != n {
            return Err("Dimension mismatch between A and b".to_string());
        }
        
        // Initialize preconditioner
        self.preconditioner.compute(matrix)?;
        
        // Initial guess x = 0
        let mut x = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, 1)?;
        
        let start_residual = b.norm();
        if start_residual <= T::default() {
             return Ok(x);
        }
        
        let threshold = self.tolerance * start_residual;
        self.iterations = 0;
        
        let m = self.restart;
        
        // Main restart loop
        for _cycle in 0..self.max_iter.div_ceil(m) {
            // 1. Calculate residual r = M^-1 (b - A x)
            // A * x
            let ax = matrix.mul_dense(&x)?;
            // b - A x
            let mut r_raw = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, 1)?;
            // r_raw = b - ax.
            // b is &Matrix. ax is Matrix.
            // b - &ax should work.
            r_raw.assign(&(b - &ax))?;
            
            // Apply preconditioner: r = M^-1 r_raw
            let r = self.preconditioner.solve(&r_raw)?;
            
            let beta = r.norm();
            self.error = beta;
            
            if beta <= threshold {
                return Ok(x);
            }
            
            // 2. Initialize Arnoldi
            let mut v: Vec<Matrix<T, DynamicStorage<T>>> = Vec::with_capacity(m + 1);
            
            // v[0] = r / beta
            let mut v0 = r.clone();
            v0.scale(T::from_usize(1) / beta);
            v.push(v0);
            
            // Hessenberg matrix H
            let mut h = Matrix::<T, DynamicStorage<T>>::new_dynamic(m + 1, m)?;
            
            // Right hand side for the least squares problem
            let mut g = Vec::with_capacity(m + 1);
            g.push(beta);
            for _ in 0..m { g.push(T::default()); }
            
            // Rotations
            let mut c = vec![T::default(); m];
            let mut s = vec![T::default(); m];
            
            let mut k = 0; // Steps in this restart
            
            for j in 0..m {
                if self.iterations >= self.max_iter {
                    break;
                }
                self.iterations += 1;
                k = j;
                
                // Arnoldi Step
                // w = M^-1 A v[j]
                let av = matrix.mul_dense(&v[j])?;
                let mut w = self.preconditioner.solve(&av)?;
                
                // Modified Gram-Schmidt
                for i in 0..=j {
                    // H(i, j) = v[i] . w
                    let hij = v[i].dot(&w);
                    *h.get_mut(i, j).unwrap() = hij;
                    
                    // w = w - hij * v[i]
                    // Manual update to avoid trait matching issues
                    let rows = w.rows();
                    for r_idx in 0..rows {
                        let val = *w.get(r_idx, 0).unwrap() - (*v[i].get(r_idx, 0).unwrap() * hij);
                        *w.get_mut(r_idx, 0).unwrap() = val;
                    }
                }
                
                let w_norm = w.norm();
                *h.get_mut(j + 1, j).unwrap() = w_norm;
                
                // Check breakdown
                let is_breakdown = w_norm.abs() < T::epsilon();
                
                if !is_breakdown {
                    let mut w_new = w.clone();
                    w_new.scale(T::from_usize(1) / w_norm);
                    v.push(w_new);
                }
                
                // Apply Givens rotations to the new column of H
                for i in 0..j {
                    let h_ij = *h.get(i, j).unwrap();
                    let h_ip1_j = *h.get(i + 1, j).unwrap();
                    
                    let temp = c[i] * h_ij + s[i] * h_ip1_j;
                    *h.get_mut(i + 1, j).unwrap() = -s[i] * h_ij + c[i] * h_ip1_j;
                    *h.get_mut(i, j).unwrap() = temp;
                }
                
                // Generate new rotation
                let h_jj = *h.get(j, j).unwrap();
                let h_jp1_j = *h.get(j + 1, j).unwrap();
                
                let (cj, sj, r_val) = Self::givens(h_jj, h_jp1_j);
                c[j] = cj;
                s[j] = sj;
                
                *h.get_mut(j, j).unwrap() = r_val;
                *h.get_mut(j + 1, j).unwrap() = T::default();
                
                // Apply rotation to g
                let g_j = g[j];
                g[j] = cj * g_j;
                g[j + 1] = -sj * g_j;
                
                self.error = g[j + 1].abs();
                
                if is_breakdown || self.error <= threshold {
                    k = j; // Converged at step j
                    break;
                }
            }
            
            // Solve H_k y = g_k (Upper triangular)
            let size = k + 1;
            let mut y = vec![T::default(); size];
            for i in (0..size).rev() {
                let mut sum = g[i];
                for j in i + 1..size {
                    sum -= *h.get(i, j).unwrap() * y[j];
                }
                y[i] = sum / *h.get(i, i).unwrap();
            }
            
            // Update x = x + V y
            for (i, val) in y.iter().enumerate() {
                let vi = &v[i];
                let factor = *val;
                for r_idx in 0..n {
                    let current = *x.get(r_idx, 0).unwrap();
                    let update = *vi.get(r_idx, 0).unwrap() * factor;
                    *x.get_mut(r_idx, 0).unwrap() = current + update;
                }
            }
            
            if self.error <= threshold {
                break;
            }
        }
        Ok(x)
    }
    
    fn givens(a: T, b: T) -> (T, T, T) {
        if b == T::default() {
            return (T::from_usize(1), T::from_usize(0), a);
        }
        // Robust calculation
        let hypot = (a * a + b * b).sqrt();
        (a / hypot, b / hypot, hypot)
    }
}
