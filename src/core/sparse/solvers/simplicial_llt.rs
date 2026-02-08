//! Simplicial Cholesky (LLT) factorization for sparse symmetric positive-definite matrices.

use crate::core::scalar::Scalar;
use crate::core::sparse::sparse_matrix::{SparseMatrix, StorageOrder};
use crate::core::sparse::iterators::InnerIterator;
use crate::core::matrix::Matrix;
use crate::core::storage::{Storage, DynamicStorage};

/// Simplicial Cholesky (LLT) factorization of a sparse symmetric positive-definite matrix.
pub struct SimplicialLLT<T: Scalar> {
    l: SparseMatrix<T>,
    is_factorized: bool,
}

impl<T: Scalar> Default for SimplicialLLT<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> SimplicialLLT<T> {
    /// Creates a new uninitialized solver.
    pub fn new() -> Self {
        Self {
            l: SparseMatrix::new(0, 0, StorageOrder::ColMajor),
            is_factorized: false,
        }
    }

    /// Computes the factorization of the given matrix.
    pub fn compute(&mut self, matrix: &SparseMatrix<T>) -> Result<(), String> {
        self.analyze_pattern(matrix)?;
        self.factorize(matrix)
    }

    /// Symbolic factorization: analyzes the sparsity pattern and prepares the L matrix.
    /// Initially, we assume a natural ordering (no permutation).
    pub fn analyze_pattern(&mut self, matrix: &SparseMatrix<T>) -> Result<(), String> {
        if matrix.rows() != matrix.cols() {
            return Err("Matrix must be square for Cholesky factorization".to_string());
        }
        let n = matrix.rows();
        
        // In a simplicial LLT, we need to know the pattern of L.
        // For symmetric A, L has the same pattern as the lower triangular part of A plus fill-ins.
        // For simplicity in this initial version, we will compute the pattern during factorization
        // or use a conservative estimate. Eigen's SimplicialLLT is more sophisticated.
        // We'll prepare an empty L with the correct dimensions.
        self.l = SparseMatrix::new(n, n, StorageOrder::ColMajor);
        self.is_factorized = false;
        Ok(())
    }

    /// Numerical factorization: computes the actual LLT decomposition.
    pub fn factorize(&mut self, matrix: &SparseMatrix<T>) -> Result<(), String> {
        let n = matrix.rows();
        // We'll store columns of L as they are computed.
        let mut l_cols: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
        let mut l_dense = vec![T::default(); n]; // Temporary dense column
        
        for j in 0..n {
            // 1. Initialize dense column with A's j-th column (lower part)
            let mut it = InnerIterator::new(matrix, j);
            while it.is_valid() {
                let r = it.row();
                if r >= j {
                    l_dense[r] = it.value();
                }
                it.next();
            }

            // 2. Update with previous columns of L
            for k in 0..j {
                // Find L(j,k)
                let mut l_jk = T::default();
                for &(r, val) in &l_cols[k] {
                    if r == j {
                        l_jk = val;
                        break;
                    }
                }

                if l_jk != T::default() {
                    // Update current dense column with col k
                    for &(r, val) in &l_cols[k] {
                        if r >= j {
                            l_dense[r] -= l_jk * val;
                        }
                    }
                }
            }

            // 3. Finalize column j
            let l_jj_sq = l_dense[j];
            if l_jj_sq <= T::default() {
                return Err("Matrix is not positive definite".to_string());
            }
            let l_jj = l_jj_sq.sqrt();
            l_dense[j] = l_jj;
            
            l_cols[j].push((j, l_jj));
            for i in (j + 1)..n {
                let val = l_dense[i] / l_jj;
                if val != T::default() {
                    l_cols[j].push((i, val));
                }
                l_dense[i] = T::default(); // Reset for next use
            }
            l_dense[j] = T::default(); // Reset
        }

        // Convert l_cols to SparseMatrix
        let mut l_triplets = Vec::new();
        for j in 0..n {
            for &(i, val) in &l_cols[j] {
                l_triplets.push(crate::core::sparse::sparse_matrix::Triplet::new(i, j, val));
            }
        }

        self.l.set_from_triplets(l_triplets);
        self.is_factorized = true;
        Ok(())
    }

    /// Solves the linear system Ax = b.
    pub fn solve<S: Storage<T>>(&self, b: &Matrix<T, S>) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        if !self.is_factorized {
            return Err("Solver is not factorized".to_string());
        }
        let n = self.l.rows();
        if b.rows() != n {
            return Err("Incompatible dimensions for solve".to_string());
        }

        let mut x = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, b.cols())?;
        for k in 0..b.cols() {
            let mut sol = vec![T::default(); n];
            for i in 0..n {
                sol[i] = *b.get(i, k).unwrap();
            }

            // Forward substitution L * y = b
            for j in 0..n {
                let mut it = InnerIterator::new(&self.l, j);
                // First element should be L(j,j)
                if it.is_valid() && it.row() == j {
                    let l_jj = it.value();
                    sol[j] /= l_jj;
                    it.next();
                } else {
                    return Err("Missing diagonal element in L".to_string());
                }
                
                let s_j = sol[j];
                while it.is_valid() {
                    let r = it.row();
                    sol[r] -= s_j * it.value();
                    it.next();
                }
            }

            // Backward substitution L^T * x = y
            // L^T is RowMajor if L is ColMajor.
            // We need to solve L^T * x = y.
            for j in (0..n).rev() {
                // Since L is ColMajor, we need the j-th row of L^T, which is the j-th column of L.
                // But wait, the j-th row of L^T is the j-th column of L.
                // No, L^T * x = y means sum(L^T(j, i) * x_i) = y_j for i >= j.
                // which is sum(L(i, j) * x_i) = y_j.
                
                let mut it = InnerIterator::new(&self.l, j);
                let mut l_jj = T::default();
                if it.is_valid() && it.row() == j {
                    l_jj = it.value();
                    it.next();
                }
                
                let mut sum = T::default();
                while it.is_valid() {
                    sum += it.value() * sol[it.row()];
                    it.next();
                }
                sol[j] = (sol[j] - sum) / l_jj;
            }

            for i in 0..n {
                *x.get_mut(i, k).unwrap() = sol[i];
            }
        }

        Ok(x)
    }

    pub fn matrix_l(&self) -> &SparseMatrix<T> {
        &self.l
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::sparse::Triplet;

    #[test]
    fn test_sparse_llt_basic() {
        // A = [ 2 -1  0 ]
        //     [-1  2 -1 ]
        //     [ 0 -1  2 ]
        // This is a common SPD matrix (Poisson 1D).
        let mut a = SparseMatrix::<f64>::new(3, 3, StorageOrder::ColMajor);
        a.set_from_triplets(vec![
            Triplet::new(0, 0, 2.0), Triplet::new(0, 1, -1.0),
            Triplet::new(1, 0, -1.0), Triplet::new(1, 1, 2.0), Triplet::new(1, 2, -1.0),
            Triplet::new(2, 1, -1.0), Triplet::new(2, 2, 2.0),
        ]);

        let mut llt = SimplicialLLT::new();
        llt.compute(&a).unwrap();

        let l = llt.matrix_l();
        assert_eq!(l.rows(), 3);
        assert_eq!(l.cols(), 3);

        let b = Matrix::<f64, DynamicStorage<f64>>::from_vec(3, 1, vec![1.0, 0.0, 1.0]).unwrap();
        let x = llt.solve(&b).unwrap();

        assert!((x.get(0, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((x.get(1, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((x.get(2, 0).unwrap() - 1.0).abs() < 1e-10);
    }
}
