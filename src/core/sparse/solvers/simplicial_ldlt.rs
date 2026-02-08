//! Simplicial LDLT factorization for sparse symmetric matrices.
//!
//! Performs the factorization A = L * D * L^T where L is lower triangular with unit diagonal
//! and D is diagonal.

use crate::core::scalar::Scalar;
use crate::core::sparse::sparse_matrix::{SparseMatrix, StorageOrder};
use crate::core::sparse::iterators::InnerIterator;
use crate::core::matrix::Matrix;
use crate::core::storage::{Storage, DynamicStorage};

/// Simplicial LDLT factorization of a sparse symmetric matrix.
pub struct SimplicialLDLT<T: Scalar> {
    l: SparseMatrix<T>,
    d: Vec<T>,
    inv_d: Vec<T>,
    is_factorized: bool,
}

impl<T: Scalar> Default for SimplicialLDLT<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> SimplicialLDLT<T> {
    /// Creates a new uninitialized solver.
    pub fn new() -> Self {
        Self {
            l: SparseMatrix::new(0, 0, StorageOrder::ColMajor),
            d: Vec::new(),
            inv_d: Vec::new(),
            is_factorized: false,
        }
    }

    /// Computes the factorization of the given matrix.
    pub fn compute(&mut self, matrix: &SparseMatrix<T>) -> Result<(), String> {
        self.analyze_pattern(matrix)?;
        self.factorize(matrix)
    }

    /// Symbolic factorization: analyzes the sparsity pattern.
    pub fn analyze_pattern(&mut self, matrix: &SparseMatrix<T>) -> Result<(), String> {
        if matrix.rows() != matrix.cols() {
            return Err("Matrix must be square for LDLT factorization".to_string());
        }
        let n = matrix.rows();
        self.l = SparseMatrix::new(n, n, StorageOrder::ColMajor);
        self.d = vec![T::default(); n];
        self.inv_d = vec![T::default(); n];
        self.is_factorized = false;
        Ok(())
    }

    /// Numerical factorization: computes A = L D L^T.
    pub fn factorize(&mut self, matrix: &SparseMatrix<T>) -> Result<(), String> {
        let n = matrix.rows();
        // Columns of L. We explicitly store the unit diagonal for simplicity in usage?
        // Actually, if we use standard SparseMatrix, we should store them.
        let mut l_cols: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
        let mut l_dense = vec![T::default(); n];
        
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
            // A_{rj} - sum_{k<j} L_{rk} D_{kk} L_{jk}
            for k in 0..j {
                let mut l_jk = T::default();
                // Find L_jk in l_cols[k]. 
                // Since l_cols is sorted by row index? No, purely pushed.
                // We assume sorted or we search.
                // For efficiency, usually standard impls use a linked list or similar for updating.
                // O(nnz) search here is naive but correct for now.
                for &(r, val) in &l_cols[k] {
                    if r == j {
                        l_jk = val;
                        break;
                    }
                }

                if l_jk != T::default() {
                    let d_kk = self.d[k];
                    let factor = l_jk * d_kk;
                    
                    for &(r, val) in &l_cols[k] {
                        if r >= j {
                            l_dense[r] -= val * factor;
                        }
                    }
                }
            }

            // 3. Finalize column j
            let d_jj = l_dense[j];
        
            if d_jj == T::default() {
                 // Zero pivot? For LDLT strictly, this is an issue unless we do pivoting.
                 // We return error for now.
                 return Err(format!("Zero pivot at index {}", j));
            }
            
            self.d[j] = d_jj;
            self.inv_d[j] = T::from_f64(1.0) / d_jj;
            
            // L_jj = 1.0
            l_cols[j].push((j, T::from_f64(1.0)));
            
            for i in (j + 1)..n {
                let val = l_dense[i] / d_jj;
                if val != T::default() {
                    l_cols[j].push((i, val));
                }
                l_dense[i] = T::default();
            }
            l_dense[j] = T::default();
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

            // 1. Forward substitution L * z = b
            // L has unit diagonal.
            for j in 0..n {
                // Diagonal is 1, so no division needed.
                // z_j = b_j - sum_{k<j} L_{jk} z_k
                // But we act column-wise on L?
                // Forward sub:
                // Iterate columns of L?
                // L stores columns.
                // For j=0..n:
                //   z[j] is determined (already accumulated updates).
                //   Update future z[i] using column j of L.
                //   z[i] -= L_{ij} * z[j]
                
                let z_j = sol[j];
                let mut it = InnerIterator::new(&self.l, j);
                while it.is_valid() {
                     let row = it.row();
                     if row > j { // Strict lower part
                         let val = it.value();
                         sol[row] -= val * z_j;
                     }
                     it.next();
                }
            }
            
            // 2. Diagonal solve D * y = z
            for i in 0..n {
                sol[i] *= self.inv_d[i];
            }
            
            // 3. Backward substitution L^T * x = y
            // L^T is upper triangular with unit diagonal.
            // Solve x backwards.
            for j in (0..n).rev() {
                // x[j] = y[j] - sum_{k>j} L^T_{jk} x[k]
                // L^T_{jk} = L_{kj}
                // x[j] -= L_{kj} * x[k]
                // Iterate row j of L^T -> column j of L.
                // The column j of L has entries L_{ij} for i >= j.
                // Entries i > j are L_{ij}.
                // We need sum over k > j of L_{kj} * x[k].
                // So iterate column j of L, look at row indices i > j.
                // sum += L_{ij} * sol[i]
                
                let mut sum = T::default();
                let mut it = InnerIterator::new(&self.l, j);
                while it.is_valid() {
                    let row = it.row();
                    if row > j {
                        sum += it.value() * sol[row];
                    }
                    it.next();
                }
                sol[j] -= sum; // Div by 1.0
            }
            
            for i in 0..n {
                *x.get_mut(i, k).unwrap() = sol[i];
            }
        }
        Ok(x)
    }
}
