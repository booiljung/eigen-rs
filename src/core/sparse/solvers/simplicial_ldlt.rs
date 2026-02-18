
use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::sparse::iterators::InnerIterator;
use crate::core::sparse::sparse_matrix::{SparseMatrix, StorageOrder};
use crate::core::storage::{DynamicStorage, Storage};
use crate::core::sparse::ordering::{Ordering, Permutation, COLAMD};

/// Simplicial LDLT factorization of a sparse symmetric matrix.
pub struct SimplicialLDLT<T: Scalar> {
    l: SparseMatrix<T>,
    d: Vec<T>,
    inv_d: Vec<T>,
    p: Option<Permutation>,
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
            p: None,
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
        
        let ordering = COLAMD;
        let p = ordering.compute(matrix);
        self.p = Some(p);
        
        // n is needed to init d/inv_d if we want to pre-allocate?
        // Actually factorize calls init. 
        // But if we want consistent state after analyze:
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
        
        // 1. Permute Matrix: A_prime = P * A * P^T
        let a_prime = if let Some(ref p) = self.p {
             p.permute_symmetric(matrix)
        } else {
             matrix.clone()
        };
        
        self.d = vec![T::default(); n];
        self.inv_d = vec![T::default(); n];
        
        // Columns of L. We explicitly store the unit diagonal for simplicity in usage?
        // Actually, if we use standard SparseMatrix, we should store them.
        let mut l_cols: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
        let mut l_dense = vec![T::default(); n];

        for j in 0..n {
            // 1. Initialize dense column with A's j-th column (lower part)
            let mut it = InnerIterator::new(&a_prime, j);
            while it.is_valid() {
                let r = it.row();
                if r >= j {
                    l_dense[r] = it.value();
                }
                it.next();
            }

            // 2. Update with previous columns of L
            // A_{rj} - sum_{k<j} L_{rk} D_{kk} L_{jk}
            for (k, l_col_k) in l_cols.iter().enumerate().take(j) {
                let mut l_jk = T::default();
                for &(r, val) in l_col_k {
                    if r == j {
                        l_jk = val;
                        break;
                    }
                }

                if l_jk != T::default() {
                    let d_kk = self.d[k];
                    let factor = l_jk * d_kk;

                    for &(r, val) in l_col_k {
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

            for (i, val) in l_dense.iter_mut().enumerate().take(n).skip(j + 1) {
                *val /= d_jj;
                if *val != T::default() {
                    l_cols[j].push((i, *val));
                }
                *val = T::default();
            }
            l_dense[j] = T::default();
        }

        // Convert l_cols to SparseMatrix
        let mut l_triplets = Vec::new();
        for (j, l_col_j) in l_cols.iter().enumerate().take(n) {
            for &(i, val) in l_col_j {
                l_triplets.push(crate::core::sparse::sparse_matrix::Triplet::new(i, j, val));
            }
        }

        self.l = SparseMatrix::new(n, n, StorageOrder::ColMajor);
        self.l.set_from_triplets(l_triplets);
        self.is_factorized = true;
        Ok(())
    }

    /// Solves the linear system Ax = b.
    pub fn solve<S: Storage<T>>(
        &self,
        b: &Matrix<T, S>,
    ) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        if !self.is_factorized {
            return Err("Solver is not factorized".to_string());
        }
        let n = self.l.rows();
        if b.rows() != n {
            return Err("Incompatible dimensions for solve".to_string());
        }

        let mut x = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, b.cols())?;
        
        let p = self.p.as_ref().unwrap();
        let p_inv = p.inverse_indices();
        
        for k in 0..b.cols() {
            let mut sol = vec![T::default(); n];
            // 1. Permute b -> c (c stored in sol)
             for i in 0..n {
                let new_idx = p_inv[i];
                sol[new_idx] = *b.get(i, k).unwrap();
            }
            
            // 1. Forward substitution L * z = c
            // L has unit diagonal.
            #[allow(clippy::needless_range_loop)]
            for j in 0..n {
                let z_j = sol[j];
                let mut it = InnerIterator::new(&self.l, j);
                while it.is_valid() {
                    let row = it.row();
                    if row > j {
                        // Strict lower part
                        let l_val = it.value();
                        sol[row] -= l_val * z_j;
                    }
                    it.next();
                }
            }

            // 2. Diagonal solve D * y = z
            for (i, val) in sol.iter_mut().enumerate().take(n) {
                *val *= self.inv_d[i];
            }

            // 3. Backward substitution L^T * x_perm = y
            // L^T is upper triangular with unit diagonal.
            // Solve x_perm backwards.
            for j in (0..n).rev() {
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
            
            // 4. Permute x_perm -> x
            for i in 0..n {
                let new_idx = p_inv[i];
                *x.get_mut(i, k).unwrap() = sol[new_idx];
            }
        }
        Ok(x)
    }
}
